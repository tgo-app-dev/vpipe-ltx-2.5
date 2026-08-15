#ifndef VPIPE_LTX25_AUDIO_ENCODER_H
#define VPIPE_LTX25_AUDIO_ENCODER_H

#include "ltx25-audio-vae.h"
#include "ltx25-metal-ops.h"

#include "generative-models/weight-set.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace ltx25 {

// ---------------------------------------------------------------------
// The mel FRONT END: waveform -> log-mel, on the host in f32.
//
// A `torchaudio.transforms.MelSpectrogram` transcribed, and it is the
// half of the audio encoder with all the silent knobs. Every one of
// these produces a plausible spectrogram and a wrong latent:
//
//   * POWER 1, not 2. The reference asks for magnitudes; squaring them
//     doubles the log and the whitening absorbs the error into
//     something that is merely wrong rather than obviously wrong.
//   * SLANEY mel scale AND slaney normalisation, not HTK. The scale
//     moves every filter's centre; the norm scales the low bins.
//   * CENTRE + REFLECT padding, so frame t is centred on sample
//     t * hop. center=False shifts the whole spectrogram by n_fft/2.
//   * log(CLAMP(mel, 1e-5)), not log1p and not a different floor.
//
// AND IT IS NOT CAUSAL, even though the checkpoint's
// `preprocessing.stft.causal` is true and the DECODE side's vocoder mel
// is. `AudioProcessor` ignores `is_causal` and always centres. That
// asymmetry belongs to the reference; a port that "fixed" it would be
// encoding a different spectrogram from the one the VAE was trained on.
//
// Runs on the host because it is small -- a 5 s clip is ~500 frames of a
// 1024-point DFT per channel -- and because the filter bank is NOT in
// the checkpoint (unlike the BWE's, which reads its bases from
// `mel_stft.stft_fn.*`), so it has to be built from the scale definition
// either way.
class Ltx25AudioMel {
public:
  struct Config {
    int sample_rate = 16000;
    int n_fft       = 1024;
    int win_length  = 1024;
    int hop_length  = 160;
    int mel_bins    = 64;
    double f_min    = 0.0;
    double f_max    = 8000.0;
  };

  explicit Ltx25AudioMel(const Config& cfg);

  // The filter bank, [mel_bins][n_fft/2 + 1], slaney-normalised. Exposed
  // so a test can pin the MATRIX rather than only the spectrogram it
  // produced -- a wrong scale and a wrong norm look alike downstream.
  const std::vector<float>& filters() const { return _fb; }
  int n_freqs() const { return _cfg.n_fft / 2 + 1; }

  // How many frames `n_samples` produces: 1 + n / hop, the centred
  // convention.
  int frames_for(int n_samples) const;

  // `pcm` is f32 PLANAR [channels][n_samples]. Writes f32
  // [channels][frames][mel_bins] -- the [C][TIME][MEL] layout the
  // encoder takes, which is the reference's own permute.
  void compute(const float* pcm, int channels, int n_samples,
               std::vector<float>* out, std::array<int, 3>* shape) const;

private:
  Config _cfg;
  std::vector<float> _window;   // hann, PERIODIC (torch's default)
  std::vector<float> _fb;       // [mel_bins][n_freqs]
};

// ---------------------------------------------------------------------
// The audio VAE ENCODER on the GPU: log-mel -> whitened latent rows.
//
// The mirror of Ltx25AudioVaeDecoder and it reuses that decoder's ops
// wholesale -- the stride-1 convolutions have the same causal-in-TIME
// padding, and PixelNorm is the same parameterless one. Three things are
// its own, and each has a kernel: the mel upload, the STRIDE-2
// downsample (whose padding is (0,1,2,0), not the stride-1 (1,1,2,0)),
// and the head.
//
// IT EMITS ROWS. The head keeps the first `z_channels` of its 2z output
// -- the means; the rest is a log-variance the reference discards --
// patchifies `b c t f -> b t (c f)` and whitens with the per-(channel,
// mel-bin) statistics. [rows, z * mel] IS what `generate-video`'s
// `ref_audio_rows` takes and what the DiT's audio patchify assumes, so
// nothing downstream has to know this layout was ever 3-D.
class Ltx25AudioVaeEncoder {
public:
  static std::unique_ptr<Ltx25AudioVaeEncoder>
  load(const AudioVaeConfig& cfg,
       std::shared_ptr<vpipe::genai::WeightSet> ws, const MetalOps& ops,
       std::string* err);

  // `mel` is f32 [in_channels][frames][mel_bins] as Ltx25AudioMel writes
  // it. Writes f32 ROWS [frames / 4][z_channels * latent_mel_bins].
  bool encode(const float* mel, int frames, std::vector<float>* rows,
              std::array<int, 2>* shape, std::string* err);

  // Intermediates, keyed "conv_in", "down0".."down2", "mid", "head".
  void set_capture(bool on) { _capture = on; }
  const std::vector<float>* tap(const std::string& name) const;
  bool tap_shape(const std::string& name, std::array<int, 3>* shape) const;

  void set_max_im2col_elems(std::size_t n) { _max_im2col = n; }
  std::uint64_t resident_bytes() const { return _resident; }
  const AudioVaeConfig& config() const { return _cfg; }

private:
  Ltx25AudioVaeEncoder() = default;

  struct Conv {
    vpipe::metal_compute::SharedBuffer w, b;
    int cout = 0, cin = 0, k = 3;
    bool ok() const { return cout > 0; }
  };
  struct ResBlock { Conv conv1, conv2, nin; };
  struct Level {
    std::vector<ResBlock> res;
    Conv down;                    // empty on the last level
  };

  bool load_conv_(vpipe::genai::WeightSet& ws, const std::string& prefix,
                  Conv& out, bool optional, std::string* err);
  bool load_res_(vpipe::genai::WeightSet& ws, const std::string& prefix,
                 ResBlock& out, std::string* err);
  // Stride-1 3x3 (or 1x1 for a nin_shortcut), channel-last.
  void conv_(vpipe::metal_compute::CommandStream& stream, const Conv& c,
             const vpipe::metal_compute::SharedBuffer& x, int H, int W,
             const vpipe::metal_compute::SharedBuffer& y);
  void res_(vpipe::metal_compute::CommandStream& stream, const ResBlock& rb,
            vpipe::metal_compute::SharedBuffer& x, int& cc, int H, int W);
  void grab_(vpipe::metal_compute::CommandStream& stream,
             const std::string& name,
             const vpipe::metal_compute::SharedBuffer& x, int C, int H,
             int W);

  AudioVaeConfig _cfg;
  std::shared_ptr<vpipe::genai::WeightSet> _ws;
  const MetalOps* _ops = nullptr;
  Conv _conv_in, _conv_out;
  std::vector<Level> _levels;
  ResBlock _mid1, _mid2;
  vpipe::metal_compute::SharedBuffer _std_of_means, _mean_of_means;
  vpipe::metal_compute::SharedBuffer _im2col;
  std::size_t _max_im2col = 32ull * 1024 * 1024;
  std::uint64_t _resident = 0;

  bool _capture = false;
  std::vector<std::pair<std::string, std::vector<float>>> _taps;
  std::vector<std::pair<std::string, std::array<int, 3>>> _tap_shapes;
};

}  // namespace ltx25

#endif
