#ifndef VPIPE_LTX25_BWE_H
#define VPIPE_LTX25_BWE_H

#include "ltx25-vocoder.h"

#include <array>
#include <memory>
#include <utility>
#include <string>
#include <vector>

namespace ltx25 {

// `config.vocoder.bwe`'s front-end numbers (the generator's own shape is
// a VocoderConfig, parsed the same way as the main one).
struct BweConfig {
  int n_fft = 512;
  int hop_length = 80;
  int win_length = 512;
  int num_mels = 64;
  int input_sampling_rate = 16000;
  int output_sampling_rate = 48000;

  int ratio() const {
    return input_sampling_rate > 0
               ? output_sampling_rate / input_sampling_rate : 1;
  }
  int n_freqs() const { return n_fft / 2 + 1; }
};

bool parse_bwe_config(const vpipe::FlexData& cfg, BweConfig& out,
                      std::string* err);

// Bandwidth extension: 16 kHz stereo -> 48 kHz stereo.
//
// The chain, which is the reference's `VocoderWithBWE.forward`:
//
//   x        = vocoder(mel)                  # (2, T_low) at 16 kHz
//   x        = pad x up to a multiple of hop_length
//   mel2     = causal log-mel of x
//   residual = bwe_generator(mel2)           # rates [6,5,2,2,2] -> x240
//   skip     = resample(x)                   # sinc, x3
//   out      = clamp(residual + skip, -1, 1)[:, :T_low * 3]
//
// The generator is the same BigVGAN as the main vocoder with different
// rates, so it is an `Ltx25Vocoder` under a different prefix and needs
// no code of its own. What IS new is the front end:
//
//  * the STFT is CAUSAL -- `win_length - hop_length` (432) samples of
//    ZERO padding on the LEFT ONLY, so a frame never looks ahead. A
//    centred STFT is a perfectly good spectrogram that shifts the
//    residual in time;
//  * its DFT bases come FROM THE CHECKPOINT (`mel_stft.stft_fn.
//    forward_basis`, real rows then imaginary), which is deliberate on
//    the reference's part -- it wants the exact bases from training so
//    the values fed to the generator are what it saw;
//  * the resampler's filter is NOT in the checkpoint
//    (`persistent=False`) and is a HANN-window sinc, NOT the kaiser one
//    the activations use. It has to be rebuilt exactly.
class Ltx25Bwe {
public:
  // `ws` must be the audio-VAE checkpoint; both generators load from it.
  static std::unique_ptr<Ltx25Bwe>
  load(const vpipe::FlexData& cfg, std::shared_ptr<vpipe::genai::WeightSet> ws,
       vpipe::metal_compute::MetalCompute* mc, std::string* err);

  // `mel` is f32 [2][T][mel_bins] from the audio VAE decoder. Writes f32
  // [2][T * 160 * 3] at 48 kHz.
  bool synthesize(const float* mel, int T, int mel_bins,
                  std::vector<float>* out, std::array<int, 2>* shape,
                  std::string* err);

  // The 16 kHz waveform the main vocoder produced, as of the last
  // synthesize() -- the BWE's own input, kept so a caller can emit
  // either rate without running the stack twice.
  const std::vector<float>& low_rate() const { return _low; }
  int low_rate_samples() const { return _low_len; }

  int output_sampling_rate() const { return _bcfg.output_sampling_rate; }
  std::uint64_t resident_bytes() const;

  // Intermediates from the last synthesize(), for localising a
  // mismatch: "low", "mel2" ([2][T_frames][mel]), "residual", "skip".
  const std::vector<float>* tap(const std::string& name) const;
  void set_capture(bool on) { _capture = on; }

  // The HANN-window sinc the resampler rebuilds. Exposed because it is
  // the one tensor in this chain with no checkpoint to check against.
  const std::vector<float>& resampler_filter() const { return _filt_host; }

private:
  Ltx25Bwe() = default;

  using Buf = vpipe::metal_compute::SharedBuffer;

  // The causal log-mel of one channel: [T] -> [frames][num_mels].
  bool mel_of_(const float* wav, int T, int frames,
               std::vector<float>* out, std::string* err);
  bool resample_(const std::vector<float>& x, int T, int C,
                 std::vector<float>* out, std::string* err);

  BweConfig _bcfg;
  VocoderConfig _vcfg, _gcfg;
  std::shared_ptr<vpipe::genai::WeightSet> _ws;
  vpipe::metal_compute::MetalCompute* _mc = nullptr;
  std::unique_ptr<Ltx25Vocoder> _voc, _gen;

  vpipe::metal_compute::ComputeLibrary _lib;
  vpipe::metal_compute::ComputeFunction _fn_stft, _fn_mag, _fn_logc,
      _fn_gemm, _fn_upr;

  Buf _fwd_basis, _mel_basis, _zero_bias, _filt;
  std::vector<float> _filt_host;
  int _filt_k = 0, _filt_pad = 0, _filt_pad_left = 0;

  std::vector<float> _low;
  int _low_len = 0;
  std::vector<std::pair<std::string, std::vector<float>>> _taps;
  bool _capture = false;
};

}  // namespace ltx25

#endif
