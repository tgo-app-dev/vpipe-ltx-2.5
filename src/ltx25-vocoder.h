#ifndef VPIPE_LTX25_VOCODER_H
#define VPIPE_LTX25_VOCODER_H

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include "common/flex-data.h"
#include "generative-models/weight-set.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace ltx25 {

// What `config.vocoder.<vocoder|bwe>` says.
struct VocoderConfig {
  int upsample_initial_channel = 1536;
  std::vector<int> upsample_rates{5, 2, 2, 2, 2, 2};
  std::vector<int> upsample_kernel_sizes{11, 4, 4, 4, 4, 4};
  std::vector<int> resblock_kernel_sizes{3, 7, 11};
  std::vector<std::vector<int>> resblock_dilation_sizes;
  std::string resblock = "AMP1";
  std::string activation = "snakebeta";
  bool use_tanh_at_final = false;
  bool use_bias_at_final = false;
  bool apply_final_activation = true;
  int output_sampling_rate = 16000;

  int num_upsamples() const { return (int)upsample_rates.size(); }
  int num_kernels() const { return (int)resblock_kernel_sizes.size(); }
  // Total samples per input mel frame: the product of the rates.
  int hop() const;
};

// Parse `config.vocoder.<which>`; `which` is "vocoder" or "bwe".
bool parse_vocoder_config(const vpipe::FlexData& cfg, const std::string& which,
                          VocoderConfig& out, std::string* err);

// The BigVGAN vocoder: log-mel -> waveform.
//
// ---- WHY THIS RUNS IN f32 ----
//
// The reference forces its whole vocoder pass to fp32 and documents the
// reason: bf16 accumulation compounds through its 108 sequential
// convolutions and degrades spectral metrics (mel_l1, MRSTFT) by 40-90%.
// The WEIGHTS are bf16 in the reference too -- it upcasts them per-op --
// so what has to be wide is the ACTIVATIONS and the accumulation. This
// port therefore widens the weights once at load and runs the f32
// twin of the plugin's metallib. Running it in bf16 would not only sound
// worse, it would put the error far above anything that could
// distinguish a port bug from rounding.
//
// ---- WHAT IS NOT PLAIN HiFiGAN ----
//
//  * SnakeBeta, x + (1/(beta+1e-9)) * sin(x*alpha)^2, with alpha and
//    beta stored in LOG SCALE;
//  * every activation is ANTI-ALIASED (upsample 2x -> snake -> lowpass
//    downsample 2x) with 12-tap kaiser-sinc filters that come FROM THE
//    CHECKPOINT, so no window is recomputed here;
//  * each upsample stage runs its three resblocks on the SAME input and
//    takes the MEAN, rather than chaining them.
//
// Layout is channel-last [T][C] throughout, as elsewhere in this port,
// so a Conv1d is im2col + GEMM over the checkpoint weight as it sits.
class Ltx25Vocoder {
public:
  // `prefix` is where this generator's tensors live, e.g.
  // "vocoder.vocoder." or "vocoder.bwe_generator.".
  static std::unique_ptr<Ltx25Vocoder>
  load(const VocoderConfig& cfg, std::shared_ptr<vpipe::genai::WeightSet> ws,
       vpipe::metal_compute::MetalCompute* mc, const std::string& prefix,
       std::string* err);

  // `mel` is f32 [2][T][mel_bins] -- the audio VAE decoder's output as
  // it stands. Writes f32 [2][T * hop()].
  bool synthesize(const float* mel, int T, int mel_bins,
                  std::vector<float>* out, std::array<int, 2>* shape,
                  std::string* err);

  std::uint64_t resident_bytes() const { return _resident; }

  // Run the first resblock's first anti-aliased activation on its own,
  // returning all three intermediates (after the 2x upsample, after
  // SnakeBeta, after the lowpass). The activation is the novel op here
  // and there are 108 of them in the stack -- checking it in isolation
  // is what makes a failure say WHICH of the three steps was wrong
  // rather than just that the audio differs. Channel-first [C][*].
  bool probe_activation(const float* x, int C, int T, std::vector<float>* up,
                        std::vector<float>* snake, std::vector<float>* down,
                        std::string* err);

  // Stage captures, channel-first [C][T], for localising a mismatch.
  void set_capture(bool on) { _capture = on; }
  const std::vector<float>* tap(const std::string& name) const;
  bool tap_shape(const std::string& name, std::array<int, 2>* out) const;

private:
  Ltx25Vocoder() = default;

  using Buf = vpipe::metal_compute::SharedBuffer;
  using Stream = vpipe::metal_compute::CommandStream;

  struct Conv {                       // Conv1d
    Buf w, b;                         // [Cout][Cin*K] f32, [Cout] f32
    int cout = 0, cin = 0, k = 1, dil = 1, pad = 0;
    bool has_bias = true;
  };
  struct ConvT {                      // ConvTranspose1d
    Buf w, b;                         // [Cin][Cout][K] f32, [Cout] f32
    int cout = 0, cin = 0, k = 0, stride = 1, pad = 0;
  };
  // Activation1d: upsample 2x -> SnakeBeta -> lowpass downsample 2x.
  struct Act {
    Buf alpha, beta;                  // [C] f32, LOG scale
    Buf up_filt, down_filt;           // [12] f32, from the checkpoint
    int c = 0, kup = 12, kdown = 12;
  };
  struct AmpBlock {                   // AMPBlock1
    std::vector<Conv> convs1, convs2;
    std::vector<Act> acts1, acts2;
  };

  // ---- the f32 op set ------------------------------------------------
  bool init_ops_(std::string* err);
  Buf alloc_(std::size_t n) const;    // f32 elements
  // A checkpoint tensor widened to f32 and kept.
  Buf f32_(const std::string& name, std::string* err);

  bool load_conv_(const std::string& p, int k, int dil, int pad, bool bias,
                  Conv& out, std::string* err);
  bool load_convt_(const std::string& p, int k, int stride, int pad,
                   ConvT& out, std::string* err);
  bool load_act_(const std::string& p, Act& out, std::string* err);

  void conv1d_(Stream& s, const Conv& c, const Buf& x, int T, const Buf& y);
  void convt_(Stream& s, const ConvT& c, const Buf& x, int T, const Buf& y);
  // Length-preserving; returns the result. `tag`, when set and capture
  // is on, grabs the two intermediates as "<tag>_up" and "<tag>_snake".
  Buf activation_(Stream& s, const Act& a, const Buf& x, int C, int T,
                  const char* tag = nullptr);
  Buf amp_(Stream& s, const AmpBlock& blk, const Buf& x, int C, int T);
  void grab_(Stream& s, const std::string& name, const Buf& x, int C, int T);

  VocoderConfig _cfg;
  std::shared_ptr<vpipe::genai::WeightSet> _ws;
  vpipe::metal_compute::MetalCompute* _mc = nullptr;
  std::string _prefix;

  vpipe::metal_compute::ComputeLibrary _lib;
  vpipe::metal_compute::ComputeFunction _fn_mel_in, _fn_im2col, _fn_gemm,
      _fn_convt, _fn_up2x, _fn_down2x, _fn_snake, _fn_mean3, _fn_out,
      _fn_add, _fn_copy;

  Conv _conv_pre, _conv_post;
  std::vector<ConvT> _ups;
  std::vector<AmpBlock> _resblocks;   // num_upsamples * num_kernels
  Act _act_post;

  Buf _im2col;
  std::size_t _max_im2col = 64ull * 1024 * 1024;
  std::uint64_t _resident = 0;

  struct Tap {
    std::string name;
    Buf buf;
    std::array<int, 2> shape{};
    std::vector<float> host;
  };
  std::vector<Tap> _taps;
  bool _capture = false;
};

}  // namespace ltx25

#endif
