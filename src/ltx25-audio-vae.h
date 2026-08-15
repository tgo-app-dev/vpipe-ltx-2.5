#ifndef VPIPE_LTX25_AUDIO_VAE_H
#define VPIPE_LTX25_AUDIO_VAE_H

#include "ltx25-metal-ops.h"

#include "common/flex-data.h"
#include "generative-models/weight-set.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace ltx25 {

// What the checkpoint's `config.audio_vae.model.params.ddconfig` says.
struct AudioVaeConfig {
  int ch = 128;                       // base channels
  int out_ch = 2;                     // stereo
  std::vector<int> ch_mult{1, 2, 4};
  int num_res_blocks = 2;             // the path builds this PLUS ONE
  int z_channels = 8;
  int mel_bins = 64;
  std::string norm_type = "pixel";
  std::string causality_axis = "height";
  bool mid_block_add_attention = false;

  // Levels, high to low, as the decoder runs them.
  int num_levels() const { return (int)ch_mult.size(); }
  // The mel bins the LATENT carries: the encoder halves the axis once
  // per level boundary, so 64 / 2^(levels-1) = 16. This is also the
  // stride through the per-(channel, mel-bin) statistics.
  int latent_mel_bins() const;
  // Statistics length: z_channels * latent_mel_bins.
  int stats_len() const { return z_channels * latent_mel_bins(); }
};

bool parse_audio_vae_config(const vpipe::FlexData& cfg, AudioVaeConfig& out,
                            std::string* err);

// The audio VAE decoder on the GPU: latent -> log-mel spectrogram.
//
// An LDM 2D decoder, so structurally much plainer than the video VAE --
// no depth-to-space, no attention, and `norm_type: pixel` means the
// normalisations are PARAMETERLESS (the checkpoint carries no norm
// tensors at all). It reuses the video decoder's channel-last im2col +
// GEMM shape one axis shorter.
//
// ---- THE THREE CONVENTIONS THAT ARE ITS OWN ----
//
// The tensor is (batch, channels, FRAMES, MEL_BINS), so torch's H is
// TIME and W is frequency. Everything below follows from that:
//
//  * `causality_axis: height` is causal in TIME. CausalConv2d puts ALL
//    the height padding on TOP and none at the bottom, while width is
//    padded symmetrically. Using the symmetric form on both axes shifts
//    the spectrogram a frame in time and still sounds like audio.
//  * Upsample interpolates 2x on both axes, convolves, and then drops
//    the FIRST ROW. Two upsamples take F -> 4F-3, which is exactly the
//    frame count `_denormalize_latents` targets, so the reference's
//    crop/pad step is always a no-op and this port does not implement
//    it.
//  * The per-channel statistics are NOT per channel. See
//    `ltx_audio_denorm_in`.
class Ltx25AudioVaeDecoder {
public:
  static std::unique_ptr<Ltx25AudioVaeDecoder>
  load(const AudioVaeConfig& cfg,
       std::shared_ptr<vpipe::genai::WeightSet> ws, const MetalOps& ops,
       std::string* err);

  // `latent` is f32 [z_channels][F][latent_mel_bins]. Writes f32
  // [out_ch][4F-3][mel_bins].
  bool decode(const float* latent, int F, std::vector<float>* out,
              std::array<int, 3>* shape, std::string* err);

  void set_max_im2col_elems(std::size_t n) { _max_im2col = n; }
  std::size_t max_im2col_elems() const { return _max_im2col; }
  std::uint64_t resident_bytes() const { return _resident; }

  // Capture the stage outputs so a mismatch names the stage that caused
  // it rather than just the spectrogram. Off by default; the copies are
  // taken INSIDE the stream because the resnet blocks add their
  // residual in place, so holding the buffer alone would hand back a
  // tensor a later block has already overwritten.
  //
  // Taps are CHANNEL-FIRST [C][H][W], matching the goldens.
  void set_capture(bool on) { _capture = on; }
  const std::vector<float>* tap(const std::string& name) const;
  bool tap_shape(const std::string& name, std::array<int, 3>* out) const;

private:
  Ltx25AudioVaeDecoder() = default;

  struct Conv {
    vpipe::metal_compute::SharedBuffer w, b;   // [Cout][Cin*k*k], [Cout]
    int cout = 0, cin = 0, k = 3;
  };
  // norm1 -> SiLU -> conv1 -> norm2 -> SiLU -> conv2, plus x (or
  // nin_shortcut(x)). `nin.cout == 0` means in == out and the shortcut
  // is Identity.
  struct ResBlock { Conv conv1, conv2, nin; };
  struct Level {
    std::vector<ResBlock> blocks;
    Conv upsample;              // cout == 0 on the last level
  };

  bool load_conv_(vpipe::genai::WeightSet& ws, const std::string& prefix,
                  Conv& out, bool optional, std::string* err);
  bool load_res_(vpipe::genai::WeightSet& ws, const std::string& prefix,
                 ResBlock& out, std::string* err);

  // x[H*W][cin] -> y[H*W][cout]. k==1 is a plain GEMM (a 1x1
  // CausalConv2d pads by nothing); k==3 is chunked im2col + GEMM.
  void conv_(vpipe::metal_compute::CommandStream& stream, const Conv& c,
             const vpipe::metal_compute::SharedBuffer& x, int H, int W,
             const vpipe::metal_compute::SharedBuffer& y);
  // The full resnet block, in place on `x` (which it may replace).
  void res_(vpipe::metal_compute::CommandStream& stream, const ResBlock& rb,
            vpipe::metal_compute::SharedBuffer& x, int& cc, int H, int W);
  // Queue a channel-last -> channel-first copy of `x` under `name`.
  void grab_(vpipe::metal_compute::CommandStream& stream,
             const std::string& name,
             const vpipe::metal_compute::SharedBuffer& x, int C, int H,
             int W);

  struct Tap {
    std::string name;
    vpipe::metal_compute::SharedBuffer buf;    // f32, channel-first
    std::array<int, 3> shape{};
    std::vector<float> host;
  };
  std::vector<Tap> _taps;
  bool _capture = false;

  AudioVaeConfig _cfg;
  std::shared_ptr<vpipe::genai::WeightSet> _ws;
  const MetalOps* _ops = nullptr;
  Conv _conv_in, _conv_out;
  ResBlock _mid1, _mid2;
  std::vector<Level> _levels;                  // indexed by LEVEL
  vpipe::metal_compute::SharedBuffer _std, _mean;
  vpipe::metal_compute::SharedBuffer _im2col;
  std::size_t _max_im2col = 64ull * 1024 * 1024;
  std::uint64_t _resident = 0;
};

}  // namespace ltx25

#endif
