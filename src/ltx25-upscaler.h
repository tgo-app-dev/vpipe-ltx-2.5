#ifndef VPIPE_LTX25_UPSCALER_H
#define VPIPE_LTX25_UPSCALER_H

#include "ltx25-metal-ops.h"
#include "ltx25-upscaler-ref.h"

#include "generative-models/weight-set.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace ltx25 {

// The LATENT upscalers on the GPU: x2 spatial and x2 temporal.
//
// Semantics are Ltx25UpscalerRef's, which is verified against the
// released checkpoints at ~4e-6 -- so this is checked against THAT on a
// small case rather than argued from the source.
//
// WHAT THESE ARE FOR. The two-stage pipeline generates at a small
// geometry, upscales the LATENT, and denoises again at the larger one.
// Upscaling in latent space is what makes stage 2 cheap: no VAE decode
// and re-encode between the stages, and no pixel-space round trip to
// lose detail in.
//
// THE CONVOLUTIONS ARE ZERO-PADDED ON EVERY AXIS, which is why this uses
// `ups_im2col` and not either of the VAE's two gathers -- see the note
// on those in ltx25-metal-ops.h. The pixel shuffles reuse `vae_d2s`
// unchanged, because its `(c p1 p2 p3)` split with width fastest IS
// PixelShuffleND's nesting, and it already drops the first frame.
//
// The un-normalize / re-normalize either side of the model is
// `vae_denorm_in` and `vae_whiten_out`, the same pair the VAE uses -- so
// the wrapper the reference calls `upsample_video` costs nothing extra
// here. Both take the VAE's per-channel statistics, which are what the
// upscaler's own checkpoint does NOT carry.
class Ltx25Upscaler {
public:
  // `std_of_means` / `mean_of_means` are the VAE's per-channel
  // statistics, `in_channels` long. THEY BELONG TO THE VAE, not to this
  // checkpoint -- which carries none -- because the model works in the
  // VAE's UN-normalized latent space while everything upstream and
  // downstream holds a whitened one. Passing them explicitly is what
  // keeps that dependency visible instead of implied; empty vectors mean
  // identity, which is only right for a caller that has already
  // un-whitened.
  static std::unique_ptr<Ltx25Upscaler>
  load(const UpscalerConfig& cfg, const std::vector<float>& std_of_means,
       const std::vector<float>& mean_of_means,
       std::shared_ptr<vpipe::genai::WeightSet> ws, const MetalOps& ops,
       std::string* err);

  const UpscalerConfig& config() const { return _cfg; }

  // `latent` is f32 [in_channels][F][H][W] in the WHITENED space the DiT
  // generates in. Writes the same space at
  // [in_channels][out_frames(F)][out_height(H)][out_width(W)].
  bool upscale(const float* latent, int F, int H, int W,
               std::vector<float>* out, std::array<int, 4>* shape,
               std::string* err);

  // The im2col chunk ceiling, in ELEMENTS. Bounds the scratch by a
  // configured size rather than by the clip.
  void set_max_im2col_elems(std::size_t n) { _max_im2col = n; }

  std::uint64_t resident_bytes() const { return _resident; }
  void release_idle();

  // Intermediates, keyed as the reference keys them ("initial", "res",
  // "upsampled", "post"). Downloading them costs a copy per stage, so
  // this is off unless a test asks -- and a test should, because a
  // whole-model mismatch says nothing about which layer moved.
  void set_capture(bool on) { _capture = on; }
  const std::vector<float>* tap(const std::string& name) const;

private:
  Ltx25Upscaler() = default;

  struct Conv {
    vpipe::metal_compute::SharedBuffer w, b;
    int cout = 0, cin = 0, kf = 0, kh = 0, kw = 0;
    bool two_d() const { return kf == 1 && kh == 3 && kw == 3; }
  };
  struct Norm { vpipe::metal_compute::SharedBuffer g, b; };
  struct ResBlock { Conv conv1, conv2; Norm norm1, norm2; };

  bool load_conv_(vpipe::genai::WeightSet& ws, const std::string& p,
                  Conv& out, std::string* err);
  bool load_norm_(vpipe::genai::WeightSet& ws, const std::string& p,
                  Norm& out, std::string* err);
  // 3x3x3 (or per-frame 3x3) convolution, chunked over output cells.
  void conv_(vpipe::metal_compute::CommandStream& stream, const Conv& c,
             const vpipe::metal_compute::SharedBuffer& x, int F, int H, int W,
             const vpipe::metal_compute::SharedBuffer& y);
  void block_(vpipe::metal_compute::CommandStream& stream, const ResBlock& b,
              vpipe::metal_compute::SharedBuffer& x, int F, int H, int W);

  UpscalerConfig _cfg;
  std::shared_ptr<vpipe::genai::WeightSet> _ws;
  const MetalOps* _ops = nullptr;
  Conv _initial_conv, _final_conv, _up_conv;
  Norm _initial_norm;
  std::vector<ResBlock> _res, _post;
  vpipe::metal_compute::SharedBuffer _std, _mean;
  vpipe::metal_compute::SharedBuffer _im2col, _a, _b;
  std::size_t _max_im2col = 64u * 1024u * 1024u;
  std::uint64_t _resident = 0;
  bool _capture = false;
  std::vector<std::pair<std::string, std::vector<float>>> _taps;
  void keep_(const char* name, const vpipe::metal_compute::SharedBuffer& b,
             std::size_t n);
};

}  // namespace ltx25

#endif
