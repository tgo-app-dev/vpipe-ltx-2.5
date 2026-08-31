#ifndef VPIPE_LTX25_UPSCALER_REF_H
#define VPIPE_LTX25_UPSCALER_REF_H

#include "generative-models/weight-set.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace ltx25 {

// The LATENT upscalers -- x2 spatial and x2 temporal -- as a
// from-scratch CPU reference, f32.
//
// These upscale a VAE latent WITHOUT decoding it to pixels, which is
// what makes the two-stage pipeline cheap: generate at a small geometry,
// upscale the latent, denoise again at the larger one.
//
// WHY A REFERENCE FIRST. Everything risky here is layout, and every one
// of these mistakes yields a correctly-shaped tensor:
//
//   * GroupNorm(32) reduces over (channels-in-group x F x H x W) and
//     scales PER CHANNEL. Reducing per channel instead is a different
//     operator that still normalises something.
//   * the pixel shuffle splits the channel with the LAST factor
//     FASTEST -- (c p1 p2) means idx = ((c*2)+p1)*2+p2. This port has
//     already been bitten once by the opposite nesting: the VAE's
//     DepthToSpace and its unpatchify nest in OPPOSITE orders.
//   * the SPATIAL model's upsampler is a Conv2d run PER FRAME
//     (b c f h w -> (b f) c h w and back), not a Conv3d. Its weight is
//     4-D, which is the only thing that says so.
//   * the TEMPORAL model DROPS FRAME 0 after the shuffle, because the
//     first latent frame encodes a single pixel frame. Keep it and every
//     frame is off by one -- 8*(F-1)+1 versus 8F, the same asymmetry the
//     VAE decoder has.
//
// AND ONE TRAP THAT IS NOT LAYOUT. The temporal checkpoint's config says
// `rational_resampler: true`, and it is a RED HERRING: the module only
// consults that flag inside `elif spatial_upsample:`, and this
// checkpoint has spatial_upsample false. It takes the plain
// Sequential(Conv3d, PixelShuffleND(1)) branch. Believing the flag
// builds a different module that still loads.
//
// Not fast, and not meant to be. Use it for correctness.

struct UpscalerConfig {
  int  in_channels = 128;
  int  mid_channels = 512;
  int  num_blocks_per_stage = 4;
  int  dims = 3;
  bool spatial_upsample = true;
  bool temporal_upsample = false;
  double spatial_scale = 2.0;
  bool rational_resampler = false;

  // Read from the checkpoint's `__metadata__["config"]`, which is a
  // JSON STRING inside the metadata object rather than a nested object.
  // Defaults stay put for a key the file does not carry.
  static bool from_metadata(const std::string& file, UpscalerConfig* out,
                            std::string* err);

  // What this checkpoint does to a [C][F][H][W] latent.
  int out_frames(int f) const
  {
    return temporal_upsample ? 2 * f - 1 : f;
  }
  int out_height(int h) const { return spatial_upsample ? 2 * h : h; }
  int out_width(int w) const { return spatial_upsample ? 2 * w : w; }
};

class Ltx25UpscalerRef {
public:
  static std::unique_ptr<Ltx25UpscalerRef>
  load(const UpscalerConfig& cfg, vpipe::genai::WeightSet& ws,
       vpipe::metal_compute::MetalCompute* mc, std::string* err);

  const UpscalerConfig& config() const { return _cfg; }

  // `latent` is f32 [in_channels][F][H][W] in the VAE's UN-NORMALIZED
  // space -- the caller un-whitens before and re-whitens after, which is
  // what the reference's upsample_video() wrapper does. Writes
  // [in_channels][out_frames(F)][out_height(H)][out_width(W)].
  bool forward(const float* latent, int F, int H, int W,
               std::vector<float>* out, std::array<int, 4>* shape,
               std::string* err);

  // Intermediates, keyed the way the goldens are ("initial", "res",
  // "shuffled", "upsampled", "post"). A whole-model mismatch says
  // nothing about WHERE.
  void set_capture(bool on) { _capture = on; }
  const std::vector<float>* tap(const std::string& name) const;
  bool tap_shape(const std::string& name, std::array<int, 4>* shape) const;

private:
  Ltx25UpscalerRef() = default;

  struct Conv {
    std::vector<float> w, b;
    int cout = 0, cin = 0, kf = 0, kh = 0, kw = 0;
  };
  struct Norm { std::vector<float> w, b; };
  struct ResBlock { Conv conv1, conv2; Norm norm1, norm2; };

  UpscalerConfig _cfg;
  Conv _initial_conv, _final_conv, _up_conv;
  Norm _initial_norm;
  std::vector<ResBlock> _res, _post;
  bool _capture = false;
  std::vector<std::pair<std::string, std::vector<float>>> _taps;
  std::vector<std::pair<std::string, std::array<int, 4>>> _tap_shapes;
};

}  // namespace ltx25

#endif
