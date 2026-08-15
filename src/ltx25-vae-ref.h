#ifndef VPIPE_LTX25_VAE_REF_H
#define VPIPE_LTX25_VAE_REF_H

#include "ltx25-vae-config.h"

#include "generative-models/weight-set.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace ltx25 {

// The conv video VAE decoder as a from-scratch CPU reference, f32.
//
// WHY A CPU REFERENCE AT ALL. Everything risky in this decoder is
// LAYOUT, and every one of the layout mistakes produces a
// correctly-shaped tensor:
//
//   * DepthToSpaceUpsample splits the channel as (c p1 p2 p3) with p3
//     (WIDTH) fastest; unpatchify splits as (c p r q) with q (HEIGHT)
//     fastest. The two nest in OPPOSITE orders, so one implementation
//     reused for both is wrong exactly half the time.
//   * every stride-2-in-time upsample DROPS ITS FIRST FRAME, which is
//     the whole reason the output is 8*(F-1)+1 and not 8F.
//   * convolutions REPLICATE-pad in time and ZERO-pad in space. Using
//     one mode for both is a quiet edge-only error.
//   * PixelNorm's eps is 1e-8 -- the class default, NOT the 1e-6 that
//     build_normalization_layer passes. The resnet blocks construct
//     `PixelNorm()` with no arguments.
//
// This runs the whole graph in f32 against reference goldens, so those
// are pinned before any kernel is written -- and afterwards it stays as
// the thing the GPU path is checked against on a small case.
//
// It is NOT fast and is not meant to be: ~2.5 GMAC for the tiny golden.
// Use it for correctness, not for pixels.
class Ltx25VaeRef {
public:
  static std::unique_ptr<Ltx25VaeRef>
  load(const VaeConfig& cfg, vpipe::genai::WeightSet& ws,
       vpipe::metal_compute::MetalCompute* mc, std::string* err);

  // `latent` is f32 [latent_channels][F][H][W]; the decode is
  // deterministic (timestep_conditioning is off in this checkpoint, so
  // there is no noise injection).
  //
  // Writes [3][8*(F-1)+1][32*H][32*W] and reports that shape.
  bool decode(const float* latent, int F, int H, int W,
              std::vector<float>* out, std::array<int, 4>* shape,
              std::string* err);

  // Intermediates, kept when `capture` is on, keyed the way the goldens
  // are ("conv_in", "up1", ... "up8"). A whole-graph mismatch says
  // nothing about WHERE; these do.
  void set_capture(bool on) { _capture = on; }
  const std::vector<float>* tap(const std::string& name) const;
  bool tap_shape(const std::string& name, std::array<int, 4>* shape) const;

private:
  Ltx25VaeRef() = default;

  // One 3x3x3 convolution: [Cout][Cin][3][3][3] plus [Cout].
  struct Conv {
    std::vector<float> w, b;
    int cout = 0, cin = 0;
    bool ok() const { return cout > 0 && cin > 0; }
  };
  struct ResBlock { Conv conv1, conv2; };
  struct UpBlock {
    // Exactly one of these is populated.
    std::vector<ResBlock> res;
    Conv conv;                        // the depth-to-space projection
    int st = 1, sh = 1, sw = 1;       // its stride
    bool is_res = true;
  };

  bool load_conv_(vpipe::genai::WeightSet& ws,
                  vpipe::metal_compute::MetalCompute* mc,
                  const std::string& prefix, Conv& out, std::string* err);

  VaeConfig _cfg;
  Conv _conv_in, _conv_out;
  std::vector<UpBlock> _ups;
  std::vector<float> _std_of_means, _mean_of_means;

  bool _capture = false;
  std::vector<std::pair<std::string, std::vector<float>>> _taps;
  std::vector<std::pair<std::string, std::array<int, 4>>> _tap_shapes;
};

// The conv video VAE ENCODER as a from-scratch CPU reference, f32.
//
// The mirror of Ltx25VaeRef, and it exists for the same reason: every
// risky thing in it is LAYOUT, and each mistake produces a
// correctly-shaped latent that generates a plausible, wrong video.
//
//   * every convolution here is CAUSAL, unlike the decoder's. Causal
//     pads TWO copies of frame 0 at the FRONT and nothing at the back,
//     so a token never reads a later frame; the decoder's symmetric
//     replicate padding in the same place is a quiet edge-only error
//     that the middle of a clip hides.
//   * patchify nests `(c r q)` with q (HEIGHT) fastest, while the
//     space-to-depth downsample nests `(c p1 p2 p3)` with p3 (WIDTH)
//     fastest -- the same opposite-nesting trap the decoder has, from
//     the other side.
//   * SpaceToDepthDownsample is not just a strided conv: it is a
//     stride-1 conv, space-to-depth, PLUS a mean-pooled space-to-depth
//     of its own INPUT added as a skip. Dropping the skip runs and
//     encodes something.
//   * a time-halving block DUPLICATES frame 0 first, which is what makes
//     1 + 8k frames encode to 1 + k latent frames rather than losing
//     one at each of the three stages.
//   * the head emits 129 channels: 128 means and ONE shared log-variance
//     that is discarded. Taking the first 128 of a 129-wide tensor is
//     right; taking a 128-wide chunk of a doubled one is what the
//     per-channel variance mode would need, and this checkpoint is
//     `uniform`.
//   * the latent is WHITENED on the way out -- (x - mean) / std, the
//     exact inverse of the decoder's denormalise. An unwhitened latent
//     is off by the dataset's own scale and produces washed-out video.
class Ltx25VaeEncoderRef {
public:
  static std::unique_ptr<Ltx25VaeEncoderRef>
  load(const VaeConfig& cfg, vpipe::genai::WeightSet& ws,
       vpipe::metal_compute::MetalCompute* mc, std::string* err);

  // `pixels` is f32 [3][F][H][W] in [-1, 1]. F must be 1 + k*temporal,
  // H and W multiples of the spatial factor; anything else is refused
  // rather than cropped, because a stage that silently drops frames
  // hands the DiT a reference for a clip it is not generating.
  //
  // Writes the WHITENED latent [latent_channels][1 + (F-1)/8][H/32][W/32]
  // and reports that shape.
  bool encode(const float* pixels, int F, int H, int W,
              std::vector<float>* out, std::array<int, 4>* shape,
              std::string* err);

  // Intermediates, keyed "conv_in", "down0".."down8", "head".
  void set_capture(bool on) { _capture = on; }
  const std::vector<float>* tap(const std::string& name) const;
  bool tap_shape(const std::string& name, std::array<int, 4>* shape) const;

  int latent_channels() const { return _cfg.latent_channels; }

private:
  Ltx25VaeEncoderRef() = default;

  struct Conv {
    std::vector<float> w, b;
    int cout = 0, cin = 0;
  };
  struct ResBlock { Conv conv1, conv2; };
  struct DownBlock {
    std::vector<ResBlock> res;
    Conv conv;                        // the space-to-depth projection
    int st = 1, sh = 1, sw = 1;
    int group_size = 1;               // the skip's channel pooling
    bool is_res = true;
  };

  bool load_conv_(vpipe::genai::WeightSet& ws,
                  vpipe::metal_compute::MetalCompute* mc,
                  const std::string& prefix, Conv& out, std::string* err);

  VaeConfig _cfg;
  Conv _conv_in, _conv_out;
  std::vector<DownBlock> _downs;
  std::vector<float> _std_of_means, _mean_of_means;

  bool _capture = false;
  std::vector<std::pair<std::string, std::vector<float>>> _taps;
  std::vector<std::pair<std::string, std::array<int, 4>>> _tap_shapes;
};

}  // namespace ltx25

#endif
