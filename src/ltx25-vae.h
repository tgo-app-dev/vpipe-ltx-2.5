#ifndef VPIPE_LTX25_VAE_H
#define VPIPE_LTX25_VAE_H

#include "ltx25-metal-ops.h"
#include "ltx25-vae-config.h"

#include "generative-models/weight-set.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace ltx25 {

// The conv video VAE decoder on the GPU.
//
// Semantics are the CPU reference's (ltx25-vae-ref.h), which is verified
// against the reference implementation at ~1e-6 -- so this file is a
// port of a pinned specification, not a second reading of it.
//
// ---- CHANNEL-LAST, and why ----
//
// Everything between the two ends is [F][H][W][C]. The convolution runs
// as im2col + GEMM and a GEMM's output is [cells][C_out], which IS
// channel-last; PixelNorm reduces over channels, which becomes one
// contiguous row per cell; and the im2col gather reads consecutive
// channels in consecutive lanes. Channel-FIRST would need a transpose
// around every one of those.
//
// The latent arrives channel-first and the pixels leave channel-first,
// so exactly two kernels convert and nothing in between does.
//
// ---- MEMORY ----
//
// im2col is the large transient: cells x C x 27 elements. At 512x512 x 9
// frames that is ~1 GB for a single block if built whole, so the
// convolution is CHUNKED over output cells and the scratch is bounded by
// `max_im2col_elems()` rather than by the picture size. Bigger chunks
// are faster (one GEMM instead of several), so the bound is a memory
// ceiling, not a tuning knob to minimise.
class Ltx25VaeDecoder {
public:
  static std::unique_ptr<Ltx25VaeDecoder>
  load(const VaeConfig& cfg, std::shared_ptr<vpipe::genai::WeightSet> ws,
       const MetalOps& ops, std::string* err);

  // `latent` is f32 [latent_channels][F][H][W]. Writes f32
  // [3][8*(F-1)+1][32*H][32*W].
  bool decode(const float* latent, int F, int H, int W,
              std::vector<float>* out, std::array<int, 4>* shape,
              std::string* err);

  // The im2col chunk ceiling, in ELEMENTS. Default 64 M (128 MB at
  // bf16).
  void set_max_im2col_elems(std::size_t n) { _max_im2col = n; }
  std::size_t max_im2col_elems() const { return _max_im2col; }

  // Bytes this decoder holds (weights).
  std::uint64_t resident_bytes() const { return _resident; }

private:
  Ltx25VaeDecoder() = default;

  struct Conv {
    vpipe::metal_compute::SharedBuffer w, b;   // [Cout][Cin*27], [Cout]
    int cout = 0, cin = 0;
  };
  struct ResBlock { Conv conv1, conv2; };
  struct UpBlock {
    std::vector<ResBlock> res;
    Conv conv;
    int st = 1, sh = 1, sw = 1;
    bool is_res = true;
  };

  bool load_conv_(vpipe::genai::WeightSet& ws, const std::string& prefix,
                  Conv& out, std::string* err);

  // x[cells][cin] -> y[cells][cout], 3x3x3, chunked over cells.
  void conv3d_(vpipe::metal_compute::CommandStream& stream, const Conv& c,
               const vpipe::metal_compute::SharedBuffer& x, int F, int H,
               int W, const vpipe::metal_compute::SharedBuffer& y);

  VaeConfig _cfg;
  std::shared_ptr<vpipe::genai::WeightSet> _ws;
  const MetalOps* _ops = nullptr;
  Conv _conv_in, _conv_out;
  std::vector<UpBlock> _ups;
  vpipe::metal_compute::SharedBuffer _std_of_means, _mean_of_means;
  vpipe::metal_compute::SharedBuffer _im2col;
  std::size_t _max_im2col = 64ull * 1024 * 1024;
  std::uint64_t _resident = 0;
};

// The conv video VAE ENCODER on the GPU: pixels -> whitened latent.
//
// The mirror of Ltx25VaeDecoder, same channel-last convention and the
// same chunked im2col + GEMM convolution -- but CAUSAL in time
// everywhere, which is its own im2col rather than a flag on the shared
// one. Semantics are Ltx25VaeEncoderRef's, verified against the
// reference implementation at ~2e-6.
//
// WHY THIS EXISTS AT ALL. The image / video reference paths take a
// latent, and nothing else in this tree can produce an LTX one: the
// stock vae-encode's built-in families all emit 8x or 16x latents at
// 16 or 32 channels, and this VAE is 32x at 128. Without it those ports
// are wired to a producer that does not exist.
class Ltx25VaeEncoder {
public:
  static std::unique_ptr<Ltx25VaeEncoder>
  load(const VaeConfig& cfg, std::shared_ptr<vpipe::genai::WeightSet> ws,
       const MetalOps& ops, std::string* err);

  // `pixels` is f32 [3][F][H][W] in [-1, 1]. Writes the WHITENED latent
  // f32 [latent_channels][1 + (F-1)/8][H/32][W/32].
  //
  // A frame count that is not 1 + 8k, or a size that is not a multiple
  // of 32, is REFUSED rather than cropped -- a reference silently
  // trimmed is a reference for a clip the DiT is not generating.
  bool encode(const float* pixels, int F, int H, int W,
              std::vector<float>* out, std::array<int, 4>* shape,
              std::string* err);

  void set_max_im2col_elems(std::size_t n) { _max_im2col = n; }
  std::size_t max_im2col_elems() const { return _max_im2col; }

  std::uint64_t resident_bytes() const { return _resident; }
  int latent_channels() const { return _cfg.latent_channels; }
  int spatial_factor() const { return _cfg.spatial_factor(); }
  int temporal_factor() const { return _cfg.temporal_factor(); }

private:
  Ltx25VaeEncoder() = default;

  struct Conv {
    vpipe::metal_compute::SharedBuffer w, b;
    int cout = 0, cin = 0;
  };
  struct ResBlock { Conv conv1, conv2; };
  struct DownBlock {
    std::vector<ResBlock> res;
    Conv conv;
    int st = 1, sh = 1, sw = 1;
    int group_size = 1;
    bool is_res = true;
  };

  bool load_conv_(vpipe::genai::WeightSet& ws, const std::string& prefix,
                  Conv& out, std::string* err);
  void conv3d_(vpipe::metal_compute::CommandStream& stream, const Conv& c,
               const vpipe::metal_compute::SharedBuffer& x, int F, int H,
               int W, const vpipe::metal_compute::SharedBuffer& y);

  VaeConfig _cfg;
  std::shared_ptr<vpipe::genai::WeightSet> _ws;
  const MetalOps* _ops = nullptr;
  Conv _conv_in, _conv_out;
  std::vector<DownBlock> _downs;
  vpipe::metal_compute::SharedBuffer _std_of_means, _mean_of_means;
  vpipe::metal_compute::SharedBuffer _im2col;
  std::size_t _max_im2col = 64ull * 1024 * 1024;
  std::uint64_t _resident = 0;
};

}  // namespace ltx25

#endif
