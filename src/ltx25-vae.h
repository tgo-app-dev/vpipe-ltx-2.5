#ifndef VPIPE_LTX25_VAE_H
#define VPIPE_LTX25_VAE_H

#include "ltx25-metal-ops.h"
#include "ltx25-vae-config.h"

#include "generative-models/weight-set.h"

#include <array>
#include <functional>
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
// im2col is the large transient WITHIN one convolution: cells x C x 27
// elements. At 512x512 x 9 frames that is ~1 GB for a single block if
// built whole, so the convolution is CHUNKED over output cells and the
// scratch is bounded by `max_im2col_elems()` rather than by the picture
// size. Bigger chunks are faster (one GEMM instead of several), so the
// bound is a memory ceiling, not a tuning knob to minimise.
//
// ACROSS convolutions the transient is the feature map, and the rule
// that bounds it is: COMMIT AND WAIT AT EVERY BLOCK BOUNDARY, then reuse
// the buffers. An un-committed Metal command buffer RETAINS every
// resource it references, so a SharedBuffer whose destructor has already
// run stays allocated until that buffer completes -- which means a
// decoder that encodes its whole graph before committing holds every
// intermediate of every level at once, however carefully the C++ scopes
// are written.
//
// MEASURED at 960x544x121, decoding the whole graph into one command
// buffer: 32.6 GB peak process footprint, 20.3 GB of it still standing
// when the frame sink was called. Committing per block and reusing three
// slots takes the same decode to ~4 GB. The scaling is in the CELL
// count, so it is invisible at test geometries -- the same decoder at
// 9 frames of 64x64 moves 129 MB, of which 128 MB is the fixed im2col
// buffer.
class Ltx25VaeDecoder {
public:
  static std::unique_ptr<Ltx25VaeDecoder>
  load(const VaeConfig& cfg, std::shared_ptr<vpipe::genai::WeightSet> ws,
       const MetalOps& ops, std::string* err);

  // `latent` is f32 [latent_channels][F][H][W]. Writes f32
  // [3][8*(F-1)+1][32*H][32*W] into `out`, which is UMA memory the
  // caller may read directly -- a frame sink gets `contents()` without a
  // copy. At this model's real geometry that copy is 758 MB.
  //
  // `progress`, when set, is called once per UP BLOCK with (done,
  // total). That is the unit the decode already commits and waits on --
  // nine of them at this model's real geometry, tens of seconds of GPU
  // apiece -- so it is the only place inside a decode where a report
  // costs nothing and means something. Returning false cancels: the
  // decode stops at the next block boundary and returns false with
  // `err` saying so.
  bool decode(const float* latent, int F, int H, int W,
              vpipe::metal_compute::SharedBuffer* out,
              std::array<int, 4>* shape, std::string* err,
              const std::function<bool(int done, int total)>& progress = {});

  // The same decode, copied into host memory. For callers holding the
  // picture as a plain vector (the reference tests); the real path uses
  // the buffer form above.
  bool decode(const float* latent, int F, int H, int W,
              std::vector<float>* out, std::array<int, 4>* shape,
              std::string* err);

  // Drop what a decode leaves behind -- the im2col scratch, which is
  // `max_im2col_elems()` wide and outlives the call so consecutive beats
  // do not rebuild it. Everything else is already released per block.
  void release_idle();

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
