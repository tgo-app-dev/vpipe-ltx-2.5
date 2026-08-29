#ifndef VPIPE_LTX25_METAL_OPS_H
#define VPIPE_LTX25_METAL_OPS_H

#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "generative-models/shared/i8-gemm.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace ltx25 {

// The op vocabulary the LTX-2.5 blocks are written in.
//
// Thin on purpose: each call is one dispatch with the argument order
// spelled out once, so the block code below reads as the reference's
// forward rather than as buffer bookkeeping. Nothing here is LTX-
// specific except the three kernels the plugin ships; the rest resolve
// out of libvpipe's own embedded libraries by name.
//
// EVERYTHING IS BF16 except the RoPE tables, which stay f32. The tables
// are angles that every block reuses and they cost O(tokens x dim/2);
// narrowing them would spend the model's precision where it is cheapest
// to keep.
// ONE linear's weights, dense or group-affine quantized.
//
// The two are one type because every GEMM in this model goes through
// MetalOps::linear, and a quantized checkpoint quantizes SOME tensors
// and not others -- the f32 scale/shift tables and the per-head gate
// logits stay dense in any pack. A per-call branch on `quantized` keeps
// the block code identical either way; a second parallel weight struct
// would put that branch at all seven block call sites instead.
//
// The quantized layout is vpipe's MLX-affine one, as `model-quantize`
// writes it and the `affine_qmm_steel_*` kernels read it: `codes` is
// u32-packed weights, `scales`/`biases` are f16 per group of `group`
// input columns. `bits` is per TENSOR, not per checkpoint, so a mixed
// 4/8 pack loads as it sits.
struct QWeight {
  vpipe::metal_compute::SharedBuffer w;                  // dense bf16
  vpipe::metal_compute::SharedBuffer codes, scales, qbias;
  int  bits = 0;                 // 4 or 8, per TENSOR
  int  group = 0;                // 32 or 64, per tensor
  bool quantized = false;

  bool empty() const noexcept
  {
    return quantized ? (codes.empty() || scales.empty() || qbias.empty())
                     : w.empty();
  }
};

class MetalOps {
public:
  // False (with `err` set) when a library or entry point is missing --
  // which on this path means a version skew between the plugin and the
  // libvpipe it loaded into, so it names what it could not find.
  bool init(vpipe::metal_compute::MetalCompute* mc, std::string* err);

  vpipe::metal_compute::MetalCompute* mc() const { return _mc; }

  // A bf16 scratch buffer of `n` elements.
  vpipe::metal_compute::SharedBuffer alloc(std::size_t n) const;
  // An f32 buffer, for the RoPE tables.
  vpipe::metal_compute::SharedBuffer alloc_f32(std::size_t n) const;
  // Upload host f32 as bf16 / as f32.
  vpipe::metal_compute::SharedBuffer
  upload_bf16(const std::vector<float>& v) const;
  vpipe::metal_compute::SharedBuffer
  upload_f32(const std::vector<float>& v) const;
  static std::vector<float>
  download_bf16(const vpipe::metal_compute::SharedBuffer& b, std::size_t n);

  // y[M][N] = x[M][K] @ W[N][K]^T (+ bias[N]). W is PyTorch's [out][in]
  // as it sits in the checkpoint, so nothing is transposed at load.
  void linear(vpipe::metal_compute::ComputeEncoder& enc,
              const vpipe::metal_compute::SharedBuffer& x,
              const vpipe::metal_compute::SharedBuffer& w,
              const vpipe::metal_compute::SharedBuffer* bias,
              const vpipe::metal_compute::SharedBuffer& y,
              int M, int K, int N) const;

  // The same, over a weight that may be quantized. Dense weights go
  // straight to the overload above; quantized ones dispatch the affine
  // qmm and then add the bias SEPARATELY, because the qmm kernel has no
  // bias slot.
  //
  // The kernel is picked from the weight's OWN bits and group, so a
  // mixed pack needs no state here. When the affine library did not
  // resolve at init this asserts rather than dispatching nothing --
  // an unvalidated ComputeFunction is a silent no-op, and for a GEMM
  // that means a block that runs at full speed over uninitialised
  // memory.
  void linear(vpipe::metal_compute::ComputeEncoder& enc,
              const vpipe::metal_compute::SharedBuffer& x,
              const QWeight& w,
              const vpipe::metal_compute::SharedBuffer* bias,
              const vpipe::metal_compute::SharedBuffer& y,
              int M, int K, int N) const;

  // y[M][N] += (x[M][K] @ A[rank][K]^T) @ B[N][rank]^T -- a low-rank
  // adapter's contribution to a linear that has ALREADY been computed
  // into `y`.
  //
  // Two ordinary GEMMs and an add, because that is all it is: `a` and
  // `b` sit in the same [out][in] order every other weight here does,
  // so the existing dense path serves both halves and nothing new is
  // dispatched. It is applied to the OUTPUT rather than folded into the
  // weight so a streamed or quantized base is untouched -- see
  // ltx25-lora.h for why that is not negotiable.
  //
  // `r` and `d` are CALLER scratch, [M][rank] and [M][N]. They are
  // passed in rather than owned here because the whole 48-block stack
  // shares one arena: an op that allocated its own would put a
  // 267 MB buffer behind every block.
  //
  // The cost is 2*rank/K of the linear it rides on -- 1.6% at rank 32,
  // ~22% at the distilled adapter's rank 450.
  void lora_add(vpipe::metal_compute::ComputeEncoder& enc,
                const vpipe::metal_compute::SharedBuffer& x,
                const vpipe::metal_compute::SharedBuffer& a,
                const vpipe::metal_compute::SharedBuffer& b,
                const vpipe::metal_compute::SharedBuffer& y,
                const vpipe::metal_compute::SharedBuffer& r,
                const vpipe::metal_compute::SharedBuffer& d,
                int M, int K, int N, int rank) const;

  // Will lora_add take the matrix-core pair for this shape, or its
  // portable linear+linear+add? Exposed because the two paths differ
  // only in speed, so a correctness test that did not ASSERT this would
  // pass while measuring the fallback -- which is what the shipped
  // shape table did: every rank in it is under the 16-column floor.
  bool lora_on_matrix_cores(int M, int K, int N, int rank) const noexcept;

  // Force the adapter onto one path or the other. The A/B hook for the
  // pair above, and the only way to measure it: VPIPE_LTX25_NO_LORA_MMA
  // is read once at init, so a process that wants both arms interleaved
  // -- which is the only ordering that measures anything, see
  // dense_mma_'s note -- has to toggle it per round.
  void set_lora_matrix_cores(bool on) noexcept { _lora_mma_off = !on; }

  // Turn the accelerated int8 tier on for this model. Rebuilds the
  // context, because whether it is wanted is a per-graph decision and
  // the env override is read in the constructor.
  void set_i8_gemm(bool on);

  // Will `linear` take the int8 tier for this shape? Same reason
  // lora_on_matrix_cores exists: the tiers compute the same product to
  // within their tolerance, so a test that did not ASSERT the path would
  // pass while measuring another one.
  bool i8_takes(int M, int K, int N) const noexcept;

  // Did the affine qmm kernels resolve at init? They are libvpipe's,
  // embedded in the host binary and reached by name, so this is only
  // false against a host too old to ship them -- and then a DENSE
  // checkpoint still runs, which is why init() does not fail on it. The
  // loader checks this before accepting a quantized checkpoint, so the
  // failure names the cause instead of surfacing as noise.
  bool quant_available() const noexcept { return _quant_ok; }

  // y[M][N] += bias[N], broadcast down the rows.
  void bias_add(vpipe::metal_compute::ComputeEncoder& enc,
                const vpipe::metal_compute::SharedBuffer& y,
                const vpipe::metal_compute::SharedBuffer& bias,
                int M, int N) const;

  // out = rms_norm(x) is NOT here: the reference's ada_zero fuses the
  // normalisation with the modulation, and splitting them would write
  // the intermediate. rms_norm(x) alone appears only as `post_sa`'s
  // second normalisation, which is `rms_norm_plain`.
  //
  // out[i] = (1 + scale[i % N]) * x[i] + shift[i % N]
  void modulate(vpipe::metal_compute::ComputeEncoder& enc,
                const vpipe::metal_compute::SharedBuffer& x,
                const vpipe::metal_compute::SharedBuffer& scale,
                const vpipe::metal_compute::SharedBuffer& shift,
                const vpipe::metal_compute::SharedBuffer& out,
                int N, int rows) const;

  // As modulate(), but reading `scale` and `shift` at byte OFFSETS into
  // one buffer. prompt_scale_shift_table is [2][dim] with shift FIRST,
  // so the two halves have to be bound explicitly -- relying on argument
  // order to line up with table order is how they get swapped.
  void modulate_off(vpipe::metal_compute::ComputeEncoder& enc,
                    const vpipe::metal_compute::SharedBuffer& x,
                    const vpipe::metal_compute::SharedBuffer& scale,
                    std::size_t scale_off,
                    const vpipe::metal_compute::SharedBuffer& shift,
                    std::size_t shift_off,
                    const vpipe::metal_compute::SharedBuffer& out,
                    int N, int rows) const;

  // out = rms_norm(x) * (1 + scale) + shift, fused. The reference's
  // ada_zero, and the block's most-used op -- composing libvpipe's
  // rms_norm and adaln_modulate would write a [rows][N] intermediate
  // every time.
  void ada_zero(vpipe::metal_compute::ComputeEncoder& enc,
                const vpipe::metal_compute::SharedBuffer& x,
                const vpipe::metal_compute::SharedBuffer& scale,
                const vpipe::metal_compute::SharedBuffer& shift,
                const vpipe::metal_compute::SharedBuffer& out,
                int N, int rows) const;

  // ---- the per-token modulation trio ---------------------------------
  //
  // The same three operations with `scale` / `shift` / `gate` read from
  // row `level[token]` of a [levels][N] table instead of broadcast. This
  // is how a conditioned generation modulates a token carrying given
  // content as clean while its neighbours are modulated at the
  // schedule's sigma -- see ltx25-kernels.metal for why it is a level
  // TABLE and not one vector per token.
  //
  // `level` is int32, one per row. Nothing calls these when a generation
  // has a single level: the block keeps the broadcast forms, so an
  // unconditioned forward is unchanged down to the dispatch.
  void ada_zero_g(vpipe::metal_compute::ComputeEncoder& enc,
                  const vpipe::metal_compute::SharedBuffer& x,
                  const vpipe::metal_compute::SharedBuffer& scale,
                  const vpipe::metal_compute::SharedBuffer& shift,
                  const vpipe::metal_compute::SharedBuffer& level,
                  const vpipe::metal_compute::SharedBuffer& out,
                  int N, int rows) const;

  void modulate_g(vpipe::metal_compute::ComputeEncoder& enc,
                  const vpipe::metal_compute::SharedBuffer& x,
                  const vpipe::metal_compute::SharedBuffer& scale,
                  std::size_t scale_off,
                  const vpipe::metal_compute::SharedBuffer& shift,
                  std::size_t shift_off,
                  const vpipe::metal_compute::SharedBuffer& level,
                  const vpipe::metal_compute::SharedBuffer& out,
                  int N, int rows, int level_stride) const;

  void gated_residual_g(vpipe::metal_compute::ComputeEncoder& enc,
                        const vpipe::metal_compute::SharedBuffer& h,
                        const vpipe::metal_compute::SharedBuffer& gate,
                        const vpipe::metal_compute::SharedBuffer& sub,
                        const vpipe::metal_compute::SharedBuffer& level,
                        int N, int rows) const;

  // x[t] += row[.] for the first `rows` tokens -- the keyframe absolute
  // position embedding. A row COUNT rather than a mask because the
  // marked set (the target's first latent frame) is contiguous at the
  // front of the sequence; see ltx25-kernels.metal.
  void add_row_prefix(vpipe::metal_compute::ComputeEncoder& enc,
                      const vpipe::metal_compute::SharedBuffer& x,
                      const vpipe::metal_compute::SharedBuffer& row,
                      int N, int rows) const;

  // out = LayerNorm(x) with NO affine -- mean-subtracting, unlike every
  // other normalisation in this model. The DiT's output head uses it
  // (`norm_out = LayerNorm(dim, elementwise_affine=False)`), and using
  // an RMSNorm there is a plausible image with the residual stream's
  // mean left in.
  void layer_norm_plain(vpipe::metal_compute::ComputeEncoder& enc,
                        const vpipe::metal_compute::SharedBuffer& x,
                        const vpipe::metal_compute::SharedBuffer& out,
                        int N, int rows) const;

  // out = rms_norm(x), OUT OF PLACE. `post_sa` keeps both the residual
  // sum and its normalisation, so the in-place form will not do.
  void rms_norm_out(vpipe::metal_compute::ComputeEncoder& enc,
                    const vpipe::metal_compute::SharedBuffer& x,
                    const vpipe::metal_compute::SharedBuffer& out,
                    int N, int rows) const;

  // out = a + b, each read at a byte offset. Builds the adaLN driver
  // rows: table[row] + timestep[row].
  void add(vpipe::metal_compute::ComputeEncoder& enc,
           const vpipe::metal_compute::SharedBuffer& a, std::size_t a_off,
           const vpipe::metal_compute::SharedBuffer& b, std::size_t b_off,
           const vpipe::metal_compute::SharedBuffer& out, int n,
           std::size_t out_off = 0) const;

  // x[t] <- registers[t % R] for every t >= n_valid. The connector's
  // padded-position substitution; see ltx25-kernels.metal.
  void fill_registers(vpipe::metal_compute::ComputeEncoder& enc,
                      const vpipe::metal_compute::SharedBuffer& x,
                      const vpipe::metal_compute::SharedBuffer& regs,
                      int dim, int n_registers, int n_valid,
                      int tokens) const;

  void copy(vpipe::metal_compute::ComputeEncoder& enc,
            const vpipe::metal_compute::SharedBuffer& src,
            const vpipe::metal_compute::SharedBuffer& dst, int n) const;

  // x <- rms_norm(x), row width N, no gain.
  void rms_norm_plain(vpipe::metal_compute::ComputeEncoder& enc,
                      const vpipe::metal_compute::SharedBuffer& x,
                      int N, int rows) const;

  // x <- rms_norm(x) * gain, row width N. The q/k norms: the width is
  // the WHOLE projection, not the head dim.
  void rms_norm_gain(vpipe::metal_compute::ComputeEncoder& enc,
                     const vpipe::metal_compute::SharedBuffer& x,
                     const vpipe::metal_compute::SharedBuffer& gain,
                     int N, int rows) const;

  // h[i] += gate[i % N] * sub[i]
  void gated_residual(vpipe::metal_compute::ComputeEncoder& enc,
                      const vpipe::metal_compute::SharedBuffer& h,
                      const vpipe::metal_compute::SharedBuffer& gate,
                      const vpipe::metal_compute::SharedBuffer& sub,
                      int N, int rows) const;

  void gelu(vpipe::metal_compute::ComputeEncoder& enc,
            const vpipe::metal_compute::SharedBuffer& x,
            const vpipe::metal_compute::SharedBuffer& out, int n) const;

  // LTX's rotary embedding, in place on TOKEN-MAJOR [T][H*D]. The tables
  // are f32 and PER HEAD ([H][T][D/2]) -- see ltx25-kernels.metal for
  // why libvpipe's rope_half_table cannot serve.
  void rope(vpipe::metal_compute::ComputeEncoder& enc,
            const vpipe::metal_compute::SharedBuffer& x,
            const vpipe::metal_compute::SharedBuffer& cos,
            const vpipe::metal_compute::SharedBuffer& sin,
            int heads, int tokens, int head_dim) const;

  // y[T][H*D] *= 2 * sigmoid(logits[T][H]) per head.
  void gate_heads(vpipe::metal_compute::ComputeEncoder& enc,
                  const vpipe::metal_compute::SharedBuffer& y,
                  const vpipe::metal_compute::SharedBuffer& logits,
                  int heads, int tokens, int head_dim) const;

  // [A][B][D] -> [B][A][D]. Used both ways around attention: the GEMMs
  // produce token-major and sdpa_full wants head-major.
  void transpose_abd(vpipe::metal_compute::ComputeEncoder& enc,
                     const vpipe::metal_compute::SharedBuffer& in,
                     const vpipe::metal_compute::SharedBuffer& out,
                     int A, int B, int D) const;

  // Full (non-causal) attention over HEAD-MAJOR q[H][Tq][D] and
  // k/v[H][Tkv][D], writing head-major out[H][Tq][D]. The DiT has no
  // causal mask on either stream -- every token attends every token --
  // so the causal variants would be both wrong and slower.
  //
  // This is the FALLBACK, kept as a correctness A/B and for a head_dim
  // steel does not carry. It runs one simdgroup per (head, query token)
  // over the whole key sequence -- O(Tq*Tkv) at a few percent of peak --
  // which at 8160 video tokens x 48 blocks is the whole step. Prefer
  // sdpa_steel below.
  void sdpa_full(vpipe::metal_compute::ComputeEncoder& enc,
                 const vpipe::metal_compute::SharedBuffer& q,
                 const vpipe::metal_compute::SharedBuffer& k,
                 const vpipe::metal_compute::SharedBuffer& v,
                 const vpipe::metal_compute::SharedBuffer& out,
                 int heads, int tq, int tkv, int head_dim) const;

  // ---- steel flash attention ----------------------------------------
  //
  // libvpipe's register-resident flash kernel, in the same head-major
  // layout sdpa_full already takes -- so this is a dispatch swap, not a
  // second data path.
  //
  // A plan is PER SHAPE because the kernel's edge handling is two
  // FUNCTION CONSTANTS (is the query length a whole number of query
  // tiles, is the key length a whole number of key tiles), so a
  // different shape is a differently specialised pipeline, not different
  // arguments. It also carries the AttnParams buffer that shape needs.
  // The caller caches plans; all 48 blocks meet the same handful of
  // shapes.
  //
  // UNLIKE the sibling DiTs this model attends across streams, so qL and
  // kL genuinely differ (text cross-attention, and both audio<->video
  // directions). The two lengths are therefore kept apart everywhere --
  // in the strides, in NK, and in constant 201, which comes from the KEY
  // length. A self-attention-only port can get away with kL = qL there;
  // this one cannot.
  struct SteelAttn {
    vpipe::metal_compute::ComputeFunction fn;
    vpipe::metal_compute::SharedBuffer    params;
    int heads = 0, tq = 0, tkv = 0, head_dim = 0;
    int bq = 0;                   // query tile, 32 (ALU) or 64 (M5 nax)
  };

  // Can steel serve this head dim? False without the library, under
  // VPIPE_LTX25_NO_STEEL_ATTN, or for a width with no entry point --
  // and then the caller stays on sdpa_full, which is correct and slow.
  bool steel_attn_available(int head_dim) const noexcept;

  // Build (or rebuild) `p` for one shape. False leaves the caller on
  // the fallback.
  bool steel_attn_plan(SteelAttn* p, int heads, int tq, int tkv,
                       int head_dim) const;

  void sdpa_steel(vpipe::metal_compute::ComputeEncoder& enc,
                  const SteelAttn& p,
                  const vpipe::metal_compute::SharedBuffer& q,
                  const vpipe::metal_compute::SharedBuffer& k,
                  const vpipe::metal_compute::SharedBuffer& v,
                  const vpipe::metal_compute::SharedBuffer& out) const;

  // Which attention kernel init() settled on, for the log line. One of
  // "steel-nax" (M5 matrix cores), "steel", or "scalar".
  const char* attn_kernel() const noexcept;

  // ---- the conv video VAE decoder ---------------------------------
  //
  // All of these work CHANNEL-LAST ([F][H][W][C]) except the two that
  // convert at the ends. See ltx25-vae.h for why.

  // latent [C][F][H][W] f32 -> [F][H][W][C] bf16, x*std + mean.
  void vae_denorm_in(vpipe::metal_compute::ComputeEncoder& enc,
                     const vpipe::metal_compute::SharedBuffer& latent,
                     const vpipe::metal_compute::SharedBuffer& stdv,
                     const vpipe::metal_compute::SharedBuffer& meanv,
                     const vpipe::metal_compute::SharedBuffer& out,
                     int C, int F, int H, int W) const;

  // [F][H][W][C] -> [n_cells][C*27] for one chunk of output cells.
  // Column order is [ci][kf][kh][kw], matching the checkpoint weight's
  // own [C_out][C_in][3][3][3] flattening, so no weight is permuted.
  void vae_im2col(vpipe::metal_compute::ComputeEncoder& enc,
                  const vpipe::metal_compute::SharedBuffer& x,
                  const vpipe::metal_compute::SharedBuffer& out,
                  int C, int F, int H, int W, int cell0, int n_cells) const;

  // PixelNorm(eps 1e-8) then SiLU, in place, per channel-last row.
  void vae_pixel_norm_silu(vpipe::metal_compute::ComputeEncoder& enc,
                           const vpipe::metal_compute::SharedBuffer& x,
                           int C, int cells) const;

  // y += x.
  void vae_add_into(vpipe::metal_compute::ComputeEncoder& enc,
                    const vpipe::metal_compute::SharedBuffer& y,
                    const vpipe::metal_compute::SharedBuffer& x,
                    std::size_t n) const;

  // depth-to-space + the dropped first frame.
  void vae_d2s(vpipe::metal_compute::ComputeEncoder& enc,
               const vpipe::metal_compute::SharedBuffer& x,
               const vpipe::metal_compute::SharedBuffer& out,
               int Cin, int F, int H, int W, int p1, int p2, int p3,
               int drop) const;

  // channel-last -> the channel-first f32 picture.
  void vae_unpatchify(vpipe::metal_compute::ComputeEncoder& enc,
                      const vpipe::metal_compute::SharedBuffer& x,
                      const vpipe::metal_compute::SharedBuffer& out,
                      int Cin, int F, int H, int W, int patch) const;

  // ---- the video VAE ENCODER ----------------------------------------
  //
  // The mirror of the four above. Only the TIME handling differs from
  // the decoder's shared ops, and it differs everywhere: every
  // convolution here is causal.

  // f32 [Cin][F][H][W] -> channel-last [F][H/p][W/p][Cin*p*p], with the
  // channel split HEIGHT-fastest.
  void vae_patchify_in(vpipe::metal_compute::ComputeEncoder& enc,
                       const vpipe::metal_compute::SharedBuffer& pixels,
                       const vpipe::metal_compute::SharedBuffer& out,
                       int Cin, int F, int H, int W, int patch) const;

  // im2col with CAUSAL time padding (two copies of frame 0 at the front,
  // none at the back). NOT a flag on vae_im2col: that is the hottest
  // gather in the VAE and a branch there costs more than a second entry
  // point.
  void vae_im2col_causal(vpipe::metal_compute::ComputeEncoder& enc,
                         const vpipe::metal_compute::SharedBuffer& x,
                         const vpipe::metal_compute::SharedBuffer& out,
                         int C, int F, int H, int W, int cell0,
                         int n_cells) const;

  // space-to-depth, WIDTH-fastest, with an optional channel-group mean
  // (`group` > 1 only on a downsample's skip path).
  void vae_s2d(vpipe::metal_compute::ComputeEncoder& enc,
               const vpipe::metal_compute::SharedBuffer& x,
               const vpipe::metal_compute::SharedBuffer& out,
               int Cin, int F, int H, int W, int p1, int p2, int p3,
               int group) const;

  // [F][H][W][C] -> [F+1][H][W][C], frame 0 duplicated.
  void vae_dup_frame0(vpipe::metal_compute::ComputeEncoder& enc,
                      const vpipe::metal_compute::SharedBuffer& x,
                      const vpipe::metal_compute::SharedBuffer& out,
                      int C, int F, int H, int W) const;

  // channel-last [F][H][W][Chead] -> f32 [C][F][H][W], keeping the first
  // C channels and whitening them.
  void vae_whiten_out(vpipe::metal_compute::ComputeEncoder& enc,
                      const vpipe::metal_compute::SharedBuffer& x,
                      const vpipe::metal_compute::SharedBuffer& stdv,
                      const vpipe::metal_compute::SharedBuffer& meanv,
                      const vpipe::metal_compute::SharedBuffer& out,
                      int Chead, int C, int F, int H, int W) const;

  // ---- the audio VAE decoder ---------------------------------------
  //
  // Channel-last [H][W][C] with H = FRAMES and W = MEL BINS. Only four
  // ops are its own; pixel_norm_silu, add_into and linear are shared
  // with the video decoder above, and the 1x1 nin_shortcut is a plain
  // linear (a 1x1 CausalConv2d pads by nothing).

  // latent [C][H][W] f32 -> [H][W][C] bf16. `stdv`/`meanv` are C*W
  // long and indexed ci*W + wi -- per (channel, mel bin), NOT per
  // channel. See ltx_audio_denorm_in.
  void audio_denorm_in(vpipe::metal_compute::ComputeEncoder& enc,
                       const vpipe::metal_compute::SharedBuffer& latent,
                       const vpipe::metal_compute::SharedBuffer& stdv,
                       const vpipe::metal_compute::SharedBuffer& meanv,
                       const vpipe::metal_compute::SharedBuffer& out,
                       int C, int H, int W) const;

  // [H][W][C] -> [n_cells][C*9], CAUSAL in height (all padding on top),
  // symmetric in width.
  void audio_im2col(vpipe::metal_compute::ComputeEncoder& enc,
                    const vpipe::metal_compute::SharedBuffer& x,
                    const vpipe::metal_compute::SharedBuffer& out,
                    int C, int H, int W, int cell0, int n_cells) const;

  // nearest 2x on both axes. The first-row drop is NOT here -- it
  // happens after the following conv, as a subview.
  void audio_up2x(vpipe::metal_compute::ComputeEncoder& enc,
                  const vpipe::metal_compute::SharedBuffer& x,
                  const vpipe::metal_compute::SharedBuffer& out,
                  int C, int H, int W) const;

  // ---- the audio VAE ENCODER ----------------------------------------
  //
  // The stride-1 convolutions reuse audio_im2col above: same causal
  // padding, same zero fill. Only these three are the encoder's own.

  // f32 [C][frames][mel] -> channel-last [frames][mel][C].
  void audio_mel_in(vpipe::metal_compute::ComputeEncoder& enc,
                    const vpipe::metal_compute::SharedBuffer& mel,
                    const vpipe::metal_compute::SharedBuffer& out,
                    int C, int H, int W) const;

  // im2col for the STRIDE-2 downsample, whose padding is (0, 1, 2, 0)
  // rather than the stride-1 convolutions' (1, 1, 2, 0).
  void audio_im2col_down(vpipe::metal_compute::ComputeEncoder& enc,
                         const vpipe::metal_compute::SharedBuffer& x,
                         const vpipe::metal_compute::SharedBuffer& out,
                         int C, int H, int W, int Wo, int cell0,
                         int n_cells) const;

  // The head: keep the first Z of Chead channels, patchify to rows
  // (column = c * W + f) and whiten. Writes f32 [H][Z * W].
  void audio_whiten_rows(vpipe::metal_compute::ComputeEncoder& enc,
                         const vpipe::metal_compute::SharedBuffer& x,
                         const vpipe::metal_compute::SharedBuffer& stdv,
                         const vpipe::metal_compute::SharedBuffer& meanv,
                         const vpipe::metal_compute::SharedBuffer& out,
                         int Chead, int Z, int H, int W) const;

  // channel-last -> the channel-first f32 spectrogram.
  void audio_out(vpipe::metal_compute::ComputeEncoder& enc,
                 const vpipe::metal_compute::SharedBuffer& x,
                 const vpipe::metal_compute::SharedBuffer& out,
                 int C, int H, int W) const;

private:
  vpipe::metal_compute::MetalCompute* _mc = nullptr;
  vpipe::metal_compute::ComputeLibrary _lib_gemm, _lib_elt, _lib_sdpa,
      _lib_ltx;
  vpipe::metal_compute::ComputeFunction _fn_gemm, _fn_modulate, _fn_gated,
      _fn_gelu, _fn_transpose, _fn_sdpa, _fn_rms, _fn_rope, _fn_gate_heads,
      _fn_rms_gain, _fn_ada_zero, _fn_rms_out, _fn_add, _fn_copy,
      _fn_layer_norm, _fn_fill_regs,
      _fn_ada_zero_g, _fn_modulate_g, _fn_gated_g, _fn_add_row_prefix,
      _fn_vae_denorm, _fn_vae_im2col, _fn_vae_pns, _fn_vae_add,
      _fn_vae_d2s, _fn_vae_unpatch,
      _fn_vae_patch_in, _fn_vae_im2col_c, _fn_vae_s2d, _fn_vae_dup,
      _fn_vae_whiten,
      _fn_aud_denorm, _fn_aud_im2col, _fn_aud_up2x, _fn_aud_out,
      _fn_aud_mel_in, _fn_aud_im2col_dn, _fn_aud_whiten,
      _fn_bias_add;

  // [bits 4|8][group 32|64][bm 32|64]. Resolved in init() and OPTIONAL
  // -- a host without them still runs a dense checkpoint.
  vpipe::metal_compute::ComputeLibrary  _lib_qmm;
  vpipe::metal_compute::ComputeFunction _fn_qmm[2][2][2];
  bool _quant_ok = false;

  // Steel flash attention, and its M5 matrix-core twin. Both OPTIONAL
  // for the same reason the qmm kernels are: a host that does not ship
  // them still runs, on sdpa_full.
  vpipe::metal_compute::ComputeLibrary _lib_attn, _lib_attn_nax;
  bool _steel_ok  = false;
  bool _attn_nax  = false;

  // The M5 matrix-core dense GEMM. Off on every pre-M5 GPU, where the
  // library's entry points are build-time stubs -- so this is gated on
  // the DEVICE capability, never on the load succeeding.
  //
  // Three N-region tiles, picked by K exactly as the sibling DiTs do:
  // 128 for the shallow projections, the TN=2 (512-wide) tile for the
  // mid band, 256 for deep K.
  vpipe::metal_compute::ComputeLibrary  _lib_dense_mma;
  vpipe::metal_compute::ComputeFunction _fn_dense_mma;
  vpipe::metal_compute::ComputeFunction _fn_dense_mma_deep;
  vpipe::metal_compute::ComputeFunction _fn_dense_mma_tn2;
  int _mma_min_m = 64;
  // VPIPE_LTX25_MMA_TILE forces one N-region (128|256|512) for every
  // shape, which is how the routing in dense_mma_ was measured.
  int _tile_force = 0;

  // Dequant-once: expand a quantized weight into `_w_deq` and run the
  // SAME dense matmul2d, instead of the fused dequant-in-loop affine
  // qmm. MEASURED on M5 at the DiT's own shapes: 2.7-3.5x over
  // affine_qmm_steel at both w4 and w8, agreeing to 4.6e-5 .. 8.4e-5
  // rel-L2. This is the single largest M5 lever in the model, because
  // the DiT ships quantized (bf16 is 39 GB) and its GEMMs dominate.
  //
  // The scratch is shared across every projection. That is safe only
  // because encoders are SERIAL (CommandStream::DispatchType::Serial),
  // so Metal's hazard tracking orders each dequant against the matmul
  // that reads it; a concurrent encoder would need one scratch per
  // in-flight GEMM.
  // ACCELERATED INT8, one tier below the dense matmul2d and taking the
  // SAME weight -- the dense one for an f16/bf16 checkpoint, `_w_deq`
  // for a quantized one, so it composes with dequant-once rather than
  // replacing it. The activation is quantized per (row, 512-group) on
  // the fly and the weight per (out-channel, 512-group) per call, so the
  // mode has no persistent memory cost.
  //
  // LOSSY -- int8, rel-L2 ~1e-2 per GEMM -- so it is OPT-IN and off by
  // default, exactly as in the in-tree DiTs. `i8_gemm` in the graph's
  // config turns it on; VPIPE_I8_GEMM=0|1 overrides either way, which is
  // how it was A/B'd.
  //
  // mutable because I8GemmContext::gemm grows its own scratches, and
  // every dispatch helper here is const.
  mutable std::unique_ptr<vpipe::genai::I8GemmContext> _i8;

  vpipe::metal_compute::ComputeLibrary  _lib_dequant;
  vpipe::metal_compute::ComputeFunction _fn_dequant[2][2];  // [w4|w8][g32|g64]
  mutable vpipe::metal_compute::SharedBuffer _w_deq;

  // The adapter's two halves on the matrix cores, folding the accumulate
  // into the second tile instead of materializing the delta.
  //
  // lora_add's portable form is linear + linear + add, and the add is
  // what costs: the second GEMM writes [M][N] to `d`, then the add reads
  // `d`, reads `y` and writes `y` -- 4 passes over the projection's full
  // output for a delta whose arithmetic is 2*rank/K of the linear it
  // rides on. At the feed-forward's N=16384 that scratch is as large as
  // `ff` itself.
  //
  // The `_acc` tile seeds its cooperative tensor FROM y and stores once,
  // so the delta lands in the register file and `d` is never written.
  // The `_scaled` tile is the A half, and at 64 wide it is the one that
  // fits the x2 upscaler's rank 32 without three quarters of the tile
  // hanging past N -- the plain 128-region tile computes four times the
  // columns that exist.
  //
  // The scale rides on A at LOAD here (see ltx25-lora.h), so the scaled
  // tile is dispatched at 1.0 and is being used for its SHAPE, not its
  // coefficient.
  //
  // Returns false for a shape it will not take -- a rank under the
  // 16-column floor, a missing entry point, or an operand past the 2 GB
  // line -- and lora_add then runs the portable form. VPIPE_LTX25_NO_LORA_MMA
  // forces that, which is how the pair was A/B'd.
  vpipe::metal_compute::ComputeFunction _fn_lora_a64, _fn_lora_a128;
  vpipe::metal_compute::ComputeFunction _fn_lora_b128, _fn_lora_b256;
  bool _lora_mma_off = false;
  bool lora_mma_(vpipe::metal_compute::ComputeEncoder& enc,
                 const vpipe::metal_compute::SharedBuffer& x,
                 const vpipe::metal_compute::SharedBuffer& a,
                 const vpipe::metal_compute::SharedBuffer& b,
                 const vpipe::metal_compute::SharedBuffer& y,
                 const vpipe::metal_compute::SharedBuffer& r,
                 int M, int K, int N, int rank) const;

  // Is this shape safe and worth running on the matrix cores? Checks the
  // tile threshold AND the 2 GB operand limit below.
  bool mma_eligible_(int M, int K, int N) const noexcept;

  // Dispatch the dense matmul2d over an ALREADY-DENSE weight, plus the
  // separate bias pass. Callers must have checked mma_eligible_ first.
  void dense_mma_(vpipe::metal_compute::ComputeEncoder& enc,
                  const vpipe::metal_compute::SharedBuffer& x,
                  const vpipe::metal_compute::SharedBuffer& w,
                  const vpipe::metal_compute::SharedBuffer* bias,
                  const vpipe::metal_compute::SharedBuffer& y,
                  int M, int K, int N) const;

  // Expand `w` into `_w_deq` and run dense_mma_ over it. False when the
  // shape is not eligible, the entry point is missing, or the scratch
  // could not be allocated -- and then the caller stays on the qmm.
  bool dequant_mma_(vpipe::metal_compute::ComputeEncoder& enc,
                    const vpipe::metal_compute::SharedBuffer& x,
                    const QWeight& w,
                    const vpipe::metal_compute::SharedBuffer* bias,
                    const vpipe::metal_compute::SharedBuffer& y,
                    int M, int K, int N) const;
};

}  // namespace ltx25

#endif
