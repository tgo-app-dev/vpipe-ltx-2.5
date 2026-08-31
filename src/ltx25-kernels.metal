// The Metal kernels LTX-2.5 needs that libvpipe does not already have.
//
// DELIBERATELY SHORT. libvpipe's embedded libraries are reachable from a
// plugin by name (dense_gemm_bf16, rms_norm_bf16, llm_elementwise_bf16,
// attn_steel, ...), so the GEMMs, RMSNorm, adaLN modulation, gated
// residual, gelu-tanh feed-forward and flash attention are all reused.
// Only two operations are genuinely LTX's own, and both are here.
//
// VPIPE_ELT is set at compile time (bfloat for the DiT, half for a f16
// twin) exactly as the in-tree kernels do it.

#include <metal_stdlib>
using namespace metal;

#ifndef VPIPE_ELT
#define VPIPE_ELT bfloat
#endif

// ---------------------------------------------------------------------
// LTX rotary embedding: the half ("split") rotation over a PER-HEAD
// table, on TOKEN-MAJOR activations.
//
// libvpipe's rope_half_table_ftab does the same arithmetic, and cannot be
// used, for two reasons that are easy to miss:
//
//   * its table is SHARED across heads (indexed `t * half + i`), because
//     every other model in the tree gives all heads the same frequencies.
//     LTX splits one 2048-wide row of angles ACROSS the heads, so head h
//     uses the slice [h*half, (h+1)*half) -- a different table per head.
//   * it expects head-major [H][T][D]; this runs directly on the
//     [T][H*D] a GEMM produces, so there is no transpose in between.
//
// Using the shared-table kernel would run cleanly and give every head
// head 0's angles.
//
//   x    [T][H*D]  in/out
//   cos  [H][T][D/2]
//   sin  [H][T][D/2]
kernel void ltx_rope_half_perhead(
    device VPIPE_ELT*    x    [[buffer(0)]],
    const device float*  cosb [[buffer(1)]],
    const device float*  sinb [[buffer(2)]],
    constant int&        H    [[buffer(3)]],
    constant int&        T    [[buffer(4)]],
    constant int&        D    [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
  const int half_d = D / 2;
  const int i = (int)gid.x;
  if (i >= half_d) { return; }
  const int t = (int)gid.y;
  const int h = (int)gid.z;
  if (t >= T || h >= H) { return; }

  // Per-head table: h selects the slice, then (t, i) within it.
  const uint cb = ((uint)h * (uint)T + (uint)t) * (uint)half_d + (uint)i;
  const float c = cosb[cb];
  const float s = sinb[cb];

  // Token-major activation: token t's head h starts at t*H*D + h*D.
  const uint base = ((uint)t * (uint)H + (uint)h) * (uint)D + (uint)i;
  const float x1 = float(x[base]);
  const float x2 = float(x[base + (uint)half_d]);
  // Both halves are read before either is written: the rotation mixes
  // them, so an in-place update one at a time feeds the new first half
  // into the second half's term.
  x[base]                    = VPIPE_ELT(x1 * c - x2 * s);
  x[base + (uint)half_d]     = VPIPE_ELT(x2 * c + x1 * s);
}

// ---------------------------------------------------------------------
// Per-head attention gating: `out_head *= 2 * sigmoid(logit)`.
//
// `apply_gated_attention` is on for every attention in this model. The
// logits come from the attention's INPUT (a [T][H] projection computed
// by a GEMM before this), not from its output, and the scaling is
// applied BEFORE to_out. Both are easy to get subtly wrong and neither
// changes a shape.
//
// The factor 2 matters: sigmoid alone would halve every head on average,
// which reads as a model that has simply lost confidence.
//
//   y      [T][H*D]  in/out (the attention result)
//   logits [T][H]
kernel void ltx_gate_heads(
    device VPIPE_ELT*       y      [[buffer(0)]],
    const device VPIPE_ELT* logits [[buffer(1)]],
    constant int&           H      [[buffer(2)]],
    constant int&           T      [[buffer(3)]],
    constant int&           D      [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])
{
  const int d = (int)gid.x;
  if (d >= D) { return; }
  const int t = (int)gid.y;
  const int h = (int)gid.z;
  if (t >= T || h >= H) { return; }

  const float lg = float(logits[(uint)t * (uint)H + (uint)h]);
  const float g  = 2.0f / (1.0f + exp(-lg));
  const uint  i  = ((uint)t * (uint)H + (uint)h) * (uint)D + (uint)d;
  y[i] = VPIPE_ELT(float(y[i]) * g);
}

// ---------------------------------------------------------------------
// RMSNorm with a learnable gain over the WHOLE inner dim.
//
// libvpipe's rms_norm_fast normalises per ROW of a given width, which is
// what this is -- but the q/k norms here run over `H*D` (the full
// projection width) and not per head, so the row width passed matters.
// Kept as its own entry point rather than reusing rms_norm_fast because
// getting that argument wrong is silent: per-head normalisation is a
// different operator that lands close enough to look plausible.
//
// One threadgroup per row; `tg` must be a power of two <= 1024.
kernel void ltx_rms_norm_gain(
    device VPIPE_ELT*       x    [[buffer(0)]],
    const device VPIPE_ELT* gain [[buffer(1)]],
    constant int&           N    [[buffer(2)]],   // row width
    constant float&         eps  [[buffer(3)]],
    uint  gid  [[threadgroup_position_in_grid]],
    uint  lid  [[thread_position_in_threadgroup]],
    uint  tgsz [[threads_per_threadgroup]])
{
  threadgroup float part[256];
  device VPIPE_ELT* row = x + (uint)N * gid;

  float acc = 0.0f;
  for (uint i = lid; i < (uint)N; i += tgsz) {
    const float v = float(row[i]);
    acc += v * v;
  }
  part[lid] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = tgsz / 2; s > 0; s >>= 1) {
    if (lid < s) { part[lid] += part[lid + s]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float inv = rsqrt(part[0] / float(N) + eps);
  for (uint i = lid; i < (uint)N; i += tgsz) {
    row[i] = VPIPE_ELT(float(row[i]) * inv * float(gain[i]));
  }
}

// ---------------------------------------------------------------------
// ada_zero, fused: out = rms_norm(x) * (1 + scale) + shift.
//
// The reference's `PytorchAdaZeroFunction`, and the single most-used
// operation in the block -- six times per block, twice more per
// audio<->video direction. libvpipe has `adaln_modulate` (the affine
// half) and `rms_norm` separately, and composing them would write a
// full [tokens][dim] intermediate every time. `scale` and `shift` are
// one dim-wide row each, broadcast over tokens.
//
// One threadgroup per row.
kernel void ltx_ada_zero(
    const device VPIPE_ELT* x     [[buffer(0)]],
    const device VPIPE_ELT* scale [[buffer(1)]],
    const device VPIPE_ELT* shift [[buffer(2)]],
    device VPIPE_ELT*       out   [[buffer(3)]],
    constant int&           N     [[buffer(4)]],
    constant float&         eps   [[buffer(5)]],
    uint  gid  [[threadgroup_position_in_grid]],
    uint  lid  [[thread_position_in_threadgroup]],
    uint  tgsz [[threads_per_threadgroup]])
{
  threadgroup float part[256];
  const device VPIPE_ELT* xr = x + (uint)N * gid;
  device VPIPE_ELT*       o  = out + (uint)N * gid;

  float acc = 0.0f;
  for (uint i = lid; i < (uint)N; i += tgsz) {
    const float v = float(xr[i]);
    acc += v * v;
  }
  part[lid] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = tgsz / 2; s > 0; s >>= 1) {
    if (lid < s) { part[lid] += part[lid + s]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float inv = rsqrt(part[0] / float(N) + eps);
  for (uint i = lid; i < (uint)N; i += tgsz) {
    o[i] = VPIPE_ELT(float(xr[i]) * inv * (1.0f + float(scale[i]))
                     + float(shift[i]));
  }
}

// out = rms_norm(x), OUT OF PLACE and with no gain.
//
// `post_sa`'s second normalisation: the block keeps BOTH the residual
// sum and its normalisation (the first goes on down the residual
// stream, the second feeds the text cross-attention), so this cannot be
// the in-place rms_norm libvpipe provides.
kernel void ltx_rms_norm_out(
    const device VPIPE_ELT* x   [[buffer(0)]],
    device VPIPE_ELT*       out [[buffer(1)]],
    constant int&           N   [[buffer(2)]],
    constant float&         eps [[buffer(3)]],
    uint  gid  [[threadgroup_position_in_grid]],
    uint  lid  [[thread_position_in_threadgroup]],
    uint  tgsz [[threads_per_threadgroup]])
{
  threadgroup float part[256];
  const device VPIPE_ELT* xr = x + (uint)N * gid;
  device VPIPE_ELT*       o  = out + (uint)N * gid;

  float acc = 0.0f;
  for (uint i = lid; i < (uint)N; i += tgsz) {
    const float v = float(xr[i]);
    acc += v * v;
  }
  part[lid] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = tgsz / 2; s > 0; s >>= 1) {
    if (lid < s) { part[lid] += part[lid + s]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float inv = rsqrt(part[0] / float(N) + eps);
  for (uint i = lid; i < (uint)N; i += tgsz) {
    o[i] = VPIPE_ELT(float(xr[i]) * inv);
  }
}

// out[i] = a[i] + b[i], with both inputs read at a caller-set offset.
// Used to build the adaLN driver rows: `table[row] + timestep[row]`.
kernel void ltx_add(
    const device VPIPE_ELT* a   [[buffer(0)]],
    const device VPIPE_ELT* b   [[buffer(1)]],
    device VPIPE_ELT*       out [[buffer(2)]],
    constant int&           n   [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)n) { return; }
  out[gid] = VPIPE_ELT(float(a[gid]) + float(b[gid]));
}

// ---------------------------------------------------------------------
// THE PER-TOKEN MODULATION TRIO.
//
// LTX conditions a generation by giving each token its OWN noise level:
// `timesteps = denoise_mask * sigma`, so a token carrying given content
// (an image anchor, a reference clip) is modulated as if it were clean
// while its neighbours are modulated at the schedule's sigma. Everything
// downstream of the timestep MLP therefore becomes per-token.
//
// It is NOT one modulation vector per token. The mask takes a handful of
// distinct values -- 1.0 for what is being generated, `1 - strength` per
// conditioning item -- so the host runs the adaLN chain once per LEVEL
// and hands the GPU a [levels][dim] table plus an int per token. A
// per-token chain would be a 4096x36864 projection thousands of times
// over; this is the same arithmetic with the redundancy removed.
//
// `lvl` is one int per ROW. The unconditioned case never reaches these
// kernels -- the block keeps calling the broadcast versions above, so a
// text-to-video forward is byte-for-byte what it was.

// ada_zero with a per-token level: out = rms_norm(x) * (1 + scale) + shift
// where scale/shift are row `lvl[token]` of a [levels][N] table.
kernel void ltx_ada_zero_g(
    const device VPIPE_ELT* x     [[buffer(0)]],
    const device VPIPE_ELT* scale [[buffer(1)]],
    const device VPIPE_ELT* shift [[buffer(2)]],
    const device int*       lvl   [[buffer(3)]],
    device VPIPE_ELT*       out   [[buffer(4)]],
    constant int&           N     [[buffer(5)]],
    constant float&         eps   [[buffer(6)]],
    uint  gid  [[threadgroup_position_in_grid]],
    uint  lid  [[thread_position_in_threadgroup]],
    uint  tgsz [[threads_per_threadgroup]])
{
  threadgroup float part[256];
  const device VPIPE_ELT* xr = x + (uint)N * gid;
  device VPIPE_ELT*       o  = out + (uint)N * gid;
  const uint base = (uint)N * (uint)lvl[gid];

  float acc = 0.0f;
  for (uint i = lid; i < (uint)N; i += tgsz) {
    const float v = float(xr[i]);
    acc += v * v;
  }
  part[lid] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = tgsz / 2; s > 0; s >>= 1) {
    if (lid < s) { part[lid] += part[lid + s]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float inv = rsqrt(part[0] / float(N) + eps);
  for (uint i = lid; i < (uint)N; i += tgsz) {
    o[i] = VPIPE_ELT(float(xr[i]) * inv * (1.0f + float(scale[base + i]))
                     + float(shift[base + i]));
  }
}

// The affine half alone, per token: out = (1 + scale) * x + shift.
//
// `stride` is how far apart two LEVELS sit in the scale/shift table, and
// is not always N: the block's tables are [levels][dim] (stride N), while
// the output head's is [levels][2][dim] with the two rows interleaved by
// base offset, so a level there is 2*dim away. Deriving it from N would
// modulate the head's later levels from the middle of an earlier one.
kernel void ltx_modulate_g(
    const device VPIPE_ELT* x      [[buffer(0)]],
    const device VPIPE_ELT* scale  [[buffer(1)]],
    const device VPIPE_ELT* shift  [[buffer(2)]],
    const device int*       lvl    [[buffer(3)]],
    device VPIPE_ELT*       out    [[buffer(4)]],
    constant int&           N      [[buffer(5)]],
    constant int&           stride [[buffer(6)]],
    constant int&           total  [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)total) { return; }
  const uint col = gid % (uint)N;
  const uint row = gid / (uint)N;
  const uint b   = (uint)stride * (uint)lvl[row] + col;
  out[gid] = VPIPE_ELT((1.0f + float(scale[b])) * float(x[gid])
                       + float(shift[b]));
}

// h += gate * sub, with the gate row chosen per token.
kernel void ltx_gated_residual_g(
    device VPIPE_ELT*       h     [[buffer(0)]],
    const device VPIPE_ELT* gate  [[buffer(1)]],
    const device VPIPE_ELT* sub   [[buffer(2)]],
    const device int*       lvl   [[buffer(3)]],
    constant int&           N     [[buffer(4)]],
    constant int&           total [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)total) { return; }
  const uint col = gid % (uint)N;
  const uint row = gid / (uint)N;
  h[gid] = VPIPE_ELT(float(h[gid])
                     + float(gate[(uint)N * (uint)lvl[row] + col])
                       * float(sub[gid]));
}

// ---------------------------------------------------------------------
// x[t][d] += row[d], for t < rows.
//
// The keyframe absolute-position embedding: a single learned [1, dim]
// vector added, right after patchify_proj, to the tokens whose latent
// encodes ONE standalone pixel frame. Because the video VAE is causal
// that set is the target's first latent frame -- which is contiguous at
// the front of the sequence, so the marker is a row count rather than a
// mask. Appended conditioning tokens are never marked; only generated
// keyframe SLOTS are, and this port has none.
kernel void ltx_add_row_prefix(
    device VPIPE_ELT*       x    [[buffer(0)]],
    const device VPIPE_ELT* row  [[buffer(1)]],
    constant int&           N    [[buffer(2)]],
    constant int&           total [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)total) { return; }
  x[gid] = VPIPE_ELT(float(x[gid]) + float(row[gid % (uint)N]));
}

// Copy, for taking the pre-cross snapshot of a stream.
kernel void ltx_copy(
    const device VPIPE_ELT* src [[buffer(0)]],
    device VPIPE_ELT*       dst [[buffer(1)]],
    constant int&           n   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)n) { return; }
  dst[gid] = src[gid];
}

// ---------------------------------------------------------------------
// Replace PADDED positions with the connector's learnable registers.
//
//   x[t][d] = (t < n_valid) ? x[t][d] : registers[(t % R)][d]
//
// The register table is [R, D] and TILES over the sequence, which is why
// the caption is padded to a multiple of R (128). After this the
// reference throws the attention mask away entirely -- every position
// now holds a real vector, so the connector's attention is plain full
// attention. A port that keeps masking the padding instead computes
// something the model never does.
kernel void ltx_fill_registers(
    device VPIPE_ELT*       x    [[buffer(0)]],
    const device VPIPE_ELT* regs [[buffer(1)]],
    constant int&           D    [[buffer(2)]],
    constant int&           R    [[buffer(3)]],
    constant int&           n_valid [[buffer(4)]],
    constant int&           T    [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int d = (int)gid.x;
  const int t = (int)gid.y;
  if (d >= D || t >= T || t < n_valid) { return; }
  x[(uint)t * (uint)D + (uint)d] =
      regs[(uint)(t % R) * (uint)D + (uint)d];
}

// =====================================================================
// The conv video VAE decoder.
//
// Everything below works CHANNEL-LAST -- [F][H][W][C] -- while the
// checkpoint's latent and the decoded pixels are channel-first. That is
// not a stylistic choice:
//
//   * the convolution runs as im2col + GEMM, and the GEMM wants its
//     output as [cells][C_out], which IS channel-last;
//   * PixelNorm reduces over CHANNELS, which channel-last makes one
//     contiguous row per cell instead of a strided gather;
//   * the im2col gather then reads consecutive channels in consecutive
//     lanes, which is the coalesced direction.
//
// The two conversions live at the ends (ltx_vae_denorm_in on the way in,
// ltx_vae_unpatchify on the way out) and nothing in between transposes.
// =====================================================================

// Latent [C][F][H][W] f32 -> channel-last [F][H][W][C] bf16, applying
// the per-channel denormalization x*std + mean on the way.
// 0:latent(f32) 1:std(f32) 2:mean(f32) 3:out 4:C 5:F 6:H 7:W
kernel void ltx_vae_denorm_in(
    const device float*  latent [[buffer(0)]],
    const device float*  stdv   [[buffer(1)]],
    const device float*  meanv  [[buffer(2)]],
    device VPIPE_ELT*    out    [[buffer(3)]],
    constant int&        C      [[buffer(4)]],
    constant int&        F      [[buffer(5)]],
    constant int&        H      [[buffer(6)]],
    constant int&        W      [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int cell = (int)gid.y;             // f*H*W + h*W + w
  const int ci   = (int)gid.x;
  if (ci >= C || cell >= F * H * W) { return; }
  const float v = latent[(uint)ci * (uint)(F * H * W) + (uint)cell];
  out[(uint)cell * (uint)C + (uint)ci] =
      (VPIPE_ELT)(v * stdv[ci] + meanv[ci]);
}

// im2col for a 3x3x3 convolution over a chunk of output cells.
//
// Column order is [ci][kf][kh][kw] -- channel OUTER, tap inner -- because
// that is exactly how the checkpoint's [C_out][C_in][3][3][3] weight
// flattens. Any other column order needs the weight permuted at load,
// and a mismatch here is a GEMM that runs happily on scrambled inputs.
//
// Padding: REPLICATE in time (the frame index is clamped to the edge),
// ZERO in space. That asymmetry is the decoder's, not a simplification.
// 0:x 1:out 2:C 3:F 4:H 5:W 6:cell0 7:n_cells
kernel void ltx_vae_im2col(
    const device VPIPE_ELT* x       [[buffer(0)]],
    device VPIPE_ELT*       out     [[buffer(1)]],
    constant int&           C       [[buffer(2)]],
    constant int&           F       [[buffer(3)]],
    constant int&           H       [[buffer(4)]],
    constant int&           W       [[buffer(5)]],
    constant int&           cell0   [[buffer(6)]],
    constant int&           n_cells [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int ci = (int)gid.x;
  const int j  = (int)gid.y;               // cell within the chunk
  if (ci >= C || j >= n_cells) { return; }
  const int cell = cell0 + j;
  const int w0 = cell % W;
  const int h0 = (cell / W) % H;
  const int f0 = cell / (W * H);

  device VPIPE_ELT* dst = out + (uint)j * (uint)(C * 27) + (uint)ci * 27u;
  for (int kf = 0; kf < 3; ++kf) {
    int sf = f0 + kf - 1;
    sf = max(0, min(F - 1, sf));           // replicate in TIME
    for (int kh = 0; kh < 3; ++kh) {
      const int sh = h0 + kh - 1;
      for (int kw = 0; kw < 3; ++kw) {
        const int sw = w0 + kw - 1;
        const int t = (kf * 3 + kh) * 3 + kw;
        if (sh < 0 || sh >= H || sw < 0 || sw >= W) {
          dst[t] = (VPIPE_ELT)0.0f;        // zero in SPACE
        } else {
          dst[t] = x[(((uint)sf * (uint)H + (uint)sh) * (uint)W + (uint)sw)
                     * (uint)C + (uint)ci];
        }
      }
    }
  }
}

// PixelNorm + SiLU, fused, in place over channel-last rows.
//
// PixelNorm is x / sqrt(mean(x^2 over C) + 1e-8). The eps is the
// PixelNorm class DEFAULT -- build_normalization_layer's 1e-6 is not
// what the resnet blocks or conv_norm_out construct.
//
// One threadgroup per cell; the reduction is over the row.
// 0:x 1:C
kernel void ltx_vae_pixel_norm_silu(
    device VPIPE_ELT* x [[buffer(0)]],
    constant int&     C [[buffer(1)]],
    uint  tgid [[threadgroup_position_in_grid]],
    uint  lid  [[thread_index_in_threadgroup]],
    uint  nth  [[threads_per_threadgroup]])
{
  device VPIPE_ELT* row = x + (uint)tgid * (uint)C;
  threadgroup float part[256];
  float ss = 0.0f;
  for (uint i = lid; i < (uint)C; i += nth) {
    const float v = (float)row[i];
    ss += v * v;
  }
  part[lid] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = nth / 2; s > 0; s >>= 1) {
    if (lid < s) { part[lid] += part[lid + s]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float inv = 1.0f / sqrt(part[0] / (float)C + 1e-8f);
  for (uint i = lid; i < (uint)C; i += nth) {
    const float v = (float)row[i] * inv;
    row[i] = (VPIPE_ELT)(v / (1.0f + exp(-v)));   // SiLU
  }
}

// y += x, the resnet shortcut (Identity here: in == out everywhere, so
// the checkpoint carries no conv_shortcut / norm3 tensors).
// 0:y 1:x 2:N
kernel void ltx_vae_add_into(
    device VPIPE_ELT*       y [[buffer(0)]],
    const device VPIPE_ELT* x [[buffer(1)]],
    constant int&           N [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)N) { return; }
  y[gid] = (VPIPE_ELT)((float)y[gid] + (float)x[gid]);
}

// depth-to-space: `(c p1 p2 p3) f h w -> c (f p1) (h p2) (w p3)`, on
// channel-last data, with the FIRST output frame dropped when p1 == 2.
//
// p3 (WIDTH) is the FASTEST channel axis, then p2 (height), then p1
// (time). ltx_vae_unpatchify below nests the OPPOSITE way; reusing one
// for the other is wrong exactly half the time and both produce a
// correctly-shaped tensor.
//
// Indexed by DESTINATION so every output element is written exactly once.
// 0:x 1:out 2:Cin 3:F 4:H 5:W 6:p1 7:p2 8:p3 9:drop
kernel void ltx_vae_d2s(
    const device VPIPE_ELT* x    [[buffer(0)]],
    device VPIPE_ELT*       out  [[buffer(1)]],
    constant int&           Cin  [[buffer(2)]],
    constant int&           F    [[buffer(3)]],
    constant int&           H    [[buffer(4)]],
    constant int&           W    [[buffer(5)]],
    constant int&           p1   [[buffer(6)]],
    constant int&           p2   [[buffer(7)]],
    constant int&           p3   [[buffer(8)]],
    constant int&           drop [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int Cout = Cin / (p1 * p2 * p3);
  const int Fo = F * p1 - drop, Ho = H * p2, Wo = W * p3;
  const int co   = (int)gid.x;
  const int ocell = (int)gid.y;
  if (co >= Cout || ocell >= Fo * Ho * Wo) { return; }
  const int wo = ocell % Wo;
  const int ho = (ocell / Wo) % Ho;
  const int fo = ocell / (Wo * Ho);

  const int fs = fo + drop;                 // undo the dropped frame
  const int a = fs % p1, f = fs / p1;
  const int b = ho % p2, h = ho / p2;
  const int d = wo % p3, w = wo / p3;
  const int ci = ((co * p1 + a) * p2 + b) * p3 + d;
  out[(uint)ocell * (uint)Cout + (uint)co] =
      x[(((uint)f * (uint)H + (uint)h) * (uint)W + (uint)w) * (uint)Cin
        + (uint)ci];
}

// unpatchify: `(c p r q) f h w -> c f (h q) (w r)` with p == 1, reading
// channel-last and writing the CHANNEL-FIRST f32 picture.
//
// q (HEIGHT) is the FASTEST channel axis and r is width -- the opposite
// nesting to ltx_vae_d2s.
// 0:x 1:out(f32) 2:Cin 3:F 4:H 5:W 6:patch
// ---------------------------------------------------------------------
// THE ENCODER'S FOUR KERNELS.
//
// Everything between them is shared with the decoder -- im2col + GEMM,
// pixel_norm_silu, add_into -- because the two graphs are the same
// graph run in opposite directions. What is NOT shared is time: the
// encoder is CAUSAL everywhere and the decoder is not, so it needs its
// own im2col rather than a flag on the shared one (a flag would put a
// branch in the hottest gather in the model).

// patchify: f32 [3][F][H][W] -> bf16 channel-last [F][H/p][W/p][3*p*p].
//
// Channel = (c * p + r) * p + q with q (HEIGHT) FASTEST -- the exact
// inverse of ltx_vae_unpatchify's split, and the OPPOSITE nesting to
// ltx_vae_s2d below. Getting the two the same way round is the single
// most likely way to produce a correctly-shaped wrong latent.
kernel void ltx_vae_patchify_in(
    const device float*  pixels [[buffer(0)]],
    device VPIPE_ELT*    out    [[buffer(1)]],
    constant int&        Cin    [[buffer(2)]],
    constant int&        F      [[buffer(3)]],
    constant int&        H      [[buffer(4)]],   // pixel height
    constant int&        W      [[buffer(5)]],
    constant int&        patch  [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int Ho = H / patch, Wo = W / patch;
  const int Cout = Cin * patch * patch;
  const int co   = (int)gid.x;
  const int ocell = (int)gid.y;
  if (co >= Cout || ocell >= F * Ho * Wo) { return; }
  const int wo = ocell % Wo;
  const int ho = (ocell / Wo) % Ho;
  const int f  = ocell / (Wo * Ho);
  const int q  = co % patch;                 // height offset
  const int r  = (co / patch) % patch;       // width offset
  const int ci = co / (patch * patch);
  out[(uint)ocell * (uint)Cout + (uint)co] =
      (VPIPE_ELT)pixels[((uint)ci * (uint)F + (uint)f) * (uint)(H * W)
                        + (uint)((ho * patch + q) * W + wo * patch + r)];
}

// im2col with CAUSAL time padding: two copies of frame 0 at the front
// and nothing at the back, so the time offset is `kf - 2` and only the
// low end clamps. Space is zero-padded, as in the decoder's.
//
// A token here never reads a later frame. Reusing the decoder's
// symmetric im2col instead is wrong only near the clip's start, which
// is exactly where a single-frame test cannot tell.
kernel void ltx_vae_im2col_causal(
    const device VPIPE_ELT* x       [[buffer(0)]],
    device VPIPE_ELT*       out     [[buffer(1)]],
    constant int&           C       [[buffer(2)]],
    constant int&           F       [[buffer(3)]],
    constant int&           H       [[buffer(4)]],
    constant int&           W       [[buffer(5)]],
    constant int&           cell0   [[buffer(6)]],
    constant int&           n_cells [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int ci = (int)gid.x;
  const int j  = (int)gid.y;
  if (ci >= C || j >= n_cells) { return; }
  const int cell = cell0 + j;
  const int w0 = cell % W;
  const int h0 = (cell / W) % H;
  const int f0 = cell / (W * H);

  device VPIPE_ELT* dst = out + (uint)j * (uint)(C * 27) + (uint)ci * 27u;
  for (int kf = 0; kf < 3; ++kf) {
    const int sf = max(0, f0 + kf - 2);      // causal: replicate frame 0
    for (int kh = 0; kh < 3; ++kh) {
      const int sh = h0 + kh - 1;
      for (int kw = 0; kw < 3; ++kw) {
        const int sw = w0 + kw - 1;
        const int t = (kf * 3 + kh) * 3 + kw;
        if (sh < 0 || sh >= H || sw < 0 || sw >= W) {
          dst[t] = (VPIPE_ELT)0.0f;
        } else {
          dst[t] = x[(((uint)sf * (uint)H + (uint)sh) * (uint)W + (uint)sw)
                     * (uint)C + (uint)ci];
        }
      }
    }
  }
}

// space-to-depth: `b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w`,
// channel-last on both sides. p3 (WIDTH) is the fastest channel axis --
// the inverse of ltx_vae_d2s.
//
// `group` averages consecutive channel groups after the regroup, which
// is only ever used on the SKIP path -- the conv path passes 1. That
// mean is the whole skip: there is no projection to bind, which is why
// leaving it out produces a model that loads cleanly and encodes
// something plausible.
kernel void ltx_vae_s2d(
    const device VPIPE_ELT* x     [[buffer(0)]],
    device VPIPE_ELT*       out   [[buffer(1)]],
    constant int&           Cin   [[buffer(2)]],
    constant int&           F     [[buffer(3)]],
    constant int&           H     [[buffer(4)]],
    constant int&           W     [[buffer(5)]],
    constant int&           p1    [[buffer(6)]],
    constant int&           p2    [[buffer(7)]],
    constant int&           p3    [[buffer(8)]],
    constant int&           group [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int Fo = F / p1, Ho = H / p2, Wo = W / p3;
  const int Cfull = Cin * p1 * p2 * p3;
  const int Cout = Cfull / group;
  const int co    = (int)gid.x;
  const int ocell = (int)gid.y;
  if (co >= Cout || ocell >= Fo * Ho * Wo) { return; }
  const int wo = ocell % Wo;
  const int ho = (ocell / Wo) % Ho;
  const int fo = ocell / (Wo * Ho);

  float acc = 0.0f;
  for (int g = 0; g < group; ++g) {
    const int cf = co * group + g;           // index into the regrouped C
    const int d  = cf % p3;
    const int b  = (cf / p3) % p2;
    const int a  = (cf / (p3 * p2)) % p1;
    const int ci = cf / (p1 * p2 * p3);
    const int sf = fo * p1 + a;
    acc += (float)x[(((uint)sf * (uint)H + (uint)(ho * p2 + b)) * (uint)W
                     + (uint)(wo * p3 + d)) * (uint)Cin + (uint)ci];
  }
  out[(uint)ocell * (uint)Cout + (uint)co] = (VPIPE_ELT)(acc / (float)group);
}

// Prepend a copy of frame 0: [F][H][W][C] -> [F+1][H][W][C].
//
// What a time-halving block does to its input before BOTH its conv and
// its skip see it, and the reason 1 + 8k frames survive three halvings
// as 1 + k rather than losing one at each. Materialised rather than
// folded into the two readers as an index shift, because the conv
// reader is an im2col whose time indices are already causal -- putting a
// second offset in there is how the two halves of one block come to
// disagree about which frame is which.
kernel void ltx_vae_dup_frame0(
    const device VPIPE_ELT* x   [[buffer(0)]],
    device VPIPE_ELT*       out [[buffer(1)]],
    constant int&           C   [[buffer(2)]],
    constant int&           F   [[buffer(3)]],
    constant int&           H   [[buffer(4)]],
    constant int&           W   [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int ci   = (int)gid.x;
  const int ocell = (int)gid.y;
  if (ci >= C || ocell >= (F + 1) * H * W) { return; }
  const int rest = ocell % (H * W);
  const int fo   = ocell / (H * W);
  const int sf   = (fo == 0) ? 0 : fo - 1;
  out[(uint)ocell * (uint)C + (uint)ci] =
      x[((uint)sf * (uint)(H * W) + (uint)rest) * (uint)C + (uint)ci];
}

// The encoder's head: channel-last [F][H][W][Chead] -> f32 channel-first
// [C][F][H][W], keeping the first C channels and WHITENING them.
//
// Chead is 129 -- 128 means and ONE shared log-variance the `uniform`
// mode discards. Keeping a PREFIX is right; taking a half of a doubled
// tensor is what the per-channel-variance mode would need, and this
// checkpoint is not that.
kernel void ltx_vae_whiten_out(
    const device VPIPE_ELT* x     [[buffer(0)]],
    const device float*     stdv  [[buffer(1)]],
    const device float*     meanv [[buffer(2)]],
    device float*           out   [[buffer(3)]],
    constant int&           Chead [[buffer(4)]],
    constant int&           C     [[buffer(5)]],
    constant int&           F     [[buffer(6)]],
    constant int&           H     [[buffer(7)]],
    constant int&           W     [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int ci   = (int)gid.x;
  const int cell = (int)gid.y;
  if (ci >= C || cell >= F * H * W) { return; }
  const float v = (float)x[(uint)cell * (uint)Chead + (uint)ci];
  const float s = stdv[ci];
  out[(uint)ci * (uint)(F * H * W) + (uint)cell] =
      (v - meanv[ci]) / (s != 0.0f ? s : 1.0f);
}

kernel void ltx_vae_unpatchify(
    const device VPIPE_ELT* x     [[buffer(0)]],
    device float*           out   [[buffer(1)]],
    constant int&           Cin   [[buffer(2)]],
    constant int&           F     [[buffer(3)]],
    constant int&           H     [[buffer(4)]],
    constant int&           W     [[buffer(5)]],
    constant int&           patch [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int Cout = Cin / (patch * patch);
  const int Ho = H * patch, Wo = W * patch;
  const int co    = (int)gid.x;
  const int ocell = (int)gid.y;
  if (co >= Cout || ocell >= F * Ho * Wo) { return; }
  const int wo = ocell % Wo;
  const int ho = (ocell / Wo) % Ho;
  const int f  = ocell / (Wo * Ho);
  const int q = ho % patch, h = ho / patch;
  const int r = wo % patch, w = wo / patch;
  const int ci = (co * patch + r) * patch + q;
  out[((uint)co * (uint)F + (uint)f) * (uint)(Ho * Wo)
      + (uint)(ho * Wo + wo)] =
      (float)x[(((uint)f * (uint)H + (uint)h) * (uint)W + (uint)w)
               * (uint)Cin + (uint)ci];
}

// =====================================================================
// The AUDIO VAE decoder.
//
// Same channel-last idea as the video decoder above, one axis shorter:
// everything between the ends is [H][W][C] with H = FRAMES (time) and
// W = MEL BINS. That the tensor is (b, c, frames, mel_bins) is what
// makes torch's "height" the time axis, so `causality_axis: height`
// means causal in TIME -- and the padding below is asymmetric in H and
// symmetric in W for exactly that reason.
//
// Three of the video decoder's kernels serve unchanged here
// (pixel_norm_silu, add_into, and the GEMM behind `linear`), and the
// 1x1 `nin_shortcut` is a plain `linear` because a 1x1 CausalConv2d
// pads by nothing at all.
// =====================================================================

// Latent [C][H][W] f32 -> channel-last [H][W][C] bf16, un-normalising
// on the way.
//
// THE STATISTICS ARE NOT PER CHANNEL. The reference denormalises the
// PATCHIFIED latent, and AudioPatchifier.patchify is
// `b c t f -> b t (c f)` -- so the 128 statistics are indexed
// ci*W + wi, channel-major with the latent mel bin FASTEST, and this is
// a per-(channel, mel-bin) affine. Reading them as 8 per-channel values
// denormalises with the wrong constants and still produces a perfectly
// plausible spectrogram, which is why there is a golden for exactly
// this tensor.
// 0:latent(f32) 1:std(f32) 2:mean(f32) 3:out 4:C 5:H 6:W
kernel void ltx_audio_denorm_in(
    const device float*  latent [[buffer(0)]],
    const device float*  stdv   [[buffer(1)]],
    const device float*  meanv  [[buffer(2)]],
    device VPIPE_ELT*    out    [[buffer(3)]],
    constant int&        C      [[buffer(4)]],
    constant int&        H      [[buffer(5)]],
    constant int&        W      [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int cell = (int)gid.y;             // h*W + w
  const int ci   = (int)gid.x;
  if (ci >= C || cell >= H * W) { return; }
  const int wi = cell % W;                 // the mel bin
  const int s  = ci * W + wi;              // (c f), mel FASTEST
  const float v = latent[(uint)ci * (uint)(H * W) + (uint)cell];
  out[(uint)cell * (uint)C + (uint)ci] =
      (VPIPE_ELT)(v * stdv[s] + meanv[s]);
}

// im2col for a 3x3 convolution over a chunk of output cells.
//
// Column order is [ci][kh][kw] -- channel OUTER -- because that is how
// the checkpoint's [C_out][C_in][3][3] weight flattens, so no weight is
// permuted at load.
//
// PADDING IS THE CAUSAL ONE. CausalConv2d with causality_axis=HEIGHT
// builds the F.pad tuple (pad_w/2, pad_w-pad_w/2, pad_h, 0): all of the
// height padding on TOP and none at the bottom, width symmetric. So the
// source row is h0 + kh - 2, not h0 + kh - 1. Using the symmetric form
// shifts the whole spectrogram one frame in time and still decodes to
// something that sounds like audio.
// 0:x 1:out 2:C 3:H 4:W 5:cell0 6:n_cells
kernel void ltx_audio_im2col(
    const device VPIPE_ELT* x       [[buffer(0)]],
    device VPIPE_ELT*       out     [[buffer(1)]],
    constant int&           C       [[buffer(2)]],
    constant int&           H       [[buffer(3)]],
    constant int&           W       [[buffer(4)]],
    constant int&           cell0   [[buffer(5)]],
    constant int&           n_cells [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int ci = (int)gid.x;
  const int j  = (int)gid.y;
  if (ci >= C || j >= n_cells) { return; }
  const int cell = cell0 + j;
  const int w0 = cell % W;
  const int h0 = cell / W;

  device VPIPE_ELT* dst = out + (uint)j * (uint)(C * 9) + (uint)ci * 9u;
  for (int kh = 0; kh < 3; ++kh) {
    const int sh = h0 + kh - 2;            // ALL padding on top
    for (int kw = 0; kw < 3; ++kw) {
      const int sw = w0 + kw - 1;          // symmetric in width
      const int t = kh * 3 + kw;
      if (sh < 0 || sh >= H || sw < 0 || sw >= W) {
        dst[t] = (VPIPE_ELT)0.0f;          // zero on BOTH axes
      } else {
        dst[t] = x[((uint)sh * (uint)W + (uint)sw) * (uint)C + (uint)ci];
      }
    }
  }
}

// Nearest-neighbour 2x on BOTH axes: [H][W][C] -> [2H][2W][C].
//
// The reference's Upsample interpolates first and convolves after, and
// then drops the first ROW of the CONVOLVED result -- so the drop is
// not here. It costs nothing on the host side: channel-last rows are
// contiguous, so dropping row 0 is a subview at offset W*C.
// 0:x 1:out 2:C 3:H 4:W
// ---------------------------------------------------------------------
// THE AUDIO ENCODER'S THREE KERNELS.
//
// Everything else it needs is the decoder's: ltx_audio_im2col (stride 1,
// causal in TIME and zero-padded on both axes) and pixel_norm_silu.

// The log-mel in: f32 [C][T][mel] -> bf16 channel-last [T][mel][C].
kernel void ltx_audio_mel_in(
    const device float* mel [[buffer(0)]],
    device VPIPE_ELT*   out [[buffer(1)]],
    constant int&       C   [[buffer(2)]],
    constant int&       H   [[buffer(3)]],   // frames
    constant int&       W   [[buffer(4)]],   // mel bins
    uint2 gid [[thread_position_in_grid]])
{
  const int ci   = (int)gid.x;
  const int cell = (int)gid.y;
  if (ci >= C || cell >= H * W) { return; }
  out[(uint)cell * (uint)C + (uint)ci] =
      (VPIPE_ELT)mel[(uint)ci * (uint)(H * W) + (uint)cell];
}

// im2col for the STRIDE-2 downsample, which pads differently from every
// other convolution here.
//
// The reference pads (left 0, right 1, top 2, bottom 0) and then runs a
// stride-2 kernel with no padding of its own. So output (i, j) reads
// input rows 2i-2 .. 2i and input columns 2j .. 2j+2, zero outside --
// causal in TIME (nothing later than 2i) and one-sided in FREQUENCY.
// Reusing the stride-1 im2col with a doubled step instead gets the
// column offsets wrong by one and shifts the spectrum half a bin.
kernel void ltx_audio_im2col_down(
    const device VPIPE_ELT* x       [[buffer(0)]],
    device VPIPE_ELT*       out     [[buffer(1)]],
    constant int&           C       [[buffer(2)]],
    constant int&           H       [[buffer(3)]],   // INPUT frames
    constant int&           W       [[buffer(4)]],   // INPUT mel bins
    constant int&           Wo      [[buffer(5)]],   // OUTPUT mel bins
    constant int&           cell0   [[buffer(6)]],
    constant int&           n_cells [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int ci = (int)gid.x;
  const int j  = (int)gid.y;
  if (ci >= C || j >= n_cells) { return; }
  const int cell = cell0 + j;
  const int wo = cell % Wo;
  const int ho = cell / Wo;

  device VPIPE_ELT* dst = out + (uint)j * (uint)(C * 9) + (uint)ci * 9u;
  for (int kh = 0; kh < 3; ++kh) {
    const int sh = 2 * ho + kh - 2;
    for (int kw = 0; kw < 3; ++kw) {
      const int sw = 2 * wo + kw;
      const int t = kh * 3 + kw;
      if (sh < 0 || sh >= H || sw < 0 || sw >= W) {
        dst[t] = (VPIPE_ELT)0.0f;
      } else {
        dst[t] = x[((uint)sh * (uint)W + (uint)sw) * (uint)C + (uint)ci];
      }
    }
  }
}

// The head: channel-last [T][mel][Chead] -> f32 ROWS [T][z * mel].
//
// Three things at once, because they are one indexing question:
//   * keep the first `z` of the head's `Chead` channels. Chead is 2z:
//     the means and a log-variance the encoder discards;
//   * patchify `b c t f -> b t (c f)`, so the row index is t and the
//     column is c * mel + f -- the layout generate-video's
//     `ref_audio_rows` takes and the one the DiT's audio patchify
//     assumes;
//   * WHITEN with the per-(channel, mel-bin) statistics, which are
//     `z * mel` long and indexed by that same c * mel + f. They are NOT
//     per channel, which is the trap the decoder's denorm documents.
kernel void ltx_audio_whiten_rows(
    const device VPIPE_ELT* x     [[buffer(0)]],
    const device float*     stdv  [[buffer(1)]],
    const device float*     meanv [[buffer(2)]],
    device float*           out   [[buffer(3)]],
    constant int&           Chead [[buffer(4)]],
    constant int&           Z     [[buffer(5)]],
    constant int&           H     [[buffer(6)]],   // frames
    constant int&           W     [[buffer(7)]],   // latent mel bins
    uint2 gid [[thread_position_in_grid]])
{
  const int col = (int)gid.x;              // c * W + f
  const int t   = (int)gid.y;
  if (col >= Z * W || t >= H) { return; }
  const int c = col / W;
  const int f = col % W;
  const float v =
      (float)x[((uint)t * (uint)W + (uint)f) * (uint)Chead + (uint)c];
  const float s = stdv[col];
  out[(uint)t * (uint)(Z * W) + (uint)col] =
      (v - meanv[col]) / (s != 0.0f ? s : 1.0f);
}

kernel void ltx_audio_up2x(
    const device VPIPE_ELT* x   [[buffer(0)]],
    device VPIPE_ELT*       out [[buffer(1)]],
    constant int&           C   [[buffer(2)]],
    constant int&           H   [[buffer(3)]],
    constant int&           W   [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int Ho = H * 2, Wo = W * 2;
  const int ci    = (int)gid.x;
  const int ocell = (int)gid.y;
  if (ci >= C || ocell >= Ho * Wo) { return; }
  const int wo = ocell % Wo;
  const int ho = ocell / Wo;
  out[(uint)ocell * (uint)C + (uint)ci] =
      x[((uint)(ho / 2) * (uint)W + (uint)(wo / 2)) * (uint)C + (uint)ci];
}

// channel-last [H][W][C] bf16 -> the channel-first f32 spectrogram
// [C][H][W]. The mirror of ltx_audio_denorm_in, and the only other
// place a transpose happens.
// 0:x 1:out(f32) 2:C 3:H 4:W
kernel void ltx_audio_out(
    const device VPIPE_ELT* x   [[buffer(0)]],
    device float*           out [[buffer(1)]],
    constant int&           C   [[buffer(2)]],
    constant int&           H   [[buffer(3)]],
    constant int&           W   [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int cell = (int)gid.y;
  const int ci   = (int)gid.x;
  if (ci >= C || cell >= H * W) { return; }
  out[(uint)ci * (uint)(H * W) + (uint)cell] =
      (float)x[(uint)cell * (uint)C + (uint)ci];
}

// =====================================================================
// The BigVGAN VOCODER: log-mel -> waveform.
//
// These are built as the f32 twin (ltx25_kernels_f32) and run there.
// The reference forces its whole vocoder pass to fp32 because bf16
// accumulation across its 108 sequential convolutions degrades spectral
// metrics 40-90%; the weights are bf16 either way, so what has to be
// wide is the ACTIVATIONS and the accumulation.
//
// Layout is CHANNEL-LAST [T][C], as everywhere else in this port, so a
// Conv1d is im2col + GEMM and the checkpoint's [C_out][C_in][K] weight
// IS the [C_out][C_in*K] the GEMM wants.
// =====================================================================

// mel (2, T, 64) -> channel-last [T][128].
//
// The reference does `x.transpose(2,3)` then `b s c t -> b (s c) t`, so
// the channel index is s*64 + mel_bin: STEREO SIDE OUTER, mel bin inner.
// The other nesting is the same 128 numbers in a different order and
// vocodes to something that still sounds like audio.
// 0:mel 1:out 2:T 3:mel_bins
kernel void ltx_voc_mel_in(
    const device float* mel  [[buffer(0)]],
    device float*       out  [[buffer(1)]],
    constant int&       T    [[buffer(2)]],
    constant int&       MB   [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int c = (int)gid.x;                // s*MB + bin
  const int t = (int)gid.y;
  if (c >= 2 * MB || t >= T) { return; }
  const int s = c / MB, b = c % MB;
  out[(uint)t * (uint)(2 * MB) + (uint)c] =
      mel[((uint)s * (uint)T + (uint)t) * (uint)MB + (uint)b];
}

// im2col for a 1-D convolution: [T][C] -> [n_out][C*K].
//
// Column order is [ci][k] -- channel OUTER -- matching how the
// checkpoint's [C_out][C_in][K] flattens, so no weight is permuted.
// Padding is SYMMETRIC and ZERO (PyTorch's `padding=` argument), with
// the reference's `get_padding(k, d) = d*(k-1)/2` supplied by the host.
// 0:x 1:out 2:C 3:T 4:K 5:dil 6:pad 7:cell0 8:n_cells
kernel void ltx_voc_im2col(
    const device float* x       [[buffer(0)]],
    device float*       out     [[buffer(1)]],
    constant int&       C       [[buffer(2)]],
    constant int&       T       [[buffer(3)]],
    constant int&       K       [[buffer(4)]],
    constant int&       dil     [[buffer(5)]],
    constant int&       pad     [[buffer(6)]],
    constant int&       cell0   [[buffer(7)]],
    constant int&       n_cells [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int ci = (int)gid.x;
  const int j  = (int)gid.y;
  if (ci >= C || j >= n_cells) { return; }
  const int t0 = cell0 + j;
  device float* dst = out + (uint)j * (uint)(C * K) + (uint)ci * (uint)K;
  for (int k = 0; k < K; ++k) {
    const int st = t0 + k * dil - pad;
    dst[k] = (st < 0 || st >= T)
                 ? 0.0f
                 : x[(uint)st * (uint)C + (uint)ci];
  }
}

// y[M][N] = x[M][K] @ w[N][K]^T (+ bias[N]), f32 in and out.
//
// A plain shared-memory tiled GEMM. libvpipe's dense_gemm is built for
// half and bfloat only, and the whole reason this path exists is that
// narrow accumulation is what the reference warns about -- so the
// vocoder brings its own rather than borrowing one that would undo the
// point.
// 0:x 1:w 2:bias 3:y 4:M 5:N 6:K 7:has_bias
#define VOC_TILE 16
kernel void ltx_voc_gemm(
    const device float* x    [[buffer(0)]],
    const device float* w    [[buffer(1)]],
    const device float* bias [[buffer(2)]],
    device float*       y    [[buffer(3)]],
    constant int&       M    [[buffer(4)]],
    constant int&       N    [[buffer(5)]],
    constant int&       K    [[buffer(6)]],
    constant int&       hasb [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]])
{
  threadgroup float xs[VOC_TILE][VOC_TILE];
  threadgroup float ws[VOC_TILE][VOC_TILE];
  const int n = (int)gid.x, m = (int)gid.y;
  const uint tx = lid.x, ty = lid.y;
  float acc = 0.0f;
  for (int k0 = 0; k0 < K; k0 += VOC_TILE) {
    const int kx = k0 + (int)tx, ky = k0 + (int)ty;
    xs[ty][tx] = (m < M && kx < K)
                     ? x[(uint)m * (uint)K + (uint)kx] : 0.0f;
    ws[ty][tx] = (n < N && ky < K)
                     ? w[(uint)n * (uint)K + (uint)ky] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < VOC_TILE; ++i) { acc += xs[ty][i] * ws[i][tx]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (m >= M || n >= N) { return; }
  if (hasb != 0) { acc += bias[n]; }
  y[(uint)m * (uint)N + (uint)n] = acc;
}

// ConvTranspose1d, the `ups` layers: x[T][Cin] -> y[To][Cout] with
// To = (T-1)*stride - 2*pad + K.
//
// PyTorch stores a ConvTranspose1d weight as [C_IN][C_OUT][K] -- the
// opposite of Conv1d -- so this reads it that way. Computed as a
// GATHER: for output `to`, the contributing taps are the k with
// (to + pad - k) divisible by stride, which is why there is no scatter
// and no atomics.
// 0:x 1:w 2:bias 3:y 4:Cin 5:Cout 6:T 7:K 8:stride 9:pad
kernel void ltx_voc_convt(
    const device float* x      [[buffer(0)]],
    const device float* w      [[buffer(1)]],
    const device float* bias   [[buffer(2)]],
    device float*       y      [[buffer(3)]],
    constant int&       Cin    [[buffer(4)]],
    constant int&       Cout   [[buffer(5)]],
    constant int&       T      [[buffer(6)]],
    constant int&       K      [[buffer(7)]],
    constant int&       stride [[buffer(8)]],
    constant int&       pad    [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int To = (T - 1) * stride - 2 * pad + K;
  const int co = (int)gid.x;
  const int to = (int)gid.y;
  if (co >= Cout || to >= To) { return; }
  float acc = bias[co];
  for (int k = 0; k < K; ++k) {
    const int rem = to + pad - k;
    if (rem < 0 || (rem % stride) != 0) { continue; }
    const int ti = rem / stride;
    if (ti < 0 || ti >= T) { continue; }
    const device float* xr = x + (uint)ti * (uint)Cin;
    for (int ci = 0; ci < Cin; ++ci) {
      acc += xr[ci] * w[((uint)ci * (uint)Cout + (uint)co) * (uint)K
                        + (uint)k];
    }
  }
  y[(uint)to * (uint)Cout + (uint)co] = acc;
}

// The anti-aliased activation's 2x UPSAMPLE: [T][C] -> [2T][C].
//
// UpSample1d replicate-pads by `pad`, runs a grouped conv_transpose1d
// of stride 2 with the checkpoint's 12-tap kaiser-sinc filter, scales
// by the ratio, and crops [pad_left : -pad_right]. Folding all of that
// into one gather is what keeps this a single dispatch. For ratio 2 and
// a 12-tap filter the host passes pad=5, pad_left=15.
//
// The filter comes from the CHECKPOINT (`.upsample.filter`), so no
// kaiser window is recomputed here and the beta/cutoff conventions
// cannot drift.
// 0:x 1:filt 2:out 3:C 4:T 5:K 6:pad 7:pad_left
kernel void ltx_voc_up2x(
    const device float* x        [[buffer(0)]],
    const device float* filt     [[buffer(1)]],
    device float*       out      [[buffer(2)]],
    constant int&       C        [[buffer(3)]],
    constant int&       T        [[buffer(4)]],
    constant int&       K        [[buffer(5)]],
    constant int&       pad      [[buffer(6)]],
    constant int&       pad_left [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int c  = (int)gid.x;
  const int to = (int)gid.y;
  if (c >= C || to >= 2 * T) { return; }
  const int Tp = T + 2 * pad;              // the replicate-padded length
  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    const int rem = to + pad_left - k;
    if (rem < 0 || (rem & 1) != 0) { continue; }
    const int ti = rem >> 1;
    if (ti < 0 || ti >= Tp) { continue; }
    // replicate padding, folded into the read
    int si = ti - pad;
    si = max(0, min(T - 1, si));
    acc += filt[k] * x[(uint)si * (uint)C + (uint)c];
  }
  out[(uint)to * (uint)C + (uint)c] = 2.0f * acc;   // scaled by the ratio
}

// The anti-aliased activation's 2x DOWNSAMPLE: [2T][C] -> [T][C].
//
// LowPassFilter1d: replicate-pad (pad_left, pad_right), then a grouped
// stride-2 conv1d with the checkpoint's filter. For a 12-tap even
// filter the host passes pad_left = 5.
// 0:x 1:filt 2:out 3:C 4:Tin 5:K 6:pad_left
kernel void ltx_voc_down2x(
    const device float* x        [[buffer(0)]],
    const device float* filt     [[buffer(1)]],
    device float*       out      [[buffer(2)]],
    constant int&       C        [[buffer(3)]],
    constant int&       Tin      [[buffer(4)]],
    constant int&       K        [[buffer(5)]],
    constant int&       pad_left [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int c  = (int)gid.x;
  const int to = (int)gid.y;
  const int To = Tin / 2;
  if (c >= C || to >= To) { return; }
  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    int si = to * 2 + k - pad_left;
    si = max(0, min(Tin - 1, si));         // replicate
    acc += filt[k] * x[(uint)si * (uint)C + (uint)c];
  }
  out[(uint)to * (uint)C + (uint)c] = acc;
}

// SnakeBeta, in place: x + (1/(beta+eps)) * sin(x*alpha)^2.
//
// alpha and beta are stored in LOG SCALE (alpha_logscale=True is the
// default and what these checkpoints use), so both are exponentiated
// first. Using them raw is a smooth, plausible nonlinearity that is not
// this one. eps is 1e-9.
// 0:x 1:alpha 2:beta 3:C 4:T
kernel void ltx_voc_snakebeta(
    device float*       x     [[buffer(0)]],
    const device float* alpha [[buffer(1)]],
    const device float* beta  [[buffer(2)]],
    constant int&       C     [[buffer(3)]],
    constant int&       T     [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int c = (int)gid.x;
  const int t = (int)gid.y;
  if (c >= C || t >= T) { return; }
  const float a = exp(alpha[c]);
  const float b = exp(beta[c]);
  const uint i = (uint)t * (uint)C + (uint)c;
  const float v = x[i];
  const float s = sin(v * a);
  x[i] = v + (1.0f / (b + 1e-9f)) * s * s;
}

// out = (a + b + c) / 3.
//
// Each upsample stage runs its THREE resblocks on the SAME input and
// takes the mean -- they are not chained. Chaining them is a working
// vocoder that is not this one.
// 0:out 1:a 2:b 3:c 4:n
kernel void ltx_voc_mean3(
    device float*       out [[buffer(0)]],
    const device float* a   [[buffer(1)]],
    const device float* b   [[buffer(2)]],
    const device float* c   [[buffer(3)]],
    constant int&       n   [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)n) { return; }
  out[gid] = (a[gid] + b[gid] + c[gid]) * (1.0f / 3.0f);
}

// channel-last [T][2] -> channel-first [2][T], clamped to [-1, 1].
//
// use_tanh_at_final is false for these checkpoints, so the final
// activation is a CLAMP, not a tanh.
// 0:x 1:out 2:C 3:T 4:clamp
kernel void ltx_voc_out(
    const device float* x     [[buffer(0)]],
    device float*       out   [[buffer(1)]],
    constant int&       C     [[buffer(2)]],
    constant int&       T     [[buffer(3)]],
    constant int&       docl  [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int c = (int)gid.x;
  const int t = (int)gid.y;
  if (c >= C || t >= T) { return; }
  float v = x[(uint)t * (uint)C + (uint)c];
  if (docl != 0) { v = clamp(v, -1.0f, 1.0f); }
  out[(uint)c * (uint)T + (uint)t] = v;
}

// =====================================================================
// The BWE stage: 16 kHz -> 48 kHz.
//
// A second generator over a CAUSAL log-mel of the main vocoder's output,
// added to a sinc-resampled skip. Only the mel front end and the
// resampler are new; the generator itself is the same BigVGAN above with
// different rates.
// =====================================================================

// im2col for the causal STFT: a mono waveform -> [frames][win].
//
// _STFTFn pads win_length - hop_length samples on the LEFT ONLY (432
// here) with ZEROS, so each frame looks strictly backwards -- that is
// the whole point of the causal STFT and the reason this is not
// centred. Frame f then reads padded[f*hop .. f*hop+win-1].
// 0:y 1:out 2:T 3:win 4:hop 5:left_pad 6:frames
kernel void ltx_voc_stft_im2col(
    const device float* y        [[buffer(0)]],
    device float*       out      [[buffer(1)]],
    constant int&       T        [[buffer(2)]],
    constant int&       win      [[buffer(3)]],
    constant int&       hop      [[buffer(4)]],
    constant int&       left_pad [[buffer(5)]],
    constant int&       frames   [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int k = (int)gid.x;
  const int f = (int)gid.y;
  if (k >= win || f >= frames) { return; }
  const int j = f * hop + k - left_pad;
  out[(uint)f * (uint)win + (uint)k] =
      (j < 0 || j >= T) ? 0.0f : y[j];
}

// [T][2*nf] -> [T][nf]: the DFT bases are stored REAL ROWS FIRST then
// IMAGINARY, so the pair for bin f is (spec[f], spec[f + nf]).
// 0:spec 1:out 2:nf 3:T
kernel void ltx_voc_magnitude(
    const device float* spec [[buffer(0)]],
    device float*       out  [[buffer(1)]],
    constant int&       nf   [[buffer(2)]],
    constant int&       T    [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int f = (int)gid.x;
  const int t = (int)gid.y;
  if (f >= nf || t >= T) { return; }
  const uint base = (uint)t * (uint)(2 * nf);
  const float re = spec[base + (uint)f];
  const float im = spec[base + (uint)(f + nf)];
  out[(uint)t * (uint)nf + (uint)f] = sqrt(re * re + im * im);
}

// x <- log(max(x, 1e-5)), the mel compression.
// 0:x 1:n
kernel void ltx_voc_log_clamp(
    device float* x [[buffer(0)]],
    constant int& n [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)n) { return; }
  x[gid] = log(max(x[gid], 1e-5f));
}

// The general-ratio anti-aliased upsample, for the BWE's x3 skip.
//
// Same shape as ltx_voc_up2x but with the ratio as an argument, and
// used ONCE rather than 108 times -- which is why up2x keeps its
// hardcoded shifts. NOTE the filter here is a HANN-window sinc built at
// construction, NOT the kaiser one the activations use and NOT in the
// checkpoint (`persistent=False`).
// 0:x 1:filt 2:out 3:C 4:T 5:K 6:pad 7:pad_left 8:ratio
kernel void ltx_voc_up_ratio(
    const device float* x        [[buffer(0)]],
    const device float* filt     [[buffer(1)]],
    device float*       out      [[buffer(2)]],
    constant int&       C        [[buffer(3)]],
    constant int&       T        [[buffer(4)]],
    constant int&       K        [[buffer(5)]],
    constant int&       pad      [[buffer(6)]],
    constant int&       pad_left [[buffer(7)]],
    constant int&       ratio    [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int c  = (int)gid.x;
  const int to = (int)gid.y;
  if (c >= C || to >= ratio * T) { return; }
  const int Tp = T + 2 * pad;
  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    const int rem = to + pad_left - k;
    if (rem < 0 || (rem % ratio) != 0) { continue; }
    const int ti = rem / ratio;
    if (ti < 0 || ti >= Tp) { continue; }
    int si = ti - pad;
    si = max(0, min(T - 1, si));           // replicate
    acc += filt[k] * x[(uint)si * (uint)C + (uint)c];
  }
  out[(uint)to * (uint)C + (uint)c] = (float)ratio * acc;
}

// ---------------------------------------------------------------------
// y[M][N] += bias[N], broadcast down the rows.
//
// Only the QUANTIZED path needs this. The dense GEMM
// (dense_gemm_t_bm64_f16) folds the bias into its epilogue, but the
// affine qmm kernels have no bias slot -- they take codes/scales/biases
// where "biases" is the QUANTIZATION zero-point, an entirely different
// tensor from the linear's bias -- so a quantized linear with a bias
// has to add it in a second pass.
// ---------------------------------------------------------------------
kernel void
ltx_bias_add(
    device VPIPE_ELT*       y    [[buffer(0)]],
    const device VPIPE_ELT* bias [[buffer(1)]],
    constant int&           N    [[buffer(2)]],
    constant int&           total[[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
  const int i = (int)gid;
  if (i >= total) { return; }
  y[i] = VPIPE_ELT(float(y[i]) + float(bias[i % N]));
}

// ---------------------------------------------------------------------
// The LATENT UPSCALERS
// ---------------------------------------------------------------------
//
// These need their own im2col even though the VAE has two already, and
// the reason is padding. ltx_vae_im2col REPLICATES in time and ZEROS in
// space, which is the decoder's asymmetry; ltx_vae_im2col_causal pads
// two copies of frame 0 at the front. The upsampler's convolutions are
// plain torch Conv3d(padding=1), which is ZERO on EVERY axis. Reusing
// either VAE gather here is wrong only at the first and last frame --
// invisible in a mean, visible as a temporal seam.
//
// Following the VAE's own precedent, this is a separate entry point
// rather than a flag: the gather is the hot path and a branch in it
// costs more than a second kernel.

// im2col for a 3x3x3 convolution, ZERO-padded on all three axes.
// Column order [ci][kf][kh][kw], matching a [C_out][C_in][3][3][3]
// weight's own flattening, so no weight is permuted at load.
// 0:x 1:out 2:C 3:F 4:H 5:W 6:cell0 7:n_cells
kernel void ltx_ups_im2col(
    const device VPIPE_ELT* x       [[buffer(0)]],
    device VPIPE_ELT*       out     [[buffer(1)]],
    constant int&           C       [[buffer(2)]],
    constant int&           F       [[buffer(3)]],
    constant int&           H       [[buffer(4)]],
    constant int&           W       [[buffer(5)]],
    constant int&           cell0   [[buffer(6)]],
    constant int&           n_cells [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int ci = (int)gid.x;
  const int j  = (int)gid.y;
  if (ci >= C || j >= n_cells) { return; }
  const int cell = cell0 + j;
  const int w0 = cell % W;
  const int h0 = (cell / W) % H;
  const int f0 = cell / (W * H);

  device VPIPE_ELT* dst = out + (uint)j * (uint)(C * 27) + (uint)ci * 27u;
  for (int kf = 0; kf < 3; ++kf) {
    const int sf = f0 + kf - 1;
    for (int kh = 0; kh < 3; ++kh) {
      const int sh = h0 + kh - 1;
      for (int kw = 0; kw < 3; ++kw) {
        const int sw = w0 + kw - 1;
        const int t = (kf * 3 + kh) * 3 + kw;
        if (sf < 0 || sf >= F || sh < 0 || sh >= H || sw < 0 || sw >= W) {
          dst[t] = (VPIPE_ELT)0.0f;        // ZERO in time as well
        } else {
          dst[t] = x[(((uint)sf * (uint)H + (uint)sh) * (uint)W + (uint)sw)
                     * (uint)C + (uint)ci];
        }
      }
    }
  }
}

// im2col for a 3x3 convolution applied PER FRAME -- the spatial
// upsampler, whose weight is 4-D and whose module reshapes to
// (b f) c h w before running a Conv2d. Column order [ci][kh][kw].
// 0:x 1:out 2:C 3:F 4:H 5:W 6:cell0 7:n_cells
kernel void ltx_ups_im2col2d(
    const device VPIPE_ELT* x       [[buffer(0)]],
    device VPIPE_ELT*       out     [[buffer(1)]],
    constant int&           C       [[buffer(2)]],
    constant int&           F       [[buffer(3)]],
    constant int&           H       [[buffer(4)]],
    constant int&           W       [[buffer(5)]],
    constant int&           cell0   [[buffer(6)]],
    constant int&           n_cells [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int ci = (int)gid.x;
  const int j  = (int)gid.y;
  if (ci >= C || j >= n_cells) { return; }
  const int cell = cell0 + j;
  const int w0 = cell % W;
  const int h0 = (cell / W) % H;
  const int f0 = cell / (W * H);
  if (f0 >= F) { return; }

  device VPIPE_ELT* dst = out + (uint)j * (uint)(C * 9) + (uint)ci * 9u;
  for (int kh = 0; kh < 3; ++kh) {
    const int sh = h0 + kh - 1;
    for (int kw = 0; kw < 3; ++kw) {
      const int sw = w0 + kw - 1;
      const int t = kh * 3 + kw;
      if (sh < 0 || sh >= H || sw < 0 || sw >= W) {
        dst[t] = (VPIPE_ELT)0.0f;
      } else {
        dst[t] = x[(((uint)f0 * (uint)H + (uint)sh) * (uint)W + (uint)sw)
                   * (uint)C + (uint)ci];
      }
    }
  }
}

// GroupNorm over channel-last [cells][C], in place, affine per channel.
//
// THE REDUCTION SPANS THE WHOLE GROUP -- channels-in-group x cells --
// which is what makes it a GROUP norm. Reducing per channel instead is
// an InstanceNorm, and it produces a normalised tensor of exactly the
// same shape.
//
// TWO PASSES over the data for the variance rather than E[x^2]-E[x]^2:
// the inputs here are bf16 and the group spans tens of thousands of
// elements, where the one-pass form cancels. One threadgroup per group.
// 0:x 1:gamma 2:beta 3:C 4:cells 5:groups 6:eps
kernel void ltx_ups_group_norm(
    device VPIPE_ELT*     x      [[buffer(0)]],
    const device float*   gamma  [[buffer(1)]],
    const device float*   beta   [[buffer(2)]],
    constant int&         C      [[buffer(3)]],
    constant int&         cells  [[buffer(4)]],
    constant int&         groups [[buffer(5)]],
    constant float&       eps    [[buffer(6)]],
    uint  tgid [[threadgroup_position_in_grid]],
    uint  lid  [[thread_index_in_threadgroup]],
    uint  nth  [[threads_per_threadgroup]])
{
  threadgroup float part[256];
  const int g = (int)tgid;
  if (g >= groups) { return; }
  const int per = C / groups;
  const uint n = (uint)per * (uint)cells;

  float s = 0.0f;
  for (uint i = lid; i < n; i += nth) {
    const uint k = i % (uint)per;
    const uint cell = i / (uint)per;
    s += (float)x[cell * (uint)C + (uint)(g * per) + k];
  }
  part[lid] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint st = nth / 2; st > 0; st >>= 1) {
    if (lid < st) { part[lid] += part[lid + st]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float mean = part[0] / (float)n;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float q = 0.0f;
  for (uint i = lid; i < n; i += nth) {
    const uint k = i % (uint)per;
    const uint cell = i / (uint)per;
    const float d = (float)x[cell * (uint)C + (uint)(g * per) + k] - mean;
    q += d * d;
  }
  part[lid] = q;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint st = nth / 2; st > 0; st >>= 1) {
    if (lid < st) { part[lid] += part[lid + st]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float inv = rsqrt(part[0] / (float)n + eps);

  for (uint i = lid; i < n; i += nth) {
    const uint k = i % (uint)per;
    const uint cell = i / (uint)per;
    const int ci = g * per + (int)k;
    const uint o = cell * (uint)C + (uint)ci;
    const float v = (float)x[o];
    x[o] = (VPIPE_ELT)(((v - mean) * inv) * gamma[ci] + beta[ci]);
  }
}

// SiLU in place. Standalone because the ResBlock needs it in two places
// that are NOT both preceded by a norm: once after GroupNorm, and once
// after the residual has already been added.
// 0:x 1:N
kernel void ltx_ups_silu(
    device VPIPE_ELT* x [[buffer(0)]],
    constant int&     N [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)N) { return; }
  const float v = (float)x[gid];
  x[gid] = (VPIPE_ELT)(v / (1.0f + exp(-v)));
}
