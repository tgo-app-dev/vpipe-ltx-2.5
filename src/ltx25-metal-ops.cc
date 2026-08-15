#include "ltx25-metal-ops.h"
#include "ltx25-config.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using vpipe::metal_compute::ComputeEncoder;
using vpipe::metal_compute::MetalCompute;
using vpipe::metal_compute::SharedBuffer;

namespace ltx25 {

namespace {

// bf16 <-> f32, round-to-nearest-even. Only the host side needs these:
// every kernel reads and writes bf16 directly.
std::uint16_t
to_bf16_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  const std::uint32_t r = ((u >> 16) & 1u) + 0x7fffu;
  return (std::uint16_t)((u + r) >> 16);
}

float
from_bf16_(std::uint16_t h)
{
  const std::uint32_t u = (std::uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// A 1-D dispatch over `n` elements at 256 threads.
void
dispatch_1d_(ComputeEncoder& enc, std::size_t n)
{
  const unsigned g = (unsigned)((n + 255) / 256 * 256);
  enc.dispatch({g, 1, 1}, {256, 1, 1});
}

// C++ mirror of mlx::steel::AttnParams, which lives in the vendored
// steel headers and is not on the plugin's include path. Field order and
// types are the contract; nothing here may be reordered.
struct SteelAttnParams {
  int B, H, D;
  int qL, kL;
  int gqa_factor;
  float scale;
  int NQ, NK, NQ_aligned, NK_aligned, qL_rem, kL_rem, qL_off;
  std::int64_t Q_strides[3], K_strides[3], V_strides[3], O_strides[3];
};

}  // namespace

bool
MetalOps::init(MetalCompute* mc, std::string* err)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  if (mc == nullptr || !mc->valid()) { return fail("no usable Metal device"); }
  _mc = mc;

  // libvpipe's own libraries. The entry points keep an `_f16` suffix in
  // BOTH dtype twins -- the LIBRARY name is what selects bf16 -- which
  // reads wrong and is correct.
  _lib_gemm = mc->load_library("dense_gemm_bf16");
  _lib_elt  = mc->load_library("llm_elementwise_bf16");
  _lib_sdpa = mc->load_library("sdpa_bf16");
  _lib_ltx  = mc->load_library(kMetalLibBf16);
  if (!_lib_gemm.valid()) { return fail("no dense_gemm_bf16 library"); }
  if (!_lib_elt.valid())  { return fail("no llm_elementwise_bf16 library"); }
  if (!_lib_sdpa.valid()) { return fail("no sdpa_bf16 library"); }
  if (!_lib_ltx.valid()) {
    return fail(std::string("no '") + kMetalLibBf16 +
                "' library -- the plugin's own kernels were not registered");
  }

  _fn_gemm       = _lib_gemm.function("dense_gemm_t_bm64_f16");
  _fn_modulate   = _lib_elt.function("adaln_modulate_f16");
  _fn_gated      = _lib_elt.function("gated_residual_f16");
  _fn_gelu       = _lib_elt.function("gelu_tanh_ff_f16");
  _fn_transpose  = _lib_elt.function("transpose_abd_f16");
  _fn_sdpa       = _lib_sdpa.function("sdpa_full_f16");
  _fn_rms        = mc->load_library("rms_norm_bf16").function("rms_norm_f16");
  _fn_rope       = _lib_ltx.function("ltx_rope_half_perhead");
  _fn_gate_heads = _lib_ltx.function("ltx_gate_heads");
  _fn_rms_gain   = _lib_ltx.function("ltx_rms_norm_gain");
  _fn_ada_zero   = _lib_ltx.function("ltx_ada_zero");
  _fn_rms_out    = _lib_ltx.function("ltx_rms_norm_out");
  _fn_add        = _lib_ltx.function("ltx_add");
  _fn_ada_zero_g = _lib_ltx.function("ltx_ada_zero_g");
  _fn_modulate_g = _lib_ltx.function("ltx_modulate_g");
  _fn_gated_g    = _lib_ltx.function("ltx_gated_residual_g");
  _fn_add_row_prefix = _lib_ltx.function("ltx_add_row_prefix");
  _fn_copy       = _lib_ltx.function("ltx_copy");
  _fn_layer_norm = _lib_elt.function("layer_norm_plain_f16");
  _fn_fill_regs  = _lib_ltx.function("ltx_fill_registers");
  _fn_vae_denorm  = _lib_ltx.function("ltx_vae_denorm_in");
  _fn_vae_im2col  = _lib_ltx.function("ltx_vae_im2col");
  _fn_vae_pns     = _lib_ltx.function("ltx_vae_pixel_norm_silu");
  _fn_vae_add     = _lib_ltx.function("ltx_vae_add_into");
  _fn_vae_d2s     = _lib_ltx.function("ltx_vae_d2s");
  _fn_vae_unpatch = _lib_ltx.function("ltx_vae_unpatchify");
  _fn_vae_patch_in  = _lib_ltx.function("ltx_vae_patchify_in");
  _fn_vae_im2col_c  = _lib_ltx.function("ltx_vae_im2col_causal");
  _fn_vae_s2d       = _lib_ltx.function("ltx_vae_s2d");
  _fn_vae_dup       = _lib_ltx.function("ltx_vae_dup_frame0");
  _fn_vae_whiten    = _lib_ltx.function("ltx_vae_whiten_out");
  _fn_aud_denorm  = _lib_ltx.function("ltx_audio_denorm_in");
  _fn_aud_im2col  = _lib_ltx.function("ltx_audio_im2col");
  _fn_aud_up2x    = _lib_ltx.function("ltx_audio_up2x");
  _fn_aud_out     = _lib_ltx.function("ltx_audio_out");
  _fn_aud_mel_in    = _lib_ltx.function("ltx_audio_mel_in");
  _fn_aud_im2col_dn = _lib_ltx.function("ltx_audio_im2col_down");
  _fn_aud_whiten    = _lib_ltx.function("ltx_audio_whiten_rows");
  _fn_bias_add    = _lib_ltx.function("ltx_bias_add");

  // The affine qmm kernels, for a QUANTIZED checkpoint. libvpipe's own,
  // embedded in the host binary and reached by name exactly as
  // dense_gemm_bf16 is (docs/PLUGINS.md lists them).
  //
  // OPTIONAL, and deliberately not in the need[] table below: a host
  // that does not ship them still runs a dense checkpoint perfectly
  // well, and failing init() here would turn a missing OPTIONAL kernel
  // into "the plugin does not load". The loader checks
  // quant_available() when it actually meets a quantized weight, so the
  // error names the cause at the point it matters.
  _lib_qmm = mc->load_library("affine_qmm_steel_bf16");
  if (_lib_qmm.valid()) {
    _quant_ok = true;
    const int bits[2] = {4, 8};
    const int grp[2]  = {32, 64};
    for (int b = 0; b < 2; ++b) {
      for (int g = 0; g < 2; ++g) {
        const std::string base = "affine_qmm_steel_w" +
                                 std::to_string(bits[b]) + "g" +
                                 std::to_string(grp[g]);
        _fn_qmm[b][g][0] = _lib_qmm.function(base);
        // The WIDE tile only exists at group 64 -- there is no
        // affine_qmm_steel_w4g32_bm64 -- so it is optional and falls
        // back to the narrow one. Requiring it turned quantization off
        // wholesale the first time this was written.
        _fn_qmm[b][g][1] = _lib_qmm.function(base + "_bm64");
        if (!_fn_qmm[b][g][0].valid()) { _quant_ok = false; }
      }
    }
  }

  // Steel flash attention. libvpipe's, embedded and reached by name,
  // and OPTIONAL for the same reason the qmm kernels are -- except that
  // here the fallback is not merely slower in the abstract: sdpa_full
  // runs one simdgroup per (head, query) across the whole key sequence,
  // which at video token counts is the entire denoise step. It is kept
  // as the A/B, not as a path to run.
  //
  // VPIPE_LTX25_NO_STEEL_ATTN forces the fallback.
  _lib_attn = mc->load_library("attn_steel");
  _steel_ok = _lib_attn.valid() &&
              std::getenv("VPIPE_LTX25_NO_STEEL_ATTN") == nullptr;

  // On M5 the same flash attention runs on the matrix units
  // (attn_steel_nax, bq=64/bk=32) instead of the simdgroup ALU kernel
  // (bq=32/bk=16). Identical AttnParams, function constants and
  // threadgroup contract -- only the tile sizes and the entry point
  // differ -- which is why this is a function swap rather than a second
  // code path.
  //
  // Gated on the DEVICE, never on the load: on a pre-M5 GPU the
  // library's entry points are build-time stubs that write one zero, so
  // a valid() library there would bind a kernel that produces nothing.
  // VPIPE_LTX25_NO_ATTN_NAX forces the ALU kernel (A/B).
  if (_steel_ok && mc->supports_matrix_cores() &&
      std::getenv("VPIPE_LTX25_NO_ATTN_NAX") == nullptr) {
    _lib_attn_nax = mc->load_library("attn_steel_nax");
    _attn_nax = _lib_attn_nax.valid();
  }

  // The M5 matrix-core dense GEMM, on the same device gate. It serves
  // BOTH linears: a dense weight goes straight to it, and a quantized
  // one is expanded into `_w_deq` by affine_dequant first and then runs
  // the same kernel. The DiT ships quantized -- bf16 is 39 GB -- so
  // without that second path the model's dominant term would stay on the
  // ALU qmm and the matrix cores would serve only the VAEs.
  //
  // VPIPE_LTX25_NO_MMA forces the steel dense kernel; VPIPE_LTX25_MMA_MIN_M
  // moves the M below which the 128-row tile is not worth entering;
  // VPIPE_LTX25_NO_DEQUANT_MMA keeps quantized weights on the qmm (the
  // A/B for the paragraph above).
  if (mc->supports_matrix_cores() &&
      std::getenv("VPIPE_LTX25_NO_MMA") == nullptr) {
    _lib_dense_mma = mc->load_library("dense_gemm_mma_bf16");
    if (_lib_dense_mma.valid()) {
      _fn_dense_mma = _lib_dense_mma.function("dense_gemm_mma_t_n128_f16");
      // Deeper-K tiles. OPTIONAL: a host that ships only the 128-region
      // entry point still runs, on that one for every K.
      _fn_dense_mma_deep =
          _lib_dense_mma.function("dense_gemm_mma_t_n128x256_f16");
      _fn_dense_mma_tn2 =
          _lib_dense_mma.function("dense_gemm_mma_t_n128x256_tn2_f16");
    }
    if (const char* e = std::getenv("VPIPE_LTX25_MMA_MIN_M")) {
      const int v = std::atoi(e);
      if (v > 0) { _mma_min_m = v; }
    }
    if (const char* e = std::getenv("VPIPE_LTX25_MMA_TILE")) {
      const int v = std::atoi(e);
      if (v == 128 || v == 256 || v == 512) { _tile_force = v; }
    }
    // The dequant-once expansion feeding it. Also OPTIONAL -- without it
    // quantized weights simply stay on the qmm, which is what every
    // pre-M5 box does anyway.
    if (_fn_dense_mma.valid() &&
        std::getenv("VPIPE_LTX25_NO_DEQUANT_MMA") == nullptr) {
      _lib_dequant = mc->load_library("affine_dequant_bf16");
      if (_lib_dequant.valid()) {
        const int bits[2] = {4, 8};
        const int grp[2]  = {32, 64};
        for (int b = 0; b < 2; ++b) {
          for (int g = 0; g < 2; ++g) {
            _fn_dequant[b][g] = _lib_dequant.function(
                "affine_dequant_w" + std::to_string(bits[b]) + "g" +
                std::to_string(grp[g]));
          }
        }
      }
    }
  }

  // Named individually: an unvalidated ComputeFunction dispatches as a
  // SILENT NO-OP, so a missing entry point would show up as a model that
  // runs at full cost and produces noise.
  struct { const vpipe::metal_compute::ComputeFunction* f; const char* n; }
  need[] = {
      {&_fn_gemm, "dense_gemm_t_bm64_f16"},
      {&_fn_modulate, "adaln_modulate_f16"},
      {&_fn_gated, "gated_residual_f16"},
      {&_fn_gelu, "gelu_tanh_ff_f16"},
      {&_fn_transpose, "transpose_abd_f16"},
      {&_fn_sdpa, "sdpa_full_f16"},
      {&_fn_rms, "rms_norm_f16"},
      {&_fn_rope, "ltx_rope_half_perhead"},
      {&_fn_gate_heads, "ltx_gate_heads"},
      {&_fn_rms_gain, "ltx_rms_norm_gain"},
      {&_fn_ada_zero, "ltx_ada_zero"},
      {&_fn_rms_out, "ltx_rms_norm_out"},
      {&_fn_add, "ltx_add"},
      {&_fn_copy, "ltx_copy"},
      {&_fn_ada_zero_g, "ltx_ada_zero_g"},
      {&_fn_modulate_g, "ltx_modulate_g"},
      {&_fn_gated_g, "ltx_gated_residual_g"},
      {&_fn_add_row_prefix, "ltx_add_row_prefix"},
      {&_fn_layer_norm, "layer_norm_plain_f16"},
      {&_fn_fill_regs, "ltx_fill_registers"},
      {&_fn_vae_denorm, "ltx_vae_denorm_in"},
      {&_fn_vae_im2col, "ltx_vae_im2col"},
      {&_fn_vae_pns, "ltx_vae_pixel_norm_silu"},
      {&_fn_vae_add, "ltx_vae_add_into"},
      {&_fn_vae_d2s, "ltx_vae_d2s"},
      {&_fn_vae_unpatch, "ltx_vae_unpatchify"},
      {&_fn_vae_patch_in, "ltx_vae_patchify_in"},
      {&_fn_vae_im2col_c, "ltx_vae_im2col_causal"},
      {&_fn_vae_s2d, "ltx_vae_s2d"},
      {&_fn_vae_dup, "ltx_vae_dup_frame0"},
      {&_fn_vae_whiten, "ltx_vae_whiten_out"},
      {&_fn_aud_denorm, "ltx_audio_denorm_in"},
      {&_fn_aud_im2col, "ltx_audio_im2col"},
      {&_fn_aud_up2x, "ltx_audio_up2x"},
      {&_fn_aud_out, "ltx_audio_out"},
      {&_fn_aud_mel_in, "ltx_audio_mel_in"},
      {&_fn_aud_im2col_dn, "ltx_audio_im2col_down"},
      {&_fn_aud_whiten, "ltx_audio_whiten_rows"},
      {&_fn_bias_add, "ltx_bias_add"},
  };
  for (const auto& e : need) {
    if (!e.f->valid()) {
      return fail(std::string("kernel '") + e.n + "' did not resolve");
    }
  }
  return true;
}

SharedBuffer
MetalOps::alloc(std::size_t n) const
{
  return _mc->make_shared_buffer(n * 2);
}

SharedBuffer
MetalOps::alloc_f32(std::size_t n) const
{
  return _mc->make_shared_buffer(n * 4);
}

SharedBuffer
MetalOps::upload_bf16(const std::vector<float>& v) const
{
  SharedBuffer b = alloc(v.size());
  auto* p = static_cast<std::uint16_t*>(b.contents());
  for (std::size_t i = 0; i < v.size(); ++i) { p[i] = to_bf16_(v[i]); }
  return b;
}

SharedBuffer
MetalOps::upload_f32(const std::vector<float>& v) const
{
  SharedBuffer b = alloc_f32(v.size());
  std::memcpy(b.contents(), v.data(), v.size() * 4);
  return b;
}

std::vector<float>
MetalOps::download_bf16(const SharedBuffer& b, std::size_t n)
{
  const auto* p = static_cast<const std::uint16_t*>(b.contents());
  std::vector<float> v(n);
  for (std::size_t i = 0; i < n; ++i) { v[i] = from_bf16_(p[i]); }
  return v;
}

bool
MetalOps::mma_eligible_(int M, int K, int N) const noexcept
{
  // Tall enough to fill the 128-row tile, and N >= 16 keeps a degenerate
  // projection off it.
  if (!_fn_dense_mma.valid() || M < _mma_min_m || N < 16) { return false; }

  // matmul2d addresses a tensor operand with a SIGNED 32-BIT BYTE
  // offset, so an operand of 2 GB or more computes garbage from the
  // 128-row tile that straddles the boundary onward -- silently, with no
  // error and no fault.
  //
  // MEASURED on M5 against the steel GEMM, which is int64-safe: at
  // K=3456 the first wrong row is 310784 and at K=864 it is 1242880 --
  // in both cases exactly 2^30/K rounded up to a 128-row tile, and in
  // both cases the rows below it are correct to 5e-5. Under the
  // boundary the two kernels agree at ~3e-5 at every M tested.
  //
  // Nothing in the model reaches this today: the DiT is bounded by its
  // token count and the VAEs chunk their im2col to 64 M ELEMENTS, which
  // is 128 MB and so 16x under the limit. It is here because that VAE
  // bound is a public setter whose own comment invites raising it
  // ("bigger is better until it does not fit"), and the failure it would
  // buy is a corrupt lower half of the picture rather than an error.
  const std::size_t kOperandMax = 2ull * 1024 * 1024 * 1024;
  const std::size_t elt = 2;   // bf16
  if ((std::size_t)M * K * elt >= kOperandMax ||
      (std::size_t)M * N * elt >= kOperandMax ||
      (std::size_t)N * K * elt >= kOperandMax) {
    return false;
  }
  return true;
}

void
MetalOps::dense_mma_(ComputeEncoder& enc, const SharedBuffer& x,
                     const SharedBuffer& w, const SharedBuffer* bias,
                     const SharedBuffer& y, int M, int K, int N) const
{
  // Which N-region a threadgroup owns: 128, 256, or the TN=2 tile's 512.
  //
  // The sibling DiTs pick this from K alone, and that rule is WRONG here
  // -- not by a little. It was fitted to a DiT whose N is never below
  // 4096, and this model runs the same kernel over a VAE whose N is the
  // channel count, 1024 down to 48. MEASURED on M5, K alone mispicks the
  // two largest VAE levels (1.2x) AND the two largest DiT GEMMs (1.1x),
  // in opposite directions.
  //
  // It cannot be rescued by moving the K threshold, because the winner
  // is not a function of K: (M=8160, K=4096, N=4096) wants the 512 tile
  // and (M=1024, K=4096, N=4096) wants the 128 one. The rule below is
  // fitted to MEASURED shapes, and each branch has a mechanism:
  //
  //   N <= 128        a wider region is mostly padding -- at N=48 the
  //                   512 tile computes 10x the columns that exist.
  //   deep K, wide N  the 256 tile's K-blocking wins outright; the 512
  //                   tile is 2x SLOWER here (the DiT's ff_out).
  //   deep K          each threadgroup already does enough K work to
  //                   saturate, so the widest region wins even when
  //                   there are few row-tiles. Fitting this on M alone
  //                   cost 1.70x at (M=800, K=27648, N=1024), which is
  //                   a real VAE level -- a small chunk of a deep one.
  //   M >= 2048       shallower K, but enough row-tiles to fill the GPU
  //                   with the widest region, which amortizes the
  //                   weight loads.
  //   otherwise       too few row-tiles; take the narrow region for the
  //                   threadgroup count instead.
  //
  // This is a fit over one model's shapes, not a law -- the same kind
  // of fit as the K rule it replaces, but at least fitted to THIS
  // model, and CHECKED on shapes held out of the fit. Measured by
  // per-round voting with every arm run once per round: measuring the
  // arms SEQUENTIALLY is what makes a rule like this wrong, and the
  // in-tree shared/mma-tile.h records that error turning a 7%
  // regression into an apparent 1.2x win. Every shape here resolves by
  // 8% or more, so the rule is not straddling a tie.
  // VPIPE_LTX25_MMA_TILE=128|256|512 forces one, to re-measure.
  int RN = _tile_force > 0 ? _tile_force : 128;
  if (_tile_force <= 0) {
    if (N <= 128) {
      RN = 128;
    } else if (K >= 12288) {
      RN = N >= 4096 ? 256 : 512;
    } else if (M >= 2048) {
      RN = 512;
    }
  }
  // Each wider tile falls back to the 128 one when its entry point is
  // absent, so a host shipping only that still runs every shape.
  const vpipe::metal_compute::ComputeFunction* fn = &_fn_dense_mma;
  if (RN == 256 && _fn_dense_mma_deep.valid()) {
    fn = &_fn_dense_mma_deep;
  } else if (RN == 512 && _fn_dense_mma_tn2.valid()) {
    fn = &_fn_dense_mma_tn2;
  } else {
    RN = 128;
  }
  enc.set_function(*fn);
  enc.set_buffer(0, x);
  enc.set_buffer(1, w);
  enc.set_buffer(2, w);          // bias slot, unread
  enc.set_buffer(3, y);
  enc.set_constant(4, K);
  enc.set_constant(5, N);
  enc.set_constant(6, M);
  enc.set_constant(7, 0);
  enc.dispatch({(unsigned)(((N + RN - 1) / RN) * 256),
                (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
  // This kernel voids its bias argument -- the epilogue is not
  // written -- so the bias is a second pass, exactly as on the
  // quantized path below.
  if (bias != nullptr) { bias_add(enc, y, *bias, M, N); }
}

bool
MetalOps::dequant_mma_(ComputeEncoder& enc, const SharedBuffer& x,
                       const QWeight& w, const SharedBuffer* bias,
                       const SharedBuffer& y, int M, int K, int N) const
{
  if (!mma_eligible_(M, K, N)) { return false; }
  const int bi = w.bits == 8 ? 1 : 0;
  const int gi = w.group == 64 ? 1 : 0;
  const vpipe::metal_compute::ComputeFunction& dq = _fn_dequant[bi][gi];
  if (!dq.valid()) { return false; }

  // One [N][K] bf16 expansion, reused by every projection. Grows to the
  // largest the model asks for (the DiT's ff, 16384x4096 = 134 MB) and
  // then stops. A failed alloc is not fatal: the caller falls back to
  // the qmm, which needs no scratch.
  const std::size_t need = (std::size_t)N * K * 2;
  if (_w_deq.empty() || _w_deq.byte_size() < need) {
    _w_deq = _mc->make_shared_buffer(need);
    if (_w_deq.empty()) { return false; }
  }

  // codes/scales/qbias -> _w_deq[N][K]. One thread per packed u32 word:
  // w4 packs 8 nibbles per word, w8 four bytes.
  enc.set_function(dq);
  enc.set_buffer(0, w.codes);
  enc.set_buffer(1, w.scales);
  enc.set_buffer(2, w.qbias);
  enc.set_buffer(3, _w_deq);
  enc.set_constant(4, K);
  enc.set_constant(5, N);
  const unsigned words = (unsigned)(w.bits == 8 ? (K / 4) : (K / 8));
  enc.dispatch({words, (unsigned)N, 1}, {64, 1, 1});

  dense_mma_(enc, x, _w_deq, bias, y, M, K, N);
  return true;
}

void
MetalOps::linear(ComputeEncoder& enc, const SharedBuffer& x,
                 const SharedBuffer& w, const SharedBuffer* bias,
                 const SharedBuffer& y, int M, int K, int N) const
{
  if (mma_eligible_(M, K, N)) {
    dense_mma_(enc, x, w, bias, y, M, K, N);
    return;
  }

  enc.set_function(_fn_gemm);
  enc.set_buffer(0, x);
  enc.set_buffer(1, w);
  // The bias slot must be bound even when unused: an unbound buffer
  // argument is undefined behaviour, not a null pointer the kernel can
  // test. Bind the weight and let has_bias gate the read.
  enc.set_buffer(2, bias != nullptr ? *bias : w);
  enc.set_buffer(3, y);
  enc.set_constant(4, K);
  enc.set_constant(5, N);
  enc.set_constant(6, M);
  enc.set_constant(7, bias != nullptr ? 1 : 0);
  // BM=64, BN=32, threadgroup {32,2,2}.
  enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                (unsigned)(((M + 63) / 64) * 2), 2}, {32, 2, 2});
}

void
MetalOps::linear(ComputeEncoder& enc, const SharedBuffer& x, const QWeight& w,
                 const SharedBuffer* bias, const SharedBuffer& y,
                 int M, int K, int N) const
{
  if (!w.quantized) {
    linear(enc, x, w.w, bias, y, M, K, N);
    return;
  }

  // M5: expand the weight once and run the dense matrix-core GEMM
  // instead of the fused dequant-in-loop qmm below. 2.7-3.5x at the
  // DiT's shapes; see dequant_mma_.
  if (dequant_mma_(enc, x, w, bias, y, M, K, N)) { return; }

  // BM=64 above the tile threshold, BM=32 below. The DiT's token counts
  // are small (672 at 768x448) next to an LM's prefill, so most calls
  // take the narrow tile; the wide one earns its keep on the caption
  // projections and long-context video.
  const int bi = w.bits == 8 ? 1 : 0;
  const int gi = w.group == 64 ? 1 : 0;
  const bool bm64 = M >= 64 && _fn_qmm[bi][gi][1].valid();
  enc.set_function(_fn_qmm[bi][gi][bm64 ? 1 : 0]);
  enc.set_buffer(0, w.codes);
  enc.set_buffer(1, w.scales);
  enc.set_buffer(2, w.qbias);
  enc.set_buffer(3, x);
  enc.set_buffer(4, y);
  enc.set_constant(5, K);
  enc.set_constant(6, N);
  enc.set_constant(7, M);
  const int bm = bm64 ? 64 : 32;
  enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                (unsigned)(((M + bm - 1) / bm) * 2), 2}, {32, 2, 2});

  // The qmm kernel has no bias slot -- its buffer(2) is the quantization
  // zero-point, not the linear's bias -- so the bias is a second pass.
  if (bias != nullptr) { bias_add(enc, y, *bias, M, N); }
}

void
MetalOps::bias_add(ComputeEncoder& enc, const SharedBuffer& y,
                   const SharedBuffer& bias, int M, int N) const
{
  const std::size_t total = (std::size_t)M * N;
  enc.set_function(_fn_bias_add);
  enc.set_buffer(0, y);
  enc.set_buffer(1, bias);
  enc.set_constant(2, N);
  enc.set_constant(3, (int)total);
  dispatch_1d_(enc, total);
}

void
MetalOps::modulate(ComputeEncoder& enc, const SharedBuffer& x,
                   const SharedBuffer& scale, const SharedBuffer& shift,
                   const SharedBuffer& out, int N, int rows) const
{
  const std::size_t total = (std::size_t)N * rows;
  enc.set_function(_fn_modulate);
  enc.set_buffer(0, x);
  enc.set_buffer(1, scale);
  enc.set_buffer(2, shift);
  enc.set_buffer(3, out);
  enc.set_constant(4, N);
  enc.set_constant(5, (int)total);
  dispatch_1d_(enc, total);
}

void
MetalOps::rms_norm_plain(ComputeEncoder& enc, const SharedBuffer& x, int N,
                         int rows) const
{
  // rms_norm_f16 is in-place over rows of width N: 0:x 1:N 2:eps.
  enc.set_function(_fn_rms);
  enc.set_buffer(0, x);
  enc.set_constant(1, N);
  enc.set_constant(2, 1.0e-6f);
  enc.dispatch({(unsigned)(rows * 256), 1, 1}, {256, 1, 1});
}

void
MetalOps::rms_norm_gain(ComputeEncoder& enc, const SharedBuffer& x,
                        const SharedBuffer& gain, int N, int rows) const
{
  enc.set_function(_fn_rms_gain);
  enc.set_buffer(0, x);
  enc.set_buffer(1, gain);
  enc.set_constant(2, N);
  enc.set_constant(3, 1.0e-6f);
  enc.dispatch({(unsigned)(rows * 256), 1, 1}, {256, 1, 1});
}

void
MetalOps::modulate_off(ComputeEncoder& enc, const SharedBuffer& x,
                       const SharedBuffer& scale, std::size_t scale_off,
                       const SharedBuffer& shift, std::size_t shift_off,
                       const SharedBuffer& out, int N, int rows) const
{
  const std::size_t total = (std::size_t)N * rows;
  enc.set_function(_fn_modulate);
  enc.set_buffer(0, x);
  enc.set_buffer(1, scale, scale_off);
  enc.set_buffer(2, shift, shift_off);
  enc.set_buffer(3, out);
  enc.set_constant(4, N);
  enc.set_constant(5, (int)total);
  dispatch_1d_(enc, total);
}

void
MetalOps::ada_zero(ComputeEncoder& enc, const SharedBuffer& x,
                   const SharedBuffer& scale, const SharedBuffer& shift,
                   const SharedBuffer& out, int N, int rows) const
{
  enc.set_function(_fn_ada_zero);
  enc.set_buffer(0, x);
  enc.set_buffer(1, scale);
  enc.set_buffer(2, shift);
  enc.set_buffer(3, out);
  enc.set_constant(4, N);
  enc.set_constant(5, 1.0e-6f);
  enc.dispatch({(unsigned)(rows * 256), 1, 1}, {256, 1, 1});
}

void
MetalOps::ada_zero_g(ComputeEncoder& enc, const SharedBuffer& x,
                     const SharedBuffer& scale, const SharedBuffer& shift,
                     const SharedBuffer& level, const SharedBuffer& out,
                     int N, int rows) const
{
  enc.set_function(_fn_ada_zero_g);
  enc.set_buffer(0, x);
  enc.set_buffer(1, scale);
  enc.set_buffer(2, shift);
  enc.set_buffer(3, level);
  enc.set_buffer(4, out);
  enc.set_constant(5, N);
  enc.set_constant(6, 1.0e-6f);
  enc.dispatch({(unsigned)(rows * 256), 1, 1}, {256, 1, 1});
}

void
MetalOps::modulate_g(ComputeEncoder& enc, const SharedBuffer& x,
                     const SharedBuffer& scale, std::size_t scale_off,
                     const SharedBuffer& shift, std::size_t shift_off,
                     const SharedBuffer& level, const SharedBuffer& out,
                     int N, int rows, int level_stride) const
{
  const std::size_t total = (std::size_t)N * rows;
  enc.set_function(_fn_modulate_g);
  enc.set_buffer(0, x);
  enc.set_buffer(1, scale, scale_off);
  enc.set_buffer(2, shift, shift_off);
  enc.set_buffer(3, level);
  enc.set_buffer(4, out);
  enc.set_constant(5, N);
  enc.set_constant(6, level_stride);
  enc.set_constant(7, (int)total);
  dispatch_1d_(enc, total);
}

void
MetalOps::gated_residual_g(ComputeEncoder& enc, const SharedBuffer& h,
                           const SharedBuffer& gate, const SharedBuffer& sub,
                           const SharedBuffer& level, int N, int rows) const
{
  const std::size_t total = (std::size_t)N * rows;
  enc.set_function(_fn_gated_g);
  enc.set_buffer(0, h);
  enc.set_buffer(1, gate);
  enc.set_buffer(2, sub);
  enc.set_buffer(3, level);
  enc.set_constant(4, N);
  enc.set_constant(5, (int)total);
  dispatch_1d_(enc, total);
}

void
MetalOps::add_row_prefix(ComputeEncoder& enc, const SharedBuffer& x,
                         const SharedBuffer& row, int N, int rows) const
{
  if (rows <= 0) { return; }
  const std::size_t total = (std::size_t)N * rows;
  enc.set_function(_fn_add_row_prefix);
  enc.set_buffer(0, x);
  enc.set_buffer(1, row);
  enc.set_constant(2, N);
  enc.set_constant(3, (int)total);
  dispatch_1d_(enc, total);
}

void
MetalOps::layer_norm_plain(ComputeEncoder& enc, const SharedBuffer& x,
                           const SharedBuffer& out, int N, int rows) const
{
  enc.set_function(_fn_layer_norm);
  enc.set_buffer(0, x);
  enc.set_buffer(1, out);
  enc.set_constant(2, N);
  enc.set_constant(3, 1.0e-6f);
  // One threadgroup per row: the kernel indexes rows off tid.y.
  enc.dispatch({256, (unsigned)rows, 1}, {256, 1, 1});
}

void
MetalOps::rms_norm_out(ComputeEncoder& enc, const SharedBuffer& x,
                       const SharedBuffer& out, int N, int rows) const
{
  enc.set_function(_fn_rms_out);
  enc.set_buffer(0, x);
  enc.set_buffer(1, out);
  enc.set_constant(2, N);
  enc.set_constant(3, 1.0e-6f);
  enc.dispatch({(unsigned)(rows * 256), 1, 1}, {256, 1, 1});
}

void
MetalOps::add(ComputeEncoder& enc, const SharedBuffer& a, std::size_t a_off,
              const SharedBuffer& b, std::size_t b_off,
              const SharedBuffer& out, int n, std::size_t out_off) const
{
  enc.set_function(_fn_add);
  enc.set_buffer(0, a, a_off);
  enc.set_buffer(1, b, b_off);
  enc.set_buffer(2, out, out_off);
  enc.set_constant(3, n);
  dispatch_1d_(enc, (std::size_t)n);
}

void
MetalOps::fill_registers(ComputeEncoder& enc, const SharedBuffer& x,
                         const SharedBuffer& regs, int dim, int n_registers,
                         int n_valid, int tokens) const
{
  if (n_valid >= tokens) { return; }   // nothing padded
  enc.set_function(_fn_fill_regs);
  enc.set_buffer(0, x);
  enc.set_buffer(1, regs);
  enc.set_constant(2, dim);
  enc.set_constant(3, n_registers);
  enc.set_constant(4, n_valid);
  enc.set_constant(5, tokens);
  enc.dispatch({(unsigned)dim, (unsigned)tokens, 1}, {64, 1, 1});
}

void
MetalOps::copy(ComputeEncoder& enc, const SharedBuffer& src,
               const SharedBuffer& dst, int n) const
{
  enc.set_function(_fn_copy);
  enc.set_buffer(0, src);
  enc.set_buffer(1, dst);
  enc.set_constant(2, n);
  dispatch_1d_(enc, (std::size_t)n);
}

void
MetalOps::gated_residual(ComputeEncoder& enc, const SharedBuffer& h,
                         const SharedBuffer& gate, const SharedBuffer& sub,
                         int N, int rows) const
{
  const std::size_t total = (std::size_t)N * rows;
  enc.set_function(_fn_gated);
  enc.set_buffer(0, h);
  enc.set_buffer(1, gate);
  enc.set_buffer(2, sub);
  enc.set_constant(3, N);
  enc.set_constant(4, (int)total);
  dispatch_1d_(enc, total);
}

void
MetalOps::gelu(ComputeEncoder& enc, const SharedBuffer& x,
               const SharedBuffer& out, int n) const
{
  enc.set_function(_fn_gelu);
  enc.set_buffer(0, x);
  enc.set_buffer(1, out);
  enc.set_constant(2, n);
  dispatch_1d_(enc, (std::size_t)n);
}

void
MetalOps::rope(ComputeEncoder& enc, const SharedBuffer& x,
               const SharedBuffer& cos, const SharedBuffer& sin, int heads,
               int tokens, int head_dim) const
{
  enc.set_function(_fn_rope);
  enc.set_buffer(0, x);
  enc.set_buffer(1, cos);
  enc.set_buffer(2, sin);
  enc.set_constant(3, heads);
  enc.set_constant(4, tokens);
  enc.set_constant(5, head_dim);
  enc.dispatch({(unsigned)(head_dim / 2), (unsigned)tokens, (unsigned)heads},
               {64, 1, 1});
}

void
MetalOps::gate_heads(ComputeEncoder& enc, const SharedBuffer& y,
                     const SharedBuffer& logits, int heads, int tokens,
                     int head_dim) const
{
  enc.set_function(_fn_gate_heads);
  enc.set_buffer(0, y);
  enc.set_buffer(1, logits);
  enc.set_constant(2, heads);
  enc.set_constant(3, tokens);
  enc.set_constant(4, head_dim);
  enc.dispatch({(unsigned)head_dim, (unsigned)tokens, (unsigned)heads},
               {64, 1, 1});
}

void
MetalOps::transpose_abd(ComputeEncoder& enc, const SharedBuffer& in,
                        const SharedBuffer& out, int A, int B, int D) const
{
  enc.set_function(_fn_transpose);
  enc.set_buffer(0, in);
  enc.set_buffer(1, out);
  enc.set_constant(2, A);
  enc.set_constant(3, B);
  enc.set_constant(4, D);
  enc.dispatch({(unsigned)D, (unsigned)B, (unsigned)A}, {32, 1, 1});
}

void
MetalOps::sdpa_full(ComputeEncoder& enc, const SharedBuffer& q,
                    const SharedBuffer& k, const SharedBuffer& v,
                    const SharedBuffer& out, int heads, int tq, int tkv,
                    int head_dim) const
{
  enc.set_function(_fn_sdpa);
  enc.set_buffer(0, q);
  enc.set_buffer(1, k);
  enc.set_buffer(2, v);
  enc.set_buffer(3, out);
  enc.set_constant(4, (float)(1.0 / std::sqrt((double)head_dim)));
  enc.set_constant(5, tkv);
  enc.set_constant(6, head_dim);
  enc.set_constant(7, heads);      // Hq
  enc.set_constant(8, heads);      // Hkv -- no GQA anywhere in this DiT
  enc.set_constant(9, tq);
  enc.set_constant(10, tkv);       // kv_stride
  enc.dispatch({32, (unsigned)heads, (unsigned)tq}, {32, 1, 1});
}

const char*
MetalOps::attn_kernel() const noexcept
{
  if (!_steel_ok) { return "scalar"; }
  return _attn_nax ? "steel-nax" : "steel";
}

bool
MetalOps::steel_attn_available(int head_dim) const noexcept
{
  // 128 is the video stream and the text cross-attention; 64 is the
  // audio stream and both audio<->video directions. Any other width
  // would need its own instantiation in attn_steel.metal, so it falls
  // back rather than binding a kernel compiled for the wrong D.
  return _steel_ok && (head_dim == 64 || head_dim == 128);
}

bool
MetalOps::steel_attn_plan(SteelAttn* p, int heads, int tq, int tkv,
                          int head_dim) const
{
  if (p == nullptr || !steel_attn_available(head_dim)) { return false; }
  if (heads <= 0 || tq <= 0 || tkv <= 0) { return false; }

  const int bq = _attn_nax ? 64 : 32;
  const int bk = _attn_nax ? 32 : 16;

  if (p->params.empty()) {
    p->params = _mc->make_shared_buffer(sizeof(SteelAttnParams));
    if (p->params.empty()) { return false; }
  }
  auto* s = static_cast<SteelAttnParams*>(p->params.contents());
  s->B = 1;
  s->H = heads;
  s->D = head_dim;
  s->qL = tq;
  s->kL = tkv;
  s->gqa_factor = 1;               // no GQA anywhere in this DiT
  s->scale = (float)(1.0 / std::sqrt((double)head_dim));
  s->NQ = (tq + bq - 1) / bq;
  s->NK = (tkv + bk - 1) / bk;
  s->NQ_aligned = tq / bq;
  s->NK_aligned = tkv / bk;
  s->qL_rem = tq - s->NQ_aligned * bq;
  s->kL_rem = tkv - s->NK_aligned * bk;
  s->qL_off = 0;                   // only a causal mask reads this
  // Q and O are tq rows per head; K and V are tkv. Filling all four from
  // one length is the bug a self-attention-only port never sees.
  s->Q_strides[0] = (std::int64_t)heads * tq * head_dim;
  s->Q_strides[1] = (std::int64_t)tq * head_dim;
  s->Q_strides[2] = head_dim;
  s->K_strides[0] = (std::int64_t)heads * tkv * head_dim;
  s->K_strides[1] = (std::int64_t)tkv * head_dim;
  s->K_strides[2] = head_dim;
  for (int i = 0; i < 3; ++i) {
    s->V_strides[i] = s->K_strides[i];
    s->O_strides[i] = s->Q_strides[i];
  }

  // 200 says the last query tile is full, 201 the last key tile -- so
  // 201 comes from the KEY length, which for this model's cross
  // attentions is not the query length.
  vpipe::metal_compute::FunctionConstants fc;
  fc.set_bool(200, (tq % bq) == 0)
      .set_bool(201, (tkv % bk) == 0)
      .set_bool(300, false)        // has_mask
      .set_bool(301, false)        // do_causal
      .set_bool(302, false);       // has_sinks
  const char* name =
      _attn_nax ? (head_dim == 128 ? "attn_steel_nax_h_bd128_bf16"
                                   : "attn_steel_nax_h_bd64_bf16")
                : (head_dim == 128 ? "attn_steel_h_bd128_bf16"
                                   : "attn_steel_h_bd64_bf16");
  p->fn = (_attn_nax ? _lib_attn_nax : _lib_attn).function(name, fc);
  if (!p->fn.valid()) { return false; }

  p->heads = heads;
  p->tq = tq;
  p->tkv = tkv;
  p->head_dim = head_dim;
  p->bq = bq;
  return true;
}

void
MetalOps::sdpa_steel(ComputeEncoder& enc, const SteelAttn& p,
                     const SharedBuffer& q, const SharedBuffer& k,
                     const SharedBuffer& v, const SharedBuffer& out) const
{
  enc.set_function(p.fn);
  enc.set_buffer(0, q);
  enc.set_buffer(1, k);
  enc.set_buffer(2, v);
  enc.set_buffer(3, out);
  enc.set_buffer(4, p.params);
  // The mask and sink buffers (5, 6) are guarded by function constants
  // 300/302 and are not declared in this specialisation, so binding them
  // would be binding arguments the pipeline does not have.
  enc.dispatch({32 * (unsigned)((p.tq + p.bq - 1) / p.bq),
                4 * (unsigned)p.heads, 1}, {32, 4, 1});
}


// ---- the conv video VAE decoder --------------------------------------

void
MetalOps::vae_denorm_in(ComputeEncoder& enc, const SharedBuffer& latent,
                        const SharedBuffer& stdv, const SharedBuffer& meanv,
                        const SharedBuffer& out, int C, int F, int H,
                        int W) const
{
  enc.set_function(_fn_vae_denorm);
  enc.set_buffer(0, latent);
  enc.set_buffer(1, stdv);
  enc.set_buffer(2, meanv);
  enc.set_buffer(3, out);
  enc.set_constant(4, C);
  enc.set_constant(5, F);
  enc.set_constant(6, H);
  enc.set_constant(7, W);
  enc.dispatch({(unsigned)C, (unsigned)(F * H * W), 1}, {32, 1, 1});
}

void
MetalOps::vae_patchify_in(ComputeEncoder& enc, const SharedBuffer& pixels,
                          const SharedBuffer& out, int Cin, int F, int H,
                          int W, int patch) const
{
  const int cout = Cin * patch * patch;
  enc.set_function(_fn_vae_patch_in);
  enc.set_buffer(0, pixels);
  enc.set_buffer(1, out);
  enc.set_constant(2, Cin);
  enc.set_constant(3, F);
  enc.set_constant(4, H);
  enc.set_constant(5, W);
  enc.set_constant(6, patch);
  enc.dispatch({(unsigned)cout,
                (unsigned)(F * (H / patch) * (W / patch)), 1}, {32, 1, 1});
}

void
MetalOps::vae_im2col_causal(ComputeEncoder& enc, const SharedBuffer& x,
                            const SharedBuffer& out, int C, int F, int H,
                            int W, int cell0, int n_cells) const
{
  enc.set_function(_fn_vae_im2col_c);
  enc.set_buffer(0, x);
  enc.set_buffer(1, out);
  enc.set_constant(2, C);
  enc.set_constant(3, F);
  enc.set_constant(4, H);
  enc.set_constant(5, W);
  enc.set_constant(6, cell0);
  enc.set_constant(7, n_cells);
  enc.dispatch({(unsigned)C, (unsigned)n_cells, 1}, {32, 1, 1});
}

void
MetalOps::vae_s2d(ComputeEncoder& enc, const SharedBuffer& x,
                  const SharedBuffer& out, int Cin, int F, int H, int W,
                  int p1, int p2, int p3, int group) const
{
  const int cout = Cin * p1 * p2 * p3 / group;
  enc.set_function(_fn_vae_s2d);
  enc.set_buffer(0, x);
  enc.set_buffer(1, out);
  enc.set_constant(2, Cin);
  enc.set_constant(3, F);
  enc.set_constant(4, H);
  enc.set_constant(5, W);
  enc.set_constant(6, p1);
  enc.set_constant(7, p2);
  enc.set_constant(8, p3);
  enc.set_constant(9, group);
  enc.dispatch({(unsigned)cout,
                (unsigned)((F / p1) * (H / p2) * (W / p3)), 1}, {32, 1, 1});
}

void
MetalOps::vae_dup_frame0(ComputeEncoder& enc, const SharedBuffer& x,
                         const SharedBuffer& out, int C, int F, int H,
                         int W) const
{
  enc.set_function(_fn_vae_dup);
  enc.set_buffer(0, x);
  enc.set_buffer(1, out);
  enc.set_constant(2, C);
  enc.set_constant(3, F);
  enc.set_constant(4, H);
  enc.set_constant(5, W);
  enc.dispatch({(unsigned)C, (unsigned)((F + 1) * H * W), 1}, {32, 1, 1});
}

void
MetalOps::vae_whiten_out(ComputeEncoder& enc, const SharedBuffer& x,
                         const SharedBuffer& stdv, const SharedBuffer& meanv,
                         const SharedBuffer& out, int Chead, int C, int F,
                         int H, int W) const
{
  enc.set_function(_fn_vae_whiten);
  enc.set_buffer(0, x);
  enc.set_buffer(1, stdv);
  enc.set_buffer(2, meanv);
  enc.set_buffer(3, out);
  enc.set_constant(4, Chead);
  enc.set_constant(5, C);
  enc.set_constant(6, F);
  enc.set_constant(7, H);
  enc.set_constant(8, W);
  enc.dispatch({(unsigned)C, (unsigned)(F * H * W), 1}, {32, 1, 1});
}

void
MetalOps::vae_im2col(ComputeEncoder& enc, const SharedBuffer& x,
                     const SharedBuffer& out, int C, int F, int H, int W,
                     int cell0, int n_cells) const
{
  enc.set_function(_fn_vae_im2col);
  enc.set_buffer(0, x);
  enc.set_buffer(1, out);
  enc.set_constant(2, C);
  enc.set_constant(3, F);
  enc.set_constant(4, H);
  enc.set_constant(5, W);
  enc.set_constant(6, cell0);
  enc.set_constant(7, n_cells);
  enc.dispatch({(unsigned)C, (unsigned)n_cells, 1}, {32, 1, 1});
}

void
MetalOps::vae_pixel_norm_silu(ComputeEncoder& enc, const SharedBuffer& x,
                              int C, int cells) const
{
  // One threadgroup per cell; the threadgroup reduces the row. 256 is
  // the `part[]` size the kernel declares, and the reduction assumes a
  // POWER OF TWO, so it is fixed rather than derived from C.
  enc.set_function(_fn_vae_pns);
  enc.set_buffer(0, x);
  enc.set_constant(1, C);
  enc.dispatch({(unsigned)(256 * cells), 1, 1}, {256, 1, 1});
}

void
MetalOps::vae_add_into(ComputeEncoder& enc, const SharedBuffer& y,
                       const SharedBuffer& x, std::size_t n) const
{
  enc.set_function(_fn_vae_add);
  enc.set_buffer(0, y);
  enc.set_buffer(1, x);
  enc.set_constant(2, (int)n);
  enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
}

void
MetalOps::vae_d2s(ComputeEncoder& enc, const SharedBuffer& x,
                  const SharedBuffer& out, int Cin, int F, int H, int W,
                  int p1, int p2, int p3, int drop) const
{
  const int cout = Cin / (p1 * p2 * p3);
  const int ocells = (F * p1 - drop) * (H * p2) * (W * p3);
  enc.set_function(_fn_vae_d2s);
  enc.set_buffer(0, x);
  enc.set_buffer(1, out);
  enc.set_constant(2, Cin);
  enc.set_constant(3, F);
  enc.set_constant(4, H);
  enc.set_constant(5, W);
  enc.set_constant(6, p1);
  enc.set_constant(7, p2);
  enc.set_constant(8, p3);
  enc.set_constant(9, drop);
  enc.dispatch({(unsigned)cout, (unsigned)ocells, 1}, {32, 1, 1});
}

void
MetalOps::vae_unpatchify(ComputeEncoder& enc, const SharedBuffer& x,
                         const SharedBuffer& out, int Cin, int F, int H,
                         int W, int patch) const
{
  const int cout = Cin / (patch * patch);
  const int ocells = F * (H * patch) * (W * patch);
  enc.set_function(_fn_vae_unpatch);
  enc.set_buffer(0, x);
  enc.set_buffer(1, out);
  enc.set_constant(2, Cin);
  enc.set_constant(3, F);
  enc.set_constant(4, H);
  enc.set_constant(5, W);
  enc.set_constant(6, patch);
  enc.dispatch({(unsigned)cout, (unsigned)ocells, 1}, {32, 1, 1});
}

void
MetalOps::audio_denorm_in(ComputeEncoder& enc, const SharedBuffer& latent,
                          const SharedBuffer& stdv, const SharedBuffer& meanv,
                          const SharedBuffer& out, int C, int H, int W) const
{
  enc.set_function(_fn_aud_denorm);
  enc.set_buffer(0, latent);
  enc.set_buffer(1, stdv);
  enc.set_buffer(2, meanv);
  enc.set_buffer(3, out);
  enc.set_constant(4, C);
  enc.set_constant(5, H);
  enc.set_constant(6, W);
  enc.dispatch({(unsigned)C, (unsigned)(H * W), 1}, {32, 1, 1});
}

void
MetalOps::audio_im2col(ComputeEncoder& enc, const SharedBuffer& x,
                       const SharedBuffer& out, int C, int H, int W,
                       int cell0, int n_cells) const
{
  enc.set_function(_fn_aud_im2col);
  enc.set_buffer(0, x);
  enc.set_buffer(1, out);
  enc.set_constant(2, C);
  enc.set_constant(3, H);
  enc.set_constant(4, W);
  enc.set_constant(5, cell0);
  enc.set_constant(6, n_cells);
  enc.dispatch({(unsigned)C, (unsigned)n_cells, 1}, {32, 1, 1});
}

void
MetalOps::audio_up2x(ComputeEncoder& enc, const SharedBuffer& x,
                     const SharedBuffer& out, int C, int H, int W) const
{
  enc.set_function(_fn_aud_up2x);
  enc.set_buffer(0, x);
  enc.set_buffer(1, out);
  enc.set_constant(2, C);
  enc.set_constant(3, H);
  enc.set_constant(4, W);
  enc.dispatch({(unsigned)C, (unsigned)(4 * H * W), 1}, {32, 1, 1});
}

void
MetalOps::audio_mel_in(ComputeEncoder& enc, const SharedBuffer& mel,
                       const SharedBuffer& out, int C, int H, int W) const
{
  enc.set_function(_fn_aud_mel_in);
  enc.set_buffer(0, mel);
  enc.set_buffer(1, out);
  enc.set_constant(2, C);
  enc.set_constant(3, H);
  enc.set_constant(4, W);
  enc.dispatch({(unsigned)C, (unsigned)(H * W), 1}, {2, 1, 1});
}

void
MetalOps::audio_im2col_down(ComputeEncoder& enc, const SharedBuffer& x,
                            const SharedBuffer& out, int C, int H, int W,
                            int Wo, int cell0, int n_cells) const
{
  enc.set_function(_fn_aud_im2col_dn);
  enc.set_buffer(0, x);
  enc.set_buffer(1, out);
  enc.set_constant(2, C);
  enc.set_constant(3, H);
  enc.set_constant(4, W);
  enc.set_constant(5, Wo);
  enc.set_constant(6, cell0);
  enc.set_constant(7, n_cells);
  enc.dispatch({(unsigned)C, (unsigned)n_cells, 1}, {32, 1, 1});
}

void
MetalOps::audio_whiten_rows(ComputeEncoder& enc, const SharedBuffer& x,
                            const SharedBuffer& stdv,
                            const SharedBuffer& meanv,
                            const SharedBuffer& out, int Chead, int Z, int H,
                            int W) const
{
  enc.set_function(_fn_aud_whiten);
  enc.set_buffer(0, x);
  enc.set_buffer(1, stdv);
  enc.set_buffer(2, meanv);
  enc.set_buffer(3, out);
  enc.set_constant(4, Chead);
  enc.set_constant(5, Z);
  enc.set_constant(6, H);
  enc.set_constant(7, W);
  enc.dispatch({(unsigned)(Z * W), (unsigned)H, 1}, {32, 1, 1});
}

void
MetalOps::audio_out(ComputeEncoder& enc, const SharedBuffer& x,
                    const SharedBuffer& out, int C, int H, int W) const
{
  enc.set_function(_fn_aud_out);
  enc.set_buffer(0, x);
  enc.set_buffer(1, out);
  enc.set_constant(2, C);
  enc.set_constant(3, H);
  enc.set_constant(4, W);
  enc.dispatch({(unsigned)C, (unsigned)(H * W), 1}, {32, 1, 1});
}

}  // namespace ltx25
