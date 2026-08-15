// The QUANTIZED linear, against the dense one it replaces.
//
// WHY THIS EXISTS RATHER THAN JUST CONVERTING THE CHECKPOINT. Quantizing
// the 42 GB DiT is a long one-way run, and if the result is wrong every
// symptom looks the same as every other DiT bug: a latent that is finite
// and structured and simply not the right picture. This test asks the
// only question that matters -- does `MetalOps::linear` over a packed
// weight compute what the dense GEMM computes -- with no checkpoint, no
// model, and a few hundred KB of synthetic weights.
//
// It packs the codes BY HAND rather than calling vpipe's own
// AffineQuantizer (which is not in the installed SDK anyway). That is
// the point: the layout is then verified independently instead of being
// trusted twice, so a misreading of the packing shows up HERE as a
// failed comparison rather than downstream as a bad video.
//
// The five things it pins, each of which is a silent wrong answer if
// missed:
//
//   * the U32 code packing -- LOW bits first, `32/bits` codes per word;
//   * `scales`/`biases` are F16 in a checkpoint and BFLOAT to the
//     kernel, so the loader must convert (see get_as_bf16_);
//   * the qmm kernel has NO bias slot, so a linear's bias is a second
//     pass (ltx_bias_add), not buffer 2 -- buffer 2 is the quantization
//     zero-point, a different tensor with a confusingly similar name;
//   * the BM=32 / BM=64 tile split at M, both arms;
//   * on a matrix-core GPU, the three matmul2d tiles `linear` routes
//     between by K -- each with its own grid divisor, and none of them
//     reachable from the K = 256 / 512 the cases above use.
//
// NOTE for a matrix-core host: the cases above compare the quantized
// linear against the dense one, and on M5 BOTH now expand to a dense
// bf16 weight and run the same matmul2d, so they agree exactly and the
// affine qmm kernel is no longer what they measure. It still runs on
// every pre-M5 GPU and as the fallback whenever the mma path declines a
// shape, and `VPIPE_LTX25_NO_MMA=1` puts this whole file back on it.

#include "ltx25-config.h"
#include "ltx25-metal-ops.h"

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/metal-compute.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

extern "C" const unsigned char ltx25_kernels_bf16_metallib[];
extern "C" const unsigned long ltx25_kernels_bf16_metallib_len;

namespace {

int g_fail = 0;

void
check(bool ok, const std::string& what)
{
  std::printf("  [ %s ] %s\n", ok ? "OK" : "FAIL", what.c_str());
  if (!ok) { ++g_fail; }
}

std::uint16_t
to_bf16(float v)
{
  std::uint32_t u;
  std::memcpy(&u, &v, 4);
  const std::uint32_t r = ((u >> 16) & 1u) + 0x7fffu;   // RNE
  return (std::uint16_t)((u + r) >> 16);
}

float
from_bf16(std::uint16_t h)
{
  const std::uint32_t u = (std::uint32_t)h << 16;
  float v;
  std::memcpy(&v, &u, 4);
  return v;
}

// Group-affine quantize one [N][K] row-major weight, MLX layout:
//   bias  = group min
//   scale = (max - min) / (2^bits - 1)
//   w ~= scale * q + bias
// Codes pack LOW bits first, `32 / bits` per u32 word. `deq` receives the
// weight as the kernel will see it, so the comparison below is against
// the dequantized weight and isolates the GEMM from the quantization
// error -- otherwise a passing tolerance could hide either one.
void
quantize(const std::vector<float>& w, int N, int K, int bits, int group,
         std::vector<std::uint32_t>& codes, std::vector<std::uint16_t>& scales,
         std::vector<std::uint16_t>& biases, std::vector<float>& deq)
{
  const int per_word = 32 / bits;
  const int gcols    = K / group;
  const int wcols    = K / per_word;
  const float qmax   = (float)((1 << bits) - 1);

  codes.assign((std::size_t)N * wcols, 0u);
  scales.assign((std::size_t)N * gcols, 0);
  biases.assign((std::size_t)N * gcols, 0);
  deq.assign((std::size_t)N * K, 0.0f);

  for (int n = 0; n < N; ++n) {
    for (int g = 0; g < gcols; ++g) {
      const int k0 = g * group;
      float lo = w[(std::size_t)n * K + k0], hi = lo;
      for (int j = 1; j < group; ++j) {
        const float v = w[(std::size_t)n * K + k0 + j];
        lo = std::fmin(lo, v);
        hi = std::fmax(hi, v);
      }
      const float sc = (hi - lo) / qmax;
      // The kernel reads scale/bias at the checkpoint's precision, so
      // round them here too -- otherwise the reference dequant below is
      // more accurate than anything the GPU can be.
      const std::uint16_t sc_b = to_bf16(sc);
      const std::uint16_t lo_b = to_bf16(lo);
      const float scr = from_bf16(sc_b), lor = from_bf16(lo_b);
      scales[(std::size_t)n * gcols + g] = sc_b;
      biases[(std::size_t)n * gcols + g] = lo_b;

      for (int j = 0; j < group; ++j) {
        const int k = k0 + j;
        int q = scr > 0.0f
                    ? (int)std::lround((w[(std::size_t)n * K + k] - lor) / scr)
                    : 0;
        q = q < 0 ? 0 : (q > (int)qmax ? (int)qmax : q);
        deq[(std::size_t)n * K + k] = scr * (float)q + lor;
        const int word = k / per_word;
        const int slot = k % per_word;
        codes[(std::size_t)n * wcols + word] |=
            ((std::uint32_t)q) << (bits * slot);
      }
    }
  }
}

double
rel_l2(const std::vector<float>& a, const std::vector<float>& b)
{
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = (double)a[i] - (double)b[i];
    num += d * d;
    den += (double)b[i] * (double)b[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

}  // namespace

int
main()
{
  using vpipe::metal_compute::CommandStream;
  using vpipe::metal_compute::MetalCompute;
  using vpipe::metal_compute::SharedBuffer;

  MetalCompute mc(nullptr);
  if (!mc.valid()) {
    std::printf("SKIPPED: no Metal device. NOTHING was checked.\n");
    return 0;
  }
  mc.register_metal_library(ltx25::kMetalLibBf16,
                            ltx25_kernels_bf16_metallib,
                            ltx25_kernels_bf16_metallib_len);

  ltx25::MetalOps ops;
  std::string err;
  if (!ops.init(&mc, &err)) {
    std::printf("ops.init: %s\n", err.c_str());
    return 1;
  }
  if (!ops.quant_available()) {
    std::printf("this host has no affine_qmm_steel_bf16 -- skipping\n");
    return 0;
  }
  check(true, "the affine qmm kernels resolved");

  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.0f, 0.05f);

  // Both tile arms: M=16 takes BM=32, M=192 takes BM=64. The split is a
  // host-side choice, so a wrong threshold is invisible to any golden.
  const struct { int M, N, K, bits, group; double tol; } cases[] = {
      {16,  128, 256, 8, 64, 3e-3},
      {192, 128, 256, 8, 64, 3e-3},
      {16,  128, 256, 4, 64, 3e-3},
      {192, 128, 256, 4, 64, 3e-3},
      {192, 128, 256, 4, 32, 3e-3},
      {192, 256, 512, 8, 32, 3e-3},
  };

  for (const auto& c : cases) {
    const std::string tag = "M=" + std::to_string(c.M) + " N=" +
                            std::to_string(c.N) + " K=" +
                            std::to_string(c.K) + " w" +
                            std::to_string(c.bits) + "g" +
                            std::to_string(c.group);

    std::vector<float> x((std::size_t)c.M * c.K), w((std::size_t)c.N * c.K);
    for (auto& v : x) { v = nd(rng); }
    for (auto& v : w) { v = nd(rng); }

    std::vector<std::uint32_t> codes;
    std::vector<std::uint16_t> sc, bi;
    std::vector<float> deq;
    quantize(w, c.N, c.K, c.bits, c.group, codes, sc, bi, deq);

    // Upload. `deq` is the weight the kernel will effectively use, so
    // the DENSE reference runs on deq -- this compares two GEMMs, not a
    // GEMM against a quantization.
    const SharedBuffer xb  = ops.upload_bf16(x);
    const SharedBuffer wb  = ops.upload_bf16(deq);
    const SharedBuffer y_d = mc.make_shared_buffer((std::size_t)c.M * c.N * 2);
    const SharedBuffer y_q = mc.make_shared_buffer((std::size_t)c.M * c.N * 2);

    SharedBuffer cb = mc.make_shared_buffer(codes.size() * 4);
    std::memcpy(cb.contents(), codes.data(), codes.size() * 4);
    SharedBuffer sb = mc.make_shared_buffer(sc.size() * 2);
    std::memcpy(sb.contents(), sc.data(), sc.size() * 2);
    SharedBuffer bb = mc.make_shared_buffer(bi.size() * 2);
    std::memcpy(bb.contents(), bi.data(), bi.size() * 2);

    ltx25::QWeight q;
    q.codes = std::move(cb);
    q.scales = std::move(sb);
    q.qbias = std::move(bb);
    q.bits = c.bits;
    q.group = c.group;
    q.quantized = true;

    ltx25::QWeight dense;
    dense.w = wb.subview(0, wb.byte_size());

    {
      CommandStream cs = mc.make_command_stream();
      {
        auto enc = cs.begin_compute();
        ops.linear(enc, xb, dense, nullptr, y_d, c.M, c.K, c.N);
        ops.linear(enc, xb, q, nullptr, y_q, c.M, c.K, c.N);
      }
      cs.commit().wait();
    }

    const auto a = ltx25::MetalOps::download_bf16(y_d,
                                                  (std::size_t)c.M * c.N);
    const auto b = ltx25::MetalOps::download_bf16(y_q,
                                                  (std::size_t)c.M * c.N);
    const double r = rel_l2(b, a);
    std::printf("       %s: rel-L2 %.3e\n", tag.c_str(), r);
    check(r < c.tol, tag + " matches the dense GEMM");

    // Not all zero -- a kernel that never ran also compares "equal" to
    // itself, and an unvalidated ComputeFunction is exactly a no-op.
    double mag = 0.0;
    for (float v : b) { mag += std::fabs((double)v); }
    check(mag > 0.0, tag + " produced a non-zero result");
  }

  // ---- NEGATIVE CONTROL --------------------------------------------
  //
  // Every case above came back at rel-L2 EXACTLY 0.000e+00. That is a
  // believable result -- both kernels accumulate in f32 over the same K
  // order and round once to bf16 -- but "exactly equal" is also what a
  // vacuous test returns, so it does not get taken on trust. Perturb ONE
  // code word and the answer must move. If it does not, the qmm result
  // is not a function of the codes and nothing above measured anything.
  {
    const int M = 64, N = 128, K = 256, bits = 4, group = 64;
    std::vector<float> x((std::size_t)M * K), w((std::size_t)N * K);
    for (auto& v : x) { v = nd(rng); }
    for (auto& v : w) { v = nd(rng); }

    std::vector<std::uint32_t> codes;
    std::vector<std::uint16_t> sc, bi;
    std::vector<float> deq;
    quantize(w, N, K, bits, group, codes, sc, bi, deq);

    const SharedBuffer xb = ops.upload_bf16(x);
    const SharedBuffer y0 = mc.make_shared_buffer((std::size_t)M * N * 2);
    const SharedBuffer y1 = mc.make_shared_buffer((std::size_t)M * N * 2);

    auto make = [&](const std::vector<std::uint32_t>& cc) {
      ltx25::QWeight q;
      q.codes = mc.make_shared_buffer(cc.size() * 4);
      std::memcpy(q.codes.contents(), cc.data(), cc.size() * 4);
      q.scales = mc.make_shared_buffer(sc.size() * 2);
      std::memcpy(q.scales.contents(), sc.data(), sc.size() * 2);
      q.qbias = mc.make_shared_buffer(bi.size() * 2);
      std::memcpy(q.qbias.contents(), bi.data(), bi.size() * 2);
      q.bits = bits;
      q.group = group;
      q.quantized = true;
      return q;
    };
    std::vector<std::uint32_t> bad = codes;
    bad[0] ^= 0xFu;                    // one 4-bit code in row 0

    const ltx25::QWeight qa = make(codes);
    const ltx25::QWeight qb = make(bad);
    {
      CommandStream cs = mc.make_command_stream();
      {
        auto enc = cs.begin_compute();
        ops.linear(enc, xb, qa, nullptr, y0, M, K, N);
        ops.linear(enc, xb, qb, nullptr, y1, M, K, N);
      }
      cs.commit().wait();
    }
    const auto a = ltx25::MetalOps::download_bf16(y0, (std::size_t)M * N);
    const auto b = ltx25::MetalOps::download_bf16(y1, (std::size_t)M * N);
    const double r = rel_l2(b, a);
    std::printf("       control (one code flipped): rel-L2 %.3e\n", r);
    check(r > 0.0, "the qmm result DEPENDS on the codes (not vacuous)");
  }

  // The BIAS, which the qmm kernel cannot take: it must arrive through
  // the second ltx_bias_add pass. Checked on its own because a dropped
  // bias is a small, plausible-looking shift rather than a blow-up.
  {
    const int M = 64, N = 128, K = 256, bits = 4, group = 64;
    std::vector<float> x((std::size_t)M * K), w((std::size_t)N * K),
        bias((std::size_t)N);
    for (auto& v : x) { v = nd(rng); }
    for (auto& v : w) { v = nd(rng); }
    for (auto& v : bias) { v = nd(rng) * 4.0f; }

    std::vector<std::uint32_t> codes;
    std::vector<std::uint16_t> sc, bi;
    std::vector<float> deq;
    quantize(w, N, K, bits, group, codes, sc, bi, deq);

    const SharedBuffer xb = ops.upload_bf16(x);
    const SharedBuffer wb = ops.upload_bf16(deq);
    const SharedBuffer bsb = ops.upload_bf16(bias);
    const SharedBuffer y_d = mc.make_shared_buffer((std::size_t)M * N * 2);
    const SharedBuffer y_q = mc.make_shared_buffer((std::size_t)M * N * 2);

    ltx25::QWeight q;
    q.codes = mc.make_shared_buffer(codes.size() * 4);
    std::memcpy(q.codes.contents(), codes.data(), codes.size() * 4);
    q.scales = mc.make_shared_buffer(sc.size() * 2);
    std::memcpy(q.scales.contents(), sc.data(), sc.size() * 2);
    q.qbias = mc.make_shared_buffer(bi.size() * 2);
    std::memcpy(q.qbias.contents(), bi.data(), bi.size() * 2);
    q.bits = bits;
    q.group = group;
    q.quantized = true;

    ltx25::QWeight dense;
    dense.w = wb.subview(0, wb.byte_size());

    {
      CommandStream cs = mc.make_command_stream();
      {
        auto enc = cs.begin_compute();
        ops.linear(enc, xb, dense, &bsb, y_d, M, K, N);
        ops.linear(enc, xb, q, &bsb, y_q, M, K, N);
      }
      cs.commit().wait();
    }
    const auto a = ltx25::MetalOps::download_bf16(y_d, (std::size_t)M * N);
    const auto b = ltx25::MetalOps::download_bf16(y_q, (std::size_t)M * N);
    const double r = rel_l2(b, a);
    std::printf("       with bias: rel-L2 %.3e\n", r);
    check(r < 3e-3, "the quantized linear's BIAS pass matches");

    // And that the bias is actually present: without it the result would
    // differ from the biased dense reference by the bias itself, which
    // this rel-L2 would catch -- but only if the bias is large enough to
    // see. Assert that directly rather than assume.
    double bmag = 0.0;
    for (float v : bias) { bmag += std::fabs((double)v); }
    double omag = 0.0;
    for (float v : a) { omag += std::fabs((double)v); }
    check(bmag / (double)N > 0.1 * (omag / (double)a.size()),
          "the bias is large enough for its absence to fail the test");
  }

  // ---- the M5 TILE ROUTING -----------------------------------------
  //
  // On a matrix-core GPU `linear` picks one of three matmul2d entry
  // points -- N-regions of 128, 256 and the TN=2 tile's 512 -- from M, K
  // and N together, and each needs its OWN grid divisor. Get that
  // divisor wrong and the GEMM computes a wrong answer at full speed;
  // nothing above would see it, because every case in this file uses
  // K = 256 or 512 and lands in one corner of the rule.
  //
  // The cases below cover every BRANCH of that rule, not every K: the
  // narrow-N guard, both deep-K arms, the large-M arm, and both sides of
  // each boundary. What they check is the DIVISOR, which is why the bar
  // is numeric agreement rather than a tile name -- the rule is tuned by
  // measurement and is expected to move.
  //
  // The reference is the STEEL GEMM the routing replaces, reached by
  // initialising a second MetalOps with VPIPE_LTX25_NO_MMA set. The knob
  // is read in init(), so both routes are live in one process and the
  // comparison needs no subprocess and no second run.
  //
  // LTX reaches all three bands: the video projections at K=4096, the
  // audio ff_out at K=8192 and a VAE level at K=6912, the video ff_out
  // at K=16384. N is deliberately not always a multiple of the region,
  // so the grid's round-up is exercised too.
  if (!mc.supports_matrix_cores()) {
    std::printf("       (no matrix cores -- tile routing not applicable)\n");
  } else {
    ltx25::MetalOps ops_steel;
    std::string serr;
    ::setenv("VPIPE_LTX25_NO_MMA", "1", 1);
    const bool ok_steel = ops_steel.init(&mc, &serr);
    ::unsetenv("VPIPE_LTX25_NO_MMA");
    check(ok_steel, "the steel reference ops initialised");

    const struct { const char* band; int M, K, N; } bands[] = {
        {"small M, shallow K", 512, 4096, 4096},
        {"large M, shallow K (DiT attn)", 2048, 4096, 4096},
        {"large M, deep-K just under", 2048, 12287, 512},
        {"deep K, narrow N (VAE L1)", 2427, 13824, 512},
        {"deep K, wide N (DiT ff_out)", 256, 16384, 4096},
        {"deep K at the bound", 256, 12288, 512},
        {"narrow-N guard (conv_out)", 2048, 3456, 48},
        {"narrow-N guard at N=128", 2048, 3456, 128},
        {"N=129, just past the guard", 2048, 3456, 129},
        {"N not a multiple of 512", 2048, 8192, 300},
        {"N not a multiple of 256", 192, 16384, 260},
        {"VAE encoder conv_in (K=81)", 4096, 81, 128},
    };
    for (const auto& b : bands) {
      std::vector<float> x((std::size_t)b.M * b.K), w((std::size_t)b.N * b.K);
      for (auto& v : x) { v = nd(rng); }
      for (auto& v : w) { v = nd(rng); }
      const SharedBuffer xb = ops.upload_bf16(x);
      const SharedBuffer wb = ops.upload_bf16(w);
      const std::size_t yn = (std::size_t)b.M * b.N;
      const SharedBuffer y_m = mc.make_shared_buffer(yn * 2);
      const SharedBuffer y_s = mc.make_shared_buffer(yn * 2);

      ltx25::QWeight dense;
      dense.w = wb.subview(0, wb.byte_size());
      {
        CommandStream cs = mc.make_command_stream();
        {
          auto enc = cs.begin_compute();
          ops.linear(enc, xb, dense, nullptr, y_m, b.M, b.K, b.N);
          ops_steel.linear(enc, xb, dense, nullptr, y_s, b.M, b.K, b.N);
        }
        cs.commit().wait();
      }
      const auto a = ltx25::MetalOps::download_bf16(y_m, yn);
      const auto s = ltx25::MetalOps::download_bf16(y_s, yn);
      const double r = rel_l2(a, s);
      std::printf("       %-32s rel-L2 %.3e\n", b.band, r);
      // The two kernels accumulate in different orders, so this is a
      // rounding bar, not an equality one. A mis-divided grid does not
      // land near it: it reads the wrong weight columns entirely.
      check(r < 3e-3 && std::isfinite(r),
            std::string("tile routing ") + b.band + " matches steel");
    }
  }

  if (g_fail != 0) {
    std::printf("FAILURES: %d\n", g_fail);
    return 1;
  }
  std::printf("ALL PASSED\n");
  return 0;
}
