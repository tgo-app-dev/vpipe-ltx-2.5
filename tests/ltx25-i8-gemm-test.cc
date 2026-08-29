// The accelerated int8 tier, against the bf16 tiles it sits on top of.
//
// `i8_gemm` quantizes the activation per (row, 512-group) on the fly and
// the weight per (out-channel, 512-group) per call, and runs the product
// on the matrix units' int8 pipe. It takes the SAME dense weight the
// matmul2d tiles would have read -- the dequant-once expansion for a
// quantized checkpoint -- so it composes with that rather than replacing
// it, and a shape it declines falls through with nothing encoded.
//
// It is LOSSY, which is the whole reason it is opt-in, so what this file
// has to establish is three things and not one:
//
//   * the error is where the mode says it is (~1e-2 rel-L2 per GEMM) and
//     not somewhere else, at the shapes this DiT actually dispatches;
//   * the tier is TAKEN at those shapes and REFUSED at the ones its
//     gates exclude -- because both tiers compute the same product to
//     within that tolerance, a test that did not assert the path would
//     pass while measuring the one it was not testing;
//   * it is faster, measured with the arms interleaved.
//
// No checkpoint and no golden: the weights are drawn from a fixed PRNG.

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include "ltx25-config.h"
#include "ltx25-metal-ops.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

extern "C" const unsigned char ltx25_kernels_bf16_metallib[];
extern "C" const unsigned int ltx25_kernels_bf16_metallib_len;

namespace {

int g_failed = 0;

void
check(bool ok, const std::string& what)
{
  std::printf("  [%s] %s\n", ok ? " OK " : "FAIL", what.c_str());
  if (!ok) { ++g_failed; }
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
  if (!mc.supports_matrix_cores()) {
    std::printf("SKIPPED: no matrix cores, so there is no int8 pipe to "
                "measure. NOTHING was checked.\n");
    return 0;
  }
  ops.set_i8_gemm(true);
  if (!ops.i8_takes(8160, 4096, 4096)) {
    std::printf("SKIPPED: this host did not resolve the int8 kernels. "
                "NOTHING was checked.\n");
    return 0;
  }
  check(true, "the int8 tier resolved and accepts the DiT's own shape");

  // The DiT's real projections at a token count a clip reaches, plus the
  // two shapes the tier's gates are supposed to REFUSE.
  struct Shape { const char* what; int M, K, N; bool want; };
  const Shape shapes[] = {
      {"attn projection      ", 8160,  4096,  4096, true},
      {"ff widening          ", 8160,  4096, 16384, true},
      {"ff narrowing         ", 8160, 16384,  4096, true},
      // caption 3840 is NOT a multiple of 512: it pads to 4096, which is
      // 6.7% and inside the 10% cap, so the tier takes it. This is the
      // row that would break first if that cap moved.
      {"cross-attn k (padded)", 8160,  3840,  4096, true},
      // Refused: under the 1024-row crossover, and a contraction too
      // shallow for the int8 rate to pay back the two quantize passes.
      {"short M (refused)    ",  512,  4096,  4096, false},
      {"shallow K (refused)  ", 8160,   512,  4096, false},
  };

  std::mt19937 rng(11);
  std::normal_distribution<float> nd(0.0f, 0.05f);

  for (const Shape& sh : shapes) {
    const std::string tag = sh.what;
    if (ops.i8_takes(sh.M, sh.K, sh.N) != sh.want) {
      check(false, tag + (sh.want ? ": expected the int8 tier, got the "
                                    "bf16 tiles"
                                  : ": expected the bf16 tiles, got the "
                                    "int8 tier"));
      continue;
    }
    std::vector<float> x((std::size_t)sh.M * sh.K);
    std::vector<float> w((std::size_t)sh.N * sh.K);
    for (auto& v : x) { v = nd(rng); }
    for (auto& v : w) { v = nd(rng); }
    auto xb = ops.upload_bf16(x);
    auto wb = ops.upload_bf16(w);
    auto yb = ops.alloc((std::size_t)sh.M * sh.N);
    if (xb.empty() || wb.empty() || yb.empty()) {
      std::printf("  %s: allocation failed, skipped\n", sh.what);
      continue;
    }
    auto run = [&](bool i8) {
      ops.set_i8_gemm(i8);
      auto stream = mc.make_command_stream();
      {
        auto enc = stream.begin_compute();
        ops.linear(enc, xb, wb, nullptr, yb, sh.M, sh.K, sh.N);
      }
      stream.commit().wait();
      return ltx25::MetalOps::download_bf16(yb,
                                            (std::size_t)sh.M * sh.N);
    };
    const std::vector<float> base = run(false);
    const std::vector<float> acc  = run(true);
    const double r = rel_l2(acc, base);
    std::printf("       %-22s rel-L2 %.3e  [%s]\n", sh.what, r,
                sh.want ? "int8" : "bf16 (tier refused)");
    if (sh.want) {
      // The mode documents ~1e-2 per GEMM. 3e-2 is loose enough not to
      // fail on a different draw and an order below a transposition,
      // which lands at O(1).
      check(r < 3e-2, tag + ": within the int8 tier's stated error");
      // And it must actually BE lossy here: a rel-L2 of zero would mean
      // both arms ran the same kernel and the A/B measured nothing.
      check(r > 1e-5, tag + ": the two tiers really differ");
    } else {
      check(r == 0.0, tag + ": refused, so both arms are the same kernel");
    }
  }

  // THE A/B. Arms interleaved one per round, best-of: measuring them
  // sequentially lets the first absorb first-touch costs and leaves the
  // two in different SoC power states, which is how a 7% regression once
  // read as a 1.2x win in the in-tree shared/mma-tile.h.
  std::printf("  --- rate: int8 vs the bf16 tiles ---\n");
  for (const Shape& sh : shapes) {
    if (!sh.want) { continue; }
    auto xb = ops.alloc((std::size_t)sh.M * sh.K);
    auto wb = ops.alloc((std::size_t)sh.N * sh.K);
    auto yb = ops.alloc((std::size_t)sh.M * sh.N);
    if (xb.empty() || wb.empty() || yb.empty()) { continue; }
    auto once = [&](bool i8) {
      ops.set_i8_gemm(i8);
      auto stream = mc.make_command_stream();
      {
        auto enc = stream.begin_compute();
        for (int i = 0; i < 4; ++i) {
          ops.linear(enc, xb, wb, nullptr, yb, sh.M, sh.K, sh.N);
        }
      }
      const auto t0 = std::chrono::steady_clock::now();
      stream.commit().wait();
      return std::chrono::duration<double, std::milli>(
                 std::chrono::steady_clock::now() - t0).count() / 4.0;
    };
    once(true); once(false);                  // warm both before timing
    double i8 = 1e9, bf = 1e9;
    for (int round = 0; round < 5; ++round) {
      i8 = std::min(i8, once(true));
      bf = std::min(bf, once(false));
    }
    const double gflop = 2.0 * sh.M * sh.K * sh.N / 1e9;
    std::printf("       %-22s int8 %7.2f ms (%6.0f GFLOP/s)  bf16 %7.2f ms"
                "  %.2fx\n", sh.what, i8, gflop / (i8 / 1000.0), bf, bf / i8);
  }
  ops.set_i8_gemm(false);

  std::printf("ltx25-i8-gemm: %s\n",
              g_failed == 0 ? "all checks passed" : "FAILURES");
  return g_failed == 0 ? 0 : 1;
}
