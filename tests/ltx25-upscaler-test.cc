// The latent upscalers on the GPU, against the CPU reference.
//
// The reference is the thing pinned to the released checkpoints
// (ltx25-upscaler-ref-test, ~4e-6 at every tap), so this compares the
// two implementations rather than re-deriving the model. What it is
// really checking is the three places the GPU path does something the
// reference does not:
//
//   * a ZERO-padded im2col. The VAE's gathers replicate in time or pad
//     frame 0 twice; using either here is wrong only at the first and
//     last frame, which a mean would hide.
//   * GroupNorm reducing across a whole group on channel-LAST data,
//     where the reference walks channel-first.
//   * vae_d2s standing in for PixelShuffleND. The claim is that its
//     (c p1 p2 p3) split with width fastest IS PixelShuffleND's
//     nesting; if it is not, the output is a correctly-shaped
//     permutation.
//
// bf16 against f32 through ~17 convolutions and 10 GroupNorms.

#include "ltx25-config.h"
#include "ltx25-metal-ops.h"
#include "ltx25-upscaler-ref.h"
#include "ltx25-upscaler.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "generative-models/weight-set.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

extern "C" const unsigned char ltx25_kernels_bf16_metallib[];
extern "C" const unsigned long ltx25_kernels_bf16_metallib_len;

using vpipe::genai::WeightSet;
using vpipe::metal_compute::MetalCompute;

namespace {

int g_fail = 0, g_ran = 0;

void
check(bool ok, const std::string& what)
{
  std::printf("  [%s] %s\n", ok ? " OK " : "FAIL", what.c_str());
  if (!ok) { ++g_fail; }
  ++g_ran;
}

double
rel_l2(const std::vector<float>& a, const std::vector<float>& b)
{
  if (a.size() != b.size() || a.empty()) { return 1e9; }
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = (double)a[i] - (double)b[i];
    num += d * d;
    den += (double)b[i] * (double)b[i];
  }
  return (den > 0.0) ? std::sqrt(num / den) : std::sqrt(num);
}

struct Rng {
  std::uint64_t s;
  explicit Rng(std::uint64_t seed) : s(seed * 6364136223846793005ull + 1) {}
  float next()
  {
    s = s * 6364136223846793005ull + 1442695040888963407ull;
    return 2.0f * ((float)(std::uint32_t)(s >> 33) / 4294967296.0f) - 1.0f;
  }
};

void
run_one(const std::string& root, MetalCompute& mc, ltx25::MetalOps& ops,
        const std::string& kind, const std::string& file)
{
  std::printf("\n%s upscaler\n", kind.c_str());
  const std::string path = root + "/latent_upscale_models/" + file;
  ltx25::UpscalerConfig cfg;
  std::string err;
  if (!ltx25::UpscalerConfig::from_metadata(path, &cfg, &err)) {
    check(false, "config: " + err);
    return;
  }
  auto ws = WeightSet::open(path, nullptr);
  if (!ws) { check(false, "open " + path); return; }

  // No VAE statistics here: identity either side, so this compares the
  // MODEL. The un/re-normalize pair is vae_denorm_in / vae_whiten_out,
  // which the VAE's own tests already pin.
  const std::vector<float> none;
  auto gpu = ltx25::Ltx25Upscaler::load(cfg, none, none, ws, ops, &err);
  if (!gpu) { check(false, "GPU load: " + err); return; }
  auto ref = ltx25::Ltx25UpscalerRef::load(cfg, *ws, &mc, &err);
  if (!ref) { check(false, "reference load: " + err); return; }
  check(true, "both paths loaded");

  // TWO GEOMETRIES. The tiny one is fast and exercises every layout
  // decision; the second is a REAL clip's latent (9 pixel frames at
  // 960x544) shrunk until the naive reference finishes: the reference is
  // O(cells x Cout x Cin x 27), so a real 1020-cell latent is ~490 GFLOP
  // single-threaded. F=2 is the case where the temporal branch emits
  // 2F-1 = 3 frames, and H != W catches an axis swap that a square grid
  // cannot.
  for (const auto& geo : std::vector<std::array<int, 3>>{{3, 2, 2},
                                                         {2, 6, 8}}) {
  const int C = cfg.in_channels, F = geo[0], H = geo[1], W = geo[2];
  std::printf("     --- F=%d H=%d W=%d ---\n", F, H, W);
  std::vector<float> in((std::size_t)C * F * H * W);
  Rng r(kind == "spatial" ? 21u : 22u);
  for (float& v : in) { v = r.next(); }

  ref->set_capture(true);
  gpu->set_capture(true);
  std::vector<float> a, b;
  std::array<int, 4> sa{}, sb{};
  if (!ref->forward(in.data(), F, H, W, &a, &sa, &err)) {
    check(false, "reference forward: " + err);
    return;
  }
  if (!gpu->upscale(in.data(), F, H, W, &b, &sb, &err)) {
    check(false, "GPU upscale: " + err);
    return;
  }
  check(sa == sb, "both produce [" + std::to_string(sb[0]) + "," +
        std::to_string(sb[1]) + "," + std::to_string(sb[2]) + "," +
        std::to_string(sb[3]) + "]");
  // TAPS FIRST: the first stage to move is the one that broke, and a
  // whole-model number cannot say which.
  //
  // The GPU keeps its intermediates channel-LAST while the reference is
  // channel-FIRST, so each is compared after transposing the GPU's --
  // otherwise every tap would read as a permutation and say nothing.
  for (const char* t : {"initial", "res", "upsampled", "post"}) {
    const std::vector<float>* rv = ref->tap(t);
    const std::vector<float>* gv = gpu->tap(t);
    if (rv == nullptr || gv == nullptr || rv->size() != gv->size()) {
      std::printf("       %-10s (not captured on both)\n", t);
      continue;
    }
    // channel-last [cells][C] -> channel-first [C][cells]
    std::array<int, 4> sh{};
    if (!ref->tap_shape(t, &sh)) { continue; }
    const std::size_t cellsn =
        (std::size_t)sh[1] * (std::size_t)sh[2] * (std::size_t)sh[3];
    std::vector<float> gt(gv->size());
    for (std::size_t cell = 0; cell < cellsn; ++cell) {
      for (int c = 0; c < sh[0]; ++c) {
        gt[(std::size_t)c * cellsn + cell] =
            (*gv)[cell * (std::size_t)sh[0] + (std::size_t)c];
      }
    }
    std::printf("       %-10s rel-L2 %.3e\n", t, rel_l2(gt, *rv));
  }

  // THE UNINSTRUMENTED PATH, which is the one a graph runs. Capture
  // forces a commit between stages, and this model was once correct
  // ONLY with it on: built as one command buffer it returned rel-L2
  // 0.97 against the reference, deterministically and with no GPU error
  // reported. So the production path is asserted separately, or the
  // diagnostics above would be hiding the defect they exist to find.
  {
    gpu->set_capture(false);
    std::vector<float> plain;
    std::array<int, 4> sp{};
    std::string e2;
    if (!gpu->upscale(in.data(), F, H, W, &plain, &sp, &e2)) {
      check(false, "uninstrumented upscale: " + e2);
    } else {
      const double dp = rel_l2(plain, a);
      std::printf("       uninstrumented    rel-L2 %.3e\n", dp);
      check(dp < 6e-2, "and WITHOUT capture -- the path a graph takes");
    }
    gpu->set_capture(true);
  }

  const double d = rel_l2(b, a);
  std::printf("       GPU vs reference  rel-L2 %.3e\n", d);

  // WHERE the error sits, not just how big it is. bf16 over depth is
  // FLAT across frames and across channel groups; a padding mistake
  // spikes the FIRST and LAST frame (those are the only cells whose
  // window leaves the volume), and a GroupNorm mistake spikes one group.
  // Both would still be small enough to pass an aggregate bar.
  {
    const int oF = sb[1], oH = sb[2], oW = sb[3], oc = sb[0];
    const std::size_t plane = (std::size_t)oH * oW;
    double worst_f = 0.0, best_f = 1e9;
    for (int f = 0; f < oF; ++f) {
      double num = 0.0, den = 0.0;
      for (int c = 0; c < oc; ++c) {
        for (std::size_t i = 0; i < plane; ++i) {
          const std::size_t k =
              ((std::size_t)c * oF + f) * plane + i;
          const double e = (double)b[k] - (double)a[k];
          num += e * e;
          den += (double)a[k] * (double)a[k];
        }
      }
      const double rf = den > 0.0 ? std::sqrt(num / den) : 0.0;
      worst_f = std::max(worst_f, rf);
      best_f = std::min(best_f, rf);
      std::printf("         frame %d  rel-L2 %.3e\n", f, rf);
    }
    // A padding error at the edges would put the first/last frame well
    // clear of the middle. 3x is generous for bf16 spread and far below
    // what a wrong pad produces.
    check(worst_f < 3.0 * std::max(best_f, 1e-9),
          "the error is FLAT across frames (worst/best " +
          std::to_string(worst_f / std::max(best_f, 1e-9)) +
          "x) -- an edge-padding mistake would spike frame 0 and the last");

    double worst_g = 0.0, best_g = 1e9;
    const int per = oc / 32;
    for (int g = 0; g < 32; ++g) {
      double num = 0.0, den = 0.0;
      for (int k = 0; k < per; ++k) {
        const int c = g * per + k;
        for (std::size_t i = 0; i < (std::size_t)oF * plane; ++i) {
          const std::size_t o = (std::size_t)c * oF * plane + i;
          const double e = (double)b[o] - (double)a[o];
          num += e * e;
          den += (double)a[o] * (double)a[o];
        }
      }
      const double rg = den > 0.0 ? std::sqrt(num / den) : 0.0;
      worst_g = std::max(worst_g, rg);
      best_g = std::min(best_g, rg);
    }
    std::printf("         channel groups: best %.3e worst %.3e\n",
                best_g, worst_g);
    check(worst_g < 4.0 * std::max(best_g, 1e-9),
          "and FLAT across the 32 channel groups (worst/best " +
          std::to_string(worst_g / std::max(best_g, 1e-9)) +
          "x) -- a GroupNorm mistake would isolate to one group");
  }
  // bf16 through ~17 convolutions and 10 GroupNorms. A layout mistake
  // lands at O(1), not here.
  check(d < 6e-2, "the GPU path matches the reference");
  }
}

}  // namespace

int
main()
{
  const char* root = std::getenv("VPIPE_LTX25_TEST_MODEL_PATH");
  if (root == nullptr) {
    std::printf("SKIPPED: needs VPIPE_LTX25_TEST_MODEL_PATH. NOTHING was "
                "checked.\n");
    return 0;
  }
  MetalCompute mc(nullptr);
  if (!mc.valid()) {
    std::printf("SKIPPED: no Metal device. NOTHING was checked.\n");
    return 0;
  }
  mc.register_metal_library(ltx25::kMetalLibBf16, ltx25_kernels_bf16_metallib,
                            ltx25_kernels_bf16_metallib_len);
  ltx25::MetalOps ops;
  std::string err;
  if (!ops.init(&mc, &err)) {
    std::printf("FAILED ops init: %s\n", err.c_str());
    return 1;
  }
  run_one(root, mc, ops, "spatial",
          "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors");
  run_one(root, mc, ops, "temporal",
          "ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors");
  if (g_ran == 0) { std::printf("nothing ran\n"); return 0; }
  std::printf("\nltx25-upscaler: %d checks, %d failed\n", g_ran, g_fail);
  return g_fail == 0 ? 0 : 1;
}
