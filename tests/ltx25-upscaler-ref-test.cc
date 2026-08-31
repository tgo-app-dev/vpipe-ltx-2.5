// The latent upscalers' CPU reference, against the released checkpoints.
//
// TAPS FIRST, then the whole model. A whole-forward mismatch says
// nothing about where it started, and this model has five places where a
// wrong answer is correctly shaped: GroupNorm's grouping, the residual
// joining before the SiLU rather than after, the pixel-shuffle nesting,
// the spatial model's per-frame Conv2d, and the temporal model's dropped
// first frame.
//
// The bar is f32 against f32 -- both sides run the same arithmetic in
// the same precision, so anything above accumulation noise is a
// transcription bug, not a dtype.
//
// Needs VPIPE_LTX25_TEST_MODEL_PATH (for the checkpoints) and
// VPIPE_LTX25_GOLDENS. SKIPS and says so without them.

#include "ltx25-upscaler-ref.h"
#include "npy.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "generative-models/weight-set.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using vpipe::genai::WeightSet;
using vpipe::metal_compute::MetalCompute;

namespace {

int g_fail = 0;
int g_ran = 0;

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

// f32 against f32 over ~17 convolutions; this is accumulation order,
// not precision.
constexpr double kBar = 2e-5;

void
run_one(const std::string& root, const std::string& gdir,
        const std::string& kind, const std::string& file)
{
  std::printf("\n%s upscaler\n", kind.c_str());
  const std::string path = root + "/latent_upscale_models/" + file;
  ltx25::UpscalerConfig cfg;
  std::string err;
  if (!ltx25::UpscalerConfig::from_metadata(path, &cfg, &err)) {
    check(false, "read the config from __metadata__: " + err);
    return;
  }
  std::printf("       in %d mid %d blocks %d dims %d  spatial %d temporal %d "
              "scale %.1f rational %d\n",
              cfg.in_channels, cfg.mid_channels, cfg.num_blocks_per_stage,
              cfg.dims, (int)cfg.spatial_upsample, (int)cfg.temporal_upsample,
              cfg.spatial_scale, (int)cfg.rational_resampler);
  // The two values a default would have got wrong.
  if (kind == "spatial") {
    check(cfg.mid_channels == 1024,
          "mid_channels is 1024, NOT the module's 512 default (got " +
          std::to_string(cfg.mid_channels) + ")");
  } else {
    check(cfg.rational_resampler,
          "the temporal checkpoint really does set rational_resampler -- "
          "and it is IGNORED, because the module reads it only on the "
          "spatial branch");
    check(!cfg.spatial_upsample && cfg.temporal_upsample,
          "temporal only, which is what sends it down the plain "
          "Sequential(Conv3d, PixelShuffle) branch");
  }

  MetalCompute mc(nullptr);
  auto ws = WeightSet::open(path, nullptr);
  if (!ws) { check(false, "open " + path); return; }
  auto m = ltx25::Ltx25UpscalerRef::load(cfg, *ws, &mc, &err);
  if (!m) { check(false, "load: " + err); return; }
  check(true, "loaded");
  m->set_capture(true);

  npy::Array in = npy::load(gdir + "/ups_" + kind + "_in.npy");
  if (!in.ok) { check(false, "load the input golden: " + in.err); return; }
  const int F = (int)in.shape[1], H = (int)in.shape[2], W = (int)in.shape[3];

  std::vector<float> out;
  std::array<int, 4> shape{};
  if (!m->forward(in.data.data(), F, H, W, &out, &shape, &err)) {
    check(false, "forward: " + err);
    return;
  }

  // The shape rule, stated rather than inherited from the golden.
  check(shape[0] == cfg.in_channels && shape[1] == cfg.out_frames(F)
            && shape[2] == cfg.out_height(H) && shape[3] == cfg.out_width(W),
        "output is [" + std::to_string(shape[0]) + "," +
        std::to_string(shape[1]) + "," + std::to_string(shape[2]) + "," +
        std::to_string(shape[3]) + "] from a [" + std::to_string(F) + "," +
        std::to_string(H) + "," + std::to_string(W) + "] latent");

  // TAPS, in order. The first one to move is the one that broke.
  const char* taps[] = {"initial", "res", "shuffled", "upsampled", "post"};
  for (const char* t : taps) {
    npy::Array g = npy::load(gdir + "/ups_" + kind + "_" + t + ".npy");
    if (!g.ok) { continue; }         // temporal-only taps
    const std::vector<float>* got = m->tap(t);
    if (got == nullptr) {
      check(false, std::string(t) + ": not captured");
      continue;
    }
    const double r = rel_l2(*got, g.data);
    std::printf("       %-10s rel-L2 %.3e\n", t, r);
    check(r < kBar, std::string(t) + " matches");
  }

  npy::Array want = npy::load(gdir + "/ups_" + kind + "_out.npy");
  if (!want.ok) { check(false, "load the output golden: " + want.err); return; }
  const double r = rel_l2(out, want.data);
  std::printf("       %-10s rel-L2 %.3e\n", "out", r);
  check(r < kBar, "the whole forward matches");
}

}  // namespace

int
main()
{
  const char* root = std::getenv("VPIPE_LTX25_TEST_MODEL_PATH");
  const char* gdir = std::getenv("VPIPE_LTX25_GOLDENS");
  if (root == nullptr || gdir == nullptr) {
    std::printf("SKIPPED: needs VPIPE_LTX25_TEST_MODEL_PATH and "
                "VPIPE_LTX25_GOLDENS. NOTHING was checked.\n");
    return 0;
  }
  run_one(root, gdir, "spatial",
          "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors");
  run_one(root, gdir, "temporal",
          "ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors");
  if (g_ran == 0) {
    std::printf("ltx25-upscaler-ref: nothing ran\n");
    return 0;
  }
  std::printf("\nltx25-upscaler-ref: %d checks, %d failed\n", g_ran, g_fail);
  return g_fail == 0 ? 0 : 1;
}
