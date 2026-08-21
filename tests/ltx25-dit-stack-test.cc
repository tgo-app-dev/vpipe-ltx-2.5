// The whole 48-block stack, once, on the real checkpoint.
//
// There is no golden for this and there deliberately is not one: a
// reference forward of the full DiT means holding 39 GB of bf16 in
// PyTorch, and every PIECE it composes is already pinned --
//
//   the block          6.7e-3 / 3.9e-3 on real layer-0 weights
//   the adaLN chains   ~1e-6, all eight, at two sigmas
//   patchify           exact
//   the output head    3.4e-3
//
// -- so what is left to establish is that the composition RUNS and does
// not produce garbage. That is what this checks, and it is also the
// first time 39 GB is bound at once, so it doubles as the memory test.
//
// The bar is deliberately about SHAPE OF THE ANSWER rather than a
// number: every value finite (a mis-wired adaLN driver NaNs, which is
// how the F32 scale/shift tables were caught), and a velocity whose
// scale is within an order of magnitude of the latent it was asked
// about. A quiet wrong answer inside those bounds is what the per-piece
// goldens above are for.

#include "ltx25-config.h"
#include "ltx25-dit.h"
#include "ltx25-metal-ops.h"

#include "generative-models/weight-set.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

extern "C" const unsigned char ltx25_kernels_bf16_metallib[];
extern "C" const unsigned long ltx25_kernels_bf16_metallib_len;

using vpipe::genai::WeightSet;
using vpipe::metal_compute::MetalCompute;

namespace {

int g_fail = 0;

void
check(bool ok, const std::string& what)
{
  std::printf("  [%s] %s\n", ok ? " OK " : "FAIL", what.c_str());
  if (!ok) { ++g_fail; }
}

double
rms(const std::vector<float>& v)
{
  if (v.empty()) { return 0.0; }
  double s = 0.0;
  for (float f : v) { s += (double)f * f; }
  return std::sqrt(s / (double)v.size());
}

}  // namespace

// The REAL coordinates -- pixels and seconds -- because this drives the
// actual forward. Using identity() here would exercise a path no
// generation takes.
const ltx25::RopeGeometry kGeo{ltx25::kTemporalCompression,
                               ltx25::kSpatialCompression,
                               ltx25::kSpatialCompression, 24.0, true};

int
main()
{
  const char* root = std::getenv("VPIPE_LTX25_TEST_MODEL_PATH");
  if (root == nullptr) {
    std::printf("SKIPPED: needs VPIPE_LTX25_TEST_MODEL_PATH. "
                "NOTHING was checked.\n");
    return 0;
  }
  MetalCompute mc(nullptr);
  if (!mc.valid()) {
    std::printf("SKIPPED: no Metal device. NOTHING was checked.\n");
    return 0;
  }
  mc.register_metal_library(ltx25::kMetalLibBf16, ltx25_kernels_bf16_metallib,
                            ltx25_kernels_bf16_metallib_len);

  ltx25::Config cfg;
  std::string err;
  if (!ltx25::resolve(root, cfg, &err, {})) {
    std::printf("SKIPPED: %s\n", err.c_str());
    return 0;
  }
  std::shared_ptr<WeightSet> ws = WeightSet::open(cfg.dit_file, nullptr);
  if (!ws) { std::printf("SKIPPED: cannot open the DiT\n"); return 0; }

  ltx25::MetalOps ops;
  if (!ops.init(&mc, &err)) {
    std::printf("FAILED ops: %s\n", err.c_str());
    return 1;
  }

  // A SMALL geometry on purpose: 2 latent frames x 4 x 6 = 48 video
  // tokens. The point is the depth (48 blocks, 39 GB of weights), not
  // the sequence -- a real 121-frame clip at 960x544 is 8160 tokens and
  // minutes per step, which is a perf question and not this one.
  const int F = 2, LH = 4, LW = 6;
  const int A_TOK = 16, T_TOK = 8;

  auto t0 = std::chrono::steady_clock::now();
  auto dit = ltx25::Ltx25Dit::load(cfg, ws, ops, /*stream_blocks=*/false,
                                   0, 0, 0, &err);
  if (!dit) {
    check(false, "load: " + err);
    std::printf("FAILURES\n");
    return 1;
  }
  auto t1 = std::chrono::steady_clock::now();
  const double load_s =
      std::chrono::duration<double>(t1 - t0).count();
  std::printf("  loaded %d blocks + trunk in %.1f s\n", dit->num_layers(),
              load_s);
  check(dit->num_layers() == cfg.dit.num_layers,
        "all " + std::to_string(cfg.dit.num_layers) + " blocks bound");
  std::printf("  connector weights in this checkpoint: %s "
              "(not implemented; context is taken already connected)\n",
              dit->connector_weights_present() ? "yes" : "no");

  if (!dit->set_geometry(F, LH, LW, A_TOK, T_TOK, kGeo, &err)) {
    check(false, "set_geometry: " + err);
    std::printf("FAILURES\n");
    return 1;
  }
  check(true, "geometry set (RoPE tables + per-block scratch)");

  // A unit-variance latent, as the sampler's first step holds.
  std::mt19937 rng(5);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  const int zc = cfg.dit.in_channels;
  std::vector<float> vlat((std::size_t)zc * F * LH * LW);
  std::vector<float> alat((std::size_t)zc * A_TOK);
  for (auto& v : vlat) { v = nd(rng); }
  for (auto& v : alat) { v = nd(rng); }
  // The context, as the connector would have produced it.
  std::vector<float> vctx((std::size_t)T_TOK * cfg.dit.cross_attention_dim);
  std::vector<float> actx((std::size_t)T_TOK *
                          cfg.dit.audio_cross_attention_dim);
  for (auto& v : vctx) { v = nd(rng) * 0.5f; }
  for (auto& v : actx) { v = nd(rng) * 0.5f; }
  auto vctx_b = ops.upload_bf16(vctx);
  auto actx_b = ops.upload_bf16(actx);

  ltx25::Ltx25Dit::Input in;
  in.video = vlat.data();
  in.audio = alat.data();
  in.context = vctx_b.contents();
  in.audio_context = actx_b.contents();
  in.sigma = 1.0;          // the distilled schedule's first step
  in.audio_sigma = 1.0;

  ltx25::Ltx25Dit::Output out;
  auto t2 = std::chrono::steady_clock::now();
  if (!dit->forward(in, &out, &err)) {
    check(false, "forward: " + err);
    std::printf("FAILURES\n");
    return 1;
  }
  auto t3 = std::chrono::steady_clock::now();
  const double step_s = std::chrono::duration<double>(t3 - t2).count();
  std::printf("  one 48-block forward at %d video / %d audio tokens: "
              "%.2f s\n", F * LH * LW, A_TOK, step_s);

  check(out.video.size() == (std::size_t)cfg.dit.out_channels * F * LH * LW,
        "video velocity has the latent's shape");
  check(out.audio.size() == (std::size_t)cfg.dit.audio_out_channels * A_TOK,
        "audio velocity has the latent's shape");

  std::size_t bad_v = 0, bad_a = 0;
  for (float f : out.video) { if (!std::isfinite(f)) { ++bad_v; } }
  for (float f : out.audio) { if (!std::isfinite(f)) { ++bad_a; } }
  check(bad_v == 0, "every video value is finite (" +
        std::to_string(bad_v) + " bad)");
  check(bad_a == 0, "every audio value is finite (" +
        std::to_string(bad_a) + " bad)");

  const double rv = rms(out.video), ra = rms(out.audio), rl = rms(vlat);
  std::printf("  rms: latent %.3f -> video velocity %.3f, audio %.3f\n",
              rl, rv, ra);
  // A velocity that has collapsed to zero or blown up by orders of
  // magnitude is the signature of a driver wired to the wrong place --
  // the model runs, and what comes out means nothing.
  check(rv > 0.02 * rl && rv < 50.0 * rl,
        "the video velocity is within an order of magnitude of the latent");
  check(ra > 0.02 * rl && ra < 50.0 * rl,
        "the audio velocity is within an order of magnitude of the latent");

  // Two different sigmas must give DIFFERENT velocities: if they do not,
  // the timestep is not reaching the blocks at all -- which is exactly
  // what an adaLN driver built once and never re-read looks like.
  ltx25::Ltx25Dit::Output out2;
  in.sigma = 0.421875;
  in.audio_sigma = 0.421875;
  if (!dit->forward(in, &out2, &err)) {
    check(false, "second forward: " + err);
  } else {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < out.video.size(); ++i) {
      const double d = (double)out2.video[i] - out.video[i];
      num += d * d;
      den += (double)out.video[i] * out.video[i];
    }
    const double rel = den > 0 ? std::sqrt(num / den) : 0.0;
    std::printf("  sigma 1.0 vs 0.421875: relative change %.3f\n", rel);
    check(rel > 0.01, "the timestep actually reaches the blocks");
  }

  // The DiT can also OWN the connectors and take a raw caption. That
  // path adds ~4 GB on top of the 39 GB, so it is a separate load and a
  // separate decision -- see the note in ltx25-dit.h.
  if (std::getenv("VPIPE_LTX25_STACK_CONNECTORS") != nullptr) {
    std::printf("  --- with connectors (raw caption in) ---\n");
    auto dit2 = ltx25::Ltx25Dit::load(cfg, ws, ops, /*stream_blocks=*/false,
                                      0, 0, 0, &err,
                                      /*with_connectors=*/true);
    if (!dit2) {
      check(false, "load with connectors: " + err);
    } else {
      check(dit2->connectors_loaded(), "both connectors loaded");
      // The caption must be a whole number of register tiles.
      const int TT = 256;
      if (!dit2->set_geometry(F, LH, LW, A_TOK, TT, kGeo, &err)) {
        check(false, "set_geometry: " + err);
      } else {
        std::vector<float> rv((std::size_t)TT * cfg.dit.cross_attention_dim);
        std::vector<float> ra_((std::size_t)TT *
                               cfg.dit.audio_cross_attention_dim);
        for (auto& v : rv)  { v = nd(rng) * 0.5f; }
        for (auto& v : ra_) { v = nd(rng) * 0.5f; }
        auto rv_b = ops.upload_bf16(rv);
        auto ra_b = ops.upload_bf16(ra_);
        ltx25::Ltx25Dit::Input in2 = in;
        in2.context = rv_b.contents();
        in2.audio_context = ra_b.contents();
        in2.n_valid_text = 100;      // a padded caption, as the pipeline pads
        in2.sigma = 1.0;
        in2.audio_sigma = 1.0;
        ltx25::Ltx25Dit::Output o2;
        if (!dit2->forward(in2, &o2, &err)) {
          check(false, "forward with connectors: " + err);
        } else {
          std::size_t bad = 0;
          for (float f : o2.video) { if (!std::isfinite(f)) { ++bad; } }
          check(bad == 0, "connector path: every video value is finite");
          std::printf("  rms with connectors: video %.3f, audio %.3f\n",
                      rms(o2.video), rms(o2.audio));
          check(rms(o2.video) > 0.02 * rl && rms(o2.video) < 50.0 * rl,
                "connector path: the velocity is plausible");
        }
      }
    }
  }

  std::printf("%s\n", g_fail == 0 ? "ALL PASSED" : "FAILURES");
  return g_fail == 0 ? 0 : 1;
}
