// The 48-block stack against a REFERENCE golden.
//
// This is the gap every other test left open. Block 0 is pinned on real
// weights, every adaLN chain is at ~1e-6, patchify is exact and the
// output head is 3.4e-3 -- but the COMPOSITION of blocks 1..47 was
// checked by nothing, and ltx25-dit-stack-test says so in its header and
// settles for finiteness + RMS. A defect anywhere past block 0 passed
// the entire suite.
//
// The reason there was no golden is recorded there too: a reference
// forward "means holding 39 GB of bf16 in PyTorch". It does not. The
// blocks apply in sequence and nothing needs two at once, so both sides
// STREAM -- load block i, run it, free it. This side binds each block
// with stream=true and drops it; peak is one block.
//
// TAPS AT 0, 1, 3, 7, 15, 31, 47, and they matter: a stack that
// diverges gradually (an accumulating bf16 gap) looks nothing like one
// that is fine until block k and wrong after, and only the intermediate
// taps tell those apart. Block 0's tap is also a free harness check --
// it must reproduce the existing blk0 golden, because the generator
// starts the stack from exactly those inputs.

#include "ltx25-block-metal.h"
#include "ltx25-config.h"
#include "ltx25-dit-weights.h"
#include "ltx25-metal-ops.h"
#include "ltx25-rope.h"

#include "generative-models/weight-set.h"
#include "npy.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

extern "C" const unsigned char ltx25_kernels_bf16_metallib[];
extern "C" const unsigned long ltx25_kernels_bf16_metallib_len;

namespace {

int g_fail = 0;
std::string g_dir;

void
check(bool ok, const std::string& what)
{
  std::printf("  [%s] %s\n", ok ? " OK " : "FAIL", what.c_str());
  if (!ok) { ++g_fail; }
}

npy::Array
g(const std::string& n)
{
  return npy::load(g_dir + "/" + n + ".npy");
}

std::string
env_(const char* k)
{
  const char* v = std::getenv(k);
  return (v != nullptr) ? std::string(v) : std::string();
}

constexpr int kF = 2, kH = 3, kW = 4;
constexpr int kTV = kF * kH * kW, kTA = 8, kTT = 7;
// The taps the generator wrote.
constexpr int kTaps[] = {0, 1, 3, 7, 15, 31, 47};

}  // namespace

int
main()
{
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const std::string root = env_("VPIPE_LTX25_TEST_MODEL_PATH");
  g_dir = env_("VPIPE_LTX25_GOLDENS");
  if (root.empty() || g_dir.empty()) {
    std::printf("SKIPPED: needs VPIPE_LTX25_TEST_MODEL_PATH and "
                "VPIPE_LTX25_GOLDENS. NOTHING was checked.\n");
    return 0;
  }
  if (!g("stack_vx_in").ok) {
    std::printf("SKIPPED: no stack goldens (run gen_goldens.py stack -- it "
                "streams the 38 GB DiT and takes a few minutes). NOTHING "
                "was checked.\n");
    return 0;
  }
  vpipe::metal_compute::MetalCompute mc(nullptr);
  if (!mc.valid()) {
    std::printf("SKIPPED: no Metal device. NOTHING was checked.\n");
    return 0;
  }
  mc.register_metal_library(ltx25::kMetalLibBf16, ltx25_kernels_bf16_metallib,
                            ltx25_kernels_bf16_metallib_len);

  ltx25::Config cfg;
  std::string err;
  if (!ltx25::resolve(root, cfg, &err)) {
    check(false, "resolve: " + err);
    std::printf("FAILURES\n");
    return 1;
  }
  auto ws = vpipe::genai::open_weight_set(cfg.dit_file, nullptr);
  if (!ws) {
    check(false, "could not open " + cfg.dit_file);
    std::printf("FAILURES\n");
    return 1;
  }
  ltx25::MetalOps ops;
  if (!ops.init(&mc, &err)) {
    check(false, "MetalOps::init: " + err);
    std::printf("FAILURES\n");
    return 1;
  }

  const auto& d = cfg.dit;
  const int n_layers = d.num_layers;
  check(n_layers == 48, "48 layers");

  // The same tables the generator used.
  const ltx25::RopeTable vpe = ltx25::build_video_rope(
      kF, kH, kW, d.positional_embedding_max_pos, d.inner_dim(),
      d.num_attention_heads, ltx25::RopeGeometry::identity(),
      d.positional_embedding_theta, d.rope_f64());
  const ltx25::RopeTable ape = ltx25::build_index_rope(
      kTA, d.audio_positional_embedding_max_pos, d.audio_inner_dim(),
      d.audio_num_attention_heads, d.positional_embedding_theta,
      d.rope_f64());
  std::vector<double> vt;
  for (int f = 0; f < kF; ++f) {
    for (int i = 0; i < kH * kW; ++i) { vt.push_back(f); }
  }
  const int cross_max = d.positional_embedding_max_pos.empty()
                            ? 20 : d.positional_embedding_max_pos[0];
  const ltx25::RopeTable vcpe = ltx25::build_cross_rope(
      vt, cross_max, d.audio_inner_dim(), d.audio_num_attention_heads,
      d.positional_embedding_theta, d.rope_f64());

  // The stack's state. forward() writes IN PLACE, so chaining is just
  // handing the same two buffers to every block.
  auto vx_b   = ops.upload_bf16(g("stack_vx_in").data);
  auto ax_b   = ops.upload_bf16(g("stack_ax_in").data);
  auto vctx_b = ops.upload_bf16(g("blk0_vctx").data);
  auto vts_b  = ops.upload_bf16(g("blk0_v_timesteps").data);
  auto vcss_b = ops.upload_bf16(g("blk0_v_cross_ss").data);
  auto vcg_b  = ops.upload_bf16(g("blk0_v_cross_gate").data);
  auto actx_b = ops.upload_bf16(g("blk0_actx").data);
  auto ats_b  = ops.upload_bf16(g("blk0_a_timesteps").data);
  auto acss_b = ops.upload_bf16(g("blk0_a_cross_ss").data);
  auto acg_b  = ops.upload_bf16(g("blk0_a_cross_gate").data);
  auto vpts_b = ops.upload_bf16(g("blk0_v_prompt_ts").data);
  auto apts_b = ops.upload_bf16(g("blk0_a_prompt_ts").data);

  ltx25::GpuStreamInput gv, ga;
  gv.x = &vx_b; gv.context = &vctx_b; gv.timesteps = &vts_b;
  gv.cross_scale_shift = &vcss_b; gv.cross_gate = &vcg_b;
  gv.prompt_timestep = &vpts_b;
  gv.tokens = kTV; gv.text_tokens = kTT;
  ga.x = &ax_b; ga.context = &actx_b; ga.timesteps = &ats_b;
  ga.cross_scale_shift = &acss_b; ga.cross_gate = &acg_b;
  ga.prompt_timestep = &apts_b;
  ga.tokens = kTA; ga.text_tokens = kTT;

  // The bar. Block 0 alone sits at 6.7e-3 against this reference, and a
  // 48-deep bf16 chain accumulates on top of that -- so a growing number
  // is expected and only its SHAPE is diagnostic. Anything O(1) is a
  // wrong answer, not rounding.
  const double kBar = 2.5e-1;
  auto cmp = [&](const std::string& what, const std::vector<float>& got,
                 const std::string& golden) -> double {
    npy::Array w = g(golden);
    if (!w.ok) { check(false, golden + ": " + w.err); return -1.0; }
    if (got.size() != w.data.size()) {
      check(false, golden + ": " + std::to_string(got.size()) + " vs " +
                       std::to_string(w.data.size()));
      return -1.0;
    }
    return npy::rel_l2(got, w.data);
  };

  const auto t0 = std::chrono::steady_clock::now();
  int tap_i = 0;
  bool localized = false;
  for (int i = 0; i < n_layers; ++i) {
    ltx25::GpuBlockWeights gw;
    // stream=true: bind, run, drop. Holding 48 of these is the 38 GB the
    // reference side avoids the same way.
    if (!ltx25::bind_block(*ws, &mc, d, i, /*stream=*/true, gw, &err)) {
      check(false, "bind block " + std::to_string(i) + ": " + err);
      std::printf("FAILURES\n");
      return 1;
    }
    auto blk = ltx25::MetalBlock::create(ops, std::move(gw), &err);
    if (!blk || !blk->reserve(kTV, kTA, kTT, &err)) {
      check(false, "block " + std::to_string(i) + ": " + err);
      std::printf("FAILURES\n");
      return 1;
    }
    blk->set_rope(&vpe, &ape, &vcpe, &ape);
    auto stream = mc.make_command_stream();
    {
      auto enc = stream.begin_compute();
      if (!blk->forward(enc, gv, ga, &err)) {
        check(false, "forward block " + std::to_string(i) + ": " + err);
        std::printf("FAILURES\n");
        return 1;
      }
    }
    stream.commit().wait();
    blk.reset();                       // the streaming part

    if (tap_i < (int)(sizeof(kTaps) / sizeof(kTaps[0])) &&
        kTaps[tap_i] == i) {
      char tag[32];
      std::snprintf(tag, sizeof(tag), "%02d", i);
      const std::vector<float> vout = ltx25::MetalOps::download_bf16(
          vx_b, (std::size_t)kTV * d.inner_dim());
      const std::vector<float> aout = ltx25::MetalOps::download_bf16(
          ax_b, (std::size_t)kTA * d.audio_inner_dim());
      const double rv = cmp("video", vout, std::string("stack_vx_b") + tag);
      const double ra = cmp("audio", aout, std::string("stack_ax_b") + tag);
      std::printf("       block %-2d  video %.3e   audio %.3e\n", i, rv, ra);
      // The FIRST tap that fails is the one that matters; the rest are
      // consequences, so stop reporting after it.
      if (!localized && (rv >= kBar || ra >= kBar || rv < 0.0 || ra < 0.0)) {
        check(false, "the stack diverges by block " + std::to_string(i));
        localized = true;
      }
      ++tap_i;
    }
  }
  const double secs = std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - t0).count();
  std::printf("       48 blocks streamed in %.1f s\n", secs);
  if (!localized) {
    check(true, "the whole 48-block stack tracks the reference");
  }

  std::printf("%s\n", g_fail == 0 ? "ALL PASSED" : "FAILURES");
  return g_fail == 0 ? 0 : 1;
}
