// The Metal block on the REAL checkpoint's layer 0, at real dimensions.
//
// Everything before this ran on a dim-64 toy. This binds block 0 of the
// released 22B DiT straight off the 39 GB safetensors -- 4096-wide video
// and 2048-wide audio, 32 heads each, head_dim 128 and 64 -- and runs it
// against the reference's own output for the same inputs.
//
// It is the first check that exercises:
//   * the WEIGHT NAMES as the checkpoint actually spells them,
//   * the asymmetric cross-attention (a 4096 query projected to the
//     audio head size), which the toy config could only approximate,
//   * the video feed-forward having NO bias where the audio one does,
//   * the real 3-axis video RoPE against the 1-axis audio one.
//
// Gated on VPIPE_LTX25_TEST_MODEL_PATH + VPIPE_LTX25_GOLDENS; skips
// loudly without them.

#include "ltx25-block-metal.h"
#include "ltx25-config.h"
#include "ltx25-dit-weights.h"
#include "ltx25-metal-ops.h"
#include "ltx25-rope.h"
#include "npy.h"

#include "generative-models/weight-set.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

extern "C" const unsigned char ltx25_kernels_bf16_metallib[];
extern "C" const unsigned long ltx25_kernels_bf16_metallib_len;

using vpipe::genai::WeightSet;
using vpipe::metal_compute::MetalCompute;

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

// gen_goldens.py's `block` geometry.
constexpr int kF = 2, kH = 3, kW = 4;
constexpr int kTV = kF * kH * kW, kTA = 8, kTT = 7;

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
  g_dir = gdir;

  MetalCompute mc(nullptr);
  if (!mc.valid()) {
    std::printf("SKIPPED: no usable Metal device. NOTHING was checked.\n");
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
  std::printf("checkpoint: %s DiT, %d layers, video %d / audio %d\n",
              ltx25::variant_name(cfg.variant), cfg.dit.num_layers,
              cfg.dit.inner_dim(), cfg.dit.audio_inner_dim());

  // The DiT is ONE .safetensors file; WeightSet::open takes the path.
  std::shared_ptr<WeightSet> ws = WeightSet::open(cfg.dit_file, nullptr);
  if (!ws) {
    std::printf("SKIPPED: could not open %s\n", cfg.dit_file.c_str());
    return 0;
  }

  ltx25::MetalOps ops;
  if (!ops.init(&mc, &err)) {
    std::printf("FAILED to init ops: %s\n", err.c_str());
    return 1;
  }

  // Bind layer 0. `stream=false` here: this is one block, and the
  // streaming path is about not RETAINING 48 of them.
  ltx25::GpuBlockWeights gw;
  if (!ltx25::bind_block(*ws, &mc, cfg.dit, 0, /*stream=*/false, gw, &err)) {
    check(false, "bind block 0: " + err);
    std::printf("FAILURES\n");
    return 1;
  }
  check(true, "block 0 bound straight off the checkpoint (bf16, no copy)");

  auto blk = ltx25::MetalBlock::create(ops, std::move(gw), &err);
  if (!blk || !blk->reserve(kTV, kTA, kTT, &err)) {
    std::printf("FAILED: %s\n", err.c_str());
    return 1;
  }

  // The same tables gen_goldens.py used: 3-axis video self, 1-axis audio
  // self, and the TIME-ONLY cross tables at the AUDIO width for both.
  const auto& d = cfg.dit;
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
  blk->set_rope(&vpe, &ape, &vcpe, &ape);

  // GpuStreamInput BORROWS its buffers, so they must outlive the
  // forward -- named locals here rather than temporaries.
  auto vx_b   = ops.upload_bf16(g("blk0_vx_in").data);
  auto vctx_b = ops.upload_bf16(g("blk0_vctx").data);
  auto vts_b  = ops.upload_bf16(g("blk0_v_timesteps").data);
  auto vcss_b = ops.upload_bf16(g("blk0_v_cross_ss").data);
  auto vcg_b  = ops.upload_bf16(g("blk0_v_cross_gate").data);
  auto ax_b   = ops.upload_bf16(g("blk0_ax_in").data);
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

  auto stream = mc.make_command_stream();
  {
    auto enc = stream.begin_compute();
    if (!blk->forward(enc, gv, ga, &err)) {
      std::printf("FAILED forward: %s\n", err.c_str());
      return 1;
    }
  }
  stream.commit().wait();

  const std::vector<float> vout =
      ltx25::MetalOps::download_bf16(vx_b, (std::size_t)kTV * d.inner_dim());
  const std::vector<float> aout =
      ltx25::MetalOps::download_bf16(ax_b,
                                     (std::size_t)kTA * d.audio_inner_dim());

  // Wider than the toy's bar: 4096-wide GEMMs accumulate over 4096 terms
  // in bf16 where the toy accumulated over 64.
  const double kBar = 6e-2;
  auto cmp = [&](const char* what, const std::vector<float>& got,
                 const char* golden) {
    npy::Array w = g(golden);
    if (!w.ok) { check(false, std::string(golden) + ": " + w.err); return; }
    const double r = npy::rel_l2(got, w.data);
    std::printf("       %-9s rel-L2 %.3e (bar %.0e)\n", what, r, kBar);
    check(r < kBar, std::string(what) + " matches the reference");
  };
  cmp("video", vout, "blk0_vx_out");
  cmp("audio", aout, "blk0_ax_out");

  std::printf("%s\n", g_fail == 0 ? "ALL PASSED" : "FAILURES");
  return g_fail == 0 ? 0 : 1;
}
