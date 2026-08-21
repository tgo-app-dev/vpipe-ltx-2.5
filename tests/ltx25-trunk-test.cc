// The DiT trunk on the real checkpoint: the adaLN chains, patchify, and
// the output head.
//
// These are exactly the pieces the 48-block loop COMPOSES and that no
// other test touches. Each has a quiet failure mode:
//
//   the sinusoid    flip_sin_to_cos is TRUE, so the row is [cos | sin].
//                   The other order is a clean forward with every
//                   modulation driven by the wrong half.
//   the chain       THREE linears (emb.linear_1, emb.linear_2, then
//                   `linear` at the adaLN root), and `embedded_timestep`
//                   is the SECOND one's output -- not the third's.
//   the head        LayerNorm, not RMSNorm; and the same dim-wide
//                   embedding is added to BOTH rows of the 2-row table.
//   patchify        channel-major [z][F][H][W] -> [T][z] with w fastest.

#include "ltx25-config.h"
#include "ltx25-dit.h"
#include "ltx25-metal-ops.h"
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

std::string
tag(const std::string& name, double sig)
{
  std::string s = (sig == 1.0) ? "1_0" : "0_421875";
  return "trunk_" + name + "_s" + s;
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
  g_dir = gdir;
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

  // Only the TRUNK is bound here -- 48 blocks would be 39 GB and none of
  // them is under test.
  ltx25::DitTrunk trunk;
  if (!ltx25::bind_trunk(*ws, &mc, cfg.dit, trunk, &err)) {
    check(false, "bind_trunk: " + err);
    std::printf("FAILURES\n");
    return 1;
  }
  check(true, "trunk bound (6 adaLN MLPs, 2 patchify, 2 heads)");

  // The adaLN chain runs on the host, so this needs no Ltx25Dit -- but
  // it must be the SAME code the DiT uses, hence adaln_public.
  ltx25::Config small = cfg;
  small.dit.num_layers = 0;      // no blocks: only the trunk is wanted
  auto dit = ltx25::Ltx25Dit::load(small, ws, ops, /*stream_blocks=*/false,
                                   0, 0, 0, &err);
  if (!dit) { std::printf("FAILED to build a 0-block DiT: %s\n", err.c_str()); return 1; }

  struct { const char* name; const ltx25::DitTrunk::AdaLN* a; } kAda[] = {
      {"adaln_single", &trunk.video},
      {"audio_adaln_single", &trunk.audio},
      {"prompt_adaln_single", &trunk.prompt},
      {"audio_prompt_adaln_single", &trunk.audio_prompt},
      {"av_ca_video_scale_shift_adaln_single", &trunk.av_video_ss},
      {"av_ca_audio_scale_shift_adaln_single", &trunk.av_audio_ss},
      {"av_ca_a2v_gate_adaln_single", &trunk.av_a2v_gate},
      {"av_ca_v2a_gate_adaln_single", &trunk.av_v2a_gate},
  };
  // The chain runs in f32 on the host over bf16 WEIGHTS, so the floor is
  // the weights' own precision, not f32's.
  const double kBar = 4e-3;
  for (const auto& e : kAda) {
    for (double sig : {1.0, 0.421875}) {
      npy::Array wd = g(tag(e.name, sig) + "_drv");
      npy::Array we = g(tag(e.name, sig) + "_emb");
      if (!wd.ok || !we.ok) { check(false, std::string(e.name) + ": no golden"); continue; }
      std::vector<float> emb;
      // sigma * timestep_scale_multiplier, as the DiT feeds it.
      const std::vector<float> drv =
          dit->adaln_public(*e.a, sig * cfg.dit.timestep_scale_multiplier,
                            &emb);
      const double rd = npy::rel_l2(drv, wd.data);
      const double re = npy::rel_l2(emb, we.data);
      std::printf("       %-42s s=%-9g drv %.2e  emb %.2e\n", e.name, sig,
                  rd, re);
      check(rd < kBar && re < kBar, std::string(e.name) + " @ sigma " +
            std::to_string(sig));
    }
  }

  // ---- patchify: the token packing, then the projection --------------
  {
    npy::Array lat = g("trunk_patch_latent");     // [1,128,2,3,4]
    npy::Array tok = g("trunk_patch_tokens");     // [1,24,128]
    npy::Array out = g("trunk_patch_out");        // [1,24,4096]
    // Check the LOAD. A golden that failed to parse compares as a size
    // mismatch, which reads as a huge model error and sends the search
    // in entirely the wrong direction.
    if (!lat.ok || !tok.ok || !out.ok) {
      check(false, "patchify goldens: " + lat.err + tok.err + out.err);
      std::printf("%s\n", "FAILURES");
      return 1;
    }
    const int z = 128, T = 24;
    // The pack the DiT does inline: channel-major -> token-major.
    std::vector<float> packed((std::size_t)T * z);
    for (int t = 0; t < T; ++t) {
      for (int c = 0; c < z; ++c) {
        packed[(std::size_t)t * z + c] = lat.data[(std::size_t)c * T + t];
      }
    }
    const double rp = npy::rel_l2(packed, tok.data);
    std::printf("       patchify pack   %.3e  (packed %zu, golden %zu)\n",
                rp, packed.size(), tok.data.size());
    check(rp < 1e-6,
          "the 1x1x1 patchify packs channel-major -> [T][z], w fastest");

    auto in_b = ops.upload_bf16(packed);
    auto out_b = ops.alloc((std::size_t)T * cfg.dit.inner_dim());
    auto st = mc.make_command_stream();
    {
      auto enc = st.begin_compute();
      ops.linear(enc, in_b, trunk.patchify_w, &trunk.patchify_b, out_b, T, z,
                 cfg.dit.inner_dim());
    }
    st.commit().wait();
    const std::vector<float> got = ltx25::MetalOps::download_bf16(
        out_b, (std::size_t)T * cfg.dit.inner_dim());
    const double r = npy::rel_l2(got, out.data);
    std::printf("       patchify_proj                              %.2e\n", r);
    check(r < 2e-2, "patchify_proj matches (bf16 GEMM over K=128)");
  }

  // ---- the output head ------------------------------------------------
  {
    npy::Array x = g("trunk_head_x");       // [1,6,4096]
    npy::Array e = g("trunk_head_emb");     // [1,1,4096]
    npy::Array w = g("trunk_head_out");     // [1,6,128]
    if (!x.ok || !e.ok || !w.ok) {
      check(false, "head goldens: " + x.err + e.err + w.err);
      std::printf("FAILURES\n");
      return 1;
    }
    const int vd = cfg.dit.inner_dim(), T = 6, zc = cfg.dit.out_channels;

    // shift = table[0] + emb, scale = table[1] + emb -- the SAME
    // embedding added to both rows.
    std::vector<float> ss((std::size_t)2 * vd);
    {
      const auto* tab = static_cast<const std::uint16_t*>(
          trunk.scale_shift_out.contents());
      for (int r = 0; r < 2; ++r) {
        for (int i = 0; i < vd; ++i) {
          const std::size_t k = (std::size_t)r * vd + i;
          std::uint32_t u = (std::uint32_t)tab[k] << 16;
          float tv;
          std::memcpy(&tv, &u, 4);
          ss[k] = tv + e.data[(std::size_t)i];
        }
      }
    }
    auto x_b  = ops.upload_bf16(x.data);
    auto ss_b = ops.upload_bf16(ss);
    auto tmp  = ops.alloc((std::size_t)T * vd);
    auto out_b = ops.alloc((std::size_t)T * zc);
    auto st = mc.make_command_stream();
    {
      auto enc = st.begin_compute();
      ops.layer_norm_plain(enc, x_b, tmp, vd, T);
      ops.modulate_off(enc, tmp, ss_b, (std::size_t)vd * 2, ss_b, 0, tmp, vd,
                       T);
      ops.linear(enc, tmp, trunk.proj_out_w, &trunk.proj_out_b, out_b, T, vd,
                 zc);
    }
    st.commit().wait();
    const std::vector<float> got =
        ltx25::MetalOps::download_bf16(out_b, (std::size_t)T * zc);
    const double r = npy::rel_l2(got, w.data);
    std::printf("       output head (LayerNorm + 2-row ss + proj)  %.2e\n", r);
    check(r < 4e-2, "the output head matches");
  }

  std::printf("%s\n", g_fail == 0 ? "ALL PASSED" : "FAILURES");
  return g_fail == 0 ? 0 : 1;
}
