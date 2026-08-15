// The Metal block against the SAME golden the CPU reference is checked
// on -- and, more usefully, against the CPU reference itself.
//
// Two comparisons, because they fail differently:
//
//   vs the CPU reference   the two run identical math at different
//                          precision, so anything above bf16 round-off
//                          is a TRANSCRIPTION bug (a wrong adaLN row, a
//                          missing snapshot, a swapped scale/shift).
//   vs the golden          catches a bug the CPU reference and the
//                          Metal path would have to share, which after
//                          the reference is pinned at 3e-7 means
//                          essentially nothing -- but it is free.
//
// The bar is bf16: the block is ~40 dependent bf16 ops deep, so error
// accumulates well past a single op's 2e-3.

#include "ltx25-block-metal.h"
#include "ltx25-block-ref.h"
#include "ltx25-config.h"
#include "ltx25-metal-ops.h"
#include "ltx25-rope.h"
#include "npy.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

extern "C" const unsigned char ltx25_kernels_bf16_metallib[];
extern "C" const unsigned long ltx25_kernels_bf16_metallib_len;

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

constexpr int kVD = 64, kVH = 4, kAD = 32, kAH = 4;
constexpr int kTV = 6, kTA = 4, kTT = 3;

npy::Array
g(const std::string& n)
{
  return npy::load(g_dir + "/" + n + ".npy");
}

// Same name list as the CPU-reference test: the two must agree on what a
// weight name means, which is why both go through load_block_weights.
bool
load_weights(ltx25::BlockWeights& w)
{
  static const char* kNames[] = {
    "attn1.to_q.weight","attn1.to_q.bias","attn1.to_k.weight","attn1.to_k.bias",
    "attn1.to_v.weight","attn1.to_v.bias","attn1.to_out.0.weight",
    "attn1.to_out.0.bias","attn1.q_norm.weight","attn1.k_norm.weight",
    "attn1.to_gate_logits.weight","attn1.to_gate_logits.bias",
    "attn2.to_q.weight","attn2.to_q.bias","attn2.to_k.weight","attn2.to_k.bias",
    "attn2.to_v.weight","attn2.to_v.bias","attn2.to_out.0.weight",
    "attn2.to_out.0.bias","attn2.q_norm.weight","attn2.k_norm.weight",
    "attn2.to_gate_logits.weight","attn2.to_gate_logits.bias",
    "ff.net.0.proj.weight","ff.net.2.weight",
    "scale_shift_table","prompt_scale_shift_table",
    "audio_attn1.to_q.weight","audio_attn1.to_q.bias","audio_attn1.to_k.weight",
    "audio_attn1.to_k.bias","audio_attn1.to_v.weight","audio_attn1.to_v.bias",
    "audio_attn1.to_out.0.weight","audio_attn1.to_out.0.bias",
    "audio_attn1.q_norm.weight","audio_attn1.k_norm.weight",
    "audio_attn1.to_gate_logits.weight","audio_attn1.to_gate_logits.bias",
    "audio_attn2.to_q.weight","audio_attn2.to_q.bias","audio_attn2.to_k.weight",
    "audio_attn2.to_k.bias","audio_attn2.to_v.weight","audio_attn2.to_v.bias",
    "audio_attn2.to_out.0.weight","audio_attn2.to_out.0.bias",
    "audio_attn2.q_norm.weight","audio_attn2.k_norm.weight",
    "audio_attn2.to_gate_logits.weight","audio_attn2.to_gate_logits.bias",
    "audio_ff.net.0.proj.weight","audio_ff.net.0.proj.bias",
    "audio_ff.net.2.weight","audio_ff.net.2.bias",
    "audio_scale_shift_table","audio_prompt_scale_shift_table",
    "audio_to_video_attn.to_q.weight","audio_to_video_attn.to_q.bias",
    "audio_to_video_attn.to_k.weight","audio_to_video_attn.to_k.bias",
    "audio_to_video_attn.to_v.weight","audio_to_video_attn.to_v.bias",
    "audio_to_video_attn.to_out.0.weight","audio_to_video_attn.to_out.0.bias",
    "audio_to_video_attn.q_norm.weight","audio_to_video_attn.k_norm.weight",
    "audio_to_video_attn.to_gate_logits.weight",
    "audio_to_video_attn.to_gate_logits.bias",
    "video_to_audio_attn.to_q.weight","video_to_audio_attn.to_q.bias",
    "video_to_audio_attn.to_k.weight","video_to_audio_attn.to_k.bias",
    "video_to_audio_attn.to_v.weight","video_to_audio_attn.to_v.bias",
    "video_to_audio_attn.to_out.0.weight","video_to_audio_attn.to_out.0.bias",
    "video_to_audio_attn.q_norm.weight","video_to_audio_attn.k_norm.weight",
    "video_to_audio_attn.to_gate_logits.weight",
    "video_to_audio_attn.to_gate_logits.bias",
    "scale_shift_table_a2v_ca_audio","scale_shift_table_a2v_ca_video",
  };
  std::unordered_map<std::string, ltx25::NamedTensor> t;
  for (const char* n : kNames) {
    std::string f = "small_w__";
    for (const char* p = n; *p; ++p) { f += (*p == '.') ? '_' : *p; }
    npy::Array a = g(f);
    if (!a.ok) { check(false, "load " + std::string(n) + ": " + a.err); return false; }
    t[n] = ltx25::NamedTensor{a.data, a.shape};
  }
  std::string err;
  if (!ltx25::load_block_weights(t, kVD, kVH, kAD, kAH, w, &err)) {
    check(false, "assemble: " + err);
    return false;
  }
  return true;
}

ltx25::Mat
mat(const npy::Array& a, int rows, int cols)
{
  ltx25::Mat m; m.rows = rows; m.cols = cols; m.v = a.data; return m;
}

}  // namespace

int
main()
{
  const char* dir = std::getenv("VPIPE_LTX25_GOLDENS");
  if (dir == nullptr) {
    std::printf("SKIPPED: set VPIPE_LTX25_GOLDENS. NOTHING was checked.\n");
    return 0;
  }
  g_dir = dir;
  MetalCompute mc(nullptr);
  if (!mc.valid()) {
    std::printf("SKIPPED: no usable Metal device. NOTHING was checked.\n");
    return 0;
  }
  mc.register_metal_library(ltx25::kMetalLibBf16, ltx25_kernels_bf16_metallib,
                            ltx25_kernels_bf16_metallib_len);

  ltx25::MetalOps ops;
  std::string err;
  if (!ops.init(&mc, &err)) {
    std::printf("FAILED to init ops: %s\n", err.c_str());
    return 1;
  }
  ltx25::BlockWeights w;
  if (!load_weights(w)) { std::printf("FAILURES\n"); return 1; }

  auto blk = ltx25::MetalBlock::create(ops, w, &err);
  if (!blk) { std::printf("FAILED to build the block: %s\n", err.c_str()); return 1; }
  if (!blk->reserve(kTV, kTA, kTT, &err)) {
    std::printf("FAILED to reserve: %s\n", err.c_str());
    return 1;
  }
  check(true, "block uploaded and scratch reserved");

  // for_each_weight must reach EVERY weight buffer in the block. It is
  // what the prefetch walks, and a slot left out of it is silent: the
  // block still runs, it just runs with that tensor cold. So the count
  // is pinned here, and a new field in GpuAttn / GpuStream /
  // GpuBlockWeights has to come here and be counted.
  //
  //   per attention   4 projections + 4 biases + 2 norms + 2 gate  = 12
  //   per stream      attn1 + attn2 + 2 ff + 2 ff bias + 3 tables  = 31
  //   audio block     video + audio + a2v + v2a = 31 + 31 + 12 + 12 = 86
  {
    std::size_t seen = 0, non_empty = 0, bytes = 0;
    ltx25::for_each_weight(
        blk->weights(),
        [&](const vpipe::metal_compute::SharedBuffer& b) {
          ++seen;
          if (!b.empty()) { ++non_empty; bytes += b.byte_size(); }
        });
    check(blk->weights().have_audio, "the test block carries audio");
    check(seen == 86,
          "for_each_weight reaches all 86 weight slots (saw " +
              std::to_string(seen) + ")");
    // Every slot of a dense audio block is filled EXCEPT the four
    // to_gate_logits biases, which this checkpoint ships without.
    check(non_empty >= 82,
          "at least 82 of them are bound (" + std::to_string(non_empty) + ")");
    check(bytes > 0, "and they carry bytes (" + std::to_string(bytes) + ")");
  }

  // The same 1-D tables the golden used; the cross tables are at the
  // AUDIO width for BOTH streams.
  const ltx25::RopeTable vpe = ltx25::build_index_rope(kTV, {20}, kVD, kVH);
  const ltx25::RopeTable ape = ltx25::build_index_rope(kTA, {20}, kAD, kAH);
  const ltx25::RopeTable vcpe = ltx25::build_index_rope(kTV, {20}, kAD, kAH);
  blk->set_rope(&vpe, &ape, &vcpe, &ape);

  // ---- the CPU reference, on the same inputs -------------------------
  ltx25::StreamInput cv, ca;
  cv.x = mat(g("small_vx_in"), kTV, kVD);
  cv.context = mat(g("small_vctx"), kTT, kVD);
  cv.timesteps = g("small_v_ts").data;
  cv.cross_scale_shift = g("small_v_css").data;
  cv.cross_gate = g("small_v_cg").data;
  cv.prompt_timestep = g("small_v_pts").data;
  cv.pe = &vpe; cv.cross_pe = &vcpe;
  ca.x = mat(g("small_ax_in"), kTA, kAD);
  ca.context = mat(g("small_actx"), kTT, kAD);
  ca.timesteps = g("small_a_ts").data;
  ca.cross_scale_shift = g("small_a_css").data;
  ca.cross_gate = g("small_a_cg").data;
  ca.prompt_timestep = g("small_a_pts").data;
  ca.pe = &ape; ca.cross_pe = &ape;
  if (!ltx25::block_forward(w, cv, ca, &err)) {
    std::printf("CPU reference failed: %s\n", err.c_str());
    return 1;
  }

  // ---- the GPU block -------------------------------------------------
  // GpuStreamInput BORROWS its buffers, so they must outlive the
  // forward -- named locals here rather than temporaries.
  auto vx_b   = ops.upload_bf16(g("small_vx_in").data);
  auto vctx_b = ops.upload_bf16(g("small_vctx").data);
  auto vts_b  = ops.upload_bf16(g("small_v_ts").data);
  auto vcss_b = ops.upload_bf16(g("small_v_css").data);
  auto vcg_b  = ops.upload_bf16(g("small_v_cg").data);
  auto ax_b   = ops.upload_bf16(g("small_ax_in").data);
  auto actx_b = ops.upload_bf16(g("small_actx").data);
  auto ats_b  = ops.upload_bf16(g("small_a_ts").data);
  auto acss_b = ops.upload_bf16(g("small_a_css").data);
  auto acg_b  = ops.upload_bf16(g("small_a_cg").data);
  auto vpts_b = ops.upload_bf16(g("small_v_pts").data);
  auto apts_b = ops.upload_bf16(g("small_a_pts").data);

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

  const std::vector<float> gvx =
      ltx25::MetalOps::download_bf16(vx_b, (std::size_t)kTV * kVD);
  const std::vector<float> gax =
      ltx25::MetalOps::download_bf16(ax_b, (std::size_t)kTA * kAD);

  // ~40 dependent bf16 ops deep. Anything much below this is the dtype;
  // anything much above is a transcription bug.
  const double kBar = 4e-2;
  auto cmp = [&](const char* what, const std::vector<float>& got,
                 const std::vector<float>& want, const char* against) {
    const double r = npy::rel_l2(got, want);
    std::printf("       %-10s vs %-14s rel-L2 %.3e\n", what, against, r);
    check(r < kBar, std::string(what) + " vs " + against);
  };
  cmp("video", gvx, cv.x.v, "the CPU ref");
  cmp("audio", gax, ca.x.v, "the CPU ref");
  cmp("video", gvx, g("small_vx_out").data, "the golden");
  cmp("audio", gax, g("small_ax_out").data, "the golden");

  // ---- the PER-TOKEN modulation path ----------------------------------
  //
  // A conditioned generation gives each token its own noise level, so
  // the block reads its shift/scale/gate from row `level[token]` of a
  // [levels][dim] table instead of one broadcast row. Three runs pin it,
  // and all three demand a BIT-IDENTICAL result rather than a tolerance:
  // this is the same arithmetic reading the same numbers through one
  // more indirection, so any difference at all is a bug in the
  // indexing, not in the dtype.
  //
  //   A  two levels, every token on row 0, row 1 poisoned
  //   B  two levels, every token on row 1, row 0 poisoned
  //   C  two levels holding the SAME driver, tokens split between them
  //
  // A and B together prove the row is actually selected (a stride bug
  // fails one of them); C proves a mixed index does not disturb the
  // arithmetic. What none of them can check is that two DIFFERENT levels
  // produce different output -- attention couples the tokens, so there
  // is nothing uniform to compare a mixed run against. That is what the
  // conditioning golden and the end-to-end run are for.
  {
    std::printf("  --- per-token modulation ---\n");
    const std::vector<float>& v_ts = g("small_v_ts").data;
    const std::vector<float>& a_ts = g("small_a_ts").data;
    // A driver that is WRONG on purpose. Poisoning the unused row is
    // what turns "the levels happen to agree" into a real check.
    auto poison = [](const std::vector<float>& v) {
      std::vector<float> p(v.size());
      for (std::size_t i = 0; i < v.size(); ++i) { p[i] = v[i] * -3.0f + 1.0f; }
      return p;
    };
    auto two_rows = [](const std::vector<float>& lo,
                       const std::vector<float>& hi) {
      std::vector<float> t;
      t.reserve(lo.size() + hi.size());
      t.insert(t.end(), lo.begin(), lo.end());
      t.insert(t.end(), hi.begin(), hi.end());
      return t;
    };
    auto index_buf = [&](int n, int value, int split) {
      // `split` < 0 means a uniform index; otherwise tokens below it get
      // 0 and the rest 1.
      auto b = mc.make_shared_buffer((std::size_t)n * sizeof(std::int32_t));
      auto* p = static_cast<std::int32_t*>(b.contents());
      for (int i = 0; i < n; ++i) {
        p[i] = (split < 0) ? value : (i < split ? 0 : 1);
      }
      return b;
    };

    if (!blk->reserve(kTV, kTA, kTT, &err, /*max_levels=*/2)) {
      check(false, "reserve for two levels: " + err);
    }

    struct Case { const char* name; int index; int split; bool hi; };
    const Case cases[] = {
        {"every token on level 0", 0, -1, /*driver in row 1 is poison*/ false},
        {"every token on level 1", 1, -1, true},
        {"a mixed index over two identical levels", 0, kTV / 2, false},
    };
    for (const Case& c : cases) {
      // Row layout: the real driver in the row the tokens point at, the
      // poison in the other -- except the mixed case, where BOTH rows
      // hold the real one.
      const bool same = (c.split >= 0);
      auto vts2 = ops.upload_bf16(
          c.hi ? two_rows(poison(v_ts), v_ts)
               : two_rows(v_ts, same ? v_ts : poison(v_ts)));
      auto ats2 = ops.upload_bf16(
          c.hi ? two_rows(poison(a_ts), a_ts)
               : two_rows(a_ts, same ? a_ts : poison(a_ts)));
      auto vlvl = index_buf(kTV, c.index, c.split);
      auto albl = index_buf(kTA, c.index, c.split >= 0 ? kTA / 2 : -1);

      auto vx2 = ops.upload_bf16(g("small_vx_in").data);
      auto ax2 = ops.upload_bf16(g("small_ax_in").data);
      ltx25::GpuStreamInput v2 = gv, a2 = ga;
      v2.x = &vx2; v2.timesteps = &vts2; v2.level = &vlvl; v2.n_levels = 2;
      a2.x = &ax2; a2.timesteps = &ats2; a2.level = &albl; a2.n_levels = 2;

      auto s2 = mc.make_command_stream();
      {
        auto e2 = s2.begin_compute();
        if (!blk->forward(e2, v2, a2, &err)) {
          check(false, std::string(c.name) + ": forward: " + err);
          continue;
        }
      }
      s2.commit().wait();

      const std::vector<float> v_got =
          ltx25::MetalOps::download_bf16(vx2, (std::size_t)kTV * kVD);
      const std::vector<float> a_got =
          ltx25::MetalOps::download_bf16(ax2, (std::size_t)kTA * kAD);
      std::size_t vbad = 0, abad = 0;
      for (std::size_t i = 0; i < v_got.size(); ++i) {
        if (v_got[i] != gvx[i]) { ++vbad; }
      }
      for (std::size_t i = 0; i < a_got.size(); ++i) {
        if (a_got[i] != gax[i]) { ++abad; }
      }
      check(vbad == 0 && abad == 0,
            std::string(c.name) + " is bit-identical to the broadcast run (" +
            std::to_string(vbad) + " video / " + std::to_string(abad) +
            " audio values differ)");
    }
  }

  // ---- ONE ARENA FOR THE WHOLE STACK ----------------------------------
  //
  // The blocks run strictly sequentially and nothing in the scratch
  // survives a forward, so 48 blocks can share one arena instead of
  // allocating 48. At the geometry this model is built for that is
  // 56.8 GB against 1.18 GB -- the difference between swapping a 64 GB
  // box to death and having 39 GB spare.
  //
  // The claim is that it changes NOTHING, so the bar is BIT-IDENTICAL,
  // not a tolerance. Two blocks are run in sequence over the same
  // weights, once with private arenas and once sharing: if block 1 could
  // see block 0's leftovers, or if a shared buffer were aliased where a
  // private one was not, the second result would differ.
  {
    std::printf("  --- one shared scratch arena ---\n");
    auto run_pair = [&](bool shared, std::vector<float>* vout,
                        std::vector<float>* aout) {
      std::string e;
      auto b0 = ltx25::MetalBlock::create(ops, w, &e);
      auto b1 = ltx25::MetalBlock::create(ops, w, &e);
      if (!b0 || !b1) { check(false, "second block: " + e); return false; }
      std::shared_ptr<ltx25::BlockScratch> arena;
      for (auto* b : {b0.get(), b1.get()}) {
        if (!b->reserve(kTV, kTA, kTT, &e, 1, shared ? &arena : nullptr)) {
          check(false, "reserve: " + e);
          return false;
        }
        b->set_rope(&vpe, &ape, &vcpe, &ape);
      }
      if (shared) {
        std::printf("       arena holds %.2f MB, shared by both blocks\n",
                    (double)arena->bytes() / (1024.0 * 1024.0));
        // Both blocks must be pointing at the SAME object, or the test
        // is measuring two private arenas and proving nothing.
        check(b0->scratch().get() == b1->scratch().get(),
              "both blocks hold the same arena");
      } else {
        check(b0->scratch().get() != b1->scratch().get(),
              "private arenas really are separate");
      }
      auto vx = ops.upload_bf16(g("small_vx_in").data);
      auto ax = ops.upload_bf16(g("small_ax_in").data);
      ltx25::GpuStreamInput v2 = gv, a2 = ga;
      v2.x = &vx;
      a2.x = &ax;
      for (auto* b : {b0.get(), b1.get()}) {
        auto st = mc.make_command_stream();
        {
          auto en = st.begin_compute();
          if (!b->forward(en, v2, a2, &e)) {
            check(false, "forward: " + e);
            return false;
          }
        }
        st.commit().wait();
      }
      *vout = ltx25::MetalOps::download_bf16(vx, (std::size_t)kTV * kVD);
      *aout = ltx25::MetalOps::download_bf16(ax, (std::size_t)kTA * kAD);
      return true;
    };

    std::vector<float> pv, pa, sv, sa;
    if (run_pair(false, &pv, &pa) && run_pair(true, &sv, &sa)) {
      std::size_t vbad = 0, abad = 0;
      for (std::size_t i = 0; i < pv.size(); ++i) {
        if (pv[i] != sv[i]) { ++vbad; }
      }
      for (std::size_t i = 0; i < pa.size(); ++i) {
        if (pa[i] != sa[i]) { ++abad; }
      }
      check(vbad == 0 && abad == 0,
            "two blocks sharing one arena are bit-identical to two with "
            "their own (" + std::to_string(vbad) + " video / " +
            std::to_string(abad) + " audio values differ)");
    }
  }

  std::printf("%s\n", g_fail == 0 ? "ALL PASSED" : "FAILURES");
  return g_fail == 0 ? 0 : 1;
}
