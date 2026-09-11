// Sol-Attn wired into this DiT: does the routing reach the attention it
// is supposed to reach, and stay off the ones it is not?
//
// The METHOD is libvpipe's and is tested there -- the routing kernels,
// the block-sparse traversal, the merge, all of it against dense on the
// same kernel. What is new HERE is the wiring, and every way it can be
// wrong is invisible in a rendered clip:
//
//   * routed where it must not be. The text cross-attention and both
//     audio<->video directions have a key set that is NOT the query
//     set, so summarising it by centroids summarises the wrong
//     sequence. The audio stream's self-attention is 64 wide where the
//     method is specified at 128.
//   * routed when the configuration said dense. `sol_dense_layers`
//     leaves the leading blocks exact, and a block that ignored it
//     renders -- slightly differently, and nothing says so.
//   * not routed at all. A lossy mode that silently never engaged is
//     indistinguishable from one that engaged and bought nothing, and
//     the timings get believed either way.
//
// So the file drives the WHOLE BLOCK, and its controls are as important
// as its comparisons: tau = -inf keeps every block exact and must
// therefore reproduce dense through the entire routing machinery, and a
// `dense_layers` that covers this block must reproduce it BYTE for byte.
// Between them, a wiring that did nothing and a wiring that did the
// wrong thing fail differently.
//
// Needs no checkpoint and no goldens: the weights are drawn from a fixed
// PRNG, so this runs everywhere the GPU does. The widths are chosen for
// the method rather than for speed -- head_dim 128 because that is what
// it is specified at, and 2048 video tokens because below 16 routing
// blocks sol_takes() declines and there would be nothing to measure.

#include "ltx25-block-metal.h"
#include "ltx25-block-ref.h"
#include "ltx25-config.h"
#include "ltx25-metal-ops.h"
#include "ltx25-rope.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

extern "C" const unsigned char ltx25_kernels_bf16_metallib[];
extern "C" const unsigned long ltx25_kernels_bf16_metallib_len;

using vpipe::metal_compute::MetalCompute;

namespace {

int g_fail = 0;
int g_ran  = 0;

void
check(bool ok, const std::string& what)
{
  std::printf("  [%s] %s\n", ok ? " OK " : "FAIL", what.c_str());
  if (!ok) { ++g_fail; }
  ++g_ran;
}

// head_dim 128 on the video stream, which is what Sol-Attn is specified
// at and what the shipped model uses; 32 on the audio stream, which is
// what makes the audio self-attention decline for a reason of its own.
constexpr int kVD = 256, kVH = 2, kAD = 64, kAH = 2;
// 2048 video tokens: 32 routing blocks of 64, twice the floor
// sol_takes() imposes. The audio and text sequences stay short, as they
// are in the model.
constexpr int kTV = 2048, kTA = 64, kTT = 16;

struct Rng {
  std::uint64_t s;
  explicit Rng(std::uint64_t seed) : s(seed * 6364136223846793005ull + 1) {}
  float next(float lo, float hi)
  {
    s = s * 6364136223846793005ull + 1442695040888963407ull;
    const std::uint32_t x = (std::uint32_t)(s >> 33);
    return lo + (hi - lo) * ((float)x / 4294967296.0f);
  }
};

std::uint64_t
hash_(const std::string& s)
{
  std::uint64_t h = 1469598103934665603ull;
  for (char c : s) {
    h = (h ^ (std::uint64_t)(unsigned char)c) * 1099511628211ull;
  }
  return h;
}

std::vector<float>
draw(const std::string& name, std::size_t n, float amp)
{
  Rng r(hash_(name));
  std::vector<float> v(n);
  for (auto& z : v) { z = r.next(-amp, amp); }
  return v;
}

ltx25::Mat
draw_mat(const std::string& name, int rows, int cols)
{
  ltx25::Mat m;
  m.rows = rows;
  m.cols = cols;
  m.v = draw(name, (std::size_t)rows * cols, 1.0f / std::sqrt((float)cols));
  return m;
}

// The same construction ltx25-lora-apply-test draws its block from, at
// wider heads. The scale/gate rows are lifted by 1 for the reason stated
// there: a table drawn around zero makes the softmax flat and the
// attention blind to its own query, which is exactly the state in which
// an approximation of that attention looks perfect.
void
fill_attn(ltx25::AttnWeights& a, const std::string& p, int query_dim,
          int ctx_dim, int heads, int head_dim)
{
  const int inner = heads * head_dim;
  a.heads = heads;
  a.head_dim = head_dim;
  a.q_w = draw_mat(p + ".q", inner, query_dim);
  a.k_w = draw_mat(p + ".k", inner, ctx_dim);
  a.v_w = draw_mat(p + ".v", inner, ctx_dim);
  a.o_w = draw_mat(p + ".o", query_dim, inner);
  a.q_b = draw(p + ".qb", (std::size_t)inner, 0.05f);
  a.k_b = draw(p + ".kb", (std::size_t)inner, 0.05f);
  a.v_b = draw(p + ".vb", (std::size_t)inner, 0.05f);
  a.o_b = draw(p + ".ob", (std::size_t)query_dim, 0.05f);
  a.q_norm = draw(p + ".qn", (std::size_t)inner, 0.1f);
  a.k_norm = draw(p + ".kn", (std::size_t)inner, 0.1f);
  for (auto& z : a.q_norm) { z += 4.0f; }
  for (auto& z : a.k_norm) { z += 4.0f; }
  a.gate_w = draw_mat(p + ".g", heads, query_dim);
  a.gate_b = draw(p + ".gb", (std::size_t)heads, 0.05f);
}

void
fill_stream(ltx25::StreamWeights& s, const std::string& p, int dim,
            int text_dim, int heads)
{
  s.dim = dim;
  const int hd = dim / heads;
  fill_attn(s.attn1, p + ".attn1", dim, dim, heads, hd);
  fill_attn(s.attn2, p + ".attn2", dim, text_dim, heads, hd);
  s.ff_in  = draw_mat(p + ".ffin", 4 * dim, dim);
  s.ff_out = draw_mat(p + ".ffout", dim, 4 * dim);
  s.ff_in_b  = draw(p + ".ffinb", (std::size_t)4 * dim, 0.05f);
  s.ff_out_b = draw(p + ".ffoutb", (std::size_t)dim, 0.05f);
  s.scale_shift = draw(p + ".ss", (std::size_t)9 * dim, 0.2f);
  s.prompt_scale_shift = draw(p + ".pss", (std::size_t)2 * dim, 0.2f);
  s.cross_table = draw(p + ".ct", (std::size_t)5 * dim, 0.2f);
  auto lift = [&](std::vector<float>& t, std::initializer_list<int> rows) {
    for (int r : rows) {
      for (int i = 0; i < dim; ++i) { t[(std::size_t)r * dim + i] += 1.0f; }
    }
  };
  lift(s.scale_shift, {2, 5, 8});
  lift(s.cross_table, {4});
}

ltx25::BlockWeights
make_weights()
{
  ltx25::BlockWeights w;
  fill_stream(w.video, "v", kVD, kVD, kVH);
  fill_stream(w.audio, "a", kAD, kAD, kAH);
  fill_attn(w.a2v, "a2v", kVD, kAD, kVH, kVD / kVH);
  fill_attn(w.v2a, "v2a", kAD, kVD, kAH, kAD / kAH);
  return w;
}

double
rel_l2(const std::vector<float>& a, const std::vector<float>& b)
{
  if (a.size() != b.size() || a.empty()) { return 1.0e9; }
  double n = 0.0, d = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double e = (double)a[i] - (double)b[i];
    n += e * e;
    d += (double)b[i] * (double)b[i];
  }
  return d > 0.0 ? std::sqrt(n / d) : std::sqrt(n);
}

bool
identical(const std::vector<float>& a, const std::vector<float>& b)
{
  if (a.size() != b.size() || a.empty()) { return false; }
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i]) { return false; }
  }
  return true;
}

}  // namespace

int
main()
{
  // The env override reaches set_sol, so a stray one in the environment
  // would silently turn every arm below into the same arm.
  ::unsetenv("VPIPE_SOL_ATTN");
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

  // ---- the predicate, which needs no forward ------------------------
  {
    vpipe::genai::sol::Config c;
    c.enabled = true;
    c.key_block = 64;
    std::string e;
    if (!ops.set_sol(c, &e)) {
      std::printf("FAILED to build Sol-Attn: %s\n", e.c_str());
      return 1;
    }
    check(ops.sol_takes(kVH, kTV, kTV, 128),
          "the video self-attention at 2048 tokens IS a candidate");
    check(!ops.sol_takes(kAH, kTA, kTA, 32),
          "the audio self-attention is not: head_dim 32, and the method "
          "is specified at 128");
    check(!ops.sol_takes(kVH, kTV, kTT, 128),
          "the text cross-attention is not: its keys are not its queries");
    check(!ops.sol_takes(kVH, kTV, kTA, 128),
          "neither is an audio<->video direction, for the same reason");
    check(!ops.sol_takes(kVH, 512, 512, 128),
          "nor a sequence of 8 routing blocks, where the local band and "
          "the tail already cover most of it");
    vpipe::genai::sol::Config off;
    off.enabled = false;
    check(ops.set_sol(off, &e) && !ops.sol_takes(kVH, kTV, kTV, 128),
          "and nothing is a candidate with the tier off");
  }

  const ltx25::BlockWeights base = make_weights();
  const ltx25::RopeTable vpe = ltx25::build_index_rope(kTV, {20}, kVD, kVH);
  const ltx25::RopeTable ape = ltx25::build_index_rope(kTA, {20}, kAD, kAH);
  const ltx25::RopeTable vcpe = ltx25::build_index_rope(kTV, {20}, kAD, kAH);

  const std::vector<float> vx0  = draw("in/vx", (std::size_t)kTV * kVD, 0.8f);
  const std::vector<float> vctx = draw("in/vctx", (std::size_t)kTT * kVD, 0.8f);
  const std::vector<float> vts  = draw("in/vts", (std::size_t)9 * kVD, 0.2f);
  const std::vector<float> vcss = draw("in/vcss", (std::size_t)4 * kVD, 0.2f);
  const std::vector<float> vcg  = draw("in/vcg", (std::size_t)kVD, 0.2f);
  const std::vector<float> vpts = draw("in/vpts", (std::size_t)2 * kVD, 0.2f);
  const std::vector<float> ax0  = draw("in/ax", (std::size_t)kTA * kAD, 0.8f);
  const std::vector<float> actx = draw("in/actx", (std::size_t)kTT * kAD, 0.8f);
  const std::vector<float> ats  = draw("in/ats", (std::size_t)9 * kAD, 0.2f);
  const std::vector<float> acss = draw("in/acss", (std::size_t)4 * kAD, 0.2f);
  const std::vector<float> acg  = draw("in/acg", (std::size_t)kAD, 0.2f);
  const std::vector<float> apts = draw("in/apts", (std::size_t)2 * kAD, 0.2f);

  // One GPU forward at the given layer index, over the Sol settings
  // currently on `ops`. Returns [video | audio].
  auto gpu = [&](int layer, std::string* e_out) {
    std::vector<float> out;
    std::string e;
    auto blk = ltx25::MetalBlock::create(ops, base, &e);
    if (!blk) { if (e_out) { *e_out = "block: " + e; } return out; }
    if (!blk->reserve(kTV, kTA, kTT, &e)) {
      if (e_out) { *e_out = "reserve: " + e; }
      return out;
    }
    blk->set_rope(&vpe, &ape, &vcpe, &ape);
    auto vx_b = ops.upload_bf16(vx0);
    auto ax_b = ops.upload_bf16(ax0);
    auto vctx_b = ops.upload_bf16(vctx);
    auto vts_b = ops.upload_bf16(vts);
    auto vcss_b = ops.upload_bf16(vcss);
    auto vcg_b = ops.upload_bf16(vcg);
    auto vpts_b = ops.upload_bf16(vpts);
    auto actx_b = ops.upload_bf16(actx);
    auto ats_b = ops.upload_bf16(ats);
    auto acss_b = ops.upload_bf16(acss);
    auto acg_b = ops.upload_bf16(acg);
    auto apts_b = ops.upload_bf16(apts);
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
      if (!blk->forward(enc, gv, ga, &e, nullptr, layer)) {
        if (e_out) { *e_out = "forward: " + e; }
        return out;
      }
    }
    stream.commit().wait();
    out = ltx25::MetalOps::download_bf16(vx_b, (std::size_t)kTV * kVD);
    const auto a = ltx25::MetalOps::download_bf16(ax_b, (std::size_t)kTA * kAD);
    out.insert(out.end(), a.begin(), a.end());
    return out;
  };

  auto set_sol = [&](bool on, float tau, int dense_layers, int radius = 1) {
    vpipe::genai::sol::Config c;
    c.enabled = on;
    c.tau = tau;
    c.key_block = 64;
    c.dense_layers = dense_layers;
    c.local_radius = radius;
    std::string e;
    if (!ops.set_sol(c, &e)) {
      std::printf("       set_sol: %s\n", e.c_str());
      return false;
    }
    return true;
  };

  // ---- the dense baseline -------------------------------------------
  if (!set_sol(false, 1.0f, 0)) { return 1; }
  std::string ferr;
  const std::vector<float> dense = gpu(/*layer=*/7, &ferr);
  if (dense.empty()) {
    std::printf("FAILED to run the dense block: %s\n", ferr.c_str());
    return 1;
  }

  // ---- tau = -inf: every block exact, so nothing may change ---------
  //
  // The strongest control in the file. The routing runs -- summaries,
  // threshold, CSR, both partial softmaxes and the merge -- and keeps
  // every key block, so the answer has to come back through all of it
  // unchanged. A wiring that fed the routing the wrong buffer, the wrong
  // head count or the wrong scale fails HERE, where an approximation
  // that is merely poor would not.
  if (!set_sol(true, -INFINITY, 0)) { return 1; }
  {
    const std::vector<float> got = gpu(7, &ferr);
    if (got.empty()) {
      std::printf("FAILED to run the routed block: %s\n", ferr.c_str());
      return 1;
    }
    const double r = rel_l2(got, dense);
    std::printf("  tau = -inf vs dense: rel-L2 %.3e\n", r);
    check(r <= 1.0e-6, "keeping every block exact reproduces the dense "
                       "block through the whole routing machinery");
    check(ops.sol_exact_blocks() > 0,
          "...and it says so: the routing counter is non-zero, so the "
          "block really did take this path");
    std::printf("  Sol owns %.2f MB after the lend\n",
                (double)ops.sol_resident_bytes() / 1048576.0);
    check(ops.sol_resident_bytes() == 0,
          "the whole scratch came out of the arena the block lent it");
  }

  // ---- dense_layers covers this block: byte for byte ----------------
  //
  // Not `close`. A leading block left dense is the ablation that says
  // the filter does anything, so it has to mean UNCHANGED MODEL and not
  // "the same answer by a different route".
  if (!set_sol(true, 1.0f, 8)) { return 1; }
  {
    ops.sol_reset_counts();
    const std::vector<float> got = gpu(/*layer=*/7, &ferr);
    check(!got.empty() && identical(got, dense),
          "a block below sol_dense_layers is BYTE-IDENTICAL to the plain "
          "block");
    check(ops.sol_exact_blocks() == 0,
          "...and routed nothing, which is what makes that identity a "
          "fact about the gate rather than about the threshold");
  }

  // ---- tau = 1.0: routed, and close ---------------------------------
  if (!set_sol(true, 1.0f, 0)) { return 1; }
  {
    ops.sol_reset_counts();
    const std::vector<float> got = gpu(7, &ferr);
    if (got.empty()) {
      std::printf("FAILED to run the routed block: %s\n", ferr.c_str());
      return 1;
    }
    const double r = rel_l2(got, dense);
    const long long ex = ops.sol_exact_blocks();
    const long long tot = ops.sol_total_blocks();
    std::printf("  tau = 1.0 vs dense: rel-L2 %.3e, kept %lld of %lld key "
                "blocks exact (%.1f%%)\n", r, ex, tot,
                tot > 0 ? 100.0 * (double)ex / (double)tot : 0.0);
    check(r < 0.25, "the routed block is close to the dense one");
    check(ex > 0 && tot > 0 && ex < tot,
          "and the routing dropped something: the kept fraction is "
          "strictly between nothing and everything");
    // NO FLOOR UNDER THAT BAR, and the reason is the most interesting
    // measurement in the file. At tau 1.0 this block comes back rel-L2
    // 0.000e+00 against dense while the counter says 70% of key blocks
    // were approximated -- and that is not the routing failing to run.
    // It is RoPE. The positions here span 2048 tokens over a max_pos of
    // 20, so q.k decays with distance and the attention is strongly
    // local; the argmax of nearly every row falls inside the local band,
    // which is exact by construction. Approximating the other 70%
    // therefore costs nothing, because those blocks were contributing
    // nothing.
    //
    // That is the regime the method is FOR, so it is the wrong place to
    // demand a difference: asserting one here would be asserting that
    // Sol is inaccurate. The materiality control is the arm below, which
    // takes the band away so that the THRESHOLD is the only thing
    // deciding.
  }

  // ---- THE MATERIALITY CONTROL --------------------------------------
  //
  // Everything above is an equality: dense reproduced, bytes unchanged,
  // a counter moving. All of it passes if the routed path is somehow a
  // slow spelling of the dense one. So one arm has to make the
  // approximation BITE and check that the block notices.
  //
  // tau at 1e30 admits nothing on its own merits, and local_radius -1
  // takes away the band that would otherwise cover for it -- so ONE
  // block of 32 stays exact, the tail, and everything the attention
  // actually wanted is approximated. A wiring that quietly ran dense
  // returns the dense answer here; this one must not.
  //
  // THE BAND IS OFF ON PURPOSE and it is what makes this a control
  // rather than a repeat of the arm above. With it on, the blocks the
  // attention leans on are kept whatever the threshold says, so an
  // implementation that ignored the threshold entirely would still come
  // back right.
  if (!set_sol(true, 1.0e30f, 0, /*radius=*/-1)) { return 1; }
  {
    ops.sol_reset_counts();
    const std::vector<float> got = gpu(7, &ferr);
    if (got.empty()) {
      std::printf("FAILED to run the crude block: %s\n", ferr.c_str());
      return 1;
    }
    const double r = rel_l2(got, dense);
    const long long ex = ops.sol_exact_blocks();
    const long long tot = ops.sol_total_blocks();
    std::printf("  tau = 1e30, no local band, vs dense: rel-L2 %.3e, kept "
                "%lld of %lld (%.1f%%)\n", r, ex, tot,
                tot > 0 ? 100.0 * (double)ex / (double)tot : 0.0);
    // MEASURED at 4.5e-3, against a dense block that a doubling of
    // attn1.to_v moves by 3.1e-1 -- so the floor is a fifth of what was
    // seen and two orders under what the attention is worth here.
    check(r > 1.0e-3,
          "approximating the blocks the attention leans on MOVES the "
          "block -- so the routed path is not a slow spelling of the "
          "dense one");
    bool finite = true;
    for (float z : got) { if (!std::isfinite(z)) { finite = false; break; } }
    check(finite, "...and what it returns is finite");
  }

  // ---- the override, both ways --------------------------------------
  {
    ::setenv("VPIPE_SOL_ATTN", "0", 1);
    vpipe::genai::sol::Config c;
    c.enabled = true;
    std::string e;
    check(ops.set_sol(c, &e) && !ops.sol_config().enabled,
          "VPIPE_SOL_ATTN=0 takes the tier away from a graph that asked");
    ::setenv("VPIPE_SOL_ATTN", "1", 1);
    c.enabled = false;
    check(ops.set_sol(c, &e) && ops.sol_config().enabled,
          "VPIPE_SOL_ATTN=1 gives it to one that did not");
    ::unsetenv("VPIPE_SOL_ATTN");
  }

  std::printf("ltx25-sol-attn: %d checks, %d failed\n", g_ran, g_fail);
  return g_fail == 0 ? 0 : 1;
}
