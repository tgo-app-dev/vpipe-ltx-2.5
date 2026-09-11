// SageAttention wired into this DiT: the int8 QK^T product, on the
// attentions that have a matrix-core kernel to run it.
//
// WHAT THIS FILE CAN CHECK DEPENDS ON THE BOX, and it says which.
//
// The int8 fragment MMA has no ALU fallback, so on a GPU without matrix
// cores the tier declines and the model runs dense. That is not a
// failure -- refusing would make a graph that runs today un-runnable on
// half the boxes it runs on now -- but it IS the property most worth
// pinning there, and it is pinned first: asking for Sage on a box that
// cannot do it must change the answer by NOTHING. A tier that quietly
// perturbed a dense run would be the worst of both.
//
// On a box that has them, the rest runs: the twin specialises, both head
// widths take it, the answer moves and stays close, and the scratch
// comes out of the planes the block lent rather than out of the process.
//
// AND THE TWO TIERS TOGETHER. Sol and Sage are orthogonal -- Sol decides
// which key blocks are read, Sage how the ones that are read are
// computed -- and both carve from arenas the block lends them. Lending
// either one the other's region would be two allocators over one buffer,
// so "both on at once" is a correctness case and not just a supported
// combination.
//
// Needs no checkpoint and no goldens: the weights are drawn from a fixed
// PRNG, so this runs everywhere the GPU does.

#include "ltx25-block-metal.h"
#include "ltx25-block-ref.h"
#include "ltx25-config.h"
#include "ltx25-metal-ops.h"
#include "ltx25-rope.h"

#include "generative-models/shared/metal-sage-attention.h"

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

// head_dim 128 on the video stream and 64 on the audio one -- BOTH of
// which the matrix-core steel kernel has an entry point for, and both of
// which therefore take the int8 twin. That is the difference from Sol,
// which is specified at 128 and declines the audio stream outright.
constexpr int kVD = 256, kVH = 2, kAD = 128, kAH = 2;
constexpr int kTV = 2048, kTA = 128, kTT = 16;

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
  ::unsetenv("VPIPE_SAGE_ATTN");
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
  const bool cores = mc.supports_matrix_cores();
  std::printf("  matrix cores: %s (attention kernel '%s')\n",
              cores ? "yes" : "no", ops.attn_kernel());

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

  // A BLOCK, and a run OVER a block, kept apart on purpose: the two
  // together are `gpu()` below, and separately they are what the
  // stack-lifetime case needs -- one block, two generations, different
  // settings.
  auto make_blk = [&](std::string* e_out) {
    std::string e;
    auto blk = ltx25::MetalBlock::create(ops, base, &e);
    if (!blk) { if (e_out) { *e_out = "block: " + e; } return blk; }
    if (!blk->reserve(kTV, kTA, kTT, &e)) {
      if (e_out) { *e_out = "reserve: " + e; }
      return std::unique_ptr<ltx25::MetalBlock>();
    }
    blk->set_rope(&vpe, &ape, &vcpe, &ape);
    return blk;
  };
  auto run_blk = [&](ltx25::MetalBlock* blk, int layer, std::string* e_out) {
    std::vector<float> out;
    std::string e;
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
  auto gpu = [&](int layer, std::string* e_out) {
    auto blk = make_blk(e_out);
    if (!blk) { return std::vector<float>(); }
    return run_blk(blk.get(), layer, e_out);
  };

  auto set_sage = [&](bool on, int dense_layers) {
    vpipe::genai::sage::Config c;
    c.enabled = on;
    c.dense_layers = dense_layers;
    std::string e;
    const bool ok = ops.set_sage(c, &e);
    if (!ok) { std::printf("       set_sage: %s\n", e.c_str()); }
    return ok;
  };
  auto set_sol = [&](bool on, float tau) {
    vpipe::genai::sol::Config c;
    c.enabled = on;
    c.tau = tau;
    c.key_block = 64;
    c.dense_layers = 0;
    c.local_radius = 1;
    std::string e;
    return ops.set_sol(c, &e);
  };

  // ---- the dense baseline -------------------------------------------
  if (!set_sage(false, 0) || !set_sol(false, 1.0f)) { return 1; }
  std::string ferr;
  const std::vector<float> dense = gpu(/*layer=*/7, &ferr);
  if (dense.empty()) {
    std::printf("FAILED to run the dense block: %s\n", ferr.c_str());
    return 1;
  }

  // ---- asking for it is never a failure -----------------------------
  check(set_sage(true, 0),
        "asking for sage_attn is accepted -- a box with no matrix cores "
        "declines, and declining is not a failure");
  check(ops.sage_takes(128) == cores,
        "the 128-wide attentions take it exactly when this GPU can");
  check(ops.sage_takes(64) == cores,
        "...and so do the 64-wide ones: the matrix-core steel kernel has "
        "an entry point for both, which is where Sol and Sage differ");
  check(!ops.sage_takes(96),
        "a width steel has no entry point for has no int8 twin either");

  if (!cores) {
    // THE PROPERTY THAT MATTERS ON THIS BOX. A tier that cannot run must
    // not perturb the run it cannot join -- not "close to", not "within
    // bf16": the same bytes, because nothing about the dispatch changed.
    const std::vector<float> got = gpu(7, &ferr);
    check(!got.empty() && identical(got, dense),
          "and a declined tier leaves the block BYTE-IDENTICAL to dense");
  }

  // ---- what only a matrix-core box can answer -----------------------
  if (cores) {
    const std::vector<float> got = gpu(7, &ferr);
    if (got.empty()) {
      std::printf("FAILED to run the int8 block: %s\n", ferr.c_str());
      return 1;
    }
    const double r = rel_l2(got, dense);
    std::printf("  sage vs dense: rel-L2 %.3e, sage owns %.2f MB\n", r,
                (double)ops.sage_resident_bytes() / 1048576.0);
    check(r > 0.0, "the int8 QK CHANGES the answer -- so the twin was "
                   "dispatched, not the f16 kernel a second time");
    check(r < 0.05, "...and stays close to it");
    check(ops.sage_resident_bytes() == 0,
          "its scratch came out of the planes the block lent it");

    // dense_layers covers this block: back to the f16 kernel, exactly.
    if (!set_sage(true, 8)) { return 1; }
    const std::vector<float> gated = gpu(/*layer=*/7, &ferr);
    check(!gated.empty() && identical(gated, dense),
          "a block below sage_dense_layers is BYTE-IDENTICAL to dense");
  } else {
    std::printf("  SKIPPED on this GPU: whether the int8 twin computes "
                "the right thing needs matrix cores\n");
  }

  // ---- A STACK OUTLIVES A GENERATION, and the twin has to too -------
  //
  // The attention plans -- the specialised pipeline and its params -- are
  // cached in the arena for the life of the stack, where `sage_attn` is
  // decided per generation. So a twin built from the SETTING would be
  // absent for every generation that turned the tier on after one that
  // had it off, and the second clip would quietly render dense while the
  // log said otherwise. The twin is therefore built from what the GPU
  // can do, and this is the case that says so: ONE block, its plans
  // built by a run with Sage off, then a run with it on.
  if (cores) {
    if (!set_sage(false, 0) || !set_sol(false, 1.0f)) { return 1; }
    auto blk = make_blk(&ferr);
    if (!blk) {
      std::printf("FAILED to build the reused block: %s\n", ferr.c_str());
      return 1;
    }
    const std::vector<float> first = run_blk(blk.get(), 7, &ferr);
    check(!first.empty() && identical(first, dense),
          "a block whose plans were built with Sage off runs dense");
    if (!set_sage(true, 0)) { return 1; }
    const std::vector<float> second = run_blk(blk.get(), 7, &ferr);
    check(!second.empty() && !identical(second, dense),
          "...and the SAME block takes the int8 twin once the next "
          "generation asks for it, rather than being stuck on the plans "
          "the first one built");
  }

  // ---- both tiers at once -------------------------------------------
  //
  // They are independent by construction and they carve from DIFFERENT
  // lent regions, which is the part that would fail loudly rather than
  // subtly. On a box with no matrix cores Sage is inert, so what this
  // arm proves there is that enabling it disturbs nothing -- which is
  // the same shape as the in-tree sol/i8 pair's own check.
  if (!set_sage(true, 0) || !set_sol(true, 1.0f)) { return 1; }
  {
    const std::vector<float> both = gpu(7, &ferr);
    check(!both.empty(), "sol_attn and sage_attn are settable together");
    bool finite = !both.empty();
    for (float z : both) { if (!std::isfinite(z)) { finite = false; break; } }
    check(finite, "...and what the pair returns is finite");
    if (!cores) {
      if (!set_sage(false, 0)) { return 1; }
      const std::vector<float> sol_only = gpu(7, &ferr);
      check(!sol_only.empty() && identical(both, sol_only),
            "with no matrix cores, adding sage to sol changes nothing");
    }
  }

  // ---- the override, both ways --------------------------------------
  {
    ::setenv("VPIPE_SAGE_ATTN", "0", 1);
    vpipe::genai::sage::Config c;
    c.enabled = true;
    std::string e;
    check(ops.set_sage(c, &e) && !ops.sage_config().enabled,
          "VPIPE_SAGE_ATTN=0 takes the tier away from a graph that asked");
    ::setenv("VPIPE_SAGE_ATTN", "1", 1);
    c.enabled = false;
    // On a box with no matrix cores =1 still ends OFF, because the
    // driver declines -- the override reaches the config, not the
    // hardware. Both outcomes are correct and which one is expected is a
    // property of the box, so it is asserted as such rather than as a
    // constant.
    check(ops.set_sage(c, &e) && ops.sage_config().enabled == cores,
          "VPIPE_SAGE_ATTN=1 gives it to one that did not, where the GPU "
          "allows");
    ::unsetenv("VPIPE_SAGE_ATTN");
  }

  std::printf("ltx25-sage-attn: %d checks, %d failed\n", g_ran, g_fail);
  return g_fail == 0 ? 0 : 1;
}
