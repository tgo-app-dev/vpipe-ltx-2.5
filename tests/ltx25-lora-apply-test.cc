// Does the adapter reach the linear it names, at the point a fused
// weight would have?
//
// ltx25-lora-test checks that the published files LOAD -- the names, the
// widths, the mixed rank. It cannot check that the pairs are then
// APPLIED, and that is the half where being wrong is invisible: bind
// `to_q`'s delta to `to_k` and the block still runs every dispatch at
// full speed and returns a plausible tensor.
//
// TWO LAYERS, because one of them cannot cover everything.
//
// LAYER ONE -- `lora_add` against f64 on the host, at the six shapes the
// block binds it at. This pins the arithmetic: that A is read [rank][k]
// and B is [n][rank], that neither is transposed, that the result is
// added rather than assigned. It does not depend on the block noticing.
//
// LAYER TWO -- the whole block, runtime against FOLDED. Applying
// `y += (x@A^T)@B^T` at a linear's output is by construction the same
// thing as `W + B@A` folded into that linear, so:
//
//   ARM A   the GPU block with the adapter passed to forward()
//   ARM B   the CPU reference (pinned at 3e-7 against the model's own
//           implementation) over weights with B@A ADDED IN
//
// must agree. They disagree if a pair is bound to the wrong linear, if
// it is applied after the RMSNorm instead of before it, if the scratch
// is the wrong shape, or if M/K/N are transposed.
//
// AGREEING IS NOT ENOUGH, because it is also what an adapter that does
// nothing produces. So each site carries two controls: the fold must
// move the reference well past the tolerance, and the same GPU block run
// WITHOUT the adapter must NOT match -- the latter being what fails when
// a call site is missing, which is the actual bug this file exists to
// catch.
//
// SIX SITES CANNOT CARRY THOSE CONTROLS, and they are named rather than
// quietly dropped. Every `to_q` and `to_k` reaches the output only
// through a softmax over a handful of tokens, and that softmax is
// dominated by its argmax; MEASURED, replacing `a2v.to_k` outright --
// 2.6x the weight's own norm -- moves the block by less than 5e-7.
// Those sites are still checked for agreement, and are additionally
// asserted to be STILL BLIND, so one that becomes visible after a
// change here is a failure and not a silent demotion. What pins their
// binding instead is layer one plus the loader's width checks.
//
// Needs no checkpoint and no goldens: the weights are drawn from a fixed
// PRNG, so this runs everywhere the GPU does.

#include "ltx25-block-metal.h"
#include "ltx25-block-ref.h"
#include "ltx25-lora.h"
#include "ltx25-metal-ops.h"
#include "ltx25-rope.h"

#include <cmath>
#include <cstdint>
#include <chrono>
#include <cstdio>
#include <functional>
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

// The same widths ltx25-block-metal-test uses, so the two exercise the
// same head geometry: video 64 wide over 4 heads (head_dim 16), audio 32
// over 4 (head_dim 8).
constexpr int kVD = 64, kVH = 4, kAD = 32, kAH = 4;
constexpr int kTV = 12, kTA = 8, kTT = 6;
constexpr int kRank = 3;

// A deterministic PRNG. Seeded per tensor from its NAME, so two sites
// never get the same numbers -- a q/k swap that reused one draw would
// pass against itself.
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
  for (char c : s) { h = (h ^ (std::uint64_t)(unsigned char)c) * 1099511628211ull; }
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
  // Fan-in scaled, so a 256-wide feed-forward does not saturate what a
  // 32-wide attention leaves tame.
  m.v = draw(name, (std::size_t)rows * cols, 1.0f / std::sqrt((float)cols));
  return m;
}

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
  // 4, NOT 1. The RMSNorm gain sets the logit scale, and at gain 1 the
  // softmax over a handful of keys is so flat that the attention output
  // barely depends on its own query -- MEASURED: a 2.5x change to
  // `attn2.to_q` moved the block by 1e-5, three orders below the
  // tolerance, so the case could not tell a correct binding from a
  // missing one. At gain 4 the same perturbation moves it by a margin
  // the comparison can see.
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
  // THE SCALES AND GATES SIT AROUND 1, THE SHIFTS AROUND 0 -- and which
  // row is which matters, because both mistakes make the test vacuous
  // in the same way and neither shows up as a failure.
  //
  // MEASURED while writing this file. With every table drawn around
  // zero, the modulated keys are near zero, the softmax is uniform, and
  // the attention is INSENSITIVE TO ITS QUERY: perturbing `attn2.to_q`
  // by half its own magnitude moved the block by 1e-6. Adding 1 to the
  // whole table instead makes the SHIFT ~1 as well, which swamps the
  // per-token variation -- every key/value token becomes nearly the
  // same vector, so now the attention is insensitive to its KEYS
  // (a2v.to_k moved by <5e-4 at any amplitude).
  //
  // Both failures pass the runtime-vs-folded comparison. Only the
  // materiality control catches them, which is why it is not optional.
  s.scale_shift = draw(p + ".ss", (std::size_t)9 * dim, 0.2f);
  s.prompt_scale_shift = draw(p + ".pss", (std::size_t)2 * dim, 0.2f);
  s.cross_table = draw(p + ".ct", (std::size_t)5 * dim, 0.2f);
  auto lift = [&](std::vector<float>& t, std::initializer_list<int> rows) {
    for (int r : rows) {
      for (int i = 0; i < dim; ++i) { t[(std::size_t)r * dim + i] += 1.0f; }
    }
  };
  // 9 rows as (shift, scale, gate) x {self-attn, ff, text-cross}.
  // Only the GATES. The scales are used as `x * (1 + scale) + shift`,
  // so a scale drawn around 0 already means "leave it alone" -- lifting
  // those too would double every modulation, and lifting the SHIFTS
  // pushes every token toward the same vector, which is the second way
  // of making the attention blind to its keys.
  lift(s.scale_shift, {2, 5, 8});
  lift(s.cross_table, {4});
}

ltx25::BlockWeights
make_weights()
{
  ltx25::BlockWeights w;
  fill_stream(w.video, "v", kVD, kVD, kVH);
  fill_stream(w.audio, "a", kAD, kAD, kAH);
  // a2v: video queries over audio keys/values, and the reverse.
  fill_attn(w.a2v, "a2v", kVD, kAD, kVH, kVD / kVH);
  fill_attn(w.v2a, "v2a", kAD, kVD, kAH, kAD / kAH);
  return w;
}

// ---- the sites -------------------------------------------------------
//
// Each names one adapted linear twice: where its pair lives in a
// LoraBlock, and which base matrix a fold has to land in. The whole
// point of the file is that those two must be the SAME linear, so they
// are written side by side and nothing derives one from the other.
struct Site {
  const char* name;
  // Which residual stream this linear's delta lands in -- 'v' or 'a'.
  // Materiality is measured THERE and not on the concatenation, or the
  // audio sites are diluted by the video half being four times larger
  // and read as insensitive when they are not.
  char stream;
  std::function<ltx25::LoraPair&(ltx25::LoraBlock&)> pair;
  std::function<ltx25::Mat&(ltx25::BlockWeights&)>   base;
  // True when the BLOCK cannot see this delta -- see kBlind below.
  bool blind = false;
};

std::vector<Site>
sites()
{
  using LB = ltx25::LoraBlock;
  using BW = ltx25::BlockWeights;
  // `blind` marks a site whose delta the BLOCK OUTPUT cannot see, and
  // it is every `to_q` and `to_k` in the file. MEASURED: replacing
  // `a2v.to_k` outright -- a perturbation 2.6x the weight's own norm --
  // moves the reference by less than 5e-7, and `attn1.to_q` by 6e-3
  // against a bf16 floor of 3e-3. Queries and keys reach the output
  // only through a softmax over a handful of tokens, and that softmax
  // is dominated by its argmax; the argmax does not move.
  //
  // So these sites are checked for AGREEMENT (a wrong binding that
  // happened to be visible would still fail) but they cannot carry the
  // omission control, and pretending otherwise is what a materiality
  // check exists to prevent. What pins them instead is the op-level
  // test above -- `lora_add` at the exact shapes `to_q` and `to_k` are
  // bound at -- plus the loader test's width checks, which a q/k swap
  // in either CROSS-attention would fail outright (their keys read the
  // other stream's width).
  //
  // The list is EXPLICIT rather than inferred from the measurement, so
  // a site that becomes blind after a change to this block is a failure
  // and not a silent demotion.
  return {
    // The five slots of ONE attention, so each kind is pinned once.
    {"video attn1 to_q", 'v',
     [](LB& l) -> ltx25::LoraPair& { return l.video_attn1.q; },
     [](BW& w) -> ltx25::Mat& { return w.video.attn1.q_w; }, true},
    {"video attn1 to_k", 'v',
     [](LB& l) -> ltx25::LoraPair& { return l.video_attn1.k; },
     [](BW& w) -> ltx25::Mat& { return w.video.attn1.k_w; }, true},
    {"video attn1 to_v", 'v',
     [](LB& l) -> ltx25::LoraPair& { return l.video_attn1.v; },
     [](BW& w) -> ltx25::Mat& { return w.video.attn1.v_w; }},
    {"video attn1 to_out", 'v',
     [](LB& l) -> ltx25::LoraPair& { return l.video_attn1.o; },
     [](BW& w) -> ltx25::Mat& { return w.video.attn1.o_w; }},
    // The odd one out: `heads` rows, not `inner`. Bound against the
    // wrong width this is still a valid GEMM.
    {"video attn1 to_gate_logits", 'v',
     [](LB& l) -> ltx25::LoraPair& { return l.video_attn1.gate; },
     [](BW& w) -> ltx25::Mat& { return w.video.attn1.gate_w; }},
    // One slot of every OTHER module, which is what pins the module
    // binding rather than the slot binding.
    {"video attn2 (text cross) to_q", 'v',
     [](LB& l) -> ltx25::LoraPair& { return l.video_attn2.q; },
     [](BW& w) -> ltx25::Mat& { return w.video.attn2.q_w; }, true},
    {"video attn2 (text cross) to_out", 'v',
     [](LB& l) -> ltx25::LoraPair& { return l.video_attn2.o; },
     [](BW& w) -> ltx25::Mat& { return w.video.attn2.o_w; }},
    {"video ff.net.0.proj", 'v',
     [](LB& l) -> ltx25::LoraPair& { return l.video_ff_in; },
     [](BW& w) -> ltx25::Mat& { return w.video.ff_in; }},
    {"video ff.net.2", 'v',
     [](LB& l) -> ltx25::LoraPair& { return l.video_ff_out; },
     [](BW& w) -> ltx25::Mat& { return w.video.ff_out; }},
    {"audio attn1 to_q", 'a',
     [](LB& l) -> ltx25::LoraPair& { return l.audio_attn1.q; },
     [](BW& w) -> ltx25::Mat& { return w.audio.attn1.q_w; }, true},
    {"audio attn2 (text cross) to_v", 'a',
     [](LB& l) -> ltx25::LoraPair& { return l.audio_attn2.v; },
     [](BW& w) -> ltx25::Mat& { return w.audio.attn2.v_w; }},
    {"audio ff.net.0.proj", 'a',
     [](LB& l) -> ltx25::LoraPair& { return l.audio_ff_in; },
     [](BW& w) -> ltx25::Mat& { return w.audio.ff_in; }},
    {"audio ff.net.2", 'a',
     [](LB& l) -> ltx25::LoraPair& { return l.audio_ff_out; },
     [](BW& w) -> ltx25::Mat& { return w.audio.ff_out; }},
    // The two cross-attention directions. Their key/value width is the
    // OTHER stream's -- the case a square test cannot tell apart -- and
    // a2v writes VIDEO while v2a writes AUDIO.
    {"a2v to_k (audio keys)", 'v',
     [](LB& l) -> ltx25::LoraPair& { return l.a2v.k; },
     [](BW& w) -> ltx25::Mat& { return w.a2v.k_w; }, true},
    {"v2a to_k (video keys)", 'a',
     [](LB& l) -> ltx25::LoraPair& { return l.v2a.k; },
     [](BW& w) -> ltx25::Mat& { return w.v2a.k_w; }, true},
    {"a2v to_v (audio values)", 'v',
     [](LB& l) -> ltx25::LoraPair& { return l.a2v.v; },
     [](BW& w) -> ltx25::Mat& { return w.a2v.v_w; }},
    {"a2v to_out", 'v',
     [](LB& l) -> ltx25::LoraPair& { return l.a2v.o; },
     [](BW& w) -> ltx25::Mat& { return w.a2v.o_w; }},
    {"v2a to_out", 'a',
     [](LB& l) -> ltx25::LoraPair& { return l.v2a.o; },
     [](BW& w) -> ltx25::Mat& { return w.v2a.o_w; }},
  };
}

// A and B for one site, held on the host so the same numbers can be
// folded into the reference and uploaded to the GPU.
struct HostPair {
  std::vector<float> a;   // [rank][k]
  std::vector<float> b;   // [n][rank]
  int rank = 0, k = 0, n = 0;
};

HostPair
make_lora_pair(const std::string& name, const ltx25::Mat& base)
{
  HostPair p;
  p.rank = kRank;
  p.k = base.cols;
  p.n = base.rows;
  // Scaled so the delta is the same order as the weight it rides on:
  // small enough not to blow the block up, large enough that the
  // materiality control below clears the bf16 floor by a wide margin.
  const float amp = 3.0f / std::sqrt((float)(kRank * p.k));
  p.a = draw(name + "/A", (std::size_t)p.rank * p.k, amp);
  p.b = draw(name + "/B", (std::size_t)p.n * p.rank, 1.0f);
  return p;
}

// W += B @ A, in f32, into the reference's own weights.
void
fold(ltx25::Mat& w, const HostPair& p)
{
  for (int o = 0; o < p.n; ++o) {
    for (int i = 0; i < p.k; ++i) {
      double acc = 0.0;
      for (int j = 0; j < p.rank; ++j) {
        acc += (double)p.b[(std::size_t)o * p.rank + j] *
               (double)p.a[(std::size_t)j * p.k + i];
      }
      w.v[(std::size_t)o * w.cols + i] += (float)acc;
    }
  }
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

// ---- layer one: the op itself ---------------------------------------
//
// `lora_add` at the shapes the block binds it at, against f64 on the
// host. This is what pins the ARITHMETIC -- that A is read [rank][k]
// and B is [n][rank], that neither is transposed, that the result is
// ADDED and not assigned -- independently of whether the block it rides
// in happens to be sensitive to the linear in question.

void
test_op(ltx25::MetalOps& ops, MetalCompute& mc)
{
  std::printf("  --- lora_add, against f64 on the host ---\n");
  struct Shape { const char* what; int M, K, N, rank; };
  const Shape shapes[] = {
    {"square self-attention projection", 12, 64, 64, 3},
    {"cross-attention key (narrow in)", 8, 32, 64, 3},
    {"cross-attention key (wide in)", 12, 64, 32, 3},
    {"the per-head gate (4 columns out)", 12, 64, 4, 3},
    {"the feed-forward widening", 12, 64, 256, 5},
    {"rank 1", 6, 64, 64, 1},
    // AND THE SHAPES THE PUBLISHED ADAPTERS ACTUALLY RUN. Every row
    // above is under BOTH matrix-core gates -- M under the 64-row tile
    // floor and rank under the 16-column one -- so they exercise the
    // portable linear+linear+add and nothing else. The x2 upscaler is
    // rank 32 over thousands of tokens, which is a different pair of
    // kernels; the assertion below pins which one each row took.
    {"x2 upscaler: attn projection", 128, 4096,  4096,  32},
    {"x2 upscaler: ff widening",     128, 4096, 16384,  32},
    {"distilled adapter: rank 450",   64, 1024,  1024, 450},
  };
  for (const Shape& sh : shapes) {
    const std::string tag = sh.what;
    const std::vector<float> x = draw(tag + "/x", (std::size_t)sh.M * sh.K, 1.0f);
    const std::vector<float> a = draw(tag + "/a", (std::size_t)sh.rank * sh.K,
                                      0.5f);
    const std::vector<float> b = draw(tag + "/b", (std::size_t)sh.N * sh.rank,
                                      0.5f);
    const std::vector<float> y0 = draw(tag + "/y", (std::size_t)sh.M * sh.N,
                                       1.0f);
    // The host answer, in f64: y + (x @ A^T) @ B^T.
    std::vector<float> want(y0);
    for (int m = 0; m < sh.M; ++m) {
      std::vector<double> r((std::size_t)sh.rank, 0.0);
      for (int j = 0; j < sh.rank; ++j) {
        double acc = 0.0;
        for (int i = 0; i < sh.K; ++i) {
          acc += (double)x[(std::size_t)m * sh.K + i] *
                 (double)a[(std::size_t)j * sh.K + i];
        }
        r[(std::size_t)j] = acc;
      }
      for (int o = 0; o < sh.N; ++o) {
        double acc = 0.0;
        for (int j = 0; j < sh.rank; ++j) {
          acc += r[(std::size_t)j] * (double)b[(std::size_t)o * sh.rank + j];
        }
        want[(std::size_t)m * sh.N + o] += (float)acc;
      }
    }
    auto xb = ops.upload_bf16(x);
    auto ab = ops.upload_bf16(a);
    auto bb = ops.upload_bf16(b);
    auto yb = ops.upload_bf16(y0);
    auto rb = ops.alloc((std::size_t)sh.M * sh.rank);
    auto db = ops.alloc((std::size_t)sh.M * sh.N);
    auto stream = mc.make_command_stream();
    {
      auto enc = stream.begin_compute();
      ops.lora_add(enc, xb, ab, bb, yb, rb, db, sh.M, sh.K, sh.N, sh.rank);
    }
    stream.commit().wait();
    const std::vector<float> got =
        ltx25::MetalOps::download_bf16(yb, (std::size_t)sh.M * sh.N);
    const double r = rel_l2(got, want);
    // Two bf16 GEMMs and an add over K up to 256: 2e-2 is generous for
    // that and still an order below any transposition error, which
    // lands at O(1).
    // WHICH PATH RAN. Without this the table below reads as coverage it
    // does not have: the matrix-core pair and the portable form compute
    // the same product, so a shape that quietly took the fallback still
    // passes every tolerance here.
    const bool mma = ops.lora_on_matrix_cores(sh.M, sh.K, sh.N, sh.rank);
    std::printf("       %-36s rel-L2 %.3e  [%s]\n", sh.what, r,
                mma ? "matrix cores" : "portable");
    check(r < 2e-2, std::string("lora_add: ") + sh.what);
    // The rank-32 rows are the upscaler's, and they must reach the
    // matrix-core pair or this file is testing the fallback twice.
    if (sh.rank >= 16 && sh.M >= 64 && mc.supports_matrix_cores()) {
      check(mma, std::string("lora_add takes the matrix cores: ") + sh.what);
    }
  }
}

// THE ADAPTER'S TWO PATHS, TIMED AGAINST EACH OTHER.
//
// The portable form writes the delta to a [M][N] scratch and then reads
// y, reads the scratch and writes y again; the matrix-core pair seeds
// the accumulator from y and stores once. So the arithmetic is the same
// and the traffic is not, which is what this measures.
//
// ARMS INTERLEAVED, one of each per round, best-of over the rounds. The
// in-tree shared/mma-tile.h records what measuring arms SEQUENTIALLY did
// to a rule like this: the first arm absorbed the first-touch cost of
// freshly allocated buffers and the two sat in different SoC power
// states, which turned a 7% regression into an apparent 1.2x win.
//
// VPIPE_LTX25_LORA_BENCH=1 to run; it is not part of the correctness
// suite and it needs matrix cores.
void
bench_lora(ltx25::MetalOps& ops, MetalCompute& mc)
{
  if (std::getenv("VPIPE_LTX25_LORA_BENCH") == nullptr) { return; }
  if (!mc.supports_matrix_cores()) {
    std::printf("  --- lora bench: no matrix cores, skipped ---\n");
    return;
  }
  std::printf("  --- lora_add: matrix cores vs portable ---\n");
  // The x2 upscaler is rank 32 and adapts 480 modules; these are its
  // per-site shapes at the DiT's inner 4096 / ff 16384, over a token
  // count in the range a real clip reaches.
  struct B { const char* what; int M, K, N, rank; };
  const B rows[] = {
      {"attn proj    8k tok", 8160, 4096,  4096,  32},
      {"ff widening  8k tok", 8160, 4096, 16384,  32},
      {"ff narrowing 8k tok", 8160, 16384, 4096,  32},
      {"attn proj   32k tok (x2 upscaled)", 32640, 4096, 4096, 32},
      {"ff widening 32k tok (x2 upscaled)", 32640, 4096, 16384, 32},
      {"rank 450     8k tok", 8160, 4096,  4096, 450},
  };
  for (const B& b : rows) {
    auto xb = ops.alloc((std::size_t)b.M * b.K);
    auto ab = ops.alloc((std::size_t)b.rank * b.K);
    auto bb = ops.alloc((std::size_t)b.N * b.rank);
    auto yb = ops.alloc((std::size_t)b.M * b.N);
    auto rb = ops.alloc((std::size_t)b.M * b.rank);
    auto db = ops.alloc((std::size_t)b.M * b.N);
    if (xb.empty() || yb.empty() || db.empty()) {
      std::printf("       %-34s allocation failed, skipped\n", b.what);
      continue;
    }
    auto once = [&](bool on) {
      ops.set_lora_matrix_cores(on);
      auto stream = mc.make_command_stream();
      {
        auto enc = stream.begin_compute();
        for (int i = 0; i < 8; ++i) {
          ops.lora_add(enc, xb, ab, bb, yb, rb, db, b.M, b.K, b.N, b.rank);
        }
      }
      const auto t0 = std::chrono::steady_clock::now();
      stream.commit().wait();
      return std::chrono::duration<double, std::milli>(
                 std::chrono::steady_clock::now() - t0).count();
    };
    once(true); once(false);                  // warm both before timing
    double mma = 1e9, port = 1e9;
    for (int round = 0; round < 5; ++round) {
      mma  = std::min(mma,  once(true));
      port = std::min(port, once(false));
    }
    ops.set_lora_matrix_cores(true);
    // The traffic the pair does not do: the scratch write, plus reading
    // it back and reading y, per adapted site.
    const double saved_mb =
        (double)((std::size_t)b.M * b.N * 2 * 3) / (1024.0 * 1024.0);
    std::printf("       %-34s mma %7.2f ms  portable %7.2f ms  %.2fx  "
                "(%.0f MB/site not moved)\n",
                b.what, mma / 8, port / 8, port / mma, saved_mb);
  }
}

}  // namespace

int
main()
{
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

  const ltx25::BlockWeights base = make_weights();
  const ltx25::RopeTable vpe = ltx25::build_index_rope(kTV, {20}, kVD, kVH);
  const ltx25::RopeTable ape = ltx25::build_index_rope(kTA, {20}, kAD, kAH);
  const ltx25::RopeTable vcpe = ltx25::build_index_rope(kTV, {20}, kAD, kAH);

  // The per-forward inputs, drawn once and reused: every arm has to see
  // the same activations or nothing below compares anything.
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

  // One CPU forward over the given weights, returning [video | audio].
  auto cpu = [&](const ltx25::BlockWeights& w) {
    ltx25::StreamInput cv, ca;
    cv.x.rows = kTV; cv.x.cols = kVD; cv.x.v = vx0;
    cv.context.rows = kTT; cv.context.cols = kVD; cv.context.v = vctx;
    cv.timesteps = vts;
    cv.cross_scale_shift = vcss;
    cv.cross_gate = vcg;
    cv.prompt_timestep = vpts;
    cv.pe = &vpe;
    cv.cross_pe = &vcpe;
    ca.x.rows = kTA; ca.x.cols = kAD; ca.x.v = ax0;
    ca.context.rows = kTT; ca.context.cols = kAD; ca.context.v = actx;
    ca.timesteps = ats;
    ca.cross_scale_shift = acss;
    ca.cross_gate = acg;
    ca.prompt_timestep = apts;
    ca.pe = &ape;
    ca.cross_pe = &ape;
    std::string e;
    std::vector<float> out;
    if (!ltx25::block_forward(w, cv, ca, &e)) {
      std::printf("       CPU reference failed: %s\n", e.c_str());
      return out;
    }
    out = cv.x.v;
    out.insert(out.end(), ca.x.v.begin(), ca.x.v.end());
    return out;
  };

  // One GPU forward over the UNMODIFIED weights, with `lora` applied at
  // runtime. `max_out` sizes the wider of the two adapter planes.
  auto gpu = [&](const ltx25::LoraBlock* lora, int rank, int max_out) {
    std::vector<float> out;
    std::string e;
    auto blk = ltx25::MetalBlock::create(ops, base, &e);
    if (!blk) { std::printf("       block: %s\n", e.c_str()); return out; }
    if (!blk->reserve(kTV, kTA, kTT, &e, 1, nullptr, rank, max_out)) {
      std::printf("       reserve: %s\n", e.c_str());
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
      if (!blk->forward(enc, gv, ga, &e, lora)) {
        std::printf("       forward: %s\n", e.c_str());
        return out;
      }
    }
    stream.commit().wait();
    out = ltx25::MetalOps::download_bf16(vx_b, (std::size_t)kTV * kVD);
    const auto a = ltx25::MetalOps::download_bf16(ax_b, (std::size_t)kTA * kAD);
    out.insert(out.end(), a.begin(), a.end());
    return out;
  };

  test_op(ops, mc);
  bench_lora(ops, mc);

  // ~40 dependent bf16 ops deep, the bar ltx25-block-metal-test states
  // for the same block against the same f32 reference. MEASURED here at
  // 2e-3 to 7e-3, so this is roughly 6x the observed spread.
  const double kBar = 4e-2;
  // The fold has to move the answer by MUCH more than that, or agreeing
  // with it proves nothing. Measured on the stream the site WRITES,
  // where the passing sites land between 0.2 and 3.
  const double kMaterial = 0.15;

  const std::vector<float> plain_cpu = cpu(base);
  const std::vector<float> plain_gpu = gpu(nullptr, 0, 0);
  if (plain_cpu.empty() || plain_gpu.empty()) {
    std::printf("FAILED to run the unadapted block\n");
    return 1;
  }
  {
    const double r = rel_l2(plain_gpu, plain_cpu);
    std::printf("  baseline (no adapter) GPU vs CPU rel-L2 %.3e\n", r);
    check(r < kBar, "the unadapted block agrees with the CPU reference");
  }

  // The half of the concatenated result a given stream wrote.
  auto half = [&](const std::vector<float>& v, char which) {
    const std::size_t nv = (std::size_t)kTV * kVD;
    return (which == 'v')
               ? std::vector<float>(v.begin(), v.begin() + nv)
               : std::vector<float>(v.begin() + nv, v.end());
  };

  const std::vector<Site> ss = sites();
  for (const Site& s : ss) {
    std::printf("  --- %s ---\n", s.name);
    ltx25::BlockWeights folded = base;
    const HostPair hp = make_lora_pair(s.name, s.base(folded));
    const std::vector<float> w_before = s.base(folded).v;
    fold(s.base(folded), hp);
    const double w_moved = rel_l2(s.base(folded).v, w_before);

    ltx25::LoraBlock lb;
    ltx25::LoraPair& lp = s.pair(lb);
    lp.a = ops.upload_bf16(hp.a);
    lp.b = ops.upload_bf16(hp.b);
    lp.rank = hp.rank;
    lp.k = hp.k;
    lp.n = hp.n;

    const std::vector<float> want = cpu(folded);
    if (want.empty()) {
      check(false, std::string(s.name) + ": CPU fold");
      continue;
    }
    const std::vector<float> got = gpu(&lb, hp.rank, hp.n);
    if (got.empty()) {
      check(false, std::string(s.name) + ": GPU forward");
      continue;
    }
    // Measured on the stream this linear WRITES, so an audio site is
    // not diluted by a video half three times its size.
    const double moved =
        rel_l2(half(want, s.stream), half(plain_cpu, s.stream));
    const double r = rel_l2(got, want);
    const double omitted = rel_l2(half(plain_gpu, s.stream),
                                  half(want, s.stream));
    std::printf("       weight moved %.2f | output moved %.4f | "
                "adapted %.3e | omitted %.4f\n", w_moved, moved, r, omitted);
    // The property under test, asserted for EVERY site including the
    // blind ones: a binding wrong enough to be visible still fails here.
    check(r < kBar, std::string(s.name) + ": runtime == folded");
    // And the two controls, which only the sites the block can see are
    // able to carry. A site declared blind that turns out NOT to be is
    // a failure too -- the list must describe the block, not excuse it.
    if (s.blind) {
      check(moved < kMaterial,
            std::string(s.name) + ": still blind (" +
                std::to_string(moved) + "); if this now moves the block, "
                "drop its `blind` flag and let it carry the controls");
    } else {
      check(moved > kMaterial,
            std::string(s.name) + ": the fold is material");
      check(omitted > kMaterial,
            std::string(s.name) + ": omitting the adapter does NOT match");
    }
  }

  // ---- every site at once ---------------------------------------------
  //
  // The isolated cases cannot catch a pair that is applied TWICE, or one
  // whose scratch collides with a neighbour's. Running them together
  // can, because a shared [tokens][rank] plane is reused by every
  // dispatch in the block.
  {
    std::printf("  --- all %zu sites together ---\n", ss.size());
    ltx25::BlockWeights folded = base;
    ltx25::LoraBlock lb;
    int max_out = 0;
    for (const Site& s : ss) {
      const HostPair hp = make_lora_pair(std::string("all/") + s.name,
                                    s.base(folded));
      fold(s.base(folded), hp);
      ltx25::LoraPair& lp = s.pair(lb);
      lp.a = ops.upload_bf16(hp.a);
      lp.b = ops.upload_bf16(hp.b);
      lp.rank = hp.rank;
      lp.k = hp.k;
      lp.n = hp.n;
      if (hp.n > max_out) { max_out = hp.n; }
    }
    const std::vector<float> want = cpu(folded);
    const std::vector<float> got = gpu(&lb, kRank, max_out);
    if (want.empty() || got.empty()) {
      check(false, "all sites: a forward failed");
    } else {
      const double r = rel_l2(got, want);
      const double moved = rel_l2(want, plain_cpu);
      std::printf("       fold moved %.3f | adapted %.3e\n", moved, r);
      check(moved > kMaterial, "all sites: the combined fold is material");
      check(r < kBar, "all sites: runtime == folded");
    }
  }

  // ---- the refusal ----------------------------------------------------
  //
  // A LoRA handed to a block whose arena was never sized for one would
  // dispatch two GEMMs into empty buffers. forward() must say so rather
  // than fault.
  {
    ltx25::LoraBlock lb;
    ltx25::LoraPair& lp = lb.video_attn1.q;
    lp.a = ops.upload_bf16(std::vector<float>((std::size_t)kRank * kVD, 0.1f));
    lp.b = ops.upload_bf16(std::vector<float>((std::size_t)kVD * kRank, 0.1f));
    lp.rank = kRank; lp.k = kVD; lp.n = kVD;
    std::string e;
    auto blk = ltx25::MetalBlock::create(ops, base, &e);
    bool refused = false;
    if (blk && blk->reserve(kTV, kTA, kTT, &e, 1, nullptr, 0, 0)) {
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
        refused = !blk->forward(enc, gv, ga, &e, &lb);
      }
    }
    check(refused, "an unsized arena REFUSES the adapter rather than "
                   "dispatching into empty buffers");
  }

  std::printf("ltx25-lora-apply: %d checks, %d failed\n", g_ran, g_fail);
  return g_fail == 0 ? 0 : 1;
}
