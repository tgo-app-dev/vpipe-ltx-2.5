// Steel flash attention against the scalar kernel it replaces.
//
// The DiT's goldens cover this too, but they need the 103 GB checkpoint,
// and what they cannot isolate is the ONE thing about the steel contract
// that is LTX-specific: this is the first model in the family whose
// attention is not square. Text cross-attention, a2v and v2a all attend
// a sequence that is not their own length, so `qL` and `kL` genuinely
// differ -- in the K/V strides, in NK/NK_aligned, and in function
// constant 201, which comes from the KEY length.
//
// Every sibling port sets kL = qL because it never has to. A port that
// copies that would still pass every SQUARE case, which is exactly why
// the cases below are mostly non-square, and why the tail lengths are
// chosen to fall off both the query tile (32) and the key tile (16).
//
// The bar is bf16 round-off. The two kernels do the same arithmetic in a
// different order -- steel keeps an online softmax in f32 registers,
// sdpa_full accumulates per query row -- so bit-identity is not on offer
// and would be the wrong thing to assert.

#include "ltx25-config.h"
#include "ltx25-metal-ops.h"

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/metal-compute.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

extern "C" const unsigned char ltx25_kernels_bf16_metallib[];
extern "C" const unsigned long ltx25_kernels_bf16_metallib_len;

using vpipe::metal_compute::MetalCompute;
using vpipe::metal_compute::SharedBuffer;

namespace {

int g_fail = 0;

void
check(bool ok, const std::string& what)
{
  std::printf("  [%s] %s\n", ok ? " OK " : "FAIL", what.c_str());
  if (!ok) { ++g_fail; }
}

double
rel_l2(const std::vector<float>& a, const std::vector<float>& b)
{
  double num = 0, den = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = (double)a[i] - (double)b[i];
    num += d * d;
    den += (double)b[i] * (double)b[i];
  }
  return std::sqrt(num / (den + 1e-30));
}

// One shape, both kernels, compared.
void
one(const ltx25::MetalOps& ops, int heads, int tq, int tkv, int head_dim,
    std::uint64_t seed)
{
  const std::string label =
      "H" + std::to_string(heads) + " D" + std::to_string(head_dim) +
      "  q " + std::to_string(tq) + " x kv " + std::to_string(tkv);

  std::mt19937_64 rng(seed);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  auto fill = [&](std::size_t n) {
    std::vector<float> v(n);
    for (float& x : v) { x = nd(rng); }
    return v;
  };
  const std::size_t nq = (std::size_t)heads * tq * head_dim;
  const std::size_t nk = (std::size_t)heads * tkv * head_dim;

  const SharedBuffer q = ops.upload_bf16(fill(nq));
  const SharedBuffer k = ops.upload_bf16(fill(nk));
  const SharedBuffer v = ops.upload_bf16(fill(nk));
  const SharedBuffer o_steel = ops.alloc(nq);
  const SharedBuffer o_scalar = ops.alloc(nq);

  ltx25::MetalOps::SteelAttn plan;
  if (!ops.steel_attn_plan(&plan, heads, tq, tkv, head_dim)) {
    check(false, label + ": steel plan");
    return;
  }

  auto stream = ops.mc()->make_command_stream();
  {
    auto enc = stream.begin_compute();
    ops.sdpa_steel(enc, plan, q, k, v, o_steel);
    ops.sdpa_full(enc, q, k, v, o_scalar, heads, tq, tkv, head_dim);
  }
  stream.commit().wait();

  const std::vector<float> a = ltx25::MetalOps::download_bf16(o_steel, nq);
  const std::vector<float> b = ltx25::MetalOps::download_bf16(o_scalar, nq);

  bool finite = true;
  for (float x : a) {
    if (!std::isfinite(x)) { finite = false; break; }
  }
  const double e = rel_l2(a, b);
  std::printf("       %-28s rel-L2 %.3e\n", label.c_str(), e);
  check(finite && e < 8e-3, label);
}

}  // namespace

int
main()
{
  MetalCompute mc(nullptr);
  if (!mc.valid()) {
    std::printf("SKIPPED: no Metal device. NOTHING was checked.\n");
    return 0;
  }
  // MetalOps::init insists on the plugin's own metallib, and a test
  // binary is not a plugin, so nothing has registered it for us.
  if (!mc.register_metal_library(ltx25::kMetalLibBf16,
                                 ltx25_kernels_bf16_metallib,
                                 ltx25_kernels_bf16_metallib_len)) {
    std::printf("FAILED to register the metallib\n");
    return 1;
  }
  ltx25::MetalOps ops;
  std::string err;
  if (!ops.init(&mc, &err)) {
    std::printf("FAILED: %s\n", err.c_str());
    return 1;
  }
  std::printf("attention kernel: %s\n", ops.attn_kernel());
  if (!ops.steel_attn_available(128)) {
    std::printf("SKIPPED: steel attention is off (no library, or "
                "VPIPE_LTX25_NO_STEEL_ATTN). NOTHING was checked.\n");
    return 0;
  }

  // head_dim 128 -- the video stream and the text cross-attention.
  std::printf("head_dim 128\n");
  one(ops, 4, 128, 128, 128, 1);      // square, both lengths aligned
  one(ops, 4, 120, 128, 128, 2);      // query tail only
  one(ops, 4, 128, 100, 128, 3);      // key tail only
  one(ops, 4,  96, 259, 128, 4);      // cross: kv >> q, key tail
  one(ops, 4, 259,  96, 128, 5);      // cross the other way
  one(ops, 2,   7,  13, 128, 6);      // shorter than one tile either way

  // head_dim 64 -- the audio stream and both audio<->video directions.
  std::printf("head_dim 64\n");
  one(ops, 4, 128, 128, 64, 11);
  one(ops, 4, 120, 100, 64, 12);      // both tails at once
  one(ops, 4,  64, 301, 64, 13);      // a2v: audio queries, video keys
  one(ops, 4, 301,  64, 64, 14);      // v2a: video queries, audio keys
  one(ops, 2,   5,   9, 64, 15);

  std::printf("%s\n", g_fail == 0 ? "ALL PASSED" : "FAILURES");
  return g_fail == 0 ? 0 : 1;
}
