// The adapter on the TRUNK: the eight host adaLN chains, and the naming
// of the four projections.
//
// ltx25-lora-apply-test covers the blocks -- 10 linears x 48 -- and the
// `lora_add` op itself. Neither reaches here, because the adaLN chains
// do NOT go through that op: they run on the HOST, in f32, once per step
// per chain, so `lora_gemv_host_` is a second implementation of the same
// idea and has to be checked as one.
//
// HOW, given the adapter is bound from a file and there is no public
// constructor: this WRITES one. A synthetic .safetensors naming exactly
// the modules under test, at a rank nothing published uses, so the
// values are known and the leftover check is exercised too.
//
// THE COMPARISON, and why the first half is not filler. The chain is
// sinusoid -> linear -> SiLU -> linear -> SiLU -> linear, and a delta
// belongs inside it, before each activation -- so nothing can be checked
// by adding something to the model's answer afterwards. What this does
// instead is run the chain independently here, and:
//
//   1. with NO adapter it must equal `adaln_public` exactly. That is
//      what makes the reimplementation trustworthy -- the sinusoid, the
//      bf16 reads, the SiLU, the three widths.
//   2. with the adapter it must equal `adaln_public` again, now that
//      both sides carry the deltas.
//
// Step 1 alone proves nothing about LoRA; step 2 without step 1 proves
// nothing at all. Together they say the delta went into the right
// linear at the right point in the chain.
//
// Only the TRUNK is loaded (num_layers = 0), so this costs ~0.85 GB and
// seconds, not the 39 GB stack.

#include "ltx25-config.h"
#include "ltx25-dit.h"
#include "ltx25-dit-weights.h"
#include "ltx25-lora.h"
#include "ltx25-metal-ops.h"

#include "generative-models/weight-set.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" const unsigned char ltx25_kernels_bf16_metallib[];
extern "C" const unsigned long ltx25_kernels_bf16_metallib_len;

using vpipe::genai::WeightSet;
using vpipe::metal_compute::MetalCompute;
using vpipe::metal_compute::SharedBuffer;

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

float
bf16_to_f32(std::uint16_t h)
{
  const std::uint32_t u = (std::uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// Round-to-nearest-even, the same rounding the loader's own scaling uses.
std::uint16_t
f32_to_bf16(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  const std::uint32_t r = (u >> 16) & 1u;
  u += 0x7fffu + r;
  return (std::uint16_t)(u >> 16);
}

struct Rng {
  std::uint64_t s;
  explicit Rng(std::uint64_t seed) : s(seed * 6364136223846793005ull + 1) {}
  float next(float amp)
  {
    s = s * 6364136223846793005ull + 1442695040888963407ull;
    const std::uint32_t x = (std::uint32_t)(s >> 33);
    return amp * (2.0f * ((float)x / 4294967296.0f) - 1.0f);
  }
};

// One tensor destined for the synthetic adapter: bf16, row-major,
// [rows][cols] in PyTorch's [out][in] order.
struct Tensor {
  std::string name;
  int rows = 0, cols = 0;
  std::vector<float> v;
};

Tensor
draw(const std::string& name, int rows, int cols, float amp)
{
  Tensor t;
  t.name = name;
  t.rows = rows;
  t.cols = cols;
  t.v.resize((std::size_t)rows * cols);
  Rng r(std::hash<std::string>{}(name));
  for (auto& z : t.v) { z = r.next(amp); }
  return t;
}

// A minimal safetensors writer: 8 bytes of little-endian header length,
// the JSON header, then the bf16 payload in the order the header names
// the offsets.
bool
write_safetensors(const std::string& path, const std::vector<Tensor>& ts)
{
  std::string hdr = "{";
  std::size_t off = 0;
  for (std::size_t i = 0; i < ts.size(); ++i) {
    const std::size_t n = (std::size_t)ts[i].rows * ts[i].cols * 2;
    if (i > 0) { hdr += ","; }
    hdr += "\"" + ts[i].name + "\":{\"dtype\":\"BF16\",\"shape\":["
         + std::to_string(ts[i].rows) + "," + std::to_string(ts[i].cols)
         + "],\"data_offsets\":[" + std::to_string(off) + ","
         + std::to_string(off + n) + "]}";
    off += n;
  }
  // A metadata block, because an adapter without one is a different code
  // path in the loader and this is not the test for it.
  hdr += ",\"__metadata__\":{\"format\":\"pt\"}}";
  while (hdr.size() % 8 != 0) { hdr += " "; }
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (f == nullptr) { return false; }
  const std::uint64_t n = hdr.size();
  bool ok = std::fwrite(&n, 1, 8, f) == 8
         && std::fwrite(hdr.data(), 1, hdr.size(), f) == hdr.size();
  for (const Tensor& t : ts) {
    std::vector<std::uint16_t> b(t.v.size());
    for (std::size_t i = 0; i < t.v.size(); ++i) {
      b[i] = f32_to_bf16(t.v[i]);
    }
    const std::size_t bytes = b.size() * 2;
    ok = ok && std::fwrite(b.data(), 1, bytes, f) == bytes;
  }
  std::fclose(f);
  return ok;
}

// y[o] = sum_i x[i] * W[o][i] + b[o], W and b bf16 in UMA memory. The
// same arithmetic ltx25-dit.cc's gemv_host_ does; step 1 above is what
// holds the two together.
void
gemv(const std::vector<float>& x, const SharedBuffer& w,
     const SharedBuffer& b, int in_dim, int out_dim, std::vector<float>& y)
{
  const auto* wp = static_cast<const std::uint16_t*>(w.contents());
  const auto* bp = b.empty() ? nullptr
                             : static_cast<const std::uint16_t*>(b.contents());
  y.assign((std::size_t)out_dim, 0.0f);
  for (int o = 0; o < out_dim; ++o) {
    const std::uint16_t* row = wp + (std::size_t)o * in_dim;
    double acc = bp != nullptr ? (double)bf16_to_f32(bp[o]) : 0.0;
    for (int i = 0; i < in_dim; ++i) {
      acc += (double)x[(std::size_t)i] * bf16_to_f32(row[i]);
    }
    y[(std::size_t)o] = (float)acc;
  }
}

// y += B @ (A @ x), from the HOST-side tensors the adapter file was
// written from -- deliberately not from the adapter's own buffers, so a
// loader that mangled them on the way in shows up here.
void
lora(const std::vector<float>& x, const Tensor& a, const Tensor& b,
     std::vector<float>& y)
{
  std::vector<double> r((std::size_t)a.rows, 0.0);
  for (int j = 0; j < a.rows; ++j) {
    double acc = 0.0;
    for (int i = 0; i < a.cols; ++i) {
      acc += (double)x[(std::size_t)i]
           * (double)bf16_to_f32(f32_to_bf16(a.v[(std::size_t)j * a.cols + i]));
    }
    r[(std::size_t)j] = acc;
  }
  for (int o = 0; o < b.rows; ++o) {
    double acc = 0.0;
    for (int j = 0; j < b.cols; ++j) {
      acc += r[(std::size_t)j]
           * (double)bf16_to_f32(f32_to_bf16(b.v[(std::size_t)o * b.cols + j]));
    }
    y[(std::size_t)o] += (float)acc;
  }
}

void
silu(std::vector<float>& v)
{
  for (auto& z : v) { z = z / (1.0f + std::exp(-z)); }
}

std::vector<float>
sinusoid(double t)
{
  const int half = 128;
  std::vector<float> sin_part((std::size_t)half), cos_part((std::size_t)half);
  const double log_period = std::log(10000.0);
  for (int i = 0; i < half; ++i) {
    const double a = t * std::exp(-log_period * (double)i / (double)half);
    sin_part[(std::size_t)i] = (float)std::sin(a);
    cos_part[(std::size_t)i] = (float)std::cos(a);
  }
  std::vector<float> out;
  out.reserve(256);
  out.insert(out.end(), cos_part.begin(), cos_part.end());
  out.insert(out.end(), sin_part.begin(), sin_part.end());
  return out;
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

}  // namespace

int
main()
{
  const char* root = std::getenv("VPIPE_LTX25_TEST_MODEL_PATH");
  if (root == nullptr) {
    std::printf("SKIPPED: needs VPIPE_LTX25_TEST_MODEL_PATH. NOTHING was "
                "checked.\n");
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
  // The TRUNK only. The 48 blocks are 39 GB and none of them is under
  // test here.
  ltx25::Config small = cfg;
  small.dit.num_layers = 0;
  auto dit = ltx25::Ltx25Dit::load(small, ws, ops, /*stream_blocks=*/false,
                                   0, 0, 0, &err);
  if (!dit) {
    std::printf("FAILED to load the trunk: %s\n", err.c_str());
    return 1;
  }
  const ltx25::DitTrunk::AdaLN& a = dit->trunk().video;
  check(a.valid && a.dim > 0 && a.out_dim > 0,
        "the video adaLN chain is bound (dim " + std::to_string(a.dim) +
        " -> " + std::to_string(a.out_dim) + ")");
  if (!a.valid) { std::printf("FAILURES\n"); return 1; }

  const int kRank = 3;
  const double kSigma = 0.421875 * small.dit.timestep_scale_multiplier;

  // ---- the synthetic adapter -------------------------------------------
  //
  // The video adaLN chain's three linears, plus the two VIDEO trunk
  // projections. The projections are named but not numerically checked
  // here: they go through MetalOps::lora_add, which ltx25-lora-apply-test
  // pins at these very shapes. What is checked is that the loader BINDS
  // them at the right widths -- a name that resolved to nothing would
  // leave the pair invalid, and a wrong width is refused outright.
  const std::string pre = "diffusion_model.adaln_single.";
  const std::string emb = pre + "emb.timestep_embedder.";
  std::vector<Tensor> ts = {
    // Scaled per linear so the whole chain is perturbed by roughly its
    // own magnitude rather than swamped: the driver has to MOVE for the
    // comparison to mean anything, but a delta 600x the weight would
    // make the two SiLUs saturate and hide everything upstream of them.
    draw(emb + "linear_1.lora_A.weight", kRank, 256, 0.04f),
    draw(emb + "linear_1.lora_B.weight", a.dim, kRank, 0.10f),
    draw(emb + "linear_2.lora_A.weight", kRank, a.dim, 0.004f),
    draw(emb + "linear_2.lora_B.weight", a.dim, kRank, 0.10f),
    draw(pre + "linear.lora_A.weight", kRank, a.dim, 0.004f),
    draw(pre + "linear.lora_B.weight", a.out_dim, kRank, 0.10f),
    draw("diffusion_model.patchify_proj.lora_A.weight", kRank,
         small.dit.in_channels, 0.05f),
    draw("diffusion_model.patchify_proj.lora_B.weight", a.dim, kRank, 0.5f),
    draw("diffusion_model.proj_out.lora_A.weight", kRank, a.dim, 0.01f),
    draw("diffusion_model.proj_out.lora_B.weight", small.dit.out_channels,
         kRank, 0.5f),
  };
  const char* tmp = std::getenv("TMPDIR");
  const std::string file = std::string(tmp != nullptr ? tmp : "/tmp")
                         + "/ltx25-synthetic-lora.safetensors";
  if (!write_safetensors(file, ts)) {
    check(false, "write the synthetic adapter to " + file);
    std::printf("FAILURES\n");
    return 1;
  }
  check(true, "wrote a synthetic rank-" + std::to_string(kRank) +
        " adapter naming 5 trunk modules");

  auto aws = WeightSet::open(file, nullptr);
  if (!aws) {
    check(false, "open the synthetic adapter");
    std::printf("FAILURES\n");
    return 1;
  }
  std::vector<std::string> unmapped;
  std::string lerr;
  auto adapter = ltx25::LoraAdapter::load(*aws, &mc, small.dit, 1.0, file,
                                          &unmapped, &lerr);
  if (!adapter) {
    check(false, "load the synthetic adapter: " + lerr);
    for (std::size_t i = 0; i < unmapped.size() && i < 4; ++i) {
      std::printf("        unmapped: %s\n", unmapped[i].c_str());
    }
    std::printf("FAILURES\n");
    return 1;
  }
  check(adapter->modules() == 5,
        "5 modules bound (got " + std::to_string(adapter->modules()) + ")");
  const ltx25::LoraTrunk* lt = adapter->trunk();
  check(lt != nullptr && lt->video.emb1.valid() && lt->video.emb2.valid()
            && lt->video.out.valid(),
        "the video adaLN chain's three linears bound");
  check(lt != nullptr && lt->patchify.valid() && lt->proj_out.valid(),
        "and both video trunk projections -- the names resolve");
  if (lt != nullptr) {
    check(lt->video.out.n == a.out_dim && lt->video.out.k == a.dim,
          "the chain's OUTPUT linear is bound at " +
          std::to_string(a.dim) + " -> " + std::to_string(a.out_dim) +
          " (got " + std::to_string(lt->video.out.k) + " -> " +
          std::to_string(lt->video.out.n) + ")");
    check(!lt->audio.emb1.valid() && !lt->prompt.emb1.valid(),
          "and NO other chain -- the adapter named none of them");
  }

  // ---- step 1: the chain, reimplemented, with no adapter ---------------
  auto host_chain = [&](bool with_lora) {
    std::vector<float> x = sinusoid(kSigma);
    std::vector<float> h1, h2, out;
    gemv(x, a.emb1_w, a.emb1_b, 256, a.dim, h1);
    if (with_lora) { lora(x, ts[0], ts[1], h1); }
    silu(h1);
    gemv(h1, a.emb2_w, a.emb2_b, a.dim, a.dim, h2);
    if (with_lora) { lora(h1, ts[2], ts[3], h2); }
    std::vector<float> act = h2;
    silu(act);
    gemv(act, a.out_w, a.out_b, a.dim, a.out_dim, out);
    if (with_lora) { lora(act, ts[4], ts[5], out); }
    return out;
  };

  const std::vector<float> plain = dit->adaln_public(a, kSigma, nullptr);
  const std::vector<float> plain_host = host_chain(false);
  const double r0 = rel_l2(plain_host, plain);
  std::printf("       reimplemented chain vs adaln_public  rel-L2 %.3e\n", r0);
  check(r0 < 1e-6,
        "the chain reimplemented here matches the model's own -- which is "
        "what makes the comparison below mean anything");

  // ---- step 2: the same, with the adapter ------------------------------
  std::string serr;
  if (!dit->set_lora(std::shared_ptr<const ltx25::LoraAdapter>(
                         std::move(adapter)), &serr)) {
    check(false, "set_lora: " + serr);
    std::printf("FAILURES\n");
    return 1;
  }
  const std::vector<float> got = dit->adaln_public(a, kSigma, nullptr);
  const std::vector<float> want = host_chain(true);
  const double r1 = rel_l2(got, want);
  // The delta must MOVE the driver, or agreeing with it says nothing.
  const double moved = rel_l2(want, plain);
  std::printf("       adapted: driver moved %.3f, vs the host chain "
              "rel-L2 %.3e\n", moved, r1);
  check(moved > 0.10,
        "the adapter moves the modulation driver materially (" +
        std::to_string(moved) + ")");
  check(r1 < 1e-5,
        "and the model's adapted chain matches the host's, so each delta "
        "went into the right linear BEFORE its activation");
  // The control that fails if the adapter never arrived.
  check(rel_l2(got, plain) > 0.10,
        "the adapted driver differs from the unadapted one -- the adapter "
        "reached the chain at all");

  // ---- the ordering rule -----------------------------------------------
  //
  // After the bake the adaLN projections are released, so an adapter
  // arriving then would reach the blocks and NOT the modulation. It has
  // to be refused rather than half-applied.
  {
    ltx25::RopeGeometry g;
    g.scale_t = 8; g.scale_h = 32; g.scale_w = 32; g.fps = 24.0;
    g.causal_fix = true;
    std::string gerr, berr;
    if (!dit->set_geometry(1, 1, 1, 0, 1, g, &gerr)) {
      std::printf("       (set_geometry declined: %s)\n", gerr.c_str());
    }
    if (dit->bake_adaln({1.0, 0.5}, &berr)) {
      std::string s2;
      check(!dit->set_lora(nullptr, &s2),
            "after bake_adaln, CHANGING the adapter is refused rather than "
            "applied to half the model");
    } else {
      std::printf("       (bake_adaln declined: %s -- ordering rule not "
                  "exercised)\n", berr.c_str());
    }
  }

  std::remove(file.c_str());
  std::printf("ltx25-lora-trunk: %d checks, %d failed\n", g_ran, g_fail);
  return g_fail == 0 ? 0 : 1;
}
