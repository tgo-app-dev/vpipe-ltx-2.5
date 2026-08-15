// The plugin's own Metal kernels, on the GPU, against the CPU reference.
//
// These are the two operations libvpipe does not already provide, and
// both are ones a wrong implementation runs cleanly:
//
//   ltx_rope_half_perhead  the half rotation over a PER-HEAD table, on
//                          token-major activations. libvpipe's
//                          rope_half_table_ftab does the same arithmetic
//                          against a table SHARED by all heads, so using
//                          it would give every head head 0's angles.
//   ltx_gate_heads         2*sigmoid per head, from the attention's
//                          INPUT, applied before to_out.
//
// plus ltx_rms_norm_gain, which exists as its own entry point so the row
// width is stated at the call site (LTX norms q/k over the WHOLE
// projection width, not per head).
//
// The bar is bf16 round-off, not f32: the DiT runs bf16 and these
// kernels are its bf16 twin, so ~3 decimal digits is all there is. A
// tight f32 bar here would be checking the wrong thing.

#include "ltx25-config.h"
#include "ltx25-rope.h"

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// The embedded metallib this test registers itself. A test binary is not
// a plugin, so nothing has registered it for us.
extern "C" const unsigned char ltx25_kernels_bf16_metallib[];
extern "C" const unsigned long ltx25_kernels_bf16_metallib_len;

using vpipe::metal_compute::ComputeFunction;
using vpipe::metal_compute::ComputeLibrary;
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

// bf16 is the top 16 bits of an f32, round-to-nearest-even.
std::uint16_t
to_bf16(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  const std::uint32_t r = ((u >> 16) & 1u) + 0x7fffu;
  return (std::uint16_t)((u + r) >> 16);
}

float
from_bf16(std::uint16_t h)
{
  const std::uint32_t u = (std::uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

double
rel_l2(const std::vector<float>& got, const std::vector<float>& want)
{
  if (got.size() != want.size() || want.empty()) { return 1e30; }
  double n = 0.0, d = 0.0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    const double e = (double)got[i] - (double)want[i];
    n += e * e;
    d += (double)want[i] * (double)want[i];
  }
  return d == 0.0 ? (n == 0.0 ? 0.0 : 1e30) : std::sqrt(n / d);
}

SharedBuffer
upload_bf16(const MetalCompute& mc, const std::vector<float>& v)
{
  SharedBuffer b = mc.make_shared_buffer(v.size() * 2);
  auto* p = static_cast<std::uint16_t*>(b.contents());
  for (std::size_t i = 0; i < v.size(); ++i) { p[i] = to_bf16(v[i]); }
  return b;
}

std::vector<float>
download_bf16(const SharedBuffer& b, std::size_t n)
{
  const auto* p = static_cast<const std::uint16_t*>(b.contents());
  std::vector<float> v(n);
  for (std::size_t i = 0; i < n; ++i) { v[i] = from_bf16(p[i]); }
  return v;
}

SharedBuffer
upload_f32(const MetalCompute& mc, const std::vector<float>& v)
{
  SharedBuffer b = mc.make_shared_buffer(v.size() * 4);
  std::memcpy(b.contents(), v.data(), v.size() * 4);
  return b;
}

// bf16 has 8 mantissa bits, so a value round-trips to ~2^-8 relative.
// Anything at or below this is the dtype, not the kernel.
constexpr double kBf16Bar = 6e-3;

void
test_rope(MetalCompute& mc, const ComputeLibrary& lib)
{
  std::printf("ltx_rope_half_perhead (per-head table, token-major)\n");
  const ComputeFunction fn = lib.function("ltx_rope_half_perhead");
  if (!fn.valid()) { check(false, "resolve ltx_rope_half_perhead"); return; }

  // The real video geometry in miniature: 2 latent frames x 3 x 4 cells.
  const int F = 2, H_ = 3, W = 4;
  const int tokens = F * H_ * W;
  const int heads = 32, head_dim = 128, inner = heads * head_dim;
  const ltx25::RopeTable t =
      ltx25::build_video_rope(F, H_, W, {20, 2048, 2048}, inner, heads, ltx25::RopeGeometry::identity());

  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> x((std::size_t)tokens * inner);
  for (auto& v : x) { v = nd(rng); }

  // The CPU reference runs on the SAME bf16 values the GPU sees, so the
  // comparison isolates the kernel rather than measuring the narrowing
  // twice.
  std::vector<float> host = x;
  for (auto& v : host) { v = from_bf16(to_bf16(v)); }
  std::vector<float> want = host;
  ltx25::apply_rope(t, want.data(), tokens, heads, head_dim);

  SharedBuffer xb = upload_bf16(mc, host);
  SharedBuffer cb = upload_f32(mc, t.cos);
  SharedBuffer sb = upload_f32(mc, t.sin);

  auto stream = mc.make_command_stream();
  {
    auto enc = stream.begin_compute();
    enc.set_function(fn);
    enc.set_buffer(0, xb);
    enc.set_buffer(1, cb);
    enc.set_buffer(2, sb);
    enc.set_constant(3, heads);
    enc.set_constant(4, tokens);
    enc.set_constant(5, head_dim);
    enc.dispatch({(unsigned)(head_dim / 2), (unsigned)tokens,
                  (unsigned)heads}, {64, 1, 1});
  }
  stream.commit().wait();

  const std::vector<float> got = download_bf16(xb, x.size());
  const double r = rel_l2(got, want);
  std::printf("       rel-L2 %.3e (bf16 bar %.0e)\n", r, kBf16Bar);
  check(r < kBf16Bar, "matches the CPU reference");

  // The point of the custom kernel: heads must NOT share a table. If they
  // did, every head would equal head 0 -- so a shared-table kernel passes
  // any test where the table happens to be uniform, and this one is not.
  bool per_head_differs = false;
  for (int j = 0; j < head_dim / 2 && !per_head_differs; ++j) {
    if (t.cos[(std::size_t)j] !=
        t.cos[((std::size_t)1 * tokens) * (head_dim / 2) + j]) {
      per_head_differs = true;
    }
  }
  check(per_head_differs,
        "the table really does differ per head (else this proves nothing)");
}

void
test_gate(MetalCompute& mc, const ComputeLibrary& lib)
{
  std::printf("ltx_gate_heads (2*sigmoid per head)\n");
  const ComputeFunction fn = lib.function("ltx_gate_heads");
  if (!fn.valid()) { check(false, "resolve ltx_gate_heads"); return; }

  const int tokens = 24, heads = 32, head_dim = 128;
  std::mt19937 rng(11);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> y((std::size_t)tokens * heads * head_dim);
  std::vector<float> lg((std::size_t)tokens * heads);
  for (auto& v : y)  { v = from_bf16(to_bf16(nd(rng))); }
  for (auto& v : lg) { v = from_bf16(to_bf16(nd(rng))); }

  std::vector<float> want = y;
  for (int t = 0; t < tokens; ++t) {
    for (int h = 0; h < heads; ++h) {
      const float g = 2.0f / (1.0f + std::exp(-lg[(std::size_t)t * heads + h]));
      for (int d = 0; d < head_dim; ++d) {
        want[((std::size_t)t * heads + h) * head_dim + d] *= g;
      }
    }
  }

  SharedBuffer yb = upload_bf16(mc, y);
  SharedBuffer lb = upload_bf16(mc, lg);
  auto stream = mc.make_command_stream();
  {
    auto enc = stream.begin_compute();
    enc.set_function(fn);
    enc.set_buffer(0, yb);
    enc.set_buffer(1, lb);
    enc.set_constant(2, heads);
    enc.set_constant(3, tokens);
    enc.set_constant(4, head_dim);
    enc.dispatch({(unsigned)head_dim, (unsigned)tokens, (unsigned)heads},
                 {64, 1, 1});
  }
  stream.commit().wait();

  const std::vector<float> got = download_bf16(yb, y.size());
  const double r = rel_l2(got, want);
  std::printf("       rel-L2 %.3e\n", r);
  check(r < kBf16Bar, "matches the CPU reference");

  // The factor of two is load-bearing: sigmoid alone halves every head on
  // average, which reads as a model that lost confidence rather than a
  // bug. Check the mean gain is ~1, not ~0.5.
  double sg = 0.0, sw = 0.0;
  for (std::size_t i = 0; i < y.size(); ++i) {
    sg += std::fabs((double)got[i]);
    sw += std::fabs((double)y[i]);
  }
  check(sg / sw > 0.8, "mean |gain| is ~1, not ~0.5 (the factor 2 is there)");
}

void
test_rms(MetalCompute& mc, const ComputeLibrary& lib)
{
  std::printf("ltx_rms_norm_gain (over the WHOLE projection width)\n");
  const ComputeFunction fn = lib.function("ltx_rms_norm_gain");
  if (!fn.valid()) { check(false, "resolve ltx_rms_norm_gain"); return; }

  const int rows = 24, n = 4096;
  std::mt19937 rng(3);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> x((std::size_t)rows * n), gain((std::size_t)n);
  for (auto& v : x)    { v = from_bf16(to_bf16(nd(rng))); }
  for (auto& v : gain) { v = from_bf16(to_bf16(1.0f + 0.1f * nd(rng))); }

  std::vector<float> want = x;
  for (int r = 0; r < rows; ++r) {
    float* row = want.data() + (std::size_t)r * n;
    double ss = 0.0;
    for (int i = 0; i < n; ++i) { ss += (double)row[i] * row[i]; }
    const double inv = 1.0 / std::sqrt(ss / n + 1e-6);
    for (int i = 0; i < n; ++i) { row[i] = (float)(row[i] * inv * gain[i]); }
  }

  SharedBuffer xb = upload_bf16(mc, x);
  SharedBuffer gb = upload_bf16(mc, gain);
  auto stream = mc.make_command_stream();
  {
    auto enc = stream.begin_compute();
    enc.set_function(fn);
    enc.set_buffer(0, xb);
    enc.set_buffer(1, gb);
    enc.set_constant(2, n);
    enc.set_constant(3, 1.0e-6f);
    enc.dispatch({(unsigned)(rows * 256), 1, 1}, {256, 1, 1});
  }
  stream.commit().wait();

  const std::vector<float> got = download_bf16(xb, x.size());
  const double r = rel_l2(got, want);
  std::printf("       rel-L2 %.3e\n", r);
  check(r < kBf16Bar, "matches the CPU reference");
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
  // Register the plugin's metallib the way the plugin does at load. A
  // test binary is not a plugin, so this is the same public API the
  // plugin context forwards to.
  if (!mc.register_metal_library(ltx25::kMetalLibBf16,
                                 ltx25_kernels_bf16_metallib,
                                 ltx25_kernels_bf16_metallib_len)) {
    std::printf("FAILED to register the metallib\n");
    return 1;
  }
  const ComputeLibrary lib = mc.load_library(ltx25::kMetalLibBf16);
  if (!lib.valid()) {
    std::printf("FAILED to load the metallib after registering it\n");
    return 1;
  }
  std::printf("registered + loaded '%s'\n", ltx25::kMetalLibBf16);

  test_rope(mc, lib);
  test_gate(mc, lib);
  test_rms(mc, lib);

  std::printf("%s\n", g_fail == 0 ? "ALL PASSED" : "FAILURES");
  return g_fail == 0 ? 0 : 1;
}
