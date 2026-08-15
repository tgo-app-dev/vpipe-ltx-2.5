// The ancestral Euler step against the reference.
//
// LTX-2.5 samples stage 1 ancestrally (eta = 1, s_noise = 1), not with
// the deterministic res_2s stepper the reference defaults to -- see
// ltx25-sampler.h. This pins the arithmetic with FIXED x / denoised /
// noise, so it checks the step and not an RNG that legitimately differs
// between implementations.

#include "ltx25-sampler.h"
#include "npy.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

int g_fail = 0;

void
check(bool ok, const std::string& what)
{
  std::printf("  [%s] %s\n", ok ? " OK " : "FAIL", what.c_str());
  if (!ok) { ++g_fail; }
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
  const std::string d(dir);
  npy::Array sig = npy::load(d + "/samp_sigmas.npy");
  npy::Array x   = npy::load(d + "/samp_x.npy");
  npy::Array den = npy::load(d + "/samp_denoised.npy");
  npy::Array noi = npy::load(d + "/samp_noise.npy");
  npy::Array out = npy::load(d + "/samp_out.npy");
  npy::Array e0  = npy::load(d + "/samp_out_eta0.npy");
  if (!sig.ok || !x.ok || !den.ok || !noi.ok || !out.ok || !e0.ok) {
    check(false, "load goldens: " + sig.err + x.err + den.err + noi.err +
          out.err + e0.err);
    std::printf("FAILURES\n");
    return 1;
  }
  const std::size_t n = x.data.size();
  const int steps = (int)sig.data.size() - 1;
  check((std::size_t)steps * n == out.data.size(),
        "the golden has one row per step index");

  double worst = 0.0;
  for (int i = 0; i < steps; ++i) {
    std::vector<float> got(n);
    ltx25::euler_ancestral_step(x.data.data(), den.data.data(),
                                noi.data.data(), sig.data[(std::size_t)i],
                                sig.data[(std::size_t)i + 1],
                                ltx25::kAncestralEta, ltx25::kAncestralSNoise,
                                got.data(), n);
    const std::vector<float> want(out.data.begin() + (std::size_t)i * n,
                                  out.data.begin() + (std::size_t)(i + 1) * n);
    const double r = npy::rel_l2(got, want);
    if (r > worst) { worst = r; }
    std::printf("       step %d  sigma %.6f -> %.6f   rel-L2 %.2e\n", i,
                sig.data[(std::size_t)i], sig.data[(std::size_t)i + 1], r);
  }
  // Pure f32 arithmetic on both sides: the bar is round-off, not a
  // tolerance.
  check(worst < 1e-6, "every ancestral step matches to f32 round-off");

  // The LAST step's sigma_next is 0, which must return the denoised
  // prediction unchanged. That is how a run ends, and getting it wrong
  // adds a final burst of noise to the finished frame.
  {
    std::vector<float> got(n);
    ltx25::euler_ancestral_step(x.data.data(), den.data.data(),
                                noi.data.data(), sig.data[(std::size_t)steps - 1],
                                0.0, 1.0, 1.0, got.data(), n);
    check(npy::rel_l2(got, den.data) == 0.0,
          "sigma_next == 0 returns the denoised sample exactly");
  }

  // eta = 0 must be a plain Euler step and must not need noise at all.
  {
    std::vector<float> got(n);
    ltx25::euler_ancestral_step(x.data.data(), den.data.data(), nullptr,
                                sig.data[0], sig.data[1], 0.0, 1.0,
                                got.data(), n);
    const double r = npy::rel_l2(got, e0.data);
    std::printf("       eta=0 (plain Euler, no noise)   rel-L2 %.2e\n", r);
    check(r < 1e-6, "eta=0 matches the reference with no noise supplied");
  }

  // to_denoised is the velocity -> x0 conversion every step consumes.
  {
    std::vector<float> got(n), want(n);
    for (std::size_t i = 0; i < n; ++i) {
      want[i] = (float)((double)x.data[i] - (double)den.data[i] * 0.725);
    }
    ltx25::to_denoised(x.data.data(), den.data.data(), 0.725, got.data(), n);
    check(npy::rel_l2(got, want) < 1e-7, "to_denoised is x - v*sigma");
  }

  // The noise draw is reproducible and roughly standard normal -- not a
  // match to torch (see the header), but a run must repeat itself.
  {
    std::vector<float> a(4096), b(4096);
    ltx25::fill_normal(a, 42);
    ltx25::fill_normal(b, 42);
    check(npy::rel_l2(a, b) == 0.0, "the same seed gives the same noise");
    double m = 0.0, v = 0.0;
    for (float f : a) { m += f; }
    m /= (double)a.size();
    for (float f : a) { v += ((double)f - m) * ((double)f - m); }
    v /= (double)a.size();
    std::printf("       noise mean %.4f, variance %.4f\n", m, v);
    check(std::fabs(m) < 0.05 && v > 0.9 && v < 1.1,
          "the noise is standard normal");
  }

  std::printf("%s\n", g_fail == 0 ? "ALL PASSED" : "FAILURES");
  return g_fail == 0 ? 0 : 1;
}
