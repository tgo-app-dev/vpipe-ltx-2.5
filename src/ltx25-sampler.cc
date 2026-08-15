#include "ltx25-sampler.h"

#include <cmath>
#include <cstdint>
#include <vector>

namespace ltx25 {

void
to_denoised(const float* x, const float* velocity, double sigma, float* out,
            std::size_t n)
{
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = (float)((double)x[i] - (double)velocity[i] * sigma);
  }
}

void
to_denoised_masked(const float* x, const float* velocity, double sigma,
                   const float* mask, int channels, int tokens, float* out)
{
  for (int c = 0; c < channels; ++c) {
    const std::size_t base = (std::size_t)c * tokens;
    for (int t = 0; t < tokens; ++t) {
      const double s = sigma * (double)mask[t];
      out[base + t] = (float)((double)x[base + t] -
                              (double)velocity[base + t] * s);
    }
  }
}

void
hold_conditioned(float* x, const float* clean, const float* mask,
                 int channels, int tokens)
{
  for (int c = 0; c < channels; ++c) {
    const std::size_t base = (std::size_t)c * tokens;
    for (int t = 0; t < tokens; ++t) {
      const float m = mask[t];
      if (m == 1.0f) { continue; }
      x[base + t] = x[base + t] * m + clean[base + t] * (1.0f - m);
    }
  }
}

void
euler_ancestral_step(const float* x, const float* denoised, const float* noise,
                     double sigma, double sigma_next, double eta,
                     double s_noise, float* out, std::size_t n)
{
  // The schedule's last entry is 0, so this is how a run ENDS -- the
  // denoised prediction is the answer, with nothing left to renoise.
  if (sigma_next == 0.0 || sigma == 0.0) {
    for (std::size_t i = 0; i < n; ++i) { out[i] = denoised[i]; }
    return;
  }

  // The scalar coefficients are computed in FLOAT, not double, because
  // that is where the reference computes them (`sigmas[i].to(float32)`)
  // -- and near sigma = 1 it matters more than it looks. eta = 1 gives
  // sigma_down = sigma_next^2 / sigma, so `alpha_down = 1 - sigma_down`
  // is ~0.019 at the schedule's second step, and the renoise coefficient
  // is a difference of two nearly-equal terms divided by it. Computing
  // that in double is MORE accurate and differs from the reference by
  // ~1e-6 relative; matching it is the point.
  const float sigf = (float)sigma, sig_nf = (float)sigma_next;
  const float downstep_ratio = 1.0f + (sig_nf / sigf - 1.0f) * (float)eta;
  const float sigma_down = sig_nf * downstep_ratio;
  const float r = sigma_down / sigf;

  if (eta <= 0.0 || noise == nullptr) {
    // A plain Euler step: interpolate between x and x0. `noise` is
    // allowed to be null only here, which is why eta is checked first.
    for (std::size_t i = 0; i < n; ++i) {
      out[i] = r * x[i] + (1.0f - r) * denoised[i];
    }
    return;
  }

  // Renoise from sigma_down back up to sigma_next, rescaling the signal
  // by alpha_next/alpha_down so the transition is variance-preserving.
  // alpha = 1 - sigma is the RECTIFIED-FLOW parameterisation; the
  // DDIM/variance-exploding coefficients give a different sigma_down and
  // a different amount of noise for the same eta, and agree only at
  // eta = 0.
  const float alpha_next = 1.0f - sig_nf;
  const float alpha_down = 1.0f - sigma_down;
  const float ratio = alpha_next / alpha_down;
  float c2 = sig_nf * sig_nf - sigma_down * sigma_down * ratio * ratio;
  if (c2 < 0.0f) { c2 = 0.0f; }      // clamp(min=0), as the reference does
  const float c = std::sqrt(c2);

  for (std::size_t i = 0; i < n; ++i) {
    const float step = r * x[i] + (1.0f - r) * denoised[i];
    out[i] = ratio * step + noise[i] * (float)s_noise * c;
  }
}

void
fill_normal(std::vector<float>& out, std::uint64_t seed)
{
  // splitmix64 + Box-Muller. Small, dependency-free and reproducible;
  // see the note in the header on why matching torch is not the goal.
  std::uint64_t s = seed ? seed : 0x9e3779b97f4a7c15ULL;
  auto next = [&s]() {
    s += 0x9e3779b97f4a7c15ULL;
    std::uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  };
  auto uniform = [&next]() {
    // (0, 1]: log(0) is not a value Box-Muller survives.
    return ((double)(next() >> 11) + 1.0) * (1.0 / 9007199254740992.0);
  };
  for (std::size_t i = 0; i < out.size(); i += 2) {
    const double u1 = uniform(), u2 = uniform();
    const double r = std::sqrt(-2.0 * std::log(u1));
    const double t = 6.283185307179586 * u2;
    out[i] = (float)(r * std::cos(t));
    if (i + 1 < out.size()) { out[i + 1] = (float)(r * std::sin(t)); }
  }
}

}  // namespace ltx25
