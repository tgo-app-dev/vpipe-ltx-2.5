#ifndef VPIPE_LTX25_SAMPLER_H
#define VPIPE_LTX25_SAMPLER_H

#include <cstdint>
#include <vector>

namespace ltx25 {

// LTX-2.5's sampler.
//
// WHICH ONE, AND WHY IT MATTERS. The reference ships two: a deterministic
// second-order `res_2s` exponential-integrator step, and an ANCESTRAL
// (SDE) Euler step. `should_use_ancestral_sampler` picks the ancestral
// one for any checkpoint of generation >= 2.5, at `eta = 1.0` and
// `s_noise = 1.0` -- so LTX-2.5 is sampled ancestrally and the res_2s
// default does not apply to it.
//
// Reaching for res_2s anyway would be wrong twice over: a different
// trajectory, and TWICE the cost, because res_2s evaluates the model at
// a midpoint as well. On a 22B model that is the difference between 8
// and 16 forwards for the distilled schedule.
//
// This is the rectified-flow parameterisation (`alpha = 1 - sigma`), NOT
// the DDIM/variance-exploding ancestral coefficients. The reference says
// so explicitly, and the two agree only at eta = 0.

// x0 from a velocity prediction: `x - v * sigma`.
//
// The DiT predicts a VELOCITY; every step below wants the denoised
// sample. Keeping the conversion here rather than in the model is what
// lets the same DiT serve a schedule that wants one and a schedule that
// wants the other.
void to_denoised(const float* x, const float* velocity, double sigma,
                 float* out, std::size_t n);

// x0 from a velocity when each token has its OWN noise level.
//
// `x`, `velocity` and `out` are CHANNEL-major [channels][tokens]; `mask`
// is one value per token and the effective level is `sigma * mask[t]`.
// A token the mask zeroes therefore comes back unchanged -- it is
// already what it should be -- which is the reference's
// `to_denoised(latent, v, timesteps)` with `timesteps = mask * sigma`.
//
// Separate from to_denoised() rather than a default argument because
// the unconditioned path is the hot one and should not pay a per-token
// load for a value it knows is 1.
void to_denoised_masked(const float* x, const float* velocity, double sigma,
                        const float* mask, int channels, int tokens,
                        float* out);

// out[c][t] = out[c][t] * mask[t] + clean[c][t] * (1 - mask[t]).
//
// The reference's `post_process_latent`, IN PLACE. It runs twice per
// step -- once on the denoised prediction and once on the stepped latent
// -- and that second application is what actually holds a conditioned
// token still: the stepper takes a scalar sigma, so without it the
// anchor would be renoised along with everything else while the model
// was being told it is clean.
void hold_conditioned(float* x, const float* clean, const float* mask,
                      int channels, int tokens);

// One ancestral Euler step.
//
//   downstep_ratio = 1 + (sigma_next/sigma - 1) * eta
//   sigma_down     = sigma_next * downstep_ratio
//   x              = (sigma_down/sigma) x + (1 - sigma_down/sigma) x0
//   and, when eta > 0, renoise from sigma_down back up to sigma_next:
//   x              = (alpha_next/alpha_down) x + noise * s_noise * c
//   with c = sqrt(max(0, sigma_next^2 - sigma_down^2 alpha_next^2/alpha_down^2))
//
// `noise` may be null only when eta == 0. When `sigma_next` is 0 the
// step returns the denoised sample unchanged -- the schedule's last
// entry is 0, so this is the ordinary way a run ends, not a special
// case.
void euler_ancestral_step(const float* x, const float* denoised,
                          const float* noise, double sigma, double sigma_next,
                          double eta, double s_noise, float* out,
                          std::size_t n);

// The eta / s_noise LTX-2.5 samples with. Named rather than defaulted at
// each call site so a graph that wants a deterministic run has one thing
// to change.
inline constexpr double kAncestralEta    = 1.0;
inline constexpr double kAncestralSNoise = 1.0;

// A deterministic normal draw for the renoise term.
//
// Deliberately NOT an attempt to reproduce torch's Philox: the two would
// diverge anyway, and a port that chases bit-identical noise is chasing
// the one thing that legitimately differs between implementations. What
// matters is that the SAME seed gives the SAME video here, which this
// does.
void fill_normal(std::vector<float>& out, std::uint64_t seed);

}  // namespace ltx25

#endif
