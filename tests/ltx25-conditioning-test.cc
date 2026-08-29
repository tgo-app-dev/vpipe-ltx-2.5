// The image / video / audio REFERENCE paths, against the reference's own
// conditioning classes.
//
// LTX-2.5 conditions a generation by rewriting a per-token denoise mask
// and appending tokens -- there is no i2v model and no extra weights --
// so the entire mechanism is bookkeeping over latents and can be pinned
// exactly, with no checkpoint and no GPU. That is what this does: the
// goldens come from `VideoConditionByLatentIndex`,
// `VideoConditionByKeyframeIndex` and `AudioConditionByReferenceLatent`
// themselves (gen_goldens.py `cond`), so a plausible-but-wrong placement
// fails here rather than three hours into a generation.
//
// WHAT WOULD OTHERWISE BE SILENT. Each of these produces a clean run and
// a wrong video:
//   * an anchor written at the right tokens with the mask left at 1 --
//     the model is told to generate over content it was given
//   * a closing keyframe given the target grid's temporal span instead
//     of one pixel frame -- a still image that "lasts" eight frames
//   * a keyframe placed with causal_fix on -- its position is clamped
//     against the clip's first frame rather than its own
//   * a reference soundtrack at POSITIVE seconds -- it overlaps the
//     soundtrack being generated on the one axis the two share
//
// Point VPIPE_LTX25_GOLDENS at the goldens directory; without it this
// SKIPS and says so.

#include "ltx25-conditioning.h"
#include "npy.h"

#include <cmath>
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

// The golden's geometry, fixed by gen_goldens.py's gen_conditioning().
constexpr int kC = 128;
constexpr int kF = 3, kH = 4, kW = 4;
constexpr int kTarget = kF * kH * kW;          // 48
constexpr int kPerFrame = kH * kW;             // 16
constexpr int kLastPixel = (kF - 1) * 8;       // 16
constexpr double kSigma = 0.703125;
constexpr int kAT = 10, kAREF = 6, kAC = 8, kAMEL = 16;

ltx25::RopeGeometry
geo_()
{
  ltx25::RopeGeometry g;
  g.scale_t = 8; g.scale_h = 32; g.scale_w = 32;
  g.fps = 24.0;
  g.causal_fix = true;
  return g;
}

// The reference's latents are [C, F, H, W] channel-major, which is the
// layout VideoAnchor takes, so the golden loads straight in.
void
test_video(const std::string& dir)
{
  std::printf("video: an image anchor at frame 0 and a closing keyframe\n");
  npy::Array anchor = npy::load(dir + "/cond_anchor.npy");
  npy::Array closing = npy::load(dir + "/cond_closing.npy");
  npy::Array w_mask = npy::load(dir + "/cond_v_denoise_mask.npy");
  npy::Array w_clean = npy::load(dir + "/cond_v_clean.npy");
  npy::Array w_pos = npy::load(dir + "/cond_v_positions.npy");
  npy::Array w_ts = npy::load(dir + "/cond_v_timesteps.npy");
  npy::Array w_din = npy::load(dir + "/cond_v_denoised_in.npy");
  npy::Array w_dout = npy::load(dir + "/cond_v_denoised_out.npy");
  for (const npy::Array* a : {&anchor, &closing, &w_mask, &w_clean, &w_pos,
                              &w_ts, &w_din, &w_dout}) {
    if (!a->ok) { check(false, "load goldens: " + a->err); return; }
  }

  std::vector<ltx25::VideoAnchor> va(2);
  va[0].latent = anchor.data.data();
  va[0].channels = kC; va[0].frames = 1; va[0].h = kH; va[0].w = kW;
  va[0].strength = 1.0;
  va[0].placement = ltx25::VideoAnchor::Placement::kLatentIndex;
  va[0].index = 0;
  va[1].latent = closing.data.data();
  va[1].channels = kC; va[1].frames = 1; va[1].h = kH; va[1].w = kW;
  va[1].strength = 1.0;
  va[1].placement = ltx25::VideoAnchor::Placement::kKeyframe;
  va[1].index = kLastPixel;

  ltx25::BuiltConditioning b;
  std::string err;
  if (!ltx25::build_conditioning(kF, kH, kW, kC, /*audio_tokens=*/0,
                                 /*audio_dim=*/0, geo_(), va, {}, &b, &err)) {
    check(false, "build_conditioning: " + err);
    return;
  }

  const int total = (int)w_mask.shape[0];
  check(b.v_tokens == total,
        "token count " + std::to_string(b.v_tokens) + " (target " +
        std::to_string(b.v_target) + " + appended " +
        std::to_string(b.v_tokens - b.v_target) + ")");
  if (b.v_tokens != total) { return; }

  // ---- the mask ------------------------------------------------------
  std::size_t bad = 0;
  for (int t = 0; t < total; ++t) {
    if (b.v_mask[(std::size_t)t] != w_mask.data[(std::size_t)t]) { ++bad; }
  }
  check(bad == 0, "the denoise mask matches token for token (" +
        std::to_string(bad) + " differ)");
  // Named explicitly as well, because "matches" hides which end is
  // wrong when both are.
  check(b.v_mask[0] == 0.0f && b.v_mask[kPerFrame - 1] == 0.0f,
        "the anchor zeroes latent frame 0 (tokens 0.." +
        std::to_string(kPerFrame - 1) + ")");
  check(b.v_mask[kPerFrame] == 1.0f && b.v_mask[kTarget - 1] == 1.0f,
        "every later target token is still generated");
  check(b.v_mask[kTarget] == 0.0f && b.v_mask[total - 1] == 0.0f,
        "the appended keyframe tokens are given, not generated");

  // ---- the clean latent ----------------------------------------------
  // The golden is TOKEN-major [tokens][C]; the port keeps it
  // channel-major, the layout the DiT reads.
  double worst = 0.0;
  for (int t = 0; t < total; ++t) {
    for (int c = 0; c < kC; ++c) {
      const float got = b.v_clean[(std::size_t)c * total + t];
      const float want = w_clean.data[(std::size_t)t * kC + c];
      worst = std::max(worst, (double)std::fabs(got - want));
    }
  }
  check(worst == 0.0, "the clean latent matches exactly (worst |diff| " +
        std::to_string(worst) + ")");

  // ---- the positions --------------------------------------------------
  // The golden is [3][tokens][2]; build the same table the DiT would.
  std::vector<std::vector<double>> s, e;
  ltx25::video_spans(kF, kH, kW, geo_(), 0, false, true, &s, &e);
  for (int a = 0; a < 3; ++a) {
    s[(std::size_t)a].insert(s[(std::size_t)a].end(),
                             b.cond.v_extra_starts[(std::size_t)a].begin(),
                             b.cond.v_extra_starts[(std::size_t)a].end());
    e[(std::size_t)a].insert(e[(std::size_t)a].end(),
                             b.cond.v_extra_ends[(std::size_t)a].begin(),
                             b.cond.v_extra_ends[(std::size_t)a].end());
  }
  double pos_worst = 0.0;
  for (int a = 0; a < 3; ++a) {
    for (int t = 0; t < total; ++t) {
      const float* w = w_pos.data.data() +
                       ((std::size_t)a * total + t) * 2;
      pos_worst = std::max(pos_worst,
                           std::fabs(s[(std::size_t)a][(std::size_t)t] - w[0]));
      pos_worst = std::max(pos_worst,
                           std::fabs(e[(std::size_t)a][(std::size_t)t] - w[1]));
    }
  }
  // f32 goldens against f64 spans: the seconds are k/24, so the bar is
  // the golden's own rounding, not ours.
  check(pos_worst < 1e-6, "every position span matches (worst |diff| " +
        std::to_string(pos_worst) + ")");
  // The one that is easy to get wrong on its own: a still image placed
  // at pixel frame 16 must span [16/24, 17/24), not the VAE's stride.
  const double kf_start = b.cond.v_extra_starts[0][0];
  const double kf_end = b.cond.v_extra_ends[0][0];
  check(std::fabs(kf_start - kLastPixel / 24.0) < 1e-9 &&
        std::fabs(kf_end - (kLastPixel + 1) / 24.0) < 1e-9,
        "the closing keyframe spans ONE pixel frame at " +
        std::to_string(kf_start) + ".." + std::to_string(kf_end) + " s");

  // ---- timesteps and the clean-latent blend ---------------------------
  //
  // These are what the mask is FOR. `timesteps = mask * sigma` is what
  // the DiT modulates from, and the blend is what the sampler applies
  // after every step; both are checked against the reference's own
  // helpers rather than re-derived.
  std::size_t ts_bad = 0;
  for (int t = 0; t < total; ++t) {
    const double want = w_ts.data[(std::size_t)t];
    const double got = b.v_mask[(std::size_t)t] * kSigma;
    if (std::fabs(got - want) > 1e-7) { ++ts_bad; }
  }
  check(ts_bad == 0, "timesteps = denoise_mask * sigma (" +
        std::to_string(ts_bad) + " differ)");

  std::vector<float> blended((std::size_t)kC * total);
  for (int t = 0; t < total; ++t) {
    const float m = b.v_mask[(std::size_t)t];
    for (int c = 0; c < kC; ++c) {
      blended[(std::size_t)c * total + t] =
          w_din.data[(std::size_t)t * kC + c] * m +
          b.v_clean[(std::size_t)c * total + t] * (1.0f - m);
    }
  }
  double blend_worst = 0.0;
  for (int t = 0; t < total; ++t) {
    for (int c = 0; c < kC; ++c) {
      blend_worst = std::max(
          blend_worst,
          (double)std::fabs(blended[(std::size_t)c * total + t] -
                            w_dout.data[(std::size_t)t * kC + c]));
    }
  }
  check(blend_worst == 0.0,
        "denoised * mask + clean * (1 - mask) matches (worst |diff| " +
        std::to_string(blend_worst) + ")");
}

void
test_audio(const std::string& dir)
{
  std::printf("audio: an appended reference soundtrack\n");
  npy::Array tokens = npy::load(dir + "/cond_a_ref_tokens.npy");
  npy::Array w_pos = npy::load(dir + "/cond_a_positions.npy");
  npy::Array w_mask = npy::load(dir + "/cond_a_denoise_mask.npy");
  npy::Array w_clean = npy::load(dir + "/cond_a_clean.npy");
  for (const npy::Array* a : {&tokens, &w_pos, &w_mask, &w_clean}) {
    if (!a->ok) { check(false, "load goldens: " + a->err); return; }
  }

  ltx25::AudioAnchor aa;
  aa.tokens = tokens.data.data();
  aa.rows = kAREF;
  aa.dim = kAC * kAMEL;
  aa.strength = 1.0;

  ltx25::BuiltConditioning b;
  std::string err;
  if (!ltx25::build_conditioning(kF, kH, kW, kC, kAT, kAC * kAMEL, geo_(),
                                 {}, {aa}, &b, &err)) {
    check(false, "build_conditioning: " + err);
    return;
  }
  const int total = kAT + kAREF;
  check(b.a_tokens == total, "audio tokens " + std::to_string(b.a_tokens) +
        " (" + std::to_string(kAT) + " target + " + std::to_string(kAREF) +
        " reference)");
  if (b.a_tokens != total) { return; }

  std::size_t bad = 0;
  for (int t = 0; t < total; ++t) {
    if (b.a_mask[(std::size_t)t] != w_mask.data[(std::size_t)t]) { ++bad; }
  }
  check(bad == 0, "the audio denoise mask matches (" + std::to_string(bad) +
        " differ)");

  // The positions: the target's own seconds, then the reference's --
  // which must be BELOW zero. A port that leaves them positive puts the
  // reference on top of the soundtrack it is meant to guide.
  std::vector<double> s = ltx25::audio_time_seconds(kAT);
  std::vector<double> e0 = ltx25::audio_time_seconds(kAT + 1);
  std::vector<double> e(e0.begin() + 1, e0.end());
  s.insert(s.end(), b.cond.a_extra_starts.begin(),
           b.cond.a_extra_starts.end());
  e.insert(e.end(), b.cond.a_extra_ends.begin(), b.cond.a_extra_ends.end());
  double worst = 0.0;
  for (int t = 0; t < total; ++t) {
    const float* w = w_pos.data.data() + (std::size_t)t * 2;
    worst = std::max(worst, std::fabs(s[(std::size_t)t] - w[0]));
    worst = std::max(worst, std::fabs(e[(std::size_t)t] - w[1]));
  }
  check(worst < 1e-6, "every audio position matches (worst |diff| " +
        std::to_string(worst) + ")");
  bool all_negative = true;
  for (double v : b.cond.a_extra_ends) {
    if (v > 0.0) { all_negative = false; }
  }
  check(all_negative && !b.cond.a_extra_ends.empty(),
        "the reference sits entirely before zero on the shared time axis");

  // The rows arrive TOKEN-major and must land channel-major.
  double cw = 0.0;
  for (int t = 0; t < total; ++t) {
    for (int c = 0; c < kAC * kAMEL; ++c) {
      cw = std::max(cw, (double)std::fabs(
          b.a_clean[(std::size_t)c * total + t] -
          w_clean.data[(std::size_t)t * (kAC * kAMEL) + c]));
    }
  }
  check(cw == 0.0, "the reference tokens land transposed into the clean "
        "latent (worst |diff| " + std::to_string(cw) + ")");
}

// The first latent frame's marker, which is present in EVERY generation
// -- conditioned or not -- because the causal VAE makes that frame span
// one pixel frame. It selects the tokens the trunk adds
// `keyframes_abs_pos_embedding` to, and the port applies it as a token
// PREFIX rather than a mask, so what is checked is that the golden's
// marker really is the prefix the port assumes.
void
test_keyframe_marker(const std::string& dir)
{
  std::printf("the first-latent-frame marker (keyframes_abs_pos_embedding)\n");
  npy::Array plain = npy::load(dir + "/cond_keyframes_mask_plain.npy");
  npy::Array withc = npy::load(dir + "/cond_v_keyframes_mask.npy");
  if (!plain.ok || !withc.ok) {
    check(false, "load goldens: " + plain.err + withc.err);
    return;
  }
  std::size_t bad = 0;
  for (std::size_t t = 0; t < plain.count(); ++t) {
    const bool want = t < (std::size_t)kPerFrame;
    if ((plain.data[t] != 0.0f) != want) { ++bad; }
  }
  check(bad == 0 && plain.count() == (std::size_t)kTarget,
        "an unconditioned generation marks exactly the first " +
        std::to_string(kPerFrame) + " tokens");

  bad = 0;
  for (std::size_t t = 0; t < withc.count(); ++t) {
    const bool want = t < (std::size_t)kPerFrame;
    if ((withc.data[t] != 0.0f) != want) { ++bad; }
  }
  check(bad == 0, "appended given content is NOT marked, so the marker "
        "stays a prefix (" + std::to_string(bad) + " differ)");
}

// Refusals. Every one of these would otherwise place content somewhere
// the caller did not mean and generate something that merely looks off.
void
test_refusals()
{
  std::printf("anchors that do not fit are refused\n");
  std::vector<float> junk((std::size_t)kC * kPerFrame, 0.0f);
  auto one = [&](int frames, int h, int w, int index,
                 ltx25::VideoAnchor::Placement p, double strength) {
    ltx25::VideoAnchor a;
    a.latent = junk.data();
    a.channels = kC; a.frames = frames; a.h = h; a.w = w;
    a.index = index; a.placement = p; a.strength = strength;
    return a;
  };
  ltx25::BuiltConditioning b;
  std::string err;
  const auto k = ltx25::VideoAnchor::Placement::kLatentIndex;

  check(!ltx25::build_conditioning(kF, kH, kW, kC, 0, 0, geo_(),
                                   {one(1, kH + 1, kW, 0, k, 1.0)}, {}, &b,
                                   &err),
        "a reference at the wrong latent resolution: " + err);
  check(!ltx25::build_conditioning(kF, kH, kW, kC, 0, 0, geo_(),
                                   {one(1, kH, kW, kF, k, 1.0)}, {}, &b, &err),
        "a reference past the end of the clip: " + err);
  check(!ltx25::build_conditioning(kF, kH, kW, kC, 0, 0, geo_(),
                                   {one(1, kH, kW, 0, k, 1.5)}, {}, &b, &err),
        "a strength outside [0, 1]: " + err);

  ltx25::AudioAnchor aa;
  std::vector<float> arow((std::size_t)kAREF * kAC * kAMEL, 0.0f);
  aa.tokens = arow.data(); aa.rows = kAREF; aa.dim = kAC * kAMEL;
  check(!ltx25::build_conditioning(kF, kH, kW, kC, /*audio_tokens=*/0,
                                   kAC * kAMEL, geo_(), {}, {aa}, &b, &err),
        "an audio reference with no audio stream: " + err);
}

// An unconditioned build must come out as the geometry the port had
// before conditioning existed: one level, no appended tokens, a mask of
// ones. That is what keeps every text-to-video forward on the broadcast
// path.
void
test_unconditioned()
{
  std::printf("no references at all\n");
  ltx25::BuiltConditioning b;
  std::string err;
  check(ltx25::build_conditioning(kF, kH, kW, kC, kAT, kAC * kAMEL, geo_(),
                                  {}, {}, &b, &err),
        "builds: " + err);
  check(b.v_tokens == kTarget && b.a_tokens == kAT,
        "no tokens are appended");
  check(b.cond.v_levels.size() == 1 && b.cond.a_levels.size() == 1,
        "one denoise level, so the per-token modulation stays off");
  check(!b.cond.conditioned(), "reports itself unconditioned");
  bool ones = true;
  for (float m : b.v_mask) { if (m != 1.0f) { ones = false; } }
  check(ones, "every token is being generated");
}

}  // namespace


// ---- the IC-LoRA reference clip ---------------------------------------
//
// An in-context LoRA is conditioned on a WHOLE reference clip, encoded at
// 1/factor of the target's resolution and appended to the sequence. What
// makes it a third placement rather than a bigger keyframe is the
// POSITIONS: a reference token stands for factor x factor of the
// target's cells, so its spatial span is stretched by the factor. Leave
// that out and the whole reference sits in the top-left corner of the
// frame -- a run that finishes cleanly and ignores most of its input,
// which is the exact failure this file exists to catch.
//
// Goldens from `VideoConditionByReferenceLatent`, the model author's own
// class (gen_goldens.py `iclora`).
void
test_iclora(const std::string& dir)
{
  std::printf("\nIC-LoRA reference conditioning\n");
  npy::Array ref = npy::load(dir + "/iclora_ref.npy");
  npy::Array w_mask = npy::load(dir + "/iclora_v_denoise_mask.npy");
  npy::Array w_clean = npy::load(dir + "/iclora_v_clean.npy");
  npy::Array w_pos = npy::load(dir + "/iclora_v_positions.npy");
  npy::Array ref1 = npy::load(dir + "/iclora_ref_f1.npy");
  npy::Array w_pos1 = npy::load(dir + "/iclora_f1_positions.npy");
  npy::Array w_mixed = npy::load(dir + "/iclora_mixed_denoise_mask.npy");
  for (const npy::Array* a : {&ref, &w_mask, &w_clean, &w_pos, &ref1,
                              &w_pos1, &w_mixed}) {
    if (!a->ok) { check(false, "load goldens: " + a->err); return; }
  }

  constexpr int kFactor = 2;
  constexpr int kRH = kH / kFactor, kRW = kW / kFactor;   // 2 x 2
  constexpr int kRefTokens = kF * kRH * kRW;              // 12

  std::vector<ltx25::VideoAnchor> va(1);
  va[0].latent = ref.data.data();
  va[0].channels = kC;
  va[0].frames = kF;
  va[0].h = kRH;
  va[0].w = kRW;
  va[0].strength = 1.0;
  va[0].placement = ltx25::VideoAnchor::Placement::kIcLoraReference;
  va[0].downscale = kFactor;

  ltx25::BuiltConditioning b;
  std::string err;
  if (!ltx25::build_conditioning(kF, kH, kW, kC, 0, 0, geo_(), va, {}, &b,
                                 &err)) {
    check(false, "build_conditioning: " + err);
    return;
  }
  const int total = (int)w_mask.shape[0];
  check(b.v_tokens == total && b.v_tokens == kTarget + kRefTokens,
        "token count " + std::to_string(b.v_tokens) + " = " +
        std::to_string(kTarget) + " target + " +
        std::to_string(kRefTokens) + " reference (the reference is "
        "COARSER, so it is not one block per target frame)");
  if (b.v_tokens != total) { return; }

  std::size_t bad = 0;
  for (int t = 0; t < total; ++t) {
    if (b.v_mask[(std::size_t)t] != w_mask.data[(std::size_t)t]) { ++bad; }
  }
  check(bad == 0, "the denoise mask matches token for token (" +
        std::to_string(bad) + " differ)");
  check(b.v_mask[0] == 1.0f && b.v_mask[kTarget - 1] == 1.0f,
        "the whole target grid is still being generated -- a reference "
        "clip conditions, it does not overwrite");

  double worst = 0.0;
  for (int t = 0; t < total; ++t) {
    for (int c = 0; c < kC; ++c) {
      const float got = b.v_clean[(std::size_t)c * total + t];
      const float want = w_clean.data[(std::size_t)t * kC + c];
      worst = std::max(worst, (double)std::fabs(got - want));
    }
  }
  check(worst == 0.0, "the reference's tokens land where the golden puts "
        "them (worst |diff| " + std::to_string(worst) + ")");

  // ---- the positions, which are the whole point ------------------------
  auto full_spans = [&](const ltx25::BuiltConditioning& bc,
                        std::vector<std::vector<double>>* s,
                        std::vector<std::vector<double>>* e) {
    ltx25::video_spans(kF, kH, kW, geo_(), 0, false, true, s, e);
    for (int a = 0; a < 3; ++a) {
      (*s)[(std::size_t)a].insert(
          (*s)[(std::size_t)a].end(),
          bc.cond.v_extra_starts[(std::size_t)a].begin(),
          bc.cond.v_extra_starts[(std::size_t)a].end());
      (*e)[(std::size_t)a].insert(
          (*e)[(std::size_t)a].end(),
          bc.cond.v_extra_ends[(std::size_t)a].begin(),
          bc.cond.v_extra_ends[(std::size_t)a].end());
    }
  };
  std::vector<std::vector<double>> s, e;
  full_spans(b, &s, &e);
  double pos_worst = 0.0;
  for (int a = 0; a < 3; ++a) {
    for (int t = 0; t < total; ++t) {
      const float* w = w_pos.data.data() + ((std::size_t)a * total + t) * 2;
      pos_worst = std::max(pos_worst,
                           std::fabs(s[(std::size_t)a][(std::size_t)t] - w[0]));
      pos_worst = std::max(pos_worst,
                           std::fabs(e[(std::size_t)a][(std::size_t)t] - w[1]));
    }
  }
  check(pos_worst < 1e-6, "every position span matches (worst |diff| " +
        std::to_string(pos_worst) + ")");

  // Said again in the concrete, because "matches" is exactly what an
  // unstretched reference would also report if the golden were wrong.
  // The reference's two columns must cover the full 128-pixel frame.
  const double h0s = b.cond.v_extra_starts[1][0];
  const double h0e = b.cond.v_extra_ends[1][0];
  const double w1s = b.cond.v_extra_starts[2][1];
  const double w1e = b.cond.v_extra_ends[2][1];
  check(h0s == 0.0 && h0e == kFactor * 32.0,
        "a reference token spans " + std::to_string(kFactor * 32) +
        " pixels, not 32 -- it stands for " + std::to_string(kFactor) +
        "x" + std::to_string(kFactor) + " target cells");
  check(w1s == kFactor * 32.0 && w1e == (double)kW * 32.0,
        "and the LAST column ends at the frame's edge (" +
        std::to_string(kW * 32) + "), so the reference covers the whole "
        "frame rather than its top-left corner");
  // The time axis is NOT stretched: the reference runs at the target's
  // frame rate. Scaling it too is the symmetric mistake and it is not
  // caught by anything spatial.
  check(b.cond.v_extra_starts[0][0] == 0.0 &&
        std::fabs(b.cond.v_extra_ends[0][0] - 1.0 / 24.0) < 1e-9,
        "the TIME span is untouched -- the first reference frame is the "
        "clip's first frame, one pixel frame long");

  // ---- factor 1, the same code path ------------------------------------
  {
    std::vector<ltx25::VideoAnchor> v1(1);
    v1[0].latent = ref1.data.data();
    v1[0].channels = kC;
    v1[0].frames = kF;
    v1[0].h = kH;
    v1[0].w = kW;
    v1[0].strength = 1.0;
    v1[0].placement = ltx25::VideoAnchor::Placement::kIcLoraReference;
    v1[0].downscale = 1;
    ltx25::BuiltConditioning b1;
    std::string e1;
    if (!ltx25::build_conditioning(kF, kH, kW, kC, 0, 0, geo_(), v1, {}, &b1,
                                   &e1)) {
      check(false, "factor 1: " + e1);
    } else {
      std::vector<std::vector<double>> s1, e1v;
      full_spans(b1, &s1, &e1v);
      const int n1 = (int)w_pos1.shape[1];
      double d1 = 0.0;
      for (int a = 0; a < 3; ++a) {
        for (int t = 0; t < n1; ++t) {
          const float* w = w_pos1.data.data() + ((std::size_t)a * n1 + t) * 2;
          d1 = std::max(d1, std::fabs(s1[(std::size_t)a][(std::size_t)t] - w[0]));
          d1 = std::max(d1, std::fabs(e1v[(std::size_t)a][(std::size_t)t] - w[1]));
        }
      }
      check(d1 < 1e-6, "factor 1 appends at the target's own resolution "
            "(worst |diff| " + std::to_string(d1) + ") -- the stretch is "
            "the factor's doing, not the placement's");
    }
  }

  // ---- a strength the mask has to carry --------------------------------
  {
    std::vector<ltx25::VideoAnchor> v2(2);
    // The golden's first item is an anchor over latent frame 0; its
    // latent is not saved, and nothing here reads its VALUES -- only the
    // mask, which depends on strength alone.
    std::vector<float> dummy((std::size_t)kC * kPerFrame, 0.0f);
    v2[0].latent = dummy.data();
    v2[0].channels = kC;
    v2[0].frames = 1;
    v2[0].h = kH;
    v2[0].w = kW;
    v2[0].strength = 1.0;
    v2[0].placement = ltx25::VideoAnchor::Placement::kLatentIndex;
    v2[0].index = 0;
    v2[1] = va[0];
    v2[1].strength = 0.75;
    ltx25::BuiltConditioning b2;
    std::string e2;
    if (!ltx25::build_conditioning(kF, kH, kW, kC, 0, 0, geo_(), v2, {}, &b2,
                                   &e2)) {
      check(false, "mixed strengths: " + e2);
    } else {
      std::size_t bad2 = 0;
      const int n2 = (int)w_mixed.shape[0];
      for (int t = 0; t < n2 && t < b2.v_tokens; ++t) {
        if (std::fabs(b2.v_mask[(std::size_t)t]
                      - w_mixed.data[(std::size_t)t]) > 1e-7) { ++bad2; }
      }
      check(b2.v_tokens == n2 && bad2 == 0,
            "a reference at strength 0.75 beside an anchor at 1.0 gives "
            "the golden's mask (" + std::to_string(bad2) + " differ)");
      check(b2.cond.v_levels.size() == 3,
            "and three denoise LEVELS -- generated, the anchor, the "
            "reference (got " + std::to_string(b2.cond.v_levels.size()) +
            ")");
    }
  }

  // ---- the refusals ----------------------------------------------------
  //
  // Both of these are wrong in a way that still RUNS if it is let
  // through, so they are errors rather than warnings.
  {
    std::vector<ltx25::VideoAnchor> v3 = va;
    v3[0].h = kH;                 // the target's grid, not 1/factor of it
    v3[0].w = kW;
    ltx25::BuiltConditioning b3;
    std::string e3;
    check(!ltx25::build_conditioning(kF, kH, kW, kC, 0, 0, geo_(), v3, {},
                                     &b3, &e3),
          "a reference at the WRONG resolution for its factor is refused");
  }
  {
    // A target grid the factor does not divide. 5 is odd, so a factor-2
    // reference cannot tile it.
    std::vector<ltx25::VideoAnchor> v4 = va;
    ltx25::BuiltConditioning b4;
    std::string e4;
    check(!ltx25::build_conditioning(kF, 5, kW, kC, 0, 0, geo_(), v4, {},
                                     &b4, &e4),
          "a target grid the factor does not divide is refused");
  }
}

int
main()
{
  const char* dir = std::getenv("VPIPE_LTX25_GOLDENS");
  test_refusals();
  test_unconditioned();
  if (dir == nullptr || *dir == '\0') {
    std::printf("\nSKIP the golden comparisons: set VPIPE_LTX25_GOLDENS to "
                "the directory gen_goldens.py wrote\n");
  } else {
    test_video(dir);
    test_audio(dir);
    test_iclora(dir);
    test_keyframe_marker(dir);
  }
  std::printf("\n%s\n", g_fail == 0 ? "ALL PASS" : "FAILURES");
  return g_fail == 0 ? 0 : 1;
}
