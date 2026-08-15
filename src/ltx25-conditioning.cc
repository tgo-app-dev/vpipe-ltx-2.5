#include "ltx25-conditioning.h"

#include "common/vpipe-format.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

using vpipe::fmt;

namespace ltx25 {

namespace {

// The level index for a strength, appending the level if it is new.
//
// Levels are compared EXACTLY. Two anchors at 0.9 and 0.9000001 would
// get two levels and two adaLN chains, which is wasteful but correct; a
// tolerance here would instead modulate one of them at the other's noise
// level, which is not.
std::uint8_t
level_for_(std::vector<double>* levels, double strength)
{
  const double v = 1.0 - strength;
  for (std::size_t i = 0; i < levels->size(); ++i) {
    if ((*levels)[i] == v) { return (std::uint8_t)i; }
  }
  levels->push_back(v);
  return (std::uint8_t)(levels->size() - 1);
}

}  // namespace

void
audio_reference_spans(int rows, std::vector<double>* starts,
                      std::vector<double>* ends)
{
  if (starts == nullptr || ends == nullptr || rows <= 0) { return; }
  const std::vector<double> s = audio_time_seconds(rows);
  const std::vector<double> e0 = audio_time_seconds(rows + 1);
  std::vector<double> e(e0.begin() + 1, e0.end());
  // One audio token, in seconds: the VAE's temporal downsample times the
  // hop over the sample rate (4 * 160 / 16000 = 0.04). Computed from the
  // same numbers audio_time_seconds uses rather than written as 0.04, so
  // a checkpoint with a different hop moves both together.
  const double one_token = 4.0 * 160.0 / 16000.0;
  double dur = 0.0;
  for (double v : e) { dur = std::max(dur, v); }
  const double shift = dur + one_token;
  starts->assign(s.begin(), s.end());
  ends->assign(e.begin(), e.end());
  for (double& v : *starts) { v -= shift; }
  for (double& v : *ends) { v -= shift; }
}

bool
build_conditioning(int latent_frames, int latent_h, int latent_w,
                   int channels, int audio_tokens, int audio_dim,
                   const RopeGeometry& geo,
                   const std::vector<VideoAnchor>& video,
                   const std::vector<AudioAnchor>& audio,
                   BuiltConditioning* out, std::string* err)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  if (out == nullptr) { return fail("no output"); }
  if (latent_frames <= 0 || latent_h <= 0 || latent_w <= 0 || channels <= 0) {
    return fail("degenerate target geometry");
  }
  *out = BuiltConditioning{};

  const int per_frame = latent_h * latent_w;
  const int v_target = latent_frames * per_frame;
  const int a_target = std::max(0, audio_tokens);

  // ---- pass 1: sizes and validation ------------------------------------
  //
  // Everything is checked before anything is written, so a graph with one
  // bad anchor gets a message rather than a half-built sequence.
  int v_extra = 0;
  for (const VideoAnchor& a : video) {
    if (a.latent == nullptr || a.frames <= 0) {
      return fail("a video reference has no latent");
    }
    if (a.channels != channels) {
      return fail(fmt("a video reference has {} channels but this DiT takes "
                      "{} -- it was encoded by a different VAE",
                      a.channels, channels)());
    }
    if (a.h != latent_h || a.w != latent_w) {
      return fail(fmt("a video reference is {}x{} in latent space but the "
                      "clip is {}x{}. Encode the reference at the SAME "
                      "resolution as the output",
                      a.h, a.w, latent_h, latent_w)());
    }
    if (a.strength < 0.0 || a.strength > 1.0) {
      return fail(fmt("a video reference's strength is {:.3f}; it must be in "
                      "[0, 1]", a.strength)());
    }
    if (a.placement == VideoAnchor::Placement::kLatentIndex) {
      if (a.index < 0 || a.index + a.frames > latent_frames) {
        return fail(fmt("a video reference of {} latent frames at index {} "
                        "does not fit a {}-frame clip", a.frames, a.index,
                        latent_frames)());
      }
    } else {
      v_extra += a.frames * per_frame;
    }
  }
  int a_extra = 0;
  for (const AudioAnchor& a : audio) {
    if (a.tokens == nullptr || a.rows <= 0) {
      return fail("an audio reference has no rows");
    }
    if (audio_dim > 0 && a.dim != audio_dim) {
      return fail(fmt("an audio reference is {} wide but this DiT patchifies "
                      "audio at {}", a.dim, audio_dim)());
    }
    if (a_target <= 0) {
      return fail("an audio reference was wired but this generation has no "
                  "audio stream to attach it to");
    }
    a_extra += a.rows;
  }

  const int v_tokens = v_target + v_extra;
  const int a_tokens = a_target + a_extra;

  out->v_target = v_target;
  out->a_target = a_target;
  out->v_tokens = v_tokens;
  out->a_tokens = a_tokens;
  out->v_clean.assign((std::size_t)channels * v_tokens, 0.0f);
  out->a_clean.assign((std::size_t)channels * a_tokens, 0.0f);
  // 1.0 everywhere to start: every token is being generated until an
  // anchor says otherwise.
  out->v_mask.assign((std::size_t)v_tokens, 1.0f);
  out->a_mask.assign((std::size_t)a_tokens, 1.0f);
  out->cond.v_levels = {1.0};
  out->cond.a_levels = {1.0};
  out->cond.v_level.assign((std::size_t)v_tokens, 0);
  out->cond.a_level.assign((std::size_t)a_tokens, 0);
  if (v_extra > 0) {
    out->cond.v_extra_starts.resize(3);
    out->cond.v_extra_ends.resize(3);
  }

  // ---- pass 2: place the video references -------------------------------
  int append_at = v_target;
  for (const VideoAnchor& a : video) {
    const std::uint8_t g = level_for_(&out->cond.v_levels, a.strength);
    const double m = 1.0 - a.strength;
    const int n = a.frames * per_frame;
    const int base = (a.placement == VideoAnchor::Placement::kLatentIndex)
                         ? a.index * per_frame
                         : append_at;
    // The reference's tokens, channel-major into the sequence's own
    // channel-major clean latent. Both sides flatten (f, h, w) with w
    // fastest, so this is a straight column copy.
    for (int ch = 0; ch < channels; ++ch) {
      const float* src = a.latent + (std::size_t)ch * n;
      float* dst = out->v_clean.data() + (std::size_t)ch * v_tokens + base;
      for (int t = 0; t < n; ++t) { dst[t] = src[t]; }
    }
    for (int t = 0; t < n; ++t) {
      out->v_mask[(std::size_t)base + t] = (float)m;
      out->cond.v_level[(std::size_t)base + t] = g;
    }
    if (a.placement == VideoAnchor::Placement::kKeyframe) {
      // causal_fix follows the reference: ON only when the block sits at
      // pixel frame 0, because only there is its first latent frame the
      // clip's own first frame. `single_pixel_frame` narrows a one-frame
      // latent's temporal span to one pixel frame -- without it the
      // model is told a still image lasts the VAE's whole stride.
      video_spans(a.frames, latent_h, latent_w, geo, a.index,
                  /*single_pixel_frame=*/a.frames == 1,
                  /*causal_fix=*/a.index == 0,
                  &out->cond.v_extra_starts, &out->cond.v_extra_ends);
      append_at += n;
    }
  }

  // ---- pass 3: the audio references ------------------------------------
  int a_append_at = a_target;
  for (const AudioAnchor& a : audio) {
    const std::uint8_t g = level_for_(&out->cond.a_levels, a.strength);
    const double m = 1.0 - a.strength;
    for (int ch = 0; ch < channels && ch < a.dim; ++ch) {
      float* dst = out->a_clean.data() + (std::size_t)ch * a_tokens
                   + a_append_at;
      for (int t = 0; t < a.rows; ++t) {
        // The rows arrive TOKEN-major [row][dim]; the DiT reads the
        // audio latent channel-major. Same transpose the generator does
        // on the way out, run backwards.
        dst[t] = a.tokens[(std::size_t)t * a.dim + ch];
      }
    }
    for (int t = 0; t < a.rows; ++t) {
      out->a_mask[(std::size_t)a_append_at + t] = (float)m;
      out->cond.a_level[(std::size_t)a_append_at + t] = g;
    }
    std::vector<double> s, e;
    audio_reference_spans(a.rows, &s, &e);
    out->cond.a_extra_starts.insert(out->cond.a_extra_starts.end(),
                                    s.begin(), s.end());
    out->cond.a_extra_ends.insert(out->cond.a_extra_ends.end(),
                                  e.begin(), e.end());
    a_append_at += a.rows;
  }

  return true;
}

}  // namespace ltx25
