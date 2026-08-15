#include "ltx25-rope.h"

#include <algorithm>

#include <cmath>
#include <string>
#include <vector>

namespace ltx25 {

namespace {

// The frequency ladder: `theta ** linspace(0, 1, n)` scaled by pi/2.
//
// The reference spells the endpoints as log(1)/log(theta) and
// log(theta)/log(theta) -- 0 and 1 -- which is a log-space linspace from
// 1 to theta. Written out that way here because "0 to 1" hides what the
// two ends ARE, and the ends are what a reader has to check against the
// checkpoint's theta.
//
// PRECISION IS PART OF THE CONTRACT, not an implementation detail. The
// reference has two generators and the checkpoint picks one
// (`frequencies_precision`): f64 builds the ladder in double and narrows
// to f32 at the end, f32 builds it in single throughout. LTX-2.5 asks
// for f64.
//
// It matters more than "a few ulps" suggests. These frequencies run up
// to `theta * pi/2` ~ 15708, so a RELATIVE slip of 1e-7 is an ABSOLUTE
// angle error of ~1e-3 radians -- and cos/sin of an angle that large are
// pure phase, so that lands directly in the table. Measured: the two
// generators differ by 6e-4 rel-L2, which is 500x the f32 round-off this
// is checked to.
//
// Returned as f32 because that is where the reference narrows -- before
// the angles are formed, not after. Being "more accurate" by staying in
// double past this point would mean not matching the model the weights
// were trained against.
std::vector<float>
freq_ladder_(double theta, int n_axes, int inner_dim, bool f64_precision)
{
  const int n_elem = 2 * n_axes;
  const int n      = inner_dim / n_elem;
  std::vector<float> out;
  if (n <= 0) { return out; }
  out.reserve((std::size_t)n);
  for (int i = 0; i < n; ++i) {
    if (f64_precision) {
      // np.linspace(0, 1, n) then np.power(theta, .) * pi / 2, all f64,
      // narrowed once at the end.
      const double t = (n == 1) ? 0.0 : (double)i / (double)(n - 1);
      out.push_back((float)(std::pow(theta, t) * 3.14159265358979323846 / 2.0));
    } else {
      // torch.linspace(..., dtype=float32) then theta ** x, in f32.
      const float step = (n == 1) ? 0.0f : 1.0f / (float)(n - 1);
      const float t    = (float)i * step;
      float v = std::pow((float)theta, t);
      v = v * 3.14159265358979323846f;
      out.push_back(v / 2.0f);
    }
  }
  return out;
}

}  // namespace

RopeTable
build_rope(const std::vector<std::vector<double>>& starts,
           const std::vector<std::vector<double>>& ends,
           const std::vector<int>& max_pos, int inner_dim, int heads,
           double theta, bool f64_precision, std::string* err)
{
  RopeTable t;
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return RopeTable{};
  };
  const int n_axes = (int)starts.size();
  if (n_axes == 0) { return fail("no position axes"); }
  if ((int)ends.size() != n_axes) {
    return fail("starts/ends disagree on the axis count");
  }
  if ((int)max_pos.size() != n_axes) {
    // The one mismatch worth refusing: max_pos is a DIVISOR per axis, so
    // a short list silently rescales the axes it does cover and drops
    // the rest, which produces a plausible table for a geometry nobody
    // asked for.
    return fail("max_pos has " + std::to_string(max_pos.size()) +
                " entries for " + std::to_string(n_axes) + " axes");
  }
  const int tokens = (int)starts[0].size();
  for (int a = 0; a < n_axes; ++a) {
    if ((int)starts[a].size() != tokens || (int)ends[a].size() != tokens) {
      return fail("axis " + std::to_string(a) + " has a different length");
    }
  }
  if (heads <= 0 || inner_dim <= 0 || inner_dim % heads != 0) {
    return fail("inner_dim " + std::to_string(inner_dim) +
                " is not a multiple of heads " + std::to_string(heads));
  }
  const int head_dim = inner_dim / heads;
  const int half     = head_dim / 2;

  const std::vector<float> ladder =
      freq_ladder_(theta, n_axes, inner_dim, f64_precision);
  const int n_freq = (int)ladder.size();
  // The reference pads the FRONT with cos=1 / sin=0 to fill head_dim/2.
  // The pad exists because n_freq * n_axes rarely divides evenly: video
  // is 4096/6 = 682 frequencies x 3 axes = 2046, two short of 2048.
  const int used = n_freq * n_axes;
  const int pad  = half * heads - used;
  if (pad < 0) {
    return fail("frequency table (" + std::to_string(used) +
                ") overruns head_dim/2 * heads (" +
                std::to_string(half * heads) + ")");
  }

  t.heads  = heads;
  t.tokens = tokens;
  t.half   = half;
  t.cos.assign(t.size(), 0.0f);
  t.sin.assign(t.size(), 0.0f);

  // One token's row, before it is split across heads: `pad` identity
  // entries then (freq, axis) pairs with AXIS FASTEST -- the reference's
  // transpose(-1,-2).flatten(2) over a [.., axes, freqs] tensor, which
  // lands as index n_freq-major, axis-minor.
  std::vector<float> row_cos((std::size_t)(pad + used));
  std::vector<float> row_sin((std::size_t)(pad + used));
  for (int i = 0; i < pad; ++i) {
    row_cos[(std::size_t)i] = 1.0f;
    row_sin[(std::size_t)i] = 0.0f;
  }
  for (int tok = 0; tok < tokens; ++tok) {
    for (int f = 0; f < n_freq; ++f) {
      for (int a = 0; a < n_axes; ++a) {
        // The MIDPOINT of the token's (start, end) extent, made
        // fractional against this axis's declared maximum and mapped to
        // [-1, 1].
        // f32 from here on, which is where the reference computes:
        // `indices` (f32) * (fractional_positions (f32) * 2 - 1).
        const float mid  = (float)((starts[a][(std::size_t)tok] +
                                    ends[a][(std::size_t)tok]) * 0.5);
        const float frac = mid / (float)max_pos[(std::size_t)a];
        const float ang  = ladder[(std::size_t)f] * (frac * 2.0f - 1.0f);
        const std::size_t k = (std::size_t)pad + (std::size_t)f * n_axes + a;
        row_cos[k] = std::cos(ang);
        row_sin[k] = std::sin(ang);
      }
    }
    // Split the row across heads: the reference reshapes [B,T,W] to
    // [B,T,heads,W/heads] and swaps to [B,H,T,W/heads], so head h takes
    // the CONTIGUOUS slice [h*half, (h+1)*half).
    for (int h = 0; h < heads; ++h) {
      const std::size_t dst =
          ((std::size_t)h * tokens + (std::size_t)tok) * (std::size_t)half;
      const std::size_t src = (std::size_t)h * half;
      for (int j = 0; j < half; ++j) {
        t.cos[dst + (std::size_t)j] = row_cos[src + (std::size_t)j];
        t.sin[dst + (std::size_t)j] = row_sin[src + (std::size_t)j];
      }
    }
  }
  return t;
}

RopeTable
build_video_rope(int frames, int height, int width,
                 const std::vector<int>& max_pos, int inner_dim, int heads,
                 const RopeGeometry& geo, double theta, bool f64_precision)
{
  std::vector<std::vector<double>> starts, ends;
  video_spans(frames, height, width, geo, /*pixel_frame_offset=*/0,
              /*single_pixel_frame=*/false, geo.causal_fix, &starts, &ends);
  return build_rope(starts, ends, max_pos, inner_dim, heads, theta,
                    f64_precision, nullptr);
}

void
video_spans(int frames, int height, int width, const RopeGeometry& geo,
            int pixel_frame_offset, bool single_pixel_frame, bool causal_fix,
            std::vector<std::vector<double>>* starts,
            std::vector<std::vector<double>>* ends)
{
  if (starts == nullptr || ends == nullptr) { return; }
  if (starts->empty()) { starts->resize(3); }
  if (ends->empty()) { ends->resize(3); }
  if (starts->size() != 3 || ends->size() != 3) { return; }

  // The time coordinate, once per frame, in PIXEL FRAMES -- the
  // seconds division comes last, after the offset, exactly as
  // `get_pixel_coords` then `positions[:, 0] += frame_idx` then
  // `/= fps` does it.
  std::vector<double> tsec((std::size_t)std::max(0, frames));
  std::vector<double> tend((std::size_t)std::max(0, frames));
  for (int f = 0; f < frames; ++f) {
    double t = (double)(f * geo.scale_t);
    double e = (double)((f + 1) * geo.scale_t);
    // The causal VAE's first frame spans ONE pixel frame, not scale_t of
    // them. The clamp is INSIDE get_pixel_coords, so it happens before
    // the offset -- a block placed at frame 40 does not get its own
    // negative start clamped against 40.
    if (causal_fix) {
      t = t + 1.0 - (double)geo.scale_t;
      e = e + 1.0 - (double)geo.scale_t;
      if (t < 0.0) { t = 0.0; }
      if (e < 0.0) { e = 0.0; }
    }
    t += (double)pixel_frame_offset;
    e += (double)pixel_frame_offset;
    // A latent that encodes ONE pixel frame gets a one-frame span, not
    // the VAE's whole temporal stride.
    if (single_pixel_frame) { e = t + 1.0; }
    tsec[(std::size_t)f] = t / geo.fps;
    tend[(std::size_t)f] = e / geo.fps;
  }

  // (f, h, w) with w FASTEST -- the order the patchifier's
  // "b c f (h p2) (w p3) -> b (f h w) ..." flatten produces. Emitting
  // them in any other order gives every token a neighbour's angles.
  //
  // The VALUES are pixels and seconds, not latent indices; see
  // RopeGeometry.
  for (int f = 0; f < frames; ++f) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        (*starts)[0].push_back(tsec[(std::size_t)f]);
        (*ends)[0].push_back(tend[(std::size_t)f]);
        (*starts)[1].push_back((double)(h * geo.scale_h));
        (*ends)[1].push_back((double)((h + 1) * geo.scale_h));
        (*starts)[2].push_back((double)(w * geo.scale_w));
        (*ends)[2].push_back((double)((w + 1) * geo.scale_w));
      }
    }
  }
}

std::vector<double>
video_time_seconds(int frames, const RopeGeometry& geo)
{
  std::vector<double> v((std::size_t)std::max(0, frames));
  for (int f = 0; f < frames; ++f) {
    double t = (double)(f * geo.scale_t);
    // get_pixel_coords' causal_fix: the causal VAE's first frame spans
    // ONE pixel frame, not scale_t of them.
    if (geo.causal_fix) { t = t + 1.0 - (double)geo.scale_t; }
    v[(std::size_t)f] = (t < 0.0 ? 0.0 : t) / geo.fps;
  }
  return v;
}

std::vector<double>
audio_time_seconds(int tokens, int downsample, int hop, int sample_rate,
                   bool causal)
{
  std::vector<double> v((std::size_t)std::max(0, tokens));
  const double per_frame =
      (sample_rate > 0) ? (double)hop / (double)sample_rate : 0.0;
  for (int t = 0; t < tokens; ++t) {
    double mel = (double)(t * downsample);
    if (causal) { mel = mel + 1.0 - (double)downsample; }
    v[(std::size_t)t] = (mel < 0.0 ? 0.0 : mel) * per_frame;
  }
  return v;
}

RopeTable
build_index_rope(int tokens, const std::vector<int>& max_pos, int inner_dim,
                 int heads, double theta, bool f64_precision)
{
  std::vector<std::vector<double>> starts(1), ends(1);
  starts[0].resize((std::size_t)tokens);
  ends[0].resize((std::size_t)tokens);
  for (int i = 0; i < tokens; ++i) {
    starts[0][(std::size_t)i] = i;
    ends[0][(std::size_t)i]   = i + 1;
  }
  return build_rope(starts, ends, max_pos, inner_dim, heads, theta,
                    f64_precision, nullptr);
}

RopeTable
build_audio_rope(int tokens, const std::vector<int>& max_pos, int inner_dim,
                 int heads, double theta, bool f64_precision)
{
  // SECONDS, not the token index: audio max_pos is [20], i.e. a
  // 20-second span, so an index here would collapse every token's angle
  // into a sliver the same way latent indices did on the video side.
  const std::vector<double> t0 = audio_time_seconds(tokens);
  const std::vector<double> t1 = audio_time_seconds(tokens + 1);
  std::vector<std::vector<double>> starts(1), ends(1);
  starts[0] = t0;
  ends[0].assign(t1.begin() + 1, t1.end());
  return build_rope(starts, ends, max_pos, inner_dim, heads, theta,
                    f64_precision, nullptr);
}

RopeTable
build_cross_rope(const std::vector<double>& starts,
                 const std::vector<double>& ends, int max_pos,
                 int audio_inner_dim, int heads, double theta,
                 bool f64_precision)
{
  std::vector<std::vector<double>> s(1), e(1);
  s[0] = starts;
  e[0] = ends;
  return build_rope(s, e, {max_pos}, audio_inner_dim, heads, theta,
                    f64_precision, nullptr);
}

RopeTable
build_cross_rope(const std::vector<double>& axis0, int max_pos,
                 int audio_inner_dim, int heads, double theta,
                 bool f64_precision)
{
  // end = start + 1: correct only when the positions are INDICES, which
  // is what the index-based tests use. A seconds-based caller must pass
  // its spans explicitly through the overload above.
  std::vector<double> ends(axis0.size());
  for (std::size_t i = 0; i < axis0.size(); ++i) { ends[i] = axis0[i] + 1.0; }
  return build_cross_rope(axis0, ends, max_pos, audio_inner_dim, heads,
                          theta, f64_precision);
}

bool
apply_rope(const RopeTable& t, float* x, int tokens, int heads, int head_dim)
{
  if (x == nullptr || t.heads != heads || t.tokens != tokens ||
      head_dim != 2 * t.half) {
    return false;
  }
  const int half = t.half;
  for (int tok = 0; tok < tokens; ++tok) {
    for (int h = 0; h < heads; ++h) {
      float* row = x + ((std::size_t)tok * heads + (std::size_t)h)
                       * (std::size_t)head_dim;
      const std::size_t off =
          ((std::size_t)h * tokens + (std::size_t)tok) * (std::size_t)half;
      const float* c = t.cos.data() + off;
      const float* s = t.sin.data() + off;
      for (int j = 0; j < half; ++j) {
        // Read BOTH halves before writing either: the rotation mixes
        // them, so updating in place one at a time feeds the new first
        // half into the second half's term.
        const float a = row[j];
        const float b = row[j + half];
        row[j]        = a * c[j] - b * s[j];
        row[j + half] = b * c[j] + a * s[j];
      }
    }
  }
  return true;
}

}  // namespace ltx25
