#ifndef VPIPE_LTX25_ROPE_H
#define VPIPE_LTX25_ROPE_H

#include <cstddef>
#include <string>
#include <vector>

namespace ltx25 {

// LTX-2.5's rotary embedding, which is not the one any other model in
// this tree uses. Three things differ, and each is silent when got
// wrong -- the forward runs, attention is scrambled:
//
//   FREQUENCIES are LOG-SPACED from 1 to theta over `dim / (2 * n_axes)`
//   entries and scaled by pi/2, rather than the usual
//   `theta^(-2i/d)` inverse-frequency ladder.
//
//   POSITIONS are FRACTIONAL and signed: `pos / max_pos` mapped through
//   `2f - 1` into [-1, 1]. A token's angle therefore depends on the
//   configured extent (max_pos = [20, 2048, 2048]), not on its absolute
//   index -- so the same token at the same index rotates differently in
//   a clip of a different declared size.
//
//   The GRID is (start, end) per axis per token and the MIDPOINT is
//   used (`use_middle_indices_grid`). That is what lets the model give
//   different regions different compression; collapsing it to one
//   integer position per token throws that away.
//
// The table is built ONCE per generation on the host -- it is
// O(tokens x dim/2) f32 and costs nothing beside a 22B forward -- and
// the per-step work is only the rotation.
//
// PRECISION follows the reference exactly rather than being as accurate
// as possible: the ladder is built in f64 and NARROWED to f32 before the
// angles are formed (that is what `frequencies_precision: "float64"`
// selects). Staying in double past that point measured 1.3e-4 rel-L2
// against the reference -- 4 orders of magnitude worse than matching it.
// Matching the model the weights were trained against beats being right
// in the abstract. VERIFIED: 1.8e-8 cos / 4.3e-8 sin / 4.3e-8 on a
// rotated q, i.e. f32 round-off.

// The cos / sin tables for one set of token positions.
//
// LAYOUT is [heads][tokens][half], head-major, matching the reference's
// [B, H, T, D/2] after its `swapaxes(1, 2)`. `half` is head_dim/2.
struct RopeTable {
  std::vector<float> cos;    // heads * tokens * half
  std::vector<float> sin;
  int heads  = 0;
  int tokens = 0;
  int half   = 0;            // head_dim / 2

  std::size_t size() const
  {
    return (std::size_t)heads * tokens * half;
  }
};

// Build the table.
//
// `starts` / `ends` are the position grid, [axes][tokens] each -- the
// (start, end) the midpoint is taken between. For a plain integer grid
// pass `ends[a][t] == starts[a][t] + 1`, which is what the patchifier
// emits for a 1x1x1 patch.
//
// `max_pos` has one entry per axis ([20, 2048, 2048] for video, [20] for
// audio). `inner_dim` is the stream's FULL width (heads * head_dim):
// 4096 for video, 2048 for audio -- and NOT the head dim, because the
// frequency count is derived from the whole width before being split
// across heads.
//
// Returns an empty table (and leaves `err` set) when the inputs do not
// agree -- a mismatched axis count is the one error that would otherwise
// produce a plausible table for the wrong geometry.
// `f64_precision` mirrors the checkpoint's `frequencies_precision`:
// true (LTX-2.5's value) builds the ladder in double and narrows once,
// false builds it in single throughout. It is NOT a quality knob -- the
// two differ by 6e-4 rel-L2 in the table, so picking the wrong one is
// simply a different model. See the note in freq_ladder_.
RopeTable build_rope(const std::vector<std::vector<double>>& starts,
                     const std::vector<std::vector<double>>& ends,
                     const std::vector<int>&                 max_pos,
                     int inner_dim, int heads, double theta = 10000.0,
                     bool f64_precision = true,
                     std::string* err = nullptr);

// How a latent cell maps to the coordinates the model was TRAINED on.
//
// This is not decoration. `build_rope` turns a position into an angle by
// `frac = mid / max_pos`, and LTX-2.5 declares
// positional_embedding_max_pos = [20, 2048, 2048] -- numbers that only
// make sense for SECONDS (20 s) and PIXELS (2048 px). Feeding raw latent
// indices instead (which this port did until it was caught) leaves a
// 768-wide clip with a spatial frac spanning 0..0.012 instead of
// 0..0.375: every token's spatial angle collapses into a band ~32x too
// narrow, the model gets almost no spatial position, and the output is
// structureless mush.
//
// The reference builds these as
//   positions = get_pixel_coords(latent_coords, scale_factors, causal_fix)
//   positions[:, 0, ...] /= fps
// (ltx_core/tools.py), with SpatioTemporalScaleFactors.default() =
// (time 8, height 32, width 32) and causal_fix = True.
struct RopeGeometry {
  // The VAE's scale factors. Available from VaeConfig::temporal_factor()
  // and spatial_factor(), so a graph that changed VAE cannot silently
  // keep the old ones.
  int    scale_t = 8;
  int    scale_h = 32;
  int    scale_w = 32;
  double fps     = 24.0;
  // The causal VAE's first frame spans one pixel frame, not `scale_t` of
  // them: the time coordinate is shifted by (1 - scale_t) and clamped at
  // zero. Off only for a non-causal VAE.
  bool   causal_fix = true;

  // Latent indices, unscaled -- what this port used everywhere before
  // the positions were pinned. It is the WRONG thing for a real
  // generation and is kept for exactly one purpose: the block and rope
  // goldens were generated from an integer arange grid, so they pin the
  // rope MATH, and reproducing them needs the same grid. A real forward
  // must never use it.
  static RopeGeometry identity()
  {
    return RopeGeometry{1, 1, 1, 1.0, false};
  }
};

// Convenience: the video grid for a [frames, height, width] latent, in
// the (f, h, w) order the patchifier flattens -- w fastest.
//
// Positions are PIXELS on the spatial axes and SECONDS on the time axis;
// see RopeGeometry for why passing latent indices is a silent, severe
// bug rather than a scaling nicety.
RopeTable build_video_rope(int frames, int height, int width,
                           const std::vector<int>& max_pos,
                           int inner_dim, int heads,
                           const RopeGeometry& geo,
                           double theta = 10000.0,
                           bool f64_precision = true);

// The position SPANS of a regular latent block, in the coordinates the
// model was trained on -- APPENDED to `starts` / `ends`, which are
// resized to 3 axes if they are empty.
//
// This is the grid build_video_rope does, exposed because conditioning
// needs the same thing at a different place in the clip. A keyframe
// anchor is a [z, 1, h, w] latent whose tokens ride at the END of the
// sequence but carry the positions of pixel frame `pixel_frame_offset`,
// so the model reads it as content belonging THERE rather than as extra
// frames on the front.
//
//   pixel_frame_offset  shifts the whole block along time, in PIXEL
//                       frames, before the seconds conversion. The
//                       reference's `positions[:, 0] += frame_idx`.
//   single_pixel_frame  narrows each token's temporal span to ONE pixel
//                       frame. What a keyframe latent that encodes one
//                       frame gets; without it the span is the VAE's
//                       whole temporal stride and the model is told a
//                       still image lasts 8 frames.
//   causal_fix          the first-frame correction. ON for the target
//                       grid and for a block at offset 0; OFF for a
//                       block placed later, whose first latent frame is
//                       not the clip's first frame.
void video_spans(int frames, int height, int width, const RopeGeometry& geo,
                 int pixel_frame_offset, bool single_pixel_frame,
                 bool causal_fix,
                 std::vector<std::vector<double>>* starts,
                 std::vector<std::vector<double>>* ends);

// The time coordinate, in SECONDS, of each of `frames` latent frames --
// the axis a video token contributes to the a2v/v2a cross table. Exposed
// so the DiT builds the cross table from the same numbers the self table
// used rather than re-deriving them.
std::vector<double> video_time_seconds(int frames, const RopeGeometry& geo);

// The same for audio: token t covers mel frames [4t, 4t+4), causal-fixed
// and scaled by hop/sample_rate. The reference's
// AudioPatchifier._get_audio_latent_time_in_sec.
std::vector<double> audio_time_seconds(int tokens, int downsample = 4,
                                       int hop = 160, int sample_rate = 16000,
                                       bool causal = true);

// Convenience: the 1-D audio grid (and the TIME-ONLY grid the
// audio<->video cross-attention uses for BOTH streams -- see
// build_cross_rope).
// NOTE: `tokens` positions are SECONDS, derived by audio_time_seconds --
// audio max_pos is [20], i.e. seconds, for the same reason as above.
//
// For a 1-axis table on raw INDICES (the goldens' arange grid, and the
// connector's register positions) use build_index_rope below. The two
// were the same function until the positions were pinned, which is how
// the audio table came to be built on token indices.
RopeTable build_audio_rope(int tokens, const std::vector<int>& max_pos,
                           int inner_dim, int heads, double theta = 10000.0,
                           bool f64_precision = true);

// The a2v / v2a cross-attention table, which is NOT either self-
// attention table.
//
// `audio_to_video_attn` projects the 4096-wide video query down to the
// AUDIO head size, so both streams' cross tables are built at the audio
// width; and only the TIME axis participates, because the frame index is
// the only coordinate a video token and an audio token share. Feeding
// the 3-axis video table here is a shape error at best and, at
// coincidentally-matching widths, silently wrong positions.
//
// `axis0` is each token's position on the time axis IN SECONDS --
// video_time_seconds() repeated across each frame's h*w cells for video,
// audio_time_seconds() for audio. It used to be the latent frame index,
// which put the two streams on different time bases and made the
// coupling attend to the wrong offsets.
// A 1-axis table on raw indices 0, 1, 2, ... Not a real audio table:
// this exists for the goldens, which pin the rope MATH on an arange
// grid, and for the connector's register positions (whose own golden
// verifies them as indices).
RopeTable build_index_rope(int tokens, const std::vector<int>& max_pos,
                           int inner_dim, int heads, double theta = 10000.0,
                           bool f64_precision = true);

// `starts` / `ends` are each token's SPAN on the time axis, in seconds.
// Both are passed explicitly because the ends cannot be inferred from
// the starts: every cell of one video frame shares that frame's span, so
// consecutive starts are EQUAL within a frame and a difference-based
// guess gives the last cell of each frame a span of zero.
RopeTable build_cross_rope(const std::vector<double>& starts,
                           const std::vector<double>& ends, int max_pos,
                           int audio_inner_dim, int heads,
                           double theta = 10000.0,
                           bool f64_precision = true);

// Deprecated shape kept for the index-based tests: end = start + 1.
RopeTable build_cross_rope(const std::vector<double>& axis0, int max_pos,
                           int audio_inner_dim, int heads,
                           double theta = 10000.0,
                           bool f64_precision = true);

// Apply the rotation in place to `x`, laid out [tokens][heads * head_dim]
// (the row-major activation layout, NOT the [heads][tokens] the table
// uses). head_dim is 2 * table.half.
//
//   out[.. , 0:D/2] = x[.., 0:D/2] * cos - x[.., D/2:D] * sin
//   out[.. , D/2:D] = x[.., D/2:D] * cos + x[.., 0:D/2] * sin
//
// i.e. the HALF rotation ("split"), not the interleaved-pairs one.
// False when the shapes disagree.
bool apply_rope(const RopeTable& t, float* x, int tokens, int heads,
                int head_dim);

}  // namespace ltx25

#endif
