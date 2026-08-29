#ifndef VPIPE_LTX25_CONDITIONING_H
#define VPIPE_LTX25_CONDITIONING_H

#include "ltx25-rope.h"

#include <cstdint>

#include <string>
#include <vector>

namespace ltx25 {

// Turning the graph's REFERENCES into what the DiT and the sampler need.
//
// LTX-2.5 has no image-to-video model and no reference-video model. It
// has ONE model and a per-token denoise mask: a reference is "these
// tokens already hold known content, at noise level zero", and the same
// mechanism serves an image anchor, a video prefix, a closing keyframe
// and a reference soundtrack. Everything specific to a KIND of reference
// -- where its tokens sit and what positions they carry -- is decided
// here; the DiT downstream only ever sees levels, an index per token and
// a position table.
//
// THE TWO PLACEMENTS, and why a keyframe cannot use the first one.
// A latent that belongs at the START of the clip overwrites the target
// grid in place: its tokens already exist and already have the right
// positions. A latent that belongs LATER cannot, because the target
// grid's later frames each span `scale_t` pixel frames while a still
// image spans one -- writing it into latent frame k would tell the model
// the picture holds for eight frames. So it rides as extra tokens on the
// end of the sequence carrying the positions of the pixel frame it
// belongs to, which is what `VideoConditionByKeyframeIndex` does.
//
// VERIFIED against the reference's own conditioning classes -- see
// tests/ltx25-conditioning-test.cc and the `cond` section of
// gen_goldens.py. That golden needs no weights: this whole mechanism is
// pure bookkeeping over latents, so it is pinned exactly rather than
// argued from the source.

// What the model is told about each token beyond where it sits: which
// tokens are being generated, and which carry content it must take as
// given.
//
// THIS IS LTX'S WHOLE CONDITIONING MECHANISM. There is no separate i2v
// model and no extra weights: an image anchor, a video prefix, a closing
// keyframe and a reference soundtrack are all "these tokens hold known
// content at noise level zero", expressed as a per-token `denoise_mask`
// that scales the timestep the adaLN chain sees
// (`timesteps = denoise_mask * sigma`). Tokens the mask zeroes are
// modulated as clean while their neighbours are modulated at the
// schedule's sigma; the sampler then keeps forcing them back to the
// content they were given. See ltx25-generator.cc for the loop.
//
// LEVELS, NOT A MASK PER TOKEN. The mask takes a handful of distinct
// values -- 1.0 for what is being generated and `1 - strength` per
// conditioning item -- so this carries the distinct values and a
// per-token index into them. The adaLN chain then runs once per level
// instead of once per token, which is the difference between eight
// 4096x36864 projections and tens of thousands of them.
struct Conditioning {
  // Distinct denoise levels. Entry 0 is the tokens being generated and
  // must be 1.0; a conditioning item at `strength` adds `1 - strength`.
  std::vector<double> v_levels{1.0};
  std::vector<double> a_levels{1.0};
  // Each token's level index, sized to the stream's FULL token count
  // (target grid plus anything appended). Empty puts every token at
  // level 0, which is a plain text-to-video generation.
  std::vector<std::uint8_t> v_level;
  std::vector<std::uint8_t> a_level;

  // ---- tokens APPENDED after the target grid -------------------------
  //
  // Conditioning that does not overwrite part of the clip -- a closing
  // keyframe, an IC-LoRA reference, a reference soundtrack -- rides as
  // extra tokens on the end of the sequence with positions of their own.
  // The model attends to them and their velocity is thrown away.
  //
  // Position SPANS, [axis][token]: video is 3 axes (seconds, pixels,
  // pixels) and audio 1 (seconds), the same coordinates the target grid
  // uses. Axis 0 doubles as the audio<->video cross table's time span,
  // because seconds is the one coordinate the two streams share.
  std::vector<std::vector<double>> v_extra_starts, v_extra_ends;
  std::vector<double>              a_extra_starts, a_extra_ends;

  int v_extra() const
  {
    return v_extra_starts.empty() ? 0 : (int)v_extra_starts[0].size();
  }
  int a_extra() const { return (int)a_extra_starts.size(); }
  bool conditioned() const { return v_levels.size() > 1 || a_levels.size() > 1
                                    || v_extra() > 0 || a_extra() > 0; }
};

// A video reference the graph wired.
struct VideoAnchor {
  // f32, CHANNEL-major [z][frames][h][w] -- a vae-encode's output shape.
  const float* latent = nullptr;
  int channels = 0;
  int frames = 0, h = 0, w = 0;
  // 1.0 keeps the content clean (the usual anchor); 0.0 would denoise it
  // away entirely. The reference calls this `strength` and stores
  // `1 - strength` in the mask.
  double strength = 1.0;

  enum class Placement {
    // Overwrite the target grid from latent frame `index`. The
    // image-to-video anchor (index 0, one frame) and a video prefix
    // (index 0, several) are both this.
    kLatentIndex,
    // Append tokens carrying PIXEL frame `index`'s positions. A closing
    // keyframe is this.
    kKeyframe,
    // Append a WHOLE reference clip, at 1/`downscale` of the target's
    // spatial resolution, with its positions stretched back over the
    // target's extent. What an IC-LoRA reads.
    //
    // WHAT MAKES THIS A THIRD PLACEMENT rather than a keyframe with a
    // smaller grid: the other two carry a latent on the target's own
    // h x w, so a token's position is the target's. Here the reference
    // is coarser, and each of its tokens stands for `downscale` x
    // `downscale` of the target's cells -- so its spatial spans are
    // multiplied by the factor and it is the SPAN, not the count, that
    // has to line up. Leave the spans unscaled and the whole reference
    // piles into the top-left corner of the frame; the run is clean and
    // the output ignores most of the input.
    //
    // The tokens themselves are the reference's, unchanged. An
    // implementation that dilates the latent into the target's grid and
    // then drops the filler tokens (which is how the ComfyUI node
    // states it) arrives at exactly the same spans; see the note in
    // ltx25-conditioning.cc.
    kIcLoraReference,
  };
  Placement placement = Placement::kLatentIndex;
  int index = 0;

  // kIcLoraReference only: the target-over-reference SPATIAL ratio, from
  // the adapter's `reference_downscale_factor` metadata. It must be the
  // factor the adapter was TRAINED at -- it is what preserves the
  // positional relationship the adapter learned, so a wrong value is
  // not a resolution mismatch but a different model.
  int downscale = 1;
};

// A reference soundtrack, ALREADY PATCHIFIED: [rows][channels * mel_bins]
// as `b c t f -> b t (c f)` produces it. Rows rather than a [c][t][f]
// latent because that is the shape it arrives in on the stage's port,
// and re-deriving c and f from it would be guessing.
struct AudioAnchor {
  const float* tokens = nullptr;
  int rows = 0, dim = 0;
  double strength = 1.0;
};

// Everything downstream needs, built together because the token layout
// has to agree between all of it.
struct BuiltConditioning {
  Conditioning cond;

  // The content the masked tokens are held at, CHANNEL-major
  // [z][v_tokens] / [z][a_tokens] -- the same layout `Ltx25Dit::Input`
  // takes, so the sampler blends and hands over one array.
  std::vector<float> v_clean, a_clean;
  // The denoise mask itself, one float per token. The DiT works from
  // levels; the SAMPLER works from this, because `to_denoised` and the
  // clean-latent blend are per token and per value.
  std::vector<float> v_mask, a_mask;

  int v_tokens = 0, a_tokens = 0;   // target + appended
  int v_target = 0, a_target = 0;

  bool any() const { return v_tokens != v_target || a_tokens != a_target
                            || cond.v_levels.size() > 1
                            || cond.a_levels.size() > 1; }
};

// Build it. False (with `err`) when an anchor does not fit the geometry
// -- a mismatched spatial size or an index past the clip, both of which
// would otherwise land content in the wrong place and generate something
// plausible.
bool build_conditioning(int latent_frames, int latent_h, int latent_w,
                        int channels, int audio_tokens, int audio_dim,
                        const RopeGeometry& geo,
                        const std::vector<VideoAnchor>& video,
                        const std::vector<AudioAnchor>& audio,
                        BuiltConditioning* out, std::string* err);

// The seconds spans of a reference soundtrack's tokens, placed BELOW
// zero so they cannot collide with the clip being generated.
//
// The reference's `patchify_dubit_audio_reference_latent(...,
// negative_positions=True)`: the block's own timeline shifted down by
// its duration plus one token, so the last reference token ends exactly
// one token before the generated soundtrack starts.
void audio_reference_spans(int rows, std::vector<double>* starts,
                           std::vector<double>* ends);

}  // namespace ltx25

#endif
