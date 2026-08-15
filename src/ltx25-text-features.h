#ifndef VPIPE_LTX25_TEXT_FEATURES_H
#define VPIPE_LTX25_TEXT_FEATURES_H

#include "ltx25-config.h"
#include "ltx25-metal-ops.h"

#include "generative-models/weight-set.h"

#include <memory>
#include <string>
#include <vector>

namespace ltx25 {

// The LTX-specific half of the text encoder.
//
// Gemma-4 12B produces 49 hidden states -- the embedding output plus one
// per layer -- and this turns them into the two cross-attention contexts
// the DiT's connectors consume. It is deliberately separable from the
// Gemma FORWARD: that is the host's model (and needs a tap path the
// Gemma implementation does not have yet, see the note below), while
// everything here is LTX's own and can be verified on its own.
//
// WHAT FEEDS IT. `hidden` is [tokens][3840][49] -- token-major, then
// hidden dim, then LAYER. That is the order `torch.stack(hidden_states,
// dim=-1)` produces and it is what the flattening below assumes.
//
// THREE THINGS THAT ARE SILENT WHEN WRONG:
//
//   * the flatten is [T][D][L] -> [T][D*L], so the flat index is
//     `d*L + l` -- LAYER-FASTEST. Concatenating layer-major (all of
//     layer 0, then all of layer 1, ...) is the natural guess, produces
//     a correctly-shaped 188160-wide vector, and scrambles the
//     projection.
//   * the RMS norm is over D, PER (token, layer): each layer's
//     3840-vector is normalised on its own, for each token. That is what
//     `text_encoder_norm_type: "PER_TOKEN_RMS"` names.
//   * each projection gets its OWN rescale, `sqrt(out_dim / 3840)`,
//     applied to the normalised vector before the Linear -- so the video
//     and audio paths see DIFFERENTLY scaled inputs from the same
//     normalisation.
//
// Padded positions are zeroed after the norm and before the projection,
// which is why the mask is needed here and not only at attention time.
class Ltx25TextFeatures {
public:
  // Bind the two aggregate projections out of the TEXT ENCODER file
  // (not the DiT's). They are 4096 x 188160 and 2048 x 188160 -- about
  // 2.3 GB together at bf16, which is why they are their own object and
  // not part of the DiT.
  static std::unique_ptr<Ltx25TextFeatures>
  load(const DitConfig& cfg, vpipe::genai::WeightSet& ws, const MetalOps& ops,
       std::string* err);

  bool reserve(int tokens, std::string* err);

  // `hidden` is f32 [valid_count][hidden_dim][layers] -- the REAL
  // tokens only. They land at rows [valid_begin, valid_begin +
  // valid_count) of the output and everything else comes out zeroed.
  //
  // A RANGE, not a count, because LTX-2.5 pads its captions on the
  // LEFT: the reference's tokenizer is built with PaddingSide.LEFT and
  // 1024 max length, so the real tokens are a SUFFIX. A prefix count
  // would zero exactly the rows that carry the caption and keep the
  // ones that carry nothing -- and the result is the right shape, so
  // nothing downstream would notice.
  //
  // Writes bf16 [tokens][4096] and [tokens][2048] -- the two contexts,
  // PRE-connector. The DiT's connectors run over these.
  bool compute(const float* hidden, int tokens, int valid_begin,
               int valid_count,
               const vpipe::metal_compute::SharedBuffer& video_out,
               const vpipe::metal_compute::SharedBuffer& audio_out,
               std::string* err);

  int layers() const { return _layers; }
  int hidden_dim() const { return _hidden; }
  int flat_dim() const { return _hidden * _layers; }

private:
  Ltx25TextFeatures() = default;

  const MetalOps* _ops = nullptr;
  vpipe::metal_compute::SharedBuffer _vw, _vb, _aw, _ab;
  int _hidden = 3840;
  int _layers = 49;
  int _video_dim = 4096;
  int _audio_dim = 2048;
  int _tokens = 0;
  // The normalised+flattened rows, and the two rescaled copies. Two
  // because the rescale differs per projection and the GEMM reads its
  // input as it sits.
  vpipe::metal_compute::SharedBuffer _normed, _v_in, _a_in;
};

}  // namespace ltx25

#endif
