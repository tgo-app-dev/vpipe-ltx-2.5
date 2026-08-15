#ifndef VPIPE_LTX25_TEXT_ENCODER_H
#define VPIPE_LTX25_TEXT_ENCODER_H

#include "ltx25-config.h"
#include "ltx25-metal-ops.h"
#include "ltx25-text-features.h"

#include "generative-models/hidden-state-encoder.h"
#include "generative-models/tokenizer.h"
#include "generative-models/weight-set.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ltx25 {

// The whole text path: caption in, the DiT's two cross-attention
// contexts out.
//
// It is the JOIN of two halves that are verified separately -- the
// Gemma-4 12B forward, which is the host's (reached through
// genai::HiddenStateEncoder so this plugin needs no access to
// MetalGemmaModel), and Ltx25TextFeatures, which is LTX's own norm +
// flatten + dual projection.
//
// ---- THE PADDING, which is where this differs from every other text
// ---- encoder in the tree ----
//
// LTX-2.5 pads its captions on the LEFT, to 1024 tokens
// (LTXGemmaTokenizer is constructed with PaddingSide.LEFT and
// TOKENIZER_MAX_LENGTH = 1024). So the real tokens are a SUFFIX of the
// context, not a prefix, and `valid` is a range rather than a count.
//
// The Gemma forward, though, runs on the REAL TOKENS ONLY -- no padding
// at all. That is not an approximation:
//
//   * Gemma's attention is causal and its positional information is
//     entirely relative (RoPE), so shifting every real token left by the
//     pad width leaves every query-key relationship identical.
//   * The reference masks the pads out, so no real token ever attends to
//     one. Running without them attends to exactly the same key set.
//   * Sliding-window layers see the same real keys either way: a window
//     is a fixed span of RELATIVE offsets, and the pads it would have
//     covered are masked.
//   * The pads' own hidden states are discarded -- LTX zeroes those rows
//     after the per-token RMS and before the projection.
//
// This matters because the host's Gemma has no prefix key-mask, and with
// LEFT padding it would need the opposite of one anyway (block the FIRST
// k keys, not the last). Encoding unpadded sidesteps the question rather
// than approximating it.
class Ltx25TextEncoder {
public:
  // The reference's TOKENIZER_MAX_LENGTH. Also the context width the DiT
  // sees, and a whole number of the connector's 128-register tiles.
  static constexpr int kMaxTokens = 1024;

  static std::unique_ptr<Ltx25TextEncoder>
  load(const Config& cfg, std::shared_ptr<vpipe::genai::WeightSet> ws,
       const MetalOps& ops, vpipe::metal_compute::MetalCompute* mc,
       const vpipe::SessionContextIntf* session, std::string* err);

  // Encode one caption into the two contexts.
  //
  // `pad_to` is the padded context width (kMaxTokens for the reference);
  // it must be a multiple of the connector's register count. Writes bf16
  // [pad_to][4096] and [pad_to][2048], both allocated by the caller.
  //
  // Reports the real-token RANGE it wrote, because the DiT needs it and
  // it is a suffix -- a caller that assumes a prefix builds a mask that
  // keeps precisely the empty rows.
  bool encode(const std::string& prompt, int pad_to,
              const vpipe::metal_compute::SharedBuffer& video_ctx,
              const vpipe::metal_compute::SharedBuffer& audio_ctx,
              int* valid_begin, int* valid_count, std::string* err);

  // The token ids a caption becomes, BOS included and truncated -- what
  // encode() will run. Exposed so a test can pin the tokenization
  // without paying for a 12B forward.
  std::vector<std::int32_t> tokenize(const std::string& prompt,
                                     int max_tokens) const;

  // The LM behind the encoder, so a test can compare the RAW hidden
  // states against a reference forward. Everything downstream of it is
  // verified against its own goldens; without this, the 12B forward is
  // the one link in the text path checked only against itself.
  vpipe::genai::HiddenStateEncoder* lm() { return _lm.get(); }

  int layers() const { return _feat ? _feat->layers() : 0; }
  int hidden_dim() const { return _feat ? _feat->hidden_dim() : 0; }

private:
  Ltx25TextEncoder() = default;

  std::shared_ptr<vpipe::genai::WeightSet>            _ws;
  std::unique_ptr<vpipe::genai::Tokenizer>            _tok;
  std::unique_ptr<vpipe::genai::HiddenStateEncoder>   _lm;
  std::unique_ptr<Ltx25TextFeatures>                  _feat;
  const vpipe::SessionContextIntf*                    _session = nullptr;
  std::int32_t                                        _bos = -1;
  int                                                 _reserved = 0;
  // The [valid][D][L] f32 staging the projections read. Sized to the
  // longest caption seen, not to pad_to: only the REAL rows are
  // materialised (1024 padded rows would be 771 MB of zeros).
  std::vector<float>                                  _stage;
};

}  // namespace ltx25

#endif
