#include "ltx25-text-encoder.h"

#include "stages/model-memory.h"

#include <cstdlib>

#include "generative-models/shared/comfy-checkpoint.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <filesystem>

#include <algorithm>
#include <cstring>

using vpipe::FlexData;
using vpipe::fmt;
using vpipe::genai::HiddenStateEncoderArgs;
using vpipe::genai::HiddenStateEncoderRegistry;
using vpipe::genai::HiddenTapResult;
using vpipe::genai::Tokenizer;
using vpipe::genai::WeightSet;
using vpipe::metal_compute::MetalCompute;
using vpipe::metal_compute::SharedBuffer;

namespace ltx25 {

namespace {

// bf16 -> f32. The tap comes back in the model's compute dtype and the
// per-token RMS runs in f32, so this is where the widening happens.
inline float
from_bf16_(std::uint16_t b)
{
  const std::uint32_t w = (std::uint32_t)b << 16;
  float f;
  std::memcpy(&f, &w, 4);
  return f;
}

inline float
from_f16_(std::uint16_t h)
{
  const std::uint32_t sign = (std::uint32_t)(h >> 15) << 31;
  const std::uint32_t exp  = (h >> 10) & 0x1f;
  const std::uint32_t man  = h & 0x3ff;
  std::uint32_t w = 0;
  if (exp == 0) {
    w = sign;
  } else if (exp == 0x1f) {
    w = sign | 0x7f800000u | (man << 13);
  } else {
    w = sign | ((exp + 112) << 23) | (man << 13);
  }
  float f;
  std::memcpy(&f, &w, 4);
  return f;
}

}  // namespace

std::unique_ptr<Ltx25TextEncoder>
Ltx25TextEncoder::load(const Config& cfg, std::shared_ptr<WeightSet> ws,
                       const MetalOps& ops, MetalCompute* mc,
                       const vpipe::SessionContextIntf* session,
                       std::string* err)
{
  auto fail = [&](std::string m) -> std::unique_ptr<Ltx25TextEncoder> {
    if (err != nullptr) { *err = std::move(m); }
    return nullptr;
  };
  if (cfg.enc_file.empty()) {
    return fail("this checkpoint has no text_encoders component");
  }
  if (mc == nullptr || !mc->valid()) {
    return fail("no usable metal-compute backend");
  }

  std::unique_ptr<Ltx25TextEncoder> e(new Ltx25TextEncoder());
  e->_ws      = std::move(ws);
  e->_session = session;

  // The LTX-specific half first: it is cheap and it tells us how many
  // layers the projection was built for, which is what the LM must
  // supply.
  std::string ferr;
  e->_feat = Ltx25TextFeatures::load(cfg.dit, *e->_ws, ops, &ferr);
  if (!e->_feat) { return fail("text features: " + ferr); }

  // ---- the Gemma forward, through the host's generic seam -----------
  //
  // The config is NESTED: this file's `__metadata__` carries LTX's own
  // fields alongside a `gemma_config` object, and there is no
  // config.json anywhere in the pack. Unwrapping it here and handing the
  // result in is exactly what HiddenStateEncoderArgs::config is for --
  // the host cannot know this layout.
  //
  // A QUANTIZED encoder is a directory instead, and then there is
  // nothing to unwrap: `model-quantize` writes the same `gemma_config`
  // out as an ordinary config.json (see comfy_output_config), with its
  // own `quantization` block beside it. Leaving `a.config` empty is what
  // makes the registry read that file -- and it has to be empty rather
  // than a config this side reconstructed, because the quantization
  // block is the part only the written file knows.
  FlexData gcfg;
  if (!std::filesystem::is_directory(cfg.enc_file)) {
    std::string merr;
    if (!vpipe::genai::comfy::metadata_json(cfg.enc_file, kEncMetaKey, gcfg,
                                            &merr)) {
      return fail("the text encoder's `" + std::string(kEncMetaKey) +
                  "` metadata is unreadable: " + merr);
    }
  }

  HiddenStateEncoderArgs a;
  a.dir     = cfg.enc_file;
  a.metal   = mc;
  a.session = session;
  a.config  = gcfg;
  // STREAM THE BACKBONE when the graph has no room to hold it.
  //
  // This encoder runs ONE prefill per prompt and is then destroyed, which
  // is what makes the trade sane here and not for a chat model: the cost
  // is re-reading the stack once, against a DiT that re-reads its own
  // per denoise step. MEASURED at w8: 15.3 GB resident and a 16.14 GB
  // load peak, on a box where the whole run has 16.
  //
  // The same rule the DiT uses, so the two answer the memory question
  // the same way -- and asked in the DENOISE phase for the same reason
  // it is there: what matters is what has to coexist, and this encoder
  // is gone by then.
  {
    namespace mm = vpipe::model_memory;
    // The DiT is generate-video's, not this stage's, and it has already
    // declared it -- so naming it here resolves it a SECOND time, from a
    // config that has this stage's `encoder_variant` but not that
    // stage's `variant`. Only what THIS caller is about to hold.
    const auto plan = mm::plan_streaming(session, /*dit_dir=*/std::string(),
                                         cfg.enc_file, mm::kStreamHeadroom);
    a.stream_layers = plan.stream;
    a.pin_frac      = plan.pin_frac;
    if (const char* e = std::getenv("VPIPE_LTX25_STREAM_ENCODER")) {
      a.stream_layers = (std::atoi(e) != 0);
      if (!a.stream_layers) { a.pin_frac = 0.0; }
    }
    if (a.stream_layers && session != nullptr) {
      session->info(vpipe::fmt(
          "ltx-2.5: streaming the text encoder's layers (pin_frac {:.3f}) "
          "-- it prefills once per prompt and is dropped after",
          a.pin_frac));
    }
  }
  std::string oerr;
  e->_lm = HiddenStateEncoderRegistry::get().open(a, &oerr);
  if (!e->_lm) { return fail("text encoder: " + oerr); }

  // LTX stacks EVERY hidden state -- 48 layers plus the embedding
  // output -- into one 188160-wide projection. A model that can only
  // serve part of the stack cannot drive this checkpoint, and the
  // failure would otherwise be a silently short concatenation.
  const int want = e->_feat->layers();               // 49
  if (e->_lm->max_tap_index() + 1 != want) {
    return fail(fmt("the projection was built for {} hidden states but "
                    "this encoder serves {} (layers 0..{} of {}); LTX-2.5 "
                    "needs the WHOLE stack",
                    want, e->_lm->max_tap_index() + 1,
                    e->_lm->max_tap_index(), e->_lm->n_layers())());
  }
  if (e->_lm->hidden_dim() != e->_feat->hidden_dim()) {
    return fail(fmt("the projection expects a {}-wide hidden state, the "
                    "encoder produces {}",
                    e->_feat->hidden_dim(), e->_lm->hidden_dim())());
  }

  // ---- the tokenizer, which ships INSIDE the checkpoint -------------
  //
  // read() rather than tensor(): 32 MB of JSON that is parsed once and
  // dropped. Caching it would keep a second copy alive next to the
  // parsed tokenizer for the life of the weight set.
  SharedBuffer tj = e->_ws->read("tokenizer_json", mc,
                                 WeightSet::Residency::Copied);
  if (tj.empty()) {
    return fail("the text encoder carries no `tokenizer_json` tensor");
  }
  const std::string_view json(static_cast<const char*>(tj.contents()),
                              tj.byte_size());
  e->_tok = Tokenizer::from_huggingface_string(json, "ltx-2.5/gemma",
                                               session);
  if (!e->_tok) { return fail("the embedded tokenizer.json did not parse"); }

  // Gemma-4 does NOT emit BOS from its post-processor -- the reference
  // prepends it explicitly and refuses to encode without one, so a
  // missing id is a hard failure rather than a skipped token.
  e->_bos = e->_tok->special_token_id("<bos>");
  if (e->_bos < 0) {
    return fail("the tokenizer has no <bos>, which the encode path "
                "requires (Gemma-4 does not add it automatically)");
  }
  return e;
}

std::vector<std::int32_t>
Ltx25TextEncoder::tokenize(const std::string& prompt, int max_tokens) const
{
  std::vector<std::int32_t> ids;
  if (!_tok) { return ids; }
  // The reference strips the caption before tokenizing.
  std::size_t b = prompt.find_first_not_of(" \t\n\r\f\v");
  std::size_t x = prompt.find_last_not_of(" \t\n\r\f\v");
  const std::string t =
      (b == std::string::npos) ? std::string()
                              : prompt.substr(b, x - b + 1);
  ids = _tok->encode(t);
  if (max_tokens > 0 && (int)ids.size() > max_tokens - 1) {
    ids.resize((std::size_t)max_tokens - 1);
  }
  if (ids.empty() || ids.front() != _bos) {
    ids.insert(ids.begin(), _bos);
  }
  if (max_tokens > 0 && (int)ids.size() > max_tokens) {
    ids.resize((std::size_t)max_tokens);
  }
  return ids;
}

bool
Ltx25TextEncoder::encode(const std::string& prompt, int pad_to,
                         const SharedBuffer& video_ctx,
                         const SharedBuffer& audio_ctx, int* valid_begin,
                         int* valid_count, std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  if (!_lm || !_feat || !_tok) { return fail("the encoder is not loaded"); }
  if (pad_to <= 0) { return fail("pad_to must be positive"); }

  const std::vector<std::int32_t> ids = tokenize(prompt, pad_to);
  const int n = (int)ids.size();
  if (n <= 0) { return fail("the caption tokenized to nothing"); }

  // ---- the Gemma forward, on the REAL tokens only -------------------
  HiddenTapResult tap;
  std::string terr;
  if (!_lm->encode(ids, _lm->available_indices(), &tap, &terr)) {
    return fail("the Gemma forward failed: " + terr);
  }
  const int L = tap.slots, D = tap.hidden;
  if (tap.tokens != n) {
    return fail(fmt("the encoder returned {} rows for {} tokens",
                    tap.tokens, n)());
  }
  if (L != _feat->layers() || D != _feat->hidden_dim()) {
    return fail(fmt("the tap is [{} x {}], the projection wants [{} x {}]",
                    L, D, _feat->layers(), _feat->hidden_dim())());
  }

  // ---- transpose [slot][pos][D] -> [pos][D][slot] -------------------
  //
  // LAYER-FASTEST, which is what the flatten downstream assumes and the
  // one thing here that is silently wrong when guessed the other way
  // (see ltx25-text-features.h).
  const std::size_t F = (std::size_t)D * L;
  if (_stage.size() < (std::size_t)n * F) { _stage.resize((std::size_t)n * F); }
  const auto* src = static_cast<const std::uint16_t*>(tap.data.contents());
  const bool bf16 = (tap.dtype != "f16");
  for (int l = 0; l < L; ++l) {
    const std::uint16_t* sl = src + (std::size_t)l * n * D;
    for (int t = 0; t < n; ++t) {
      const std::uint16_t* sr = sl + (std::size_t)t * D;
      float* dst = _stage.data() + (std::size_t)t * F + l;
      for (int d = 0; d < D; ++d) {
        dst[(std::size_t)d * L] = bf16 ? from_bf16_(sr[d]) : from_f16_(sr[d]);
      }
    }
  }

  // ---- LEFT padding: the caption is a SUFFIX ------------------------
  const int begin = pad_to - n;
  if (_reserved != pad_to) {
    std::string rerr;
    if (!_feat->reserve(pad_to, &rerr)) { return fail("reserve: " + rerr); }
    _reserved = pad_to;
  }
  std::string cerr;
  if (!_feat->compute(_stage.data(), pad_to, begin, n, video_ctx, audio_ctx,
                      &cerr)) {
    return fail("the projections failed: " + cerr);
  }
  if (valid_begin != nullptr) { *valid_begin = begin; }
  if (valid_count != nullptr) { *valid_count = n; }
  if (_session != nullptr) {
    _session->info(fmt("ltx-2.5: caption -> {} tokens at rows {}..{} of {}",
                       n, begin, pad_to - 1, pad_to));
  }
  return true;
}

}  // namespace ltx25
