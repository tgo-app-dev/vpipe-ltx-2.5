#include "ltx25-text-features.h"
#include "ltx25-dit-weights.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using vpipe::genai::WeightSet;
using vpipe::metal_compute::SharedBuffer;

namespace ltx25 {

namespace {

std::uint16_t
to_bf16_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  const std::uint32_t r = ((u >> 16) & 1u) + 0x7fffu;
  return (std::uint16_t)((u + r) >> 16);
}

// The projections live at the TEXT ENCODER file's root, with no
// `model.diffusion_model.` prefix -- that one is the DiT's.
constexpr const char* kProj = "text_embedding_projection.";

}  // namespace

std::unique_ptr<Ltx25TextFeatures>
Ltx25TextFeatures::load(const DitConfig& cfg, WeightSet& ws,
                        const MetalOps& ops, std::string* err)
{
  std::unique_ptr<Ltx25TextFeatures> t(new Ltx25TextFeatures());
  t->_ops = &ops;
  t->_hidden = cfg.caption_channels;              // 3840, the Gemma width
  t->_video_dim = cfg.cross_attention_dim;        // 4096
  t->_audio_dim = cfg.audio_cross_attention_dim;  // 2048

  auto get = [&](const std::string& n, SharedBuffer& out) {
    if (!ws.has(n)) {
      if (err != nullptr) { *err = "missing '" + n + "'"; }
      return false;
    }
    out = ws.tensor(n, ops.mc(), WeightSet::Residency::Mapped);
    return !out.empty();
  };
  if (!get(std::string(kProj) + "video_aggregate_embed.weight", t->_vw) ||
      !get(std::string(kProj) + "video_aggregate_embed.bias",   t->_vb) ||
      !get(std::string(kProj) + "audio_aggregate_embed.weight", t->_aw) ||
      !get(std::string(kProj) + "audio_aggregate_embed.bias",   t->_ab)) {
    return nullptr;
  }
  // The LAYER COUNT is derived from the projection's own width rather
  // than assumed: 188160 / 3840 = 49, which is 48 layers plus the
  // embedding output. A checkpoint whose encoder has a different depth
  // would otherwise be read with a silently wrong stride.
  const std::size_t flat = t->_vw.byte_size() / 2 / (std::size_t)t->_video_dim;
  if (flat % (std::size_t)t->_hidden != 0) {
    if (err != nullptr) {
      *err = "the video projection is " + std::to_string(flat) +
             " wide, which is not a multiple of the encoder's " +
             std::to_string(t->_hidden);
    }
    return nullptr;
  }
  t->_layers = (int)(flat / (std::size_t)t->_hidden);
  return t;
}

bool
Ltx25TextFeatures::reserve(int tokens, std::string* err)
{
  if (tokens <= 0) {
    if (err != nullptr) { *err = "tokens must be positive"; }
    return false;
  }
  _tokens = tokens;
  const std::size_t flat = (std::size_t)tokens * flat_dim();
  _normed = _ops->alloc(flat);
  _v_in   = _ops->alloc(flat);
  _a_in   = _ops->alloc(flat);
  return true;
}

bool
Ltx25TextFeatures::compute(const float* hidden, int tokens,
                           int valid_begin, int valid_count,
                           const SharedBuffer& video_out,
                           const SharedBuffer& audio_out, std::string* err)
{
  if (tokens != _tokens) {
    if (err != nullptr) {
      *err = "reserve() sized for " + std::to_string(_tokens) +
             " tokens, compute wants " + std::to_string(tokens);
    }
    return false;
  }
  if (hidden == nullptr) {
    if (err != nullptr) { *err = "no hidden states"; }
    return false;
  }
  const int D = _hidden, L = _layers;
  const int F = D * L;

  // The norm and the flatten, on the host in f32.
  //
  // It is O(tokens x 188160) -- 2.3 M values for a 12-token caption, 48 M
  // for a padded 256 -- against a 770 M-parameter GEMM that follows, so
  // the cost is in the projection and not here. Doing it in f32 keeps
  // the per-token RMS exact, which matters because each layer is
  // normalised on its own and a bf16 reduction over 3840 values would
  // lose more than the projection does.
  const float rv = std::sqrt((float)_video_dim / (float)D);
  const float ra = std::sqrt((float)_audio_dim / (float)D);
  auto* np = static_cast<std::uint16_t*>(_normed.contents());
  auto* vp = static_cast<std::uint16_t*>(_v_in.contents());
  auto* ap = static_cast<std::uint16_t*>(_a_in.contents());

  for (int t = 0; t < tokens; ++t) {
    // `hidden` holds the VALID rows only, so row t lives at
    // t - valid_begin. Indexing it by the absolute t would mean a
    // [1024][3840][49] f32 staging buffer -- 771 MB of mostly
    // zeros for a caption that is a dozen tokens long.
    const float* h = hidden + (std::size_t)(t - valid_begin) * F;
    std::uint16_t* nrow = np + (std::size_t)t * F;
    std::uint16_t* vrow = vp + (std::size_t)t * F;
    std::uint16_t* arow = ap + (std::size_t)t * F;
    if (t < valid_begin || t >= valid_begin + valid_count) {
      // Padding is ZEROED after the norm, so it contributes nothing to
      // the projection. Zeroing the input instead would leave the bias.
      std::memset(nrow, 0, (std::size_t)F * 2);
      std::memset(vrow, 0, (std::size_t)F * 2);
      std::memset(arow, 0, (std::size_t)F * 2);
      continue;
    }
    // One RMS per (token, LAYER): the reduction is over D with the
    // layer held fixed, and `hidden` is [D][L], so the stride is L.
    for (int l = 0; l < L; ++l) {
      double ss = 0.0;
      for (int d = 0; d < D; ++d) {
        const double v = h[(std::size_t)d * L + l];
        ss += v * v;
      }
      const float inv = (float)(1.0 / std::sqrt(ss / (double)D + 1e-6));
      for (int d = 0; d < D; ++d) {
        // LAYER-FASTEST: the flat index is d*L + l, which is exactly
        // where this value already sits in `hidden`. The reshape is a
        // no-op on the layout and a trap on the assumption.
        const std::size_t k = (std::size_t)d * L + l;
        const float n = h[k] * inv;
        nrow[k] = to_bf16_(n);
        vrow[k] = to_bf16_(n * rv);
        arow[k] = to_bf16_(n * ra);
      }
    }
  }

  auto stream = _ops->mc()->make_command_stream();
  {
    auto enc = stream.begin_compute();
    _ops->linear(enc, _v_in, _vw, &_vb, video_out, tokens, F, _video_dim);
    _ops->linear(enc, _a_in, _aw, &_ab, audio_out, tokens, F, _audio_dim);
  }
  stream.commit().wait();
  return true;
}

}  // namespace ltx25
