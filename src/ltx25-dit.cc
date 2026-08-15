#include "ltx25-dit.h"

#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <chrono>
#include <future>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <unistd.h>

using vpipe::genai::WeightSet;
using vpipe::metal_compute::SharedBuffer;

namespace ltx25 {

namespace {

float
bf16_to_f32_(std::uint16_t h)
{
  const std::uint32_t u = (std::uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

std::uint16_t
f32_to_bf16_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  const std::uint32_t r = ((u >> 16) & 1u) + 0x7fffu;
  return (std::uint16_t)((u + r) >> 16);
}

// y[o] = sum_i x[i] * W[o][i] + b[o], with W and b bf16 in UMA memory.
// One row only -- this is the adaLN chain, see the note in the header.
void
gemv_host_(const std::vector<float>& x, const SharedBuffer& w,
           const SharedBuffer& b, int in_dim, int out_dim,
           std::vector<float>& y)
{
  const auto* wp = static_cast<const std::uint16_t*>(w.contents());
  const auto* bp = b.empty() ? nullptr
                             : static_cast<const std::uint16_t*>(b.contents());
  y.assign((std::size_t)out_dim, 0.0f);
  for (int o = 0; o < out_dim; ++o) {
    const std::uint16_t* row = wp + (std::size_t)o * in_dim;
    double acc = bp != nullptr ? (double)bf16_to_f32_(bp[o]) : 0.0;
    for (int i = 0; i < in_dim; ++i) {
      acc += (double)x[(std::size_t)i] * bf16_to_f32_(row[i]);
    }
    y[(std::size_t)o] = (float)acc;
  }
}

void
silu_(std::vector<float>& v)
{
  for (auto& z : v) { z = z / (1.0f + std::exp(-z)); }
}

// The sinusoidal timestep projection: 256 channels, max_period 10000,
// downscale_freq_shift 0, and flip_sin_to_cos TRUE -- so the row is
// [cos | sin], NOT the usual [sin | cos]. Emitting them the other way
// round is a clean forward with every modulation driven by the wrong
// half of the embedding.
std::vector<float>
timestep_sinusoid_(double t, int channels = 256)
{
  const int half = channels / 2;
  std::vector<float> sin_part((std::size_t)half), cos_part((std::size_t)half);
  const double log_period = std::log(10000.0);
  for (int i = 0; i < half; ++i) {
    // downscale_freq_shift = 0, so the divisor is `half`, not half-1.
    const double exponent = -log_period * (double)i / (double)half;
    const double a = t * std::exp(exponent);
    sin_part[(std::size_t)i] = (float)std::sin(a);
    cos_part[(std::size_t)i] = (float)std::cos(a);
  }
  std::vector<float> out;
  out.reserve((std::size_t)channels);
  out.insert(out.end(), cos_part.begin(), cos_part.end());
  out.insert(out.end(), sin_part.begin(), sin_part.end());
  return out;
}

}  // namespace

std::vector<float>
Ltx25Dit::adaln_(const DitTrunk::AdaLN& a, double timestep,
                 std::vector<float>* embedded) const
{
  std::vector<float> proj = timestep_sinusoid_(timestep, 256);
  std::vector<float> h1, h2, out;
  gemv_host_(proj, a.emb1_w, a.emb1_b, 256, a.dim, h1);
  silu_(h1);
  gemv_host_(h1, a.emb2_w, a.emb2_b, a.dim, a.dim, h2);
  // h2 IS `embedded_timestep`: the embedder's output, before the SiLU
  // and the final projection. The output head wants this one, not the
  // k*dim driver -- they are different tensors and the same call
  // produces both.
  if (embedded != nullptr) { *embedded = h2; }
  std::vector<float> act = h2;
  silu_(act);
  gemv_host_(act, a.out_w, a.out_b, a.dim, a.out_dim, out);
  return out;
}

void
Ltx25Dit::compute_adaln_step_(double sigma, double audio_sigma,
                              BakedStep* b) const
{
  // The timestep the sinusoid sees is sigma * timestep_scale_multiplier
  // (1000), not sigma. Feeding the raw [0,1] sigma leaves every
  // frequency in the embedding near zero and the whole schedule
  // collapses to one modulation.
  const DitConfig& c = _cfg;
  const double m = (double)c.timestep_scale_multiplier;

  // ONE CHAIN PER DENOISE LEVEL. The reference runs the timestep MLP on
  // `denoise_mask * sigma` flattened over every token; the mask takes
  // `_v_levels.size()` distinct values, so this is the same arithmetic
  // with the repetition removed. Level 0 is 1.0 -- the tokens being
  // generated -- so an unconditioned generation is one chain, exactly
  // what it was.
  b->v_ts.resize(_v_levels.size());
  b->v_embedded.resize(_v_levels.size());
  for (std::size_t g = 0; g < _v_levels.size(); ++g) {
    b->v_ts[g] = adaln_(_trunk.video, sigma * _v_levels[g] * m,
                        &b->v_embedded[g]);
  }
  // The PROMPT driver, per stream: prompt_adaln(sigma * multiplier). It
  // is what makes the text cross-attention's K/V modulation depend on
  // the step; without it the K/V are constant across the schedule.
  //
  // The SCALAR sigma, not the per-token timesteps: the reference feeds
  // `modality.sigma` here, so the caption is modulated once for the
  // whole stream however the tokens are conditioned.
  b->v_pts = adaln_(_trunk.prompt, sigma * m, nullptr);

  if (!_have_audio) { return; }

  b->a_ts.resize(_a_levels.size());
  b->a_embedded.resize(_a_levels.size());
  for (std::size_t g = 0; g < _a_levels.size(); ++g) {
    b->a_ts[g] = adaln_(_trunk.audio, audio_sigma * _a_levels[g] * m,
                        &b->a_embedded[g]);
  }
  b->a_pts = adaln_(_trunk.audio_prompt, audio_sigma * m, nullptr);

  // The a2v/v2a drivers. Note WHICH sigma feeds which: the scale/shift
  // comes from the stream's OWN noise level, the GATE from the OTHER
  // stream's -- how much of the other modality to let in depends on how
  // noisy that modality currently is. Swapping them is a forward that
  // runs and a coupling that means nothing.
  const double f = c.av_ca_timestep_scale_multiplier / m;
  b->v_css = adaln_(_trunk.av_video_ss,  sigma * m, nullptr);
  b->a_css = adaln_(_trunk.av_audio_ss,  audio_sigma * m, nullptr);
  b->v_cg  = adaln_(_trunk.av_a2v_gate,  audio_sigma * m * f, nullptr);
  b->a_cg  = adaln_(_trunk.av_v2a_gate,  sigma * m * f, nullptr);
}

bool
Ltx25Dit::bake_adaln(const std::vector<double>& sigmas, std::string* err)
{
  auto fail = [&](const std::string& msg) {
    if (err != nullptr) { *err = msg; }
    return false;
  };
  if (_baked) { return true; }
  if (sigmas.empty()) { return fail("bake_adaln: empty schedule"); }
  // The bake runs one chain per (step, DENOISE LEVEL), so it needs the
  // levels -- which come from set_geometry. Baking first would silently
  // bake the unconditioned single level and then refuse every
  // conditioned forward, with the weights already released.
  if (!_geometry_set) {
    return fail("bake_adaln: set_geometry() must run first -- the bake is "
                "per denoise level and the levels are part of the "
                "geometry");
  }

  // LTX steps both streams on ONE schedule -- there is no per-modality
  // sigma shift in this model -- so a step index names both sigmas.
  _baked_steps.resize(sigmas.size());
  for (std::size_t i = 0; i < sigmas.size(); ++i) {
    compute_adaln_step_(sigmas[i], sigmas[i], &_baked_steps[i]);
  }

  // Count what the per-step chains were touching, then clear the
  // handles. The count is WORK AVOIDED, not memory freed: these are
  // Mapped views the WeightSet also caches, so dropping them here
  // returns no RSS. Clearing is enforcement -- after the bake nothing
  // may read them, and an empty buffer makes that a crash rather than a
  // silent success.
  const DitTrunk::AdaLN* all[] = {
      &_trunk.video, &_trunk.audio, &_trunk.prompt, &_trunk.audio_prompt,
      &_trunk.av_video_ss, &_trunk.av_audio_ss,
      &_trunk.av_a2v_gate, &_trunk.av_v2a_gate};
  std::uint64_t freed = 0;
  for (const DitTrunk::AdaLN* a : all) {
    if (!a->valid) { continue; }
    const SharedBuffer* bufs[] = {&a->emb1_w, &a->emb1_b, &a->emb2_w,
                                  &a->emb2_b, &a->out_w,  &a->out_b};
    for (const SharedBuffer* sb : bufs) { freed += sb->byte_size(); }
  }
  for (DitTrunk::AdaLN* a : {&_trunk.video, &_trunk.audio, &_trunk.prompt,
                             &_trunk.audio_prompt, &_trunk.av_video_ss,
                             &_trunk.av_audio_ss, &_trunk.av_a2v_gate,
                             &_trunk.av_v2a_gate}) {
    a->emb1_w = SharedBuffer(); a->emb1_b = SharedBuffer();
    a->emb2_w = SharedBuffer(); a->emb2_b = SharedBuffer();
    a->out_w  = SharedBuffer(); a->out_b  = SharedBuffer();
    a->valid  = false;
  }
  _baked_freed = freed;
  _baked_v_levels = _v_levels;
  _baked_a_levels = _a_levels;
  _baked = true;
  return true;
}

namespace {

// Bring a block's weight pages back into RAM, by touching one byte per
// page.
//
// WHAT IS BEING FAULTED depends on the pack, and the two are not the
// same cost:
//
//   w8g64 / w4g64   sharded, and every tensor lands 16-byte aligned, so
//                   the WeightSet maps them. These are clean file pages:
//                   the kernel drops them for free and a touch re-reads
//                   from the file (or from the buffer cache).
//   bf16            ONE file whose data section starts at 677624 == 8
//                   (mod 16), so every one of its 4349 tensors is
//                   unaligned, load_mapped refuses, and all 39 GB come
//                   in as COPIES. These are anonymous, so a touch is a
//                   decompress or a swap-in, not a file read.
//
// Not madvise(MADV_WILLNEED): it is advisory, and only the anonymous
// case has anything for it to advise about anyway. A read per page is
// what actually makes the page present, and it is cheap.
//
// `volatile` because the sum is dead and every optimiser knows it; the
// whole point of the loop is its side effect on residency.
void
warm_pages_(const GpuBlockWeights& w)
{
  const std::size_t page = (std::size_t)::getpagesize();
  volatile std::uint8_t sink = 0;
  for_each_weight(w, [&](const SharedBuffer& b) {
    if (b.empty()) { return; }
    const auto* p = static_cast<const std::uint8_t*>(b.contents());
    if (p == nullptr) { return; }
    const std::size_t n = b.byte_size();
    for (std::size_t off = 0; off < n; off += page) { sink = sink ^ p[off]; }
    if (n > 0) { sink = sink ^ p[n - 1]; }
  });
  (void)sink;
}

// Is enough of this block missing to be worth warming? Sampled coarsely
// -- a block is faulted in as a whole, so one page in 64 answers it --
// and it is what keeps the prefetch free on a box where the weights
// never leave RAM: there, this returns false for every block and no
// thread is ever started.
bool
block_is_cold_(const GpuBlockWeights& w)
{
  std::size_t ex = 0, ic = 0;
  for_each_weight(w, [&](const SharedBuffer& b) {
    if (b.empty()) { return; }
    const auto r = b.page_residency(64);
    if (!r.valid) { return; }
    ex += r.examined;
    ic += r.incore;
  });
  // 95%, not 100%: a handful of pages missing is measurement noise and
  // not worth a thread, while a block that was genuinely dropped reads
  // at a few percent (MEASURED: 1.0-17% on the first step of the 39 GB
  // bf16 pack, against 100% on every step after it).
  return ex > 0 && (double)ic < 0.95 * (double)ex;
}

}  // namespace

std::unique_ptr<Ltx25Dit>
Ltx25Dit::load(const Config& cfg, WeightSet& ws, const MetalOps& ops,
               bool stream_blocks, std::string* err, bool with_connectors)
{
  std::unique_ptr<Ltx25Dit> d(new Ltx25Dit());
  d->_ops = &ops;
  d->_cfg = cfg.dit;
  d->_have_audio = cfg.dit.use_audio_video_cross_attention;

  if (!bind_trunk(ws, ops.mc(), cfg.dit, d->_trunk, err)) { return nullptr; }
  d->_has_connector =
      ws.has(std::string(kDitPrefix) +
             "video_embeddings_connector.learnable_registers") ||
      ws.has(std::string(kDitPrefix) +
             "audio_embeddings_connector.learnable_registers");

  if (with_connectors && d->_has_connector) {
    d->_v_conn = Ltx25Connector::load(cfg.dit, ws, ops, false, err);
    if (!d->_v_conn) { return nullptr; }
    if (d->_have_audio) {
      d->_a_conn = Ltx25Connector::load(cfg.dit, ws, ops, true, err);
      if (!d->_a_conn) { return nullptr; }
    }
  }

  d->_blocks.reserve((std::size_t)cfg.dit.num_layers);
  for (int i = 0; i < cfg.dit.num_layers; ++i) {
    GpuBlockWeights gw;
    if (!bind_block(ws, ops.mc(), cfg.dit, i, stream_blocks, gw, err)) {
      return nullptr;
    }
    // A quantized checkpoint on a host whose affine kernels did not
    // resolve must fail HERE. Letting it through would dispatch an
    // invalid ComputeFunction -- a silent no-op -- so every block would
    // run at full cost and leave its output buffer untouched, which
    // reads downstream as a model that generates noise rather than as a
    // missing kernel.
    if (gw.quant_group != 0 && !ops.quant_available()) {
      if (err != nullptr) {
        *err = "block " + std::to_string(i) + " is quantized (group " +
               std::to_string(gw.quant_group) + ") but this host has no "
               "usable affine_qmm_steel_bf16 library";
      }
      return nullptr;
    }
    d->_quant_group = gw.quant_group;
    auto b = MetalBlock::create(ops, std::move(gw), err);
    if (!b) { return nullptr; }
    d->_blocks.push_back(std::move(b));
  }
  return d;
}

bool
Ltx25Dit::set_geometry(int latent_frames, int latent_h, int latent_w,
                       int audio_tokens, int text_tokens,
                       const RopeGeometry& geo, std::string* err,
                       const Conditioning& cond)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  if (latent_frames <= 0 || latent_h <= 0 || latent_w <= 0) {
    return fail("latent geometry must be positive");
  }
  if (cond.v_levels.empty() || cond.v_levels[0] != 1.0) {
    return fail("the video denoise levels must start at 1.0 -- level 0 is "
                "the tokens being generated");
  }
  if (cond.a_levels.empty() || cond.a_levels[0] != 1.0) {
    return fail("the audio denoise levels must start at 1.0");
  }
  if (!cond.v_extra_starts.empty() &&
      (cond.v_extra_starts.size() != 3 || cond.v_extra_ends.size() != 3)) {
    return fail("appended video tokens need 3 position axes");
  }
  // A geometry change is free; a LEVEL change after the bake is not, so
  // it is refused here rather than discovered as wrong modulation. The
  // adaLN projections were released by the bake, so there is nothing
  // left to recompute the new levels from.
  if (_baked &&
      (cond.v_levels != _baked_v_levels || cond.a_levels != _baked_a_levels)) {
    return fail("the adaLN schedule was baked for a different set of denoise "
                "levels, and the projections that would rebake it have been "
                "released. Build a fresh DiT for a request whose "
                "conditioning strengths differ.");
  }
  _frames = latent_frames;
  _lh = latent_h;
  _lw = latent_w;
  _v_target = latent_frames * latent_h * latent_w;
  _a_target = _have_audio ? audio_tokens : 0;
  const int v_extra = cond.v_extra();
  const int a_extra = _have_audio ? cond.a_extra() : 0;
  _video_tokens = _v_target + v_extra;
  _audio_tokens = _a_target + a_extra;
  _text_tokens = text_tokens;
  _v_levels = cond.v_levels;
  _a_levels = cond.a_levels;
  if (!cond.v_level.empty() &&
      (int)cond.v_level.size() != _video_tokens) {
    return fail("the video level index is " +
                std::to_string(cond.v_level.size()) + " long but the stream "
                "has " + std::to_string(_video_tokens) + " tokens");
  }
  if (!cond.a_level.empty() && (int)cond.a_level.size() != _audio_tokens) {
    return fail("the audio level index is " +
                std::to_string(cond.a_level.size()) + " long but the stream "
                "has " + std::to_string(_audio_tokens) + " tokens");
  }

  const DitConfig& c = _cfg;
  const int vd = c.inner_dim(), ad = c.audio_inner_dim();

  // The four tables. The two SELF tables carry each stream's own axes;
  // the two CROSS tables are TIME-ONLY and at the AUDIO width for BOTH
  // streams -- see ltx25-rope.h.
  //
  // The target grid's spans first, then whatever the conditioning
  // appended -- one table over the whole sequence, because attention
  // does not distinguish them.
  std::vector<std::vector<double>> vs, ve;
  video_spans(_frames, _lh, _lw, geo, /*pixel_frame_offset=*/0,
              /*single_pixel_frame=*/false, geo.causal_fix, &vs, &ve);
  for (int a = 0; a < 3 && v_extra > 0; ++a) {
    vs[(std::size_t)a].insert(vs[(std::size_t)a].end(),
                              cond.v_extra_starts[(std::size_t)a].begin(),
                              cond.v_extra_starts[(std::size_t)a].end());
    ve[(std::size_t)a].insert(ve[(std::size_t)a].end(),
                              cond.v_extra_ends[(std::size_t)a].begin(),
                              cond.v_extra_ends[(std::size_t)a].end());
  }
  _v_self = build_rope(vs, ve, c.positional_embedding_max_pos, vd,
                       c.num_attention_heads, c.positional_embedding_theta,
                       c.rope_f64(), err);
  if (_v_self.tokens == 0) { return fail("could not build the video RoPE"); }
  if (_have_audio && _audio_tokens > 0) {
    // The audio self table: the target's own seconds, then the appended
    // reference tokens' -- which sit on a time base the caller chose
    // (the reference's dub pipeline places them at NEGATIVE seconds, so
    // they cannot collide with the clip being generated).
    std::vector<double> as0 = audio_time_seconds(_a_target);
    std::vector<double> as1 = audio_time_seconds(_a_target + 1);
    std::vector<std::vector<double>> asx(1), aex(1);
    asx[0] = as0;
    aex[0].assign(as1.begin() + (as1.empty() ? 0 : 1), as1.end());
    if (a_extra > 0) {
      asx[0].insert(asx[0].end(), cond.a_extra_starts.begin(),
                    cond.a_extra_starts.end());
      aex[0].insert(aex[0].end(), cond.a_extra_ends.begin(),
                    cond.a_extra_ends.end());
    }
    _a_self = build_rope(asx, aex, c.audio_positional_embedding_max_pos, ad,
                         c.audio_num_attention_heads,
                         c.positional_embedding_theta, c.rope_f64(), err);
    const int cross_max = c.positional_embedding_max_pos.empty()
                              ? 20 : c.positional_embedding_max_pos[0];
    // The video stream's cross positions are the SAME seconds its self
    // table used -- axis 0 of the spans just built, appended tokens
    // included -- not the latent frame index. With the index the two
    // streams sat on different time bases (a video latent frame is
    // 0.33 s at 24 fps, an audio token 0.04 s) and the coupling attended
    // to the wrong offsets.
    _v_cross = build_cross_rope(vs[0], ve[0], cross_max, ad,
                                c.audio_num_attention_heads,
                                c.positional_embedding_theta, c.rope_f64());
    _a_cross = _a_self;   // audio's cross table IS its self table: both
                          // are the time axis at the audio width
  }

  const int levels =
      (int)std::max(_v_levels.size(), _a_levels.size());
  // ONE arena for the whole stack. The blocks run strictly sequentially
  // -- a command stream each, committed and waited on before the next --
  // and nothing in the scratch survives a forward, so 48 private arenas
  // were 47 copies of dead memory. See BlockScratch.
  _scratch.reset();
  for (auto& b : _blocks) {
    b->set_rope(&_v_self, _have_audio ? &_a_self : nullptr,
                _have_audio ? &_v_cross : nullptr,
                _have_audio ? &_a_cross : nullptr);
    if (!b->reserve(_video_tokens, _audio_tokens, _text_tokens, err,
                    levels, &_scratch)) {
      return false;
    }
  }

  // The per-token level index, int32 for the kernels. Left empty when
  // there is one level -- the blocks then never look at it, and the
  // whole per-token path stays off.
  _v_level_buf = SharedBuffer();
  _a_level_buf = SharedBuffer();
  if (_v_levels.size() > 1 && !cond.v_level.empty()) {
    _v_level_buf = _ops->mc()->make_shared_buffer(
        (std::size_t)_video_tokens * sizeof(std::int32_t));
    auto* p = static_cast<std::int32_t*>(_v_level_buf.contents());
    for (int t = 0; t < _video_tokens; ++t) {
      p[t] = (std::int32_t)cond.v_level[(std::size_t)t];
    }
  }
  if (_a_levels.size() > 1 && !cond.a_level.empty() && _audio_tokens > 0) {
    _a_level_buf = _ops->mc()->make_shared_buffer(
        (std::size_t)_audio_tokens * sizeof(std::int32_t));
    auto* p = static_cast<std::int32_t*>(_a_level_buf.contents());
    for (int t = 0; t < _audio_tokens; ++t) {
      p[t] = (std::int32_t)cond.a_level[(std::size_t)t];
    }
  }

  const MetalOps& o = *_ops;
  _vx    = o.alloc((std::size_t)_video_tokens * vd);
  _vlat  = o.alloc((std::size_t)_video_tokens * c.in_channels);
  _v_out = o.alloc((std::size_t)_video_tokens * c.out_channels);
  // One 9*dim driver and one 2*dim head scale/shift PER DENOISE LEVEL.
  // The head's is per level too because `embedded_timestep` -- which it
  // adds to `scale_shift_out` -- comes out of the same per-token chain.
  _vts   = o.alloc((std::size_t)_v_levels.size() * 9 * vd);
  _v_css = o.alloc((std::size_t)4 * vd);
  _v_cg  = o.alloc((std::size_t)vd);
  _v_head_ss = o.alloc((std::size_t)_v_levels.size() * 2 * vd);
  _v_pts = o.alloc((std::size_t)2 * vd);
  _tmp   = o.alloc((std::size_t)_video_tokens * vd);
  _vctx  = o.alloc((std::size_t)_text_tokens * c.cross_attention_dim);
  if (_v_conn != nullptr) {
    _vctx_raw = o.alloc((std::size_t)_text_tokens * c.cross_attention_dim);
    if (!_v_conn->reserve(_text_tokens, err)) { return false; }
  }
  if (_have_audio && _audio_tokens > 0) {
    _ax    = o.alloc((std::size_t)_audio_tokens * ad);
    _alat  = o.alloc((std::size_t)_audio_tokens * c.in_channels);
    _a_out = o.alloc((std::size_t)_audio_tokens * c.audio_out_channels);
    _ats   = o.alloc((std::size_t)_a_levels.size() * 9 * ad);
    _a_css = o.alloc((std::size_t)4 * ad);
    _a_cg  = o.alloc((std::size_t)ad);
    _a_head_ss = o.alloc((std::size_t)_a_levels.size() * 2 * ad);
    _a_pts = o.alloc((std::size_t)2 * ad);
    _actx = o.alloc((std::size_t)_text_tokens * c.audio_cross_attention_dim);
    if (_a_conn != nullptr) {
      _actx_raw =
          o.alloc((std::size_t)_text_tokens * c.audio_cross_attention_dim);
      if (!_a_conn->reserve(_text_tokens, err)) { return false; }
    }
  }
  _geometry_set = true;
  return true;
}

bool
Ltx25Dit::forward(const Input& in, Output* out, std::string* err)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  if (!_geometry_set) { return fail("set_geometry() has not run"); }
  if (in.video == nullptr) { return fail("no video latent"); }
  if (out == nullptr) { return fail("no output"); }

  const DitConfig& c = _cfg;
  const int vd = c.inner_dim(), ad = c.audio_inner_dim();
  const int zc = c.in_channels;
  const bool ra = _have_audio && in.audio != nullptr && _audio_tokens > 0;

  // ---- the adaLN drivers ----------------------------------------------
  //
  // Either computed here on the host, or looked up from the schedule
  // bake_adaln() precomputed. The arithmetic is identical -- the bake
  // just ran it earlier, once, and then let the 852 MB of projections
  // go.
  BakedStep local;
  const BakedStep* mod = nullptr;
  if (_baked) {
    if (in.step < 0 || (std::size_t)in.step >= _baked_steps.size()) {
      return fail("adaLN is baked, so the request must name a step in "
                  "[0, " + std::to_string(_baked_steps.size()) +
                  "); got " + std::to_string(in.step) +
                  ". The projections that would compute it have been "
                  "released.");
    }
    mod = &_baked_steps[(std::size_t)in.step];
    // A baked step carries the audio chains only if the bake saw audio
    // weights. Running audio against a video-only bake would modulate
    // the audio stream with whatever the vectors happen to hold.
    if (ra && mod->a_ts.empty()) {
      return fail("adaLN was baked without the audio chains");
    }
  } else {
    compute_adaln_step_(in.sigma, in.audio_sigma, &local);
    mod = &local;
  }

  {
    auto put = [](const SharedBuffer& b, const std::vector<float>& v,
                  std::size_t off = 0) {
      auto* p = static_cast<std::uint16_t*>(b.contents()) + off;
      for (std::size_t i = 0; i < v.size(); ++i) { p[i] = f32_to_bf16_(v[i]); }
    };
    // The drivers, LEVEL-MAJOR: level g's 9*dim block at g*9*dim. That
    // is the stride MetalBlock::ada3_ walks.
    for (std::size_t g = 0; g < mod->v_ts.size(); ++g) {
      put(_vts, mod->v_ts[g], g * 9 * (std::size_t)vd);
    }
    put(_v_pts, mod->v_pts);
    if (ra) {
      for (std::size_t g = 0; g < mod->a_ts.size(); ++g) {
        put(_ats, mod->a_ts[g], g * 9 * (std::size_t)ad);
      }
      put(_a_pts, mod->a_pts);
      put(_v_css, mod->v_css);
      put(_a_css, mod->a_css);
      put(_v_cg, mod->v_cg);
      put(_a_cg, mod->a_cg);
    }
  }

  // The output head's scale/shift: table[0] + embedded and
  // table[1] + embedded -- the SAME dim-wide embedding added to both
  // rows, not a sliced projection. Per LEVEL, because `embedded` came
  // out of the per-token chain: a conditioned token leaves the head
  // modulated as clean too, not just the blocks.
  {
    const auto* tab =
        static_cast<const std::uint16_t*>(_trunk.scale_shift_out.contents());
    auto* p = static_cast<std::uint16_t*>(_v_head_ss.contents());
    for (std::size_t g = 0; g < mod->v_embedded.size(); ++g) {
      for (int r = 0; r < 2; ++r) {
        for (int i = 0; i < vd; ++i) {
          const std::size_t k = (std::size_t)r * vd + i;
          p[g * 2 * (std::size_t)vd + k] =
              f32_to_bf16_(bf16_to_f32_(tab[k]) +
                           mod->v_embedded[g][(std::size_t)i]);
        }
      }
    }
  }
  if (ra) {
    const auto* tab = static_cast<const std::uint16_t*>(
        _trunk.audio_scale_shift_out.contents());
    auto* p = static_cast<std::uint16_t*>(_a_head_ss.contents());
    for (std::size_t g = 0; g < mod->a_embedded.size(); ++g) {
      for (int r = 0; r < 2; ++r) {
        for (int i = 0; i < ad; ++i) {
          const std::size_t k = (std::size_t)r * ad + i;
          p[g * 2 * (std::size_t)ad + k] =
              f32_to_bf16_(bf16_to_f32_(tab[k]) +
                           mod->a_embedded[g][(std::size_t)i]);
        }
      }
    }
  }

  // ---- patchify ------------------------------------------------------
  //
  // The latent arrives CHANNEL-major [z][F][H][W]; a token is one cell
  // (the DiT's patch is 1x1x1, all the spatial packing being the VAE's
  // patch_size 4). So this is a transpose to [T][z] and then one
  // Linear. Done on the host during upload: it is z*T elements, and the
  // alternative is a kernel that exists only to reorder the step's
  // smallest tensor.
  {
    auto* p = static_cast<std::uint16_t*>(_vlat.contents());
    for (int t = 0; t < _video_tokens; ++t) {
      for (int ch = 0; ch < zc; ++ch) {
        p[(std::size_t)t * zc + ch] =
            f32_to_bf16_(in.video[(std::size_t)ch * _video_tokens + t]);
      }
    }
  }
  if (ra) {
    auto* p = static_cast<std::uint16_t*>(_alat.contents());
    for (int t = 0; t < _audio_tokens; ++t) {
      for (int ch = 0; ch < zc; ++ch) {
        p[(std::size_t)t * zc + ch] =
            f32_to_bf16_(in.audio[(std::size_t)ch * _audio_tokens + t]);
      }
    }
  }

  // The context arrives as bf16 the caller owns; copy it into the
  // buffer the stack borrows for the whole forward.
  const int n_valid = in.n_valid_text > 0 ? in.n_valid_text : _text_tokens;
  if (_v_conn != nullptr) {
    // The caption arrives RAW; connect it here. Positions past
    // `n_valid` become the connector's learnable registers.
    if (in.context != nullptr) {
      std::memcpy(_vctx_raw.contents(), in.context,
                  (std::size_t)_text_tokens * c.cross_attention_dim * 2);
    }
    if (!_v_conn->forward(_vctx_raw, _vctx, _text_tokens, n_valid, err)) {
      return false;
    }
    if (ra && _a_conn != nullptr && in.audio_context != nullptr) {
      std::memcpy(_actx_raw.contents(), in.audio_context,
                  (std::size_t)_text_tokens * c.audio_cross_attention_dim * 2);
      if (!_a_conn->forward(_actx_raw, _actx, _text_tokens, n_valid, err)) {
        return false;
      }
    }
  } else {
    if (in.context != nullptr) {
      std::memcpy(_vctx.contents(), in.context,
                  (std::size_t)_text_tokens * c.cross_attention_dim * 2);
    }
    if (ra && in.audio_context != nullptr) {
      std::memcpy(_actx.contents(), in.audio_context,
                  (std::size_t)_text_tokens * c.audio_cross_attention_dim * 2);
    }
  }

  const MetalOps& o = *_ops;
  auto stream = o.mc()->make_command_stream();
  {
    auto enc = stream.begin_compute();
    o.linear(enc, _vlat, _trunk.patchify_w, &_trunk.patchify_b, _vx,
             _video_tokens, zc, vd);
    // The KEYFRAME absolute-position embedding, immediately after
    // patchify_proj and nowhere else. It marks the tokens whose latent
    // encodes ONE standalone pixel frame -- and because the video VAE is
    // causal, that is the target's first latent frame in EVERY
    // generation, conditioned or not. The reference marks it
    // unconditionally (`_first_frame_keyframes_mask`), which is why this
    // is not gated on conditioning: leaving it out drops a trained
    // [1, 4096] vector from every forward this port has ever run.
    //
    // Appended conditioning tokens are deliberately NOT marked -- given
    // content is ordinary image guidance, and only generated keyframe
    // slots (which this port has none of) carry the marker.
    if (_cfg.use_keyframes_abs_pos_embedding &&
        _trunk.keyframes_abs_pos.byte_size() > 0) {
      o.add_row_prefix(enc, _vx, _trunk.keyframes_abs_pos, vd, _lh * _lw);
    }
    if (ra) {
      o.linear(enc, _alat, _trunk.audio_patchify_w, &_trunk.audio_patchify_b,
               _ax, _audio_tokens, zc, ad);
    }
  }
  stream.commit().wait();

  // ---- the block stack -----------------------------------------------
  //
  // Committed per block rather than as one enormous command buffer: 48
  // blocks x ~60 dispatches is 2900 encodes, and a single buffer that
  // large both delays the first work and makes a mid-stack abort
  // impossible. Per block is also where `progress` can answer.
  GpuStreamInput gv, ga;
  gv.x = &_vx;
  gv.context = &_vctx;
  gv.timesteps = &_vts;
  gv.cross_scale_shift = &_v_css;
  gv.cross_gate = &_v_cg;
  gv.prompt_timestep = &_v_pts;
  gv.tokens = _video_tokens;
  gv.text_tokens = _text_tokens;
  gv.n_levels = (int)_v_levels.size();
  gv.level = (_v_level_buf.byte_size() > 0) ? &_v_level_buf : nullptr;
  ga.x = &_ax;
  ga.context = &_actx;
  ga.timesteps = &_ats;
  ga.cross_scale_shift = &_a_css;
  ga.cross_gate = &_a_cg;
  ga.prompt_timestep = &_a_pts;
  ga.tokens = _audio_tokens;
  ga.text_tokens = _text_tokens;
  ga.present = ra;
  ga.n_levels = (int)_a_levels.size();
  ga.level = (_a_level_buf.byte_size() > 0) ? &_a_level_buf : nullptr;

  // ---- per-block instrumentation (VPIPE_LTX25_BLOCK_PROFILE=1) -------
  //
  // What it answers: this model's blocks are bound ONCE, at load, and
  // never re-read, so its only per-forward memory cost is pages having
  // been EVICTED between steps and having to come back. That cost is
  // invisible in a wall clock -- it lands inside the block's own
  // commit/wait, with the GPU waiting on it -- so the probe reports,
  // per block and BEFORE it is encoded, how much of it is still in RAM.
  //
  // Resident on arrival means there is nothing for a prefetch to hide,
  // and that is the answer on any box the checkpoint fits. Run this
  // first on a box where it does not.
  static const bool kBlkProf =
      std::getenv("VPIPE_LTX25_BLOCK_PROFILE") != nullptr;
  double bp_ms = 0.0;
  std::size_t bp_examined = 0, bp_incore = 0, bp_cold_blocks = 0;

  // ---- weight prefetch (OPT-IN: VPIPE_LTX25_PREFETCH=1) -------------
  //
  // Warm block i+1's pages while the GPU runs block i -- the same idea,
  // in the same place, as the streamed-block prefetch in the host's
  // MiniMax-H3. What differs is what is being hidden, and it is worth
  // being exact about it because it decides whether this helps at all.
  //
  // H3 READS block L per forward, so it always has a read to move off
  // the critical path. This model does not: bind_block runs ONCE, at
  // load, and the blocks are then held for the model's lifetime. So the
  // only per-forward cost here is the pages having LEFT -- evicted
  // between one step and the next -- and on a box that can hold the
  // checkpoint they never do.
  //
  // MEASURED on a 64 GB M4 Pro, 512x320x9, 8 steps, both packs: blocks
  // arrive 100% resident on every step after the first, so this fires
  // 19 times on step 1 and 0 times thereafter, and the wall clock moves
  // by less than the run-to-run spread. It is OFF BY DEFAULT for that
  // reason -- a box this model fits has nothing for it to hide, and an
  // on-by-default knob that never fires is just a slower way to read
  // the same number.
  //
  // Where it should earn its keep is a box the checkpoint does NOT fit,
  // which is where H3 measured its own 4.1%: there every block is cold
  // every step. That case is NOT measured here -- deliberately, because
  // creating it means over-committing the machine during a DiT run, and
  // that is what panicked this box's kernel once already.
  //
  // Depth is structurally ONE: a single outstanding warm of a single
  // block, so nothing queues and the extra pressure is one block.
  //
  // DECLARATION ORDER: `fut` LAST, so it destroys FIRST. Its destructor
  // joins the worker, and the worker reads `_blocks` -- which every
  // early return out of this loop (abort, block failure) would otherwise
  // leave it racing against.
  static const bool pf_on =
      std::getenv("VPIPE_LTX25_PREFETCH") != nullptr;
  struct PrefetchSlot {
    int               block = -1;
    std::future<void> fut;
  } pf;
  int pf_started = 0, pf_hit = 0;
  for (int i = 0; i < (int)_blocks.size(); ++i) {
    if (in.progress && !in.progress(i, (int)_blocks.size())) {
      return fail("aborted at block " + std::to_string(i));
    }
    const auto t0 = kBlkProf ? std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::time_point{};
    if (kBlkProf) {
      // Stride 8: a page either survived or it did not, and the whole
      // block was faulted in by one contiguous read, so a sample of one
      // page in eight finds an evicted block just as well as a full walk
      // at an eighth of the mincore vector.
      std::size_t ex = 0, ic = 0;
      for_each_weight(_blocks[(std::size_t)i]->weights(),
                      [&](const SharedBuffer& b) {
                        if (b.empty()) { return; }
                        const auto r = b.page_residency(8);
                        if (!r.valid) { return; }
                        ex += r.examined;
                        ic += r.incore;
                      });
      bp_examined += ex;
      bp_incore += ic;
      if (ex > 0 && ic * 4 < ex * 3) { ++bp_cold_blocks; }   // < 75% in RAM
    }
    // The prefetch issued under block i-1's GPU work. Joining here
    // costs only the part that did not fit under that window.
    if (pf.block == i && pf.fut.valid()) {
      pf.fut.get();
      pf.block = -1;
      ++pf_hit;
    }
    auto s = o.mc()->make_command_stream();
    {
      auto enc = s.begin_compute();
      if (!_blocks[(std::size_t)i]->forward(enc, gv, ga, err)) { return false; }
    }
    // BETWEEN THE COMMIT AND THE WAIT is the whole opportunity: the GPU
    // is busy with block i and this thread has nothing to do.
    auto fence = s.commit();
    if (pf_on && pf.block < 0 && i + 1 < (int)_blocks.size()) {
      // Gated on paging() rather than on H3's fits_growth(). H3 is
      // deciding whether to ALLOCATE a second block; this decides
      // whether to fault clean file pages back in, which the kernel can
      // drop again for free and which cost no anonymous memory. What
      // both must refuse is a box already in distress, where warming
      // block i+1 evicts block i out from under the GPU still reading
      // it -- turning a hidden fault into a fault plus a re-fault.
      const auto mb = o.mc()->memory_budget();
      if (!mb.paging() &&
          block_is_cold_(_blocks[(std::size_t)(i + 1)]->weights())) {
        pf.block = i + 1;
        ++pf_started;
        pf.fut = std::async(std::launch::async, [this, i]() {
          warm_pages_(_blocks[(std::size_t)(i + 1)]->weights());
        });
      }
    }
    fence.wait();
    if (kBlkProf) {
      bp_ms += std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - t0).count();
    }
  }
  if (kBlkProf && o.mc()->session() != nullptr) {
    const auto mb = o.mc()->memory_budget();
    o.mc()->session()->log_normal(vpipe::fmt(
        "ltx-2.5: {} blocks in {:.0f} ms ({:.1f} ms/block); weight pages "
        "{:.1f}% resident on arrival ({} of {} sampled), {} block(s) below "
        "75%; box {} MB compressed, {} MB swap",
        _blocks.size(), bp_ms, bp_ms / (double)_blocks.size(),
        bp_examined > 0 ? 100.0 * (double)bp_incore / (double)bp_examined
                        : 100.0,
        bp_incore, bp_examined, bp_cold_blocks,
        mb.compressed >> 20, mb.swap_used >> 20));
    o.mc()->session()->log_normal(vpipe::fmt(
        "ltx-2.5: prefetch {}/{} hit", pf_hit, pf_started));
  }

  // ---- the output head ------------------------------------------------
  //
  // LayerNorm, NOT RMSNorm: `norm_out` is
  // `LayerNorm(inner_dim, elementwise_affine=False)`, so it subtracts
  // the mean. Every other normalisation in this model is RMS, which is
  // exactly why this one is easy to get wrong.
  {
    auto s = o.mc()->make_command_stream();
    {
      auto enc = s.begin_compute();
      o.layer_norm_plain(enc, _vx, _tmp, vd, _video_tokens);
      // The head's table is [levels][2][dim] with SHIFT first, so the
      // scale sits one dim-wide row in. `modulate_g` then strides by
      // 2*dim per level -- which is why the level stride N it is given
      // is 2*dim's worth on both operands, not dim's.
      if (gv.level != nullptr && gv.n_levels > 1) {
        o.modulate_g(enc, _tmp, _v_head_ss, (std::size_t)vd * 2,
                     _v_head_ss, 0, *gv.level, _tmp, vd, _video_tokens,
                     /*level_stride=*/2 * vd);
      } else {
        o.modulate_off(enc, _tmp, _v_head_ss, (std::size_t)vd * 2,
                       _v_head_ss, 0, _tmp, vd, _video_tokens);
      }
      o.linear(enc, _tmp, _trunk.proj_out_w, &_trunk.proj_out_b, _v_out,
               _video_tokens, vd, c.out_channels);
      if (ra) {
        o.layer_norm_plain(enc, _ax, _tmp, ad, _audio_tokens);
        if (ga.level != nullptr && ga.n_levels > 1) {
          o.modulate_g(enc, _tmp, _a_head_ss, (std::size_t)ad * 2,
                       _a_head_ss, 0, *ga.level, _tmp, ad, _audio_tokens,
                       /*level_stride=*/2 * ad);
        } else {
        o.modulate_off(enc, _tmp, _a_head_ss, (std::size_t)ad * 2,
                       _a_head_ss, 0, _tmp, ad, _audio_tokens);
        }
        o.linear(enc, _tmp, _trunk.audio_proj_out_w,
                 &_trunk.audio_proj_out_b, _a_out, _audio_tokens, ad,
                 c.audio_out_channels);
      }
    }
    s.commit().wait();
  }

  // Back to CHANNEL-major, the shape the sampler and the VAE expect.
  out->video.assign((std::size_t)c.out_channels * _video_tokens, 0.0f);
  {
    const auto* p = static_cast<const std::uint16_t*>(_v_out.contents());
    for (int t = 0; t < _video_tokens; ++t) {
      for (int ch = 0; ch < c.out_channels; ++ch) {
        out->video[(std::size_t)ch * _video_tokens + t] =
            bf16_to_f32_(p[(std::size_t)t * c.out_channels + ch]);
      }
    }
  }
  if (ra) {
    out->audio.assign((std::size_t)c.audio_out_channels * _audio_tokens, 0.0f);
    const auto* p = static_cast<const std::uint16_t*>(_a_out.contents());
    for (int t = 0; t < _audio_tokens; ++t) {
      for (int ch = 0; ch < c.audio_out_channels; ++ch) {
        out->audio[(std::size_t)ch * _audio_tokens + t] =
            bf16_to_f32_(p[(std::size_t)t * c.audio_out_channels + ch]);
      }
    }
  } else {
    out->audio.clear();
  }
  return true;
}

}  // namespace ltx25
