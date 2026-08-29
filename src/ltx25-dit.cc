#include "ltx25-dit.h"

#include "ltx25-text-encoder.h"

#include "apple-silicon/metal-compute/shared-buffer.h"

#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"

#include <chrono>
#include <cstdio>
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

// y += B @ (A @ x) -- one adapted host GEMV, in the same f32 the chain
// it rides on runs in.
//
// A and B are bf16 in UMA memory exactly as the base weight is, so this
// reads them in place rather than converting a copy. `rank` mat-vecs
// down and `n` up: at rank 450 against a 4096x36864 base that is ~2.4%
// of the chain, and the chain is 0.2 GFLOP.
void
lora_gemv_host_(const std::vector<float>& x, const LoraPair& p,
                std::vector<float>& y)
{
  if (!p.valid()) { return; }
  if ((int)x.size() != p.k || (int)y.size() != p.n) { return; }
  const auto* ap = static_cast<const std::uint16_t*>(p.a.contents());
  const auto* bp = static_cast<const std::uint16_t*>(p.b.contents());
  if (ap == nullptr || bp == nullptr) { return; }
  std::vector<float> r((std::size_t)p.rank, 0.0f);
  for (int j = 0; j < p.rank; ++j) {
    const std::uint16_t* row = ap + (std::size_t)j * p.k;
    double acc = 0.0;
    for (int i = 0; i < p.k; ++i) {
      acc += (double)x[(std::size_t)i] * bf16_to_f32_(row[i]);
    }
    r[(std::size_t)j] = (float)acc;
  }
  for (int o = 0; o < p.n; ++o) {
    const std::uint16_t* row = bp + (std::size_t)o * p.rank;
    double acc = 0.0;
    for (int j = 0; j < p.rank; ++j) {
      acc += (double)r[(std::size_t)j] * bf16_to_f32_(row[j]);
    }
    y[(std::size_t)o] += (float)acc;
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
                 std::vector<float>* embedded, const LoraAdaLN* la) const
{
  std::vector<float> proj = timestep_sinusoid_(timestep, 256);
  std::vector<float> h1, h2, out;
  gemv_host_(proj, a.emb1_w, a.emb1_b, 256, a.dim, h1);
  // Each delta goes in BEFORE the activation that follows it, which is
  // where a fused weight would have put it.
  if (la != nullptr) { lora_gemv_host_(proj, la->emb1, h1); }
  silu_(h1);
  gemv_host_(h1, a.emb2_w, a.emb2_b, a.dim, a.dim, h2);
  if (la != nullptr) { lora_gemv_host_(h1, la->emb2, h2); }
  // h2 IS `embedded_timestep`: the embedder's output, before the SiLU
  // and the final projection. The output head wants this one, not the
  // k*dim driver -- they are different tensors and the same call
  // produces both.
  if (embedded != nullptr) { *embedded = h2; }
  std::vector<float> act = h2;
  silu_(act);
  gemv_host_(act, a.out_w, a.out_b, a.dim, a.out_dim, out);
  if (la != nullptr) { lora_gemv_host_(act, la->out, out); }
  return out;
}

const LoraAdaLN*
Ltx25Dit::lora_for_(const DitTrunk::AdaLN& a) const
{
  const LoraTrunk* lt = (_lora != nullptr) ? _lora->trunk() : nullptr;
  if (lt == nullptr) { return nullptr; }
  if (&a == &_trunk.video) { return &lt->video; }
  if (&a == &_trunk.audio) { return &lt->audio; }
  if (&a == &_trunk.prompt) { return &lt->prompt; }
  if (&a == &_trunk.audio_prompt) { return &lt->audio_prompt; }
  if (&a == &_trunk.av_video_ss) { return &lt->av_video_ss; }
  if (&a == &_trunk.av_audio_ss) { return &lt->av_audio_ss; }
  if (&a == &_trunk.av_a2v_gate) { return &lt->av_a2v_gate; }
  if (&a == &_trunk.av_v2a_gate) { return &lt->av_v2a_gate; }
  return nullptr;
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
  // Every chain below takes its own adapter through lora_for_, which is
  // also what adaln_public uses -- so the test hook and the forward can
  // never route differently.

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
                        &b->v_embedded[g], lora_for_(_trunk.video));
  }
  // The PROMPT driver, per stream: prompt_adaln(sigma * multiplier). It
  // is what makes the text cross-attention's K/V modulation depend on
  // the step; without it the K/V are constant across the schedule.
  //
  // The SCALAR sigma, not the per-token timesteps: the reference feeds
  // `modality.sigma` here, so the caption is modulated once for the
  // whole stream however the tokens are conditioned.
  b->v_pts = adaln_(_trunk.prompt, sigma * m, nullptr,
                    lora_for_(_trunk.prompt));

  if (!_have_audio) { return; }

  b->a_ts.resize(_a_levels.size());
  b->a_embedded.resize(_a_levels.size());
  for (std::size_t g = 0; g < _a_levels.size(); ++g) {
    b->a_ts[g] = adaln_(_trunk.audio, audio_sigma * _a_levels[g] * m,
                        &b->a_embedded[g], lora_for_(_trunk.audio));
  }
  b->a_pts = adaln_(_trunk.audio_prompt, audio_sigma * m, nullptr,
                    lora_for_(_trunk.audio_prompt));

  // The a2v/v2a drivers. Note WHICH sigma feeds which: the scale/shift
  // comes from the stream's OWN noise level, the GATE from the OTHER
  // stream's -- how much of the other modality to let in depends on how
  // noisy that modality currently is. Swapping them is a forward that
  // runs and a coupling that means nothing.
  const double f = c.av_ca_timestep_scale_multiplier / m;
  b->v_css = adaln_(_trunk.av_video_ss, sigma * m, nullptr,
                    lora_for_(_trunk.av_video_ss));
  b->a_css = adaln_(_trunk.av_audio_ss, audio_sigma * m, nullptr,
                    lora_for_(_trunk.av_audio_ss));
  b->v_cg  = adaln_(_trunk.av_a2v_gate, audio_sigma * m * f, nullptr,
                    lora_for_(_trunk.av_a2v_gate));
  b->a_cg  = adaln_(_trunk.av_v2a_gate, sigma * m * f, nullptr,
                    lora_for_(_trunk.av_v2a_gate));
}

void
Ltx25Dit::trunk_lora_(vpipe::metal_compute::ComputeEncoder& enc,
                      const LoraPair* p, const SharedBuffer& x,
                      const SharedBuffer& y, int M) const
{
  if (p == nullptr || !p->valid() || M <= 0 || !_scratch) { return; }
  _ops->lora_add(enc, x, p->a, p->b, y, _scratch->lora_r, _scratch->lora_d,
                 M, p->k, p->n, p->rank);
}

bool
Ltx25Dit::set_lora(std::shared_ptr<const LoraAdapter> lora, std::string* err)
{
  auto fail = [&](const std::string& msg) {
    if (err != nullptr) { *err = msg; }
    return false;
  };
  if (lora == _lora) { return true; }
  // The bake has already released the adaLN projections, so eight of
  // this adapter's chains would have nothing to adapt. Refused rather
  // than partly applied -- see the note on set_lora in the header.
  if (_baked) {
    return fail("set_lora: the adaLN bake has already run and released "
                "the projections; the adapter would be applied to the "
                "blocks and not to the modulation");
  }
  if (lora != nullptr && lora->layers() != num_layers()) {
    return fail("set_lora: the adapter was built for " +
                std::to_string(lora->layers()) + " blocks but this DiT has " +
                std::to_string(num_layers()));
  }
  _lora = std::move(lora);
  // GROW THE ARENA IF ONE ALREADY EXISTS. A caller that set the geometry
  // first is not refused -- the planes are appended to the shared arena
  // and every block already holding it sees them, because it is one
  // object. Blocks that do not exist yet (the streamed ones) are built
  // inside the forward and reserve with lora_rank_() themselves.
  if (_scratch) {
    for (auto& b : _blocks) {
      if (!b) { continue; }
      if (!b->reserve(_video_tokens, _audio_tokens, _text_tokens, err,
                      _geo_levels, &_scratch, lora_rank_(), lora_out_())) {
        return false;
      }
    }
    for (auto& b : _slot) {
      if (!b) { continue; }
      if (!b->reserve(_video_tokens, _audio_tokens, _text_tokens, err,
                      _geo_levels, &_scratch, lora_rank_(), lora_out_())) {
        return false;
      }
    }
  }
  return true;
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
  // handles. On a PRELOADING model the count is work avoided rather than
  // memory freed -- the chains are Mapped views the WeightSet also
  // caches. On a STREAMING one they are this model's own bytes
  // (kept_residency), so the release is real. Clearing is enforcement
  // either way: after the bake nothing may read them, and an empty
  // buffer makes that a crash rather than a silent success.
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
  // UNWIRE BEFORE CLEARING. Destroying a wired buffer unwires it in the
  // kernel but not in the pool's counter, so these chains would be
  // charged to the pool for the rest of the run while no longer
  // existing -- and the pool would fill with bytes nothing holds.
  if (_wired_fixed) {
    if (auto* mgr = manager_()) {
      for (const DitTrunk::AdaLN* a : all) {
        const SharedBuffer* bufs[] = {&a->emb1_w, &a->emb1_b, &a->emb2_w,
                                      &a->emb2_b, &a->out_w,  &a->out_b};
        for (const SharedBuffer* sb : bufs) {
          mgr->unwire_from_pool(const_cast<SharedBuffer&>(*sb));
        }
      }
    }
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

  // THE GROUND MOVED, and the residency policy cannot see it.
  //
  // Whatever the ratchet concluded about this box a moment ago was
  // measured against a load this model no longer carries. That matters
  // because the ratchet is deliberately slow to climb -- one block per
  // three quiet forwards -- and this schedule is 8 steps, so a shed
  // taken during the bake would stand for the whole run and leave the
  // box half empty. MiniMax-H3 resets it here for the same reason and a
  // bigger one (13.2 GB of per-step projections against this model's
  // 0.85 GB).
  //
  // Deliberately not automatic: a caller that resets on nothing in
  // particular has simply turned the ratchet off. This is the one moment
  // the model KNOWS.
  if (_stream_blocks && freed > 0) { _resid.note_landscape_changed(); }
  _baked_freed = freed;
  _baked_v_levels = _v_levels;
  _baked_a_levels = _a_levels;
  _baked = true;
  return true;
}

namespace {


// Is enough of this block missing to be worth warming? Sampled coarsely
// -- a block is faulted in as a whole, so one page in 64 answers it --
// and it is what keeps the prefetch free on a box where the weights
// never leave RAM: there, this returns false for every block and no
// thread is ever started.
// Bytes one block holds, for the affordability question the prefetch
// asks before allocating a second one.
std::size_t
block_bytes_(const MetalBlock* b)
{
  if (b == nullptr) { return 0; }
  std::size_t n = 0;
  for_each_weight(b->weights(), [&](const SharedBuffer& x) {
    n += x.byte_size();
  });
  return n;
}

}  // namespace

std::unique_ptr<Ltx25Dit>
Ltx25Dit::load(const Config& cfg, std::shared_ptr<WeightSet> ws_in,
               const MetalOps& ops, bool stream_blocks,
               int plan_w, int plan_h, int plan_frames, std::string* err,
               bool with_connectors)
{
  if (!ws_in) {
    if (err != nullptr) { *err = "no weight set"; }
    return nullptr;
  }
  WeightSet& ws = *ws_in;
  std::unique_ptr<Ltx25Dit> d(new Ltx25Dit());
  d->_ops = &ops;
  d->_cfg = cfg.dit;
  d->_ws = std::move(ws_in);
  d->_stream_blocks = stream_blocks;
  d->_have_audio = cfg.dit.use_audio_video_cross_attention;

  // The trunk is KEPT for the model's life and read every step, so its
  // residency follows the model's verdict -- Copied when streaming, so a
  // 2.8 GB trunk is not competing with the streamed tail for page cache.
  const auto kept = kept_residency(stream_blocks);
  if (!bind_trunk(ws, ops.mc(), cfg.dit, d->_trunk, err, kept)) {
    return nullptr;
  }
  d->_has_connector =
      ws.has(std::string(kDitPrefix) +
             "video_embeddings_connector.learnable_registers") ||
      ws.has(std::string(kDitPrefix) +
             "audio_embeddings_connector.learnable_registers");

  if (with_connectors && d->_has_connector) {
    d->_v_conn = Ltx25Connector::load(cfg.dit, ws, ops, false, err, kept);
    if (!d->_v_conn) { return nullptr; }
    if (d->_have_audio) {
      d->_a_conn = Ltx25Connector::load(cfg.dit, ws, ops, true, err, kept);
      if (!d->_a_conn) { return nullptr; }
    }
  }

  // HOW MANY BLOCKS STAY RESIDENT.
  //
  // Not streaming: all of them, which is what every box that can hold
  // the checkpoint does. Streaming: a LEADING PREFIX sized by the shared
  // rule the other DiTs in the host tree use, so pinned + the in-flight
  // block + scratch stays inside a fraction of RAM. The tail is read per
  // forward and dropped.
  //
  // A prefix rather than a cache, and the difference is the reason this
  // works: a forward is a cyclic scan, so recency predicts nothing and
  // an LRU set of any size gives ~0% hits, while a FIXED subset gives
  // exactly its share. At least one, because the arena is sized from a
  // resident block and a stack with none could not run at all.
  d->_pinned = cfg.dit.num_layers;
  if (stream_blocks) {
    // ONE, and it is a STRUCTURAL minimum rather than a prefix.
    //
    // The fraction-of-RAM prefix this used to size is retired, for the
    // reason docs/MODEL-MEMORY.md gives: it was a share of TOTAL ram
    // decided before the run, blind to another process, to this graph's
    // peers, and to the moment a peer let go. What replaces it MEASURES
    // -- BlockResidency keeps a streamed block when the box turns out to
    // have room and gives it back the moment its pages are found outside
    // RAM. This port's own history is the argument: a hardcoded fraction
    // pinned 20 of 48 blocks on a 16 GB box against a plan that had
    // computed room for none.
    //
    // Why one and not zero, where every built-in DiT pins zero: the
    // block ARENA is allocated by MetalBlock::reserve() through a
    // resident block, so set_geometry over an empty stack would leave
    // `_scratch` null -- and with it scratch_bytes(), the residency
    // reserve, and the wired-pool charge. One block is what owns the
    // arena. It is not a memory decision and does not scale with the
    // box.
    d->_pinned = 1;
    if (ops.mc()->session() != nullptr) {
      ops.mc()->session()->log_normal(vpipe::fmt(
          "ltx-2.5: streaming {} blocks, 1 held to own the arena -- the "
          "resident set grows into free RAM as the denoise runs and is "
          "given back when the box needs it", cfg.dit.num_layers));
    }
  }

  // Sized to the FULL depth even when only the prefix is filled: an
  // empty slot is what the forward tests to decide whether to stream.
  d->_blocks.resize((std::size_t)cfg.dit.num_layers);
  for (int i = 0; i < d->_pinned; ++i) {
    GpuBlockWeights gw;
    // CACHED, because a pinned block is one the model KEEPS -- which is
    // the weight set's own rule, and it is what makes the manager's
    // accounting of this checkpoint true. Only the streamed tail is read
    // uncached, in build_block_, where it is genuinely consumed.
    if (!bind_block(ws, ops.mc(), cfg.dit, i, /*stream=*/false, gw, err,
                    kept)) {
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
    d->_blocks[(std::size_t)i] = std::move(b);
  }
  // A streamed stack still has to know its quantization group, and only
  // a BOUND block reports it -- so when the prefix is short of the whole
  // stack the value comes from the blocks that were built, which are
  // packed identically to the ones that were not.
  if (ops.mc() != nullptr && ops.mc()->session() != nullptr &&
      stream_blocks) {
    ops.mc()->session()->log_normal(vpipe::fmt(
        "ltx-2.5: streaming blocks -- {} of {} pinned resident, the rest "
        "read per forward", d->_pinned, cfg.dit.num_layers));
  }
  return d;
}

// The arena MetalBlock::reserve() will allocate, computed from the same
// widths it does. Kept next to nothing in particular on purpose: the
// authority is reserve(), and the test named in the header is what holds
// the two together.
std::uint64_t
Ltx25Dit::scratch_bytes(const DitConfig& cfg, int video_tokens,
                        int audio_tokens, int text_tokens, int levels)
{
  const bool have_audio = cfg.use_audio_video_cross_attention;
  const std::uint64_t vd = (std::uint64_t)cfg.inner_dim();
  const std::uint64_t ad = have_audio ? (std::uint64_t)cfg.audio_inner_dim()
                                      : 0;
  const std::uint64_t d = std::max(vd, ad);
  // Every stream's feed-forward is 4x its own width, so the widest is
  // 4*d -- the one buffer that is not a plane.
  const std::uint64_t f = 4 * d;
  const std::uint64_t t = (std::uint64_t)std::max(
      std::max(video_tokens, audio_tokens), text_tokens);
  const std::uint64_t l = (std::uint64_t)std::max(1, levels);
  if (d == 0 || t == 0) { return 0; }
  // BlockScratch owns the arithmetic; this only maps a DitConfig onto
  // the four widths reserve() reduces to.
  return BlockScratch::predict_bytes((std::size_t)t, (std::size_t)d,
                                     (std::size_t)f, (std::size_t)l);
}

std::size_t
Ltx25Dit::pinned_bytes() const noexcept
{
  std::size_t n = 0;
  for (const auto& b : _blocks) {
    if (!b) { continue; }
    for_each_weight(b->weights(), [&](const SharedBuffer& x) {
      n += x.byte_size();
    });
  }
  return n;
}

Ltx25Dit::~Ltx25Dit()
{
  if (!_wired_fixed) { return; }
  wire_fixed_(false);
  for (auto& b : _blocks) {
    if (b) { wire_block_(*b, false); }
  }
}

std::size_t
Ltx25Dit::held_weight_bytes() const noexcept
{
  std::size_t n = pinned_bytes();
  for_each_weight(_trunk, [&](const SharedBuffer& x) { n += x.byte_size(); });
  if (_v_conn) {
    _v_conn->for_each_weight([&](const SharedBuffer& x) {
      n += x.byte_size();
    });
  }
  if (_a_conn) {
    _a_conn->for_each_weight([&](const SharedBuffer& x) {
      n += x.byte_size();
    });
  }
  return n;
}

vpipe::genai::GenerativeModelManager*
Ltx25Dit::manager_() const
{
  const auto* mc = _ops != nullptr ? _ops->mc() : nullptr;
  if (mc == nullptr || mc->session() == nullptr) { return nullptr; }
  const auto* svc = mc->session()->services();
  return svc != nullptr ? svc->generative_model_manager() : nullptr;
}

// Every SCRATCH buffer this model holds -- the shared block arena and
// both connectors' working planes. Separated from wire_fixed_ because
// the scratch has a shorter life than everything else in it: a geometry
// change replaces it, and the replacement has to give the pool back what
// the old one was charged.
void
Ltx25Dit::each_scratch_(
    const std::function<void(SharedBuffer&)>& fn)
{
  if (!fn) { return; }
  if (_scratch) { _scratch->for_each_buffer(fn); }
  if (_v_conn) { _v_conn->for_each_scratch(fn); }
  if (_a_conn) { _a_conn->for_each_scratch(fn); }
}

// Give the pool back what the current scratch is charged, before that
// scratch is replaced.
//
// Destroying a wired buffer unwires it in the KERNEL but does not
// decrement the pool's counter -- only unwire_from_pool does -- so a
// geometry change used to leak a scratch's worth of budget. A pool that
// has lost budget to bytes nothing holds wires less of what comes next,
// and the resident set then shrinks for a reason nothing in the log
// names.
void
Ltx25Dit::unwire_scratch_()
{
  auto* mgr = manager_();
  if (mgr == nullptr || !_wired_fixed) { return; }
  std::size_t given = 0;
  each_scratch_([&](SharedBuffer& b) {
    if (b.byte_size() == 0 || !b.is_wired()) { return; }
    given += b.byte_size();
    mgr->unwire_from_pool(b);
  });
  if (given > 0 && _ops != nullptr && _ops->mc() != nullptr &&
      _ops->mc()->session() != nullptr) {
    _ops->mc()->session()->log_debug(vpipe::fmt(
        "ltx-2.5: released {} MB of wired scratch before resizing it",
        given >> 20));
  }
}

std::size_t
Ltx25Dit::wire_fixed_(bool on)
{
  auto* mgr = manager_();
  if (mgr == nullptr) { return 0; }
  std::size_t changed = 0;
  bool full = false;
  std::size_t unwirable = 0;
  auto one = [&](SharedBuffer& b) {
    if (b.byte_size() == 0 || b.is_wired() == on) { return; }
    if (full) { unwirable += b.byte_size(); return; }
    if (!on) { mgr->unwire_from_pool(b); changed += b.byte_size(); return; }
    const std::size_t got = mgr->wire_into_pool(b);
    if (got == 0) {
      // STOP, and keep what is already wired rather than unwinding it.
      // A partly wired model is partly protected, which is strictly
      // better than none -- and handing protection back on the way out
      // means competing for it again against a pool that just said no.
      full = true;
      unwirable += b.byte_size();
      return;
    }
    changed += got;
  };
  // THE SCRATCH FIRST. A forward cannot proceed without it, where a
  // resident block is an optimisation this model can shed -- so if the
  // pool runs out partway, it runs out on the half that had an
  // alternative.
  each_scratch_(one);
  // Then the TRUNK and the CONNECTORS, read on every block of every
  // forward and never shed.
  //
  // const_cast because the enumerations hand out const buffers -- they
  // exist for counting as well as wiring -- and wiring is a property of
  // the PAGES rather than of the bytes. Nothing here writes through the
  // pointer.
  for_each_weight(_trunk, [&](const SharedBuffer& b) {
    one(const_cast<SharedBuffer&>(b));
  });
  auto conn = [&](const std::unique_ptr<Ltx25Connector>& c) {
    if (!c) { return; }
    c->for_each_weight([&](const SharedBuffer& b) {
      one(const_cast<SharedBuffer&>(b));
    });
  };
  conn(_v_conn);
  conn(_a_conn);
  _unwirable = unwirable;
  return changed;
}

std::size_t
Ltx25Dit::wire_block_(MetalBlock& b, bool on)
{
  auto* mgr = manager_();
  if (mgr == nullptr) { return 0; }
  std::size_t changed = 0;
  for_each_weight(b.weights(), [&](const SharedBuffer& x) {
    if (x.byte_size() == 0 || x.is_wired() == on) { return; }
    SharedBuffer& m = const_cast<SharedBuffer&>(x);
    if (!on) { mgr->unwire_from_pool(m); changed += m.byte_size(); return; }
    changed += mgr->wire_into_pool(m);
  });
  return changed;
}

std::size_t
Ltx25Dit::wire_into_pool()
{
  const auto* sess = _ops != nullptr && _ops->mc() != nullptr
                         ? _ops->mc()->session() : nullptr;
  auto* mgr = manager_();
  if (mgr == nullptr) { return 0; }
  const std::size_t limit = mgr->wired_pool_limit();
  if (limit == 0) {
    // SAID, not skipped silently. Wiring off is a legitimate setting
    // (wired_pool_pct 0, or a box that granted nothing), and a run that
    // then sheds its resident set looks exactly like one whose policy
    // is broken. This is the line that tells the two apart.
    if (sess != nullptr && !_wired_reported) {
      _wired_reported = true;
      sess->info(vpipe::fmt(
          "ltx-2.5: the wired pool is off (wired_pool_pct={}), so resident "
          "blocks stay reclaimable and the compressor may take them back "
          "inside a forward", mgr->wired_pool_pct()));
    }
    return 0;
  }
  std::size_t n = wire_fixed_(true);
  _wired_fixed = true;
  // The blocks AFTER, and only the ones already held. A streamed block
  // is wired as it is admitted (see the residency loop), not here.
  for (auto& b : _blocks) {
    if (b) { n += wire_block_(*b, true); }
  }
  // Reported at INFO and on every geometry, including a zero. A pool
  // that silently refused reads in the log exactly like one that was
  // never asked, and the difference is what decides whether a shed
  // resident set is the policy working or the pool failing.
  if (sess != nullptr && (n > 0 || !_wired_reported)) {
    _wired_reported = true;
    // The REFUSED bytes too. A partly wired model is partly protected,
    // and the half outside the pool is the half the compressor may take
    // -- so a silent partial wire reads in the log exactly like a
    // complete one, and the difference shows up later as a shed
    // resident set with no stated cause. Zero is the ordinary case and
    // is said so plainly.
    sess->info(vpipe::fmt(
        "ltx-2.5: wired {} MB this pass{}; the pool holds {} MB of {} MB",
        n >> 20,
        _unwirable > 0
            ? vpipe::fmt(" ({} MB REFUSED and left reclaimable)",
                         _unwirable >> 20)()
            : std::string(),
        mgr->wired_pool_used() >> 20, limit >> 20));
  }
  return n;
}

void
Ltx25Dit::set_residency_reserve(std::size_t bytes)
{
  _resid.set_reserve(bytes);
}

std::size_t
Ltx25Dit::evict_tail_block_(bool allow_pinned)
{
  // From the TAIL. Highest index first because the forward is about to
  // start again at 0: giving back the block furthest from the next use
  // is the one choice that is right whatever the schedule does next.
  const int floor = allow_pinned ? 0 : _pinned;
  for (int i = (int)_blocks.size() - 1; i >= floor; --i) {
    auto& b = _blocks[(std::size_t)i];
    if (!b) { continue; }
    const std::size_t n = block_bytes_(b.get());
    // GIVE THE POOL BACK FIRST. Freeing a wired buffer unwires it in the
    // kernel but not in the pool's counter, and an evict/admit cycle is
    // exactly the loop that would turn that into a pool full of bytes
    // nothing holds.
    //
    // This is only safe because set_wired(false) leaves the buffer
    // NonVolatile. It used to mark it purgeable VOLATILE, and doing
    // that here -- to a block the GPU may still be reading, whose
    // weights can be subviews of a shard its neighbours share -- meant
    // the kernel could discard those pages mid-forward. It took SIGBUS
    // in this block's own destructor.
    if (_wired_fixed) { wire_block_(*b, false); }
    b.reset();
    // Taking one out of the PINNED prefix un-pins it. That prefix was
    // sized at load against what the box was believed to hold, and a
    // measurement saying its pages are no longer in RAM is that belief
    // being wrong. The forward decides resident-or-streamed by whether
    // the slot is EMPTY, not by this count, so it simply streams now.
    if (i < _pinned) { _pinned = i; }
    return n;
  }
  return 0;
}

void
Ltx25Dit::resident_pages_(std::size_t* examined, std::size_t* incore) const
{
  std::size_t ex = 0, ic = 0;
  for (int i = 0; i < (int)_blocks.size(); ++i) {
    const auto& b = _blocks[(std::size_t)i];
    if (!b) { continue; }
    for_each_weight(b->weights(), [&](const SharedBuffer& x) {
      if (x.empty()) { return; }
      // A WIRED BUFFER CANNOT HAVE LEFT RAM -- mlock guarantees it --
      // so the walk would spend ~57 ms per 4.3 GB to be told that.
      // Skipped per BUFFER because wire_block_ stops at the first
      // refusal, leaving the rest of a block unwired and still worth
      // measuring. With everything wired `examined` stays 0, which the
      // caller already reads as "no evidence" rather than a shortfall.
      if (x.is_wired()) { return; }
      const auto r = x.page_residency(64);
      if (!r.valid) { return; }
      ex += r.examined;
      ic += r.incore;
    });
  }
  if (examined != nullptr) { *examined = ex; }
  if (incore != nullptr) { *incore = ic; }
}

bool
Ltx25Dit::build_block_(int i, std::shared_ptr<BlockScratch> arena,
                       std::unique_ptr<MetalBlock>& out,
                       std::string* err) const
{
  GpuBlockWeights gw;
  // Uncached ALWAYS: this block is read, run and dropped, and caching it
  // would keep the whole streamed tail alive -- which is the one thing
  // streaming exists to avoid. It is also what counts the traffic, so
  // the manager can tell a bounded model from one thrashing.
  if (!bind_block(*_ws, _ops->mc(), _cfg, i, /*stream=*/true, gw, err)) {
    return false;
  }
  auto b = MetalBlock::create(*_ops, std::move(gw), err);
  if (!b) { return false; }
  b->set_rope(&_v_self, _have_audio ? &_a_self : nullptr,
              _have_audio ? &_v_cross : nullptr,
              _have_audio ? &_a_cross : nullptr);
  if (!b->reserve(_video_tokens, _audio_tokens, _text_tokens, err,
                  _geo_levels, &arena, lora_rank_(), lora_out_())) {
    return false;
  }
  out = std::move(b);
  return true;
}

bool
Ltx25Dit::refill_slot_(int layer, MetalBlock& dst, std::string* err) const
{
  // bind_block with a LIVE destination refills where it can and replaces
  // where it cannot -- see refill_into_ in ltx25-dit-weights.cc. Every
  // scalar it sets (dims, quant group, have_audio) is re-derived from
  // the checkpoint, so the block's metadata cannot end up describing the
  // previous layer's buffers.
  return bind_block(*_ws, _ops->mc(), _cfg, layer, /*stream=*/true,
                    dst.weights_mut(), err);
}

bool
Ltx25Dit::fill_slot_(int layer, int idx, std::shared_ptr<BlockScratch> arena,
                     std::string* err)
{
  auto& sl = _slot[(std::size_t)idx];
  if (sl) {
    if (refill_slot_(layer, *sl, err)) { return true; }
    // A refusal here is structural rather than per-tensor (bind_block
    // handles those itself), so the slot is suspect: drop it and build a
    // clean one. Self-healing rather than a sticky off-switch -- a slot
    // that failed once on a checkpoint quirk should not cost the run its
    // whole streaming path.
    sl.reset();
  }
  return build_block_(layer, std::move(arena), sl, err);
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
  // THE POOL FIRST, while the buffers still exist to be given back.
  unwire_scratch_();
  _scratch.reset();
  // THE SLOTS GO WITH IT. They hold the old arena and were reserved for
  // the old token counts, and nothing below re-reserves them -- the loop
  // walks `_blocks`, which slots are deliberately not part of. Dropped
  // rather than re-reserved: the next streamed block rebuilds one, which
  // is a read that was going to happen.
  _slot[0].reset();
  _slot[1].reset();
  _slot_cur = 0;
  // Remembered for the blocks that do not exist yet: a streamed one is
  // built inside the forward and has to be given the same geometry.
  _geo_levels = levels;
  for (auto& b : _blocks) {
    if (!b) { continue; }          // streamed: built per forward
    b->set_rope(&_v_self, _have_audio ? &_a_self : nullptr,
                _have_audio ? &_v_cross : nullptr,
                _have_audio ? &_a_cross : nullptr);
    if (!b->reserve(_video_tokens, _audio_tokens, _text_tokens, err,
                    levels, &_scratch, lora_rank_(), lora_out_())) {
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

  // INTO THE POOL, here rather than at load: the scratch does not exist
  // until this runs, and it is reallocated whenever the geometry grows.
  // Re-wiring is cheap and idempotent -- every site skips a buffer whose
  // wired state already matches -- so a second clip at the same size
  // wires nothing and one at a larger size wires only the new arena.
  //
  // NOT wired: this function's own per-geometry buffers (_tmp, _vctx,
  // the modulation rows). They are ~150 MB against the block arena's
  // ~1 GB at 960x544, and enumerating them here would be a list that
  // rots silently as buffers are added. Said plainly rather than left to
  // look complete.
  wire_into_pool();

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
  // The adapter's trunk, hoisted once: it is read at the patchify above
  // and again at the output head 350 lines down.
  const LoraTrunk* ltr = (_lora != nullptr) ? _lora->trunk() : nullptr;
  auto stream = o.mc()->make_command_stream();
  {
    auto enc = stream.begin_compute();
    o.linear(enc, _vlat, _trunk.patchify_w, &_trunk.patchify_b, _vx,
             _video_tokens, zc, vd);
    if (ltr != nullptr) {
      trunk_lora_(enc, &ltr->patchify, _vlat, _vx, _video_tokens);
    }
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
      if (ltr != nullptr) {
        trunk_lora_(enc, &ltr->audio_patchify, _alat, _ax, _audio_tokens);
      }
    }
  }
  {
    // Checked, for the reason spelled out at the block fence below: an
    // over-committed command buffer fails silently under a bare wait().
    std::string perr;
    if (!stream.commit().wait_ok(&perr)) {
      return fail("patchify: " +
                  (perr.empty() ? std::string("GPU error") : perr));
    }
  }

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

  // ---- streaming + prefetch -----------------------------------------
  //
  // A block with no resident slot is READ here, used, and dropped. The
  // read is the whole cost of streaming, and it sits on the critical
  // path unless something moves it: between a block's commit() and its
  // wait() the GPU is busy and this thread is not, which is exactly the
  // window the NEXT block's read fits into.
  //
  // Depth is structurally ONE -- a single outstanding read into a single
  // spare block -- so nothing queues and the extra live memory is one
  // block. That is what keeps a tight box safe: the failure mode there
  // is not slowness but thrash, and a second live block is the most this
  // can ever cost.
  //
  // DECLARATION ORDER: `fut` LAST, so it destroys FIRST. Its destructor
  // joins the worker while `blk` -- the block that worker is building --
  // is still alive. Reversing these two lines is a use-after-free on
  // every early return out of this loop (abort, a block that failed to
  // bind, a GPU error).
  const bool pf_on = _stream_blocks &&
                     std::getenv("VPIPE_LTX25_NO_PREFETCH") == nullptr;
  struct PrefetchSlot {
    // WHICH SLOT it is filling, not a block of its own. The prefetch
    // and the main path alternate between the two, so the reader always
    // writes the one the GPU is not reading.
    int                         slot  = -1;
    int                         block = -1;
    std::future<bool>           fut;
  } pf;
  int pf_started = 0, pf_hit = 0, streamed_n = 0;
  // The next layer that will actually be STREAMED. A resident one is
  // skipped: prefetching it would re-read bytes the forward already has.
  auto pf_next = [&](int from) {
    for (int n = from; n < (int)_blocks.size(); ++n) {
      if (!_blocks[(std::size_t)n]) { return n; }
    }
    return -1;
  };
  // The block the forward is running when the slot is not resident.

  // ---- growing back into free RAM (mechanism 4) ---------------------
  //
  // The scratch is allocated by set_geometry, BEFORE this reads the
  // budget, so those bytes are already out of `available_physical`.
  // Reserving them again is asking for the same room twice, which is the
  // documented way this refuses a block it could afford.
  _resid.note_reserve_allocated((std::size_t)scratch_bytes());
  const auto mb0 = o.mc()->memory_budget();
  _resid.begin_forward(mb0, [this] { return evict_tail_block_(); });

  // The one signal that is not arithmetic: are the blocks we kept STILL
  // in RAM? Free-memory figures cannot tell a cache about to be dropped
  // from one being used, so the ceiling is found by watching what
  // happened rather than by predicting it. Gated on our own compressed
  // footprint having moved, because the page walk is not free.
  bool resid_short = false;
  if ((_resid.count() > 0 || _pinned > 0) &&
      _resid.self_compression_grew(mb0.self_compressed)) {
    std::size_t ex = 0, ic = 0;
    resident_pages_(&ex, &ic);
    if (ex > 0 && ic < ex) {
      resid_short = true;
      std::size_t freed = _resid.note_weight_residency(
          ex, ic, [this] { return evict_tail_block_(); });
      // Nothing left outside the prefix and the pages are STILL leaving
      // RAM: the prefix itself is what does not fit. Give one of it back
      // rather than sit in the thrash it was meant to prevent -- a block
      // re-read from the file costs a read, a block faulted out of the
      // compressor costs the compress AND the decompress.
      if (freed == 0 && _pinned > 0) {
        freed = evict_tail_block_(/*allow_pinned=*/true);
      }
      if (freed > 0 && o.mc()->session() != nullptr) {
        o.mc()->session()->log_normal(vpipe::fmt(
            "ltx-2.5: resident weights are only {}% in RAM -- released "
            "{} MB, now {} blocks resident",
            (int)(100.0 * (double)ic / (double)ex), freed >> 20,
            _pinned + _resid.count()));
      }
    }
  }
  if (!resid_short) { _resid.note_healthy_forward(); }
  // WHAT THIS FORWARD ALLOCATES, cumulatively, so "steady state does not
  // reallocate" is a number rather than a claim. total_count never
  // decrements, so the delta across a forward is exactly the buffers
  // minted during it -- and once the slots are built and the resident
  // set has stopped growing, the honest value is zero.
  const auto alloc0 = vpipe::metal_compute::shared_buffer_memory_stats();
  int promoted_n = 0;
  for (int i = 0; i < (int)_blocks.size(); ++i) {
    if (in.progress && !in.progress(i, (int)_blocks.size())) {
      return fail("aborted at block " + std::to_string(i));
    }
    const auto t0 = kBlkProf ? std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::time_point{};

    // ---- the block: resident, prefetched, or read now ---------------
    MetalBlock* blk = _blocks[(std::size_t)i].get();
    // Which slot this block is in, so the promotion below can move it
    // out and the next fill can take the other one. -1 when the block
    // was already resident.
    int use = -1;
    if (blk == nullptr) {
      ++streamed_n;
      if (pf.block == i && pf.fut.valid()) {
        // Issued under block i-1's GPU work, so waiting here costs only
        // the part that did not fit under that window.
        const bool ok = pf.fut.get();
        pf.block = -1;
        if (!ok) { return fail("streaming block " + std::to_string(i)); }
        use = pf.slot;
        pf.slot = -1;
        ++pf_hit;
      } else {
        use = _slot_cur;
        if (!fill_slot_(i, use, _scratch, err)) { return false; }
      }
      // The OTHER one is free from here: this block's GPU work is
      // committed and waited before the next iteration reuses a slot,
      // and the prefetch below writes the one this is not reading.
      _slot_cur = 1 - use;
      blk = _slot[(std::size_t)use].get();
      if (blk == nullptr) { return fail("streaming block " +
                                        std::to_string(i)); }
    }

    if (kBlkProf) {
      // Stride 8: a page either survived or it did not, and a block is
      // faulted in as a whole, so one page in eight answers it at an
      // eighth of the mincore vector.
      std::size_t ex = 0, ic = 0;
      for_each_weight(blk->weights(), [&](const SharedBuffer& b) {
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

    auto s = o.mc()->make_command_stream();
    {
      auto enc = s.begin_compute();
      if (!blk->forward(enc, gv, ga, err,
                        _lora != nullptr ? _lora->block(i) : nullptr)) {
        return false;
      }
    }
    // BETWEEN THE COMMIT AND THE WAIT is the whole opportunity: the GPU
    // is busy with block i and this thread has nothing to do.
    auto fence = s.commit();
    if (pf_on && pf.block < 0) {
      const int nxt = pf_next(i + 1);
      // Asked PER BLOCK, with the same budget question growth asks,
      // because on a box that fits one block the failure mode is not
      // slowness but thrash -- a second live block tips the machine into
      // the compressor and the block being read is evicted before the
      // GPU reads it. A no simply makes the next iteration serial again;
      // nothing accumulates and nothing has to be unwound.
      const auto mb = o.mc()->memory_budget();
      if (nxt >= 0 && mb.recommended != 0 &&
          mb.fits_growth(block_bytes_(blk))) {
        pf.block = nxt;
        pf.slot  = _slot_cur;       // the one the main path is not on
        ++pf_started;
        // The arena is COPIED for the thread rather than read from the
        // member: `_scratch` is a shared_ptr the main thread reassigns
        // on a geometry change, and a worker reading it then would race.
        // Same reason build_block_ takes it by value.
        auto arena = _scratch;
        const int slot = pf.slot;
        pf.fut = std::async(std::launch::async,
                            [this, nxt, slot, arena]() {
          std::string perr;
          return fill_slot_(nxt, slot, arena, &perr);
        });
      }
    }
    // wait_ok, not wait: a command buffer can END IN ERROR, and the one
    // that matters here is an OUT-OF-MEMORY or page fault from
    // over-committing GPU memory -- exactly what a bounded box produces
    // under a streamed forward. A bare wait() returns happily and leaves
    // the output buffer silently corrupt, so the run finishes and the
    // clip is wrong. MiniMax-H3 checks this on the same commit; this did
    // not.
    std::string blk_err;
    if (!fence.wait_ok(&blk_err)) {
      return fail("block " + std::to_string(i) + ": " +
                  (blk_err.empty() ? std::string("GPU error") : blk_err));
    }
    // The streamed block dies HERE, after the GPU is done reading it --
    // unless there is room to KEEP it, in which case the next forward
    // finds it resident and reads one block fewer. Asked after the wait
    // so the budget reflects a settled forward rather than one with a
    // command buffer still in flight.
    if (_blocks[(std::size_t)i] == nullptr && use >= 0) {
      const std::size_t nb = block_bytes_(_slot[(std::size_t)use].get());
      if (nb > 0 && _resid.admit(o.mc(), nb)) {
        // MOVED out of the slot, which leaves it empty. The next block
        // to be streamed rebuilds it -- the read it was going to do
        // anyway -- so a promotion costs no copy and no extra read, and
        // allocations stop once the resident set stops growing.
        _blocks[(std::size_t)i] = std::move(_slot[(std::size_t)use]);
        _resid.note_admitted(nb);
        ++promoted_n;
        // INTO THE POOL as it is admitted. A block that is merely kept
        // is a block the compressor may take back inside the same
        // forward -- it is written once and then read only by the GPU,
        // so it carries no CPU reference bits and reads as cold. That is
        // what a resident set costs when it is resident only in the
        // accounting. Wiring is skipped silently when the pool is full,
        // which leaves the block held-but-reclaimable: the old
        // behaviour, and no worse than it.
        if (_wired_fixed) { wire_block_(*_blocks[(std::size_t)i], true); }
      }
    }
    if (kBlkProf && (i % 12 == 0 || i + 1 == (int)_blocks.size())) {
      // WHERE the memory goes, per block. A streaming DiT that ends a
      // forward holding more than a couple of blocks is not streaming,
      // and nothing else in this loop would say so.
      const auto st = vpipe::metal_compute::shared_buffer_memory_stats();
      std::fprintf(stderr,
                   "  [blk %2d] shared buffers live %6llu MB in %5llu "
                   "handles, resident blocks %d\n",
                   i, (unsigned long long)(st.live_bytes >> 20),
                   (unsigned long long)st.live_count,
                   _pinned + _resid.count());
    }
    if (kBlkProf) {
      bp_ms += std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - t0).count();
    }
  }
  if (kBlkProf && o.mc()->session() != nullptr) {
    const auto mb = o.mc()->memory_budget();
    const auto alloc1 = vpipe::metal_compute::shared_buffer_memory_stats();
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
        "ltx-2.5: {} of {} blocks streamed, prefetch {}/{} hit; resident "
        "set {} blocks / {} MB (+{} this forward); {} buffer(s) / {} MB "
        "allocated this forward",
        streamed_n, _blocks.size(), pf_hit, pf_started,
        _pinned + _resid.count(), _resid.bytes() >> 20, promoted_n,
        alloc1.total_count - alloc0.total_count,
        (alloc1.total_bytes - alloc0.total_bytes) >> 20));
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
      if (ltr != nullptr) {
        trunk_lora_(enc, &ltr->proj_out, _tmp, _v_out, _video_tokens);
      }
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
        if (ltr != nullptr) {
          trunk_lora_(enc, &ltr->audio_proj_out, _tmp, _a_out,
                      _audio_tokens);
        }
      }
    }
    std::string herr;
    if (!s.commit().wait_ok(&herr)) {
      return fail("output head: " +
                  (herr.empty() ? std::string("GPU error") : herr));
    }
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
