#include "ltx25-block-metal.h"

#include "ltx25-lora.h"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <string>
#include <vector>

using vpipe::metal_compute::ComputeEncoder;
using vpipe::metal_compute::SharedBuffer;

namespace ltx25 {

namespace {

SharedBuffer
up_mat_(const MetalOps& ops, const Mat& m)
{
  return ops.upload_bf16(m.v);
}

SharedBuffer
up_vec_(const MetalOps& ops, const std::vector<float>& v)
{
  return ops.upload_bf16(v);
}

bool
up_attn_(const MetalOps& ops, const AttnWeights& a, GpuAttn& g,
         std::string* err)
{
  if (a.q_w.rows == 0 || a.o_w.rows == 0) {
    if (err != nullptr) { *err = "attention weights are empty"; }
    return false;
  }
  g.heads     = a.heads;
  g.head_dim  = a.head_dim;
  g.query_dim = a.q_w.cols;    // to_q reads [inner][query_dim]
  g.ctx_dim   = a.k_w.cols;    // to_k reads [inner][context_dim]
  g.q_w.w = up_mat_(ops, a.q_w);
  g.k_w.w = up_mat_(ops, a.k_w);
  g.v_w.w = up_mat_(ops, a.v_w);
  g.o_w.w = up_mat_(ops, a.o_w);
  g.q_b = up_vec_(ops, a.q_b);
  g.k_b = up_vec_(ops, a.k_b);
  g.v_b = up_vec_(ops, a.v_b);
  g.o_b = up_vec_(ops, a.o_b);
  g.q_norm = up_vec_(ops, a.q_norm);
  g.k_norm = up_vec_(ops, a.k_norm);
  g.has_gate = a.gate_w.rows > 0;
  if (g.has_gate) {
    g.gate_w = up_mat_(ops, a.gate_w);
    g.gate_b = up_vec_(ops, a.gate_b);
  }
  return true;
}

bool
up_stream_(const MetalOps& ops, const StreamWeights& s, GpuStream& g,
           std::string* err)
{
  g.dim = s.dim;
  if (!up_attn_(ops, s.attn1, g.attn1, err) ||
      !up_attn_(ops, s.attn2, g.attn2, err)) {
    return false;
  }
  g.ff_hidden = s.ff_in.rows;
  g.ff_in.w  = up_mat_(ops, s.ff_in);
  g.ff_out.w = up_mat_(ops, s.ff_out);
  g.ff_has_bias = !s.ff_in_b.empty();
  if (g.ff_has_bias) {
    g.ff_in_b  = up_vec_(ops, s.ff_in_b);
    g.ff_out_b = up_vec_(ops, s.ff_out_b);
  }
  g.scale_shift        = up_vec_(ops, s.scale_shift);
  g.prompt_scale_shift = up_vec_(ops, s.prompt_scale_shift);
  g.cross_table        = up_vec_(ops, s.cross_table);
  return true;
}

}  // namespace

bool
MetalBlock::upload(const MetalOps& ops, const BlockWeights& w,
                   GpuBlockWeights& out, std::string* err)
{
  out.norm_eps = w.norm_eps;
  if (!up_stream_(ops, w.video, out.video, err)) { return false; }
  out.have_audio = w.audio.dim > 0 && w.audio.attn1.q_w.rows > 0;
  if (out.have_audio) {
    if (!up_stream_(ops, w.audio, out.audio, err)) { return false; }
    if (!up_attn_(ops, w.a2v, out.a2v, err)) { return false; }
    if (!up_attn_(ops, w.v2a, out.v2a, err)) { return false; }
  }
  return true;
}

namespace {

void
attn_weights_(const GpuAttn& a,
              const std::function<void(
                  const vpipe::metal_compute::SharedBuffer&)>& fn)
{
  auto q = [&](const QWeight& w) {
    if (w.quantized) { fn(w.codes); fn(w.scales); fn(w.qbias); }
    else             { fn(w.w); }
  };
  q(a.q_w); q(a.k_w); q(a.v_w); q(a.o_w);
  fn(a.q_b); fn(a.k_b); fn(a.v_b); fn(a.o_b);
  fn(a.q_norm); fn(a.k_norm);
  fn(a.gate_w); fn(a.gate_b);
}

void
stream_weights_(const GpuStream& g,
                const std::function<void(
                    const vpipe::metal_compute::SharedBuffer&)>& fn)
{
  attn_weights_(g.attn1, fn);
  attn_weights_(g.attn2, fn);
  if (g.ff_in.quantized) {
    fn(g.ff_in.codes); fn(g.ff_in.scales); fn(g.ff_in.qbias);
  } else {
    fn(g.ff_in.w);
  }
  if (g.ff_out.quantized) {
    fn(g.ff_out.codes); fn(g.ff_out.scales); fn(g.ff_out.qbias);
  } else {
    fn(g.ff_out.w);
  }
  fn(g.ff_in_b); fn(g.ff_out_b);
  fn(g.scale_shift); fn(g.prompt_scale_shift); fn(g.cross_table);
}

}  // namespace

void
for_each_weight(
    const GpuBlockWeights& w,
    const std::function<void(const vpipe::metal_compute::SharedBuffer&)>& fn)
{
  if (!fn) { return; }
  // Empty buffers are handed to `fn` too rather than filtered here: a
  // dense pack leaves the quantized slots empty and vice versa, an
  // optional bias may be absent, and every caller has to cope with that
  // anyway. Filtering would just move the check.
  stream_weights_(w.video, fn);
  if (w.have_audio) {
    stream_weights_(w.audio, fn);
    attn_weights_(w.a2v, fn);
    attn_weights_(w.v2a, fn);
  }
}

std::unique_ptr<MetalBlock>
MetalBlock::create(const MetalOps& ops, GpuBlockWeights w, std::string* err)
{
  if (w.video.dim <= 0 || w.video.attn1.q_w.empty()) {
    if (err != nullptr) { *err = "video stream weights are not bound"; }
    return nullptr;
  }
  std::unique_ptr<MetalBlock> b(new MetalBlock());
  b->_ops = &ops;
  b->_w = std::move(w);
  return b;
}

std::unique_ptr<MetalBlock>
MetalBlock::create(const MetalOps& ops, const BlockWeights& w,
                   std::string* err)
{
  GpuBlockWeights g;
  if (!upload(ops, w, g, err)) { return nullptr; }
  return create(ops, std::move(g), err);
}

bool
MetalBlock::set_rope(const RopeTable* v_self, const RopeTable* a_self,
                     const RopeTable* v_cross, const RopeTable* a_cross)
{
  auto up = [&](const RopeTable* t, RopeGpu& g) {
    if (t == nullptr || t->cos.empty()) { g.valid = false; return; }
    g.cos = _ops->upload_f32(t->cos);
    g.sin = _ops->upload_f32(t->sin);
    g.heads = t->heads;
    g.tokens = t->tokens;
    g.half = t->half;
    g.valid = true;
  };
  up(v_self, _v_self);
  up(a_self, _a_self);
  up(v_cross, _v_cross);
  up(a_cross, _a_cross);
  return true;
}

std::uint64_t
BlockScratch::predict_bytes(std::size_t t, std::size_t d, std::size_t f,
                            std::size_t l, std::size_t r,
                            std::size_t o) noexcept
{
  if (d == 0 || t == 0) { return 0; }
  if (l < 1) { l = 1; }
  // 15 planes (a,b,c,d,e / q,k,v,o / qh,kh,vh,oh / snap_v,snap_a), the
  // wide feed-forward buffer, the [tokens][64] gate logits, three
  // per-level modulation rows and the two cross-attention K/V rows --
  // reserve()'s allocation list, in its order.
  std::uint64_t elems = 15ull * t * d + (std::uint64_t)t * std::max(d, f)
                      + 64ull * t + 3ull * d * l + 2ull * d;
  // The adapter planes, allocated only when there IS an adapter -- so a
  // run without one predicts exactly what it predicted before.
  if (r > 0 && o > 0) { elems += (std::uint64_t)t * (r + o); }
  return elems * 2;               // MetalOps::alloc is bf16
}

std::uint64_t
BlockScratch::bytes() const noexcept
{
  const vpipe::metal_compute::SharedBuffer* all[] = {
      &a, &b, &c, &d, &e, &q, &k, &v, &o, &qh, &kh, &vh, &oh,
      &gate_logits, &ff, &mod_scale, &mod_shift, &mod_gate,
      &kv_scale, &kv_shift, &snap_v, &snap_a, &lora_r, &lora_d};
  std::uint64_t n = 0;
  for (const auto* b : all) { n += b->byte_size(); }
  return n;
}

void
BlockScratch::for_each_buffer(
    const std::function<void(vpipe::metal_compute::SharedBuffer&)>& fn)
{
  if (!fn) { return; }
  // The SAME list bytes() sums, and deliberately so: a buffer counted
  // but not wired is one the accounting believes is protected and the
  // kernel is free to take.
  vpipe::metal_compute::SharedBuffer* all[] = {
      &a, &b, &c, &d, &e, &q, &k, &v, &o, &qh, &kh, &vh, &oh,
      &gate_logits, &ff, &mod_scale, &mod_shift, &mod_gate,
      &kv_scale, &kv_shift, &snap_v, &snap_a, &lora_r, &lora_d};
  for (auto* p : all) { fn(*p); }
}

const MetalOps::SteelAttn*
BlockScratch::steel_for(const MetalOps& ops, int heads, int tq, int tkv,
                        int head_dim)
{
  if (!ops.steel_attn_available(head_dim)) { return nullptr; }
  for (const auto& p : attn_plans) {
    if (p->heads == heads && p->tq == tq && p->tkv == tkv &&
        p->head_dim == head_dim) {
      return p.get();
    }
  }
  auto p = std::make_unique<MetalOps::SteelAttn>();
  if (!ops.steel_attn_plan(p.get(), heads, tq, tkv, head_dim)) {
    // Not cached as a negative: a plan fails on an allocation or a
    // missing specialisation, and re-asking next block is cheaper than
    // carrying a second kind of entry. Both failures are also
    // shape-independent in practice, so the retry costs a library
    // lookup, not a dispatch.
    return nullptr;
  }
  attn_plans.push_back(std::move(p));
  return attn_plans.back().get();
}

bool
MetalBlock::reserve(int max_video_tokens, int max_audio_tokens, int max_text,
                    std::string* err, int max_levels,
                    std::shared_ptr<BlockScratch>* arena,
                    int max_lora_rank, int max_lora_out)
{
  const int vd = _w.video.dim;
  const int ad = _w.have_audio ? _w.audio.dim : 0;
  const std::size_t dim = (std::size_t)std::max(vd, ad);
  const std::size_t tok = (std::size_t)std::max(
      std::max(max_video_tokens, max_audio_tokens), max_text);
  if (dim == 0 || tok == 0) {
    if (err != nullptr) { *err = "reserve() needs a non-empty geometry"; }
    return false;
  }
  // Widest single activation any op here produces. The FF hidden is the
  // largest by a factor of four, so it sets the arena rather than the
  // attention widths.
  const std::size_t ffh =
      (std::size_t)std::max(_w.video.ff_hidden,
                            _w.have_audio ? _w.audio.ff_hidden : 0);
  const std::size_t lv = (std::size_t)std::max(1, max_levels);
  // BOTH or neither: a rank with no output width (or the reverse) would
  // allocate one half of a pair and dispatch the other into an empty
  // buffer, which is a GPU fault rather than a wrong number -- but only
  // on the run that happens to load an adapter.
  const std::size_t lr =
      (max_lora_rank > 0 && max_lora_out > 0) ? (std::size_t)max_lora_rank : 0;
  const std::size_t lo =
      (max_lora_rank > 0 && max_lora_out > 0) ? (std::size_t)max_lora_out : 0;

  // Adopt the stack's arena, or a private one when driven alone.
  if (arena != nullptr) {
    if (!*arena) { *arena = std::make_shared<BlockScratch>(); }
    _s = *arena;
  } else if (!_s) {
    _s = std::make_shared<BlockScratch>();
  }

  // ALLOCATE ONLY WHAT IS NOT ALREADY BIG ENOUGH. Every block of a stack
  // passes the same numbers, so the first sizes the arena and the other
  // 47 fall straight through -- which is the whole saving. A block that
  // genuinely needs more (a wider FF, say) grows it, and the blocks
  // already holding it see the growth, because it is one object.
  const bool grow = tok > _s->tokens || dim > _s->dim ||
                    lv > _s->levels || ffh > _s->ffh ||
                    lr > _s->lora_rank || lo > _s->lora_out;
  if (!grow) { return true; }

  const std::size_t t = std::max(tok, _s->tokens);
  const std::size_t d = std::max(dim, _s->dim);
  const std::size_t f = std::max(ffh, _s->ffh);
  const std::size_t l = std::max(lv, _s->levels);
  const std::size_t rk = std::max(lr, _s->lora_rank);
  const std::size_t ow = std::max(lo, _s->lora_out);
  const std::size_t plane = t * d;
  const std::size_t wide  = t * std::max(d, f);

  _s->a = _ops->alloc(plane);
  _s->b = _ops->alloc(plane);
  _s->c = _ops->alloc(plane);
  _s->d = _ops->alloc(plane);
  _s->e = _ops->alloc(plane);
  _s->q = _ops->alloc(plane);
  _s->k = _ops->alloc(plane);
  _s->v = _ops->alloc(plane);
  _s->o = _ops->alloc(plane);
  _s->qh = _ops->alloc(plane);
  _s->kh = _ops->alloc(plane);
  _s->vh = _ops->alloc(plane);
  _s->oh = _ops->alloc(plane);
  _s->ff = _ops->alloc(wide);
  // The gate logits are [tokens][heads] -- tiny, but sized from the
  // widest head count rather than assumed.
  _s->gate_logits = _ops->alloc(t * 64);
  // One shift/scale/gate row per DENOISE LEVEL. A generation with no
  // conditioning has one, which is the shape these were before -- three
  // dim-wide rows, a rounding error either way (4096 x 2 B x 3 levels).
  // The audio<->video cross tables write only row 0; their drivers come
  // from the stream's scalar sigma and are not per token.
  _s->mod_scale = _ops->alloc(d * l);
  _s->mod_shift = _ops->alloc(d * l);
  _s->mod_gate  = _ops->alloc(d * l);
  _s->kv_scale  = _ops->alloc(d);
  _s->kv_shift  = _ops->alloc(d);
  _s->snap_v = _ops->alloc(plane);
  _s->snap_a = _ops->alloc(plane);
  // Only when an adapter asked for them. `lora_d` is as wide as `ff`,
  // so allocating it unconditionally would put a second 267 MB plane in
  // every run at the geometry this model is built for.
  if (rk > 0 && ow > 0) {
    _s->lora_r = _ops->alloc(t * rk);
    _s->lora_d = _ops->alloc(t * ow);
  }
  _s->tokens = t;
  _s->dim = d;
  _s->levels = l;
  _s->ffh = f;
  _s->lora_rank = rk;
  _s->lora_out = ow;
  return true;
}

// The three adaLN driver rows for a modulation group: rows [lo, lo+3) of
// a `table_rows`-row table, each ADDED to the same-indexed slice of the
// timestep MLP's output. Written into the scratch modulation buffers.
//
// The ORDER is the reference's and is not uniform across the block: the
// self-attention and FF triples unpack (shift, scale, gate) while the
// audio<->video cross tables unpack (scale, shift). That is why this
// takes `lo` and the caller names what it got.
void
MetalBlock::ada3_(ComputeEncoder& enc, const SharedBuffer& table,
                  const SharedBuffer& timesteps, int table_rows, int dim,
                  int lo, int levels)
{
  (void)table_rows;   // the slices are contiguous; rows only fix the stride
  const std::size_t w = (std::size_t)dim * 2;   // bf16 bytes per row
  // The BLOCK's table is per-block and level-independent; the TIMESTEP
  // driver is per level. So the sum is taken once per level, each into
  // its own row of the scratch, and the block's own row is re-read.
  for (int g = 0; g < levels; ++g) {
    const std::size_t ts = (std::size_t)g * 9 * w;   // this level's 9*dim
    const std::size_t o  = (std::size_t)g * w;
    _ops->add(enc, table, (std::size_t)lo * w, timesteps,
              ts + (std::size_t)lo * w, _s->mod_shift, dim, o);
    _ops->add(enc, table, (std::size_t)(lo + 1) * w, timesteps,
              ts + (std::size_t)(lo + 1) * w, _s->mod_scale, dim, o);
    _ops->add(enc, table, (std::size_t)(lo + 2) * w, timesteps,
              ts + (std::size_t)(lo + 2) * w, _s->mod_gate, dim, o);
  }
}

// One adapted linear's contribution, added into the output the base
// linear just wrote. Null and empty pairs are no-ops, so a call site
// reads as one line beside its linear whether or not an adapter is
// loaded.
//
// THE PLACEMENT IS THE WHOLE POINT: this lands between the linear and
// whatever consumes it, so q/k are adapted BEFORE their RMSNorm and
// their RoPE, exactly as a fused weight would have been. Applying it
// after the norm would be a different function that still runs.
void
MetalBlock::lora_(ComputeEncoder& enc, const LoraPair* p,
                  const SharedBuffer& x, const SharedBuffer& y, int M) const
{
  if (p == nullptr || !p->valid() || M <= 0) { return; }
  _ops->lora_add(enc, x, p->a, p->b, y, _s->lora_r, _s->lora_d, M, p->k,
                 p->n, p->rank);
}

void
MetalBlock::run_attention_(ComputeEncoder& enc, const GpuAttn& a,
                           const SharedBuffer& xq, int tq,
                           const SharedBuffer& xkv, int tkv,
                           const RopeGpu* q_pe, const RopeGpu* k_pe,
                           const SharedBuffer& out, const LoraAttn* lora)
{
  const int inner = a.heads * a.head_dim;
  _ops->linear(enc, xq,  a.q_w, &a.q_b, _s->q, tq,  a.query_dim, inner);
  _ops->linear(enc, xkv, a.k_w, &a.k_b, _s->k, tkv, a.ctx_dim,   inner);
  _ops->linear(enc, xkv, a.v_w, &a.v_b, _s->v, tkv, a.ctx_dim,   inner);
  if (lora != nullptr) {
    lora_(enc, &lora->q, xq,  _s->q, tq);
    lora_(enc, &lora->k, xkv, _s->k, tkv);
    lora_(enc, &lora->v, xkv, _s->v, tkv);
  }

  // q/k RMSNorm spans the WHOLE projection width, not the head dim.
  _ops->rms_norm_gain(enc, _s->q, a.q_norm, inner, tq);
  _ops->rms_norm_gain(enc, _s->k, a.k_norm, inner, tkv);

  // RoPE on token-major, before the head-major transpose.
  if (q_pe != nullptr && q_pe->valid) {
    _ops->rope(enc, _s->q, q_pe->cos, q_pe->sin, a.heads, tq, a.head_dim);
    const RopeGpu* kt = (k_pe != nullptr && k_pe->valid) ? k_pe : q_pe;
    _ops->rope(enc, _s->k, kt->cos, kt->sin, a.heads, tkv, a.head_dim);
  }

  // Both attention kernels want [H][T][D]; the GEMMs produced [T][H*D].
  _ops->transpose_abd(enc, _s->q, _s->qh, tq,  a.heads, a.head_dim);
  _ops->transpose_abd(enc, _s->k, _s->kh, tkv, a.heads, a.head_dim);
  _ops->transpose_abd(enc, _s->v, _s->vh, tkv, a.heads, a.head_dim);
  // Steel flash attention where it has an entry point for this head
  // width, the scalar kernel otherwise. Same buffers either way -- the
  // plan carries the shape, so the choice is one dispatch or the other.
  const MetalOps::SteelAttn* st =
      _s->steel_for(*_ops, a.heads, tq, tkv, a.head_dim);
  if (st != nullptr) {
    _ops->sdpa_steel(enc, *st, _s->qh, _s->kh, _s->vh, _s->oh);
  } else {
    _ops->sdpa_full(enc, _s->qh, _s->kh, _s->vh, _s->oh, a.heads, tq, tkv,
                    a.head_dim);
  }
  _ops->transpose_abd(enc, _s->oh, _s->o, a.heads, tq, a.head_dim);

  // Per-head gating, from the attention's INPUT and before to_out.
  if (a.has_gate) {
    _ops->linear(enc, xq, a.gate_w, &a.gate_b, _s->gate_logits, tq,
                 a.query_dim, a.heads);
    // The gate writes ONE LOGIT PER HEAD, not `inner`. An adapter bound
    // against the wrong width here is a valid GEMM over the wrong
    // columns; the loader checks it, and this is the consumer that
    // depends on the check.
    if (lora != nullptr) {
      lora_(enc, &lora->gate, xq, _s->gate_logits, tq);
    }
    _ops->gate_heads(enc, _s->o, _s->gate_logits, a.heads, tq, a.head_dim);
  }
  _ops->linear(enc, _s->o, a.o_w, &a.o_b, out, tq, inner, a.query_dim);
  if (lora != nullptr) { lora_(enc, &lora->o, _s->o, out, tq); }
}

// Self-attention + text cross-attention, i.e. everything before the
// audio<->video pair. `x` is updated in place.
void
MetalBlock::stream_first_half_(ComputeEncoder& enc, const GpuStream& w,
                               GpuStreamInput& s, const RopeGpu* pe,
                               const LoraAttn* l1, const LoraAttn* l2)
{
  const int d = w.dim, n = s.tokens, tt = s.text_tokens;

  const int g = s.n_levels;
  const bool pt = (g > 1 && s.level != nullptr);

  // Rows 0..3: shift, scale, gate for the self-attention.
  ada3_(enc, w.scale_shift, (*s.timesteps), 9, d, 0, g);
  if (pt) {
    _ops->ada_zero_g(enc, (*s.x), _s->mod_scale, _s->mod_shift, *s.level,
                     _s->a, d, n);
  } else {
    _ops->ada_zero(enc, (*s.x), _s->mod_scale, _s->mod_shift, _s->a, d, n);
  }
  run_attention_(enc, w.attn1, _s->a, n, _s->a, n, pe, nullptr, _s->b, l1);
  // x = x + out * gate, THEN x_normed = rms_norm(x). The second
  // normalisation feeds the cross-attention and is NOT re-derived from
  // the pre-residual x -- keeping both is why rms_norm_out exists.
  if (pt) {
    _ops->gated_residual_g(enc, (*s.x), _s->mod_gate, _s->b, *s.level, d, n);
  } else {
    _ops->gated_residual(enc, (*s.x), _s->mod_gate, _s->b, d, n);
  }
  _ops->rms_norm_out(enc, (*s.x), _s->c, d, n);

  // Rows 6..9: shift_q, scale_q, gate for the text cross-attention.
  ada3_(enc, w.scale_shift, (*s.timesteps), 9, d, 6, g);
  // modulate, NOT ada_zero: `_s->c` is already normalised, and the
  // reference applies only the affine here. Normalising twice is a
  // plausible-looking result with the residual stream's scale thrown
  // away.
  if (pt) {
    _ops->modulate_g(enc, _s->c, _s->mod_scale, 0, _s->mod_shift, 0, *s.level,
                     _s->d, d, n, /*level_stride=*/d);
  } else {
    _ops->modulate(enc, _s->c, _s->mod_scale, _s->mod_shift, _s->d, d, n);
  }

  // The KEY/VALUE side is modulated too, from prompt_scale_shift_table
  // -- row 0 SHIFT, row 1 SCALE, and no timestep. Modulating only the
  // query is the easy mistake and costs prompt adherence, not shape.
  const std::size_t w_row = (std::size_t)d * 2;
  if (s.prompt_timestep != nullptr) {
    // Timestep-DEPENDENT K/V: table[0] + pts[0] is the shift, table[1] +
    // pts[1] the scale. Built into the kv_* scratch, which is free here
    // -- it only carries a value during the audio<->video pair.
    _ops->add(enc, w.prompt_scale_shift, 0, *s.prompt_timestep, 0,
              _s->kv_shift, d);
    _ops->add(enc, w.prompt_scale_shift, w_row, *s.prompt_timestep, w_row,
              _s->kv_scale, d);
    _ops->modulate(enc, (*s.context), _s->kv_scale, _s->kv_shift, _s->e, d, tt);
  } else {
    _ops->modulate_off(enc, (*s.context), w.prompt_scale_shift, w_row,
                       w.prompt_scale_shift, 0, _s->e, d, tt);
  }

  run_attention_(enc, w.attn2, _s->d, n, _s->e, tt, nullptr, nullptr, _s->b,
                 l2);
  if (pt) {
    _ops->gated_residual_g(enc, (*s.x), _s->mod_gate, _s->b, *s.level, d, n);
  } else {
    _ops->gated_residual(enc, (*s.x), _s->mod_gate, _s->b, d, n);
  }
}

// The feed-forward tail: rows 3..6.
void
MetalBlock::stream_ff_(ComputeEncoder& enc, const GpuStream& w,
                       GpuStreamInput& s, const LoraPair* l_in,
                       const LoraPair* l_out)
{
  const int d = w.dim, n = s.tokens, h = w.ff_hidden;
  const int g = s.n_levels;
  const bool pt = (g > 1 && s.level != nullptr);
  ada3_(enc, w.scale_shift, (*s.timesteps), 9, d, 3, g);
  if (pt) {
    _ops->ada_zero_g(enc, (*s.x), _s->mod_scale, _s->mod_shift, *s.level,
                     _s->a, d, n);
  } else {
    _ops->ada_zero(enc, (*s.x), _s->mod_scale, _s->mod_shift, _s->a, d, n);
  }
  _ops->linear(enc, _s->a, w.ff_in, w.ff_has_bias ? &w.ff_in_b : nullptr,
               _s->ff, n, d, h);
  // BEFORE the GELU. `ff.net.0.proj` is the widest adapted linear in
  // the model and the only reason the arena's `lora_d` is 4x a stream
  // plane.
  lora_(enc, l_in, _s->a, _s->ff, n);
  _ops->gelu(enc, _s->ff, _s->ff, n * h);
  _ops->linear(enc, _s->ff, w.ff_out, w.ff_has_bias ? &w.ff_out_b : nullptr,
               _s->b, n, h, d);
  lora_(enc, l_out, _s->ff, _s->b, n);
  if (pt) {
    _ops->gated_residual_g(enc, (*s.x), _s->mod_gate, _s->b, *s.level, d, n);
  } else {
    _ops->gated_residual(enc, (*s.x), _s->mod_gate, _s->b, d, n);
  }
}

// One direction of the audio<->video cross-attention.
//
// `q_side` is the stream being UPDATED, `kv_side` the one it reads.
// `lo` is 0 for a2v and 2 for v2a: the [5, dim] table holds the a2v
// scale/shift at rows 0..2, the v2a pair at 2..4, and the gate at row 4
// -- and the pair unpacks SCALE FIRST, unlike every other triple here.
void
MetalBlock::av_cross_(ComputeEncoder& enc, const GpuAttn& attn,
                      const GpuStream& q_w, GpuStreamInput& q_in,
                      const SharedBuffer& q_snap,
                      const GpuStream& kv_w, GpuStreamInput& kv_in,
                      const SharedBuffer& kv_snap, int lo,
                      const RopeGpu* q_pe, const RopeGpu* kv_pe,
                      const LoraAttn* lora)
{
  const int qd = q_w.dim, kd = kv_w.dim;
  const std::size_t qrow = (std::size_t)qd * 2, krow = (std::size_t)kd * 2;

  // The query side's scale/shift, and its gate (row 4, driven by the
  // SEPARATE cross_gate timestep rather than cross_scale_shift).
  _ops->add(enc, q_w.cross_table, (std::size_t)lo * qrow, (*q_in.cross_scale_shift),
            (std::size_t)lo * qrow, _s->mod_scale, qd);
  _ops->add(enc, q_w.cross_table, (std::size_t)(lo + 1) * qrow,
            (*q_in.cross_scale_shift), (std::size_t)(lo + 1) * qrow,
            _s->mod_shift, qd);
  _ops->add(enc, q_w.cross_table, (std::size_t)4 * qrow, (*q_in.cross_gate), 0,
            _s->mod_gate, qd);
  _ops->ada_zero(enc, q_snap, _s->mod_scale, _s->mod_shift, _s->d, qd,
                 q_in.tokens);

  // The key/value side's scale/shift. Its gate is discarded -- each
  // direction is gated by the stream it WRITES.
  _ops->add(enc, kv_w.cross_table, (std::size_t)lo * krow,
            (*kv_in.cross_scale_shift), (std::size_t)lo * krow, _s->kv_scale, kd);
  _ops->add(enc, kv_w.cross_table, (std::size_t)(lo + 1) * krow,
            (*kv_in.cross_scale_shift), (std::size_t)(lo + 1) * krow,
            _s->kv_shift, kd);
  _ops->ada_zero(enc, kv_snap, _s->kv_scale, _s->kv_shift, _s->e, kd,
                 kv_in.tokens);

  run_attention_(enc, attn, _s->d, q_in.tokens, _s->e, kv_in.tokens, q_pe,
                 kv_pe, _s->b, lora);
  _ops->gated_residual(enc, (*q_in.x), _s->mod_gate, _s->b, qd, q_in.tokens);
}

bool
MetalBlock::forward(ComputeEncoder& enc, GpuStreamInput& video,
                    GpuStreamInput& audio, std::string* err,
                    const LoraBlock* lora)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  const bool rv = video.present && video.tokens > 0;
  const bool ra = _w.have_audio && audio.present && audio.tokens > 0;
  if (!rv && !ra) { return fail("neither stream is present"); }
  const std::size_t need_tok =
      (std::size_t)std::max(std::max(video.tokens, audio.tokens),
                            std::max(video.text_tokens, audio.text_tokens));
  if (need_tok > _s->tokens) {
    return fail("forward wants " + std::to_string(need_tok) +
                " tokens but reserve() sized for " +
                std::to_string(_s->tokens));
  }
  // A LoRA with no planes to work in would dispatch two GEMMs into
  // empty buffers. REFUSED rather than skipped: an adapter the caller
  // asked for and did not get renders a plausible clip that ignored it,
  // and nothing downstream can tell.
  if (lora != nullptr && (_s->lora_rank == 0 || _s->lora_out == 0)) {
    return fail("a LoRA was passed to forward() but reserve() was not "
                "told its rank -- the adapter planes were never "
                "allocated");
  }

  if (rv) {
    stream_first_half_(enc, _w.video, video, &_v_self,
                       lora != nullptr ? &lora->video_attn1 : nullptr,
                       lora != nullptr ? &lora->video_attn2 : nullptr);
  }
  if (ra) {
    stream_first_half_(enc, _w.audio, audio, &_a_self,
                       lora != nullptr ? &lora->audio_attn1 : nullptr,
                       lora != nullptr ? &lora->audio_attn2 : nullptr);
  }

  if (rv && ra) {
    // BOTH DIRECTIONS READ THE PRE-CROSS SNAPSHOT. a2v updates the video
    // stream; if v2a then read that updated video as its keys, the
    // result would depend on which direction ran first. Nothing in the
    // shapes catches this -- it quietly couples the modalities more in
    // one direction than the other.
    _ops->copy(enc, (*video.x), _s->snap_v, video.tokens * _w.video.dim);
    _ops->copy(enc, (*audio.x), _s->snap_a, audio.tokens * _w.audio.dim);
    av_cross_(enc, _w.a2v, _w.video, video, _s->snap_v, _w.audio, audio,
              _s->snap_a, 0, &_v_cross, &_a_cross,
              lora != nullptr ? &lora->a2v : nullptr);
    av_cross_(enc, _w.v2a, _w.audio, audio, _s->snap_a, _w.video, video,
              _s->snap_v, 2, &_a_cross, &_v_cross,
              lora != nullptr ? &lora->v2a : nullptr);
  }

  if (rv) {
    stream_ff_(enc, _w.video, video,
               lora != nullptr ? &lora->video_ff_in : nullptr,
               lora != nullptr ? &lora->video_ff_out : nullptr);
  }
  if (ra) {
    stream_ff_(enc, _w.audio, audio,
               lora != nullptr ? &lora->audio_ff_in : nullptr,
               lora != nullptr ? &lora->audio_ff_out : nullptr);
  }
  return true;
}

}  // namespace ltx25
