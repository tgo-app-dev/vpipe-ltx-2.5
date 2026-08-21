#include "ltx25-connector.h"
#include "ltx25-dit-weights.h"

#include <string>
#include <vector>

using vpipe::genai::WeightSet;
using vpipe::metal_compute::SharedBuffer;

namespace ltx25 {

namespace {

SharedBuffer
get_(WeightSet& ws, vpipe::metal_compute::MetalCompute* mc,
     const std::string& n, WeightSet::Residency kept)
{
  if (!ws.has(n)) { return SharedBuffer{}; }
  return ws.tensor(n, mc, kept);
}

bool
need_(WeightSet& ws, vpipe::metal_compute::MetalCompute* mc,
      const std::string& n, SharedBuffer& out, std::string* miss,
      WeightSet::Residency kept)
{
  out = get_(ws, mc, n, kept);
  if (out.empty()) {
    if (miss != nullptr && miss->empty()) { *miss = n; }
    return false;
  }
  return true;
}

}  // namespace

std::unique_ptr<Ltx25Connector>
Ltx25Connector::load(const DitConfig& cfg, WeightSet& ws, const MetalOps& ops,
                     bool audio, std::string* err,
                     WeightSet::Residency kept)
{
  std::unique_ptr<Ltx25Connector> c(new Ltx25Connector());
  c->_ops = &ops;
  c->_heads = audio ? cfg.audio_connector_num_attention_heads
                    : cfg.connector_num_attention_heads;
  c->_head_dim = audio ? cfg.audio_connector_attention_head_dim
                       : cfg.connector_attention_head_dim;
  c->_dim = c->_heads * c->_head_dim;
  // 4x, as both connectors ship. The checkpoint states it only through
  // the tensor's shape, which WeightSet does not expose.
  c->_ff_hidden = c->_dim * 4;
  c->_n_registers = cfg.connector_num_learnable_registers;
  c->_norm_output = cfg.connector_norm_output;
  c->_theta = cfg.positional_embedding_theta;
  c->_rope_f64 = cfg.rope_f64();
  c->_max_pos = cfg.connector_positional_embedding_max_pos.empty()
                    ? 4096
                    : cfg.connector_positional_embedding_max_pos[0];

  const std::string root = std::string(kDitPrefix) +
                           (audio ? "audio_" : "video_") +
                           "embeddings_connector.";
  std::string miss, qerr;
  // Group is fixed across the connector, as it is across a block:
  // two group sizes would need two kernels bound at once, and
  // silently running the wrong one is a full-speed wrong answer.
  int qgroup = 0;
  if (c->_n_registers > 0 &&
      !need_(ws, ops.mc(), root + "learnable_registers", c->_registers,
             &miss, kept)) {
    if (err != nullptr) { *err = "missing '" + miss + "'"; }
    return nullptr;
  }

  c->_blocks.resize((std::size_t)cfg.connector_num_layers);
  for (int i = 0; i < cfg.connector_num_layers; ++i) {
    Block& b = c->_blocks[(std::size_t)i];
    const std::string p =
        root + "transformer_1d_blocks." + std::to_string(i) + ".";
    // The MATRICES through bind_qlinear (dense or affine, decided by the
    // checkpoint's own siblings); the biases and norms straight. `.weight`
    // is appended by the binder, so these name the LINEAR and not the
    // tensor. All four attention projections read the connector width;
    // ff.net.2 reads the 4x hidden.
    const bool ok =
        bind_qlinear(ws, ops.mc(), p + "attn1.to_q", c->_dim, false, b.q_w,
                     &qgroup, &miss, &qerr, kept) &&
        need_(ws, ops.mc(), p + "attn1.to_q.bias",   b.q_b, &miss, kept) &&
        bind_qlinear(ws, ops.mc(), p + "attn1.to_k", c->_dim, false, b.k_w,
                     &qgroup, &miss, &qerr, kept) &&
        need_(ws, ops.mc(), p + "attn1.to_k.bias",   b.k_b, &miss, kept) &&
        bind_qlinear(ws, ops.mc(), p + "attn1.to_v", c->_dim, false, b.v_w,
                     &qgroup, &miss, &qerr, kept) &&
        need_(ws, ops.mc(), p + "attn1.to_v.bias",   b.v_b, &miss, kept) &&
        bind_qlinear(ws, ops.mc(), p + "attn1.to_out.0", c->_dim, false,
                     b.o_w, &qgroup, &miss, &qerr, kept) &&
        need_(ws, ops.mc(), p + "attn1.to_out.0.bias",   b.o_b, &miss, kept) &&
        need_(ws, ops.mc(), p + "attn1.q_norm.weight", b.q_norm, &miss, kept) &&
        need_(ws, ops.mc(), p + "attn1.k_norm.weight", b.k_norm, &miss, kept) &&
        // The connector's feed-forward HAS bias, unlike the DiT's video
        // one. Required, not probed: a missing bias here means the
        // checkpoint disagrees with connector_ff_bias and the result
        // would be quietly shifted.
        bind_qlinear(ws, ops.mc(), p + "ff.net.0.proj", c->_dim, false,
                     b.ff_in, &qgroup, &miss, &qerr, kept) &&
        need_(ws, ops.mc(), p + "ff.net.0.proj.bias",   b.ff_in_b, &miss,
            kept) &&
        bind_qlinear(ws, ops.mc(), p + "ff.net.2", c->_ff_hidden, false,
                     b.ff_out, &qgroup, &miss, &qerr, kept) &&
        need_(ws, ops.mc(), p + "ff.net.2.bias",   b.ff_out_b, &miss, kept);
    if (!ok) {
      if (err != nullptr) {
        // A quantization-shape complaint says more than "missing X" --
        // the tensor IS there, its shapes just do not close -- so it
        // wins the message, exactly as it does for a block.
        *err = "connector block " + std::to_string(i) + ": " +
               (qerr.empty() ? "missing '" + miss + "'" : qerr);
      }
      return nullptr;
    }
    if (b.q_w.quantized && !ops.quant_available()) {
      if (err != nullptr) {
        *err = "the connector is quantized but this host's affine qmm "
               "kernels did not resolve -- an unvalidated ComputeFunction "
               "is a silent no-op, so this refuses rather than running the "
               "connector over uninitialised memory";
      }
      return nullptr;
    }
    b.gate_w = get_(ws, ops.mc(), p + "attn1.to_gate_logits.weight", kept);
    b.has_gate = !b.gate_w.empty();
    if (b.has_gate) {
      b.gate_b = get_(ws, ops.mc(), p + "attn1.to_gate_logits.bias", kept);
    }
  }
  return c;
}

bool
Ltx25Connector::reserve(int tokens, std::string* err)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  if (tokens <= 0) { return fail("tokens must be positive"); }
  if (_n_registers > 0 && tokens % _n_registers != 0) {
    // The reference asserts this. The registers TILE, so a sequence that
    // is not a whole number of tiles has no defined substitution.
    return fail("sequence length " + std::to_string(tokens) +
                " is not a multiple of the register count " +
                std::to_string(_n_registers));
  }
  _tokens = tokens;

  // A 1-D grid, one position per token, at the CONNECTOR's own max_pos
  // (4096) -- not the DiT's [20, 2048, 2048].
  // INDICES, not seconds: the connector's registers are a
  // sequence position, and ltx25-connector-test pins them that way
  // against the reference's own connector.
  _rope = build_index_rope(tokens, {_max_pos}, _dim, _heads, _theta,
                           _rope_f64);
  if (_rope.tokens == 0) { return fail("could not build the connector RoPE"); }
  _rope_cos = _ops->upload_f32(_rope.cos);
  _rope_sin = _ops->upload_f32(_rope.sin);

  const std::size_t plane = (std::size_t)tokens * _dim;
  _a  = _ops->alloc(plane);
  _b  = _ops->alloc(plane);
  _q  = _ops->alloc(plane);
  _k  = _ops->alloc(plane);
  _v  = _ops->alloc(plane);
  _o  = _ops->alloc(plane);
  _qh = _ops->alloc(plane);
  _kh = _ops->alloc(plane);
  _vh = _ops->alloc(plane);
  _oh = _ops->alloc(plane);
  _gate = _ops->alloc((std::size_t)tokens * _heads);
  _ff = _ops->alloc((std::size_t)tokens * _ff_hidden);
  return true;
}

bool
Ltx25Connector::forward(const SharedBuffer& in, const SharedBuffer& out,
                        int tokens, int n_valid, std::string* err)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  if (tokens != _tokens) {
    return fail("reserve() sized for " + std::to_string(_tokens) +
                " tokens, forward wants " + std::to_string(tokens));
  }
  const MetalOps& o = *_ops;
  if (_steel.tq != tokens) {
    _steel = MetalOps::SteelAttn{};
    if (!o.steel_attn_plan(&_steel, _heads, tokens, tokens, _head_dim)) {
      _steel = MetalOps::SteelAttn{};
    }
  }
  auto stream = o.mc()->make_command_stream();
  {
    auto enc = stream.begin_compute();

    // The running stream lives in `out` from here: the first thing done
    // is a copy, so the caller's input is never modified.
    o.copy(enc, in, out, tokens * _dim);
    if (_n_registers > 0) {
      o.fill_registers(enc, out, _registers, _dim, _n_registers, n_valid,
                       tokens);
    }

    for (const Block& b : _blocks) {
      // --- pre-norm + self-attention, PLAIN residual -----------------
      o.rms_norm_out(enc, out, _a, _dim, tokens);
      o.linear(enc, _a, b.q_w, &b.q_b, _q, tokens, _dim, _dim);
      o.linear(enc, _a, b.k_w, &b.k_b, _k, tokens, _dim, _dim);
      o.linear(enc, _a, b.v_w, &b.v_b, _v, tokens, _dim, _dim);
      o.rms_norm_gain(enc, _q, b.q_norm, _dim, tokens);
      o.rms_norm_gain(enc, _k, b.k_norm, _dim, tokens);
      o.rope(enc, _q, _rope_cos, _rope_sin, _heads, tokens, _head_dim);
      o.rope(enc, _k, _rope_cos, _rope_sin, _heads, tokens, _head_dim);
      o.transpose_abd(enc, _q, _qh, tokens, _heads, _head_dim);
      o.transpose_abd(enc, _k, _kh, tokens, _heads, _head_dim);
      o.transpose_abd(enc, _v, _vh, tokens, _heads, _head_dim);
      // FULL attention, no mask: the registers already replaced every
      // padded position, and the reference discards the mask at that
      // point.
      if (_steel.tq == tokens) {
        o.sdpa_steel(enc, _steel, _qh, _kh, _vh, _oh);
      } else {
        o.sdpa_full(enc, _qh, _kh, _vh, _oh, _heads, tokens, tokens,
                    _head_dim);
      }
      o.transpose_abd(enc, _oh, _o, _heads, tokens, _head_dim);
      if (b.has_gate) {
        o.linear(enc, _a, b.gate_w, &b.gate_b, _gate, tokens, _dim, _heads);
        o.gate_heads(enc, _o, _gate, _heads, tokens, _head_dim);
      }
      o.linear(enc, _o, b.o_w, &b.o_b, _b, tokens, _dim, _dim);
      // x = attn_out + x. PLAIN -- there is no gate on this residual,
      // unlike every residual in the DiT block. In place: ltx_add reads
      // and writes the same index, so aliasing a with out is safe.
      o.add(enc, out, 0, _b, 0, out, tokens * _dim);

      // --- pre-norm + feed-forward, PLAIN residual -------------------
      o.rms_norm_out(enc, out, _a, _dim, tokens);
      o.linear(enc, _a, b.ff_in, &b.ff_in_b, _ff, tokens, _dim, _ff_hidden);
      o.gelu(enc, _ff, _ff, tokens * _ff_hidden);
      o.linear(enc, _ff, b.ff_out, &b.ff_out_b, _b, tokens, _ff_hidden,
               _dim);
      o.add(enc, out, 0, _b, 0, out, tokens * _dim);
    }

    if (_norm_output) {
      // `connector_norm_output`. In place: rms_norm_out with the same
      // buffer both sides is elementwise-safe (each row reads only
      // itself), and the alternative is a copy of the whole context.
      o.rms_norm_out(enc, out, out, _dim, tokens);
    }
  }
  stream.commit().wait();
  return true;
}

namespace {

void
qw_(const QWeight& q,
    const std::function<void(const vpipe::metal_compute::SharedBuffer&)>& fn)
{
  if (q.quantized) { fn(q.codes); fn(q.scales); fn(q.qbias); }
  else { fn(q.w); }
}

}  // namespace

void
Ltx25Connector::for_each_weight(
    const std::function<void(const vpipe::metal_compute::SharedBuffer&)>& fn)
    const
{
  if (!fn) { return; }
  for (const Block& b : _blocks) {
    qw_(b.q_w, fn); qw_(b.k_w, fn); qw_(b.v_w, fn); qw_(b.o_w, fn);
    qw_(b.ff_in, fn); qw_(b.ff_out, fn);
    fn(b.q_b); fn(b.k_b); fn(b.v_b); fn(b.o_b);
    fn(b.q_norm); fn(b.k_norm);
    fn(b.gate_w); fn(b.gate_b);
    fn(b.ff_in_b); fn(b.ff_out_b);
  }
  // The learnable REGISTERS are weights, not scratch: they replace
  // padded positions and are read every forward.
  fn(_registers);
}

void
Ltx25Connector::for_each_scratch(
    const std::function<void(vpipe::metal_compute::SharedBuffer&)>& fn)
{
  if (!fn) { return; }
  // The RoPE tables are built once per token count and read, never
  // written during a forward -- but they are sized by reserve(), so they
  // live and die with the scratch and are wired with it.
  vpipe::metal_compute::SharedBuffer* all[] = {
      &_rope_cos, &_rope_sin, &_a, &_b, &_q, &_k, &_v, &_o,
      &_qh, &_kh, &_vh, &_oh, &_gate, &_ff};
  for (auto* p : all) { fn(*p); }
}

}  // namespace ltx25
