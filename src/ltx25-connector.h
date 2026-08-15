#ifndef VPIPE_LTX25_CONNECTOR_H
#define VPIPE_LTX25_CONNECTOR_H

#include "ltx25-config.h"
#include "ltx25-metal-ops.h"
#include "ltx25-rope.h"

#include "generative-models/weight-set.h"

#include <memory>
#include <string>
#include <vector>

namespace ltx25 {

// The embeddings connector: an 8-layer 1-D transformer that resamples
// the text encoder's projected caption into the DiT's cross-attention
// context. One per stream, and BOTH live in the DiT checkpoint even
// though they belong to the encoder side of the pipeline
// (`caption_proj_before_connector: true`).
//
// Its block is far simpler than the DiT's -- pre-norm, self-attention,
// plain residual, pre-norm, feed-forward, plain residual. No adaLN, no
// timestep, no gate on either residual. Two things still differ from the
// DiT block in ways nothing checks for you:
//
//   * the connector's feed-forward HAS bias (`connector_ff_bias`
//     defaults true and LTX-2.5 does not set it). The DiT's VIDEO
//     feed-forward does not. Same code shape, opposite answer.
//   * PADDED positions are replaced by LEARNABLE REGISTERS before the
//     first block, and the attention mask is then discarded. So the
//     attention is plain full attention -- there is no mask to get
//     wrong, and a port that instead masks the padding computes
//     something the model never does.
//
// The register table is `[128, dim]` and is TILED over the sequence:
// position i takes register `i % 128`. The sequence length must be a
// multiple of 128, which is why the pipeline pads captions to 256.
class Ltx25Connector {
public:
  // Load one connector. `audio` picks the audio twin (2048 wide, 64
  // head_dim) over the video one (4096, 128).
  static std::unique_ptr<Ltx25Connector>
  load(const DitConfig& cfg, vpipe::genai::WeightSet& ws, const MetalOps& ops,
       bool audio, std::string* err);

  // Size the scratch for a sequence of `tokens`. `tokens` must be a
  // multiple of the register count, for the tiling above.
  bool reserve(int tokens, std::string* err);

  // Resample. `in` is bf16 [tokens][dim] (the projected caption), `out`
  // the same shape.
  //
  // `n_valid` is the prompt's real length: positions at or past it are
  // replaced by registers. This is the prefix form of the reference's
  // additive mask, which is what a padded caption produces; a mask with
  // holes in the middle is not something this pipeline emits and is
  // REFUSED rather than approximated.
  bool forward(const vpipe::metal_compute::SharedBuffer& in,
               const vpipe::metal_compute::SharedBuffer& out, int tokens,
               int n_valid, std::string* err);

  int dim() const { return _dim; }
  int registers() const { return _n_registers; }

private:
  Ltx25Connector() = default;

  struct Block {
    vpipe::metal_compute::SharedBuffer q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b;
    vpipe::metal_compute::SharedBuffer q_norm, k_norm;
    vpipe::metal_compute::SharedBuffer gate_w, gate_b;
    vpipe::metal_compute::SharedBuffer ff_in, ff_in_b, ff_out, ff_out_b;
    bool has_gate = false;
  };

  const MetalOps* _ops = nullptr;
  std::vector<Block> _blocks;
  vpipe::metal_compute::SharedBuffer _registers;
  int _dim = 0, _heads = 0, _head_dim = 0, _ff_hidden = 0;
  int _n_registers = 128;
  bool _norm_output = true;
  double _theta = 10000.0;
  bool _rope_f64 = true;
  int _max_pos = 4096;

  RopeTable _rope;
  vpipe::metal_compute::SharedBuffer _rope_cos, _rope_sin;
  int _tokens = 0;
  vpipe::metal_compute::SharedBuffer _a, _b, _q, _k, _v, _o, _qh, _kh, _vh,
      _oh, _gate, _ff;

  // The steel-attention plan for this connector's one shape. Every
  // block here attends the same square sequence, so it is built once
  // per token count rather than per block. Left unbuilt (tq == 0) when
  // steel has no entry point for the head width, which drops the
  // forward back to sdpa_full.
  MetalOps::SteelAttn _steel;
};

}  // namespace ltx25

#endif
