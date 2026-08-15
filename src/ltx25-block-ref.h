#ifndef VPIPE_LTX25_BLOCK_REF_H
#define VPIPE_LTX25_BLOCK_REF_H

#include "ltx25-rope.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace ltx25 {

// A plain-C++ reference for one `BasicAVTransformerBlock`.
//
// WHY A CPU REFERENCE EXISTS AT ALL. This block is where LTX-2.5 is
// unlike anything else in the tree: nine adaLN rows sliced three ways,
// a text cross-attention that modulates K/V as well as Q, two more
// cross-attentions between the modalities that must both read the
// PRE-cross state, and per-head gating on every one of them. Every one
// of those is invisible in the output SHAPE. Writing the Metal path
// first means debugging kernel bugs and semantic bugs at the same time,
// on a 22B model, through a 48-block stack.
//
// So this comes first and is pinned against the reference on a small
// config (dim 64/32, 4 heads) where every weight fits in a golden. The
// Metal forward is then checked against THIS, locally, one op at a time
// -- which is what checking a new kernel against a serial reference
// path means in practice.
//
// It is not a fallback and must never be on a hot path: it is dense f32
// on one thread, so a real block would take minutes.

// Row-major matrix, [rows][cols].
struct Mat {
  std::vector<float> v;
  int rows = 0;
  int cols = 0;

  float*       at(int r)       { return v.data() + (std::size_t)r * cols; }
  const float* at(int r) const { return v.data() + (std::size_t)r * cols; }
};

// One attention's parameters. `to_gate_logits` is optional (null rows
// when the block does not gate), everything else is required.
struct AttnWeights {
  Mat q_w, k_w, v_w, o_w;              // [out][in]
  std::vector<float> q_b, k_b, v_b, o_b;
  std::vector<float> q_norm, k_norm;   // RMSNorm gains, inner_dim wide
  Mat gate_w;                          // [heads][query_dim], may be empty
  std::vector<float> gate_b;
  int heads = 0;
  int head_dim = 0;
};

// One stream's half of the block.
struct StreamWeights {
  AttnWeights attn1;                   // self
  AttnWeights attn2;                   // cross, to text
  Mat ff_in, ff_out;                   // [4d][d] and [d][4d]
  std::vector<float> ff_in_b, ff_out_b;   // empty when the stream has no bias
  std::vector<float> scale_shift;      // 9 * dim
  std::vector<float> prompt_scale_shift;   // 2 * dim
  std::vector<float> cross_table;      // 5 * dim (a2v/v2a scale-shift+gate)
  int dim = 0;
};

struct BlockWeights {
  StreamWeights video;
  StreamWeights audio;
  AttnWeights   a2v;    // video queries, audio keys/values
  AttnWeights   v2a;    // audio queries, video keys/values
  double        norm_eps = 1e-6;
};

// One stream's per-forward inputs.
struct StreamInput {
  Mat x;                               // [tokens][dim]
  Mat context;                         // [text_tokens][dim]
  std::vector<float> timesteps;        // 9 * dim
  std::vector<float> cross_scale_shift;   // 4 * dim
  std::vector<float> cross_gate;          // dim
  // The PROMPT adaLN driver, 2 * dim. `use_prompt_adaln_single` defaults
  // TRUE and LTX-2.5 does not override it, so the text cross-attention's
  // K/V modulation is timestep-DEPENDENT: prompt_scale_shift_table plus
  // this. Empty means the static table alone, which is what a checkpoint
  // that turns the prompt MLP off wants -- and what a golden generated
  // with prompt_timestep=None silently pins.
  std::vector<float> prompt_timestep;
  const RopeTable* pe = nullptr;       // self-attention table
  const RopeTable* cross_pe = nullptr; // a2v / v2a table (audio width)
  bool present = true;                 // false => this stream is absent
};

// Run the block. `video` / `audio` are updated in place. Returns false
// (with `err` set) on any shape disagreement, rather than computing
// something plausible from mismatched parts.
bool block_forward(const BlockWeights& w, StreamInput& video,
                   StreamInput& audio, std::string* err = nullptr);

// Build a BlockWeights from a name -> tensor map, using the reference's
// own parameter names (`attn1.to_q.weight`, `audio_ff.net.0.proj.bias`,
// `scale_shift_table_a2v_ca_video`, ...). Returns false naming the first
// tensor it could not find -- a block assembled from a partial map
// computes cleanly and matches nothing, which is the failure this
// refuses to allow.
struct NamedTensor {
  std::vector<float> data;
  std::vector<int>   shape;
};
bool load_block_weights(
    const std::unordered_map<std::string, NamedTensor>& t,
    int vdim, int vheads, int adim, int aheads, BlockWeights& out,
    std::string* err = nullptr);

// The pieces, exposed because each is separately checkable and each has
// a way to be silently wrong.

// RMSNorm with no learnable gain (`rms_norm(x, eps)` in the reference).
void rms_norm(float* x, int rows, int cols, double eps);

// `rms_norm(x) * (1 + scale) + shift`, the reference's ada-zero.
void ada_zero(const float* x, float* out, int rows, int cols, double eps,
              const float* scale, const float* shift);

// The adaLN rows for one modulation group: `table[i] + timesteps[i]`,
// for i in [lo, hi). `table` is `n_rows * dim`, `timesteps` the same.
// Returns hi-lo vectors of `dim` each.
std::vector<std::vector<float>>
ada_values(const std::vector<float>& table, const std::vector<float>& timesteps,
           int n_rows, int dim, int lo, int hi);

}  // namespace ltx25

#endif
