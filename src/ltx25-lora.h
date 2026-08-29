#ifndef VPIPE_LTX25_LORA_H
#define VPIPE_LTX25_LORA_H

#include "ltx25-config.h"

#include "generative-models/weight-set.h"
#include "apple-silicon/metal-compute/metal-compute.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace ltx25 {

// A low-rank adapter over the DiT, applied at RUNTIME.
//
// WHY NOT FUSED INTO THE WEIGHTS, which is what `lora_scale`'s "folded
// in at load" wording used to suggest. Two reasons, either decisive:
//
//   * The DiT STREAMS its 48 blocks. A fused weight is a NEW tensor, so
//     it must be cached to be worth anything -- and caching the fused
//     set is ~25.8 GB of bf16, the whole stack. Fusing would quietly
//     turn a model that runs on a 16 GB box into one that does not.
//     Re-fusing per refill instead puts a 4096x4096 rebuild on every
//     block of every step.
//   * A quantized pack holds u32 codes, not weights. There is nothing
//     to add a bf16 delta to without dequantizing first.
//
// Applied instead as `y += (x @ A^T) @ B^T` where each linear is
// consumed, which touches no base weight, works over a quantized base
// unchanged, and costs 2*rank/K -- about 1.6% of a 4096-wide linear at
// rank 32, and ~22% at the distilled adapter's rank 450.
//
// RANK IS PER MODULE, not per adapter. MEASURED on the two published
// LTX-2.5 adapters: the IC-LoRA upscaler is rank 32 throughout, while
// the distilled-450 adapter is rank 450 for the projections and rank 32
// for every `to_gate_logits` in the same file. An adapter is a bag of
// independently-shaped pairs and is read as one.
//
// SCALE. `lora_scale` multiplies the delta and is folded into A at
// load, A being the small side. An adapter whose `__metadata__` names
// `lora_alpha` and `lora_rank` also carries the usual `alpha/rank`
// factor, which is folded into the SAME place -- both published
// adapters make that 1.0 (450/450), so nothing in the tree exercises a
// non-unit alpha yet and it is applied rather than assumed away.

// One module's pair. `a` is [rank][k] and `b` is [n][rank], both bf16
// in PyTorch's [out][in] order -- the order MetalOps::linear already
// reads a weight in, so neither is transposed at load.
struct LoraPair {
  vpipe::metal_compute::SharedBuffer a, b;
  int rank = 0, k = 0, n = 0;

  bool valid() const noexcept { return rank > 0 && k > 0 && n > 0; }
};

// The adapted linears of one attention module.
struct LoraAttn {
  LoraPair q, k, v, o, gate;
};

// One transformer block's adapted linears, named to match GpuBlock-
// Weights rather than the checkpoint, because that is where they are
// consumed.
struct LoraBlock {
  LoraAttn video_attn1, video_attn2;
  LoraPair video_ff_in, video_ff_out;
  LoraAttn audio_attn1, audio_attn2;
  LoraPair audio_ff_in, audio_ff_out;
  LoraAttn a2v, v2a;

  bool any() const noexcept;
};

// One adaLN chain's three linears, matching DitTrunk::AdaLN.
//
// THESE ARE HOST GEMVs, not GPU matmuls -- `Ltx25Dit::adaln_` runs the
// chain on the CPU once per step and the bake turns it into a table.
// So the adapter here is applied on the host too, as `y += B @ (A @ x)`
// over a single vector: two mat-vecs against a rank-450 pair, against
// chains the model already walks eight times a forward. It is the
// cheapest of the three sites, not the most expensive, which is worth
// saying because "adaLN is baked" reads like the hard case.
//
// It also has to happen BEFORE the bake. bake_adaln() precomputes every
// modulation the schedule will ask for and then RELEASES the
// projections; an adapter arriving after that has nothing left to
// adapt, and the DiT refuses a re-bake rather than modulating from
// half-adapted weights.
struct LoraAdaLN {
  LoraPair emb1, emb2, out;
};

// Everything outside the blocks: the two patchify projections, the two
// output heads, and the eight model-level adaLN chains.
//
// The IC-LoRA upscaler carries none of this and the distilled-450
// adapter carries all of it, which is why the loader cannot treat the
// trunk as optional decoration -- an adapter that names these and does
// not get them applied is wrong by exactly the modules it named.
struct LoraTrunk {
  LoraPair patchify, proj_out;
  LoraPair audio_patchify, audio_proj_out;
  LoraAdaLN video, audio, prompt, audio_prompt;
  LoraAdaLN av_video_ss, av_audio_ss, av_a2v_gate, av_v2a_gate;

  bool any() const noexcept;
};

class LoraAdapter {
public:
  // Bind every `diffusion_model.transformer_blocks.{n}.{mod}.lora_{A,B}
  // .weight` pair in `ws`, folding `scale` (and any metadata alpha/rank)
  // into A.
  //
  // `file` is the adapter's own .safetensors, read only for its
  // `__metadata__` -- an IC-LoRA's reference geometry and an adapter's
  // alpha both live there rather than in its tensors. Empty skips it.
  //
  // REFUSES rather than binding what it recognised, on:
  //   * a pair whose two shapes disagree, or whose widths do not match
  //     the base linear they name;
  //   * an adapter that resolved NO module;
  //   * an adapter naming a module this does not apply.
  //
  // That last one is the important refusal. A LoRA applied to 80% of
  // the modules it names does not fail: it renders, slightly wrong,
  // and nothing in the output says which fifth was dropped. `unmapped`
  // (optional) receives the offending names so the caller can say them.
  static std::unique_ptr<LoraAdapter>
  load(vpipe::genai::WeightSet& ws,
       vpipe::metal_compute::MetalCompute* mc, const DitConfig& cfg,
       double scale, const std::string& file,
       std::vector<std::string>* unmapped, std::string* err);

  // Null when this layer carries no adapted module.
  const LoraBlock* block(int layer) const noexcept;

  // Null when the adapter touches no trunk weight (the IC-LoRA).
  const LoraTrunk* trunk() const noexcept;

  // Blocks this adapter was built for, from the config it was loaded
  // against. A model with a different depth is a different checkpoint.
  int layers() const noexcept { return (int)_blocks.size(); }
  int modules() const noexcept { return _modules; }
  int max_rank() const noexcept { return _max_rank; }
  std::size_t bytes() const noexcept { return _bytes; }

  // The widest output any adapted linear writes, so the caller can size
  // the single scratch the runtime path needs.
  int max_out() const noexcept { return _max_out; }

  // From the adapter's `__metadata__`. An IC-LoRA is trained with the
  // reference at 1/factor of the target's linear resolution; 1 means a
  // plain LoRA, which is also what an adapter publishing no such key
  // gets.
  int reference_downscale_factor() const noexcept { return _ref_downscale; }

  // The TEMPORAL ratio, target fps over reference fps. 1 for everything
  // published so far -- the pixel spatial upscaler carries only
  // `reference_downscale_factor` -- and the conditioning refuses
  // anything else rather than guessing at the time spacing.
  int reference_temporal_scale_factor() const noexcept
  {
    return _ref_temporal;
  }

  // True when this adapter expects an in-context reference. The
  // generator uses it to refuse the combination that fails SILENTLY: an
  // IC-LoRA with no reference wired renders a plausible clip that
  // ignored the input it was asked to upscale.
  bool wants_reference() const noexcept { return _ref_downscale > 1; }

private:
  std::vector<LoraBlock> _blocks;
  LoraTrunk   _trunk;
  int         _modules = 0;
  int         _max_rank = 0;
  int         _max_out = 0;
  int         _ref_downscale = 1;
  int         _ref_temporal = 1;
  std::size_t _bytes = 0;
};

}  // namespace ltx25

#endif
