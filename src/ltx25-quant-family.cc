#include "ltx25-quant-family.h"

#include "ltx25-config.h"

namespace ltx25 {

using vpipe::genai::QuantizableComponent;

std::string_view
Ltx25QuantFamily::tag() const noexcept
{
  return kFamily;
}

bool
Ltx25QuantFamily::claims(const std::string& root) const
{
  // The same probe the video family uses, and for the same reason:
  // `resolve` refuses anything whose DiT `_class_name` is not
  // AVTransformer3DModel, so a hit is both the detection and proof the
  // repo is readable, and a miss costs one safetensors header read.
  //
  // It has to hold for a PARTIALLY quantized repo too, because that is
  // exactly what the second pass of a chain is handed -- a model whose
  // `diffusion_models/` is already a directory checkpoint. `resolve_dit_`
  // accepting that shape is what makes this claim survive the first
  // pass; without it a chain would quantize the DiT and then fail to
  // recognise its own output.
  //
  // No preferred variant: asking for one here would make the claim
  // depend on which pack happens to be present, and the question being
  // asked is "is this an LTX-2.5 repo", not "which DiT would I load".
  Config cfg;
  return resolve(root, cfg, nullptr, std::string());
}

std::vector<QuantizableComponent>
Ltx25QuantFamily::components() const
{
  QuantizableComponent dit;
  dit.target   = "dit";
  dit.role     = "diffusion_models";
  dit.meta_key = kDitMetaKey;
  dit.prefer   = {"distilled", "dev"};
  // BOTH block stacks: the DiT's own `transformer_blocks.` and the two
  // text connectors' `transformer_1d_blocks.`. `scope` is a SUBSTRING and
  // `_blocks.` is the one both share, which is why it is spelled that way
  // rather than as the DiT's prefix.
  //
  // The connectors were outside the old scope and so shipped DENSE in a
  // pack asked for at w8 -- 3.75 GB of bf16, and the largest single thing
  // in a checkpoint whose whole point was to be small. Worse, they never
  // stream: the trunk is resident for the entire denoise, so on a bounded
  // box that was 3.75 GB off the top before a single block could be
  // pinned. VERIFIED against the released checkpoint, this selects 1344
  // block tensors + 96 connector tensors and leaves the connectors' 162
  // norms, biases and learnable registers (6 MB) alone -- the wholesale
  // rule's 2-D / non-norm / non-embed test does the rest.
  //
  // Ltx25Connector reads through bind_qlinear for exactly this reason. A
  // dense-only reader against such a pack does not fail; it binds the u32
  // codes as bf16 and conditions the model on noise at full speed. The
  // scope and the reader move together or not at all.
  dit.scope        = "_blocks.";
  dit.all_in_scope = true;
  // Within the stack, the two that are 2-D floats but not matrices.
  // `scale_shift` are f32 modulation TABLES; `to_gate_logits` is the
  // [32, dim] per-head gate. Quantizing either is a silent wrong answer,
  // not a quality loss.
  dit.exclude       = "scale_shift,to_gate_logits";
  dit.component_tag = "ltx-2.5-dit";

  QuantizableComponent enc;
  enc.target   = "text_encoder";
  enc.role     = "text_encoders";
  enc.meta_key = kEncMetaKey;
  enc.prefer   = {"gemma4", "with-proj"};
  // The Gemma backbone. This one prefix is the whole recipe: it takes
  // the 20.3 GB of decoder layers and leaves BOTH the 1.88 GB embedding
  // table (host-gathered as a plain table) and LTX's own 2.16 GB
  // `text_embedding_projection.` (read dense by Ltx25TextFeatures)
  // untouched, so neither needs an exclude.
  enc.scope        = "model.layers.";
  enc.all_in_scope = true;
  enc.component_tag = "ltx-2.5-text-encoder";

  // DiT first: it is the larger win and the one a caller who names no
  // target most likely means.
  return {dit, enc};
}

}  // namespace ltx25
