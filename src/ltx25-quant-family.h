#ifndef VPIPE_LTX25_QUANT_FAMILY_H
#define VPIPE_LTX25_QUANT_FAMILY_H

#include "generative-models/quantize-family-registry.h"

#include <string>
#include <string_view>
#include <vector>

namespace ltx25 {

// LTX-2.5 as a QUANTIZABLE family, so `model-quantize` packages this
// checkpoint the way it packages a built-in one: one directory holding a
// quantized DiT, a quantized text encoder, and the VAEs untouched.
//
// Without this the stage's detection -- a closed switch over in-tree DiT
// class names -- does not recognise an LTX repo, and the pass falls
// through to the single-component path. That path WORKS -- it is what a
// `model-quantize` pointed straight at one .safetensors still takes --
// but it is handed one file at a time and writes its output beside the
// file it read, so the result is a repo with packs scattered through it
// rather than a model that can be moved, registered and named as one
// thing.
//
// WHAT IS AND IS NOT DESCRIBED HERE. The two `scope`s below are the
// whole recipe, and both are load-bearing:
//
//   * the DiT quantizes `transformer_blocks.` and EXCLUDES the f32
//     modulation tables and the [32, dim] gate logits. The wholesale
//     rule would take both -- they are 2-D floats whose leaf is neither
//     a norm nor an embedding -- and quantizing either produces a
//     checkpoint that loads, runs, and generates the wrong thing.
//   * the encoder quantizes `model.layers.`, which is the 20.3 GB Gemma
//     backbone, and thereby leaves the 1.88 GB embedding table and LTX's
//     own 2.16 GB `text_embedding_projection.` dense. The embedding
//     table is host-gathered as a plain table and the projection is read
//     as a dense matrix by Ltx25TextFeatures; neither has a quantized
//     reader.
//
// The VAEs are absent on purpose rather than by omission: they are conv
// nets read through dense buffers, there is no `QWeight` path for them,
// and both together are 1.4 GB against the DiT's 42.
class Ltx25QuantFamily final : public vpipe::genai::QuantizableFamily {
public:
  std::string_view tag() const noexcept override;

  bool claims(const std::string& root) const override;

  std::vector<vpipe::genai::QuantizableComponent>
  components() const override;
};

}  // namespace ltx25

#endif
