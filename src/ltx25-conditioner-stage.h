#ifndef VPIPE_LTX25_CONDITIONER_STAGE_H
#define VPIPE_LTX25_CONDITIONER_STAGE_H

#include "ltx25-config.h"
#include "ltx25-metal-ops.h"
#include "ltx25-text-encoder.h"

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/stage-spec.h"
#include "pipeline/typed-stage.h"
#include "stages/model-memory.h"

#include <memory>
#include <string>
#include <vector>

namespace ltx25 {

// The LTX-2.5 text path as a stage: a caption in, the DiT's TWO
// cross-attention contexts out.
//
// ---- WHY THIS IS NOT `diffusion-conditioner` ----
//
// The host conditioner serves the image families and MiniMax-H3, and
// every one of them produces ONE context from an encoder it knows how
// to build (umT5, Qwen3-VL, a Qwen3 dense tap). LTX-2.5 matches neither
// half of that:
//
//   * its encoder is Gemma-4 12B read through `HiddenStateEncoder` with
//     ALL 49 hidden states tapped, then LTX's own per-token RMS,
//     layer-fastest flatten, and a 188160-wide projection -- none of
//     which the host conditioner has a shape for;
//   * it produces TWO contexts, 4096 for video and 2048 for audio, from
//     the SAME hidden states through two different projections. A
//     one-context stage cannot emit the second, and that is exactly why
//     `generate-video` has been running LTX video-only.
//
// Adding a fourth family branch to the host conditioner would put LTX's
// projections in the host, which is the thing this plugin exists to
// avoid. So the family brings its own conditioner, the same way it
// brings its own model-config source.
//
//   iport0  prompt text (FlexData string or {text: ...})
//   iport1  OPTIONAL negative prompt, for CFG on the `dev` checkpoint
//   iport2  OPTIONAL FlexDataPayload model reference (model-select),
//           overriding hf_dir the way every other model stage does
//
//   oport0  TensorBeatPayload bf16 [pad_to, 4096] -- the VIDEO context,
//           straight into `generate-video`'s iport0
//   oport1  the NEGATIVE video context, into iport1. Emitted only when
//           a non-empty negative is set AND the checkpoint is `dev`
//   oport2  TensorBeatPayload bf16 [pad_to, 2048] -- the AUDIO context
//   oport3  the NEGATIVE audio context. Separate from oport1 because
//           LTX guides the two streams at DIFFERENT scales (3.0 video
//           against 7.0 audio in the reference), so one cannot serve both
//
// The 0/1 ordering is POSITIONAL with generate-video's iports, matching
// `diffusion-conditioner`. Putting the audio context on oport1 -- where
// this stage started -- made the obvious wiring feed a 2048-wide audio
// context into the port that wants a 4096-wide negative.
//
// ---- THE SIDEBAND, and why it is not optional ----
//
// LTX pads its captions on the LEFT (PaddingSide.LEFT, 1024 tokens), so
// the real tokens are a SUFFIX. Both beats therefore carry
// `valid_begin` / `valid_count`, because a consumer that assumes a
// prefix -- which every other conditioner in this tree emits -- would
// build a mask keeping precisely the empty rows.
class Ltx25ConditionerStage final
  : public vpipe::TypedStage<Ltx25ConditionerStage> {
public:
  static constexpr const char* kTypeName = "ltx-2.5-conditioner";

  Ltx25ConditionerStage(const vpipe::SessionContextIntf* session,
                        std::string                      id,
                        std::vector<vpipe::InEdge>       iports,
                        vpipe::FlexData                  config);
  ~Ltx25ConditionerStage() override;

  const vpipe::StageSpec& spec() const noexcept override;
  static const vpipe::StageSpec* stage_spec() noexcept;

  vpipe::Job initialize(vpipe::RuntimeContext& ctx) override;
  vpipe::Job process(vpipe::RuntimeContext& ctx) override;

  // The 12B encoder is the second-largest thing an LTX graph loads, and
  // nothing else declares it -- without this the DiT stage sizes itself
  // against a box it does not know is about to hold 24 GB.
  // The model-select beat, delivered BEFORE the planning phase.
  //
  // Without this the stage learns its checkpoint at the first process(),
  // which is after every peer has sized itself -- so declare_resources()
  // below sees an empty directory and declares NOTHING, and a 15 GB text
  // encoder is invisible to the graph that has to make room for it.
  void apply_constant(unsigned iport, const vpipe::FlexData& beat) override;

  std::vector<vpipe::ResourceClaim> declare_resources() const override;

  // Pass two: re-declare the encoder as a CONDITION-phase claim once
  // every stage has declared, so a peer sizing the denoise does not count
  // an encoder that will be gone by then. Only when it will genuinely be
  // gone -- see the body.
  std::vector<vpipe::ResourceClaim> decide_resources() const override;

  void reset_run_state() override;

private:
  bool ensure_loaded_(const std::string& root);

  // Resolve _idle_action from the post-barrier picture. Idempotent.
  void resolve_idle_policy_();
  // Apply whichever of destroy/park/keep was resolved. The single call
  // site for "this prompt is done with me".
  void release_encoder_when_idle_();
  void destroy_encoder_();
  void park_encoder_();

  std::string _hf_dir;
  // config `encoder_variant`: a name substring picking a quantized
  // encoder directory, or empty for the released bf16 file.
  std::string _enc_variant;
  std::string _negative;
  bool _warned_distilled_neg = false;
  int _pad_to = Ltx25TextEncoder::kMaxTokens;
  // What happens to the encoder between prompts, and how it was asked
  // for. `_idle_action` is never kAuto once resolve_idle_policy_() has
  // run; the resolution happens AFTER the init barrier (at the first
  // process()), because at construction the graph is half loaded and a
  // stage would size against peers that have not appeared yet.
  vpipe::model_memory::UnloadPolicy _unload_cfg =
      vpipe::model_memory::UnloadPolicy::kAuto;
  vpipe::model_memory::UnloadPolicy _idle_action =
      vpipe::model_memory::UnloadPolicy::kKeep;
  bool _idle_resolved = false;
  bool _have_model_port = false;

  Config _cfg;
  MetalOps _ops;
  std::shared_ptr<vpipe::genai::WeightSet> _ws;
  std::unique_ptr<Ltx25TextEncoder> _enc;
  std::string _loaded_root;
  int _emitted = 0;
};

}  // namespace ltx25

#endif
