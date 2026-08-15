#ifndef VPIPE_LTX25_MODEL_CONFIG_STAGE_H
#define VPIPE_LTX25_MODEL_CONFIG_STAGE_H

#include "common/flex-data.h"
#include "pipeline/stage-spec.h"
#include "stages/model-config-source.h"

#include <string>
#include <vector>

namespace ltx25 {

// The LTX-2.5-specific generation knobs, as a config SOURCE.
//
// They are here rather than on `generate-video` for the reason
// stages/model-config-source.h gives at length: a stage that serves
// several families accumulates the union of their knobs, and each key is
// inert -- silently -- on whichever family is not resident. None of what
// follows means anything to Wan or MiniMax-H3, so putting it on the
// shared stage would add six keys that do nothing on two of three
// checkpoints.
//
// WHAT IS HERE AND WHY. LTX-2.5 ships two DiTs with the SAME
// architecture and different sampling:
//
//   distilled  a fixed 8-step sigma list, CFG = 1. Guidance-distilled,
//              so a negative prompt has nothing to guide away from.
//   dev        real classifier-free guidance over ~40 steps, with a
//              separate scale for each modality (the reference defaults
//              to 3.0 video / 7.0 audio -- audio needs the stronger
//              push).
//
// so `variant` picks the checkpoint and the guidance keys apply to one
// of them. Setting a guidance scale on a distilled checkpoint is
// reported by the model layer rather than silently ignored.
//
// EVERY KEY IS EMITTED ONLY WHEN SET. The model layer holds LTX-2.5's
// own defaults; a source that helpfully emitted its own would overwrite
// them with something that merely looks configured. `attr_is_set` is
// what tells the two apart.
class Ltx25ModelConfigStage
  : public vpipe::ModelConfigSourceStage<Ltx25ModelConfigStage> {
public:
  static constexpr const char* kTypeName = "ltx-2.5-model-config";

  Ltx25ModelConfigStage(const vpipe::SessionContextIntf* session,
                        std::string                      id,
                        std::vector<vpipe::InEdge>       iports,
                        vpipe::FlexData                  config);

  const vpipe::StageSpec& spec() const noexcept override;

  // The file-static spec, for the plugin's registration call. A plugin
  // attaches its spec through VpipePluginContext::register_stage rather
  // than the in-tree VPIPE_REGISTER_SPEC macro, and that takes a
  // pointer with static storage duration -- so the spec has to be
  // reachable from the plugin entry point, not just from spec().
  static const vpipe::StageSpec* stage_spec() noexcept;

  vpipe::FlexData resolved_config() const;
  void            report_config(const vpipe::FlexData& fd) const;

private:
  // Which DiT to resolve when a root holds both. "" = the shipped
  // preference (distilled).
  std::string _variant;

  // Sampling. `steps` is deliberately NOT here -- it is on
  // generate-video, where every family answers to it.
  double _guidance       = 1.0;   // video CFG (dev only)
  double _audio_guidance = 1.0;   // audio CFG (dev only)
  // Spatio-temporal guidance: a second forward with some blocks skipped,
  // blended in. The reference defaults it to 1.0 on dev and 0.0 on
  // distilled, and it costs a whole extra forward, so it is off unless
  // asked for.
  double _stg           = 0.0;
  double _audio_stg     = 0.0;
  // How strongly each modality is pulled toward the other across the
  // audio<->video cross-attention. 1.0 is as trained.
  double _modality       = 1.0;
  double _audio_modality = 1.0;

  // Soundtrack. 0 derives it from frames / fps, which is what keeps the
  // two modalities the same length by construction.
  double _audio_seconds = 0.0;
  // Generate the soundtrack at all. A graph that only wants pixels pays
  // for the audio stream anyway -- it is fused into every block -- but
  // it can skip the audio VAE and the vocoder.
  bool   _audio         = true;

  // The optional duration head: with it wired ON and no explicit frame
  // count, the model picks the clip length from the prompt.
  bool   _duration_head = false;

  // A runtime LoRA (the shipped distilled-450 adapter, or a trained
  // one). Load-time, like every adapter in this tree.
  std::string _lora;
  double      _lora_scale = 1.0;

  // Which keys the graph actually SET, so resolved_config() can emit
  // only those. See the class note.
  bool _set_guidance = false, _set_audio_guidance = false;
  bool _set_stg = false, _set_audio_stg = false;
  bool _set_modality = false, _set_audio_modality = false;
  bool _set_audio_seconds = false, _set_audio = false;
  bool _set_duration_head = false;
};

}  // namespace ltx25

#endif
