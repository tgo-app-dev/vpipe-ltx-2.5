#include "ltx25-model-config-stage.h"
#include "ltx25-config.h"

#include "common/beat-payload-intf.h"
#include "stages/model-registry.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <string>
#include <utility>

using vpipe::ConfigKey;
using vpipe::ConfigType;
using vpipe::FlexData;
using vpipe::FlexDataPayload;
using vpipe::InEdge;
using vpipe::PortSpec;
using vpipe::SessionContextIntf;
using vpipe::StageCategory;
using vpipe::StageSpec;
using vpipe::fmt;
using vpipe::model_config::make_config;

namespace ltx25 {

namespace {

const ConfigKey kAttrs[] = {
  {.key = "variant", .type = ConfigType::String, .required = false,
   .doc = "which DiT to resolve when the checkpoint root holds more than "
          "one. 'distilled' (a fixed 8-step schedule at CFG 1 -- guidance-"
          "DISTILLED, so the guidance keys below do nothing) or 'dev' "
          "(real CFG over ~40 steps, trainable). Also names a QUANTIZED "
          "pack written into diffusion_models/ by model-quantize -- "
          "'w8g64', say -- since the value is matched against what is "
          "actually there rather than against a fixed list. Empty prefers "
          "distilled bf16, which is what this port is verified against. "
          "VPIPE_LTX25_VARIANT overrides it, and warns when it does",
   .def_str = ""},

  {.key = "guidance", .type = ConfigType::Real, .required = false,
   .doc = "classifier-free guidance for the VIDEO stream, on the 'dev' "
          "checkpoint. 0 means unset -- the model layer then uses the "
          "reference default (3.0). Ignored, with a warning, on a "
          "distilled checkpoint, which has no unconditional pass to blend "
          "with; running one would double the cost of a 22B model for "
          "nothing",
   .def_real = 0.0},
  {.key = "audio_guidance", .type = ConfigType::Real, .required = false,
   .doc = "the same for the AUDIO stream. Separate because the two are "
          "not interchangeable: the reference guides audio more than twice "
          "as hard (7.0 against video's 3.0). 0 means unset",
   .def_real = 0.0},

  {.key = "stg_scale", .type = ConfigType::Real, .required = false,
   .doc = "spatio-temporal guidance for the video stream: a SECOND forward "
          "with some blocks perturbed, blended into the result. Costs a "
          "whole extra pass over 48 blocks of a 22B model, so it is off (0) "
          "unless asked for. The reference uses 1.0 on 'dev'",
   .def_real = 0.0},
  {.key = "audio_stg_scale", .type = ConfigType::Real, .required = false,
   .doc = "the same for the audio stream", .def_real = 0.0},

  {.key = "modality_scale", .type = ConfigType::Real, .required = false,
   .doc = "how hard the video stream is pulled toward the audio one across "
          "the per-block audio<->video cross-attention. 1.0 is as trained; "
          "0 means unset. Raise it when the picture ignores the "
          "soundtrack's rhythm, lower it when motion looks driven by audio "
          "rather than by the prompt",
   .def_real = 0.0},
  {.key = "audio_modality_scale", .type = ConfigType::Real, .required = false,
   .doc = "the reverse direction: how hard the soundtrack follows the "
          "picture. 0 means unset", .def_real = 0.0},

  {.key = "audio", .type = ConfigType::String, .required = false,
   .doc = "whether to produce a soundtrack: auto|on|off. This model "
          "generates both modalities in ONE packed forward, so 'off' does "
          "not make the DiT cheaper -- it skips the audio VAE and the "
          "vocoder, and leaves oport1 unwritten. 'auto' follows whether "
          "the graph wired anything to oport1",
   .def_str = "auto"},
  {.key = "audio_seconds", .type = ConfigType::Real, .required = false,
   .doc = "soundtrack duration. 0 derives it from the video's frames / fps, "
          "which is what keeps the two modalities the same length by "
          "construction -- set it only to deliberately over- or under-run "
          "the picture",
   .def_real = 0.0},

  {.key = "duration_head", .type = ConfigType::Bool, .required = false,
   .doc = "let the optional duration head pick the clip length from the "
          "prompt instead of using generate-video's `frames`. Needs "
          "model_patches/ltx-2.5-duration-head in the checkpoint; without "
          "it this is reported and ignored rather than silently doing "
          "nothing",
   .def_bool = false},

  {.key = "lora", .type = ConfigType::String, .required = false,
   .doc = "a LoRA applied at LOAD time. A registered model key (what a "
          "`model-fetch` of the shipped distilled-450 adapter writes), a "
          "directory holding one .safetensors, or a path to one. Read "
          "before the DiT is built, so a beat that changes it afterwards "
          "is reported and ignored",
   .suggest_db = vpipe::kModelRegistryDb,
   // Without a type the picker shows nothing: every LoRA is catalogued
   // as a `supplement`, and a field with no type offers plain models
   // only. Naming it also keeps a Wan or Krea-2 adapter out of this
   // field.
   .suggest_db_type = "ltx-2.5-lora"},
  {.key = "lora_scale", .type = ConfigType::Real, .required = false,
   .doc = "adapter strength, folded into A at load. 1.0 is as trained",
   .def_real = 1.0},
};

const PortSpec kIports[] = {
  {.name = "trigger",
   .doc = "OPTIONAL beat that gates re-emitting the config (a chrono tick, a "
          "prompt source, a feedback loop). Any payload -- receipt is the "
          "signal. Unwired, the stage emits once for the run",
   .type = nullptr, .clock_group = 0},
};

const PortSpec kOports[] = {
  {.name = "model_config",
   .doc = "LTX-2.5 generation parameters as one FlexData object "
          "{model_family: ltx-2.5, ...}, for a generate-video model_config "
          "iport. Only the keys the graph actually set are present, so the "
          "model layer's own defaults survive",
   .type = &typeid(FlexDataPayload),
   .tags = "model-config", .clock_group = 0},
};

const StageSpec kSpec = {
  .type_name = "ltx-2.5-model-config",
  .doc       = "Source: the LTX-2.5-specific generation parameters -- which "
               "of the two DiTs to run, the per-modality guidance and "
               "spatio-temporal guidance scales, how hard the audio and "
               "video streams pull on each other, the soundtrack duration, "
               "and an optional runtime LoRA -- as one FlexData beat for "
               "generate-video to latch. LTX-2.5 generates picture and "
               "soundtrack in ONE packed forward, which is why the audio "
               "knobs live beside the video ones rather than on the audio "
               "decode stage. One beat then done; with a trigger iport, one "
               "beat per inbound beat.",
  .display_name = "LTX-2.5 Model Config",
  .category  = StageCategory::ModelSpecificConfig,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

}  // namespace

Ltx25ModelConfigStage::Ltx25ModelConfigStage(const SessionContextIntf* s,
                                             std::string               id,
                                             std::vector<InEdge>       iports,
                                             FlexData                  config)
  : vpipe::ModelConfigSourceStage<Ltx25ModelConfigStage>(s, std::move(id),
                                                         std::move(iports),
                                                         std::move(config))
{
  // Which keys the GRAPH set, as opposed to which have a schema default.
  // The distinction is the whole contract here: the model layer holds
  // LTX-2.5's own numbers, and a source that emitted its defaults would
  // overwrite them with something that merely looks configured.
  auto was_set = [this](const char* k) {
    const FlexData& c = this->config();
    if (!c.is_object()) { return false; }
    return c.as_object().contains(k);
  };

  _variant = attr_str("variant");

  _guidance       = attr_real("guidance");
  _audio_guidance = attr_real("audio_guidance");
  _stg            = attr_real("stg_scale");
  _audio_stg      = attr_real("audio_stg_scale");
  _modality       = attr_real("modality_scale");
  _audio_modality = attr_real("audio_modality_scale");
  _audio_seconds  = attr_real("audio_seconds");
  _duration_head  = attr_bool("duration_head");
  _lora           = attr_str("lora");
  _lora_scale     = attr_real("lora_scale");

  const std::string audio = attr_str("audio");
  _audio = (audio != "off");

  _set_guidance       = was_set("guidance");
  _set_audio_guidance = was_set("audio_guidance");
  _set_stg            = was_set("stg_scale");
  _set_audio_stg      = was_set("audio_stg_scale");
  _set_modality       = was_set("modality_scale");
  _set_audio_modality = was_set("audio_modality_scale");
  _set_audio_seconds  = was_set("audio_seconds");
  _set_duration_head  = was_set("duration_head");
  _set_audio          = was_set("audio") && audio != "auto";

  // `variant` is NOT validated against a fixed set, and that is the
  // point: it is matched as a substring against what is actually under
  // `diffusion_models/`, so besides the two released DiTs it also names
  // a quantized pack (`w8g64`) or anything else `model-quantize` wrote
  // there. A closed list would have to be edited every time someone
  // produced a new pack, and would reject the one thing they most want
  // to select. A value that matches nothing falls back to the shipped
  // preference and the family LOGS which file it loaded, so a typo
  // surfaces as "not what I asked for" in one line rather than as a
  // config error about a vocabulary this stage does not own.
  if (audio != "auto" && audio != "on" && audio != "off") {
    fail_config(fmt(
        "Ltx25ModelConfigStage('{}'): audio must be auto|on|off (got '{}')",
        this->id(), audio));
  }
  // Negative scales are not a stylistic choice, they invert the guidance
  // direction and produce noise at full 22B cost.
  auto bad = [](double v) { return v < 0.0; };
  if (bad(_guidance) || bad(_audio_guidance) || bad(_stg) ||
      bad(_audio_stg) || bad(_modality) || bad(_audio_modality)) {
    fail_config(fmt(
        "Ltx25ModelConfigStage('{}'): guidance / stg / modality scales must "
        "be >= 0 (0 means unset)", this->id()));
  }
  if (_audio_seconds < 0.0) {
    fail_config(fmt(
        "Ltx25ModelConfigStage('{}'): audio_seconds must be >= 0; 0 derives "
        "it from the video (got {:.3f})", this->id(), _audio_seconds));
  }
  if (!(_lora_scale >= 0.0)) {
    fail_config(fmt(
        "Ltx25ModelConfigStage('{}'): lora_scale must be >= 0 (got {:.3f})",
        this->id(), _lora_scale));
  }
  allocate_oports(spec().oports.size());
}

const StageSpec&
Ltx25ModelConfigStage::spec() const noexcept
{
  return kSpec;
}

const StageSpec*
Ltx25ModelConfigStage::stage_spec() noexcept
{
  return &kSpec;
}

FlexData
Ltx25ModelConfigStage::resolved_config() const
{
  FlexData fd = make_config(kFamily);
  auto o = fd.as_object();
  auto put_real = [&o](bool set, const char* k, double v) {
    if (set) { o.insert_or_assign(k, FlexData::make_real(v)); }
  };
  if (!_variant.empty()) {
    o.insert_or_assign("variant", FlexData::make_string(_variant));
  }
  put_real(_set_guidance,       "guidance",             _guidance);
  put_real(_set_audio_guidance, "audio_guidance",       _audio_guidance);
  put_real(_set_stg,            "stg_scale",            _stg);
  put_real(_set_audio_stg,      "audio_stg_scale",      _audio_stg);
  put_real(_set_modality,       "modality_scale",       _modality);
  put_real(_set_audio_modality, "audio_modality_scale", _audio_modality);
  put_real(_set_audio_seconds,  "audio_seconds",        _audio_seconds);
  if (_set_audio) {
    o.insert_or_assign("audio", FlexData::make_bool(_audio));
  }
  if (_set_duration_head) {
    o.insert_or_assign("duration_head", FlexData::make_bool(_duration_head));
  }
  // As with every adapter in this tree: emitted only when named. An
  // empty `lora` would read as "the graph asked for no adapter", which
  // is indistinguishable from "the graph said nothing" -- and the
  // consumer's default is already no adapter.
  if (!_lora.empty()) {
    o.insert_or_assign("lora", FlexData::make_string(_lora));
    o.insert_or_assign("lora_scale", FlexData::make_real(_lora_scale));
  }
  return fd;
}

void
Ltx25ModelConfigStage::report_config(const FlexData& fd) const
{
  if (session() == nullptr) { return; }
  // Name what was SET, not what the object happens to contain -- an
  // almost-empty beat is the correct and common case here, and a log
  // line listing defaults would suggest otherwise.
  session()->info(fmt(
      "Ltx25ModelConfigStage('{}'): {} -> {}", this->id(),
      _variant.empty() ? std::string("variant as shipped (distilled)")
                       : fmt("variant '{}'", _variant)(),
      fd.to_json()));
}

}  // namespace ltx25
