#include "ltx25-catalog.h"
#include "ltx25-config.h"

#include <vector>

using vpipe::ModelCatalogEntry;

namespace ltx25 {

// The LTX-2.5 repo is GATED on HuggingFace: a fetch needs an accepted
// licence and a token in the session's credential store. The entries are
// still worth publishing -- the drill-down is how a user discovers the
// model exists and which files it needs, and a 403 with a licence URL is
// a better answer than a model that is simply absent.
//
// The pack is SPLIT, so each entry pins exactly the files that component
// set needs (`files`). Fetching the whole repo would pull 174 GB,
// including the int8-convrot and nvfp4 packings this build deliberately
// cannot read.
std::vector<ModelCatalogEntry>
catalog_entries()
{
  std::vector<ModelCatalogEntry> v;

  // The distilled stack: what a t2v / i2v graph actually runs. One
  // entry, not four, because these files are useless apart -- a DiT with
  // no text encoder is a directory that cannot encode a prompt, which is
  // a fetch that reports success and produces something unusable.
  {
    ModelCatalogEntry e;
    e.family      = "LTX";
    e.version     = "2.5";
    e.param_class = "22B";
    e.variant     = "distilled bf16 (video+audio, 8-step)";
    e.hf_path     = "Lightricks/LTX-2.5";
    e.model_type  = kFamily;
    e.name        = "LTX-2.5-distilled";
    e.inputs      = {"text", "image", "audio", "video"};
    e.outputs     = {"video", "audio"};
    e.weight_format = "comfyui";
    e.files = {
        "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
        "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
        "vae/ltx-2.5-audio-vae-bf16.safetensors",
        "model_patches/ltx-2.5-duration-head-bf16.safetensors",
    };
    v.push_back(std::move(e));
  }

  // The dev (trainable) DiT, alongside the same peers. A separate entry
  // rather than a `files` superset because the two DiTs are 39 GB each
  // and a user choosing one should not fetch both.
  {
    ModelCatalogEntry e;
    e.family      = "LTX";
    e.version     = "2.5";
    e.param_class = "22B";
    e.variant     = "dev bf16 (video+audio, trainable, real CFG)";
    e.hf_path     = "Lightricks/LTX-2.5";
    e.model_type  = kFamily;
    e.name        = "LTX-2.5-dev";
    e.inputs      = {"text", "image", "audio", "video"};
    e.outputs     = {"video", "audio"};
    e.weight_format = "comfyui";
    e.files = {
        "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors",
        "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
        "vae/ltx-2.5-audio-vae-bf16.safetensors",
        "model_patches/ltx-2.5-duration-head-bf16.safetensors",
    };
    v.push_back(std::move(e));
  }

  // The distilled LoRA, as a SUPPLEMENT: it attaches to the dev DiT
  // (`parent_model_type`), which is what puts it in the
  // `ltx-2.5-model-config` stage's `lora` picker and keeps a Wan or
  // Krea-2 adapter out of it.
  {
    ModelCatalogEntry e;
    e.family      = "LTX";
    e.version     = "2.5";
    e.param_class = "22B";
    e.variant     = "distilled LoRA 450 (for the dev DiT)";
    e.hf_path     = "Lightricks/LTX-2.5";
    e.model_type  = "ltx-2.5-lora";
    e.name        = "LTX-2.5-distilled-lora-450";
    e.parent_model_type = kFamily;
    e.parent_param_class = "22B";
    e.weight_format = "comfyui";
    e.files = {"loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors"};
    v.push_back(std::move(e));
  }

  // The 2x pixel spatial upscaler, an IC-LoRA -- a DIFFERENT KIND of
  // adapter from the distilled LoRA above, and the difference is not
  // in the weights.
  //
  // A plain LoRA changes what the model does with the context it
  // already had. An in-context LoRA changes the CONTEXT: the reference
  // clip rides as extra tokens alongside the noisy latents, and the
  // adapter is trained on a model that can see them. Fused into a graph
  // that passes no reference it does not fail -- it renders, ignoring
  // the clip, which is why the generator refuses that combination
  // rather than letting it produce a plausible wrong answer.
  //
  // `reference_downscale_factor` lives in the checkpoint's safetensors
  // `__metadata__` (2 here) rather than in this table, because it is the
  // ADAPTER's property and a later variant will carry its own. The
  // conditioner reads it from the file.
  //
  // It is a separate ENTRY rather than a `files` addition to the pack
  // above for the same reason the dev and distilled DiTs are separate:
  // a user who wants one should not fetch the other. This one is
  // 327 MB against the DiT's 39 GB.
  //
  // GATED on HuggingFace -- an unauthenticated fetch 401s -- and
  // PUBLIC on ModelScope, so `source=modelscope` reaches it without a
  // token where the HuggingFace path needs an accepted licence.
  {
    ModelCatalogEntry e;
    e.family      = "LTX";
    e.version     = "2.5";
    e.param_class = "22B";
    e.variant     = "IC-LoRA pixel spatial upscaler x2 (video-to-video)";
    e.hf_path     = "Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler";
    e.model_type  = "ltx-2.5-lora";
    e.name        = "LTX-2.5-ic-lora-pixel-spatial-upscaler-x2";
    e.inputs      = {"text", "video"};
    e.outputs     = {"video"};
    e.parent_model_type = kFamily;
    e.parent_param_class = "22B";
    e.weight_format = "comfyui";
    e.files = {
        "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors"};
    v.push_back(std::move(e));
  }

  // The x2 latent upscalers, for the two-stage pipeline. Catalogued
  // separately because stage 2 is optional and they are useless without
  // a stage-1 latent.
  {
    ModelCatalogEntry e;
    e.family      = "LTX";
    e.version     = "2.5";
    e.param_class = "upscalers";
    e.variant     = "latent x2 spatial + temporal (bf16)";
    e.hf_path     = "Lightricks/LTX-2.5";
    e.model_type  = "ltx-2.5-upscaler";
    e.name        = "LTX-2.5-latent-upscalers";
    e.parent_model_type = kFamily;
    e.weight_format = "comfyui";
    e.files = {
        "latent_upscale_models/"
        "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
        "latent_upscale_models/"
        "ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors",
    };
    v.push_back(std::move(e));
  }

  return v;
}

}  // namespace ltx25
