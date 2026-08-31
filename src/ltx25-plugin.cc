// vpipe-ltx-2.5 -- the LTX-2.5 (Lightricks) joint audio-video model as a
// vpipe PLUGIN.
//
// The three extension points used here, and why each one:
//
//   register_video_family     LTX-2.5 is a VIDEO family, so it plugs into
//                             `generate-video` -- the same ports, the same
//                             conditioning, the same latent-out split. The
//                             family owns its whole denoise loop.
//   register_stage            `ltx-2.5-conditioner` (the caption -> the DiT's
//                             two contexts) and `ltx-2.5-model-config`, this
//                             family's own
//                             knobs. They do not go on generate-video:
//                             none of them mean anything to Wan or
//                             MiniMax-H3, and a key that applies to one
//                             family of three is inert -- silently -- on
//                             the other two.
//   register_catalog_entries  so the checkpoint is discoverable and
//                             fetchable the way a built-in model is.
//
// The model is under the LTX-2.x Community License, which is not vpipe's
// Apache-2.0 -- which is the point of shipping it as a separate binary.

#include "plugin/plugin-abi.h"
#include "plugin/plugin-context.h"

#include "ltx25-catalog.h"
#include "ltx25-config.h"
#include "ltx25-family.h"
#include "ltx25-conditioner-stage.h"
#include "ltx25-upscale-stage.h"
#include "ltx25-model-config-stage.h"
#include "ltx25-quant-family.h"
#include "ltx25-vae-family.h"

#include <memory>

// Emitted by vpipe_add_metal_library (see CMakeLists.txt).
extern "C" const unsigned char ltx25_kernels_bf16_metallib[];
extern "C" const unsigned long ltx25_kernels_bf16_metallib_len;
extern "C" const unsigned char ltx25_kernels_f16_metallib[];
extern "C" const unsigned long ltx25_kernels_f16_metallib_len;
extern "C" const unsigned char ltx25_kernels_f32_metallib[];
extern "C" const unsigned long ltx25_kernels_f32_metallib_len;

static const VpipePluginInfo kInfo = {
    VPIPE_PLUGIN_INFO_SCHEMA,
    "ltx-2.5",
    "0.1.0",
    "T-Go LLC",
    "Apache-2.0 (plugin); the LTX-2.5 WEIGHTS are under the LTX-2.x "
    "Community License -- see https://github.com/Lightricks/LTX-2",
    "LTX-2.5: 22B joint audio-video generation for generate-video",
};

static void
ltx25_register(vpipe::VpipePluginContext* ctx)
{
  if (ctx == nullptr) { return; }

  // The family FIRST: generate-video consults the registry before its
  // built-in wan / minimax-h3 probes, so this is what makes an LTX-2.5
  // checkpoint resolve at all.
  ctx->register_video_family(std::make_unique<ltx25::Ltx25Family>());

  // The family's own config source. Registered through the plugin path
  // rather than VPIPE_REGISTER_STAGE so the StageSpec is attached and
  // the stage shows up in /api/stage-types and the web-ui composer.
  ctx->register_stage<ltx25::Ltx25ModelConfigStage>(
      ltx25::Ltx25ModelConfigStage::stage_spec());

  // The family's own conditioner. LTX's encoder is Gemma-4 12B with all
  // 49 hidden states tapped and its own dual projection, and it emits
  // TWO contexts -- neither of which the host `diffusion-conditioner`
  // has a shape for. The audio one is what generate-video needs to
  // produce a soundtrack at all.
  ctx->register_stage<ltx25::Ltx25ConditionerStage>(
      ltx25::Ltx25ConditionerStage::stage_spec());

  // The LATENT upscalers. A STAGE rather than a registry entry, because
  // unlike the VAE there is no host stage to join: nothing in the tree
  // upscales a latent, so this is a new operation rather than a new
  // backend for an existing one.

  // The LATENT upscalers. A STAGE rather than a registry entry, because
  // unlike the VAE there is no host stage to join: nothing in the tree
  // upscales a latent, so this is a new operation rather than a new
  // backend for an existing one.
  ctx->register_stage<ltx25::Ltx25UpscaleStage>(
      ltx25::Ltx25UpscaleStage::stage_spec());

  // ...and its VAE, through the registry rather than a stage of its own.
  // `vae-decode` picks its built-in decoder from a hardcoded
  // `_class_name` chain that an out-of-tree family cannot join;
  // register_vae_family is the seam for exactly that. A plugin adding a
  // video model registers BOTH families -- one makes the latent, the
  // other decodes it -- and needs no stage for either.
  ctx->register_vae_family(std::make_unique<ltx25::Ltx25VaeFamily>());

  // register_quantize_family is the seam that lets `model-quantize`
  // package this checkpoint as ONE self-contained model -- a quantized
  // DiT, a quantized text encoder and the VAEs untouched, in one
  // directory -- instead of leaving packs scattered through the fetched
  // repo. The family supplies only where its components live and which
  // of their tensors are matrices; see ltx25-quant-family.h.
  ctx->register_quantize_family(std::make_unique<ltx25::Ltx25QuantFamily>());

  // The kernels, before anything can dispatch them. Two dtype twins
  // under the names the model layer resolves with load_library(); the
  // DiT runs the bf16 one. Registration is first-wins and cannot shadow
  // a built-in name.
  ctx->register_metal_library(ltx25::kMetalLibBf16,
                              ltx25_kernels_bf16_metallib,
                              ltx25_kernels_bf16_metallib_len);
  ctx->register_metal_library(ltx25::kMetalLibF16,
                              ltx25_kernels_f16_metallib,
                              ltx25_kernels_f16_metallib_len);
  // The f32 twin is NOT optional: the vocoder and the BWE resolve it by
  // name, and the reference forces fp32 through their 108 sequential
  // convolutions (bf16 costs 40-90% of the spectral metrics). Without
  // this the audio path loads fine everywhere except a real pipeline.
  ctx->register_metal_library(ltx25::kMetalLibF32,
                              ltx25_kernels_f32_metallib,
                              ltx25_kernels_f32_metallib_len);

  ctx->register_catalog_entries(ltx25::catalog_entries());
}

VPIPE_PLUGIN_DEFINE(&kInfo, ltx25_register)
