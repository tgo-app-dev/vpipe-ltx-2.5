#ifndef VPIPE_LTX25_UPSCALE_STAGE_H
#define VPIPE_LTX25_UPSCALE_STAGE_H

#include "ltx25-metal-ops.h"
#include "ltx25-upscaler.h"

#include "common/flex-data.h"
#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/stage-spec.h"
#include "pipeline/typed-stage.h"

#include <memory>
#include <string>
#include <vector>

namespace ltx25 {

// LTX-2.5's LATENT upscalers as a stage: a latent in, a bigger latent
// out, with no VAE round trip between them.
//
// WHY THIS IS A STAGE AND THE VAE IS NOT. `vae-decode` and `vae-encode`
// reach an out-of-tree family through register_vae_family, because the
// host already has stages for those jobs and only needed the model
// swapped. Nothing in the host upscales a LATENT, so there is no stage
// to join -- this is a new operation, not a new backend for an old one.
//
// WHAT IT IS FOR. The two-stage pipeline: generate at a small geometry,
// upscale the latent, denoise again at the larger one. Doing it in
// latent space is the point -- decoding to pixels and re-encoding costs
// two VAE passes and loses whatever the round trip loses.
//
// THE STATISTICS COME FROM THE VAE, NOT FROM THIS CHECKPOINT. The
// upscaler works in the VAE's un-normalized latent space while
// generate-video emits a whitened one, so the stage reads
// `per_channel_statistics.*` out of <hf_dir>/vae and un-whitens either
// side. An upscaler pointed at a directory with no VAE would run the
// model on the wrong scale and return a plausible latent, so a missing
// VAE is an error rather than an identity fallback.
class Ltx25UpscaleStage final
  : public vpipe::TypedStage<Ltx25UpscaleStage> {
public:
  static constexpr const char* kTypeName = "ltx-2.5-latent-upscale";

  Ltx25UpscaleStage(const vpipe::SessionContextIntf* session,
                    std::string                      id,
                    std::vector<vpipe::InEdge>       iports,
                    vpipe::FlexData                  config);
  ~Ltx25UpscaleStage() override;

  const vpipe::StageSpec& spec() const noexcept override;
  static const vpipe::StageSpec* stage_spec() noexcept;

  vpipe::Job process(vpipe::RuntimeContext& ctx) override;

  std::vector<vpipe::ResourceClaim> declare_resources() const override;

private:
  bool ensure_loaded_();
  std::string model_root_() const;
  // <root>/latent_upscale_models/<the file for `mode`>, or empty with a
  // message naming what it looked for.
  std::string checkpoint_(std::string* err) const;

  std::string _hf_dir;
  std::string _mode = "spatial";
  bool        _unload_idle = false;
  bool        _tried = false;

  std::unique_ptr<MetalOps>       _ops;
  std::unique_ptr<Ltx25Upscaler>  _model;
  UpscalerConfig                  _cfg;
  unsigned                        _emitted = 0;
};

}  // namespace ltx25

#endif
