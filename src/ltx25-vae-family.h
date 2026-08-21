#ifndef VPIPE_LTX25_VAE_FAMILY_H
#define VPIPE_LTX25_VAE_FAMILY_H

#include "generative-models/vae-model-registry.h"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace ltx25 {

// LTX-2.5's conv video VAE, as the stock `vae-decode` stage sees it.
//
// The host stage detects its built-in families from the VAE config's
// `_class_name` through a hardcoded chain, which an out-of-tree family
// cannot join. `register_vae_family` is the seam for that, and this is
// the counterpart to `Ltx25Family` on the generation side: one family
// makes the latent, the other turns it into frames, and the plugin
// needs a stage of its own for neither.
//
// Everything here wraps `Ltx25VaeDecoder` (ltx25-vae.h), which is
// verified at 3.8e-3 against the reference -- bf16's own precision -- so
// this file is adaptation, not a second reading of the model.
//
// ---- WHAT IT ANSWERS, AND THE TWO THINGS THAT ARE NOT THE DEFAULT ----
//
//  * `claims` is given BOTH the root and `resolve_vae_dir(root)`, and it
//    uses the ROOT: LTX keeps its VAE as `vae/*video-vae-conv*.
//    safetensors` with the config in the file's `__metadata__`, so there
//    is no `vae/config.json` for resolve_vae_dir to find and it hands
//    back the root unchanged.
//  * `idle_peers` names `diffusion_models/` and `text_encoders/` -- the
//    COMFY spelling. The stage's own guess is the diffusers one
//    (`transformer/`, `text_encoder/`), which sums to zero here, and a
//    zero peer footprint reads as "the box is roomy, keep the VAE
//    resident" beside a 39 GB DiT.
class Ltx25VaeFamily : public vpipe::genai::VaeModelFamily {
public:
  std::string_view tag() const noexcept override;

  bool claims(const std::string& root, const std::string& vae_dir,
              const std::string& model_type) const override;

  std::vector<vpipe::ResourceClaim>
  declare_resources(const std::string& root,
                    const std::string& vae_dir) const override;

  // WHERE each half of this family's VAE actually lives. LTX-2.5 ships
  // two files under `vae/`, and the host cannot find either: there is no
  // `vae/config.json`, so its resolver returns the ROOT -- and the root
  // is the whole 142 GB repository. Every release, pool and
  // phase-release the VAE stages perform is keyed on the name they get
  // from here.
  std::string vae_path(const std::string& root, Role role) const override;

  std::vector<vpipe::StageHolding>
  declare_holdings(const std::string& root, Role role) const override;

  std::vector<std::string> idle_peers(const std::string& root) const override;

  std::unique_ptr<vpipe::genai::VaeDecoder>
  load_decoder(const vpipe::genai::VaeModelCreateArgs& args) override;

  // The ENCODER half, for `vae-encode` -- what makes the image and
  // first/last-frame reference paths producible at all. Nothing else in
  // the vpipe tree can emit an LTX latent: every built-in family the
  // stage knows is 8x or 16x at 16 or 32 channels, and this VAE is 32x
  // at 128.
  //
  // ONE IMAGE per beat, because that is what the stage supplies. A
  // multi-frame video reference would need the stage to gather frames,
  // which it does not do for any family.
  std::unique_ptr<vpipe::genai::VaeEncoder>
  load_encoder(const vpipe::genai::VaeModelCreateArgs& args) override;

  // The soundtrack: the audio VAE decoder (latent -> log-mel), the
  // BigVGAN vocoder (mel -> 16 kHz) and the BWE (16 -> 48 kHz), each
  // verified separately. Chained here so `audio-vae-decode` gets PCM
  // without knowing any of it.
  std::unique_ptr<vpipe::genai::AudioVaeDecoder>
  load_audio_decoder(const vpipe::genai::VaeModelCreateArgs& args) override;

  // The reference SOUNDTRACK, for `audio-vae-encode`: the log-mel front
  // end (its own knobs, none of them in the checkpoint) chained onto the
  // audio VAE's encoder half. Emits the [rows, z * mel] form
  // generate-video's `ref_audio_rows` takes, so nothing between here and
  // the DiT has to know the latent was ever 3-D.
  std::unique_ptr<vpipe::genai::AudioVaeEncoder>
  load_audio_encoder(const vpipe::genai::VaeModelCreateArgs& args) override;
};

}  // namespace ltx25

#endif
