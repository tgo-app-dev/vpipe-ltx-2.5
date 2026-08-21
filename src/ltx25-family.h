#ifndef VPIPE_LTX25_FAMILY_H
#define VPIPE_LTX25_FAMILY_H

#include "generative-models/video-model-registry.h"

#include <memory>
#include <string>
#include <vector>

namespace ltx25 {

// The LTX-2.5 family, as `generate-video` sees it.
//
// The family is process-wide and stateless: it answers "is this
// checkpoint mine", "what frame counts can it do", and "what will
// loading it cost" without touching a weight. Only `load` builds
// anything, and what it builds -- a VideoGenerator -- owns the whole
// denoise loop for one resident checkpoint.
class Ltx25Family : public vpipe::genai::VideoModelFamily {
public:
  std::string_view tag() const noexcept override;

  // Cheap and SURE, per the registry contract. This resolves the DiT
  // component and reads its `__metadata__` -- one JSON parse off a
  // header, no weights -- and refuses anything whose `_class_name` is
  // not AVTransformer3DModel. Claiming someone else's checkpoint here
  // would shadow a working built-in path, so the refusal is the point.
  bool claims(const std::string& root,
              const std::string& model_type) const override;

  // frames % 8 == 1, rounded up.
  int align_frames(const std::string& root, int frames) const override;
  void size_grid(const std::string& root, int* gh, int* gw) const override;

  std::vector<vpipe::ResourceClaim>
  declare_resources(const std::string& root) const override;

  // The same checkpoint in the topological plan's terms -- see
  // docs/MODEL-MEMORY.md, "Which ledger do I use?". One holding, the
  // DiT, with the floor its 48 streamed blocks can be reduced to.
  std::vector<vpipe::StageHolding>
  declare_holdings(const std::string& root) const override;

  // The beat-shaped terms nothing else can size: this family's latent
  // shape and its soundtrack. Both are LTX's own geometry, and a host
  // that substituted a built-in's formula would report a confident
  // number for the wrong model.
  std::size_t latent_bytes(const std::string& root, int width, int height,
                           int frames) const override;
  bool audio_cost(const std::string& root, int frames, double fps,
                  std::size_t* latent, std::size_t* pcm,
                  std::size_t* arena) const override;

  std::unique_ptr<vpipe::genai::VideoGenerator>
  load(const vpipe::genai::VideoModelCreateArgs& args) override;
};

}  // namespace ltx25

#endif
