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

  std::vector<vpipe::ResourceClaim>
  declare_resources(const std::string& root) const override;

  std::unique_ptr<vpipe::genai::VideoGenerator>
  load(const vpipe::genai::VideoModelCreateArgs& args) override;
};

}  // namespace ltx25

#endif
