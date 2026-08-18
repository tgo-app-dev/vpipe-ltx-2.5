#ifndef VPIPE_LTX25_GENERATOR_H
#define VPIPE_LTX25_GENERATOR_H

#include "ltx25-config.h"
#include "ltx25-dit.h"
#include "ltx25-metal-ops.h"

#include "generative-models/video-model-registry.h"
#include "generative-models/weight-set.h"

#include <memory>
#include <string>
#include <vector>

namespace ltx25 {

// The knobs `ltx-2.5-model-config` emits, parsed.
//
// PARSED IN THE MODEL LAYER, not in the stage -- which is the contract
// stages/model-config-source.h states and the reason `generate-video`
// passes the beat down unread. A knob added to the config stage later
// needs a field here and nothing in any stage.
//
// Every field starts at LTX-2.5's own default, and `from_flex` only
// overwrites what the beat actually carries. A source that emitted its
// defaults would be indistinguishable from a graph that chose them.
struct GenerationParams {
  std::string variant;          // "" | "distilled" | "dev"
  double guidance = 0.0;        // 0 = unset -> the reference default
  double audio_guidance = 0.0;
  double stg = 0.0;
  double audio_stg = 0.0;
  double modality = 0.0;
  double audio_modality = 0.0;
  double audio_seconds = 0.0;   // 0 = derive from frames / fps
  // How hard the REFERENCES are held. 1.0 (the reference's own default
  // for an image anchor) keeps the given content exactly; lower values
  // let the model denoise it away, which is how a "loose" i2v that
  // reinterprets the opening frame is asked for. Each kind gets its own
  // knob because a run commonly wants a hard first frame and a soft
  // closing one.
  double ref_strength       = 1.0;
  double ref_last_strength  = 1.0;
  double ref_audio_strength = 1.0;
  bool   audio = true;
  bool   duration_head = false;
  std::string lora;
  double lora_scale = 1.0;

  // Never throws and never half-applies: a malformed value leaves its
  // field at the default and is named in `err`.
  static GenerationParams from_flex(const vpipe::FlexData& fd,
                                    std::string* err = nullptr);
};

// LTX-2.5 as a `generate-video` family member.
//
// It owns the whole generation -- the schedule, the noise, the denoise
// loop -- which is exactly the seam VideoModelFamily draws (see
// generative-models/video-model-registry.h). What it does NOT own is
// pixels: `generate-video` emits LATENTS and `vae-decode` makes frames,
// so this is useful before the VAE is ported.
//
// WHAT IT NEEDS FROM THE GRAPH, and what it refuses without:
//
//   iport0  the caption PROJECTED to `cross_attention_dim` (4096) --
//           the text encoder's `video_aggregate_embed` output. This
//           runs the DiT's own connector over it
//           (`caption_proj_before_connector`), so the conditioning
//           handed in is PRE-connector.
//
// The AUDIO stream needs a context of its own at
// `audio_cross_attention_dim` (2048), from the encoder's separate
// `audio_aggregate_embed`. No conditioner in this tree produces one
// yet, so a request without it runs VIDEO-ONLY and says so -- the block
// supports that (the reference's `run_a2v` is conditional on audio
// being present), and inventing a zero audio context would be
// fabricating conditioning rather than declining it.
class Ltx25Generator : public vpipe::genai::VideoGenerator {
public:
  // `pin_frac` comes from model_memory::plan_streaming and is passed
  // straight to Ltx25Dit::load -- see the note there. Held as a member
  // because the DiT is rebuilt on the reload path and the fraction has to
  // be the same one, not a fresh guess taken against a different graph.
  static std::unique_ptr<Ltx25Generator>
  create(const Config& cfg, std::shared_ptr<vpipe::genai::WeightSet> ws,
         vpipe::metal_compute::MetalCompute* mc, bool stream_blocks,
         double pin_frac, int plan_w, int plan_h, int plan_frames,
         const vpipe::SessionContextIntf* session, std::string* err);

  int latent_channels() const override { return _cfg.dit.in_channels; }
  int spatial_compression() const override { return kSpatialCompression; }

  bool generate(const vpipe::genai::VideoGenRequest& req,
                vpipe::genai::VideoGenResult* out) override;

  void release_idle() override;
  std::uint64_t resident_bytes() const override { return _resident; }

private:
  Ltx25Generator() = default;

  void log_(const std::string& m) const;
  void warn_(const std::string& m) const;

  Config _cfg;
  std::shared_ptr<vpipe::genai::WeightSet> _ws;
  MetalOps _ops;
  std::unique_ptr<Ltx25Dit> _dit;
  const vpipe::SessionContextIntf* _session = nullptr;
  std::uint64_t _resident = 0;
  bool _stream_blocks = false;
  // The plan's pinned-prefix fraction, kept for the reload path.
  double _pin_frac = 0.0;
  // The clip the graph planned, for the reload path's pin sizing.
  int _plan_w = 0, _plan_h = 0, _plan_frames = 0;

public:
  // Streaming state, for the family's log line and its declaration.
  bool streaming_blocks() const noexcept;
  int  pinned_blocks() const noexcept;
  std::size_t pinned_weight_bytes() const noexcept;

private:
  // The geometry the DiT is currently sized for; a request that changes
  // it re-sizes rather than silently running the old one. `_v_tokens` /
  // `_a_tokens` are the TOTALS, so a request that only changes how much
  // conditioning it carries still re-sizes.
  int _lf = 0, _lh = 0, _lw = 0, _at = 0, _tt = 0;
  int _v_tokens = 0, _a_tokens = 0;
  // The denoise levels the geometry was built for, and the ones the
  // adaLN bake actually covers. They differ only across a request that
  // changed a reference's strength, which is what forces a rebuild --
  // the bake released the projections that would recompute the table.
  std::vector<double> _v_levels{1.0}, _a_levels{1.0};
  std::vector<double> _v_baked, _a_baked;
};

}  // namespace ltx25

#endif
