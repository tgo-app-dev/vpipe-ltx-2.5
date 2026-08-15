#ifndef VPIPE_LTX25_DIT_H
#define VPIPE_LTX25_DIT_H

#include "ltx25-block-metal.h"
#include "ltx25-conditioning.h"
#include "ltx25-config.h"
#include "ltx25-connector.h"
#include "ltx25-dit-weights.h"
#include "ltx25-metal-ops.h"
#include "ltx25-rope.h"

#include "generative-models/weight-set.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace ltx25 {

// The 48-block AVTransformer3DModel: one velocity prediction.
//
// WHAT THIS OWNS. The trunk (six adaLN MLPs, two patchify projections,
// two output heads), the block stack, and the RoPE tables for one
// geometry. It does NOT own the scheduler, the guidance, or the
// latents -- those belong to the family's denoise loop, so the same DiT
// serves the distilled 8-step schedule and dev's 40-step CFG without
// knowing which is running.
//
// THE CONNECTORS. The two `*_embeddings_connector` perceivers live in
// this checkpoint (8 layers, 128 learnable registers, per stream) and
// resample the text encoder's projected caption into the
// cross-attention context. They are OPTIONAL here, and the choice is
// the caller's for a memory reason: the video connector is ~3.2 GB on
// top of the DiT's 39 GB, and a pipeline that connects once and then
// denoises for 8 steps should not hold it for all of them.
//
//   load(..., with_connectors = true)   forward takes the RAW projected
//                                       caption and connects it
//   load(..., with_connectors = false)  forward takes it ALREADY
//                                       connected (the default, and what
//                                       a two-stage graph wants)
class Ltx25Dit {
public:
  // Load the trunk and bind the blocks.
  //
  // `stream_blocks` binds each block through the weight set's
  // non-retaining path: the 48 blocks ARE the 39 GB, so a box that
  // cannot hold them runs them this way and the manager sees the
  // traffic. The trunk is always cached -- it is a few hundred MB and
  // every step reads it.
  static std::unique_ptr<Ltx25Dit>
  load(const Config& cfg, vpipe::genai::WeightSet& ws, const MetalOps& ops,
       bool stream_blocks, std::string* err, bool with_connectors = false);

  // Fix the geometry: build the four RoPE tables and size the scratch.
  // Separate from load() because the tables depend on the REQUEST
  // (frames x height x width) while the weights do not, and a 48-block
  // stack should not rebuild them per block.
  //
  // `geo` carries the VAE scale factors and the clip's fps, which is what
  // turns latent cells into the PIXEL and SECOND coordinates the model
  // was trained on. It has no default ON PURPOSE: defaulting it is how
  // this port spent its first life feeding raw latent indices, which
  // collapses the spatial RoPE into a band 32x too narrow and produces
  // structureless output that no golden caught. See RopeGeometry.
  //
  // `cond` adds the conditioning: extra denoise levels, and extra tokens
  // appended after each stream's target grid. Defaulted, so an
  // unconditioned caller says nothing and gets exactly the geometry it
  // got before conditioning existed.
  bool set_geometry(int latent_frames, int latent_h, int latent_w,
                    int audio_tokens, int text_tokens,
                    const RopeGeometry& geo, std::string* err,
                    const Conditioning& cond = Conditioning{});

  // Token counts for the geometry now set. The TARGET counts are what a
  // caller slices its output down to; the totals include whatever the
  // conditioning appended.
  int video_tokens() const noexcept { return _video_tokens; }
  int audio_tokens() const noexcept { return _audio_tokens; }
  int video_target_tokens() const noexcept { return _v_target; }
  int audio_target_tokens() const noexcept { return _a_target; }

  // Precompute every adaLN modulation the SCHEDULE will ask for, so the
  // per-step host GEMV chains stop running.
  //
  // WHAT THIS IS WORTH, and it is NOT what MiniMax-H3's bake is worth.
  // H3 puts an adaLN projection in each of its 50 BLOCKS: 12.91 GB of a
  // 23.5 GB checkpoint (54.9%), re-read from disk every step under
  // streaming, so baking there removes half the streamed model. LTX-2.5
  // puts its adaLN at the MODEL level -- computed once per step and
  // shared by all 48 blocks. MEASURED off the shard headers: per-block
  // adaLN 0.019 GB (0.0%), model-level chains 0.852 GB (2.0%) of 42.02
  // GB. There is no 55% here and it would be wrong to claim one.
  //
  // NOR IS IT A MEMORY WIN. The chains are bound through
  // `WeightSet::tensor(..., Residency::Mapped)`, so they are views into
  // shard pages the DiT maps anyway and the set keeps its own cached
  // alias. Dropping this model's handles frees no RSS, and this class
  // deliberately does not pretend otherwise -- see
  // `adaln_bytes_per_step()`, which reports work avoided, not bytes
  // freed.
  //
  // What IS real: `adaln_()` runs its GEMV chain ON THE HOST, eight
  // times per forward, TOUCHING those 852 MB of demand-paged weights
  // every step. In a streaming run -- the only way this 65 GB stack
  // fits -- those pages compete with the streamed blocks for page
  // cache, so re-reading them per step is I/O, not just arithmetic. The
  // bake replaces it with a few kB of table per step.
  //
  // Requires the whole schedule up front, which the distilled checkpoint
  // has by construction. After this, forward() selects on `Input::step`
  // and IGNORES `sigma` for modulation -- so a request that does not
  // name a baked step is REFUSED rather than modulated for some other
  // noise level. The projections are cleared on the way out purely to
  // make that enforceable: a stale reader crashes instead of quietly
  // working.
  bool bake_adaln(const std::vector<double>& sigmas, std::string* err);
  bool adaln_baked() const noexcept { return _baked; }

  // 0 when the blocks are dense bf16; 32 or 64 for a quantized pack.
  int quant_group() const noexcept { return _quant_group; }

  // Weight bytes the per-step host chains no longer TOUCH. This is not
  // memory freed -- see above. 0 before the bake runs.
  std::uint64_t adaln_bytes_per_step() const noexcept { return _baked_freed; }

  // Bytes the block stack's shared scratch arena holds. This IS real
  // memory, and it is the term that scales with the clip: it used to be
  // multiplied by the 48 blocks.
  std::uint64_t scratch_bytes() const noexcept
  {
    return _scratch ? _scratch->bytes() : 0;
  }

  // One forward.
  struct Input {
    // The video latent, f32, CHANNEL-major [z][tokens] -- as a VAE
    // encode produces it (its [F][H][W] flattens w-fastest) and as the
    // sampler holds it. `tokens` is video_tokens(), so a conditioned
    // generation's APPENDED tokens are columns on the end of the same
    // array, carrying the content they were given.
    const float* video = nullptr;
    // The audio latent, f32 [z][audio_tokens()]. Null runs video-only.
    const float* audio = nullptr;
    // The cross-attention context, bf16:
    // [text_tokens][cross_attention_dim] for video and
    // [text_tokens][audio_cross_attention_dim] for audio.
    //
    // Already through the connector unless the DiT was loaded WITH
    // connectors, in which case this is the raw projected caption and
    // `n_valid_text` says how much of it is real.
    const void* context = nullptr;
    const void* audio_context = nullptr;
    // The caption's true length; positions at or past it are replaced by
    // the connector's learnable registers. Ignored when the DiT has no
    // connectors. 0 means "all of it is real".
    int n_valid_text = 0;
    // Noise levels in the model's own units (1 = pure noise). Both are
    // needed even for a video-only forward: the a2v gate is driven by
    // the AUDIO sigma, which is what makes the coupling depend on how
    // noisy the other modality currently is.
    double sigma = 1.0;
    double audio_sigma = 1.0;

    // Which schedule step this is. Only meaningful once bake_adaln()
    // has run; -1 means "not baked / compute it now". A baked DiT given
    // step < 0 falls back to computing, which is correct but has just
    // released the weights it would need -- so it refuses instead.
    int step = -1;
    // Called between blocks; false aborts. A 48-block forward on a 22B
    // model is minutes, so a stack that cannot be interrupted is a Stop
    // that looks like a hang.
    std::function<bool(int block, int total)> progress;
  };

  struct Output {
    // The velocity, [z][video_tokens()] -- the WHOLE sequence, appended
    // conditioning tokens included. Their velocity is meaningless (the
    // sampler holds them at their given content) and the caller drops
    // it; returning a short array instead would make the two halves of
    // one buffer disagree on length.
    std::vector<float> video;
    std::vector<float> audio;        // [z][audio_tokens()]
  };

  bool forward(const Input& in, Output* out, std::string* err);

  int num_layers() const { return (int)_blocks.size(); }
  const DitTrunk& trunk() const { return _trunk; }

  // Exposed for the trunk test: each is a piece the 48-block loop
  // composes and that nothing else checks. `adaln` is the whole chain
  // (sinusoid -> 3 linears), which is where flip_sin_to_cos and the
  // three-linear shape can be silently wrong.
  std::vector<float> adaln_public(const DitTrunk::AdaLN& a, double timestep,
                                  std::vector<float>* embedded) const
  {
    return adaln_(a, timestep, embedded);
  }

  const DitConfig& config() const { return _cfg; }
  bool connector_weights_present() const { return _has_connector; }
  bool connectors_loaded() const { return _v_conn != nullptr; }

private:
  Ltx25Dit() = default;

  // The adaLN MLP chain, on the HOST in f32.
  //
  // It runs on ONE row -- a single timestep -- so the whole chain is
  // 256x4096 + 4096x4096 + 4096x36864 MACs, about 0.2 GFLOP against the
  // block stack's ~40 TFLOP. Doing it on the host in f32 costs nothing
  // measurable and keeps full precision in the value that drives every
  // modulation in every block; a bf16 GEMV here would round the
  // schedule itself.
  //
  // Returns the k*dim driver, and (through `embedded`) the dim-wide
  // embedder output the OUTPUT HEAD needs -- the reference returns both
  // from one call for exactly that reason.
  std::vector<float> adaln_(const DitTrunk::AdaLN& a, double timestep,
                            std::vector<float>* embedded) const;

  const MetalOps* _ops = nullptr;
  DitConfig _cfg;
  DitTrunk  _trunk;
  std::vector<std::unique_ptr<MetalBlock>> _blocks;
  bool _has_connector = false;
  bool _have_audio = false;

  // 32 / 64 when the blocks are group-affine quantized, 0 when dense.
  // Reported so the family can log which pack it loaded -- the two
  // differ by 3x in footprint and nothing else in the log says which.
  int  _quant_group = 0;

  // ---- the baked adaLN schedule ---------------------------------------
  // Per step: the eight modulation vectors plus the two `embedded`
  // vectors the output head wants. A few kB a step against 852 MB of
  // projections.
  struct BakedStep {
    // Per DENOISE LEVEL: the 9*dim driver and the dim-wide embedder
    // output the output head wants. One entry each for an unconditioned
    // generation.
    std::vector<std::vector<float>> v_ts, a_ts;
    std::vector<std::vector<float>> v_embedded, a_embedded;
    // Driven by the stream's SCALAR sigma, so not per token: the prompt
    // adaLN and the audio<->video scale/shift and gates. The reference
    // feeds `modality.sigma` to those, never `modality.timesteps`.
    std::vector<float> v_pts, a_pts;
    std::vector<float> v_css, a_css, v_cg, a_cg;
  };
  // Run the eight chains for one (sigma, audio_sigma) pair, the two
  // per-token ones once per denoise level.
  void compute_adaln_step_(double sigma, double audio_sigma,
                           BakedStep* b) const;

  std::vector<BakedStep> _baked_steps;
  bool          _baked = false;
  std::uint64_t _baked_freed = 0;
  // The levels the bake was taken at. A later request that changes them
  // is REFUSED rather than modulated from the old table: the
  // projections that would recompute it are gone, so there is nothing
  // to fall back to and a silent mismatch would modulate a conditioned
  // token at the wrong noise level.
  std::vector<double> _baked_v_levels, _baked_a_levels;

  // Geometry-dependent state.
  RopeTable _v_self, _a_self, _v_cross, _a_cross;
  int _frames = 0, _lh = 0, _lw = 0;
  int _video_tokens = 0, _audio_tokens = 0, _text_tokens = 0;
  // The TARGET grid's token counts. Equal to the totals above unless
  // conditioning appended tokens, whose velocity the caller discards.
  int _v_target = 0, _a_target = 0;
  bool _geometry_set = false;

  // The conditioning the geometry was built for. `_v_levels[0]` is
  // always 1.0; a single-entry vector is the unconditioned case and the
  // whole per-token path stays switched off.
  std::vector<double> _v_levels{1.0}, _a_levels{1.0};
  // Per-token level index, int32 on the GPU. Empty when there is one
  // level, and then never bound.
  vpipe::metal_compute::SharedBuffer _v_level_buf, _a_level_buf;

  // The block stack's ONE scratch arena, held here so it outlives a
  // re-geometry and so its size can be reported. Every block holds a
  // copy of this handle; see BlockScratch for why one is enough.
  std::shared_ptr<BlockScratch> _scratch;

  // Per-forward buffers, allocated once by set_geometry.
  vpipe::metal_compute::SharedBuffer _vx, _ax;          // the token streams
  vpipe::metal_compute::SharedBuffer _vlat, _alat;      // packed latents
  vpipe::metal_compute::SharedBuffer _vts, _ats;        // 9*dim drivers
  vpipe::metal_compute::SharedBuffer _v_css, _a_css;    // 4*dim
  vpipe::metal_compute::SharedBuffer _v_cg, _a_cg;      // dim
  vpipe::metal_compute::SharedBuffer _v_pts, _a_pts;    // 2*dim
  vpipe::metal_compute::SharedBuffer _v_out, _a_out;    // the head's output
  vpipe::metal_compute::SharedBuffer _v_head_ss, _a_head_ss;  // 2*dim
  vpipe::metal_compute::SharedBuffer _tmp;
  // The connected context, copied in per forward so the block stack has
  // a stable buffer to borrow.
  vpipe::metal_compute::SharedBuffer _vctx, _actx;
  // The raw caption, when this DiT connects it itself. Separate from
  // _vctx so the connector reads one buffer and writes the other.
  vpipe::metal_compute::SharedBuffer _vctx_raw, _actx_raw;
  std::unique_ptr<Ltx25Connector> _v_conn, _a_conn;
};

}  // namespace ltx25

#endif
