#ifndef VPIPE_LTX25_DIT_H
#define VPIPE_LTX25_DIT_H

#include "ltx25-block-metal.h"
#include "ltx25-conditioning.h"
#include "ltx25-config.h"
#include "ltx25-connector.h"
#include "ltx25-dit-weights.h"
#include "ltx25-metal-ops.h"
#include "ltx25-rope.h"

#include "generative-models/generative-model-manager.h"
#include "generative-models/shared/block-residency.h"
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
  // THERE IS NO PINNED-PREFIX FRACTION any more. A share of TOTAL ram
  // decided before the run is blind to another process, to this graph's
  // peers, and to the moment a peer lets go; what replaces it measures
  // (BlockResidency below). A streaming stack holds exactly ONE block --
  // the one that owns the scratch arena -- and grows from there.
  //
  // `plan_w/h/frames` is the clip the graph INTENDS to make, used to
  // size that arena at the right order of magnitude. 0 means the stage
  // could not settle one.
  static std::unique_ptr<Ltx25Dit>
  load(const Config& cfg, std::shared_ptr<vpipe::genai::WeightSet> ws,
       const MetalOps& ops, bool stream_blocks,
       int plan_w, int plan_h, int plan_frames, std::string* err,
       bool with_connectors = false);

  // Streaming state, for the stage's log line and its declaration.
  //
  // `pinned_blocks()` is how many LEADING blocks are held resident; the
  // rest are read per forward and freed. `pinned_bytes()` is what this
  // model actually keeps, which is what revise_declaration() must be
  // told -- a streaming model that goes on declaring the whole
  // checkpoint makes every peer size against weights that are not there.
  bool        streaming() const noexcept { return _stream_blocks; }
  int         pinned_blocks() const noexcept { return _pinned; }
  std::size_t pinned_bytes() const noexcept;

  // EVERYTHING this model holds in weights: the blocks it is keeping,
  // the trunk, and both connectors.
  //
  // pinned_bytes() answers a narrower question -- the blocks alone --
  // and the difference is not small: the trunk plus the two connectors
  // is several GB that a peer sizing itself has no other way to see.
  // This is what VideoGenerator::resident_bytes() must return, and
  // docs/MODEL-MEMORY.md calls the default 0 there the single most
  // consequential wrong answer a family can give.
  std::size_t held_weight_bytes() const noexcept;

  // ---- the WIRED POOL (mechanism 7) ---------------------------------
  //
  // Hand this model's pages to the manager's pool, where the OS cannot
  // compress or swap them. Rule one is that nothing goes to swap, and
  // wiring is the only mechanism that enforces it.
  //
  // ORDER IS THE POINT and it is the opposite of intuition: the trunk
  // and the scratch go in FIRST, the blocks after. A resident block is
  // an optimisation this model can shed and stream instead; the scratch
  // is what a forward cannot proceed without and the trunk is read on
  // every block of every forward. Protecting the optional half first is
  // how a run ends up with wired blocks beside an activation buffer the
  // compressor is free to take.
  //
  // Returns the bytes it managed to wire. A refusal is not an error --
  // the pool is an UP-TO -- and it is not rolled back either: see
  // wire_fixed_.
  std::size_t wire_into_pool();

  // GIVE THE POOL BACK. Freeing a wired buffer unwires it in the kernel,
  // so the machine recovers either way -- but the pool's own counter
  // would not, and a DiT destroyed after every clip (the ordinary
  // `unload_when_idle: destroy` path) would leak its whole share of the
  // budget per clip until nothing could wire at all.
  ~Ltx25Dit();

  // ---- growing back into free RAM (mechanism 4) ---------------------
  //
  // Streaming every block on every step is the safe answer and an
  // expensive one. When there is room, a streamed block is KEPT after it
  // runs and the next forward finds it resident.
  //
  // `bytes` is what must stay clear for whatever runs AFTER this
  // forward and has not allocated yet -- the VAE decode above all. Zero
  // is a real answer ("nothing runs after me that I do not free
  // first"), and it is different from never calling this: growth stays
  // OFF until a caller sets a reserve at least once, so a model whose
  // stage never asks the question streams forever and looks in the log
  // exactly like one that chose to.
  void set_residency_reserve(std::size_t bytes);

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

  // What the shared arena WILL hold at a given geometry, before one
  // exists.
  //
  // Static, and derived from the same numbers MetalBlock::reserve()
  // allocates from, because the callers that need it have no model yet:
  // the pinned-prefix count is decided at LOAD, and sizing it against a
  // constant is what put 20 of 48 blocks on a 16 GB box. MiniMax-H3
  // states the same figure the same way (scratch_bytes there), and for
  // the same reason -- the term scales with the token count, so a
  // constant is wrong at every geometry but one.
  //
  // Checked against reality rather than trusted: ltx25-block-metal-test
  // reserves a real arena at several geometries and holds this to
  // BlockScratch::bytes(). An estimate that drifts from what reserve()
  // does is worse than none, because it is the number a bounded box
  // commits to before it can measure anything.
  //
  // `levels` is the denoise-level count (1 unconditioned).
  static std::uint64_t scratch_bytes(const DitConfig& cfg, int video_tokens,
                                     int audio_tokens, int text_tokens,
                                     int levels);

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
  // Held for this model's lifetime, per the WeightSet contract -- and
  // here it is load-bearing rather than bookkeeping: a streamed block is
  // read from this set inside the forward.
  std::shared_ptr<vpipe::genai::WeightSet> _ws;
  // Full depth ALWAYS. When streaming, only [0, _pinned) are filled and
  // the rest are null -- a null slot is the signal to read that block
  // for this forward and drop it again. Sizing the vector to the full
  // depth rather than to _pinned is what lets the forward index blocks
  // by layer without a second mapping.
  std::vector<std::unique_ptr<MetalBlock>> _blocks;
  bool _stream_blocks = false;
  int  _pinned = 0;
  // The geometry a streamed block has to be given when it is built,
  // recorded by set_geometry because the block does not exist yet when
  // that runs.
  int  _geo_levels = 1;
  // Mechanism 4. Promotion is per streamed block, eviction is from the
  // TAIL -- never into the pinned prefix, which is not its to give.
  vpipe::genai::BlockResidency _resid;

  // Free the highest-index resident block, returning the bytes freed.
  // `allow_pinned` lets it dip into the prefix -- see the definition.
  std::size_t evict_tail_block_(bool allow_pinned = false);
  // mincore over EVERY block this model is holding, prefix included.
  // The prefix is not exempt from the question: it was sized at load
  // against what the box was believed to hold, and the measurement is
  // how that belief gets checked.
  void resident_pages_(std::size_t* examined, std::size_t* incore) const;

  // Read block `i` and make it ready to run: bind its weights, adopt the
  // shared scratch arena, and give it the RoPE tables set_geometry built.
  //
  // `arena` is passed BY VALUE rather than read from `_scratch`, because
  // this is also what the prefetch thread calls: a copy taken on the
  // main thread before the thread starts is a shared_ptr the worker can
  // read without racing the member.
  bool build_block_(int i, std::shared_ptr<BlockScratch> arena,
                    std::unique_ptr<MetalBlock>& out,
                    std::string* err) const;

  // ---- STREAMING SLOTS ------------------------------------------------
  //
  // Two destinations for the whole run, refilled in place, rather than a
  // block's worth of fresh buffers per block per forward.
  //
  // What the per-block allocation cost is not throughput -- the pread
  // that replaced the mapped memcpy already took that -- it is CHURN. A
  // streamed LTX block is ~140 tensors and ~400 MB, allocated and freed
  // 48 times a forward, and on a box with no room to spare each of those
  // allocations has to come from somewhere: the compressor, or swap. A
  // fixed pair asks for the memory once and never gives it back, so the
  // demand curve is flat instead of a sawtooth. MiniMax-H3 made the same
  // change for the same reason.
  //
  // NULLABLE, and that is what makes promotion free. A block admitted to
  // the resident set is MOVED out of its slot, leaving it empty, and the
  // next block to be streamed rebuilds it -- which is the read that was
  // going to happen anyway. So allocations converge to zero once the
  // resident set stops growing, without promotion needing a copy.
  //
  // Not wired: a slot is written on every block, so it is the hottest
  // memory in the model and the compressor has no reason to take it.
  // What is wired is the resident set, which is written once and then
  // only read.
  std::unique_ptr<MetalBlock> _slot[2];
  // The slot the MAIN path will fill next; the prefetch takes the other.
  int _slot_cur = 0;

  // Refill `dst` with `layer`'s weights, reusing its buffers. Falls back
  // per TENSOR inside bind_block: one whose size or dtype a raw read
  // cannot place is replaced rather than written into, and the rest of
  // the block still refills.
  bool refill_slot_(int layer, MetalBlock& dst, std::string* err) const;

  // Get `layer` into slot `idx`, refilling it when it exists and
  // building it when it does not (the first block, and the one after a
  // promotion).
  bool fill_slot_(int layer, int idx, std::shared_ptr<BlockScratch> arena,
                  std::string* err);

  // The manager, or null outside a session (which every unit test is).
  // Everything below no-ops there rather than branching at each site.
  vpipe::genai::GenerativeModelManager* manager_() const;
  // Wire the TRUNK, the CONNECTORS and the SCRATCH -- everything this
  // model holds that is not a streamed block.
  std::size_t wire_fixed_(bool on);
  // Every scratch buffer this model holds -- the shared block arena and
  // both connectors' planes. Its own walk because the scratch has a
  // shorter life than the rest of what wire_fixed_ covers.
  void each_scratch_(
      const std::function<void(vpipe::metal_compute::SharedBuffer&)>& fn);
  // Give the pool back what the current scratch is charged, BEFORE it is
  // replaced. Destroying a wired buffer does not decrement the pool.
  void unwire_scratch_();
  // Wire one block's weights. Called as a block is admitted and undone
  // as it is evicted, so the pool's charge tracks what is really held.
  std::size_t wire_block_(MetalBlock& b, bool on);
  // Has wire_fixed_ run? The scratch is reallocated by set_geometry, so
  // the fixed half is re-wired after every geometry change rather than
  // once at load.
  bool _wired_fixed = false;
  // One report per model, so a per-geometry re-wire does not repeat it.
  bool _wired_reported = false;
  // Bytes the pool would not take on the last wire pass; see wire_fixed_.
  std::size_t _unwirable = 0;
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
