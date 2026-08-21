#ifndef VPIPE_LTX25_BLOCK_METAL_H
#define VPIPE_LTX25_BLOCK_METAL_H

#include "ltx25-block-ref.h"
#include "ltx25-metal-ops.h"
#include "ltx25-rope.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace ltx25 {

// One `BasicAVTransformerBlock` on the GPU.
//
// The forward is a transcription of ltx25-block-ref's -- same order,
// same adaLN row slices, same pre-cross snapshot -- because that
// reference is verified against the model's own implementation at 3e-7.
// Where the two must differ (bf16, head-major attention, an explicit
// scratch arena) the difference is noted at the point it appears.
//
// SCRATCH is SHARED by the whole stack and sized once from the largest
// geometry it will see, so a DiT step allocates nothing. See
// BlockScratch below for why one arena serves 48 blocks, and for what
// it cost when each block had its own.

// One attention's weights, already on the GPU.
struct GpuAttn {
  // The four projections may be quantized; everything else in the
  // struct stays dense in any pack (the norms are 1D gains, and the
  // per-head gate is [heads, dim] -- 32 rows, whose quantization noise
  // would land on a sigmoid that decides how much of each head to keep).
  QWeight q_w, k_w, v_w, o_w;
  vpipe::metal_compute::SharedBuffer q_b, k_b, v_b, o_b;
  vpipe::metal_compute::SharedBuffer q_norm, k_norm;
  vpipe::metal_compute::SharedBuffer gate_w, gate_b;
  bool has_gate = false;
  int  heads = 0;
  int  head_dim = 0;
  int  query_dim = 0;    // the width `to_q` and the gate read
  int  ctx_dim = 0;      // the width `to_k` / `to_v` read
};

struct GpuStream {
  GpuAttn attn1, attn2;
  QWeight ff_in, ff_out;
  vpipe::metal_compute::SharedBuffer ff_in_b, ff_out_b;
  bool ff_has_bias = false;
  int  ff_hidden = 0;
  // The adaLN tables, uploaded once: 9 x dim, 2 x dim, 5 x dim.
  vpipe::metal_compute::SharedBuffer scale_shift, prompt_scale_shift,
      cross_table;
  int dim = 0;
};

// One stream's inputs for a forward.
//
// Every buffer is BORROWED and spelled as a pointer, because a 48-block
// stack passes the SAME five buffers to every block -- SharedBuffer is
// move-only, so holding them by value would mean either 48 copies of the
// handle or a shuffle at each block.
struct GpuStreamInput {
  const vpipe::metal_compute::SharedBuffer* x = nullptr;  // [tokens][dim],
                                                          // UPDATED in place
  const vpipe::metal_compute::SharedBuffer* context = nullptr;
  // The raw timestep-MLP outputs -- NOT pre-summed with the per-block
  // tables. The block adds its own table, exactly as the reference does,
  // which is what lets one driver serve all 48.
  const vpipe::metal_compute::SharedBuffer* timesteps = nullptr;   // 9*dim
  const vpipe::metal_compute::SharedBuffer* cross_scale_shift = nullptr;
  const vpipe::metal_compute::SharedBuffer* cross_gate = nullptr;
  // The PROMPT adaLN driver (2*dim). Null runs the static
  // prompt_scale_shift_table alone -- correct only for a checkpoint with
  // use_prompt_adaln_single false, which LTX-2.5 is not.
  const vpipe::metal_compute::SharedBuffer* prompt_timestep = nullptr;
  const RopeTable* pe = nullptr;
  const RopeTable* cross_pe = nullptr;
  int tokens = 0;
  int text_tokens = 0;
  bool present = true;

  // ---- per-token modulation, for a CONDITIONED generation ------------
  //
  // `timesteps` above is then a [n_levels][9*dim] TABLE rather than one
  // 9*dim driver, and `level` an int per token selecting its row. This
  // is LTX's `denoise_mask * sigma` with the redundancy taken out: a
  // token holding given content is modulated as clean, its neighbours at
  // the schedule's sigma, and there are two or three such levels, not
  // one per token.
  //
  // n_levels == 1 (the default, and every text-to-video forward) leaves
  // `level` unread and every dispatch on the broadcast path it was on
  // before conditioning existed.
  const vpipe::metal_compute::SharedBuffer* level = nullptr;   // int32[tokens]
  int n_levels = 1;
};

// One block's weights, already on the GPU.
//
// Filled two ways, and both must produce the same thing: by UPLOADING a
// host BlockWeights (what the tests do, so the Metal path and the CPU
// reference are checked against one another), or by BINDING the weight
// set's bf16 buffers directly (what the real checkpoint does -- no host
// round-trip, no conversion, no second copy).
struct GpuBlockWeights {
  GpuStream video, audio;
  GpuAttn   a2v, v2a;
  bool      have_audio = false;
  double    norm_eps = 1e-6;
  // The group size every quantized weight in this block was packed at,
  // or 0 when the block is entirely dense. Reported OUT of bind_block
  // because the kernels are resolved per group and the checkpoint is the
  // only thing that knows which -- see MetalOps::enable_quant.
  int       quant_group = 0;
};

// Every weight buffer this block holds, in no particular order.
//
// The streaming arm binds all of them to MAPPED shard pages, so this
// doubles as the block's footprint in the file-backed page cache --
// which is what the residency probe and the prefetch walk. Anything
// added to GpuBlockWeights has to be added here too, or it silently
// stops being prefetched: a miss shows up as a slower block, never as
// a wrong one, which is the kind of bug that survives a test suite.
void for_each_weight(
    const GpuBlockWeights& w,
    const std::function<void(const vpipe::metal_compute::SharedBuffer&)>& fn);

// The scratch ONE block forward needs -- and, deliberately, the scratch
// the WHOLE STACK shares.
//
// It used to be a member of MetalBlock, so all 48 blocks allocated their
// own. That is 47 copies of a buffer set that is dead the moment a block
// returns: the stack runs strictly sequentially, one command stream per
// block with a commit().wait() between, and nothing here survives a
// forward -- the residual stream is the CALLER's buffer, handed in
// through GpuStreamInput::x.
//
// MEASURED, at 960x544x121 (8160 video tokens, the geometry the model is
// actually built for): 1.18 GB per block, so 56.8 GB across the stack
// against 1.18 GB shared. Beside a 24 GB quantized checkpoint that is the
// difference between 81 GB and 25 GB on a 64 GB box -- between a run that
// swaps itself to death and one with 39 GB to spare.
//
// It went unnoticed for so long because the size is
// `max(video, audio, TEXT) tokens`, and every clip this port had run had
// fewer than the 1024 text tokens. The text width set the arena, so
// 768x448x9 and 512x320x9 cost exactly the same and memory looked flat in
// the geometry.
//
// SHARING IS SAFE ONLY BECAUSE OF THAT SEQUENCING. Anything that ran two
// blocks concurrently would have to give them separate arenas again.
struct BlockScratch {
  // a/b/c/d/e are the working planes. Five because the text
  // cross-attention needs its modulated query AND its modulated
  // key/value live at once, and the audio<->video pair needs the same
  // for two different streams.
  vpipe::metal_compute::SharedBuffer a, b, c, d, e;
  vpipe::metal_compute::SharedBuffer q, k, v, o;
  vpipe::metal_compute::SharedBuffer qh, kh, vh, oh;
  vpipe::metal_compute::SharedBuffer gate_logits;
  vpipe::metal_compute::SharedBuffer ff;
  vpipe::metal_compute::SharedBuffer mod_scale, mod_shift, mod_gate;
  // The key/value side of an audio<->video direction modulates from the
  // OTHER stream's table, so it cannot share mod_scale/mod_shift --
  // those still hold the query side's when the attention runs.
  vpipe::metal_compute::SharedBuffer kv_scale, kv_shift;
  vpipe::metal_compute::SharedBuffer snap_v, snap_a;

  // What it is currently sized FOR. A block whose shape exceeds any of
  // these grows the arena in place; every block already holding it sees
  // the grown buffers, because they hold the same object.
  std::size_t tokens = 0, dim = 0, levels = 0, ffh = 0;

  // The steel-attention plans this stack has met, one per distinct
  // (heads, tq, tkv, head_dim). A forward meets at most four -- video
  // self, text cross, and the two audio<->video directions -- and all 48
  // blocks meet the SAME four, so the AttnParams buffer and the
  // function-constant specialisation are built once for the stack
  // instead of once per block. Sharing the arena is what makes that
  // free; see the note above.
  //
  // Held by pointer so a later plan cannot move an earlier one: callers
  // keep the returned pointer across the dispatch that uses it.
  std::vector<std::unique_ptr<MetalOps::SteelAttn>> attn_plans;

  // The plan for one attention shape, built on first sight. Null when
  // steel cannot serve it, which leaves the caller on sdpa_full.
  const MetalOps::SteelAttn*
  steel_for(const MetalOps& ops, int heads, int tq, int tkv, int head_dim);

  // Bytes held, for the log line that makes the saving visible.
  std::uint64_t bytes() const noexcept;

  // Every working plane, for the WIRED POOL. A forward cannot proceed
  // without these, so they belong in the pool AHEAD of any resident
  // block -- a block is an optimisation the model can shed and stream
  // instead, and protecting the optional half first is how a run ends up
  // with wired weights beside an activation buffer the compressor is
  // free to take. See docs/MODEL-MEMORY.md, "The wired pool".
  void for_each_buffer(
      const std::function<void(vpipe::metal_compute::SharedBuffer&)>& fn);

  // What reserve() WILL allocate at these widths, before an arena
  // exists. Pure arithmetic, and it lives HERE -- next to the allocation
  // it predicts -- because the two have to move together: a caller sizes
  // a pinned prefix against this at LOAD time, long before it can
  // measure anything, and an estimate that drifts from reserve() is
  // worse than no estimate at all.
  //
  // `t` is the widest of video / audio / caption tokens, `d` the widest
  // stream, `f` the widest feed-forward hidden, `l` the denoise levels
  // -- the same four reserve() reduces its arguments to.
  static std::uint64_t predict_bytes(std::size_t t, std::size_t d,
                                     std::size_t f, std::size_t l) noexcept;
};

class MetalBlock {
public:
  // Adopt weights already on the GPU. The checkpoint path.
  static std::unique_ptr<MetalBlock>
  create(const MetalOps& ops, GpuBlockWeights w, std::string* err);

  // Upload a host BlockWeights -- the CPU-reference form -- and adopt
  // it. Kept so a test can drive both paths from one set of weights and
  // hold them to each other; the two can then never diverge on what a
  // weight NAME means.
  static std::unique_ptr<MetalBlock>
  create(const MetalOps& ops, const BlockWeights& w, std::string* err);

  // Upload without building a block, for a caller assembling a stack.
  static bool upload(const MetalOps& ops, const BlockWeights& w,
                     GpuBlockWeights& out, std::string* err);

  // Size the scratch for a forward of at most these geometries. Called
  // once; a later forward that needs more grows it and says so.
  // `max_levels` sizes the modulation scratch: a conditioned generation
  // holds one shift/scale/gate row per denoise level, not one.
  //
  // `arena` is the scratch this block will USE, and passing the same one
  // to every block of a stack is the point -- see BlockScratch. The
  // first call sizes it, the rest adopt it, and one that is too small
  // for a later block grows in place. Null allocates a private arena,
  // which is what a test driving a single block wants.
  bool reserve(int max_video_tokens, int max_audio_tokens, int max_text,
               std::string* err, int max_levels = 1,
               std::shared_ptr<BlockScratch>* arena = nullptr);

  // The weights this block adopted. Read-only, and only two callers
  // want it: the residency probe and the prefetch, both of which need
  // the block's mapped byte ranges rather than its behaviour.
  const GpuBlockWeights& weights() const noexcept { return _w; }

  // The arena this block is using, so a caller can report its size.
  const std::shared_ptr<BlockScratch>& scratch() const noexcept
  {
    return _s;
  }

  // Run the block, updating `video.x` / `audio.x` in place. Encodes into
  // `enc`; the caller commits. False (with `err`) on a shape
  // disagreement.
  bool forward(vpipe::metal_compute::ComputeEncoder& enc,
               GpuStreamInput& video, GpuStreamInput& audio,
               std::string* err);

  // The RoPE tables this forward will use, uploaded. Held here rather
  // than passed per call because a 48-block stack shares one set and
  // re-uploading them per block would dominate a short sequence.
  bool set_rope(const RopeTable* v_self, const RopeTable* a_self,
                const RopeTable* v_cross, const RopeTable* a_cross);

private:
  MetalBlock() = default;

  struct RopeGpu {
    vpipe::metal_compute::SharedBuffer cos, sin;
    int heads = 0, tokens = 0, half = 0;
    bool valid = false;
  };


  // One attention, whole: projections, q/k norm, RoPE, transpose in,
  // sdpa, transpose out, per-head gate, output projection. `out` is
  // token-major [tq][query_dim].
  void run_attention_(vpipe::metal_compute::ComputeEncoder& enc,
                      const GpuAttn& a,
                      const vpipe::metal_compute::SharedBuffer& xq,
                      int tq,
                      const vpipe::metal_compute::SharedBuffer& xkv,
                      int tkv,
                      const RopeGpu* q_pe, const RopeGpu* k_pe,
                      const vpipe::metal_compute::SharedBuffer& out);

  void stream_first_half_(vpipe::metal_compute::ComputeEncoder& enc,
                          const GpuStream& w, GpuStreamInput& s,
                          const RopeGpu* pe);
  void stream_ff_(vpipe::metal_compute::ComputeEncoder& enc,
                  const GpuStream& w, GpuStreamInput& s);
  // One direction of the audio<->video cross-attention. `lo` is 0 for
  // a2v and 2 for v2a.
  void av_cross_(vpipe::metal_compute::ComputeEncoder& enc,
                 const GpuAttn& attn, const GpuStream& q_w,
                 GpuStreamInput& q_in,
                 const vpipe::metal_compute::SharedBuffer& q_snap,
                 const GpuStream& kv_w, GpuStreamInput& kv_in,
                 const vpipe::metal_compute::SharedBuffer& kv_snap, int lo,
                 const RopeGpu* q_pe, const RopeGpu* kv_pe);

  // shift/scale/gate for rows [lo, lo+3) of a 9-row table, written into
  // the scratch modulation buffers. `rows` is the table's row count, so
  // the timestep slice is read the way the reference reshapes it.
  //
  // `levels` > 1 repeats the add per level, so the scratch comes out as
  // the [levels][dim] tables the per-token kernels index.
  void ada3_(vpipe::metal_compute::ComputeEncoder& enc,
             const vpipe::metal_compute::SharedBuffer& table,
             const vpipe::metal_compute::SharedBuffer& timesteps,
             int table_rows, int dim, int lo, int levels = 1);

  const MetalOps* _ops = nullptr;
  GpuBlockWeights _w;
  RopeGpu   _v_self, _a_self, _v_cross, _a_cross;
  // BORROWED-BY-SHARING: every block of one stack holds the same
  // arena. Never written between forwards, never read across them.
  std::shared_ptr<BlockScratch> _s;
};

}  // namespace ltx25

#endif
