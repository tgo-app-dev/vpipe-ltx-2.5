#ifndef VPIPE_LTX25_DIT_WEIGHTS_H
#define VPIPE_LTX25_DIT_WEIGHTS_H

#include "ltx25-block-metal.h"
#include "ltx25-config.h"

#include "generative-models/weight-set.h"

#include <functional>
#include <string>

namespace ltx25 {

// Binding the released checkpoint's tensors to the GPU block.
//
// The bytes are taken AS THEY SIT: the DiT ships bf16 and every kernel
// here is bf16, so a block binds the weight set's buffers directly --
// no host round-trip, no conversion, no second copy. That is the whole
// reason the plugin's kernels are bf16 rather than f32.
//
// GOING THROUGH THE WEIGHT SET is not optional (see vpipe's
// docs/MODEL-MEMORY.md): the manager owns the checkpoint and a model
// borrows, so
// two stages over one checkpoint share it and the manager can see what
// is resident. A model that opened its own mmap would be invisible to
// exactly the accounting a 39 GB checkpoint most needs.

// Everything in the DiT that is NOT a transformer block: the six adaLN
// MLPs, the two patchify projections, and the output heads.
//
// Held together because they are all trunk weights -- small, always
// resident, and loaded once -- as against the 48 blocks, which are the
// 39 GB and may stream.
struct DitTrunk {
  // AdaLayerNormSingle. THREE linears, not two:
  //
  //   timestep -> sinusoid(256)
  //            -> emb.timestep_embedder.linear_1   [dim, 256]
  //            -> SiLU
  //            -> emb.timestep_embedder.linear_2   [dim, dim]
  //            -> SiLU
  //            -> linear                           [k*dim, dim]
  //
  // The last one lives at the adaLN ROOT, not inside `.emb`, and it is
  // the one that fans a single dim-wide embedding out to the k
  // modulation rows the block slices.
  struct AdaLN {
    vpipe::metal_compute::SharedBuffer emb1_w, emb1_b;
    vpipe::metal_compute::SharedBuffer emb2_w, emb2_b;
    vpipe::metal_compute::SharedBuffer out_w,  out_b;
    int dim = 0;          // the embedder's width
    int out_dim = 0;      // k * dim
    bool valid = false;
  };
  AdaLN video, audio;              // the 9-row drivers
  AdaLN prompt, audio_prompt;      // the K/V modulation drivers
  AdaLN av_video_ss, av_audio_ss;  // the 4-row cross scale/shift
  AdaLN av_a2v_gate, av_v2a_gate;  // the 1-row cross gates

  vpipe::metal_compute::SharedBuffer patchify_w, patchify_b;
  vpipe::metal_compute::SharedBuffer audio_patchify_w, audio_patchify_b;
  vpipe::metal_compute::SharedBuffer proj_out_w, proj_out_b;
  vpipe::metal_compute::SharedBuffer audio_proj_out_w, audio_proj_out_b;
  // The [2, dim] output scale/shift tables. F32 in the checkpoint, and
  // left f32: they are two rows.
  vpipe::metal_compute::SharedBuffer scale_shift_out, audio_scale_shift_out;
  vpipe::metal_compute::SharedBuffer keyframes_abs_pos;   // may be empty
};

// Every weight buffer the TRUNK holds -- the eight adaLN chains, the
// patchify and output projections, the scale/shift tables, the keyframe
// embedding.
//
// Two callers, and they want the same set for opposite reasons: one
// counts the bytes so the manager can see what this model holds, the
// other hands them to the wired pool so the OS cannot take them. The
// trunk is read on every block of every forward, so it has a better
// claim on the pool than any single resident block does.
//
// Empty buffers are passed through rather than filtered, exactly as
// for_each_weight(GpuBlockWeights) does: an optional component leaves
// its slots empty and every caller copes with that anyway.
void for_each_weight(
    const DitTrunk& t,
    const std::function<void(const vpipe::metal_compute::SharedBuffer&)>& fn);

// The tensor-name prefix every DiT weight carries in the released
// single-file checkpoint. Recorded here rather than spelled at each call
// site because a Comfy repack could change it, and then it changes once.
inline constexpr const char* kDitPrefix = "model.diffusion_model.";

// Bind block `layer`'s weights straight off the weight set.
//
// `stream` picks stream_tensor() over tensor(): the streaming path does
// NOT retain, which is what lets a 39 GB checkpoint run on a box that
// cannot hold it -- and it is counted separately by the manager, so a
// one-time load is never misread as a bounded model thrashing.
//
// False (with `err` naming the first missing tensor) rather than a
// partially-bound block: a block with one unbound projection runs at
// full cost and produces noise.
// Copied or Mapped for the bytes this model KEEPS, decided by the
// model's residency verdict rather than per call.
//
//   PRELOADING -> Mapped. Everything is resident either way, so owning
//     the bytes buys nothing and costs their KIND: clean file pages the
//     kernel drops for free become anonymous memory it can only
//     compress. load_mapped() wrapping the whole shard is the usual
//     objection and does not apply -- the whole checkpoint being
//     resident is what preloading MEANS.
//   STREAMING  -> Copied, for the PINNED PREFIX and the trunk as well as
//     the tail. A prefix the kernel can evict is not a prefix, and a
//     forward is a cyclic scan: the one access pattern an LRU page cache
//     handles worst, dropping each block exactly before it comes round
//     again.
//
// This is MiniMax-H3's rule (kept_residency_ there) and the reason it is
// spelled out again rather than left implicit: this port decided it PER
// TENSOR instead -- streamed reads Copied, everything else Mapped -- so
// a streaming model's pinned prefix came out mapped, which is the one
// combination H3's note rules out. MEASURED on a 16 GB box: the prefix
// was evicted inside the first forward, the residency check saw it leave
// RAM ("resident weights are only 88% in RAM"), and shed it -- so the
// pin bought nothing and cost a re-read per forward.
//
// The 409.7 s vs 473.5 s that favours mapping was measured PRELOADED on
// a 64 GB box; generalising it to the streaming prefix is what went
// wrong here.
inline vpipe::genai::WeightSet::Residency kept_residency(bool stream_blocks)
{
  return stream_blocks ? vpipe::genai::WeightSet::Residency::Copied
                       : vpipe::genai::WeightSet::Residency::Mapped;
}

bool bind_block(vpipe::genai::WeightSet& ws,
                vpipe::metal_compute::MetalCompute* mc, const DitConfig& cfg,
                int layer, bool stream, GpuBlockWeights& out,
                std::string* err,
                vpipe::genai::WeightSet::Residency kept =
                    vpipe::genai::WeightSet::Residency::Mapped);

// Bind the trunk. Always cached -- it is small and every step reads it.
bool bind_trunk(vpipe::genai::WeightSet& ws,
                vpipe::metal_compute::MetalCompute* mc, const DitConfig& cfg,
                DitTrunk& out, std::string* err,
                vpipe::genai::WeightSet::Residency kept =
                    vpipe::genai::WeightSet::Residency::Mapped);

// ONE linear that may be group-affine quantized, for a caller outside the
// block stack. Same rule as the blocks': the `.scales`/`.biases` siblings
// decide, and bits and group are recovered from the SHAPES rather than
// from a config, so a mixed pack loads as it sits.
//
// Exported because the connectors are quantized by the same pass as the
// blocks and must therefore be READ the same way. A dense-only reader
// against a quantized pack does not fail -- it binds the u32 codes as
// bf16 and produces plausible conditioning at full speed -- so the two
// have to move together, and sharing this is what keeps them together.
//
// `K` is the width the linear READS; `*group` is filled on the first
// quantized tensor and checked against every one after.
bool bind_qlinear(vpipe::genai::WeightSet& ws,
                  vpipe::metal_compute::MetalCompute* mc,
                  const std::string& name, int K, bool stream, QWeight& out,
                  int* group, std::string* miss, std::string* err,
                  vpipe::genai::WeightSet::Residency kept =
                      vpipe::genai::WeightSet::Residency::Mapped);

}  // namespace ltx25

#endif
