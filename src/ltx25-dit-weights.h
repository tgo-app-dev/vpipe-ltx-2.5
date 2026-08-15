#ifndef VPIPE_LTX25_DIT_WEIGHTS_H
#define VPIPE_LTX25_DIT_WEIGHTS_H

#include "ltx25-block-metal.h"
#include "ltx25-config.h"

#include "generative-models/weight-set.h"

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
bool bind_block(vpipe::genai::WeightSet& ws,
                vpipe::metal_compute::MetalCompute* mc, const DitConfig& cfg,
                int layer, bool stream, GpuBlockWeights& out,
                std::string* err);

// Bind the trunk. Always cached -- it is small and every step reads it.
bool bind_trunk(vpipe::genai::WeightSet& ws,
                vpipe::metal_compute::MetalCompute* mc, const DitConfig& cfg,
                DitTrunk& out, std::string* err);

}  // namespace ltx25

#endif
