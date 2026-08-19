#include "ltx25-dit-weights.h"

#include "generative-models/shared/streamed-refill.h"

#include <cstdint>
#include <cstring>
#include <string>

using vpipe::genai::WeightSet;
using vpipe::metal_compute::MetalCompute;
using vpipe::metal_compute::SharedBuffer;

namespace ltx25 {

namespace {

// One tensor, by full name. `stream` picks the non-retaining read.
//
// Residency is chosen PER CALL, not fixed for the model. Mapping is a
// candidate here at all because these bytes are used AS THEY SIT (bf16
// in, bf16 kernels): the `Copied` default in vpipe's
// docs/MODEL-MEMORY.md is about loaders that allocate a converted copy
// per dtype mismatch, which this is not. Which residency each call
// gets, and the measurement behind the split, is in the body.
SharedBuffer
get_(WeightSet& ws, MetalCompute* mc, const std::string& name, bool stream,
     WeightSet::Residency kept)
{
  if (!ws.has(name)) { return SharedBuffer{}; }
  // COPIED only when STREAMING, and the difference is measured rather
  // than argued.
  //
  // Streaming needs to own the bytes: a pinned prefix means nothing if
  // the kernel can evict it, and a forward is a cyclic scan, which is
  // the case an LRU page cache handles worst -- each block dropped
  // exactly before it comes round again.
  //
  // Preloading is the opposite. Everything is resident either way, so
  // owning the bytes buys nothing and costs their KIND: clean file pages
  // the kernel drops for free become anonymous memory it can only
  // compress. MEASURED at 960x544x121 on a 64 GB box, same seed, w8g64:
  // mapped 408.5 s of denoise against 473.5 s copied, with the copied
  // arm sitting at 36 GB compressed and its own weights at 2-3% resident.
  // Streamed reads own their bytes by definition. What the model KEEPS
  // is `kept` -- the MODEL's verdict, not this tensor's -- so a streaming
  // model's pinned prefix and trunk come out Copied too. kept_residency()
  // in the header is where that distinction is argued.
  const auto res = stream ? WeightSet::Residency::Copied : kept;
  if (!stream) { return ws.tensor(name, mc, res); }
  // The streamed read, which is the one that happens 140 times a block,
  // 48 blocks a step, every step. Allocate the destination and pread the
  // bytes straight into it rather than letting stream_tensor allocate
  // and then memcpy out of the shard's mmap: MEASURED on an M5 over a
  // 206 MB block, arms interleaved and their ORDER rotated, 0.86-1.48
  // GB/s the mapped way against 6.7-6.9 GB/s for the pread. The mapped
  // rate also swings 2x between rounds where pread's does not, which is
  // what a streamed forward turns into GPU occupancy that will not sit
  // still.
  //
  // Falls back on ANYTHING the raw read cannot serve, per tensor rather
  // than per block -- an f32 table among bf16 matrices costs its own old
  // path and nothing else. See generative-models/shared/streamed-refill.h
  // for why the two refusals are distinguished.
  //
  // The allocation stays. Removing it as well needs a destination that
  // outlives the block, which is a change to who owns a MetalBlock; this
  // is the part that needs no ownership change and is most of the win.
  const auto* ti = ws.src().info(name);
  // Asked BEFORE allocating: an LTX block is 140 tensors of which this
  // route serves 78, and allocating for the other 62 only to drop them
  // would add 24.5 MB of pointless allocation per block per step.
  if (ti != nullptr && ti->nbytes > 0 &&
      vpipe::genai::refill_serves(ws, name, vpipe::genai::RefillDst::kRaw)) {
    SharedBuffer dst = mc->make_shared_buffer(ti->nbytes);
    if (!dst.empty()) {
      // kRaw: this function hands back the checkpoint's OWN bytes, and
      // the f16 scales a quantized pack carries are converted by
      // need_as_bf16_ instead. Asking for kBf16 here would convert them
      // a second time -- silently, into plausible numbers off by an
      // exponent bias.
      const auto r = vpipe::genai::refill_streamed_tensor(
          ws, name, dst, vpipe::genai::RefillDst::kRaw);
      if (r == vpipe::genai::Refill::kFilled) { return dst; }
      // kFailed leaves `dst` partly written, so it is dropped rather
      // than returned; either refusal falls through to the read that
      // always worked.
    }
  }
  return ws.stream_tensor(name, mc, res);
}

// The scale/shift TABLES are F32 where every other tensor in the DiT is
// bf16 -- all 290 of them, six per block plus the two output heads.
// Reading one as bf16 is not a small error: it reinterprets each f32's
// upper half as a whole value and its lower half as the next, so the
// table comes out as alternating garbage and the block produces NaN on
// the first modulation.
//
// Converted at bind through derived(), so the cache holds one bf16 copy
// per table rather than one per model, and the key names the transform
// AND the dtype it produces (the rule in vpipe's docs/MODEL-MEMORY.md:
// the key is all the cache compares). They are tiny -- 9 x 4096 floats at the widest -- so
// this costs ~19 MB across all 48 blocks.
SharedBuffer
get_f32_as_bf16_(WeightSet& ws, MetalCompute* mc, const std::string& name)
{
  if (!ws.has(name)) { return SharedBuffer{}; }
  return ws.derived("ltx25/bf16/" + name, [&ws, mc, &name]() {
    const SharedBuffer src =
        ws.read(name, mc, WeightSet::Residency::Copied);
    if (src.empty()) { return SharedBuffer{}; }
    const std::size_t n = src.byte_size() / 4;
    SharedBuffer dst = mc->make_shared_buffer(n * 2);
    if (dst.empty()) { return SharedBuffer{}; }
    const auto* in = static_cast<const float*>(src.contents());
    auto* out = static_cast<std::uint16_t*>(dst.contents());
    for (std::size_t i = 0; i < n; ++i) {
      std::uint32_t u;
      std::memcpy(&u, &in[i], 4);
      const std::uint32_t r = ((u >> 16) & 1u) + 0x7fffu;   // RNE
      out[i] = (std::uint16_t)((u + r) >> 16);
    }
    return dst;
  });
}

bool
need_f32_as_bf16_(WeightSet& ws, MetalCompute* mc, const std::string& name,
                  SharedBuffer& out, std::string* miss)
{
  out = get_f32_as_bf16_(ws, mc, name);
  if (out.empty()) {
    if (miss != nullptr && miss->empty()) { *miss = name; }
    return false;
  }
  return true;
}

// Bind, or record the first name that was missing. A block assembled
// from a partial map runs at full cost and produces noise, so the first
// miss stops the walk.
bool
need_(WeightSet& ws, MetalCompute* mc, const std::string& name, bool stream,
      SharedBuffer& out, std::string* miss, WeightSet::Residency kept)
{
  out = get_(ws, mc, name, stream, kept);
  if (out.empty()) {
    if (miss != nullptr && miss->empty()) { *miss = name; }
    return false;
  }
  return true;
}

// Bind one linear that MAY be group-affine quantized.
//
// The three siblings `NAME.weight` (u32 codes), `NAME.scales` and
// `NAME.biases` are what `model-quantize` writes; their absence is the
// ordinary dense case, not an error, because a real pack quantizes some
// tensors and leaves others alone.
//
// HOW BITS AND GROUP ARE RECOVERED, and why no config.json is needed.
// The other loaders in this tree read `quantization.{bits,group_size}`
// out of config.json. This checkpoint is a Comfy single file whose
// config lives in the safetensors `__metadata__`, and a quantized
// output written from one gets a synthesized config.json -- so relying
// on it would make the two packings load differently. But `K` is known
// HERE from the architecture (the caller passes the width the linear
// reads), and the packing pins the rest:
//
//     scales cols = K / group     ->  group = K / scales_cols
//     codes  cols = K * bits / 32 ->  bits  = codes_cols * 32 / K
//
// Two shapes, two unknowns, no metadata. `*group` is filled on the
// first quantized tensor seen and CHECKED on every one after: a pack
// with two group sizes would need two kernels bound at once, and
// silently running the wrong one is a full-speed wrong answer.
// A tensor read as BF16 whatever the checkpoint stores it as.
//
// THE TRAP THIS CLOSES. `model-quantize` writes `.scales` and `.biases`
// as **F16** -- every writer in the tree does -- while the qmm kernels
// are the `_bf16` twin, whose buffers 1 and 2 are `VPIPE_ELT` =
// bfloat. Handing the raw F16 bytes to a bfloat kernel is not an
// approximation: the exponent bias and mantissa width both differ, so
// every scale comes out as an unrelated number and the block produces
// noise at full speed. Nothing in the shapes catches it, because the
// element COUNT is identical.
//
// Converted rather than read, so it is a `derived` entry (a transform
// the model keeps), keyed by dtype because that is what changes the
// bytes.
SharedBuffer
get_as_bf16_(WeightSet& ws, MetalCompute* mc, const std::string& name,
             bool stream, WeightSet::Residency kept)
{
  const auto* info = ws.src().info(name);
  if (info == nullptr || info->shape.empty()) { return SharedBuffer{}; }
  if (info->dtype == "BF16") { return get_(ws, mc, name, stream, kept); }

  std::size_t n = 1;
  for (auto d : info->shape) { n *= (std::size_t)d; }
  auto build = [&ws, mc, &name, info, n]() -> SharedBuffer {
    const SharedBuffer src =
        ws.read(name, mc, WeightSet::Residency::Copied);
    if (src.empty()) { return SharedBuffer{}; }
    SharedBuffer dst = mc->make_shared_buffer(n * 2);
    if (dst.empty()) { return SharedBuffer{}; }
    auto* out = static_cast<std::uint16_t*>(dst.contents());
    if (info->dtype == "F16") {
      const auto* in = static_cast<const _Float16*>(src.contents());
      for (std::size_t i = 0; i < n; ++i) {
        const float v = (float)in[i];
        std::uint32_t u;
        std::memcpy(&u, &v, 4);
        const std::uint32_t r = ((u >> 16) & 1u) + 0x7fffu;   // RNE
        out[i] = (std::uint16_t)((u + r) >> 16);
      }
      return dst;
    }
    if (info->dtype == "F32") {
      const auto* in = static_cast<const float*>(src.contents());
      for (std::size_t i = 0; i < n; ++i) {
        std::uint32_t u;
        std::memcpy(&u, &in[i], 4);
        const std::uint32_t r = ((u >> 16) & 1u) + 0x7fffu;
        out[i] = (std::uint16_t)((u + r) >> 16);
      }
      return dst;
    }
    return SharedBuffer{};
  };
  // Streamed: not retained, so it is rebuilt per read -- which is what
  // streaming means, and these are ~1.2 GB across the whole 4-bit stack.
  if (stream) { return ws.stream_derived(build); }
  return ws.derived("ltx25/bf16/" + info->dtype + "/" + name, build);
}

bool
need_as_bf16_(WeightSet& ws, MetalCompute* mc, const std::string& name,
              bool stream, SharedBuffer& out, std::string* miss,
              WeightSet::Residency kept)
{
  out = get_as_bf16_(ws, mc, name, stream, kept);
  if (out.empty()) {
    if (miss != nullptr && miss->empty()) { *miss = name; }
    return false;
  }
  return true;
}

bool
need_q_(WeightSet& ws, MetalCompute* mc, const std::string& name, bool stream,
        int K, QWeight& out, int* group, std::string* miss, std::string* err,
        WeightSet::Residency kept)
{
  const auto& src = ws.src();
  const auto* si = src.info(name + ".scales");
  const auto* ci = src.info(name + ".weight");
  if (si != nullptr && ci != nullptr && si->shape.size() == 2 &&
      ci->shape.size() == 2 && K > 0) {
    const long scols = si->shape[1];
    const long gcols = ci->shape[1];
    const long g = scols > 0 ? (long)K / scols : 0;
    const long b = (long)gcols * 32 / (long)K;
    if ((g != 32 && g != 64) || (b != 4 && b != 8) ||
        scols * g != (long)K || gcols * 32 != (long)K * b) {
      if (err != nullptr && err->empty()) {
        *err = "'" + name + "' looks quantized but its shapes do not close: "
               "K=" + std::to_string(K) + ", scales cols=" +
               std::to_string(scols) + ", codes cols=" +
               std::to_string(gcols);
      }
      return false;
    }
    if (*group == 0) {
      *group = (int)g;
    } else if (*group != (int)g) {
      if (err != nullptr && err->empty()) {
        *err = "'" + name + "' is packed at group " + std::to_string(g) +
               " but the checkpoint already used group " +
               std::to_string(*group);
      }
      return false;
    }
    out.bits = (int)b;
    out.group = (int)g;
    out.quantized = true;
    // The CODES are u32 and go through untouched; the scales and
    // biases are F16 in the checkpoint and bfloat to the kernel.
    if (!need_(ws, mc, name + ".weight", stream, out.codes, miss, kept) ||
        !need_as_bf16_(ws, mc, name + ".scales", stream, out.scales, miss,
            kept) ||
        !need_as_bf16_(ws, mc, name + ".biases", stream, out.qbias, miss,
            kept)) {
      return false;
    }
    return true;
  }
  out.quantized = false;
  return need_(ws, mc, name + ".weight", stream, out.w, miss, kept);
}

bool
bind_attn_(WeightSet& ws, MetalCompute* mc, const std::string& p, int heads,
           int head_dim, int query_dim, int ctx_dim, bool stream, GpuAttn& a,
           int* group, std::string* miss, std::string* qerr,
           WeightSet::Residency kept)
{
  a.heads = heads;
  a.head_dim = head_dim;
  a.query_dim = query_dim;
  a.ctx_dim = ctx_dim;
  // K per projection: `to_q` reads the query width, `to_k`/`to_v` read
  // the context width (they differ in cross-attention), and `to_out`
  // reads the concatenated heads.
  const int inner = heads * head_dim;
  const bool ok =
      need_q_(ws, mc, p + ".to_q", stream, query_dim, a.q_w, group, miss,
              qerr, kept) &&
      need_(ws, mc, p + ".to_q.bias",   stream, a.q_b, miss, kept) &&
      need_q_(ws, mc, p + ".to_k", stream, ctx_dim, a.k_w, group, miss,
              qerr, kept) &&
      need_(ws, mc, p + ".to_k.bias",   stream, a.k_b, miss, kept) &&
      need_q_(ws, mc, p + ".to_v", stream, ctx_dim, a.v_w, group, miss,
              qerr, kept) &&
      need_(ws, mc, p + ".to_v.bias",   stream, a.v_b, miss, kept) &&
      // to_out is a Sequential; the linear is index 0 and index 1 is an
      // Identity that carries nothing.
      need_q_(ws, mc, p + ".to_out.0", stream, inner, a.o_w, group, miss,
              qerr, kept) &&
      need_(ws, mc, p + ".to_out.0.bias",   stream, a.o_b, miss, kept) &&
      need_(ws, mc, p + ".q_norm.weight", stream, a.q_norm, miss, kept) &&
      need_(ws, mc, p + ".k_norm.weight", stream, a.k_norm, miss, kept);
  if (!ok) { return false; }
  // Gating is on for every attention in LTX-2.5, but the absence of the
  // tensors is a legal (older) checkpoint rather than an error.
  a.gate_w = get_(ws, mc, p + ".to_gate_logits.weight", stream, kept);
  a.has_gate = !a.gate_w.empty();
  if (a.has_gate) {
    a.gate_b = get_(ws, mc, p + ".to_gate_logits.bias", stream, kept);
  }
  return true;
}

bool
bind_stream_(WeightSet& ws, MetalCompute* mc, const std::string& block,
             bool is_audio, const DitConfig& cfg, bool stream, GpuStream& g,
             int* group, std::string* miss, std::string* qerr,
             WeightSet::Residency kept)
{
  const std::string pre = block + (is_audio ? "audio_" : "");
  const int dim   = is_audio ? cfg.audio_inner_dim() : cfg.inner_dim();
  const int heads = is_audio ? cfg.audio_num_attention_heads
                             : cfg.num_attention_heads;
  const int hd    = is_audio ? cfg.audio_attention_head_dim
                             : cfg.attention_head_dim;
  const int ctx   = is_audio ? cfg.audio_cross_attention_dim
                             : cfg.cross_attention_dim;
  g.dim = dim;
  // attn1 is SELF-attention, so its context width is the stream's own;
  // attn2 reads the text conditioning at `cross_attention_dim`.
  if (!bind_attn_(ws, mc, pre + "attn1", heads, hd, dim, dim, stream,
                  g.attn1, group, miss, qerr, kept) ||
      !bind_attn_(ws, mc, pre + "attn2", heads, hd, dim, ctx, stream,
                  g.attn2, group, miss, qerr, kept)) {
    return false;
  }
  // FeedForward: net.0.proj is the input linear, net.2 the output. 1 is
  // the activation and 3 the dropout, so neither carries weights.
  // ff.net.0.proj reads `dim` and writes 4*dim; ff.net.2 reads 4*dim.
  if (!need_q_(ws, mc, pre + "ff.net.0.proj", stream, dim, g.ff_in, group,
               miss, qerr, kept) ||
      !need_q_(ws, mc, pre + "ff.net.2", stream, dim * 4, g.ff_out, group,
               miss, qerr, kept)) {
    return false;
  }
  // The VIDEO feed-forward has NO bias (`ff_bias: false`) and the audio
  // one does. Reading the config rather than probing: an absent tensor
  // and a config that says there is none must agree, and the config is
  // what the model was built from.
  g.ff_has_bias = is_audio ? cfg.audio_ff_bias : cfg.ff_bias;
  if (g.ff_has_bias) {
    if (!need_(ws, mc, pre + "ff.net.0.proj.bias", stream, g.ff_in_b, miss,
        kept) ||
        !need_(ws, mc, pre + "ff.net.2.bias", stream, g.ff_out_b, miss, kept)) {
      return false;
    }
  }
  // 4x, as every LTX-2.5 stream ships. Derived rather than read because
  // the checkpoint states it only through the tensor's shape, which the
  // SDK's WeightSet does not expose.
  g.ff_hidden = dim * 4;

  // F32 in the checkpoint -- see get_f32_as_bf16_. `stream` is not
  // honoured for these: they are 100 KB against a block's 800 MB, and a
  // streamed table would be re-converted every forward.
  if (!need_f32_as_bf16_(ws, mc, pre + "scale_shift_table", g.scale_shift,
                         miss) ||
      !need_f32_as_bf16_(ws, mc, pre + "prompt_scale_shift_table",
                         g.prompt_scale_shift, miss)) {
    return false;
  }
  // NOT `audio_`-prefixed: the two cross tables are named for the
  // DIRECTION pair (a2v_ca) and then for which stream they modulate, so
  // the audio one is `scale_shift_table_a2v_ca_audio` and not
  // `audio_scale_shift_table_a2v_ca_*`.
  const std::string ct = block + "scale_shift_table_a2v_ca_" +
                         (is_audio ? "audio" : "video");
  return need_f32_as_bf16_(ws, mc, ct, g.cross_table, miss);
}

bool
bind_adaln_(WeightSet& ws, MetalCompute* mc, const std::string& p, int dim,
            int out_dim, DitTrunk::AdaLN& a, WeightSet::Residency kept)
{
  a.dim = dim;
  a.out_dim = out_dim;
  const std::string e = p + ".emb.timestep_embedder.";
  // SharedBuffer is move-only, so each slot is filled once, in place --
  // no temporaries to assign from.
  a.emb1_w = get_(ws, mc, e + "linear_1.weight", false, kept);
  a.emb1_b = get_(ws, mc, e + "linear_1.bias",   false, kept);
  a.emb2_w = get_(ws, mc, e + "linear_2.weight", false, kept);
  a.emb2_b = get_(ws, mc, e + "linear_2.bias",   false, kept);
  a.out_w  = get_(ws, mc, p + ".linear.weight",  false, kept);
  a.out_b  = get_(ws, mc, p + ".linear.bias",    false, kept);
  a.valid = !a.emb1_w.empty() && !a.emb2_w.empty() && !a.out_w.empty();
  return a.valid;
}

}  // namespace

bool
bind_qlinear(WeightSet& ws, MetalCompute* mc, const std::string& name, int K,
             bool stream, QWeight& out, int* group, std::string* miss,
             std::string* err, WeightSet::Residency kept)
{
  int local = 0;
  return need_q_(ws, mc, name, stream, K, out,
                 group != nullptr ? group : &local, miss, err, kept);
}

bool
bind_block(WeightSet& ws, MetalCompute* mc, const DitConfig& cfg, int layer,
           bool stream, GpuBlockWeights& out, std::string* err,
           WeightSet::Residency kept)
{
  std::string miss, qerr;
  // A quantization-shape complaint says far more than "missing X" -- the
  // tensor IS there, it just does not close -- so it wins the message.
  auto report = [&](std::string* err) {
    if (err == nullptr) { return; }
    *err = "block " + std::to_string(layer) + ": " +
           (qerr.empty() ? "missing '" + miss + "'" : qerr);
  };
  const std::string b =
      std::string(kDitPrefix) + "transformer_blocks." + std::to_string(layer)
      + ".";
  out.norm_eps = cfg.norm_eps;
  if (!bind_stream_(ws, mc, b, false, cfg, stream, out.video, &out.quant_group,
                    &miss, &qerr, kept)) {
    report(err);
    return false;
  }
  out.have_audio = cfg.use_audio_video_cross_attention;
  if (!out.have_audio) { return true; }
  if (!bind_stream_(ws, mc, b, true, cfg, stream, out.audio, &out.quant_group,
                    &miss, &qerr, kept)) {
    report(err);
    return false;
  }
  // BOTH cross-attentions run at the AUDIO head count and head dim --
  // the 4096-wide video query is projected down to the audio head size.
  // Their query/context widths are the two streams' own, and swapped
  // between the directions.
  const int ah = cfg.audio_num_attention_heads;
  const int ahd = cfg.audio_attention_head_dim;
  const bool ok =
      bind_attn_(ws, mc, b + "audio_to_video_attn", ah, ahd,
                 /*query=*/cfg.inner_dim(), /*ctx=*/cfg.audio_inner_dim(),
                 stream, out.a2v, &out.quant_group, &miss, &qerr, kept) &&
      bind_attn_(ws, mc, b + "video_to_audio_attn", ah, ahd,
                 /*query=*/cfg.audio_inner_dim(), /*ctx=*/cfg.inner_dim(),
                 stream, out.v2a, &out.quant_group, &miss, &qerr, kept);
  if (!ok) { report(err); }
  return ok;
}

bool
bind_trunk(WeightSet& ws, MetalCompute* mc, const DitConfig& cfg,
           DitTrunk& out, std::string* err, WeightSet::Residency kept)
{
  const std::string p(kDitPrefix);
  const int vd = cfg.inner_dim(), ad = cfg.audio_inner_dim();

  // The nine-row driver is `9 * dim` wide because cross_attention_adaln
  // is on; without it the block would want six. Sized from the config so
  // a checkpoint that disagrees fails at bind rather than at forward.
  const int k = cfg.cross_attention_adaln ? 9 : 6;
  bool ok = bind_adaln_(ws, mc, p + "adaln_single", vd, k * vd, out.video,
      kept);
  ok &= bind_adaln_(ws, mc, p + "audio_adaln_single", ad, k * ad, out.audio,
      kept);
  ok &= bind_adaln_(ws, mc, p + "prompt_adaln_single", vd, 2 * vd, out.prompt,
      kept);
  ok &= bind_adaln_(ws, mc, p + "audio_prompt_adaln_single", ad, 2 * ad,
                    out.audio_prompt, kept);
  if (cfg.use_audio_video_cross_attention) {
    ok &= bind_adaln_(ws, mc, p + "av_ca_video_scale_shift_adaln_single",
                      vd, 4 * vd, out.av_video_ss, kept);
    ok &= bind_adaln_(ws, mc, p + "av_ca_audio_scale_shift_adaln_single",
                      ad, 4 * ad, out.av_audio_ss, kept);
    ok &= bind_adaln_(ws, mc, p + "av_ca_a2v_gate_adaln_single", vd, vd,
                      out.av_a2v_gate, kept);
    ok &= bind_adaln_(ws, mc, p + "av_ca_v2a_gate_adaln_single", ad, ad,
                      out.av_v2a_gate, kept);
  }
  if (!ok) {
    if (err != nullptr) { *err = "one or more adaLN MLPs are missing"; }
    return false;
  }

  std::string miss;
  const bool bound =
      need_(ws, mc, p + "patchify_proj.weight", false, out.patchify_w, &miss,
          kept) &&
      need_(ws, mc, p + "patchify_proj.bias",   false, out.patchify_b, &miss,
          kept) &&
      need_(ws, mc, p + "proj_out.weight", false, out.proj_out_w, &miss,
          kept) &&
      need_(ws, mc, p + "proj_out.bias",   false, out.proj_out_b, &miss,
          kept) &&
      need_f32_as_bf16_(ws, mc, p + "scale_shift_table", out.scale_shift_out,
                        &miss);
  if (!bound) {
    if (err != nullptr) { *err = "missing '" + miss + "'"; }
    return false;
  }
  if (cfg.use_audio_video_cross_attention) {
    const bool ab =
        need_(ws, mc, p + "audio_patchify_proj.weight", false,
              out.audio_patchify_w, &miss, kept) &&
        need_(ws, mc, p + "audio_patchify_proj.bias", false,
              out.audio_patchify_b, &miss, kept) &&
        need_(ws, mc, p + "audio_proj_out.weight", false, out.audio_proj_out_w,
              &miss, kept) &&
        need_(ws, mc, p + "audio_proj_out.bias", false, out.audio_proj_out_b,
              &miss, kept) &&
        need_f32_as_bf16_(ws, mc, p + "audio_scale_shift_table",
                          out.audio_scale_shift_out, &miss);
    if (!ab) {
      if (err != nullptr) { *err = "missing '" + miss + "'"; }
      return false;
    }
  }
  // Optional: zero-initialised in checkpoints that predate it, and an
  // exact no-op when absent.
  out.keyframes_abs_pos = get_(ws, mc, p + "keyframes_abs_pos_embedding",
                               false, kept);
  return true;
}

}  // namespace ltx25
