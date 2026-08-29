#include "ltx25-lora.h"

#include "generative-models/shared/comfy-checkpoint.h"
#include "common/flex-data.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>

namespace ltx25 {

using vpipe::genai::WeightSet;
using vpipe::metal_compute::MetalCompute;
using vpipe::metal_compute::SharedBuffer;

namespace {

// The prefix every adapter published for this model uses. NOT the
// checkpoint's `model.diffusion_model.` -- an adapter is written against
// the module tree, so it carries one fewer level.
constexpr const char* kLoraPrefix = "diffusion_model.";

float
bf16_to_f32_(std::uint16_t h)
{
  const std::uint32_t bits = (std::uint32_t)h << 16;
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

std::uint16_t
f32_to_bf16_(float f)
{
  std::uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  // Round to nearest even on the discarded low half, the same rounding
  // a converted checkpoint was written with. Truncating instead biases
  // every scaled weight toward zero.
  const std::uint32_t lsb = (bits >> 16) & 1u;
  bits += 0x7fffu + lsb;
  return (std::uint16_t)(bits >> 16);
}

// A scaled COPY of `src`, or a refcount-sharing alias when the factor
// is exactly 1 -- which is the common case (`lora_scale` defaults to
// 1.0 and both published adapters make alpha/rank 1.0), and the case
// worth not copying 250 MB for.
SharedBuffer
scaled_(WeightSet& ws, MetalCompute* mc, const std::string& name,
        double factor, WeightSet::Residency res)
{
  SharedBuffer src = ws.tensor(name, mc, res);
  if (src.empty() || factor == 1.0) { return src; }
  const std::size_t n = src.byte_size() / sizeof(std::uint16_t);
  SharedBuffer dst = mc->make_shared_buffer(src.byte_size());
  if (dst.empty()) { return dst; }
  const auto* s = static_cast<const std::uint16_t*>(src.contents());
  auto* d = static_cast<std::uint16_t*>(dst.contents());
  for (std::size_t i = 0; i < n; ++i) {
    d[i] = f32_to_bf16_(bf16_to_f32_(s[i]) * (float)factor);
  }
  return dst;
}

// Bind one module's pair, checking it against the base linear it names.
//
// `k` and `n` are the widths the BASE weight reads and writes. They are
// the check that matters: a pair whose A is [rank][k] and B is
// [n][rank] can only be this module's, and one that is not says so here
// instead of dispatching a GEMM over the wrong shape.
bool
bind_pair_(WeightSet& ws, MetalCompute* mc, const std::string& mod,
           int k, int n, double factor, LoraPair& out,
           std::string* err)
{
  const std::string an = std::string(kLoraPrefix) + mod + ".lora_A.weight";
  const std::string bn = std::string(kLoraPrefix) + mod + ".lora_B.weight";
  if (!ws.has(an) && !ws.has(bn)) { return true; }   // not adapted
  if (!ws.has(an) || !ws.has(bn)) {
    if (err != nullptr) {
      *err = "'" + mod + "' has only one of lora_A / lora_B";
    }
    return false;
  }
  SharedBuffer a = scaled_(ws, mc, an, factor,
                           WeightSet::Residency::Copied);
  SharedBuffer b = ws.tensor(bn, mc, WeightSet::Residency::Copied);
  if (a.empty() || b.empty()) {
    if (err != nullptr) { *err = "'" + mod + "' would not bind"; }
    return false;
  }
  // Rank from A's own size against the width it must read, cross-checked
  // against B. Shapes are not carried on the buffer, and deriving the
  // rank twice from two independent tensors is a stronger check than
  // reading it from metadata that no loader can verify.
  const std::size_t ea = a.byte_size() / sizeof(std::uint16_t);
  const std::size_t eb = b.byte_size() / sizeof(std::uint16_t);
  if (k <= 0 || n <= 0 || ea % (std::size_t)k != 0
      || eb % (std::size_t)n != 0) {
    if (err != nullptr) {
      *err = "'" + mod + "' does not divide by its base widths";
    }
    return false;
  }
  const std::size_t ra = ea / (std::size_t)k;
  const std::size_t rb = eb / (std::size_t)n;
  if (ra == 0 || ra != rb) {
    if (err != nullptr) {
      *err = "'" + mod + "' rank disagrees between lora_A ("
           + std::to_string(ra) + ") and lora_B (" + std::to_string(rb)
           + ")";
    }
    return false;
  }
  out.a = std::move(a);
  out.b = std::move(b);
  out.rank = (int)ra;
  out.k = k;
  out.n = n;
  return true;
}

struct AttnDims {
  int query = 0, ctx = 0, inner = 0, heads = 0;
};

bool
bind_attn_(WeightSet& ws, MetalCompute* mc, const std::string& pre,
           const AttnDims& d, double factor, LoraAttn& out,
           std::string* err)
{
  return bind_pair_(ws, mc, pre + ".to_q", d.query, d.inner, factor,
                    out.q, err)
      && bind_pair_(ws, mc, pre + ".to_k", d.ctx, d.inner, factor,
                    out.k, err)
      && bind_pair_(ws, mc, pre + ".to_v", d.ctx, d.inner, factor,
                    out.v, err)
      && bind_pair_(ws, mc, pre + ".to_out.0", d.inner, d.query, factor,
                    out.o, err)
      // The per-head gate writes ONE logit per head, so its output is
      // `heads` and not `inner`. Both published adapters keep it at
      // rank 32 even when everything around it is 450.
      && bind_pair_(ws, mc, pre + ".to_gate_logits", d.query, d.heads,
                    factor, out.gate, err);
}

bool
bind_adaln_(WeightSet& ws, MetalCompute* mc, const std::string& pre,
            int dim, int out_dim, double factor, LoraAdaLN& out,
            std::string* err)
{
  const std::string e = pre + ".emb.timestep_embedder.";
  // linear_1 reads the 256-channel sinusoid; linear_2 is square in
  // `dim`; `.linear` writes the k*dim driver.
  return bind_pair_(ws, mc, e + "linear_1", 256, dim, factor, out.emb1, err)
      && bind_pair_(ws, mc, e + "linear_2", dim, dim, factor, out.emb2, err)
      && bind_pair_(ws, mc, pre + ".linear", dim, out_dim, factor, out.out,
                    err);
}

// Every tensor name the loader above will have consumed, so anything
// left over can be reported rather than ignored.
void
note_(std::unordered_set<std::string>& seen, const std::string& mod)
{
  seen.insert(std::string(kLoraPrefix) + mod + ".lora_A.weight");
  seen.insert(std::string(kLoraPrefix) + mod + ".lora_B.weight");
}

// Every tensor name in a safetensors file, straight off its header.
//
// The WeightSet does not enumerate -- it answers `has(name)`, which is
// the wrong direction for "what did this file contain that I did not
// ask for". Reading the header here is eight bytes of length and one
// JSON object, and it is the only way the leftover check below can be
// made at all.
std::vector<std::string>
st_tensor_names_(const std::string& file)
{
  std::vector<std::string> out;
  if (file.empty()) { return out; }
  std::FILE* f = std::fopen(file.c_str(), "rb");
  if (f == nullptr) { return out; }
  std::uint64_t n = 0;
  if (std::fread(&n, 1, sizeof(n), f) != sizeof(n) || n == 0
      || n > (std::uint64_t{1} << 30)) {
    std::fclose(f);
    return out;
  }
  std::string hdr((std::size_t)n, '\0');
  const bool ok = std::fread(hdr.data(), 1, (std::size_t)n, f)
                  == (std::size_t)n;
  std::fclose(f);
  if (!ok) { return out; }
  vpipe::FlexData doc;
  try { doc = vpipe::FlexData::from_json(hdr); }
  catch (...) { return out; }
  if (!doc.is_object()) { return out; }
  auto o = doc.as_object();
  for (auto it = o.begin(); it != o.end(); ++it) {
    const std::string k((*it).first);
    if (k != "__metadata__") { out.push_back(k); }
  }
  return out;
}

}  // namespace

bool
LoraAttn_any_(const LoraAttn& a)
{
  return a.q.valid() || a.k.valid() || a.v.valid() || a.o.valid()
      || a.gate.valid();
}

bool
LoraBlock::any() const noexcept
{
  return LoraAttn_any_(video_attn1) || LoraAttn_any_(video_attn2)
      || video_ff_in.valid() || video_ff_out.valid()
      || LoraAttn_any_(audio_attn1) || LoraAttn_any_(audio_attn2)
      || audio_ff_in.valid() || audio_ff_out.valid()
      || LoraAttn_any_(a2v) || LoraAttn_any_(v2a);
}

bool
LoraTrunk::any() const noexcept
{
  auto ad = [](const LoraAdaLN& a) {
    return a.emb1.valid() || a.emb2.valid() || a.out.valid();
  };
  return patchify.valid() || proj_out.valid() || audio_patchify.valid()
      || audio_proj_out.valid() || ad(video) || ad(audio) || ad(prompt)
      || ad(audio_prompt) || ad(av_video_ss) || ad(av_audio_ss)
      || ad(av_a2v_gate) || ad(av_v2a_gate);
}

const LoraBlock*
LoraAdapter::block(int layer) const noexcept
{
  if (layer < 0 || (std::size_t)layer >= _blocks.size()) { return nullptr; }
  return _blocks[(std::size_t)layer].any() ? &_blocks[(std::size_t)layer]
                                           : nullptr;
}

const LoraTrunk*
LoraAdapter::trunk() const noexcept
{
  return _trunk.any() ? &_trunk : nullptr;
}

std::unique_ptr<LoraAdapter>
LoraAdapter::load(WeightSet& ws, MetalCompute* mc, const DitConfig& cfg,
                  double scale, const std::string& file,
                  std::vector<std::string>* unmapped, std::string* err)
{
  auto out = std::unique_ptr<LoraAdapter>(new LoraAdapter());

  // ---- metadata: alpha/rank and the IC-LoRA reference geometry ------
  double factor = scale;
  if (!file.empty()) {
    vpipe::FlexData meta;
    std::string merr;
    if (vpipe::genai::comfy::read_metadata(file, meta, &merr)
        && meta.is_object()) {
      auto o = meta.as_object();
      auto num = [&](const char* k, double* v) {
        if (!o.contains(k)) { return false; }
        const std::string s(o.at(k).as_string(""));
        if (s.empty()) { return false; }
        try { *v = std::stod(s); } catch (...) { return false; }
        return true;
      };
      double alpha = 0.0, rank = 0.0;
      if (num("lora_alpha", &alpha) && num("lora_rank", &rank)
          && rank > 0.0 && alpha > 0.0) {
        // The usual LoRA convention. Both published adapters make this
        // exactly 1 (450/450); it is applied rather than assumed so an
        // adapter that does not is not silently over-driven.
        factor *= alpha / rank;
      }
      double rd = 0.0;
      if (num("reference_downscale_factor", &rd) && rd >= 1.0) {
        out->_ref_downscale = (int)std::lround(rd);
      }
      // The TEMPORAL companion. Read even though nothing published
      // carries it yet, so an adapter trained at a lower reference frame
      // rate is REFUSED by the conditioning rather than run at the
      // target's spacing -- which would place every reference token at
      // the wrong second and produce a clean, wrong clip.
      double rt = 0.0;
      if (num("reference_temporal_scale_factor", &rt) && rt >= 1.0) {
        out->_ref_temporal = (int)std::lround(rt);
      }
    }
  }

  const int vd = cfg.inner_dim();
  const int ad = cfg.audio_inner_dim();
  const int layers = cfg.num_layers;
  out->_blocks.resize((std::size_t)(layers > 0 ? layers : 0));

  std::unordered_set<std::string> seen;

  AttnDims v_self{vd, vd, vd, cfg.num_attention_heads};
  AttnDims v_cross{vd, cfg.cross_attention_dim, vd, cfg.num_attention_heads};
  AttnDims a_self{ad, ad, ad, cfg.audio_num_attention_heads};
  AttnDims a_cross{ad, cfg.audio_cross_attention_dim, ad,
                   cfg.audio_num_attention_heads};
  // Both cross-attentions run at the AUDIO head count and head dim; the
  // query and context widths are the two streams' own, swapped between
  // the directions. Same rule bind_block uses.
  const int cross_inner =
      cfg.audio_num_attention_heads * cfg.audio_attention_head_dim;
  AttnDims a2v{vd, ad, cross_inner, cfg.audio_num_attention_heads};
  AttnDims v2a{ad, vd, cross_inner, cfg.audio_num_attention_heads};

  for (int l = 0; l < layers; ++l) {
    const std::string b = "transformer_blocks." + std::to_string(l) + ".";
    LoraBlock& lb = out->_blocks[(std::size_t)l];
    std::string e;
    const bool ok =
        bind_attn_(ws, mc, b + "attn1", v_self, factor, lb.video_attn1, &e)
     && bind_attn_(ws, mc, b + "attn2", v_cross, factor, lb.video_attn2, &e)
     && bind_pair_(ws, mc, b + "ff.net.0.proj", vd, 4 * vd, factor,
                   lb.video_ff_in, &e)
     && bind_pair_(ws, mc, b + "ff.net.2", 4 * vd, vd, factor,
                   lb.video_ff_out, &e)
     && bind_attn_(ws, mc, b + "audio_attn1", a_self, factor,
                   lb.audio_attn1, &e)
     && bind_attn_(ws, mc, b + "audio_attn2", a_cross, factor,
                   lb.audio_attn2, &e)
     && bind_pair_(ws, mc, b + "audio_ff.net.0.proj", ad, 4 * ad, factor,
                   lb.audio_ff_in, &e)
     && bind_pair_(ws, mc, b + "audio_ff.net.2", 4 * ad, ad, factor,
                   lb.audio_ff_out, &e)
     && bind_attn_(ws, mc, b + "audio_to_video_attn", a2v, factor, lb.a2v,
                   &e)
     && bind_attn_(ws, mc, b + "video_to_audio_attn", v2a, factor, lb.v2a,
                   &e);
    if (!ok) {
      if (err != nullptr) { *err = "block " + std::to_string(l) + ": " + e; }
      return nullptr;
    }
    for (const char* m : {"attn1", "attn2", "audio_attn1", "audio_attn2",
                          "audio_to_video_attn", "video_to_audio_attn"}) {
      for (const char* s : {".to_q", ".to_k", ".to_v", ".to_out.0",
                            ".to_gate_logits"}) {
        note_(seen, b + m + s);
      }
    }
    for (const char* m : {"ff.net.0.proj", "ff.net.2",
                          "audio_ff.net.0.proj", "audio_ff.net.2"}) {
      note_(seen, b + m);
    }
  }

  // ---- the trunk ----------------------------------------------------
  {
    // The patchify projections both READ the latent channel width and
    // write their stream's inner dim; the heads run the other way. Same
    // widths ltx25-dit.cc dispatches them at, so a config that
    // disagrees is caught here rather than as a wrong-shaped GEMM.
    const int vp = cfg.in_channels;
    const int vo = cfg.out_channels;
    const int ao = cfg.audio_out_channels;
    std::string e;
    // Nine rows with cross-attention adaLN on, six without -- the same
    // question bind_trunk asks.
    const int k = cfg.cross_attention_adaln ? 9 : 6;
    const bool ok =
        bind_pair_(ws, mc, "patchify_proj", vp, vd, factor,
                   out->_trunk.patchify, &e)
     && bind_pair_(ws, mc, "proj_out", vd, vo, factor,
                   out->_trunk.proj_out, &e)
     && bind_pair_(ws, mc, "audio_patchify_proj", vp, ad, factor,
                   out->_trunk.audio_patchify, &e)
     && bind_pair_(ws, mc, "audio_proj_out", ad, ao, factor,
                   out->_trunk.audio_proj_out, &e)
     && bind_adaln_(ws, mc, "adaln_single", vd, k * vd, factor,
                    out->_trunk.video, &e)
     && bind_adaln_(ws, mc, "audio_adaln_single", ad, k * ad, factor,
                    out->_trunk.audio, &e)
     && bind_adaln_(ws, mc, "prompt_adaln_single", vd, 2 * vd, factor,
                    out->_trunk.prompt, &e)
     && bind_adaln_(ws, mc, "audio_prompt_adaln_single", ad, 2 * ad,
                    factor, out->_trunk.audio_prompt, &e)
     && bind_adaln_(ws, mc, "av_ca_video_scale_shift_adaln_single", vd,
                    4 * vd, factor, out->_trunk.av_video_ss, &e)
     && bind_adaln_(ws, mc, "av_ca_audio_scale_shift_adaln_single", ad,
                    4 * ad, factor, out->_trunk.av_audio_ss, &e)
     && bind_adaln_(ws, mc, "av_ca_a2v_gate_adaln_single", vd, vd, factor,
                    out->_trunk.av_a2v_gate, &e)
     && bind_adaln_(ws, mc, "av_ca_v2a_gate_adaln_single", ad, ad, factor,
                    out->_trunk.av_v2a_gate, &e);
    if (!ok) {
      if (err != nullptr) { *err = "trunk: " + e; }
      return nullptr;
    }
    for (const char* m : {"patchify_proj", "proj_out",
                          "audio_patchify_proj", "audio_proj_out"}) {
      note_(seen, m);
    }
    for (const char* m : {"adaln_single", "audio_adaln_single",
                          "prompt_adaln_single",
                          "audio_prompt_adaln_single",
                          "av_ca_video_scale_shift_adaln_single",
                          "av_ca_audio_scale_shift_adaln_single",
                          "av_ca_a2v_gate_adaln_single",
                          "av_ca_v2a_gate_adaln_single"}) {
      note_(seen, std::string(m) + ".emb.timestep_embedder.linear_1");
      note_(seen, std::string(m) + ".emb.timestep_embedder.linear_2");
      note_(seen, std::string(m) + ".linear");
    }
  }

  // ---- tally, and refuse anything left over -------------------------
  auto tally = [&](const LoraPair& p) {
    if (!p.valid()) { return; }
    ++out->_modules;
    out->_bytes += p.a.byte_size() + p.b.byte_size();
    if (p.rank > out->_max_rank) { out->_max_rank = p.rank; }
    if (p.n > out->_max_out) { out->_max_out = p.n; }
  };
  auto tally_attn = [&](const LoraAttn& a) {
    tally(a.q); tally(a.k); tally(a.v); tally(a.o); tally(a.gate);
  };
  auto tally_adaln = [&](const LoraAdaLN& a) {
    tally(a.emb1); tally(a.emb2); tally(a.out);
  };
  for (const LoraBlock& b : out->_blocks) {
    tally_attn(b.video_attn1); tally_attn(b.video_attn2);
    tally(b.video_ff_in); tally(b.video_ff_out);
    tally_attn(b.audio_attn1); tally_attn(b.audio_attn2);
    tally(b.audio_ff_in); tally(b.audio_ff_out);
    tally_attn(b.a2v); tally_attn(b.v2a);
  }
  tally(out->_trunk.patchify); tally(out->_trunk.proj_out);
  tally(out->_trunk.audio_patchify); tally(out->_trunk.audio_proj_out);
  tally_adaln(out->_trunk.video); tally_adaln(out->_trunk.audio);
  tally_adaln(out->_trunk.prompt); tally_adaln(out->_trunk.audio_prompt);
  tally_adaln(out->_trunk.av_video_ss);
  tally_adaln(out->_trunk.av_audio_ss);
  tally_adaln(out->_trunk.av_a2v_gate);
  tally_adaln(out->_trunk.av_v2a_gate);

  // Anything the adapter ships that the walk above never asked for.
  // This is the check that turns "applied most of it" into a refusal:
  // an adapter naming a module this port does not adapt would otherwise
  // render slightly wrong with nothing to say which part was dropped.
  std::vector<std::string> left;
  for (const std::string& n : st_tensor_names_(file)) {
    if (n.find(".lora_") == std::string::npos) { continue; }
    if (seen.count(n) == 0) { left.push_back(n); }
  }
  if (!left.empty()) {
    if (unmapped != nullptr) { *unmapped = left; }
    if (err != nullptr) {
      *err = "the adapter names " + std::to_string(left.size())
           + " tensor(s) this port does not apply, the first being '"
           + left.front()
           + "'. Applying the rest would render slightly wrong with "
             "nothing to say which modules were dropped";
    }
    return nullptr;
  }
  if (out->_modules == 0) {
    if (err != nullptr) {
      *err = "no '" + std::string(kLoraPrefix)
           + "...lora_A/lora_B' pair resolved -- not an LTX-2.5 adapter";
    }
    return nullptr;
  }
  return out;
}

}  // namespace ltx25
