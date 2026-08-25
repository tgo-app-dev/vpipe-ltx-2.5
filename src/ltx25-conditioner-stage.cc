#include "ltx25-conditioner-stage.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"
#include "generative-models/generative-model-manager.h"
#include "stages/model-registry.h"

#include <cstdlib>

#include <fstream>

#include <cstring>
#include <utility>

using vpipe::ConfigKey;
using vpipe::ConfigType;
using vpipe::FlexData;
using vpipe::FlexDataPayload;
using vpipe::InEdge;
using vpipe::Job;
using vpipe::PortSpec;
using vpipe::ResourceClaim;
using vpipe::RuntimeContext;
using vpipe::SessionContextIntf;
using vpipe::StageCategory;
using vpipe::StageSpec;
using vpipe::TensorBeat;
using vpipe::TensorBeatPayload;
using vpipe::fmt;

namespace ltx25 {

namespace {

const PortSpec kIports[] = {
  {.name = "prompt", .doc = "prompt text (FlexData string or {text: ...})",
   .type = &typeid(FlexDataPayload), .tags = "text", .clock_group = 0},
  {.name = "negative",
   .doc = "OPTIONAL negative prompt (FlexData) for classifier-free "
          "guidance on the `dev` checkpoint. Overrides the `negative` "
          "config key. Encoding one costs a SECOND 12B forward, so it "
          "happens only when there is a non-empty negative to encode -- "
          "and never on a `distilled` checkpoint, which is "
          "guidance-distilled and has no unconditional pass to blend with",
   .type = &typeid(FlexDataPayload), .tags = "text", .clock_group = 0},
  {.name = "model",
   .doc = "OPTIONAL model reference (model-select); overrides hf_dir",
   .type = &typeid(FlexDataPayload), .tags = "model", .clock_group = 0},
};

// The order is POSITIONAL with generate-video's: oport0 -> iport0 and
// oport1 -> iport1, the same convention `diffusion-conditioner` uses.
// Putting the audio context on oport1 (where it started) would have made
// the obvious wiring -- 0 to 0, 1 to 1 -- feed a 2048-wide audio context
// into the port that wants a 4096-wide negative.
const PortSpec kOports[] = {
  {.name = "conditioning",
   .doc = "the VIDEO cross-attention context, bf16 [pad_to, 4096] -- "
          "generate-video's iport0. LEFT-padded, so the sideband's "
          "valid_begin/valid_count give the real-token RANGE",
   .type = &typeid(TensorBeatPayload), .tags = "conditioning",
   .clock_group = 0},
  {.name = "neg_conditioning",
   .doc = "the negative VIDEO context, same shape -- generate-video's "
          "iport1. Emitted ONLY when a non-empty negative is set and the "
          "checkpoint is `dev`; a graph that wires this port and gets no "
          "beat is being told the negative was inert, not dropped",
   .type = &typeid(TensorBeatPayload), .tags = "conditioning",
   .clock_group = 0},
  {.name = "audio_conditioning",
   .doc = "the AUDIO cross-attention context, bf16 [pad_to, 2048], from "
          "the SAME hidden states through a second projection. Wire it "
          "to generate-video's audio_conditioning port to get a "
          "soundtrack; leave it unwired for video-only",
   .type = &typeid(TensorBeatPayload), .tags = "conditioning",
   .clock_group = 0},
  {.name = "neg_audio_conditioning",
   .doc = "the negative AUDIO context, same shape and the same rule as "
          "neg_conditioning. Separate from it because LTX guides the two "
          "streams at DIFFERENT scales (the reference defaults to 3.0 "
          "video against 7.0 audio), so one negative cannot serve both",
   .type = &typeid(TensorBeatPayload), .tags = "conditioning",
   .clock_group = 0},
};

const ConfigKey kAttrs[] = {
  {.key = "hf_dir", .type = ConfigType::String, .required = false,
   .doc = "the LTX-2.5 checkpoint root. The text encoder is resolved out "
          "of it the same way the DiT is, so both stages take the same "
          "path. A model-select source on the `model` iport (2) overrides this",
   .def_str = ""},
  {.key = "pad_to", .type = ConfigType::Int, .required = false,
   .doc = "the padded context width. The reference's "
          "TOKENIZER_MAX_LENGTH is 1024, which is also a whole number of "
          "the connector's 128-register tiles -- a value that is not a "
          "multiple of 128 is refused by the DiT rather than truncated",
   .def_int = Ltx25TextEncoder::kMaxTokens},
  {.key = "negative", .type = ConfigType::Text, .required = false,
   .doc = "a static negative prompt for classifier-free guidance on the "
          "`dev` checkpoint. A beat on iport1 overrides it. Empty (the "
          "default) means no negative is encoded at all, which saves a "
          "second 12B forward -- and is the only sensible setting on a "
          "`distilled` checkpoint, where guidance is baked in",
   .def_str = ""},
  {.key = "encoder_variant", .type = ConfigType::Text, .required = false,
   .doc = "pick a QUANTIZED text encoder from `text_encoders/` by name "
          "substring -- e.g. \"w4g64\" takes "
          "`gemma4-12b-with-proj-ltx-2.5-w4g64/`, the directory "
          "`model-quantize` writes. Empty (the default) takes the "
          "released bf16 file, so a run only changes precision when it "
          "asks to. MEASURED: the bf16 encoder is 24.5 GB and peaks at "
          "27.1 GB of footprint; w4g64 is 9.9 GB. The encoder runs once "
          "per prompt and is then destroyed, so this trades caption "
          "fidelity for the peak that decides whether a small box can "
          "run at all",
   .def_str = ""},
  {.key = "unload_when_idle", .type = ConfigType::Text, .required = false,
   .doc = "what happens to the 12B encoder between prompts. It is idle "
          "for the whole denoise, which on a 22B DiT is minutes, so this "
          "is the difference between fitting and thrashing -- and NOT "
          "only on a small box: the DiT sizes its irreversible streaming "
          "decision against this encoder's declaration, so an encoder "
          "that is kept costs 24 GB of the 64 GB one too. "
          "\"auto\" (default) decides from whether a peer streams and "
          "from RAM vs the pipeline's weight bytes; \"destroy\" frees "
          "the bytes (a reload per prompt); \"park\" hands them to the "
          "kernel as purgeable; \"keep\" pins them. Legacy: true / "
          "\"always\" = destroy, false / \"never\" = keep. NOTE that "
          "park reports 0 bytes for this encoder -- it is a Gemma LM and "
          "reads its weights uncached, so the weight set has nothing to "
          "hand over; it is logged, not hidden",
   .def_str = "auto"},
};

const StageSpec kSpec = {
  .type_name = Ltx25ConditionerStage::kTypeName,
  .doc = "LTX-2.5 text conditioning: a caption becomes the DiT's two "
         "cross-attention contexts (video 4096 and audio 2048) through "
         "Gemma-4 12B and LTX's own dual projection. The audio context "
         "is what `generate-video` needs to produce a soundtrack; no "
         "other conditioner in the tree emits one.",
  .display_name = "LTX-2.5 Conditioner",
  .category = StageCategory::Generative,
  .iports = kIports,
  .oports = kOports,
  .attrs = kAttrs,
};

// The caption out of a prompt beat: a bare string, or {text: ...}.
std::string
prompt_of_(const FlexData& fd)
{
  if (fd.is_string()) { return std::string(fd.as_string("")); }
  if (fd.is_object()) {
    const auto o = fd.as_object();
    for (const char* k : {"text", "prompt", "caption"}) {
      if (o.contains(k)) {
        const FlexData v = o.at(k);
        if (v.is_string()) { return std::string(v.as_string("")); }
      }
    }
  }
  return {};
}

// Positional, and named because two of them moved when the negative
// took iport1.
constexpr unsigned kPromptPort = 0;
constexpr unsigned kNegPort    = 1;
constexpr unsigned kModelPort  = 2;
// The least the text encoder can be held at while still being HELD:
// everything outside the layer stack, plus the two in-flight slots the
// streamer refills into.
//
// The stem is an LM's, because that is what this encoder is -- a Gemma
// backbone with LTX's aggregate projection bolted on. The projection
// itself (~2.3 GB of 4096x188160 and 2048x188160) falls OUTSIDE the
// stem and is therefore counted as trunk, which is correct: it never
// streams.
std::size_t
enc_floor_bytes(const std::string& enc)
{
  if (enc.empty()) { return 0; }
  return vpipe::model_memory::streaming_floor_bytes(
      enc, {"model.layers.", "layers."});
}

constexpr unsigned kCondOut          = 0;
constexpr unsigned kNegCondOut       = 1;
constexpr unsigned kAudioCondOut     = 2;
constexpr unsigned kNegAudioCondOut  = 3;

}  // namespace

const StageSpec*
Ltx25ConditionerStage::stage_spec() noexcept
{
  return &kSpec;
}

const StageSpec&
Ltx25ConditionerStage::spec() const noexcept
{
  return kSpec;
}

Ltx25ConditionerStage::Ltx25ConditionerStage(const SessionContextIntf* session,
                                             std::string               id,
                                             std::vector<InEdge>       iports,
                                             FlexData                  config)
  : vpipe::TypedStage<Ltx25ConditionerStage>(session, std::move(id),
                                             std::move(iports),
                                             std::move(config))
{
  // The oports have to be ALLOCATED, not just declared in the spec: the
  // spec is documentation, and a stage that skips this reports ZERO
  // oports to pipeline_from_spec, so every downstream edge is refused
  // as out of range.
  this->allocate_oports(kSpec.oports.size());

  // Deferred validation: the constructor never throws, and a bad config
  // is reported at launch, where the runtime skips the stage.
  const FlexData& c = this->config();
  if (c.is_object()) {
    const auto o = c.as_object();
    if (o.contains("hf_dir")) {
      _hf_dir = std::string(o.at("hf_dir").as_string(""));
    }
    if (o.contains("pad_to")) {
      _pad_to = (int)o.at("pad_to").as_int(Ltx25TextEncoder::kMaxTokens);
    }
    if (o.contains("negative")) {
      _negative = std::string(o.at("negative").as_string(""));
    }
    if (o.contains("encoder_variant")) {
      _enc_variant = std::string(o.at("encoder_variant").as_string(""));
    }
    if (o.contains("unload_when_idle")) {
      // Accepts the STRING vocabulary and the legacy BOOL. The key was a
      // bool here before the host grew destroy/park/keep, and shipped
      // pipeline JSON says `true`; silently reading a bool as a string
      // would resolve every such graph to "auto" and quietly change what
      // it does, so both spellings are read and the bool is mapped the
      // way it was documented (true = drop it, false = hold it).
      const FlexData& v = o.at("unload_when_idle");
      if (v.is_bool()) {
        _unload_cfg = v.as_bool(false)
                          ? vpipe::model_memory::UnloadPolicy::kDestroy
                          : vpipe::model_memory::UnloadPolicy::kKeep;
      } else {
        bool bad = false;
        _unload_cfg = vpipe::model_memory::parse_unload_policy(
            std::string(v.as_string("auto")), &bad);
        if (bad) {
          this->fail_config(fmt(
              "unload_when_idle must be auto|destroy|park|keep (or the "
              "legacy always|never|true|false); got '{}'",
              std::string(v.as_string(""))));
          return;
        }
      }
    }
  }
  if (_pad_to <= 0 || (_pad_to % 128) != 0) {
    this->fail_config(fmt("pad_to must be a positive multiple of 128 (the "
                          "connector's register count); got {}", _pad_to));
    return;
  }
  // Whether a model reference is WIRED is a runtime-graph fact, not a
  // config one, so the "no hf_dir anywhere" refusal happens at the
  // first beat rather than here -- fail_config would reject a graph
  // that legitimately gets its directory from iport1.
}

Ltx25ConditionerStage::~Ltx25ConditionerStage() = default;

// The pre-launch twin of the model-select read in process(). Bookkeeping
// only: nothing loads here, and the pipeline is not assembled yet.
//
// `model-select` has no iport, so its output is a constant of the run --
// which the runtime folds and delivers before the planning phase exactly
// so a consumer can declare against it. This stage was not taking it,
// and the cost was not subtle: a graph that names its checkpoint through
// model-select rather than through this stage's own `hf_dir` left
// `_hf_dir` empty for the whole planning phase, so declare_resources()
// returned nothing and the 15 GB encoder never reached the ledger. The
// DiT then sized its irreversible streaming decision against an on-disk
// guess for a component nobody had claimed.
//
// The beat carries a REFERENCE (a registry key or a path). It is stored
// as one; model_root_() is what turns it into a directory, at each of
// the places that actually walk the filesystem.
std::string
Ltx25ConditionerStage::model_root_() const
{
  return vpipe::resolve_model_dir(this->session(), _hf_dir);
}

void
Ltx25ConditionerStage::apply_constant(unsigned iport, const FlexData& beat)
{
  if (iport != kModelPort) { return; }
  std::string ref;
  if (!vpipe::apply_model_select_beat(beat, ref) || ref.empty()) { return; }
  // Latch the REFERENCE, the way the config path holds one, and let
  // model_root_() resolve it at each use.
  _hf_dir = ref;
}

std::vector<ResourceClaim>
Ltx25ConditionerStage::declare_resources() const
{
  if (_hf_dir.empty()) { return {}; }
  Config cfg;
  if (!resolve(model_root_(), cfg, nullptr, {}, _enc_variant)) { return {}; }
  if (cfg.enc_file.empty()) { return {}; }
  // The encoder file, not the root: the root also holds the 39 GB DiT,
  // and declaring that here would double-count what generate-video
  // already claims.
  //
  // BOTH numbers. This encoder streams its layers when the graph has no
  // room to hold it (see Ltx25TextEncoder::open), and the floor it
  // reduces to is what says whether a graph that does not fit preloaded
  // fits anyway. Declaring only the on-disk size is what makes the pool
  // check refuse a run that works -- MEASURED on the bf16 graph, where a
  // text encoder with no floor was the whole of the 9 GB overshoot.
  return {vpipe::model_memory::weight_claim_streamable(
      cfg.enc_file, enc_floor_bytes(cfg.enc_file))};
}

// The topological ledger. Same encoder, same floor, plus the thing the
// phase claims cannot express: what LEAVES this stage.
vpipe::StageMemory
Ltx25ConditionerStage::declare_memory() const
{
  vpipe::StageMemory m;
  if (_hf_dir.empty()) { return m; }
  Config cfg;
  if (!resolve(model_root_(), cfg, nullptr, {}, _enc_variant)) { return m; }
  if (cfg.enc_file.empty()) { return m; }
  namespace mm = vpipe::model_memory;
  // NAMED by the file, which is what the removable pool and every
  // revision below are keyed on.
  //
  // `releases` is the same question decide_resources() asks and has to
  // be answered the same way: only when the drop is certain from config.
  // `park` is not a release here -- this encoder is a Gemma LM reading
  // uncached, so park_weights() hands over 0 bytes and it stays entirely
  // resident.
  m.hold(cfg.enc_file, mm::dir_weights_bytes(cfg.enc_file),
         enc_floor_bytes(cfg.enc_file),
         _unload_cfg == mm::UnloadPolicy::kDestroy,
         _unload_cfg == mm::UnloadPolicy::kAuto);

  // The CONDITIONING, on four oports. bf16 [pad_to, dim] each, which is
  // exactly what process() allocates -- so the plan and the allocation
  // cannot describe different beats.
  //
  // Alive well past this stage: the denoise reads them on every step of
  // an 8-step schedule, and the plan works that lifetime out from the
  // edges rather than being told it. At the default 1024x4096 the video
  // context is 8 MB and all four together 12 MB, which is small beside a
  // 39 GB DiT and is still the honest entry -- an omitted payload is a
  // hole, and holes are what make a plan read roomier than the box.
  // From the freshly resolved `cfg`, not from the `_cfg` member: this
  // runs BEFORE the encoder loads, so the member is still
  // default-constructed and its widths are the header's defaults rather
  // than this checkpoint's. They agree on everything shipped, which is
  // exactly what would keep the mistake invisible.
  const int vdim = cfg.dit.cross_attention_dim;
  const int adim = cfg.dit.audio_cross_attention_dim;
  const std::size_t vbytes = (std::size_t)_pad_to * (std::size_t)vdim * 2;
  const std::size_t abytes = (std::size_t)_pad_to * (std::size_t)adim * 2;
  m.outputs.resize(4, 0);
  m.outputs[kCondOut]         = vbytes;
  m.outputs[kAudioCondOut]    = abytes;
  // The NEGATIVES only when one will really be emitted. A `distilled`
  // checkpoint is guidance-distilled and never produces them, and a
  // graph with no negative set does not either -- declaring them anyway
  // would put 12 MB in every plan for beats that never exist.
  if (!_negative.empty() && cfg.variant != Variant::kDistilled) {
    m.outputs[kNegCondOut]      = vbytes;
    m.outputs[kNegAudioCondOut] = abytes;
  }
  return m;
}

// Pass TWO. Everything above has been declared by every stage; nothing
// from this pass has been applied yet, so every stage deciding here sees
// the same picture.
//
// WHY THE PHASE MATTERS, and it is not bookkeeping. `generate-video`
// takes an irreversible block-streaming decision on the first
// conditioning beat -- which is after this stage has produced its output
// and before it has dropped the encoder. So an announcement at unload
// time arrives one decision too late, and the only thing that reaches the
// decision is a DECLARATION made now.
//
// MEASURED on a 16 GB box, w8 throughout: with the encoder declared for
// the whole run, the DiT read "footprint 39 GB (peers 39 GB)" -- 15.3 GB
// of which was an encoder destroyed before the first denoise step -- and
// plan_streaming's pinned-prefix fraction is gated on `ram > others +
// 5 GB`, so an inflated `others` silently left it at zero.
//
// ONLY when the encoder will really go. `park` releases NOTHING here:
// park_weights() walks a weight set's CACHED entries and this encoder is
// a Gemma LM reading uncached into its own members, so it parks 0 bytes
// and stays entirely resident. A peer that subtracted it would be short
// by the encoder's whole size. `keep` says so outright.
std::vector<ResourceClaim>
Ltx25ConditionerStage::decide_resources() const
{
  if (_hf_dir.empty()) { return {}; }
  Config cfg;
  if (!resolve(model_root_(), cfg, nullptr, {}, _enc_variant)) { return {}; }
  if (cfg.enc_file.empty()) { return {}; }
  namespace mm = vpipe::model_memory;
  // The same question resolve_idle_policy_() will answer at the first
  // beat, asked here because the answer has to be on the record before
  // any peer acts on it. `bounded` is unphased on purpose: it is the
  // no-release worst case, which is what "is this box tight" means.
  const bool releases =
      _unload_cfg == mm::UnloadPolicy::kDestroy ||
      (_unload_cfg == mm::UnloadPolicy::kAuto &&
       mm::bounded(this->session(), {cfg.enc_file, cfg.dit_file},
                   mm::kHeadroom));
  if (!releases) { return {}; }
  return mm::weight_claims_in_phase({cfg.enc_file}, mm::kPhaseCondition);
}

void
Ltx25ConditionerStage::reset_run_state()
{
  _emitted = 0;
}

bool
Ltx25ConditionerStage::ensure_loaded_(const std::string& root)
{
  if (_enc && _loaded_root == root) { return true; }
  std::string err;
  if (!resolve(root, _cfg, &err, {}, _enc_variant)) {
    this->session()->warn(fmt("{}: {}", this->id(), err));
    return false;
  }
  if (_cfg.enc_file.empty()) {
    this->session()->warn(fmt("{}: no text encoder under '{}'", this->id(),
                              root));
    return false;
  }
  auto* mc = this->session()->services()->metal_compute();
  if (mc == nullptr || !mc->valid()) {
    this->session()->warn(fmt("{}: no Metal device", this->id()));
    return false;
  }
  if (!_ops.init(mc, &err)) {
    this->session()->warn(fmt("{}: {}", this->id(), err));
    return false;
  }
  _ws = vpipe::genai::open_weight_set(_cfg.enc_file, this->session());
  if (!_ws) {
    this->session()->warn(fmt("{}: could not open '{}'", this->id(),
                              _cfg.enc_file));
    return false;
  }
  _enc = Ltx25TextEncoder::load(_cfg, _ws, _ops, mc, this->session(), &err);
  if (!_enc) {
    this->session()->warn(fmt("{}: {}", this->id(), err));
    return false;
  }
  _loaded_root = root;
  return true;
}

Job
Ltx25ConditionerStage::initialize(RuntimeContext& ctx)
{
  (void)ctx;
  // The encoder is loaded LAZILY, on the first prompt. Loading it here
  // would hold 24 GB through the whole init barrier while every peer
  // sizes itself, and a graph whose prompt never arrives would pay for
  // it anyway.
  co_return;
}

Job
Ltx25ConditionerStage::process(RuntimeContext& ctx)
{
  // The model reference moved to iport2 when the negative took iport1,
  // so read it by the named constant rather than by position.
  _have_model_port =
      ctx.num_iports() > kModelPort && ctx.iport_connected(kModelPort);
  if (_have_model_port) {
    auto mb = co_await ctx.read(kModelPort);
    if (const auto* mp = mb ? dynamic_cast<const FlexDataPayload*>(mb.get())
                            : nullptr) {
      // Parse the beat with the SHARED reader rather than reading it as
      // a string. `model-select` emits an OBJECT ({"hf_dir": ...}), and
      // as_string() on an object yields "" -- which then resolved the
      // empty key and surfaced as MDB_BAD_VALSIZE out of the registry,
      // naming neither this stage nor the beat it mis-read.
      // apply_model_select_beat also accepts the plain-string form, so
      // this reads both shapes the beat is allowed to take.
      std::string ref;
      if (vpipe::apply_model_select_beat(mp->data, ref) && !ref.empty()) {
        // Stored as the REFERENCE, matching apply_constant() above --
        // the pre-launch latch and this one have to agree, or the
        // directory declared is not the one loaded.
        _hf_dir = ref;
      }
    }
  }
  // A wired negative port is READ unconditionally, before anything can
  // return early: leaving a beat unread stalls the producer, and this
  // port is on the same clock as the prompt.
  std::string negative = _negative;
  if (ctx.num_iports() > kNegPort && ctx.iport_connected(kNegPort)) {
    auto nb = co_await ctx.read(kNegPort);
    if (const auto* np = nb ? dynamic_cast<const FlexDataPayload*>(nb.get())
                            : nullptr) {
      negative = prompt_of_(np->data);
    }
  }
  auto pb = co_await ctx.read(kPromptPort);
  if (!pb) {
    // EOS on the prompt: SIGNAL DONE, do not merely return. The driver
    // calls process() in a loop until the stage says it is finished
    // (`while (!stop_requested() && !done())`), so a bare co_return here
    // is re-entered immediately and forever -- a 99% CPU busy spin that
    // keeps the whole `vpipe --launch` alive with its weights resident
    // long after the work is done. Measured: 10 s of wall clock at 99%
    // CPU with no model loaded at all.
    ctx.signal_done();
    co_return;
  }
  const auto* fp = dynamic_cast<const FlexDataPayload*>(pb.get());
  if (fp == nullptr) { co_return; }
  const std::string prompt = prompt_of_(fp->data);
  if (prompt.empty()) {
    this->session()->warn(fmt("{}: the prompt beat carries no text; "
                              "emitting nothing", this->id()));
    co_return;
  }
  if (_hf_dir.empty()) {
    this->session()->warn(fmt("{}: no `hf_dir` and no model reference on "
                              "iport2; emitting nothing", this->id()));
    co_return;
  }
  if (!ensure_loaded_(model_root_())) { co_return; }

  // A negative is INERT on the distilled checkpoint -- it is
  // guidance-distilled, so there is no unconditional pass to blend with
  // and encoding one would spend a second 12B forward on nothing. Say
  // so once rather than silently dropping it, and rather than silently
  // paying for it.
  if (!negative.empty() && _cfg.variant == Variant::kDistilled) {
    if (!_warned_distilled_neg) {
      _warned_distilled_neg = true;
      this->session()->warn(
          fmt("{}: a negative prompt is set but this is the DISTILLED "
              "checkpoint, which is guidance-distilled -- there is no "
              "unconditional pass to blend with, so the negative is not "
              "encoded", this->id()));
    }
    negative.clear();
  }

  auto* mc = this->session()->services()->metal_compute();
  const int vdim = _cfg.dit.cross_attention_dim;
  const int adim = _cfg.dit.audio_cross_attention_dim;

  // One caption -> the two contexts, as a pair of beats. Shared by the
  // prompt and the negative so the two cannot drift in layout or in
  // what they put on the sideband.
  struct Pair { TensorBeat video, audio; bool ok = false; };
  auto encode_pair = [&](const std::string& text) -> Pair {
    Pair out;
    auto vbuf = mc->make_shared_buffer((std::size_t)_pad_to * vdim * 2);
    auto abuf = mc->make_shared_buffer((std::size_t)_pad_to * adim * 2);
    if (vbuf.empty() || abuf.empty()) {
      this->session()->warn(fmt("{}: no room for the two contexts",
                                this->id()));
      return out;
    }
    int vbegin = 0, vcount = 0;
    std::string err;
    if (!_enc->encode(text, _pad_to, vbuf, abuf, &vbegin, &vcount, &err)) {
      this->session()->warn(fmt("{}: {}", this->id(), err));
      return out;
    }
    // Debug tap (VPIPE_LTX25_COND_DUMP=<prefix>): the two contexts as raw
    // bf16, so a quantized encoder can be held against the bf16 one
    // NUMERICALLY. Pixels cannot answer that question -- a diffusion
    // sample moves to a different mode on any conditioning change, so a
    // low PSNR between two runs is equally consistent with a damaged
    // encoder and a perfectly good different sample.
    if (const char* dp = std::getenv("VPIPE_LTX25_COND_DUMP")) {
      auto wr = [&](const char* nm,
                    const vpipe::metal_compute::SharedBuffer& b, int dim) {
        std::ofstream f(std::string(dp) + "_" + nm + ".bin",
                        std::ios::binary);
        f.write(static_cast<const char*>(b.contents()),
                (std::streamsize)_pad_to * dim * 2);
      };
      wr("video", vbuf, vdim);
      wr("audio", abuf, adim);
    }
    auto beat_of = [&](const vpipe::metal_compute::SharedBuffer& b,
                       int dim) {
      TensorBeat tb;
      tb.dtype = TensorBeat::DType::Bf16;
      tb.shape = {(std::int64_t)_pad_to, (std::int64_t)dim};
      // ELEMENTS, not bytes -- resize_contiguous multiplies by
      // element_byte_size() (2 here) and clears strides /
      // storage_offset. data.resize() takes BYTES, and mixing the two
      // conventions is how a buffer ends up half the size it should be.
      tb.resize_contiguous((std::size_t)_pad_to * dim);
      std::memcpy(tb.data.data(), b.contents(),
                  (std::size_t)_pad_to * dim * 2);
      // LEFT padding: the real tokens are a SUFFIX. Every other
      // conditioner in this tree emits a prefix, so a consumer that
      // assumes one keeps precisely the empty rows -- which is why this
      // is a range and not a count.
      tb.sideband = FlexData::make_object();
      auto so = tb.sideband.as_object();
      so.insert("valid_begin", FlexData::make_int(vbegin));
      so.insert("valid_count", FlexData::make_int(vcount));
      so.insert("padding_side", FlexData::make_string("left"));
      return tb;
    };
    out.video = beat_of(vbuf, vdim);
    out.audio = beat_of(abuf, adim);
    out.ok = true;
    this->session()->info(
        fmt("{}: '{}' -> video [{},{}] + audio [{},{}], {} real tokens at "
            "[{}..{})", this->id(),
            text.size() > 48 ? text.substr(0, 45) + "..." : text,
            _pad_to, vdim, _pad_to, adim, vcount, vbegin, vbegin + vcount));
    return out;
  };

  // After the init barrier, so peers are real. Idempotent.
  resolve_idle_policy_();

  Pair pos = encode_pair(prompt);
  if (!pos.ok) { co_return; }

  // The negative costs a SECOND full 12B forward, which is why it is
  // encoded only when there is one to encode.
  Pair neg;
  if (!negative.empty()) { neg = encode_pair(negative); }

  // RELEASE BEFORE PUBLISHING, and this ordering is load-bearing too.
  //
  // `generate-video` loads its DiT lazily, on the first conditioning
  // beat -- and loading is where it takes the irreversible streaming
  // decision. So everything this stage is going to give back has to be
  // given back, and REVISED, before the beat lands. Releasing after the
  // writes (which is what this did) leaves the DiT racing a free it
  // cannot see: it wins on a fast box and loses on a slow one, from the
  // same graph.
  //
  // Both encodes are finished by here, so this is genuinely idle even
  // when a negative was needed.
  release_encoder_when_idle_();

  // THE NEGATIVES GO FIRST, and the order is load-bearing.
  // `generate-video` reads its positive with a blocking read and then
  // POLLS its negative (`backlog(1) > 0`) rather than reading it,
  // because a graph with no negative would otherwise deadlock there.
  // That poll is only correct if the negative is already queued when the
  // positive lands -- its own comment says so. Emitting the positive
  // first makes the poll a RACE that it usually loses, and losing it is
  // silent: the run simply proceeds unguided at CFG 1, which looks like
  // a weak negative prompt rather than a dropped one.
  if (neg.ok) {
    co_await ctx.write(kNegCondOut, vpipe::make_payload<TensorBeatPayload>(
                                        std::move(neg.video)));
    co_await ctx.write(kNegAudioCondOut,
                       vpipe::make_payload<TensorBeatPayload>(
                           std::move(neg.audio)));
  }
  co_await ctx.write(kCondOut, vpipe::make_payload<TensorBeatPayload>(
                                   std::move(pos.video)));
  co_await ctx.write(kAudioCondOut, vpipe::make_payload<TensorBeatPayload>(
                                        std::move(pos.audio)));
  ++_emitted;

}

// ---------------------------------------------------------------------
// The idle policy.
//
// Resolved AFTER the init barrier -- at the first process() -- because
// at construction the graph is half loaded and a stage would size
// against peers that have not appeared yet. See docs/MODEL-MEMORY.md.
// ---------------------------------------------------------------------
void
Ltx25ConditionerStage::resolve_idle_policy_()
{
  if (_idle_resolved) { return; }
  _idle_resolved = true;

  namespace mm = vpipe::model_memory;
  const bool streaming = mm::peer_streams(this->session());
  const bool tight =
      mm::bounded(this->session(), {_cfg.enc_file, _cfg.dit_file},
                  mm::kHeadroom);

  switch (_unload_cfg) {
    case mm::UnloadPolicy::kDestroy:
    case mm::UnloadPolicy::kPark:
    case mm::UnloadPolicy::kKeep:
      _idle_action = _unload_cfg;
      break;
    default:
      // DESTROY is the default answer here, and this model is the reason
      // the usual park/destroy split does not apply.
      //
      // Park would be the better trade if it could work: the next prompt
      // reuses the encoder for free when the pages survive. But parking
      // walks a weight SET's cached tensors, and this encoder is a Gemma
      // LM -- it reads uncached, holding its weights in its own members,
      // where the registry cannot see them. Parking it reports 0 bytes
      // and changes nothing.
      //
      // Meanwhile the cost of not letting go is not a small box's
      // problem. `generate-video` sizes an IRREVERSIBLE streaming
      // decision against this encoder's declaration, so a 24 GB encoder
      // that is merely idle pushes a 42 GB DiT into streaming on a 64 GB
      // machine that could hold it outright.
      _idle_action = mm::UnloadPolicy::kDestroy;
      break;
  }

  this->session()->log_debug(fmt(
      "Ltx25ConditionerStage('{}'): encoder + DiT footprint {} MB + {} MB "
      "headroom vs {} MB RAM, a peer streams={}, unload_when_idle={} -> {}",
      this->id(),
      mm::weight_footprint(this->session(), {_cfg.enc_file, _cfg.dit_file})
          >> 20,
      mm::kHeadroom >> 20, mm::phys_ram() >> 20, streaming ? "yes" : "no",
      mm::unload_policy_name(_unload_cfg),
      mm::unload_policy_name(_idle_action)));
  if (_idle_action == mm::UnloadPolicy::kKeep && !tight) {
    this->session()->log_debug(fmt(
        "Ltx25ConditionerStage('{}'): holding the encoder pinned -- note "
        "that generate-video sizes its streaming decision against it",
        this->id()));
  }
}

// Free the bytes, AND say so.
//
// The revision is the point, not the free. Declarations persist for the
// whole run at max(held, estimate) -- deliberately, because a weight set
// only accounts for what it CACHED and an uncached-reading model would
// otherwise vanish from the accounting the moment it finished loading.
// The consequence is that dropping this encoder is invisible to peers
// unless the stage that dropped it says so.
//
// And the peer that reads it is deciding something it cannot take back.
// MEASURED on the 64 GB box: with the declaration standing, the DiT
// computed "footprint 65 GB (peers 65 GB) + 8 GB headroom -> STREAM
// blocks", then spent the denoise re-faulting ~37 GB of mapped weights
// per step at ~550 MB/s -- about 67 s/step -- for want of the 24 GB this
// stage had already returned to the box.
void
Ltx25ConditionerStage::destroy_encoder_()
{
  if (!_enc && !_ws) { return; }
  auto* mgr = this->session()->services()->generative_model_manager();
  // THE PHASE CLAIM'S OTHER HALF. decide_resources() promised these
  // bytes are gone before the peers that sized against them run; this is
  // where the promise is kept, and reporting it is what stops the
  // manager warning at the end of the launch that it was not. An
  // unfalsifiable promise is worse than none -- a broken one shows up
  // only as thrash, with nothing in the log connecting it to the claim.
  if (mgr != nullptr && !_cfg.enc_file.empty()) {
    mgr->note_phase_released(_cfg.enc_file);
  }
  // TO THE POOL, and BEFORE the resets. pool_weights() finds the set
  // through a WEAK reference, so once these drop their last strong one
  // there is nothing left to pool and the call is a silent no-op.
  //
  // Only under `auto`, which is the case the pool exists for: an encoder
  // dropped because the box was tight, wanted again on the very next
  // prompt. Pooled it stays purgeable -- a peer that genuinely needs the
  // room takes it, one that does not leaves the next prompt nothing to
  // reload. `destroy` is the caller asking for the bytes back now.
  //
  // NOT specialised to anything, so it is recyclable and a relaunch over
  // the same checkpoint finds it. (The DiT's adaLN bake does not change
  // this: it clears the model's own handles and writes nothing into the
  // weight set, so no set here is schedule-specific.)
  if (mgr != nullptr && !_cfg.enc_file.empty() &&
      _unload_cfg == vpipe::model_memory::UnloadPolicy::kAuto) {
    mgr->pool_weights(_cfg.enc_file);
  }
  _enc.reset();
  _ws.reset();
  _loaded_root.clear();
  if (mgr != nullptr && !_cfg.enc_file.empty()) {
    mgr->revise_declaration(_cfg.enc_file, 0);
    // The SAME correction on the topological plan. The two ledgers are
    // computed differently and a stage that corrects only one leaves the
    // other reporting an encoder that no longer exists.
    {
      vpipe::StageMemory m = declare_memory();
      for (vpipe::StageHolding& h : m.holdings) {
        if (h.source != _cfg.enc_file) { continue; }
        h.preload = 0;
        h.floor   = 0;
      }
      this->revise_memory(m);
    }
    this->session()->log_debug(fmt(
        "Ltx25ConditionerStage('{}'): encoder dropped and its declaration "
        "revised to 0 -- peers sizing after this see the room back",
        this->id()));
  }
}

// Purgeable rather than freed. Reports what it actually managed to hand
// over, which for this encoder is 0 -- see resolve_idle_policy_(). A
// park that frees nothing looks exactly like a park that worked until
// someone measures the box, so it is logged either way.
//
// No declaration revision here on purpose: parked bytes are STILL HELD
// bytes for the accounting, and they only return to the box if the
// kernel gets round to reclaiming them. A peer deciding this step
// whether it can hold a block cannot be told room exists on that basis.
void
Ltx25ConditionerStage::park_encoder_()
{
  auto* mgr = this->session()->services()->generative_model_manager();
  if (mgr == nullptr || _cfg.enc_file.empty()) { return; }
  const std::size_t got = mgr->park_weights(_cfg.enc_file);
  this->session()->log_debug(fmt(
      "Ltx25ConditionerStage('{}'): parked {} MB of the encoder{}",
      this->id(), got >> 20,
      got == 0 ? " -- nothing parkable; it is a Gemma LM and reads its "
                 "weights uncached, so the bytes live in the model rather "
                 "than the weight set"
               : ""));
}

void
Ltx25ConditionerStage::release_encoder_when_idle_()
{
  switch (_idle_action) {
    case vpipe::model_memory::UnloadPolicy::kDestroy:
      destroy_encoder_();
      break;
    case vpipe::model_memory::UnloadPolicy::kPark:
      park_encoder_();
      break;
    default: break;                     // kKeep: hold it pinned
  }
}

}  // namespace ltx25
