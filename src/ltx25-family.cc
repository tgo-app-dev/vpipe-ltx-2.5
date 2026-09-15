#include "ltx25-family.h"
#include "ltx25-config.h"
#include "ltx25-dit.h"
#include "ltx25-generator.h"
#include "ltx25-text-encoder.h"

#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "generative-models/generative-model-manager.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "interfaces/session-services-intf.h"
#include "stages/model-memory.h"
#include "stages/model-registry.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;

using vpipe::ResourceClaim;
using vpipe::fmt;
using vpipe::genai::VideoGenerator;
using vpipe::genai::VideoModelCreateArgs;

namespace ltx25 {

std::string_view
Ltx25Family::tag() const noexcept
{
  return kFamily;
}

namespace {

// Which DiT pack to prefer, when a root holds more than one.
//
// A quantized pack (a DIRECTORY of shards) and the bf16 Comfy file can
// sit side by side under diffusion_models/, as can the `distilled` and
// `dev` DiTs, and they differ by up to 3x in footprint -- 42 GB against
// 14 GB for w4g64. That is too big a difference to decide implicitly, in
// EITHER direction: silently taking the quantized one surprises anyone
// who keeps bf16 for quality, and silently taking bf16 makes a
// deliberate quantize look like it did nothing.
//
// So it is asked for, from the `variant` key on ltx-2.5-model-config,
// which reaches load through VideoModelCreateArgs::model_config.
// `VPIPE_LTX25_VARIANT` overrides it -- the same shape as
// VPIPE_LTX25_STREAM beside it -- and SAYS SO when it does, because an
// env var that silently beats a checked-in pipeline is the trap this
// whole function exists to close, only pointed the other way.
std::string
prefer_variant_(const vpipe::genai::VideoModelCreateArgs* args,
                const vpipe::SessionContextIntf* session)
{
  std::string from_cfg;
  if (args != nullptr && args->model_config != nullptr &&
      args->model_config->is_object()) {
    const auto o = args->model_config->as_object();
    if (o.contains("variant")) {
      from_cfg = std::string(o.at("variant").as_string(""));
    }
  }
  const char* e = std::getenv("VPIPE_LTX25_VARIANT");
  if (e == nullptr || *e == '\0') { return from_cfg; }
  if (!from_cfg.empty() && from_cfg != e && session != nullptr) {
    session->warn(fmt(
        "ltx-2.5: VPIPE_LTX25_VARIANT='{}' overrides the model-config "
        "stage's variant '{}'", e, from_cfg));
  }
  return std::string(e);
}

// The widest single block under `stems`, off the checkpoint's own tensor
// table. The walk stream_floor_bytes() does -- first matching stem, the
// index up to the next '.' -- reduced to the one number that walk does
// not hand back. 0 when nothing matches.
std::size_t
widest_block_bytes_(const std::string& dit,
                    const std::vector<std::string_view>& stems)
{
  auto wts = vpipe::genai::MetalLlamaWeights::open_model(dit);
  if (!wts.has_value()) { return 0; }
  std::vector<std::vector<std::size_t>> stacks(stems.size());
  for (const std::string& nm : wts->tensor_names()) {
    std::size_t si = stems.size();
    for (std::size_t i = 0; i < stems.size(); ++i) {
      if (nm.rfind(std::string(stems[i]), 0) == 0) { si = i; break; }
    }
    if (si == stems.size()) { continue; }
    const std::size_t i0 = stems[si].size();
    const std::size_t dot = nm.find('.', i0);
    if (dot == std::string::npos) { continue; }
    const std::size_t idx =
        (std::size_t)std::atol(nm.substr(i0, dot - i0).c_str());
    const auto* ti = wts->info(nm);
    std::vector<std::size_t>& blocks = stacks[si];
    if (blocks.size() <= idx) { blocks.resize(idx + 1, 0); }
    blocks[idx] += ti != nullptr ? (std::size_t)ti->nbytes : 0;
  }
  std::size_t widest = 0;
  for (const std::vector<std::size_t>& blocks : stacks) {
    for (std::size_t b : blocks) { if (b > widest) { widest = b; } }
  }
  return widest;
}

// The least this DiT can be held at while still being HELD: everything
// outside the block stack, the two in-flight slots a streamer refills
// into, AND the one block a streaming load keeps pinned.
//
// That last term is this port's, not the host's. streaming_floor_bytes()
// is the trunk plus two slots, which is the floor of a DiT that pins
// nothing. Ltx25Dit::load pins block 0 even when streaming, because the
// shared scratch arena is allocated THROUGH a resident block (see the
// note at `d->_pinned = 1`) -- so a streaming load holds the trunk, that
// block and both slots, and a floor one block short of it under-states
// the least this model can run on. That is the direction that admits a
// graph the box cannot hold.
//
// The widest block comes off the same tensor table and the same stems
// the floor walks, so the two terms cannot describe different
// checkpoints.
//
// The stem is the checkpoint's, `model.diffusion_model.` +
// `transformer_blocks.`, and a stem that matches nothing yields 0 --
// which reads as "no smaller form", the safe answer for a pack this
// does not recognise. The trunk it leaves behind is real and not small:
// both text connectors live there and never stream.
std::size_t
dit_floor_bytes(const std::string& dit)
{
  if (dit.empty()) { return 0; }
  // Both spellings, prefixed first: the shipped packs carry the
  // `model.diffusion_model.` prefix, and a stem that matches nothing
  // costs one miss and falls through to the next.
  const std::vector<std::string_view> stems = {
      "model.diffusion_model.transformer_blocks.", "transformer_blocks."};
  const std::size_t floor =
      vpipe::model_memory::streaming_floor_bytes(dit, stems);
  if (floor == 0) { return 0; }
  return floor + widest_block_bytes_(dit, stems);
}

// The DiT the PRE-LOAD plan names for `root`: the LARGEST candidate
// under diffusion_models/, not the preferred one. Empty when `root` does
// not resolve.
//
// No beat exists yet, so the `variant` that will actually choose is
// unknowable here (see the correction in load()). Between guessing and
// guessing SAFELY, the direction matters: over-declaring costs a peer
// some unnecessary caution, which is recoverable; under-declaring tells
// the graph there is room that is not there, and every stage sizing
// against it loads. The preference order happens to name the bf16 file
// today, which is also the biggest -- but a root holding only quantized
// packs would have it pick one arbitrarily, and picking w4 while the run
// loads w8 is the dangerous direction.
//
// ONE function, because two callers have to agree on it exactly: the
// planning calls declare this key, and load() revises THIS key when the
// config selects another file. The correction used to re-resolve with no
// variant instead, which names the preference order's file -- the
// declared one only when it is also the largest. Where it was not (a
// root holding only a w4 and a w8 pack, say), the correction revised a
// key nobody had declared and the real estimate stood for the rest of
// the run beside the one for the file actually loaded.
std::string
largest_dit_(const std::string& root)
{
  Config cfg;
  // resolve() refuses a root with no DiT, so a success always names one.
  if (!resolve(root, cfg, nullptr, prefer_variant_(nullptr, nullptr))) {
    return {};
  }
  std::string biggest = cfg.dit_file;
  std::size_t most = vpipe::model_memory::dir_weights_bytes(biggest);
  std::error_code ec;
  const fs::path dm = fs::path(root) / "diffusion_models";
  for (const auto& de : fs::directory_iterator(dm, ec)) {
    if (ec) { break; }
    const std::size_t b =
        vpipe::model_memory::dir_weights_bytes(de.path().string());
    if (b > most) { most = b; biggest = de.path().string(); }
  }
  return biggest;
}

}  // namespace

bool
Ltx25Family::claims(const std::string& root, const std::string& model_type) const
{
  // The models-DB hint is enough on its own when it is there -- it is
  // what a `model-fetch` of a catalogued LTX-2.5 entry wrote, and it
  // survives a directory being renamed.
  if (model_type == kFamily) { return true; }
  // Otherwise probe. `resolve` refuses anything whose DiT `_class_name`
  // is not AVTransformer3DModel, so a hit is both the detection AND the
  // config, and a miss costs one safetensors header read.
  Config cfg;
  return resolve(root, cfg, nullptr, prefer_variant_(nullptr, nullptr));
}

int
Ltx25Family::align_frames(const std::string& root, int frames) const
{
  (void)root;   // the rule is the VAE's, and every LTX-2.5 VAE shares it
  return align_num_frames(frames);
}

void
Ltx25Family::size_grid(const std::string& root, int* gh, int* gw) const
{
  (void)root;   // the rule is the VAE's, and every LTX-2.5 VAE shares it
  // 32 in both axes: the conv VAE compresses space by 32, and the DiT's
  // patch is 1 on top of that. A size that is not a multiple comes back
  // rounded UP rather than refused, so a caller asks for the picture it
  // wants instead of deriving one from the compression ratio.
  if (gh != nullptr) { *gh = 32; }
  if (gw != nullptr) { *gw = 32; }
}

// What the planning calls name as this family's DiT under `root`.
//
// The file load() actually opened, for exactly as long as the generator
// built from it is alive; the largest candidate otherwise.
//
// WHY THE LOADED FILE AT ALL. generate-video remembers the first
// holding's `source` as the checkpoint it later hands to drop_weights,
// pool_weights and note_phase_released, and the manager matches that
// name exactly. The stage re-reads declare_holdings() straight after
// load to correct its plan, so answering with the opened file there is
// what makes the idle release name the checkpoint that is really held.
// Left at the largest candidate, a run that loaded any other pack
// released nothing when idle.
//
// WHY ONLY WHILE ITS GENERATOR LIVES. The family outlives every launch,
// and the next launch's planning phase has no beat either -- so a record
// that outlived its generator would declare the LAST run's pack before
// this run has chosen one, which is the under-declaring direction the
// largest-candidate rule exists to avoid. A generator that is alive is
// still holding its file, so naming it then is simply true.
std::string
Ltx25Family::declared_dit_(const std::string& root) const
{
  {
    std::lock_guard<std::mutex> lk(_loaded_mu);
    const auto it = _loaded.find(root);
    if (it != _loaded.end() && !it->second.lease.expired()) {
      return it->second.file;
    }
  }
  return largest_dit_(root);
}

std::vector<ResourceClaim>
Ltx25Family::declare_resources(const std::string& root) const
{
  // Declared BEFORE any driver starts, which is the only moment every
  // peer can size itself against this checkpoint. A DiT reported as 0
  // bytes is exactly the silent under-count the planning phase exists to
  // prevent, so claim the FILE that will actually be read rather than
  // the directory -- the largest candidate before a load, the opened one
  // while it is held. See declared_dit_().
  const std::string dit = declared_dit_(root);
  if (dit.empty()) {
    // Nothing resolved. Claim the root so the checkpoint is at least
    // visible; an unresolvable root will fail loudly at load anyway.
    return vpipe::model_memory::weight_claims({root});
  }
  // The text encoder is NOT claimed here. It is loaded by
  // `diffusion-conditioner`, a different stage, which declares it
  // itself -- claiming it twice would size every peer against 24 GB
  // that this stage never holds.
  //
  // BOTH numbers, not just the size on disk. This DiT streams its 48
  // blocks when the box is tight, and the floor it can be reduced to --
  // the trunk, the block it keeps pinned and two in-flight slots -- is
  // what says whether a graph that does not fit preloaded fits anyway.
  // A streamable component that declares no floor is counted at full
  // size by the pool check, and reads as a graph that cannot run when
  // it can.
  std::vector<ResourceClaim> out{vpipe::model_memory::weight_claim_streamable(
      dit, dit_floor_bytes(dit))};
  // The ANE feed-forward module (`ane_ffn`): ONE unit in the denoise phase,
  // held by CoreML where no other ledger sees it. Booked unconditionally,
  // because this call cannot see the graph's settings: generate-video
  // keeps it only when the graph asked for the tier, re-labels it as its
  // own, and tells the generator no when the plan grants nothing. The
  // architecture is fixed, so the default config's width is the width.
  for (auto& c : vpipe::model_memory::coreml_claims(
           "ltx-2.5-ane-ffn", Ltx25Dit::ane_bytes(DitConfig{}), 1,
           vpipe::model_memory::kPhaseDenoise)) {
    out.push_back(std::move(c));
  }
  return out;
}

std::vector<vpipe::StageHolding>
Ltx25Family::declare_holdings(const std::string& root) const
{
  // The SAME checkpoint the claim above names, and it has to be the
  // same string: the plan merges two stages holding one checkpoint by
  // this name, and the stage remembers it as what the removable pool
  // will later be asked for. Asked again after load, this names the file
  // load() opened -- which is what makes that later request name a
  // checkpoint that is actually held.
  std::vector<ResourceClaim> claims = declare_resources(root);
  std::vector<vpipe::StageHolding> out;
  out.reserve(claims.size());
  for (const ResourceClaim& c : claims) {
    if (c.kind != vpipe::model_memory::kWeightsKind) { continue; }
    vpipe::StageHolding h;
    h.source  = c.key;
    h.preload = vpipe::model_memory::dir_weights_bytes(c.key);
    h.floor   = dit_floor_bytes(c.key);
    // `releases` and `reclaimable` are deliberately left alone -- they
    // are generate-video's `unload_when_idle`, which a family cannot
    // see, and the stage stamps them on.
    if (h.preload > 0) { out.push_back(std::move(h)); }
  }
  return out;
}

std::size_t
Ltx25Family::latent_bytes(const std::string& root, int width, int height,
                          int frames) const
{
  if (width <= 0 || height <= 0 || frames <= 0) { return 0; }
  // The CHANNEL COUNT from the checkpoint, not from a literal here. It
  // is 128 on everything that ships, but this figure feeds a claim
  // peers size against and the config already says so -- resolve() is
  // one metadata parse off a header, no weights, which is what makes it
  // answerable in the planning phase at all. Unresolvable means "cannot
  // say", and the stage then declares nothing rather than a plausible
  // number.
  Config cfg;
  if (!resolve(root, cfg, nullptr, prefer_variant_(nullptr, nullptr))) {
    return 0;
  }
  // What Ltx25Generator actually writes into VideoGenResult::video:
  // f32, [in_channels, (F-1)/8+1, H/32, W/32] -- from the same
  // constants the generator divides by, so the two cannot end up
  // describing different clips.
  const std::size_t lf = (std::size_t)((frames - 1) / kTemporalCompression + 1);
  const std::size_t lh = (std::size_t)(height / kSpatialCompression);
  const std::size_t lw = (std::size_t)(width / kSpatialCompression);
  if (lf == 0 || lh == 0 || lw == 0 || cfg.dit.in_channels <= 0) { return 0; }
  return (std::size_t)cfg.dit.in_channels * lf * lh * lw * sizeof(float);
}

std::size_t
Ltx25Family::denoise_scratch_bytes(const std::string& root, int width,
                                   int height, int frames,
                                   const vpipe::FlexData* model_config) const
{
  if (width <= 0 || height <= 0 || frames <= 0) { return 0; }
  // The WIDTHS the arena scales with are the checkpoint's, so this
  // resolves it -- through the `variant` the beat names when the graph
  // wired one as a constant, which is the pack load() will open.
  VideoModelCreateArgs va;
  va.model_config = model_config;
  Config cfg;
  if (!resolve(root, cfg, nullptr, prefer_variant_(&va, nullptr))) {
    return 0;
  }
  const GenerationParams p = GenerationParams::from_flex(
      model_config != nullptr ? *model_config : vpipe::FlexData{});

  // THE TOKEN COUNTS, by Ltx25Generator::generate's own arithmetic: a
  // latent of (F-1)/8+1 frames over an H/32 x W/32 grid, one token a
  // cell, and audio from the clip's PIXEL frames over fps -- or from
  // `audio_seconds` when the beat sets one.
  const int lf = (frames - 1) / kTemporalCompression + 1;
  const int lh = height / kSpatialCompression;
  const int lw = width / kSpatialCompression;
  if (lf <= 0 || lh <= 0 || lw <= 0) { return 0; }
  const int v_tokens = lf * lh * lw;
  int a_tokens = 0;
  if (p.audio && cfg.dit.use_audio_video_cross_attention) {
    // fps is not an input to this question, so this takes the fallback
    // the generator itself uses when the request carries none. The arena
    // is sized by the WIDEST stream, and at 25 audio latents a second
    // against a whole latent grid every 8 frames, audio is the widest
    // only for a clip a few cells across -- or an `audio_seconds` far
    // longer than the picture.
    const double secs = p.audio_seconds > 0.0
                            ? p.audio_seconds
                            : (double)frames / 24.0;
    a_tokens = std::max(
        0, (int)std::llround(secs * kAudioLatentsPerSecond));
  }
  // The caption rows are the conditioner's `pad_to`, which is that
  // stage's key and not visible from here. Its default is the encoder's
  // own limit, so that is what an unconfigured graph carries.
  const int text_tokens = Ltx25TextEncoder::kMaxTokens;

  // WHAT THIS CANNOT SEE, all of which a plan's geometry does not carry:
  //   * references. An anchor adds denoise LEVELS and appended tokens,
  //     and an IC-LoRA appends a whole reference clip; one level and no
  //     appended tokens is the unconditioned generation.
  //   * an adapter's two planes. Its rank lives in its file, which is
  //     named per generation.
  // Each of those only ADDS to the arena, so this is the least a
  // generation at this geometry allocates, not a bound on all of them.
  return (std::size_t)Ltx25Dit::scratch_bytes(cfg.dit, v_tokens, a_tokens,
                                              text_tokens, /*levels=*/1);
}

bool
Ltx25Family::audio_cost(const std::string& root, int frames, double fps,
                        std::size_t* latent, std::size_t* pcm,
                        std::size_t* arena) const
{
  (void)root;
  if (frames <= 0 || fps <= 0.0) { return false; }
  // The reference derives the audio length from the clip's PIXEL frames
  // over fps, NOT from the video latent frame count -- see
  // kAudioLatentsPerSecond. Substituting the latter is wrong by the
  // temporal compression factor, which is 8x.
  const double secs = (double)frames / fps;
  const std::size_t at =
      (std::size_t)std::llround(secs * kAudioLatentsPerSecond);
  if (at == 0) { return false; }
  // f32 [channels, frames, mel] -- the transposed layout the generator
  // emits, not the DiT's internal packing.
  if (latent != nullptr) {
    *latent = (std::size_t)kAudioLatentChannels * at *
              (std::size_t)kAudioLatentMelBins * sizeof(float);
  }
  // f32 stereo at the BWE's output rate. An UPPER BOUND on purpose: a
  // checkpoint with no `bwe` section falls back to the vocoder alone and
  // plays at its own lower rate, so this over-declares there -- which is
  // the safe direction, and the rate is not knowable without loading the
  // pack. See docs/MODEL-MEMORY.md: an under-estimate is not a
  // conservative error.
  const std::size_t samples = (std::size_t)std::llround(secs * 48000.0);
  const std::size_t pcm_b = samples * 2 * sizeof(float);
  if (pcm != nullptr) { *pcm = pcm_b; }
  // The decode's own transient, and the honest answer is that it is not
  // computable here: it runs the audio VAE, then a 108-conv vocoder,
  // then the BWE's causal STFT, and their peak depends on channel counts
  // that live in the checkpoint's metadata. What IS certain is that the
  // finished waveform is part of it, so a bound below the PCM would be
  // one nothing could justify.
  //
  // Declared at the PCM rather than at a made-up multiple of it:
  // audio-vae-decode holds the real figure on the first clip, and a
  // marker that is at least a term of the truth beats a factor invented
  // to look conservative.
  if (arena != nullptr) { *arena = pcm_b; }
  return true;
}

std::unique_ptr<VideoGenerator>
Ltx25Family::load(const VideoModelCreateArgs& args)
{
  Config cfg;
  std::string err;
  if (!resolve(args.root, cfg, &err, prefer_variant_(&args, args.session))) {
    if (args.session != nullptr) {
      args.session->error(fmt("ltx-2.5: {}", err));
    }
    return nullptr;
  }
  // ---- CORRECT THE DECLARATION, BEFORE ANYTHING SIZES AGAINST IT ----
  //
  // declare_resources() ran in the pre-barrier planning phase, where no
  // beat exists and therefore no `variant` -- so it declared whichever
  // DiT the default preference names. If the config beat then selected a
  // different pack, the graph is now carrying an estimate for a file
  // that will never be opened, and MISSING one for the file that will.
  //
  // MEASURED before this: a run selecting the 23 GB w8g64 pack reported
  // "footprint 41 GB -> PRELOAD", sized against the 39 GB bf16 file it
  // had just declined to load. Harmless on a 64 GB box; on a smaller one
  // that over-count forces an IRREVERSIBLE streaming decision onto a
  // pack that fits comfortably.
  //
  // Done HERE, before plan_streaming reads weight_footprint, because
  // that is the reader whose answer cannot be taken back. Note the
  // asymmetry: the stale key is REVISED (it was declared, so it can be
  // corrected to nothing), while the chosen one is DECLARED (it never
  // was, and revise_declaration deliberately refuses to invent an entry
  // for a checkpoint nobody claimed).
  //
  // THE KEY THAT WAS DECLARED, from the one function that declared it.
  // This used to re-resolve with no variant, which names the preference
  // order's file -- the declared key only when that file is also the
  // largest. Otherwise the revise hit a key nobody had declared, the
  // real estimate stood, and the file actually opened was counted beside
  // it. Asked BEFORE this load records its own file, so it is still the
  // pre-load answer.
  if (args.session != nullptr && args.session->services() != nullptr) {
    auto* mgr = args.session->services()->generative_model_manager();
    if (mgr != nullptr && !cfg.dit_file.empty()) {
      const std::string pre = declared_dit_(args.root);
      if (!pre.empty() && pre != cfg.dit_file) {
        mgr->revise_declaration(pre, 0);
        args.session->log_debug(fmt(
            "ltx-2.5: released the pre-barrier declaration for '{}' -- the "
            "config selected '{}' instead", pre, cfg.dit_file));
      }
      // With its FLOOR, as the claim it stands in for carried one.
      // Declared without it, the floor ledger counts a streamable DiT at
      // its whole size.
      const std::size_t want =
          vpipe::model_memory::dir_weights_bytes(cfg.dit_file);
      if (want > 0) {
        mgr->declare_weights(cfg.dit_file, want, std::string(),
                             std::string(), dit_floor_bytes(cfg.dit_file));
      }
    }
  }

  if (args.session != nullptr) {
    args.session->info(fmt(
        "ltx-2.5: {} DiT, {} layers, video {}d/{}h + audio {}d/{}h, "
        "connector {} layers x {} registers; scheduler {}/{}",
        variant_name(cfg.variant), cfg.dit.num_layers, cfg.dit.inner_dim(),
        cfg.dit.num_attention_heads, cfg.dit.audio_inner_dim(),
        cfg.dit.audio_num_attention_heads, cfg.dit.connector_num_layers,
        cfg.dit.connector_num_learnable_registers,
        cfg.scheduler.class_name, cfg.scheduler.sampler));
    // WHICH PACK, always. Two DiTs of the same variant can differ 3x in
    // footprint, and the only visible difference is this line -- a run
    // that quietly loaded 42 GB of bf16 when the box has a 14 GB w4
    // pack looks identical in every other log.
    args.session->info(fmt("ltx-2.5: DiT weights from {} ({})",
                           cfg.dit_file,
                           cfg.dit_is_dir ? "quantized pack"
                                          : "bf16 single file"));
  }
  if (cfg.enc_file.empty() && args.session != nullptr) {
    args.session->warn(fmt(
        "ltx-2.5: no text encoder under text_encoders/ -- fetch "
        "gemma4-12b-with-proj-ltx-2.5-bf16"));
  }
  if (args.metal == nullptr) {
    if (args.session != nullptr) {
      args.session->error(fmt("ltx-2.5: no metal-compute backend"));
    }
    return nullptr;
  }

  // THE WEIGHT SET, through the manager. Not a private mmap: the
  // manager owns the checkpoint and a model borrows it, so two stages
  // over one file share it and the accounting sees what is resident --
  // which is exactly what a 39 GB checkpoint most needs. The generator
  // HOLDS the shared_ptr for its lifetime; every tensor it bound is an
  // alias into this set.
  std::shared_ptr<vpipe::genai::WeightSet> ws =
      vpipe::genai::open_weight_set(cfg.dit_file, args.session);
  if (!ws) {
    if (args.session != nullptr) {
      args.session->error(fmt("ltx-2.5: cannot open '{}'", cfg.dit_file));
    }
    return nullptr;
  }

  // STREAMING. The stage's verdict is the starting point (it sized the
  // whole graph, not just this model), but the DiT's own numbers refine
  // it: the 48 blocks ARE the 39 GB, and `plan_streaming` is the single
  // rule every DiT family in this tree uses -- copying a sixth variant
  // of it is what that helper exists to prevent.
  bool stream = args.prefer_streaming;

  // THE ADAPTER, declared before the verdict below reads the footprint.
  //
  // A LoRA is opened by the generator on its first request, which is
  // AFTER this model has irrevocably decided whether to stream -- so an
  // adapter the graph configured was a peer of the DiT that the decision
  // never saw, and the distilled-450 file is 8.9 GB. The `lora` key rides
  // the same model-config beat as `variant`, so it is readable here for
  // the same reason `variant` is.
  //
  // Declared rather than named to plan_streaming: its footprint starts
  // from the manager's declarations, where one with no phase counts in
  // every phase, the denoise included. Not refused when it does not
  // resolve -- the generator resolves it again on the first request and
  // says why there.
  if (args.model_config != nullptr && args.session != nullptr &&
      args.session->services() != nullptr) {
    const GenerationParams gp =
        GenerationParams::from_flex(*args.model_config);
    auto* mgr = args.session->services()->generative_model_manager();
    if (!gp.lora.empty() && mgr != nullptr) {
      std::string lerr;
      const std::string lora_file =
          vpipe::resolve_adapter_file(args.session, gp.lora, &lerr);
      const std::size_t lora_bytes =
          vpipe::model_memory::dir_weights_bytes(lora_file);
      if (lora_bytes > 0) {
        mgr->declare_weights(lora_file, lora_bytes);
        args.session->log_debug(fmt(
            "ltx-2.5: declared the LoRA '{}' ({} MB) ahead of the "
            "streaming decision", lora_file, lora_bytes >> 20));
      } else {
        args.session->log_debug(fmt(
            "ltx-2.5: the LoRA '{}' did not resolve at load ({}), so the "
            "streaming decision does not count it", gp.lora, lerr));
      }
    }
  }

  // NAME ONLY WHAT THIS STAGE IS ABOUT TO HOLD. The encoder is
  // `diffusion-conditioner`'s, and it has already declared it -- so the
  // manager's view carries it, at the variant that was really chosen
  // and at what it really holds after streaming its layers.
  //
  // Naming it here instead resolved it a SECOND time, from this
  // family's own resolve, which has the DiT's `variant` but not the
  // conditioner's `encoder_variant` -- so it named the bf16 pack while
  // the conditioner had loaded w8g64. weight_footprint then added the
  // whole of a checkpoint nobody would open, on top of the one that was
  // already accounted. MEASURED on this box: footprint 43868 MB against
  // a true 18821 MB, the difference being a 25 GB phantom encoder, and
  // the DiT streamed a 14.7 GB pack that fits four times over.
  const auto plan = vpipe::model_memory::plan_streaming(
      args.session, cfg.dit_file, /*enc_dir=*/std::string(),
      vpipe::model_memory::kStreamHeadroom);
  stream = stream || plan.stream;
  if (const char* e = std::getenv("VPIPE_LTX25_STREAM")) {
    stream = (std::atoi(e) != 0);
  }
  if (args.session != nullptr) {
    args.session->info(fmt(
        "ltx-2.5: footprint {} GB (peers {} GB) + {} GB headroom -> {}",
        plan.footprint >> 30, plan.others >> 30,
        vpipe::model_memory::kStreamHeadroom >> 30,
        stream ? "STREAM blocks" : "PRELOAD"));
  }

  // WHAT `stream` MEANS HERE. The blocks ARE the checkpoint, and a
  // streaming run keeps a PINNED PREFIX of them resident and reads the
  // rest per forward, dropping each after use -- the same policy, and
  // the same shared `stream_pin_count` rule, as every other streaming
  // DiT in the host tree.
  //
  // It is a prefix rather than a cache on purpose. A forward is a cyclic
  // scan over the stack, so recency predicts nothing: an LRU set of any
  // size is evicted exactly before it comes round again and gives ~0%
  // hits, while a fixed subset gives exactly its share. That is also why
  // the blocks are read `Copied` rather than `Mapped` -- letting the
  // kernel's page cache own their residency is precisely the LRU case,
  // and it degrades with no control on a box that cannot hold the file.
  // The clip the graph plans, so the pinned prefix is sized against the
  // arena that clip will actually need rather than a constant. Zero when
  // the stage could not settle it, which Ltx25Dit reads as "unknown" and
  // falls back on.
  auto g = Ltx25Generator::create(cfg, std::move(ws), args.metal, stream,
                                  args.width, args.height,
                                  args.frames, args.session, &err);
  if (!g) {
    if (args.session != nullptr) {
      args.session->error(fmt("ltx-2.5: {}", err));
    }
    return nullptr;
  }
  // A streaming DiT keeps its pinned prefix, not the checkpoint, so the
  // load-time claim would go on sizing every peer against the whole
  // thing. Peers that then decline to free something are the difference
  // between a run that fits and one that swaps.
  //
  // REVISED TO THE FLOOR -- the trunk, the pinned block and two slots,
  // the same figure the claim declared -- and not to the pinned block
  // alone, which is what this used to say. That left out the trunk and
  // both connectors, several GB that never streams, and the two slots
  // the first forward builds, so the declaration sat below the floor it
  // had just been planned against. What the resident set grows to after
  // that is reported live, through resident_bytes(); the larger of the
  // two is taken here so a revision never reads below what is held.
  if (g->streaming_blocks() && args.session != nullptr) {
    auto* mgr = args.session->services()->generative_model_manager();
    if (mgr != nullptr) {
      const std::size_t held =
          std::max(dit_floor_bytes(cfg.dit_file),
                   (std::size_t)g->resident_bytes());
      mgr->revise_declaration(cfg.dit_file, held);
      args.session->info(fmt(
          "ltx-2.5: streaming keeps {} of {} blocks resident ({} MB); "
          "declaration revised from {} MB to {} MB",
          g->pinned_blocks(), cfg.dit.num_layers,
          g->pinned_weight_bytes() >> 20,
          vpipe::model_memory::dir_weights_bytes(cfg.dit_file) >> 20,
          held >> 20));
    }
  }
  // THE FILE THIS OPENED, for the planning calls -- see declared_dit_().
  // The generator owns the lease, so the record answers for exactly as
  // long as the checkpoint is held, and generate-video's re-read of
  // declare_holdings() right after this returns names this file.
  {
    auto lease = std::make_shared<char>(0);
    g->set_family_lease(lease);
    std::lock_guard<std::mutex> lk(_loaded_mu);
    _loaded[args.root] = Loaded{cfg.dit_file, lease};
  }
  return g;
}

}  // namespace ltx25
