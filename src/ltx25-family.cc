#include "ltx25-family.h"
#include "ltx25-config.h"
#include "ltx25-generator.h"

#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "generative-models/generative-model-manager.h"
#include "interfaces/session-services-intf.h"
#include "stages/model-memory.h"

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>

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

std::vector<ResourceClaim>
Ltx25Family::declare_resources(const std::string& root) const
{
  // Declared BEFORE any driver starts, which is the only moment every
  // peer can size itself against this checkpoint. A DiT reported as 0
  // bytes is exactly the silent under-count the planning phase exists to
  // prevent, so claim the FILES that will actually be read rather than
  // the directory.
  Config cfg;
  std::string err;
  if (!resolve(root, cfg, &err, prefer_variant_(nullptr, nullptr))) {
    // Nothing resolved. Claim the root so the checkpoint is at least
    // visible; an unresolvable root will fail loudly at load anyway.
    return vpipe::model_memory::weight_claims({root});
  }
  // THE LARGEST candidate, not the preferred one.
  //
  // No beat exists yet, so the `variant` that will actually choose is
  // unknowable here (see the correction in load()). Between guessing and
  // guessing SAFELY, the direction matters: over-declaring costs a peer
  // some unnecessary caution, which is recoverable; under-declaring
  // tells the graph there is room that is not there, and every stage
  // sizing against it loads. The preference order happens to name the
  // bf16 file today, which is also the biggest -- but a root holding
  // only quantized packs would have it pick one arbitrarily, and picking
  // w4 while the run loads w8 is the dangerous direction.
  std::vector<std::string> dirs;
  if (!cfg.dit_file.empty()) {
    std::string biggest = cfg.dit_file;
    std::size_t most = vpipe::model_memory::dir_weights_bytes(biggest);
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path dm = fs::path(root) / "diffusion_models";
    for (const auto& de : fs::directory_iterator(dm, ec)) {
      if (ec) { break; }
      const std::size_t b =
          vpipe::model_memory::dir_weights_bytes(de.path().string());
      if (b > most) { most = b; biggest = de.path().string(); }
    }
    dirs.push_back(biggest);
  }
  // The text encoder is NOT claimed here. It is loaded by
  // `diffusion-conditioner`, a different stage, which declares it
  // itself -- claiming it twice would size every peer against 24 GB
  // that this stage never holds.
  return vpipe::model_memory::weight_claims(dirs);
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
  if (args.session != nullptr && args.session->services() != nullptr) {
    auto* mgr = args.session->services()->generative_model_manager();
    if (mgr != nullptr && !cfg.dit_file.empty()) {
      Config base;
      if (resolve(args.root, base, nullptr, {}) && !base.dit_file.empty() &&
          base.dit_file != cfg.dit_file) {
        mgr->revise_declaration(base.dit_file, 0);
        args.session->log_debug(fmt(
            "ltx-2.5: released the pre-barrier declaration for '{}' -- the "
            "config selected '{}' instead", base.dit_file, cfg.dit_file));
      }
      const std::size_t want =
          vpipe::model_memory::dir_weights_bytes(cfg.dit_file);
      if (want > 0) { mgr->declare_weights(cfg.dit_file, want); }
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
  const auto plan = vpipe::model_memory::plan_streaming(
      args.session, cfg.dit_file, cfg.enc_file,
      vpipe::model_memory::kStreamHeadroom);
  stream = stream || plan.stream;
  if (const char* e = std::getenv("VPIPE_LTX25_STREAM")) {
    stream = (std::atoi(e) != 0);
  }
  // THE PINNED-PREFIX FRACTION, from the same plan. Every built-in DiT in
  // the host tree threads this; this port used to drop it and pass a
  // hardcoded 0.60 down instead, which is the whole reason a 16 GB box
  // pinned 20 of 48 blocks against a plan that had computed room for
  // none. It is a property of the GRAPH -- what else is resident during
  // the denoise -- so the model cannot derive it.
  //
  // Overridable in the same shape as VPIPE_LTX25_STREAM beside it: a
  // forced `stream=0` makes the fraction meaningless (nothing streams, so
  // there is no prefix), and a forced `stream=1` on a roomy box wants a
  // way to ask for a real prefix anyway.
  double pin_frac = plan.pin_frac;
  if (std::getenv("VPIPE_LTX25_STREAM") != nullptr && !stream) {
    pin_frac = 0.0;
  }
  if (const char* e = std::getenv("VPIPE_LTX25_PIN_FRAC")) {
    pin_frac = std::atof(e);
  }
  if (args.session != nullptr) {
    args.session->info(fmt(
        "ltx-2.5: footprint {} GB (peers {} GB) + {} GB headroom -> {} "
        "(pin_frac {:.3f})",
        plan.footprint >> 30, plan.others >> 30,
        vpipe::model_memory::kStreamHeadroom >> 30,
        stream ? "STREAM blocks" : "PRELOAD", pin_frac));
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
                                  pin_frac, args.width, args.height,
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
  if (g->streaming_blocks() && args.session != nullptr) {
    auto* mgr = args.session->services()->generative_model_manager();
    if (mgr != nullptr) {
      const std::size_t held = g->pinned_weight_bytes();
      mgr->revise_declaration(cfg.dit_file, held);
      args.session->info(fmt(
          "ltx-2.5: streaming keeps {} of {} blocks resident ({} MB); "
          "declaration revised from {} MB",
          g->pinned_blocks(), cfg.dit.num_layers, held >> 20,
          vpipe::model_memory::dir_weights_bytes(cfg.dit_file) >> 20));
    }
  }
  return g;
}

}  // namespace ltx25
