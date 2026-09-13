#include "ltx25-upscale-stage.h"
#include "ltx25-config.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"
#include "generative-models/generative-model-manager.h"
#include "stages/model-memory.h"
#include "stages/model-registry.h"

#include <cstdint>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

using vpipe::ConfigKey;
using vpipe::ConfigType;
using vpipe::FlexData;
using vpipe::InEdge;
using vpipe::PortSpec;
using vpipe::SessionContextIntf;
using vpipe::StageCategory;
using vpipe::StageSpec;
using vpipe::TensorBeat;
using vpipe::TensorBeatPayload;
using vpipe::fmt;
using vpipe::genai::WeightSet;

namespace ltx25 {

namespace {

const ConfigKey kAttrs[] = {
  {.key = "hf_dir", .type = ConfigType::String, .required = false,
   .doc = "the LTX-2.5 model dir. The upscalers are read from "
          "<hf_dir>/latent_upscale_models/ and the per-channel statistics "
          "they need from <hf_dir>/vae -- both, because the upscaler's own "
          "checkpoint carries no statistics and works in the VAE's "
          "un-normalized latent space",
   // A picker of its own, filtered to the UPSCALER entry rather than to
   // the family: `ltx-2.5` records name the same directory, but only
   // this one's presence says latent_upscale_models/ was actually
   // fetched -- they are a separate download. Without the allow-list the
   // browser would offer nothing at all: the entry carries a
   // parent_model_type, so it is a SUPPLEMENT, and an untyped model
   // field shows plain models only.
   //
   // NOT on the "diffusion-model" channel, deliberately. A channel
   // consumer contributes its types to what model-select offers, and an
   // upscaler-only record is not something generate-video can run.
   // The other half of what this stage needs -- a VAE beside the
   // upscalers -- no model_type filter can express, so it stays the
   // runtime check that names what is missing.
   .suggest_db = vpipe::kModelRegistryDb,
   .suggest_db_type = "ltx-2.5-upscaler"},
  {.key = "mode", .type = ConfigType::String, .required = false,
   .doc = "'spatial' (the default) doubles height and width and leaves the "
          "frame count alone; 'temporal' doubles the frames and leaves the "
          "picture alone. A temporal pass emits 2F-1 latent frames, not "
          "2F: the first latent frame encodes a single pixel frame, so the "
          "shuffle's first output frame is dropped -- the same asymmetry "
          "the VAE decoder has. Chain two stages to do both",
   .def_str = "spatial"},
  {.key = "unload_when_idle", .type = ConfigType::String, .required = false,
   .doc = "what happens to the upscaler between clips. The same "
          "vocabulary every model-holding stage takes. \"auto\" (the "
          "default) resolves from the box and lands on park here. "
          "\"park\" lets the model go and hands its checkpoint to the "
          "kernel as purgeable: the pages survive unless something else "
          "needs the RAM, so the next clip rebuilds without a disk read "
          "if nothing took them. \"destroy\" gives the bytes back and "
          "pays a full reload. \"keep\" holds the model and its scratch "
          "pinned. The legacy always|never and a legacy BOOL are "
          "accepted. Anything but keep also releases the scratch -- the "
          "im2col and the two working planes, which scale with the clip "
          "and are the part worth giving back to a denoise that follows",
   .def_str = "auto"},
};

const PortSpec kIports[] = {
  {.name = "latent",
   .doc = "f32 [z, T, h, w] -- a whitened video latent, as generate-video "
          "emits it and as vae-decode consumes it",
   .type = &typeid(TensorBeatPayload), .tags = "latent", .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "latent",
   .doc = "the upscaled latent, same z and the same whitened space. "
          "spatial: [z, T, 2h, 2w]; temporal: [z, 2T-1, h, w]",
   .type = &typeid(TensorBeatPayload), .tags = "latent", .clock_group = 0},
};

const StageSpec kSpec = {
  .type_name = Ltx25UpscaleStage::kTypeName,
  .doc       = "LTX-2.5 latent upscaler: x2 spatial or x2 temporal, in "
               "LATENT space. The second half of the two-stage pipeline -- "
               "generate small, upscale the latent, denoise again larger -- "
               "with no VAE decode and re-encode in between.",
  .display_name = "LTX-2.5 Latent Upscale",
  .category  = StageCategory::Generative,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

}  // namespace

Ltx25UpscaleStage::Ltx25UpscaleStage(const SessionContextIntf* session,
                                     std::string id,
                                     std::vector<InEdge> iports,
                                     FlexData config)
  : vpipe::TypedStage<Ltx25UpscaleStage>(session, std::move(id),
                                         std::move(iports),
                                         std::move(config))
{
  // The oports have to be ALLOCATED before any early return: a stage
  // that skips this reports ZERO oports to pipeline_from_spec, and a
  // bad-config refusal would then surface as an out-of-range edge
  // instead of the message that says what is actually wrong.
  allocate_oports(kSpec.oports.size());

  _hf_dir = attr_str("hf_dir");
  _mode = attr_str("mode");
  if (_mode.empty()) { _mode = "spatial"; }
  // ACCEPTS THE STRING VOCABULARY AND A LEGACY BOOL, the way the
  // conditioner beside it does. This key was a bool here -- `true` meant
  // "drop the scratch", which is the cheap half of what park does -- and
  // shipped pipeline JSON says so. Reading that bool as a string would
  // resolve every such graph to "auto" and quietly change what it does,
  // so both spellings are read and `true` lands on the policy closest to
  // what it was documented to mean.
  const FlexData& c = this->config();
  if (c.is_object()) {
    const auto o = c.as_object();
    if (o.contains("unload_when_idle")) {
      const FlexData& v = o.at("unload_when_idle");
      if (v.is_bool()) {
        _unload_cfg = v.as_bool(false)
                          ? vpipe::model_memory::UnloadPolicy::kPark
                          : vpipe::model_memory::UnloadPolicy::kKeep;
      } else {
        bool bad = false;
        _unload_cfg = vpipe::model_memory::parse_unload_policy(
            std::string(v.as_string("auto")), &bad);
        if (bad) {
          fail_config(fmt(
              "Ltx25UpscaleStage('{}'): unload_when_idle must be "
              "auto|destroy|park|keep (or the legacy always|never|"
              "true|false); got '{}'",
              this->id(), std::string(v.as_string(""))));
          return;
        }
      }
    }
  }
  if (_mode != "spatial" && _mode != "temporal") {
    fail_config(fmt("Ltx25UpscaleStage('{}'): mode must be 'spatial' or "
                    "'temporal' (got '{}')", this->id(), _mode));
  }
}

Ltx25UpscaleStage::~Ltx25UpscaleStage() = default;

const StageSpec&
Ltx25UpscaleStage::spec() const noexcept
{
  return kSpec;
}

const StageSpec*
Ltx25UpscaleStage::stage_spec() noexcept
{
  return &kSpec;
}

std::string
Ltx25UpscaleStage::model_root_() const
{
  return vpipe::resolve_model_dir(this->session(), _hf_dir);
}

std::string
Ltx25UpscaleStage::checkpoint_(std::string* err) const
{
  namespace fs = std::filesystem;
  const std::string root = model_root_();
  const fs::path dir = fs::path(root) / "latent_upscale_models";
  std::error_code ec;
  if (!fs::is_directory(dir, ec)) {
    if (err != nullptr) {
      *err = fmt("no latent_upscale_models/ under '{}'. The upscalers are a "
                 "SEPARATE download from the DiT -- fetch the "
                 "'LTX-2.5-latent-upscalers' catalogue entry", root)();
    }
    return {};
  }
  // Matched on the MODE rather than on a pinned filename, so a renamed
  // or re-versioned checkpoint still resolves.
  const std::string want = _mode;
  std::string found;
  for (const auto& e : fs::directory_iterator(dir, ec)) {
    if (!e.is_regular_file()) { continue; }
    const std::string n = e.path().filename().string();
    if (n.size() < 12 || n.rfind(".safetensors") == std::string::npos) {
      continue;
    }
    if (n.find(want) != std::string::npos) {
      if (!found.empty()) {
        if (err != nullptr) {
          *err = fmt("two {} upscalers in '{}' ('{}' and '{}'); this cannot "
                     "choose", want, dir.string(),
                     fs::path(found).filename().string(), n)();
        }
        return {};
      }
      found = e.path().string();
    }
  }
  if (found.empty() && err != nullptr) {
    *err = fmt("no '{}' upscaler in '{}'", want, dir.string())();
  }
  return found;
}

std::vector<vpipe::ResourceClaim>
Ltx25UpscaleStage::declare_resources() const
{
  // Declared so a peer sizing against this graph sees the upscaler.
  // ~950 MB for the spatial model, ~260 MB for the temporal one -- small
  // beside the DiT, but a stage that declares nothing is invisible
  // exactly when the box is tight.
  std::string e;
  const std::string ck = checkpoint_(&e);
  if (ck.empty()) { return {}; }
  return vpipe::model_memory::weight_claims({ck});
}

// The topological ledger. The SAME key the claim above names and
// ensure_loaded_() opens, so the manager's ledger, the plan and the
// release in destroy_model_() / park_model_() all describe one
// checkpoint under one name. Without it this stage was a hole in the
// plan, which reads as room that is not there.
//
// No floor: the upscaler does not stream, so the least it can be held
// at is all of it.
//
// `releases` and `reclaimable` answer the questions the conditioner's
// declare_memory() answers, and reach a different answer on `park` for
// the reason resolve_idle_policy_() gives. This model binds every conv
// through WeightSet::tensor() and release_model_() drops the borrow
// before parking, so a park genuinely hands the bytes back -- where the
// conditioner's uncached Gemma parks nothing. `auto` lands on park here,
// so it is reclaimable for the same reason.
vpipe::StageMemory
Ltx25UpscaleStage::declare_memory() const
{
  vpipe::StageMemory m;
  std::string e;
  const std::string ck = checkpoint_(&e);
  if (ck.empty()) { return m; }
  namespace mm = vpipe::model_memory;
  m.hold(ck, mm::dir_weights_bytes(ck), /*floor=*/0,
         _unload_cfg == mm::UnloadPolicy::kDestroy,
         _unload_cfg == mm::UnloadPolicy::kAuto ||
             _unload_cfg == mm::UnloadPolicy::kPark);
  return m;
}

bool
Ltx25UpscaleStage::ensure_loaded_()
{
  if (_model) { return true; }
  if (_tried) { return false; }
  _tried = true;

  std::string e;
  const std::string ck = checkpoint_(&e);
  if (ck.empty()) {
    session()->warn(fmt("Ltx25UpscaleStage('{}'): {}", id(), e));
    return false;
  }
  if (!UpscalerConfig::from_metadata(ck, &_cfg, &e)) {
    session()->warn(fmt("Ltx25UpscaleStage('{}'): {}", id(), e));
    return false;
  }
  // The MODE the graph asked for has to be the mode the file does. A
  // spatial checkpoint under a temporal name would otherwise run and
  // return the wrong shape.
  const bool want_spatial = _mode == "spatial";
  if (_cfg.spatial_upsample != want_spatial
      || _cfg.temporal_upsample == want_spatial) {
    session()->warn(fmt(
        "Ltx25UpscaleStage('{}'): '{}' is a {} upscaler but mode is '{}'",
        id(), ck, _cfg.spatial_upsample ? "spatial" : "temporal", _mode));
    return false;
  }

  auto* mc = session()->services()->metal_compute();
  if (mc == nullptr) {
    session()->warn(fmt("Ltx25UpscaleStage('{}'): no metal-compute", id()));
    return false;
  }
  _ops = std::make_unique<MetalOps>();
  if (!_ops->init(mc, &e)) {
    session()->warn(fmt("Ltx25UpscaleStage('{}'): ops: {}", id(), e));
    _ops.reset();
    return false;
  }

  // THE VAE'S STATISTICS, which this checkpoint does not carry. Without
  // them the model runs on a latent scaled differently from the one it
  // was trained on and returns a plausible tensor, so a missing VAE is
  // an error rather than an identity fallback.
  namespace fs = std::filesystem;
  std::vector<float> sv, mv;
  {
    const fs::path vdir = fs::path(model_root_()) / "vae";
    std::error_code ec;
    std::string vfile;
    if (fs::is_directory(vdir, ec)) {
      for (const auto& en : fs::directory_iterator(vdir, ec)) {
        const std::string n = en.path().filename().string();
        if (n.rfind(".safetensors") != std::string::npos
            && n.find("video") != std::string::npos) {
          vfile = en.path().string();
          break;
        }
      }
    }
    if (vfile.empty()) {
      session()->warn(fmt(
          "Ltx25UpscaleStage('{}'): no video VAE under '{}'. The upscaler "
          "needs its per-channel statistics -- it works in the VAE's "
          "un-normalized latent space and the graph's latents are whitened",
          id(), vdir.string()));
      return false;
    }
    auto vws = vpipe::genai::open_weight_set(vfile, session());
    if (!vws) {
      session()->warn(fmt("Ltx25UpscaleStage('{}'): cannot open '{}'", id(),
                          vfile));
      return false;
    }
    auto stat = [&](const char* name, std::vector<float>& dst) {
      const auto* info = vws->src().info(name);
      if (info == nullptr) { return; }
      const auto b = vws->read(name, mc, WeightSet::Residency::Copied);
      if (b.empty() || info->shape.size() != 1) { return; }
      const std::size_t n = (std::size_t)info->shape[0];
      dst.assign(n, 0.0f);
      if (info->dtype == "F32") {
        std::memcpy(dst.data(), b.contents(), n * 4);
      } else if (info->dtype == "BF16") {
        const auto* p = static_cast<const std::uint16_t*>(b.contents());
        for (std::size_t i = 0; i < n; ++i) {
          const std::uint32_t u = (std::uint32_t)p[i] << 16;
          std::memcpy(&dst[i], &u, 4);
        }
      }
    };
    stat("per_channel_statistics.std-of-means", sv);
    stat("per_channel_statistics.mean-of-means", mv);
    if ((int)sv.size() != _cfg.in_channels
        || (int)mv.size() != _cfg.in_channels) {
      session()->warn(fmt(
          "Ltx25UpscaleStage('{}'): '{}' carries no usable per-channel "
          "statistics for {} channels", id(), vfile, _cfg.in_channels));
      return false;
    }
  }

  auto ws = vpipe::genai::open_weight_set(ck, session());
  if (!ws) {
    session()->warn(fmt("Ltx25UpscaleStage('{}'): cannot open '{}'", id(),
                        ck));
    return false;
  }
  _model = Ltx25Upscaler::load(_cfg, sv, mv, ws, *_ops, &e);
  if (!_model) {
    session()->warn(fmt("Ltx25UpscaleStage('{}'): load: {}", id(), e));
    return false;
  }
  _ckpt = ck;
  session()->info(fmt(
      "Ltx25UpscaleStage('{}'): {} x2 latent upscaler from '{}' -- "
      "{} channels, mid {}, {} blocks per stage, {:.0f} MB",
      id(), _mode, ck, _cfg.in_channels, _cfg.mid_channels,
      _cfg.num_blocks_per_stage,
      (double)_model->resident_bytes() / (1024.0 * 1024.0)));
  return true;
}

vpipe::Job
Ltx25UpscaleStage::process(vpipe::RuntimeContext& ctx)
{
  auto in = co_await ctx.read(0);
  if (!in) { ctx.signal_done(); co_return; }
  const auto* tbp = dynamic_cast<const TensorBeatPayload*>(in.get());
  if (tbp == nullptr || tbp->shape.size() != 4
      || tbp->dtype != TensorBeat::DType::F32) {
    session()->warn(fmt(
        "Ltx25UpscaleStage('{}'): expected an f32 [z, T, h, w] latent, got "
        "{}; skipping", id(), in->describe()));
    co_return;
  }
  if (!ensure_loaded_()) { co_return; }

  const int C = (int)tbp->shape[0], F = (int)tbp->shape[1];
  const int H = (int)tbp->shape[2], W = (int)tbp->shape[3];
  if (C != _cfg.in_channels) {
    session()->warn(fmt(
        "Ltx25UpscaleStage('{}'): the latent has {} channels but this "
        "upscaler takes {}; it was made by a different VAE. Skipping",
        id(), C, _cfg.in_channels));
    co_return;
  }

  const auto src = tbp->materialize_contiguous();
  std::vector<float> out;
  std::array<int, 4> shape{};
  std::string e;
  if (!_model->upscale(reinterpret_cast<const float*>(src.data()), F, H, W,
                       &out, &shape, &e)) {
    session()->warn(fmt("Ltx25UpscaleStage('{}'): {}", id(), e));
    co_return;
  }

  // The two distributions, because the ONE thing that cannot be seen
  // from a shape is whether the latent came back in the space it went
  // in. Both are whitened, so both should sit near zero mean and unit
  // scale; a factor between them is the un-normalize / re-normalize pair
  // composing the wrong way.
  {
    const float* ip = reinterpret_cast<const float*>(src.data());
    const std::size_t ni = (std::size_t)C * F * H * W;
    double im = 0.0, iv = 0.0, om = 0.0, ov = 0.0;
    for (std::size_t i = 0; i < ni; ++i) { im += ip[i]; }
    im /= (double)ni;
    for (std::size_t i = 0; i < ni; ++i) {
      const double d = ip[i] - im; iv += d * d;
    }
    iv = std::sqrt(iv / (double)ni);
    for (float v : out) { om += v; }
    om /= (double)out.size();
    for (float v : out) { const double d = v - om; ov += d * d; }
    ov = std::sqrt(ov / (double)out.size());
    // SPATIAL variance specifically: a latent can have a healthy overall
    // std while being CONSTANT across cells, with all the variance
    // sitting between channels -- and that decodes to a flat field.
    auto spatial = [](const float* p, int c, std::size_t cells) {
      double acc = 0.0;
      for (int ch = 0; ch < c; ++ch) {
        const float* q = p + (std::size_t)ch * cells;
        double m = 0.0;
        for (std::size_t i = 0; i < cells; ++i) { m += q[i]; }
        m /= (double)cells;
        double v = 0.0;
        for (std::size_t i = 0; i < cells; ++i) {
          const double d = q[i] - m; v += d * d;
        }
        acc += std::sqrt(v / (double)cells);
      }
      return acc / (double)c;
    };
    const double is = spatial(ip, C, (std::size_t)F * H * W);
    const double os = spatial(out.data(), C,
                              (std::size_t)shape[1] * shape[2] * shape[3]);
    session()->log_debug(fmt(
        "Ltx25UpscaleStage('{}'): latent in mean {:.3f} std {:.3f} "
        "spatial-std {:.3f} -> out mean {:.3f} std {:.3f} spatial-std {:.3f}",
        id(), im, iv, is, om, ov, os));
  }

  auto beat = std::make_unique<TensorBeatPayload>();
  beat->dtype = TensorBeat::DType::F32;
  beat->shape = {shape[0], shape[1], shape[2], shape[3]};
  beat->sideband = tbp->sideband;
  beat->resize_contiguous(out.size());
  std::memcpy(beat->as_f32(), out.data(), out.size() * sizeof(float));
  ++_emitted;
  session()->log_debug(fmt(
      "Ltx25UpscaleStage('{}'): latent #{} [{},{},{},{}] -> [{},{},{},{}]",
      id(), _emitted, C, F, H, W, shape[0], shape[1], shape[2], shape[3]));
  release_when_idle_();
  co_await ctx.write(0, std::move(beat));
}

// ---------------------------------------------------------------------
// The idle policy.
//
// Resolved at the first beat rather than in the ctor, for the reason
// docs/MODEL-MEMORY.md gives: at construction the graph is half loaded
// and a stage would size against peers that have not appeared yet.
// ---------------------------------------------------------------------
void
Ltx25UpscaleStage::resolve_idle_policy_()
{
  if (_idle_resolved) { return; }
  _idle_resolved = true;

  namespace mm = vpipe::model_memory;
  switch (_unload_cfg) {
    case mm::UnloadPolicy::kDestroy:
    case mm::UnloadPolicy::kPark:
    case mm::UnloadPolicy::kKeep:
      _idle_action = _unload_cfg;
      break;
    default:
      // PARK is the default answer here, and the conditioner's opposite
      // choice next door is not an inconsistency -- it is the same rule
      // reaching a different answer because the two models are held
      // differently.
      //
      // Parking walks a weight SET's cached tensors. This model binds
      // every conv and norm through WeightSet::tensor(), so they ARE
      // cached entries with a source name and parking genuinely hands
      // the bytes back; the conditioner's Gemma reads uncached into its
      // own members, where the registry cannot see them, so parking it
      // reports 0 and it has to destroy instead.
      //
      // And nothing sizes an irreversible decision against this model.
      // It is under a gigabyte, it loads after the denoise it would have
      // competed with, and its reload is one file -- so the cheaper,
      // reversible half of the trade is the right default.
      _idle_action = mm::UnloadPolicy::kPark;
      break;
  }
  this->session()->log_debug(fmt(
      "Ltx25UpscaleStage('{}'): unload_when_idle={} -> {}", this->id(),
      mm::unload_policy_name(_unload_cfg),
      mm::unload_policy_name(_idle_action)));
}

// BOTH POLICIES DROP THE MODEL FIRST, and that is not an implementation
// detail -- it is the manager's rule. Nothing parks or frees a BORROWED
// set: a borrower holds aliases of the very buffers a park makes
// purgeable and reads them in a forward pass that asks the set for
// nothing, so the manager refuses rather than hand a live model pages
// the kernel may discard. This model binds every conv through
// WeightSet::tensor() and keeps the set alive for its own lifetime, as
// the weight contract requires, so it IS that borrower.
//
// MEASURED, and this is why the first version of these two was wrong:
// asking to park while the model was still alive logged "still borrowed
// by 1 model(s), so it is not parked" and reclaimed 0 MB. A park that
// frees nothing looks exactly like a park that worked.
//
// `_tried` is cleared with the model on both paths. That flag exists to
// stop a FAILED load being retried every beat, and letting go on purpose
// is not a failure.
void
Ltx25UpscaleStage::release_model_()
{
  _model.reset();
  _ops.reset();
  _tried = false;
}

// Give the bytes back. The next clip pays a full reload from disk.
void
Ltx25UpscaleStage::destroy_model_()
{
  release_model_();
  auto* mgr = this->session()->services()->generative_model_manager();
  if (mgr == nullptr || _ckpt.empty()) { return; }
  const std::size_t got = mgr->drop_weights(_ckpt);
  this->session()->log_debug(fmt(
      "Ltx25UpscaleStage('{}'): dropped the upscaler, {} MB back; the "
      "next clip reloads it", this->id(), got >> 20));
}

// Hand the checkpoint to the kernel as purgeable instead. The pages
// survive unless something else needs the RAM, so the next clip rebuilds
// the model without a disk read if nothing took them -- never worse than
// destroy, and cheaper whenever the box was not actually short.
//
// NO declaration revision on purpose: parked bytes are still HELD bytes
// for the accounting, and they return to the box only if the kernel gets
// round to reclaiming them. A peer deciding this step whether it can
// hold a block must not be told room exists on that basis.
void
Ltx25UpscaleStage::park_model_()
{
  release_model_();
  auto* mgr = this->session()->services()->generative_model_manager();
  if (mgr == nullptr || _ckpt.empty()) { return; }
  const std::size_t got = mgr->park_weights(_ckpt);
  this->session()->log_debug(fmt(
      "Ltx25UpscaleStage('{}'): parked {} MB of the upscaler", this->id(),
      got >> 20));
}

void
Ltx25UpscaleStage::release_when_idle_()
{
  resolve_idle_policy_();
  switch (_idle_action) {
    case vpipe::model_memory::UnloadPolicy::kDestroy:
      destroy_model_();
      break;
    case vpipe::model_memory::UnloadPolicy::kPark:
      park_model_();
      break;
    default: break;                     // kKeep: hold it, scratch and all
  }
}

}  // namespace ltx25
