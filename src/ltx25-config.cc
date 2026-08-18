#include "ltx25-config.h"

#include "common/flex-data.h"
#include "generative-models/shared/comfy-checkpoint.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using vpipe::FlexData;

namespace ltx25 {

namespace {

std::string
lower_(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return (char)std::tolower(c); });
  return s;
}

// Read one key off a JSON object, leaving `out` alone when it is absent.
// "Leave alone" and not "write the default" on purpose: the struct is
// already initialised to LTX-2.5's shipped values, so a missing key means
// "as shipped" -- and a checkpoint that DOES say something always wins.
void
get_int_(const FlexData::ConstObjectView& o, const char* k, int& out)
{
  if (o.contains(k)) { out = (int)o.at(k).as_int(out); }
}

void
get_real_(const FlexData::ConstObjectView& o, const char* k, double& out)
{
  if (o.contains(k)) { out = o.at(k).as_real(out); }
}

void
get_bool_(const FlexData::ConstObjectView& o, const char* k, bool& out)
{
  if (o.contains(k)) { out = o.at(k).as_bool(out); }
}

void
get_str_(const FlexData::ConstObjectView& o, const char* k, std::string& out)
{
  if (o.contains(k)) { out = std::string(o.at(k).as_string(out)); }
}

// An int array, e.g. positional_embedding_max_pos [20, 2048, 2048]. A
// present-but-empty array is taken as written -- the model has one-axis
// audio positions, so an empty list is a real answer, not a miss.
void
get_int_vec_(const FlexData::ConstObjectView& o, const char* k,
             std::vector<int>& out)
{
  if (!o.contains(k)) { return; }
  const FlexData& v = o.at(k);
  if (!v.is_array()) { return; }
  const auto arr = v.as_array();
  std::vector<int> got;
  got.reserve(arr.size());
  for (std::size_t i = 0; i < arr.size(); ++i) {
    got.push_back((int)arr.at(i).as_int(0));
  }
  out = std::move(got);
}

// `_class_name` as the checkpoint spells it. The quantized-directory
// scan matches on it, and parse_dit_config checks it, so it is written
// once.
constexpr const char* kDitClassName = "AVTransformer3DModel";

bool
read_file_(const std::string& path, std::string* out)
{
  std::ifstream f(path, std::ios::binary);
  if (!f) { return false; }
  out->assign(std::istreambuf_iterator<char>(f),
              std::istreambuf_iterator<char>());
  return true;
}

// The candidate DiT in `root`, preferring the wanted variant. The
// component resolver already skips packings this build cannot read
// (int8_convrot, nvfp4), so what comes back is what can be opened.
std::string
resolve_dit_(const std::string& root, const std::string& prefer_variant,
             std::string* err)
{
  // Distilled first when nothing was asked for: it is the checkpoint
  // this port is built and verified against, and it is also the one a
  // 64 GB box can actually sample in 8 steps.
  std::vector<std::string> prefer;
  if (prefer_variant == "dev") {
    prefer = {"dev"};
  } else if (prefer_variant == "distilled" || prefer_variant.empty()) {
    prefer = {"distilled", "dev"};
  } else {
    prefer = {prefer_variant, "distilled", "dev"};
  }

  // TWO shapes of DiT can sit under diffusion_models/, and they compete
  // on the SAME preference list:
  //
  //   * a Comfy single FILE, config in its `__metadata__`; and
  //   * a DIRECTORY of shards plus a synthesized config.json, which is
  //     what `model-quantize` writes -- resolve_component cannot see it,
  //     because it looks for a file with the right metadata key.
  //
  // Collecting both before choosing is the point. An earlier version
  // returned the single file whenever one existed and only then looked
  // for directories, which meant a repo holding BOTH a bf16 file and a
  // quantized pack silently always loaded the bf16 one -- a 4x footprint
  // difference decided by resolution order, with nothing in the log
  // saying so.
  //
  // A THIRD shape, which is the same directory checkpoint one level up:
  // `model-quantize`'s self-contained output replaces the whole
  // `diffusion_models/` role with the quantized DiT, so the config.json
  // is AT the role dir rather than in a pack inside it. That layout has
  // no bf16 file beside it to choose against -- it is the only DiT in
  // the model -- so taking it silently is not the precision surprise the
  // rule below guards against; there is no choice being made.
  namespace fs = std::filesystem;
  std::error_code ec;
  const fs::path dm = fs::path(root) / "diffusion_models";
  auto is_dit_ckpt = [](const fs::path& dir) {
    std::string txt;
    if (!read_file_((dir / "config.json").string(), &txt)) { return false; }
    FlexData c;
    try { c = FlexData::from_json(txt); } catch (...) { return false; }
    if (!c.is_object()) { return false; }
    auto co = c.as_object();
    if (!co.contains("transformer")) { return false; }
    const FlexData t = co.at("transformer");
    auto to = t.as_object();
    const FlexData cn = to.contains("_class_name") ? to.at("_class_name")
                                                   : FlexData();
    return std::string(cn.as_string("")) == kDitClassName;
  };
  std::vector<std::string> cands;
  if (is_dit_ckpt(dm)) { cands.push_back(dm.string()); }
  for (const auto& de : fs::directory_iterator(dm, ec)) {
    if (ec) { break; }
    if (!de.is_directory()) { continue; }
    if (!is_dit_ckpt(de.path())) { continue; }
    cands.push_back(de.path().string());
  }

  // An explicitly ASKED-FOR variant wins outright, whichever shape it is
  // in -- that is what asking means. `prefer_variant` is how a config
  // says "the w4 pack", since the bit width is in the pack's name.
  if (!prefer_variant.empty() && prefer_variant != "distilled" &&
      prefer_variant != "dev") {
    for (const std::string& c : cands) {
      if (c.find(prefer_variant) != std::string::npos) { return c; }
    }
  }

  const std::string f = vpipe::genai::comfy::resolve_component(
      root, "diffusion_models", kDitMetaKey, prefer);
  if (!f.empty()) { return f; }

  for (const std::string& want : prefer) {
    for (const std::string& c : cands) {
      if (c.find(want) != std::string::npos) { return c; }
    }
  }
  if (!cands.empty()) { return cands.front(); }

  if (err != nullptr) {
    *err = "no readable LTX-2.5 DiT under '" + root +
           "/diffusion_models' (the int8-convrot and nvfp4 packings are "
           "ComfyUI-only and are deliberately not read)";
  }
  return {};
}

}  // namespace

int
align_num_frames(int frames)
{
  if (frames <= 1) { return 1; }
  // frames % 8 == 1, rounded UP.
  const int k = (frames - 1 + kTemporalCompression - 1) / kTemporalCompression;
  return k * kTemporalCompression + 1;
}

int
align_dimension(int px)
{
  if (px <= 0) { return kSpatialCompression; }
  return ((px + kSpatialCompression - 1) / kSpatialCompression)
         * kSpatialCompression;
}

const std::vector<double>&
distilled_sigmas()
{
  // Reference DISTILLED_SIGMA_VALUES. Descending, 1.0 = pure noise --
  // the opposite of Boogu's inverted convention, so do not "fix" it.
  static const std::vector<double> kS = {
      1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0};
  return kS;
}

const std::vector<double>&
stage2_distilled_sigmas()
{
  static const std::vector<double> kS = {0.909375, 0.725, 0.421875, 0.0};
  return kS;
}

Variant
variant_of_filename(const std::string& file)
{
  const fs::path p(file);
  std::string n = lower_(p.filename().string());
  // A SELF-CONTAINED pack names the variant on its ROOT, not on the
  // component. `model-quantize` writes `<name>/diffusion_models/`, so the
  // leaf here is the role directory and says nothing -- which is how a
  // pack called `LTX-2.5-distilled-8bit` came out kUnknown, and with it
  // every decision keyed on the variant.
  //
  // Only when the leaf IS the role directory, and only ONE level up: the
  // variant words are ordinary English and a full-path search would take
  // `/Users/dev/...` for the dev checkpoint.
  if (n == "diffusion_models" && p.has_parent_path()) {
    n = lower_(p.parent_path().filename().string());
  }
  // "distilled" is checked first because a distilled file's name does
  // not contain "dev" -- but a future "dev-distilled" spelling would,
  // and taking it for the dev checkpoint would silently sample a
  // guidance-distilled model with real CFG.
  if (n.find("distilled") != std::string::npos) { return Variant::kDistilled; }
  if (n.find("dev") != std::string::npos) { return Variant::kDev; }
  return Variant::kUnknown;
}

const char*
variant_name(Variant v)
{
  switch (v) {
    case Variant::kDistilled: return "distilled";
    case Variant::kDev:       return "dev";
    default:                  return "unknown";
  }
}

bool
parse_dit_config(const FlexData& cfg, DitConfig& out, SchedulerConfig* sched,
                 std::string* err)
{
  if (!cfg.is_object()) {
    if (err != nullptr) { *err = "DiT `config` metadata is not an object"; }
    return false;
  }
  const auto root = cfg.as_object();
  if (!root.contains("transformer")) {
    if (err != nullptr) {
      *err = "DiT `config` has no `transformer` object";
    }
    return false;
  }
  // Bind the owning FlexData to a local before taking a view: as_object()
  // returns a VIEW, and a view over a temporary dangles (bad_variant_access
  // at the first read, far from here).
  const FlexData& tf_fd = root.at("transformer");
  if (!tf_fd.is_object()) {
    if (err != nullptr) { *err = "`config.transformer` is not an object"; }
    return false;
  }
  const auto t = tf_fd.as_object();

  // IDENTITY. This is the check that decides whether the family claims a
  // checkpoint, so it must not be able to pass by DEFAULT.
  //
  // It nearly did. Every other field here is initialised to LTX-2.5's
  // shipped value and left alone when the key is absent, which is right
  // for a knob and catastrophic for an identity: `_class_name` defaulted
  // to "AVTransformer3DModel", so a config carrying a `transformer`
  // object and no class name matched. MiniMax-H3's Comfy-Org repack is
  // exactly that -- same `diffusion_models/` layout, same `config`
  // metadata key, a `transformer` object with no `_class_name` -- and
  // this family CLAIMED it, shadowing the built-in H3 path on a graph
  // that had done nothing wrong.
  //
  // So: `class_name` starts EMPTY, and there are two ways to be LTX-2.5.
  std::string cls;
  get_str_(t, "_class_name", cls);
  out.class_name = cls;
  if (!cls.empty()) {
    // Said who it is. Take it at its word, in both directions.
    if (cls != kDitClassName) {
      if (err != nullptr) {
        *err = "`_class_name` is '" + cls + "', not AVTransformer3DModel";
      }
      return false;
    }
  } else {
    // Did not say. Fall back to a STRUCTURAL signature rather than to a
    // default -- a repack that drops the class name should still
    // resolve, but only if it looks like this model and nothing else.
    // These four keys are LTX-2.5's joint audio-video shape; H3's config
    // (hidden_size / latents_dim / ffn_hidden_size / token_refiner_*)
    // has none of them.
    static constexpr const char* kSignature[] = {
        "audio_attention_head_dim", "use_audio_video_cross_attention",
        "connector_num_learnable_registers", "caption_channels"};
    for (const char* k : kSignature) {
      if (!t.contains(k)) {
        if (err != nullptr) {
          *err = std::string("no `_class_name`, and `") + k +
                 "` is missing too -- this is not an LTX-2.5 transformer "
                 "config";
        }
        return false;
      }
    }
    out.class_name = "AVTransformer3DModel";   // inferred, and say so
  }

  get_int_(t, "num_attention_heads", out.num_attention_heads);
  get_int_(t, "attention_head_dim",  out.attention_head_dim);
  get_int_(t, "in_channels",         out.in_channels);
  get_int_(t, "out_channels",        out.out_channels);
  get_int_(t, "num_layers",          out.num_layers);
  get_int_(t, "cross_attention_dim", out.cross_attention_dim);
  get_int_(t, "caption_channels",    out.caption_channels);
  get_bool_(t, "ff_bias",            out.ff_bias);

  get_int_(t, "audio_num_attention_heads", out.audio_num_attention_heads);
  get_int_(t, "audio_attention_head_dim",  out.audio_attention_head_dim);
  get_int_(t, "audio_out_channels",        out.audio_out_channels);
  get_int_(t, "audio_cross_attention_dim", out.audio_cross_attention_dim);
  get_bool_(t, "audio_ff_bias",            out.audio_ff_bias);
  get_bool_(t, "use_audio_video_cross_attention",
            out.use_audio_video_cross_attention);

  get_real_(t, "norm_eps",                   out.norm_eps);
  get_real_(t, "positional_embedding_theta", out.positional_embedding_theta);
  get_int_vec_(t, "positional_embedding_max_pos",
               out.positional_embedding_max_pos);
  get_int_vec_(t, "audio_positional_embedding_max_pos",
               out.audio_positional_embedding_max_pos);
  get_int_(t, "timestep_scale_multiplier", out.timestep_scale_multiplier);
  get_real_(t, "av_ca_timestep_scale_multiplier",
            out.av_ca_timestep_scale_multiplier);
  get_bool_(t, "use_middle_indices_grid",   out.use_middle_indices_grid);
  get_bool_(t, "apply_gated_attention",     out.apply_gated_attention);
  get_bool_(t, "cross_attention_adaln",     out.cross_attention_adaln);
  get_bool_(t, "causal_temporal_positioning", out.causal_temporal_positioning);
  get_bool_(t, "use_keyframes_abs_pos_embedding",
            out.use_keyframes_abs_pos_embedding);
  get_str_(t, "rope_type", out.rope_type);
  get_str_(t, "frequencies_precision", out.frequencies_precision);
  get_str_(t, "qk_norm",   out.qk_norm);

  get_bool_(t, "use_embeddings_connector",  out.use_embeddings_connector);
  get_int_(t, "connector_num_layers",       out.connector_num_layers);
  get_int_(t, "connector_num_attention_heads",
           out.connector_num_attention_heads);
  get_int_(t, "connector_attention_head_dim",
           out.connector_attention_head_dim);
  get_int_(t, "audio_connector_num_attention_heads",
           out.audio_connector_num_attention_heads);
  get_int_(t, "audio_connector_attention_head_dim",
           out.audio_connector_attention_head_dim);
  get_int_(t, "connector_num_learnable_registers",
           out.connector_num_learnable_registers);
  get_int_vec_(t, "connector_positional_embedding_max_pos",
               out.connector_positional_embedding_max_pos);
  get_bool_(t, "connector_apply_gated_attention",
            out.connector_apply_gated_attention);
  get_bool_(t, "connector_norm_output",     out.connector_norm_output);
  get_bool_(t, "caption_proj_before_connector",
            out.caption_proj_before_connector);

  // A `rope_type` this port does not implement is a refusal, not a
  // warning: "interleaved" is a legacy mode with a different rotation,
  // and running it as "split" produces a clean forward and scrambled
  // attention -- the exact failure mode that cost the H3 bring-up a
  // session.
  if (out.rope_type != "split") {
    if (err != nullptr) {
      *err = "rope_type '" + out.rope_type +
             "' is not implemented (this port handles 'split' only)";
    }
    return false;
  }

  if (sched != nullptr && root.contains("scheduler")) {
    const FlexData& sc_fd = root.at("scheduler");
    if (sc_fd.is_object()) {
      const auto s = sc_fd.as_object();
      get_str_(s, "_class_name", sched->class_name);
      get_str_(s, "sampler",     sched->sampler);
      get_int_(s, "num_train_timesteps", sched->num_train_timesteps);
    }
  }
  return true;
}

// The text encoder in `root`, preferring `prefer` when it names one.
//
// Two shapes live side by side under `text_encoders/`:
//
//   * the released single .safetensors, whose `__metadata__` carries the
//     backbone config under `gemma_config`; and
//   * a DIRECTORY written by `model-quantize` -- shards, an index, and a
//     config.json holding that same config plus a `quantization` block.
//
// A directory is taken only when ASKED FOR by name. Preferring one
// silently would change the precision of a run that did not request it,
// and the encoder is the component every conditioning token comes out
// of -- a quality change there reads as a port bug, not as a setting.
std::string
resolve_enc_(const std::string& root, const std::string& prefer)
{
  namespace fs = std::filesystem;
  std::error_code ec;
  const fs::path dir = fs::path(root) / "text_encoders";
  if (!prefer.empty()) {
    for (const auto& e : fs::directory_iterator(dir, ec)) {
      if (ec) { break; }
      if (!e.is_directory()) { continue; }
      if (e.path().filename().string().find(prefer) == std::string::npos) {
        continue;
      }
      // A quantized CHECKPOINT, not merely a directory whose name matched.
      if (fs::exists(e.path() / "config.json", ec)) {
        return e.path().string();
      }
    }
  }
  const std::string released = vpipe::genai::comfy::resolve_component(
      root, "text_encoders", kEncMetaKey, {"gemma4", "with-proj"});
  if (!released.empty()) { return released; }
  // `model-quantize`'s self-contained output replaces the whole
  // `text_encoders/` role with the quantized encoder, so the config.json
  // is AT the role dir. Taken only AFTER the released file, and only
  // because reaching here means there is no released file to take: the
  // rule above is "never silently prefer a quantized encoder OVER the
  // bf16 one", not "never load a model that has only a quantized one".
  // A self-contained model holds exactly one encoder, so there is no
  // preference left to express.
  if (fs::exists(dir / "config.json", ec)) { return dir.string(); }
  return {};
}

bool
resolve(const std::string& root_in, Config& out, std::string* err,
        const std::string& prefer_variant, const std::string& prefer_enc)
{
  if (err != nullptr) { err->clear(); }
  if (root_in.empty()) {
    if (err != nullptr) { *err = "empty checkpoint path"; }
    return false;
  }
  std::error_code ec;
  // `root` may be the repo dir OR the DiT file itself, so that a
  // model-select beat naming a file resolves the same as one naming the
  // directory. A file two levels down (repo/diffusion_models/x.st) means
  // the repo is its grandparent.
  std::string root = root_in;
  if (fs::is_regular_file(fs::path(root_in), ec)) {
    const fs::path p(root_in);
    root = p.parent_path().parent_path().string();
  }
  if (!fs::is_directory(fs::path(root), ec)) {
    if (err != nullptr) { *err = "'" + root + "' is not a directory"; }
    return false;
  }

  out = Config{};
  out.root     = root;
  out.dit_file = resolve_dit_(root, prefer_variant, err);
  if (out.dit_file.empty()) { return false; }

  // The config comes from the file's `__metadata__` for a Comfy single
  // file, and from config.json for a quantized directory. The two carry
  // the SAME object -- comfy_output_config writes the embedded config
  // through verbatim, scheduler block included -- so only the source
  // differs and parse_dit_config below is shared.
  FlexData cfg;
  {
    std::error_code dec;
    out.dit_is_dir = std::filesystem::is_directory(out.dit_file, dec) && !dec;
  }
  if (out.dit_is_dir) {
    const std::string cj =
        (std::filesystem::path(out.dit_file) / "config.json").string();
    std::string txt;
    if (!read_file_(cj, &txt)) {
      if (err != nullptr) { *err = "could not read '" + cj + "'"; }
      return false;
    }
    try { cfg = FlexData::from_json(txt); }
    catch (...) {
      if (err != nullptr) { *err = "'" + cj + "' is not valid JSON"; }
      return false;
    }
  } else {
    std::string merr;
    if (!vpipe::genai::comfy::metadata_json(out.dit_file, kDitMetaKey, cfg,
                                            &merr)) {
      if (err != nullptr) {
        *err = "could not read `" + std::string(kDitMetaKey) + "` from '" +
               out.dit_file + "': " + merr;
      }
      return false;
    }
  }
  if (!parse_dit_config(cfg, out.dit, &out.scheduler, err)) { return false; }
  out.variant = variant_of_filename(out.dit_file);

  // The peers. All optional at RESOLVE time -- a root holding only a DiT
  // is still recognisably LTX-2.5, and refusing here would make the
  // family's `claims` depend on which files someone happened to fetch.
  // What needs each of them says so when it loads.
  out.enc_file = resolve_enc_(root, prefer_enc);
  // The CONV VAE, not the diffusion one: the diffusion decoder is a
  // second sampling loop and is not ported yet, so naming it here would
  // resolve a file nothing can run.
  out.vae_file = vpipe::genai::comfy::resolve_component(
      root, "vae", kVaeMetaKey, {"video-vae-conv"});
  out.audio_vae_file = vpipe::genai::comfy::resolve_component(
      root, "vae", kVaeMetaKey, {"audio-vae"});
  // The duration head is a `model_patches/` component and genuinely
  // optional -- without it the frame count comes from config, which is
  // what every other family in this tree does anyway.
  const fs::path patches = fs::path(root) / "model_patches";
  if (fs::is_directory(patches, ec)) {
    for (const auto& e : fs::directory_iterator(patches, ec)) {
      const std::string n = lower_(e.path().filename().string());
      if (n.find("duration-head") != std::string::npos &&
          e.path().extension() == ".safetensors") {
        out.duration_head_file = e.path().string();
        break;
      }
    }
  }
  return true;
}

}  // namespace ltx25
