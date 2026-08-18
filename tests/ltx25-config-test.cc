// Checks the one piece of real logic the plugin has today: turning an
// LTX-2.5 checkpoint on disk into a Config.
//
// The frame / dimension rules run everywhere. The checkpoint half is
// gated on VPIPE_LTX25_TEST_MODEL_PATH and SKIPS when it is unset -- so
// read the output, not just the exit code: a vacuous skip looks exactly
// like a pass. It says which it did.

#include "ltx25-config.h"
#include "ltx25-quant-family.h"

#include <filesystem>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

int g_fail = 0;

void
check(bool ok, const std::string& what)
{
  std::printf("  [%s] %s\n", ok ? " OK " : "FAIL", what.c_str());
  if (!ok) { ++g_fail; }
}

void
test_frame_rule()
{
  std::printf("frame + dimension rules\n");
  // frames % 8 == 1, rounded UP. 121 is the reference's own example.
  check(ltx25::align_num_frames(121) == 121, "121 is already legal");
  check(ltx25::align_num_frames(120) == 121, "120 -> 121");
  check(ltx25::align_num_frames(122) == 129, "122 -> 129");
  check(ltx25::align_num_frames(1) == 1,     "1 -> 1");
  check(ltx25::align_num_frames(0) == 1,     "0 -> 1 (never zero frames)");
  check(ltx25::align_num_frames(9) == 9,     "9 is legal");
  // Every output must satisfy the rule, which is the property that
  // matters -- the table above only samples it.
  bool all_legal = true;
  for (int n = 1; n <= 400; ++n) {
    const int a = ltx25::align_num_frames(n);
    if (a % 8 != 1 || a < n) { all_legal = false; break; }
  }
  check(all_legal, "every 1..400 rounds UP to a frames%8==1 count");

  check(ltx25::align_dimension(960) == 960, "960 is a multiple of 32");
  check(ltx25::align_dimension(544) == 544, "544 is a multiple of 32");
  check(ltx25::align_dimension(950) == 960, "950 -> 960");
  check(ltx25::align_dimension(1) == 32,    "1 -> 32");

  // The distilled schedule, as the reference publishes it. Descending,
  // 1.0 = pure noise -- reversing this is a silent quality bug, not a
  // crash, so it is worth pinning.
  const auto& s = ltx25::distilled_sigmas();
  check(s.size() == 9, "the distilled schedule is 8 steps (9 sigmas)");
  check(s.front() == 1.0 && s.back() == 0.0, "it runs 1.0 -> 0.0");
  bool descending = true;
  for (std::size_t i = 1; i < s.size(); ++i) {
    if (!(s[i] < s[i - 1])) { descending = false; break; }
  }
  check(descending, "sigmas DESCEND (1 = noise)");
}

void
test_variant_from_filename()
{
  std::printf("variant detection (the two DiTs are told apart by NAME)\n");
  using ltx25::Variant;
  check(ltx25::variant_of_filename(
            "ltx-2.5-22b-distilled-transformer-bf16.safetensors")
        == Variant::kDistilled, "distilled");
  check(ltx25::variant_of_filename(
            "ltx-2.5-22b-dev-transformer-bf16.safetensors")
        == Variant::kDev, "dev");
  check(ltx25::variant_of_filename("something-else.safetensors")
        == Variant::kUnknown, "neither");

  // A SELF-CONTAINED pack names the variant on its ROOT: model-quantize
  // writes <name>/diffusion_models/, so the leaf is the role directory
  // and says nothing. This is what a quantized model actually looks like
  // on disk, and reading only the leaf reported kUnknown for it -- which
  // silenced the "your step count is being ignored" notice for exactly
  // the checkpoints most people run.
  check(ltx25::variant_of_filename(
            "/m/local/LTX-2.5-distilled-8bit/diffusion_models")
        == Variant::kDistilled, "a quantized pack, named on its root");
  check(ltx25::variant_of_filename(
            "/m/local/LTX-2.5-dev-4bit/diffusion_models")
        == Variant::kDev, "and the dev one likewise");
  // ONE level up, and only past the role directory. The variant words
  // are ordinary English, so a full-path search takes somebody's home
  // directory for the dev checkpoint.
  check(ltx25::variant_of_filename(
            "/Users/dev/models/whatever/diffusion_models")
        == Variant::kUnknown, "but not from further up the path");
  check(ltx25::variant_of_filename("/Users/dev/models/plain.safetensors")
        == Variant::kUnknown, "nor from a parent that is not the role dir");
}

void
test_real_checkpoint(const char* root)
{
  std::printf("real checkpoint at %s\n", root);
  ltx25::Config cfg;
  std::string err;
  const bool ok = ltx25::resolve(root, cfg, &err, {});
  check(ok, ok ? "resolved" : ("resolve failed: " + err));
  if (!ok) { return; }

  // Everything below is what the SHIPPED config.transformer says. These
  // are the numbers the whole port is built against, so a checkpoint
  // that disagrees should fail here rather than 48 layers later.
  check(cfg.dit.class_name == "AVTransformer3DModel", "_class_name");
  check(cfg.dit.num_layers == 48,          "48 layers");
  check(cfg.dit.inner_dim() == 4096,       "video inner dim 4096 (32 x 128)");
  check(cfg.dit.audio_inner_dim() == 2048, "audio inner dim 2048 (32 x 64)");
  check(cfg.dit.in_channels == 128 && cfg.dit.out_channels == 128,
        "128 latent channels in and out");
  check(cfg.dit.caption_channels == 3840,
        "caption channels 3840 (Gemma-4 12B hidden)");
  check(cfg.dit.cross_attention_dim == 4096,   "cross-attn dim 4096");
  check(cfg.dit.audio_cross_attention_dim == 2048, "audio cross-attn 2048");
  check(cfg.dit.apply_gated_attention,     "gated attention is ON");
  check(cfg.dit.cross_attention_adaln,     "cross-attention adaLN is ON");
  check(cfg.dit.use_middle_indices_grid,   "middle-indices RoPE grid is ON");
  check(cfg.dit.rope_type == "split",      "rope_type split");
  check(!cfg.dit.ff_bias,                  "the VIDEO ff has NO bias");
  check(cfg.dit.audio_ff_bias,             "the AUDIO ff HAS bias");
  check(cfg.dit.use_audio_video_cross_attention,
        "audio<->video cross-attention is ON");
  check(cfg.dit.positional_embedding_max_pos.size() == 3 &&
        cfg.dit.positional_embedding_max_pos[0] == 20 &&
        cfg.dit.positional_embedding_max_pos[1] == 2048 &&
        cfg.dit.positional_embedding_max_pos[2] == 2048,
        "max_pos [20, 2048, 2048]");
  check(cfg.dit.connector_num_layers == 8, "8-layer embeddings connector");
  check(cfg.dit.connector_num_learnable_registers == 128,
        "128 learnable registers");
  check(cfg.scheduler.class_name == "RectifiedFlowScheduler",
        "RectifiedFlowScheduler");
  check(cfg.scheduler.sampler == "LinearQuadratic", "LinearQuadratic sampler");

  check(!cfg.dit_file.empty(), "a DiT file resolved");
  check(cfg.variant != ltx25::Variant::kUnknown,
        std::string("variant is ") + ltx25::variant_name(cfg.variant));
  std::printf("       dit=%s\n       enc=%s\n       vae=%s\n"
              "       avae=%s\n       dur=%s\n",
              cfg.dit_file.c_str(),
              cfg.enc_file.empty()  ? "(absent)" : cfg.enc_file.c_str(),
              cfg.vae_file.empty()  ? "(absent)" : cfg.vae_file.c_str(),
              cfg.audio_vae_file.empty() ? "(absent)"
                                         : cfg.audio_vae_file.c_str(),
              cfg.duration_head_file.empty()
                  ? "(absent)" : cfg.duration_head_file.c_str());

  // The conv VAE, NOT the diffusion one: the diffusion decoder is a
  // second sampling loop and is not ported, so resolving it here would
  // name a file nothing can run.
  if (!cfg.vae_file.empty()) {
    check(cfg.vae_file.find("conv") != std::string::npos,
          "the resolved video VAE is the CONV one");
  }

  // A directory that is not LTX-2.5 must be refused. Claiming someone
  // else's checkpoint is the worst failure this family has.
  ltx25::Config other;
  check(!ltx25::resolve("/tmp", other, nullptr, {}),
        "/tmp is refused (not an LTX-2.5 pack)");
}

// The near-miss that actually happened. MiniMax-H3's Comfy-Org repack has
// the same `diffusion_models/` layout, the same `config` metadata key and
// a `transformer` object -- with NO `_class_name`. With `class_name`
// defaulted to "AVTransformer3DModel" the identity check passed and this
// family claimed H3, shadowing the built-in path.
//
// Gated separately from the LTX checkpoint because a box can have either.
void
test_rejects_lookalike(const char* root)
{
  std::printf("look-alike refusal: %s\n", root);
  ltx25::Config cfg;
  std::string err;
  const bool claimed = ltx25::resolve(root, cfg, &err, {});
  check(!claimed, claimed
        ? "CLAIMED a non-LTX checkpoint -- this shadows its real family"
        : "refused (" + err + ")");
}


// The SELF-CONTAINED layout: `model-quantize`'s registered-family path
// replaces a whole role subdir with the quantized component, so the
// config.json sits AT `diffusion_models/` and `text_encoders/` rather
// than in a pack inside them. Needs no checkpoint -- it is a question
// about directory shapes, and the answer is a silent wrong one: a
// resolver that does not know the shape reports "no readable DiT" for a
// model that plainly has one.
//
// It also pins the rule this must NOT break. A quantized encoder is
// never preferred over a released bf16 one implicitly, because every
// conditioning token comes out of it; the self-contained case is
// different only because there is no bf16 file there to prefer.
void
test_self_contained_layout()
{
  namespace fs = std::filesystem;
  std::printf("self-contained (role dir IS the checkpoint)\n");
  std::error_code ec;
  const fs::path base =
      fs::temp_directory_path(ec) / "ltx25-selfcontained-test";
  fs::remove_all(base, ec);

  auto write = [](const fs::path& f, const std::string& body) {
    std::error_code wec;
    fs::create_directories(f.parent_path(), wec);
    std::ofstream o(f);
    o << body;
  };
  // Minimal but REAL: parse_dit_config refuses anything whose
  // _class_name is not AVTransformer3DModel, so this is the same gate a
  // genuine pack passes.
  const std::string dit_cfg =
      R"({"transformer":{"_class_name":"AVTransformer3DModel"}})";
  const std::string enc_cfg = R"({"model_type":"gemma3n_text"})";

  write(base / "diffusion_models" / "config.json", dit_cfg);
  write(base / "text_encoders" / "config.json", enc_cfg);

  ltx25::Config cfg;
  std::string err;
  const bool ok = ltx25::resolve(base.string(), cfg, &err);
  check(ok, "a self-contained model resolves" +
                (ok ? std::string() : (" -- " + err)));
  if (ok) {
    check(cfg.dit_file == (base / "diffusion_models").string(),
          "the DiT is the diffusion_models/ role dir itself");
    check(cfg.dit_is_dir, "and it is reported as a directory checkpoint");
    check(cfg.enc_file == (base / "text_encoders").string(),
          "the encoder is the text_encoders/ role dir itself");
  }

  // A NESTED pack beside it still wins when it is asked for by name --
  // the role dir is a fallback for "there is nothing else", not a new
  // preference that outranks an explicit request.
  write(base / "text_encoders" / "enc-w4g64" / "config.json", enc_cfg);
  ltx25::Config c2;
  const bool ok2 = ltx25::resolve(base.string(), c2, &err, {}, "w4g64");
  check(ok2 && c2.enc_file ==
                   (base / "text_encoders" / "enc-w4g64").string(),
        "an asked-for nested pack still outranks the role dir");

  fs::remove_all(base, ec);
}

// The QUANTIZE family's claim, which is what routes an LTX repo to
// `model-quantize`'s self-contained path instead of the single-component
// fallback. Two directions matter and they fail differently: a claim
// that is too narrow silently drops the model back to the old shape,
// and one that is too broad quantizes SOMEONE ELSE'S weights with LTX's
// scope -- which writes a checkpoint that loads and is wrong.
void
test_quant_family_claim()
{
  namespace fs = std::filesystem;
  std::printf("quantize-family claim\n");
  std::error_code ec;
  const fs::path base = fs::temp_directory_path(ec) / "ltx25-claim-test";
  fs::remove_all(base, ec);
  auto write = [](const fs::path& f, const std::string& body) {
    std::error_code wec;
    fs::create_directories(f.parent_path(), wec);
    std::ofstream o(f);
    o << body;
  };

  ltx25::Ltx25QuantFamily fam;
  check(fam.tag() == std::string_view("ltx-2.5"), "it names itself ltx-2.5");

  // An LTX repo, in the shape a chain's SECOND pass is handed: the DiT
  // already quantized to a directory checkpoint. Claiming this is what
  // lets the encoder pass run at all.
  write(base / "ltx" / "diffusion_models" / "config.json",
        R"({"transformer":{"_class_name":"AVTransformer3DModel"}})");
  check(fam.claims((base / "ltx").string()),
        "claims an LTX repo whose DiT is already a directory checkpoint");

  // A look-alike: same layout, another family's DiT. Refusing is the
  // half that protects other people's checkpoints.
  write(base / "other" / "diffusion_models" / "config.json",
        R"({"transformer":{"_class_name":"WanTransformer3DModel"}})");
  check(!fam.claims((base / "other").string()),
        "refuses a repo whose DiT is another family's");
  check(!fam.claims((base / "nope").string()),
        "refuses a path that does not exist");

  // The recipe. These two scopes ARE the port's quantization contract:
  // the DiT excludes the modulation tables and gate logits, and the
  // encoder's single prefix is what leaves the embedding table and
  // LTX's projection dense.
  const auto comps = fam.components();
  check(comps.size() == 2, "describes two components");
  if (comps.size() == 2) {
    check(comps[0].target == "dit" && comps[0].role == "diffusion_models",
          "the DiT: diffusion_models/");
    // The scope by what it SELECTS, not by its spelling. It is a
    // SUBSTRING, and it has to reach two stacks that are named
    // differently: the DiT's `transformer_blocks.` and the two text
    // connectors' `transformer_1d_blocks.`. The connectors were outside
    // it once and shipped dense in a pack asked for at w8 -- 3.75 GB of
    // bf16 that never streams -- so a scope that stops matching them is
    // the regression to catch, and the literal cannot say that.
    auto in_scope = [&](const char* n) {
      return std::string(n).find(comps[0].scope) != std::string::npos;
    };
    check(in_scope("model.diffusion_model.transformer_blocks.0.attn1.to_q"
                   ".weight"),
          "the scope reaches the DiT block stack");
    check(in_scope("model.diffusion_model.video_embeddings_connector"
                   ".transformer_1d_blocks.0.attn1.to_k.weight") &&
          in_scope("model.diffusion_model.audio_embeddings_connector"
                   ".transformer_1d_blocks.7.ff.net.2.weight"),
          "and BOTH text connectors' block stacks");
    // And stops there: the trunk is read dense by bind_trunk, and the
    // registers are a learned table rather than a matrix.
    check(!in_scope("model.diffusion_model.patchify_proj.weight") &&
          !in_scope("model.diffusion_model.adaln_single.linear.weight") &&
          !in_scope("model.diffusion_model.video_embeddings_connector"
                    ".learnable_registers"),
          "but not the trunk or the connectors' learnable registers");
    check(comps[0].exclude.find("scale_shift") != std::string::npos &&
              comps[0].exclude.find("to_gate_logits") != std::string::npos,
          "and it excludes the modulation tables and the gate logits");
    check(comps[1].target == "text_encoder" &&
              comps[1].role == "text_encoders" &&
              comps[1].scope == "model.layers.",
          "the encoder: text_encoders/, scoped to the Gemma backbone");
  }
  fs::remove_all(base, ec);
}
}  // namespace

int
main()
{
  test_frame_rule();
  test_variant_from_filename();
  test_self_contained_layout();
  test_quant_family_claim();

  if (const char* root = std::getenv("VPIPE_LTX25_TEST_MODEL_PATH")) {
    test_real_checkpoint(root);
  } else {
    std::printf("SKIPPED the checkpoint tests: set "
                "VPIPE_LTX25_TEST_MODEL_PATH to the LTX-2.5 repo dir.\n"
                "Only the pure rules above actually ran.\n");
  }
  if (const char* other = std::getenv("VPIPE_LTX25_TEST_LOOKALIKE_PATH")) {
    test_rejects_lookalike(other);
  } else {
    std::printf("SKIPPED the look-alike refusal: set "
                "VPIPE_LTX25_TEST_LOOKALIKE_PATH to another Comfy-packed "
                "video checkpoint (e.g. Comfy-Org/MiniMax-H3).\n");
  }
  std::printf("%s\n", g_fail == 0 ? "ALL PASSED" : "FAILURES");
  return g_fail == 0 ? 0 : 1;
}
