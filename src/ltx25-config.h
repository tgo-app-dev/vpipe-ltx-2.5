#ifndef VPIPE_LTX25_CONFIG_H
#define VPIPE_LTX25_CONFIG_H

#include <string>
#include <vector>

namespace vpipe { class FlexData; }

namespace ltx25 {

// The LTX-2.5 family tag. It is what a `ltx-2.5-model-config` beat stamps
// into `model_family`, what VideoModelFamily::tag() returns, and what
// generate-video prints -- one spelling, so a config wired to the wrong
// checkpoint is reported instead of half-applied.
inline constexpr const char* kFamily = "ltx-2.5";

// The plugin's own metallibs, as registered at load and resolved by
// load_library(). Three dtype twins of one source; the DiT runs bf16.
inline constexpr const char* kMetalLibBf16 = "ltx25_kernels_bf16";
inline constexpr const char* kMetalLibF16  = "ltx25_kernels_f16";
// The f32 twin, which the VOCODER needs. The reference forces its whole
// pass to fp32 and says why: bf16 accumulation over its 108 sequential
// convolutions degrades spectral metrics by 40-90%. The WEIGHTS are
// bf16 in the reference too -- it is the ACTIVATIONS and the
// accumulation that have to be wide, so this twin buys correctness that
// no amount of care in bf16 could.
inline constexpr const char* kMetalLibF32  = "ltx25_kernels_f32";

// The `__metadata__` key each component's config hides under. There is no
// config.json anywhere in this pack; see ARCHITECTURE.md.
inline constexpr const char* kDitMetaKey = "config";
inline constexpr const char* kEncMetaKey = "gemma_config";
inline constexpr const char* kVaeMetaKey = "config";

// The video VAE's compression, and the frame rule that follows from it.
// Taken from the checkpoint's own block list in the reference
// (SpatioTemporalScaleFactors::from_blocks), not hardcoded upstream --
// but constant across every LTX-2.5 VAE that ships, so the loader
// asserts them rather than guessing.
inline constexpr int kSpatialCompression  = 32;
inline constexpr int kTemporalCompression = 8;

// The AUDIO latent's rate, from the audio VAE's own numbers:
// sample_rate 16000 / hop 160 / latent downsample 4 = 25 latent frames a
// second. The reference derives the audio token count from the clip's
// PIXEL frames over fps (AudioLatentShape.from_video_pixel_shape), NOT
// from the video LATENT frame count -- an easy substitution that is
// wrong by the temporal compression factor.
inline constexpr int    kAudioSampleRate       = 16000;
inline constexpr int    kAudioHopLength        = 160;
inline constexpr int    kAudioLatentDownsample = 4;
inline constexpr double kAudioLatentsPerSecond =
    (double)kAudioSampleRate / (double)kAudioHopLength /
    (double)kAudioLatentDownsample;               // 25.0
// The audio latent's own axes: 8 channels x 16 mel bins, which is the
// 128 the DiT carries per token (patchified as c*16 + f).
inline constexpr int kAudioLatentChannels = 8;
inline constexpr int kAudioLatentMelBins  = 16;

// Round `frames` UP to the nearest count the VAE can chunk: frames % 8
// == 1 (1, 9, 17, ..., 121). Any positive input is accepted and adjusted
// -- a graph must be able to change families without being re-authored,
// so this rounds rather than rejects.
int align_num_frames(int frames);

// Round a pixel dimension UP to a multiple of 32.
int align_dimension(int px);

// The DiT's architecture, exactly as the checkpoint's `config.transformer`
// object spells it. Field names track the JSON keys so the two can be
// compared by eye; the defaults are LTX-2.5's shipped values, which is
// what a missing key means.
struct DitConfig {
  // EMPTY by default, unlike every other field here. This one decides
  // whether the family claims a checkpoint, and a default that happens
  // to be the right answer makes the check unfalsifiable -- see the long
  // note in parse_dit_config.
  std::string class_name;

  // ---- video stream ----
  int    num_attention_heads = 32;
  int    attention_head_dim  = 128;      // -> inner_dim 4096
  int    in_channels         = 128;
  int    out_channels        = 128;
  int    num_layers          = 48;
  int    cross_attention_dim = 4096;
  int    caption_channels    = 3840;     // the encoder's hidden size
  bool   ff_bias             = false;

  // ---- audio stream ----
  int    audio_num_attention_heads = 32;
  int    audio_attention_head_dim  = 64;   // -> audio_inner_dim 2048
  int    audio_out_channels        = 128;
  int    audio_cross_attention_dim = 2048;
  bool   audio_ff_bias             = true;
  bool   use_audio_video_cross_attention = true;

  // ---- shared ----
  double norm_eps                     = 1e-6;
  double positional_embedding_theta   = 10000.0;
  std::vector<int> positional_embedding_max_pos       = {20, 2048, 2048};
  std::vector<int> audio_positional_embedding_max_pos = {20};
  int    timestep_scale_multiplier        = 1000;
  double av_ca_timestep_scale_multiplier  = 1000.0;
  bool   use_middle_indices_grid          = true;
  bool   apply_gated_attention            = true;
  bool   cross_attention_adaln            = true;
  bool   causal_temporal_positioning      = true;
  bool   use_keyframes_abs_pos_embedding  = true;
  std::string rope_type = "split";        // "split" | "interleaved"
  // "float64" (LTX-2.5's value) or anything else for f32. NOT a quality
  // knob -- it selects which of the reference's two frequency ladders is
  // built, and they differ by 6e-4 rel-L2. See ltx25-rope.h.
  std::string frequencies_precision = "float64";
  bool rope_f64() const { return frequencies_precision == "float64"; }
  std::string qk_norm   = "rms_norm";

  // ---- embeddings connector (the perceiver between encoder and DiT) ----
  bool use_embeddings_connector         = true;
  int  connector_num_layers             = 8;
  int  connector_num_attention_heads    = 32;
  int  connector_attention_head_dim     = 128;
  int  audio_connector_num_attention_heads = 32;
  int  audio_connector_attention_head_dim  = 64;
  int  connector_num_learnable_registers   = 128;
  std::vector<int> connector_positional_embedding_max_pos = {4096};
  bool connector_apply_gated_attention  = true;
  bool connector_norm_output            = true;
  bool caption_proj_before_connector    = true;

  int inner_dim()       const { return num_attention_heads * attention_head_dim; }
  int audio_inner_dim() const
  {
    return audio_num_attention_heads * audio_attention_head_dim;
  }
};

// The scheduler half of the DiT file's `config`.
struct SchedulerConfig {
  std::string class_name        = "RectifiedFlowScheduler";
  std::string sampler           = "LinearQuadratic";
  int         num_train_timesteps = 1000;
};

// Which DiT this is. The two ship the SAME architecture and differ only
// in how they are sampled -- distilled is guidance-distilled with a fixed
// 8-step sigma list, dev takes real CFG over ~40 steps. Nothing in the
// tensors distinguishes them, so it comes from the FILENAME, the way
// MiniMax-H3's partition comes from its packaging.
enum class Variant { kUnknown, kDistilled, kDev };

// The distilled checkpoint's fixed schedule (reference
// DISTILLED_SIGMA_VALUES). Descending, 1.0 = pure noise.
const std::vector<double>& distilled_sigmas();
// The 2x-resolution second stage's 3-step tail.
const std::vector<double>& stage2_distilled_sigmas();

// Everything resolved about one LTX-2.5 checkpoint root.
struct Config {
  std::string root;          // the repo directory
  // diffusion_models/*.safetensors, OR -- for a quantized pack -- the
  // DIRECTORY of shards `model-quantize` wrote there. `dit_is_dir` says
  // which, and everything that opens it has to branch: a WeightSet over
  // a directory reads config.json and an index, over a file it reads
  // one header.
  std::string dit_file;
  bool        dit_is_dir = false;
  std::string enc_file;      // text_encoders/*.safetensors
  std::string vae_file;      // vae/*video-vae-conv*.safetensors
  std::string audio_vae_file;
  std::string duration_head_file;   // may be empty -- it is optional
  Variant         variant = Variant::kUnknown;
  DitConfig       dit;
  SchedulerConfig scheduler;
};

// Resolve an LTX-2.5 checkpoint at `root`.
//
// `root` may be the repo directory or the DiT file itself. Returns false
// (with `err` set) when this is not an LTX-2.5 pack -- which is exactly
// what VideoModelFamily::claims needs, so detection and configuration are
// one read rather than two that can disagree.
//
// `prefer_variant` biases the DiT choice when a root holds both ("dev" /
// "distilled"); empty prefers the distilled one, which is the checkpoint
// this port is built and verified against.
// `prefer_enc` biases the TEXT ENCODER the same way, and is a separate
// knob because the two are separate choices: a box may want a quantized
// encoder beside a bf16 DiT (the encoder runs once and is destroyed; the
// DiT runs every step), or the reverse. Empty takes the released bf16
// file, which is what every run did before quantized encoders existed.
bool resolve(const std::string& root, Config& out, std::string* err = nullptr,
             const std::string& prefer_variant = {},
             const std::string& prefer_enc = {});

// Parse `config.transformer` / `config.scheduler` out of a DiT file's
// `__metadata__`. Exposed separately so a test can read a config without
// a family, and so a caller that already has the FlexData does not have
// to re-open the file.
bool parse_dit_config(const vpipe::FlexData& cfg, DitConfig& out,
                      SchedulerConfig* sched = nullptr,
                      std::string* err = nullptr);

// The variant a DiT filename names. kUnknown when it says neither.
Variant variant_of_filename(const std::string& file);
const char* variant_name(Variant v);

}  // namespace ltx25

#endif
