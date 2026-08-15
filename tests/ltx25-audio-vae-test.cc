// The audio VAE decoder (Metal) against the real weights.
//
// Latent -> log-mel spectrogram. Intermediates are checked in GRAPH
// ORDER so a mismatch names the stage that caused it; the ones after it
// are consequences.
//
// The `denorm` tap is the one that matters most. The per-channel
// statistics are 128 long against 8 latent channels because the
// reference denormalises the PATCHIFIED latent (`b c t f -> b t (c f)`),
// so they are per (channel, mel bin) with the mel bin FASTEST. Reading
// them as 8 per-channel values is arithmetically fine and produces a
// perfectly plausible spectrogram -- this tap is what distinguishes the
// two.

#include "ltx25-audio-vae.h"
#include "ltx25-config.h"
#include "ltx25-metal-ops.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/flex-data.h"
#include "generative-models/shared/comfy-checkpoint.h"
#include "generative-models/weight-set.h"
#include "npy.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

extern "C" const unsigned char ltx25_kernels_bf16_metallib[];
extern "C" const unsigned long ltx25_kernels_bf16_metallib_len;

namespace {

int g_fail = 0;

void
check(bool ok, const std::string& what)
{
  std::printf("  [%s] %s\n", ok ? " OK " : "FAIL", what.c_str());
  if (!ok) { ++g_fail; }
}

std::string
env_(const char* k)
{
  const char* v = std::getenv(k);
  return (v != nullptr) ? std::string(v) : std::string();
}

}  // namespace

int
main()
{
  // Unbuffered: this drives GPU kernels, and a crash with a block-
  // buffered stdout loses every line that said where it got to.
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const std::string root = env_("VPIPE_LTX25_TEST_MODEL_PATH");
  const std::string gdir = env_("VPIPE_LTX25_GOLDENS");
  if (root.empty() || gdir.empty()) {
    std::printf("SKIPPED: needs VPIPE_LTX25_TEST_MODEL_PATH and "
                "VPIPE_LTX25_GOLDENS. NOTHING was checked.\n");
    return 0;
  }
  vpipe::metal_compute::MetalCompute mc(nullptr);
  if (!mc.valid()) {
    std::printf("SKIPPED: no Metal device. NOTHING was checked.\n");
    return 0;
  }
  // The plugin's metallib must be registered before MetalOps resolves
  // anything out of it -- an unregistered library hands back an invalid
  // one and .function() then segfaults.
  mc.register_metal_library(ltx25::kMetalLibBf16, ltx25_kernels_bf16_metallib,
                            ltx25_kernels_bf16_metallib_len);

  ltx25::Config cfg;
  std::string err;
  if (!ltx25::resolve(root, cfg, &err)) {
    check(false, "resolve: " + err);
    std::printf("FAILURES\n");
    return 1;
  }
  if (cfg.audio_vae_file.empty()) {
    std::printf("SKIPPED: this checkpoint has no audio VAE. NOTHING was "
                "checked.\n");
    return 0;
  }

  vpipe::FlexData acfg;
  if (!vpipe::genai::comfy::metadata_json(cfg.audio_vae_file,
                                          ltx25::kVaeMetaKey, acfg, &err)) {
    check(false, "audio VAE metadata: " + err);
    std::printf("FAILURES\n");
    return 1;
  }
  ltx25::AudioVaeConfig ac;
  if (!ltx25::parse_audio_vae_config(acfg, ac, &err)) {
    check(false, "parse_audio_vae_config: " + err);
    std::printf("FAILURES\n");
    return 1;
  }
  check(ac.z_channels == 8, "8 latent channels");
  check(ac.mel_bins == 64, "64 mel bins");
  check(ac.num_levels() == 3, "3 levels (ch_mult 1,2,4)");
  check(ac.num_res_blocks == 2, "num_res_blocks 2 (so THREE blocks a level)");
  check(!ac.mid_block_add_attention, "the mid block has no attention");
  // The two numbers the patchifier packing rests on. 64 mel / 2^2 = 16,
  // and 8 * 16 = 128 = the length of the statistics vectors.
  check(ac.latent_mel_bins() == 16, "the latent carries 16 mel bins");
  check(ac.stats_len() == 128,
        "128 statistics = 8 channels x 16 mel bins (NOT 8 per-channel)");

  ltx25::MetalOps ops;
  if (!ops.init(&mc, &err)) {
    check(false, "MetalOps::init: " + err);
    std::printf("FAILURES\n");
    return 1;
  }
  auto ws = vpipe::genai::open_weight_set(cfg.audio_vae_file, nullptr);
  if (!ws) {
    check(false, "could not open " + cfg.audio_vae_file);
    std::printf("FAILURES\n");
    return 1;
  }
  auto dec = ltx25::Ltx25AudioVaeDecoder::load(ac, ws, ops, &err);
  if (!dec) {
    check(false, "load: " + err);
    std::printf("FAILURES\n");
    return 1;
  }
  std::printf("       decoder weights %.1f MB\n",
              (double)dec->resident_bytes() / (1024.0 * 1024.0));
  check(true, "the decoder bound its weights");

  auto g = [&](const std::string& n) {
    return npy::load(gdir + "/" + n + ".npy");
  };

  struct Case { const char* sfx; int frames; };
  // The 2-frame case still runs both upsamples and both first-row
  // drops (2 -> 3 -> 5); the 8-frame one is the same path at a size
  // where an off-by-one in the causal padding cannot hide.
  const Case cases[] = {{"_tiny", 2}, {"", 8}};

  for (const Case& c : cases) {
    npy::Array lat = g(std::string("audio_latent") + c.sfx);
    npy::Array mel = g(std::string("audio_mel") + c.sfx);
    if (!lat.ok || !mel.ok) {
      std::printf("  [SKIP] no audio goldens%s (run gen_goldens.py audio)\n",
                  c.sfx);
      continue;
    }
    std::printf("     --- F' = %d ---\n", c.frames);
    dec->set_capture(true);
    std::vector<float> got;
    std::array<int, 3> shape{};
    if (!dec->decode(lat.data.data(), c.frames, &got, &shape, &err)) {
      check(false, std::string("decode") + c.sfx + ": " + err);
      continue;
    }
    std::printf("       latent [%d,%d,%d] -> [%d,%d,%d]\n", ac.z_channels,
                c.frames, ac.latent_mel_bins(), shape[0], shape[1], shape[2]);
    check(shape[0] == ac.out_ch, "stereo output");
    // Two upsamples, each 2x-then-drop-a-row: F -> 2F-1 -> 4F-3. That
    // this equals the reference's target frame count exactly is why its
    // crop/pad step is always a no-op.
    check(shape[1] == 4 * c.frames - 3, "4F'-3 frames (two dropped rows)");
    check(shape[2] == ac.mel_bins, "64 mel bins");

    const char* taps[] = {"denorm", "conv_in", "mid", "up2", "up1", "up0"};
    bool localized = false;
    for (const char* t : taps) {
      npy::Array want = g(std::string("audio_") + t + c.sfx);
      const std::vector<float>* have = dec->tap(t);
      if (!want.ok || have == nullptr) { continue; }
      std::array<int, 3> s{};
      dec->tap_shape(t, &s);
      if (have->size() != want.data.size()) {
        check(false, std::string(t) + ": " + std::to_string(have->size()) +
                         " values vs the golden's " +
                         std::to_string(want.data.size()));
        localized = true;
        break;
      }
      const double r = npy::rel_l2(*have, want.data);
      std::printf("       %-8s [%d,%d,%d]  rel-L2 %.3e\n", t, s[0], s[1], s[2],
                  r);
      // The golden is f32 and this path is bf16, so the bar is bf16's
      // own rounding (~3.9e-3) with room for it to accumulate over ~20
      // convolutions -- NOT f32 agreement. A layout or padding error is
      // O(1) here, nowhere near this line.
      if (r >= 3e-2) {
        check(false, std::string("the tap after ") + t + " matches");
        localized = true;
        break;
      }
    }
    if (!localized) {
      check(true, "every intermediate matches the reference");
    }
    check(got.size() == mel.data.size(), "the spectrogram size matches");
    if (got.size() == mel.data.size()) {
      const double r = npy::rel_l2(got, mel.data);
      std::printf("       decoded mel     rel-L2 %.3e\n", r);
      check(r < 3e-2, "the decoded spectrogram matches the reference");
    }
  }

  std::printf("%s\n", g_fail == 0 ? "ALL PASSED" : "FAILURES");
  return g_fail == 0 ? 0 : 1;
}
