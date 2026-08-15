// The BWE stage (Metal, f32) against the real weights: 16 -> 48 kHz.
//
// Checked in chain order, because each step feeds the next and the
// FIRST mismatch is the cause:
//
//   low       the main vocoder's 16 kHz output (already verified on its
//             own by ltx25-vocoder-test, so a failure here is the pad)
//   mel2      the CAUSAL log-mel of it -- 432 samples of ZERO padding on
//             the LEFT ONLY. A centred STFT is a fine spectrogram that
//             shifts everything downstream in time.
//   residual  the second generator's output
//   skip      the x3 sinc resample. Its filter is the one tensor in this
//             chain with NO checkpoint to check against -- it is
//             persistent=False in the reference and rebuilt from a HANN
//             window, not the kaiser one the activations use -- so it is
//             compared against a golden of its own.

#include "ltx25-bwe.h"
#include "ltx25-config.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/flex-data.h"
#include "generative-models/shared/comfy-checkpoint.h"
#include "generative-models/weight-set.h"
#include "npy.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

extern "C" const unsigned char ltx25_kernels_f32_metallib[];
extern "C" const unsigned long ltx25_kernels_f32_metallib_len;

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
  mc.register_metal_library(ltx25::kMetalLibF32, ltx25_kernels_f32_metallib,
                            ltx25_kernels_f32_metallib_len);

  ltx25::Config cfg;
  std::string err;
  if (!ltx25::resolve(root, cfg, &err)) {
    check(false, "resolve: " + err);
    std::printf("FAILURES\n");
    return 1;
  }
  if (cfg.audio_vae_file.empty()) {
    std::printf("SKIPPED: no audio VAE / vocoder. NOTHING was checked.\n");
    return 0;
  }
  vpipe::FlexData vcfg;
  if (!vpipe::genai::comfy::metadata_json(cfg.audio_vae_file,
                                          ltx25::kVaeMetaKey, vcfg, &err)) {
    check(false, "vocoder metadata: " + err);
    std::printf("FAILURES\n");
    return 1;
  }
  ltx25::BweConfig bc;
  if (!ltx25::parse_bwe_config(vcfg, bc, &err)) {
    std::printf("SKIPPED: %s. NOTHING was checked.\n", err.c_str());
    return 0;
  }
  check(bc.input_sampling_rate == 16000, "16 kHz in");
  check(bc.output_sampling_rate == 48000, "48 kHz out");
  check(bc.ratio() == 3, "a x3 ratio");
  check(bc.n_fft == 512 && bc.hop_length == 80, "512-point FFT, hop 80");
  check(bc.n_freqs() == 257, "257 frequency bins");

  auto ws = vpipe::genai::open_weight_set(cfg.audio_vae_file, nullptr);
  if (!ws) {
    check(false, "could not open " + cfg.audio_vae_file);
    std::printf("FAILURES\n");
    return 1;
  }
  auto bwe = ltx25::Ltx25Bwe::load(vcfg, ws, &mc, &err);
  if (!bwe) {
    check(false, "load: " + err);
    std::printf("FAILURES\n");
    return 1;
  }
  std::printf("       both generators %.1f MB (f32)\n",
              (double)bwe->resident_bytes() / (1024.0 * 1024.0));
  check(true, "the BWE bound its weights");

  auto g = [&](const std::string& n) {
    return npy::load(gdir + "/" + n + ".npy");
  };

  // The rebuilt HANN-window sinc, FIRST: nothing else in this chain has
  // to be reconstructed rather than read, so it is the one place a
  // silent divergence from the reference can start.
  {
    npy::Array want = g("bwe_resampler_filter");
    if (want.ok) {
      const std::vector<float>& have = bwe->resampler_filter();
      check(have.size() == want.data.size(),
            "the resampler filter is " + std::to_string(want.data.size()) +
                " taps");
      if (have.size() == want.data.size()) {
        const double r = npy::rel_l2(have, want.data);
        std::printf("       resampler filter  rel-L2 %.3e\n", r);
        check(r < 1e-6, "the rebuilt HANN-window sinc matches the reference");
      }
    }
  }

  npy::Array mel = g("bwe_mel_in");
  npy::Array out = g("bwe_out");
  if (!mel.ok || !out.ok) {
    std::printf("  [SKIP] no bwe goldens (run gen_goldens.py bwe)\n");
    std::printf("%s\n", g_fail == 0 ? "ALL PASSED" : "FAILURES");
    return g_fail == 0 ? 0 : 1;
  }

  const int T = 16, MB = 64;
  bwe->set_capture(true);
  std::vector<float> got;
  std::array<int, 2> shape{};
  const auto t0 = std::chrono::steady_clock::now();
  if (!bwe->synthesize(mel.data.data(), T, MB, &got, &shape, &err)) {
    check(false, "synthesize: " + err);
    std::printf("FAILURES\n");
    return 1;
  }
  const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
  std::printf("       mel [2,%d,%d] -> 16 kHz [2,%d] -> 48 kHz [%d,%d] "
              "in %.0f ms\n", T, MB, bwe->low_rate_samples(), shape[0],
              shape[1], ms);
  check(shape[0] == 2, "stereo");
  check(shape[1] == bwe->low_rate_samples() * 3, "3x the 16 kHz length");

  const char* taps[] = {"low", "mel2", "residual", "skip"};
  bool localized = false;
  for (const char* t : taps) {
    npy::Array want = g(std::string("bwe_") + t);
    const std::vector<float>* have = bwe->tap(t);
    if (!want.ok || have == nullptr) { continue; }
    if (have->size() != want.data.size()) {
      check(false, std::string(t) + ": " + std::to_string(have->size()) +
                       " values vs the golden's " +
                       std::to_string(want.data.size()));
      localized = true;
      break;
    }
    const double r = npy::rel_l2(*have, want.data);
    std::printf("       %-9s rel-L2 %.3e\n", t, r);
    if (r >= 2e-4) {
      check(false, std::string("the ") + t + " matches");
      localized = true;
      break;
    }
  }
  if (!localized) { check(true, "every intermediate matches the reference"); }

  check(got.size() == out.data.size(), "the sample count matches");
  if (got.size() == out.data.size()) {
    const double r = npy::rel_l2(got, out.data);
    std::printf("       48 kHz waveform  rel-L2 %.3e\n", r);
    check(r < 2e-4, "the 48 kHz waveform matches the reference");
  }

  // ---- at a realistic length -----------------------------------------
  //
  // The goldens run 16 mel frames; a real clip is 512. This is where the
  // STFT's frame count and the x3 resample are at their working size,
  // and it is the number worth quoting.
  {
    const int T2 = 509;                 // what a 5.12 s latent decodes to
    std::vector<float> big((std::size_t)2 * T2 * MB);
    for (std::size_t i = 0; i < big.size(); ++i) {
      big[i] = -1.0f + 0.7f * std::sin(0.013f * (float)i);
    }
    bwe->set_capture(false);
    std::vector<float> w;
    std::array<int, 2> sh{};
    const auto b0 = std::chrono::steady_clock::now();
    if (!bwe->synthesize(big.data(), T2, MB, &w, &sh, &err)) {
      check(false, "big synthesize: " + err);
    } else {
      const double bms = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - b0).count();
      const double secs = (double)sh[1] / (double)bwe->output_sampling_rate();
      std::printf("       %d frames -> [%d,%d] = %.2f s at 48 kHz in "
                  "%.0f ms\n", T2, sh[0], sh[1], secs, bms);
      check(sh[1] == T2 * 160 * 3, "48 kHz length is mel frames x 160 x 3");
      double peak = 0.0, energy = 0.0;
      for (float v : w) {
        peak = std::max(peak, (double)std::fabs(v));
        energy += (double)v * (double)v;
      }
      check(peak <= 1.0, "clamped to [-1, 1]");
      check(energy > 0.0, "not silence");
    }
  }

  std::printf("%s\n", g_fail == 0 ? "ALL PASSED" : "FAILURES");
  return g_fail == 0 ? 0 : 1;
}
