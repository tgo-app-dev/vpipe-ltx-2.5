// The BigVGAN vocoder (Metal, f32) against the real weights.
//
// The ANTI-ALIASED ACTIVATION is checked FIRST and on its own, before
// anything stacks 108 of them: upsample 2x -> SnakeBeta -> lowpass
// downsample 2x, with a tap after each step. It is the novel op here,
// and inside the full stack an error in it is indistinguishable from an
// error anywhere else.
//
// Then the whole vocoder, with taps in graph order.
//
// This path is f32 END TO END on purpose -- see ltx25-vocoder.h. The
// reference forces fp32 through its 108 sequential convolutions because
// bf16 accumulation costs 40-90% of the spectral metrics, so the bar
// here is f32 agreement (~1e-5), not bf16 rounding.

#include "ltx25-audio-vae.h"
#include "ltx25-config.h"
#include "ltx25-metal-ops.h"
#include "ltx25-vocoder.h"

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

extern "C" const unsigned char ltx25_kernels_bf16_metallib[];
extern "C" const unsigned long ltx25_kernels_bf16_metallib_len;
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
  // The f32 twin, not the bf16 one -- this is the whole reason the
  // vocoder has a metallib of its own.
  mc.register_metal_library(ltx25::kMetalLibF32, ltx25_kernels_f32_metallib,
                            ltx25_kernels_f32_metallib_len);
  // The audio VAE decoder ahead of it runs bf16, so the chain check at
  // the end needs both twins registered.
  mc.register_metal_library(ltx25::kMetalLibBf16, ltx25_kernels_bf16_metallib,
                            ltx25_kernels_bf16_metallib_len);
  ltx25::MetalOps ops;
  {
    std::string oerr;
    if (!ops.init(&mc, &oerr)) {
      std::printf("SKIPPED: MetalOps::init: %s\n", oerr.c_str());
      return 0;
    }
  }

  ltx25::Config cfg;
  std::string err;
  if (!ltx25::resolve(root, cfg, &err)) {
    check(false, "resolve: " + err);
    std::printf("FAILURES\n");
    return 1;
  }
  if (cfg.audio_vae_file.empty()) {
    std::printf("SKIPPED: this checkpoint has no audio VAE / vocoder. "
                "NOTHING was checked.\n");
    return 0;
  }
  vpipe::FlexData vcfg;
  if (!vpipe::genai::comfy::metadata_json(cfg.audio_vae_file,
                                          ltx25::kVaeMetaKey, vcfg, &err)) {
    check(false, "vocoder metadata: " + err);
    std::printf("FAILURES\n");
    return 1;
  }
  ltx25::VocoderConfig vc;
  if (!ltx25::parse_vocoder_config(vcfg, "vocoder", vc, &err)) {
    check(false, "parse_vocoder_config: " + err);
    std::printf("FAILURES\n");
    return 1;
  }
  check(vc.resblock == "AMP1", "AMP1 resblocks (BigVGAN, not HiFiGAN)");
  check(vc.activation == "snakebeta", "snakebeta activation");
  check(!vc.use_tanh_at_final, "the final activation is a CLAMP, not tanh");
  check(!vc.use_bias_at_final, "conv_post has no bias");
  check(vc.upsample_initial_channel == 1536, "1536 initial channels");
  check(vc.num_upsamples() == 6, "6 upsample stages");
  check(vc.num_kernels() == 3, "3 resblocks per stage (mean-aggregated)");
  // The rates multiply to the mel hop, which is what ties the vocoder
  // to the audio VAE's 160-sample frame.
  check(vc.hop() == 160, "the rates multiply to the 160-sample hop");
  // The main vocoder's section carries NO sample rate; the reference
  // takes it from the BWE's input rate. Defaulting it to 24 kHz gives
  // correct samples that play 1.5x fast.
  check(vc.output_sampling_rate == 16000,
        "16 kHz out (taken from the BWE's input rate, not defaulted)");

  auto ws = vpipe::genai::open_weight_set(cfg.audio_vae_file, nullptr);
  if (!ws) {
    check(false, "could not open " + cfg.audio_vae_file);
    std::printf("FAILURES\n");
    return 1;
  }
  auto voc = ltx25::Ltx25Vocoder::load(vc, ws, &mc, "vocoder.vocoder.", &err);
  if (!voc) {
    check(false, "load: " + err);
    std::printf("FAILURES\n");
    return 1;
  }
  std::printf("       vocoder weights %.1f MB (f32, widened from bf16)\n",
              (double)voc->resident_bytes() / (1024.0 * 1024.0));
  check(true, "the vocoder bound its weights");

  auto g = [&](const std::string& n) {
    return npy::load(gdir + "/" + n + ".npy");
  };

  // ---- the anti-aliased activation, ON ITS OWN ----------------------
  //
  // Before 108 of them are stacked. A tap after each of the three steps,
  // so a failure says which: the kaiser-sinc upsample, the LOG-SCALE
  // SnakeBeta, or the lowpass downsample.
  {
    npy::Array ain = g("voc_act_in"), aup = g("voc_act_up"),
               asn = g("voc_act_snake"), aout = g("voc_act_out");
    if (ain.ok && aup.ok && asn.ok && aout.ok && ain.shape.size() == 2) {
      const int C = ain.shape[0], T = ain.shape[1];
      std::vector<float> up, sn, dn;
      if (!voc->probe_activation(ain.data.data(), C, T, &up, &sn, &dn,
                                 &err)) {
        check(false, "probe_activation: " + err);
      } else {
        const double r1 = npy::rel_l2(up, aup.data);
        const double r2 = npy::rel_l2(sn, asn.data);
        const double r3 = npy::rel_l2(dn, aout.data);
        std::printf("       act: up %.3e  snake %.3e  down %.3e\n", r1, r2,
                    r3);
        check(r1 < 2e-4, "the 2x kaiser-sinc upsample matches");
        check(r2 < 2e-4, "SnakeBeta matches (alpha/beta are LOG scale)");
        check(r3 < 2e-4, "the lowpass downsample matches");
      }
    } else {
      std::printf("  [SKIP] no activation goldens\n");
    }
  }

  npy::Array mel = g("voc_mel");
  npy::Array wav = g("voc_wav");
  if (!mel.ok || !wav.ok) {
    std::printf("  [SKIP] no vocoder goldens (run gen_goldens.py vocoder)\n");
    std::printf("%s\n", g_fail == 0 ? "ALL PASSED" : "FAILURES");
    return g_fail == 0 ? 0 : 1;
  }

  const int T = 16, MB = 64;
  voc->set_capture(true);
  std::vector<float> got;
  std::array<int, 2> shape{};
  const auto t0 = std::chrono::steady_clock::now();
  if (!voc->synthesize(mel.data.data(), T, MB, &got, &shape, &err)) {
    check(false, "synthesize: " + err);
    std::printf("FAILURES\n");
    return 1;
  }
  const double ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t0).count();
  std::printf("       mel [2,%d,%d] -> [%d,%d] in %.0f ms\n", T, MB, shape[0],
              shape[1], ms);
  check(shape[0] == 2, "stereo waveform");
  check(shape[1] == T * vc.hop(), "160 samples per mel frame");

  const char* taps[] = {"conv_pre", "ups0", "rb0", "ups1", "ups5",
                        "act_post"};
  bool localized = false;
  for (const char* t : taps) {
    npy::Array want = g(std::string("voc_") + t);
    const std::vector<float>* have = voc->tap(t);
    if (!want.ok || have == nullptr) { continue; }
    std::array<int, 2> s{};
    voc->tap_shape(t, &s);
    if (have->size() != want.data.size()) {
      check(false, std::string(t) + ": " + std::to_string(have->size()) +
                       " values vs the golden's " +
                       std::to_string(want.data.size()));
      localized = true;
      break;
    }
    const double r = npy::rel_l2(*have, want.data);
    std::printf("       %-9s [%d,%d]  rel-L2 %.3e\n", t, s[0], s[1], r);
    // f32 both sides: this is arithmetic-order noise only, and the bar
    // says so. A layout, padding or log-scale error is O(1).
    if (r >= 2e-4) {
      check(false, std::string("the tap after ") + t + " matches");
      localized = true;
      break;
    }
  }
  if (!localized) {
    check(true, "every intermediate matches the reference");
  }
  check(got.size() == wav.data.size(), "the sample count matches");
  if (got.size() == wav.data.size()) {
    const double r = npy::rel_l2(got, wav.data);
    std::printf("       waveform  rel-L2 %.3e\n", r);
    check(r < 2e-4, "the waveform matches the reference");
  }

  // ---- the WHOLE audio chain, at a realistic length ------------------
  //
  // latent -> log-mel -> waveform, which is the first time the two
  // halves meet. Also the only case at a size the goldens do not cover:
  // 5.12 s is what the checkpoint's preprocessing config says a clip
  // is, and it is where the im2col chunking and the f32 residency
  // actually matter.
  {
    ltx25::AudioVaeConfig ac;
    if (!ltx25::parse_audio_vae_config(vcfg, ac, &err)) {
      check(false, "parse_audio_vae_config: " + err);
    } else {
      auto dec = ltx25::Ltx25AudioVaeDecoder::load(ac, ws, ops, &err);
      if (!dec) {
        check(false, "audio decoder load: " + err);
      } else {
        // 5.12 s at 16 kHz / hop 160 = 512 mel frames, so the latent
        // is (512 + 3) / 4 frames.
        const int F = (512 + 3) / 4;
        std::vector<float> lat((std::size_t)ac.z_channels * F *
                                   ac.latent_mel_bins(),
                               0.0f);
        for (std::size_t i = 0; i < lat.size(); ++i) {
          lat[i] = 0.6f * std::sin(0.017f * (float)i);
        }
        std::vector<float> m2;
        std::array<int, 3> ms{};
        const auto c0 = std::chrono::steady_clock::now();
        if (!dec->decode(lat.data(), F, &m2, &ms, &err)) {
          check(false, "chain decode: " + err);
        } else {
          std::vector<float> w2;
          std::array<int, 2> wsh{};
          if (!voc->synthesize(m2.data(), ms[1], ms[2], &w2, &wsh, &err)) {
            check(false, "chain synthesize: " + err);
          } else {
            const double cms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - c0).count();
            const double secs =
                (double)wsh[1] / (double)vc.output_sampling_rate;
            std::printf("       chain: latent[%d,%d,%d] -> mel[%d,%d,%d] -> "
                        "wav[%d,%d] = %.2f s audio in %.0f ms\n",
                        ac.z_channels, F, ac.latent_mel_bins(), ms[0], ms[1],
                        ms[2], wsh[0], wsh[1], secs, cms);
            check(wsh[1] == ms[1] * vc.hop(),
                  "the chain's sample count is mel frames x 160");
            // Silence would also be "in range", so check it is not.
            double peak = 0.0, energy = 0.0;
            for (float v : w2) {
              peak = std::max(peak, (double)std::fabs(v));
              energy += (double)v * (double)v;
            }
            check(peak <= 1.0, "the waveform is clamped to [-1, 1]");
            check(energy > 0.0, "the waveform is not silence");
          }
        }
      }
    }
  }

  std::printf("%s\n", g_fail == 0 ? "ALL PASSED" : "FAILURES");
  return g_fail == 0 ? 0 : 1;
}
