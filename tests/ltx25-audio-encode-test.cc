// The audio VAE ENCODER: waveform -> log-mel -> whitened latent rows.
//
// Checked in graph order, because the two halves fail differently and a
// whole-chain mismatch says nothing about which:
//
//   1. the mel FILTER BANK, against the matrix torchaudio built. A
//      wrong mel scale (htk vs slaney) and a wrong normalisation
//      (none vs slaney) both survive into a plausible spectrogram, and
//      only the matrix tells them apart;
//   2. the log-mel itself -- which additionally pins power=1, the
//      periodic hann window, centre+reflect padding and the 1e-5 log
//      clamp;
//   3. the conv encoder's taps, then the whitened ROWS, which is the
//      form `generate-video`'s ref_audio_rows takes.
//
// Needs VPIPE_LTX25_TEST_MODEL_PATH and VPIPE_LTX25_GOLDENS; without
// them it SKIPS and says so.

#include "ltx25-audio-encoder.h"
#include "ltx25-audio-vae.h"
#include "ltx25-vocoder.h"
#include "ltx25-config.h"
#include "ltx25-metal-ops.h"
#include "npy.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/flex-data.h"
#include "generative-models/shared/comfy-checkpoint.h"
#include "generative-models/weight-set.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

extern "C" const unsigned char ltx25_kernels_bf16_metallib[];
extern "C" const unsigned long ltx25_kernels_bf16_metallib_len;
// The vocoder -- used only by the round trip at the end -- runs in F32
// and reads a metallib twin of its own. Registering only the bf16 one
// leaves its entry points unresolved, and an unvalidated
// ComputeFunction dispatches as a silent no-op.
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
  auto g = [&](const std::string& n) {
    return npy::load(gdir + "/" + n + ".npy");
  };

  // ---- 1: the filter bank ----------------------------------------------
  //
  // Built from the scale definition, not read from the checkpoint --
  // unlike the BWE's, LTX ships no mel bases for the encoder, so this
  // matrix is the port's own and has to be pinned on its own.
  ltx25::Ltx25AudioMel::Config mcfg;      // the checkpoint's values
  ltx25::Ltx25AudioMel mel(mcfg);
  {
    npy::Array want = g("aenc_mel_basis");
    if (!want.ok) {
      std::printf("  [SKIP] no aenc goldens (run gen_goldens.py aenc): %s\n",
                  want.err.c_str());
      std::printf("%s\n", g_fail == 0 ? "ALL PASSED" : "FAILURES");
      return g_fail == 0 ? 0 : 1;
    }
    check(mel.filters().size() == want.data.size(),
          "the filter bank is [" + std::to_string(mcfg.mel_bins) + ", " +
          std::to_string(mel.n_freqs()) + "]");
    if (mel.filters().size() == want.data.size()) {
      const double r = npy::rel_l2(mel.filters(), want.data);
      std::printf("       slaney mel filters  rel-L2 %.3e\n", r);
      // The golden is an f32 tensor and this bank is built in double
      // and narrowed, so the residual is the golden's own rounding --
      // concentrated in the near-zero filter tails, where a relative
      // measure has nothing to divide by. The LOG-MEL below is the
      // check with teeth: it applies this bank and lands an order of
      // magnitude closer.
      check(r < 1e-5, "the mel filter bank matches torchaudio's");
    }
  }

  // ---- 2: the log-mel ---------------------------------------------------
  npy::Array wav = g("aenc_wav");
  npy::Array wmel = g("aenc_mel");
  if (!wav.ok || !wmel.ok) {
    check(false, "load goldens: " + wav.err + wmel.err);
    std::printf("FAILURES\n");
    return 1;
  }
  const int CH = (int)wav.shape[0], NS = (int)wav.shape[1];
  std::vector<float> got_mel;
  std::array<int, 3> mshape{};
  {
    const auto t0 = std::chrono::steady_clock::now();
    mel.compute(wav.data.data(), CH, NS, &got_mel, &mshape);
    const double dt =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count();
    std::printf("       [%d,%d] -> mel [%d,%d,%d] in %.2f s\n", CH, NS,
                mshape[0], mshape[1], mshape[2], dt);
  }
  check(mshape[1] == (int)wmel.shape[1] && mshape[2] == (int)wmel.shape[2],
        "the mel has the reference's frame and bin counts");
  check(got_mel.size() == wmel.data.size(), "the mel element count matches");
  if (got_mel.size() == wmel.data.size()) {
    const double r = npy::rel_l2(got_mel, wmel.data);
    std::printf("       log-mel  rel-L2 %.3e\n", r);
    // f32 host arithmetic against torch's, over a 1024-point DFT: the
    // bar is round-off, not a tolerance for a different convention.
    check(r < 1e-5, "the log-mel matches the reference");
  }

  // ---- 3: the conv encoder ---------------------------------------------
  vpipe::metal_compute::MetalCompute mc(nullptr);
  if (!mc.valid()) {
    std::printf("SKIPPED the encoder: no Metal device.\n");
    std::printf("%s\n", g_fail == 0 ? "ALL PASSED" : "FAILURES");
    return g_fail == 0 ? 0 : 1;
  }
  mc.register_metal_library(ltx25::kMetalLibBf16, ltx25_kernels_bf16_metallib,
                            ltx25_kernels_bf16_metallib_len);
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
    check(false, "this checkpoint has no audio VAE");
    std::printf("FAILURES\n");
    return 1;
  }
  vpipe::FlexData acfg_meta;
  if (!vpipe::genai::comfy::metadata_json(
          cfg.audio_vae_file, ltx25::kVaeMetaKey, acfg_meta, &err)) {
    check(false, "audio VAE metadata: " + err);
    std::printf("FAILURES\n");
    return 1;
  }
  ltx25::AudioVaeConfig av;
  if (!ltx25::parse_audio_vae_config(acfg_meta, av, &err)) {
    check(false, "parse_audio_vae_config: " + err);
    std::printf("FAILURES\n");
    return 1;
  }
  ltx25::MetalOps ops;
  if (!ops.init(&mc, &err)) {
    check(false, "MetalOps init: " + err);
    std::printf("FAILURES\n");
    return 1;
  }
  auto ws = vpipe::genai::open_weight_set(cfg.audio_vae_file, nullptr);
  if (!ws) {
    check(false, "could not open " + cfg.audio_vae_file);
    std::printf("FAILURES\n");
    return 1;
  }
  auto enc = ltx25::Ltx25AudioVaeEncoder::load(av, ws, ops, &err);
  if (!enc) {
    check(false, "encoder load: " + err);
    std::printf("FAILURES\n");
    return 1;
  }
  check(true, "the audio encoder bound its weights (" +
        std::to_string(enc->resident_bytes() / (1024 * 1024)) + " MB)");

  enc->set_capture(true);
  std::vector<float> rows;
  std::array<int, 2> rshape{};
  const auto t1 = std::chrono::steady_clock::now();
  // The GOLDEN's mel, not the one just computed: this half is being
  // checked on its own, and feeding it a mel that is itself under test
  // would let two errors cancel.
  if (!enc->encode(wmel.data.data(), (int)wmel.shape[1], &rows, &rshape,
                   &err)) {
    check(false, "encode: " + err);
    std::printf("FAILURES\n");
    return 1;
  }
  {
    const double dt =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t1)
            .count();
    std::printf("       mel [%d,%d,%d] -> rows [%d,%d] in %.3f s\n",
                (int)wmel.shape[0], (int)wmel.shape[1], (int)wmel.shape[2],
                rshape[0], rshape[1], dt);
  }

  npy::Array wrows = g("aenc_rows");
  npy::Array wlat  = g("aenc_latent");
  if (wrows.ok) {
    check(rshape[0] == (int)wrows.shape[0] &&
          rshape[1] == (int)wrows.shape[1],
          "the rows have the reference's shape");
  }
  if (wlat.ok) {
    check(rshape[1] == (int)wlat.shape[0] * (int)wlat.shape[2],
          "the row width is z_channels x latent mel bins (" +
          std::to_string((int)wlat.shape[0]) + " x " +
          std::to_string((int)wlat.shape[2]) + ")");
  }

  // Taps, in graph order: the first mismatch is the block that broke.
  const char* taps[] = {"conv_in", "head"};
  for (const char* t : taps) {
    npy::Array w = g(std::string("aenc_") + t);
    const std::vector<float>* got = enc->tap(t);
    if (!w.ok || got == nullptr) { continue; }
    std::array<int, 3> s{};
    enc->tap_shape(t, &s);
    if (got->size() != w.data.size()) {
      check(false, std::string(t) + ": " + std::to_string(got->size()) +
            " values vs the golden's " + std::to_string(w.data.size()));
      break;
    }
    const double r = npy::rel_l2(*got, w.data);
    std::printf("       %-8s [%d,%d,%d]  rel-L2 %.3e\n", t, s[0], s[1], s[2],
                r);
    check(r < 3e-2, std::string("the tap after ") + t + " matches");
  }

  // The head tap's number is FLATTERING and must not be the bar. Its 16
  // channels are 8 means and 8 log-variances, and the log-variances are
  // ~5x the magnitude -- so a whole-tensor rel-L2 is dominated by the
  // half that gets discarded. Report the MEANS alone, which is what the
  // latent is made of.
  {
    npy::Array wh = g("aenc_head");
    const std::vector<float>* got = enc->tap("head");
    std::array<int, 3> s{};
    if (wh.ok && got != nullptr && enc->tap_shape("head", &s) &&
        got->size() == wh.data.size() && s[0] >= av.z_channels) {
      const std::size_t plane = (std::size_t)s[1] * s[2];
      const std::size_t nz = (std::size_t)av.z_channels * plane;
      std::vector<float> gm(got->begin(), got->begin() + nz);
      std::vector<float> wm(wh.data.begin(), wh.data.begin() + nz);
      std::vector<float> gv(got->begin() + nz, got->end());
      std::vector<float> wv(wh.data.begin() + nz, wh.data.end());
      std::printf("       head MEANS   rel-L2 %.3e\n", npy::rel_l2(gm, wm));
      std::printf("       head logvar  rel-L2 %.3e (discarded)\n",
                  npy::rel_l2(gv, wv));
    }
  }

  if (wrows.ok && rows.size() == wrows.data.size()) {
    const double r = npy::rel_l2(rows, wrows.data);
    std::printf("       whitened rows  rel-L2 %.3e\n", r);
    // MEASURED 4.0e-2, and it is bf16 depth rather than the whitening:
    // reconstructing these rows from the GOLDEN head with this port's
    // own (channel, mel-bin) indexing reproduces the golden exactly
    // (rel-L2 0.0), and the whitening's own amplification is 1.03x. What
    // is left is ~20 sequential convolutions at up to 512 channels in
    // bf16, from a per-convolution floor the conv_in tap puts at 2.8e-3.
    //
    // Higher than the audio DECODER's 1.7e-3..6.0e-3 because that
    // number is taken on a log-mel, whose dynamic range flatters a
    // relative measure; this one is taken on a whitened latent that is
    // O(1) everywhere.
    check(r < 6e-2, "the encoded rows match the reference");
  } else if (wrows.ok) {
    check(false, "the row element count matches");
  }

  // Two structural facts a numeric bar cannot state. A NaN or a
  // constant latent would sail past a rel-L2 that is already 4e-2.
  {
    std::size_t bad = 0;
    double lo = rows.empty() ? 0.0 : rows[0], hi = lo;
    for (float v : rows) {
      if (!std::isfinite(v)) { ++bad; }
      lo = std::min(lo, (double)v);
      hi = std::max(hi, (double)v);
    }
    check(bad == 0, "every row value is finite (" + std::to_string(bad) +
          " bad)");
    check(hi - lo > 1e-3, "the rows are not constant (range " +
          std::to_string(hi - lo) + ")");
  }

  // ---- 4: a ROUND TRIP through the decoder ------------------------------
  //
  // The golden pins this encoder against the reference; the round trip
  // pins it against the DECODER, which is the property a reference
  // soundtrack actually needs -- the DiT is conditioned in one latent
  // space and both halves have to speak it. It is also the only test
  // here that runs the mel front end on real audio rather than on noise.
  //
  // The bar is CORRELATION, not rel-L2: a VAE round trip is lossy by
  // construction and 4x temporal compression through a vocoder and back
  // is not an identity. What it can catch is a sign flip, a scale error,
  // a transposed layout or a whitening applied the wrong way -- all of
  // which would put the correlation near zero.
  {
    std::printf("  --- round trip: latent -> mel -> 16 kHz -> latent ---\n");
    npy::Array lat = g("aenc_latent");
    auto dec = ltx25::Ltx25AudioVaeDecoder::load(av, ws, ops, &err);
    ltx25::VocoderConfig vcfg;
    std::unique_ptr<ltx25::Ltx25Vocoder> voc;
    if (lat.ok && dec &&
        ltx25::parse_vocoder_config(acfg_meta, "vocoder", vcfg, &err)) {
      voc = ltx25::Ltx25Vocoder::load(vcfg, ws, &mc, "vocoder.vocoder.",
                                      &err);
    }
    if (!lat.ok || !dec || !voc) {
      std::printf("  [SKIP] round trip: %s\n", err.c_str());
    } else {
      std::vector<float> mel2;
      std::array<int, 3> ms{};
      std::vector<float> pcm;
      std::array<int, 2> ps{};
      if (!dec->decode(lat.data.data(), (int)lat.shape[1], &mel2, &ms,
                       &err) ||
          !voc->synthesize(mel2.data(), ms[1], ms[2], &pcm, &ps, &err)) {
        check(false, "round-trip decode: " + err);
      } else {
        std::printf("       latent [%d,%d,%d] -> mel [%d,%d,%d] -> pcm "
                    "[%d,%d] at %d Hz\n", (int)lat.shape[0],
                    (int)lat.shape[1], (int)lat.shape[2], ms[0], ms[1],
                    ms[2], ps[0], ps[1], vcfg.output_sampling_rate);
        check(vcfg.output_sampling_rate == mcfg.sample_rate,
              "the vocoder's rate is the one the encoder's mel wants");
        std::vector<float> rmel;
        std::array<int, 3> rms{};
        mel.compute(pcm.data(), ps[0], ps[1], &rmel, &rms);
        std::vector<float> back;
        std::array<int, 2> bs{};
        if (!enc->encode(rmel.data(), rms[1], &back, &bs, &err)) {
          check(false, "round-trip encode: " + err);
        } else {
          std::printf("       back to rows [%d,%d]\n", bs[0], bs[1]);
          check(bs[1] == rshape[1], "the round trip keeps the row width");
          const int n = std::min(bs[0], rshape[0]);
          double sxy = 0.0, sxx = 0.0, syy = 0.0;
          for (int t = 0; t < n; ++t) {
            for (int d = 0; d < bs[1]; ++d) {
              const double a = back[(std::size_t)t * bs[1] + d];
              const double b = wrows.data[(std::size_t)t * bs[1] + d];
              sxy += a * b; sxx += a * a; syy += b * b;
            }
          }
          const double corr =
              (sxx > 0 && syy > 0) ? sxy / std::sqrt(sxx * syy) : 0.0;
          std::printf("       round-trip correlation %.4f over %d rows\n",
                      corr, n);
          check(corr > 0.5, "the re-encoded reference correlates with the "
                "original latent");
        }
      }
    }
  }

  std::printf("%s\n", g_fail == 0 ? "ALL PASSED" : "FAILURES");
  return g_fail == 0 ? 0 : 1;
}
