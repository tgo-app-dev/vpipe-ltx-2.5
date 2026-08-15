// The conv video VAE decoder, CPU reference, against the real weights.
//
// Runs the TINY golden (latent [128,1,1,1]) because a from-scratch f32
// CPU decode of the [128,2,2,2] one is ~20x the work for the same
// semantics. Every block, every upsample kind and every layout op is
// exercised either way.
//
// The intermediates are checked BEFORE the output, in graph order, so a
// mismatch names the block that caused it rather than just the picture.

#include "ltx25-config.h"
#include "ltx25-vae-config.h"
#include "ltx25-vae-ref.h"
#include "ltx25-vae.h"
#include "ltx25-metal-ops.h"

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
  // Unbuffered: this test drives GPU kernels, and a crash with a block-
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
  // The plugin's own metallib has to be registered before MetalOps can
  // resolve anything out of it.
  mc.register_metal_library(ltx25::kMetalLibBf16, ltx25_kernels_bf16_metallib,
                            ltx25_kernels_bf16_metallib_len);

  ltx25::Config cfg;
  std::string err;
  if (!ltx25::resolve(root, cfg, &err)) {
    check(false, "resolve: " + err);
    std::printf("FAILURES\n");
    return 1;
  }
  if (cfg.vae_file.empty()) {
    check(false, "this checkpoint has no conv video VAE");
    std::printf("FAILURES\n");
    return 1;
  }

  vpipe::FlexData vcfg;
  if (!vpipe::genai::comfy::metadata_json(cfg.vae_file, ltx25::kVaeMetaKey,
                                          vcfg, &err)) {
    check(false, "VAE metadata: " + err);
    std::printf("FAILURES\n");
    return 1;
  }
  ltx25::VaeConfig vc;
  if (!ltx25::parse_vae_config(vcfg, vc, &err)) {
    check(false, "parse_vae_config: " + err);
    std::printf("FAILURES\n");
    return 1;
  }
  check(vc.class_name == "CausalVideoAutoencoder", "it is a conv video VAE");
  check(vc.latent_channels == 128, "128 latent channels");
  check(vc.patch_size == 4, "patch size 4");
  check(!vc.causal_decoder, "the decoder is NOT causal (symmetric time pad)");
  check(!vc.timestep_conditioning,
        "no timestep conditioning (so the decode is deterministic)");
  // Derived from the block list, not hardcoded -- the same way the
  // reference derives them.
  check(vc.spatial_factor() == 32, "32x spatial (8 from blocks, 4 from patch)");
  check(vc.temporal_factor() == 8, "8x temporal");
  check(vc.bottleneck_channels() == 1024, "conv_in projects to 1024");
  const auto ups = vc.up_blocks();
  check(ups.size() == 9, "9 up blocks");
  // up_blocks are decoder_blocks REVERSED; getting that backwards binds
  // the wrong tensors, so state it.
  check(ups.front().is_res() && ups.front().num_layers == 2,
        "up_blocks[0] is the LAST decoder block (res_x, 2 layers)");

  auto ws = vpipe::genai::open_weight_set(cfg.vae_file, nullptr);
  if (!ws) {
    check(false, "could not open " + cfg.vae_file);
    std::printf("FAILURES\n");
    return 1;
  }
  auto dec = ltx25::Ltx25VaeRef::load(vc, *ws, &mc, &err);
  if (!dec) {
    check(false, "load: " + err);
    std::printf("FAILURES\n");
    return 1;
  }
  check(true, "the decoder bound its weights");

  auto g = [&](const std::string& n) { return npy::load(gdir + "/" + n + ".npy"); };
  npy::Array lat = g("vae_latent_tiny");
  npy::Array out = g("vae_out_tiny");
  if (!lat.ok || !out.ok) {
    std::printf("  [SKIP] no vae goldens (run gen_goldens.py vae): %s%s\n",
                lat.err.c_str(), out.err.c_str());
    std::printf("%s\n", g_fail == 0 ? "ALL PASSED" : "FAILURES");
    return g_fail == 0 ? 0 : 1;
  }

  const int F = 1, H = 1, W = 1;
  dec->set_capture(true);
  std::vector<float> pix;
  std::array<int, 4> shape{};
  if (!dec->decode(lat.data.data(), F, H, W, &pix, &shape, &err)) {
    check(false, "decode: " + err);
    std::printf("FAILURES\n");
    return 1;
  }
  std::printf("       latent [128,%d,%d,%d] -> [%d,%d,%d,%d]\n", F, H, W,
              shape[0], shape[1], shape[2], shape[3]);
  check(shape[0] == 3, "3 output channels");
  check(shape[1] == 8 * (F - 1) + 1, "F = 8*(F'-1)+1 frames");
  check(shape[2] == 32 * H && shape[3] == 32 * W, "32x spatial");

  // Intermediates in GRAPH ORDER. The first one that fails is the block
  // that broke; the ones after it are consequences.
  const char* taps[] = {"conv_in", "up1", "up3", "up5", "up7", "up8"};
  bool localized = false;
  for (const char* t : taps) {
    npy::Array want = g(std::string("vae_") + t + "_tiny");
    const std::vector<float>* got = dec->tap(t);
    if (!want.ok || got == nullptr) { continue; }
    std::array<int, 4> s{};
    dec->tap_shape(t, &s);
    if (got->size() != want.data.size()) {
      check(false, std::string(t) + ": " + std::to_string(got->size()) +
            " values vs the golden's " + std::to_string(want.data.size()));
      localized = true;
      break;
    }
    const double r = npy::rel_l2(*got, want.data);
    std::printf("       %-8s [%d,%d,%d,%d]  rel-L2 %.3e\n", t, s[0], s[1],
                s[2], s[3], r);
    // f32 both sides, same weights: this is arithmetic-order noise only.
    if (r >= 2e-5) {
      check(false, std::string("the tap after ") + t + " matches");
      localized = true;
      break;
    }
  }
  if (!localized) {
    check(true, "every intermediate matches the reference");
  }

  check(pix.size() == out.data.size(), "the pixel count matches");
  if (pix.size() == out.data.size()) {
    const double r = npy::rel_l2(pix, out.data);
    std::printf("       decoded pixels  rel-L2 %.3e\n", r);
    check(r < 2e-5, "the decoded pixels match the reference");
  }

  // ---- the MULTI-FRAME case -----------------------------------------
  //
  // F'=1 leaves two things untested, because with a single input frame
  // they are indistinguishable:
  //
  //   * CAUSAL vs SYMMETRIC time padding -- causal pads 2 frames at the
  //     front, symmetric pads 1 at each end, and with one frame both
  //     produce [x, x, x];
  //   * the interior of the temporal convolution at all.
  //
  // F'=2 (-> 3 -> 5 -> 9 frames) separates them. It is ~20x the work, so
  // it runs only when asked -- but it runs, rather than being a comment
  // about what would be nice to check.
  if (env_("VPIPE_LTX25_VAE_FULL").empty()) {
    std::printf("  [note] set VPIPE_LTX25_VAE_FULL=1 for the multi-frame "
                "case (slow; it is what pins the TIME padding)\n");
  } else {
    npy::Array lat2 = g("vae_latent");
    npy::Array out2 = g("vae_out");
    if (!lat2.ok || !out2.ok) {
      std::printf("  [SKIP] no multi-frame vae golden\n");
    } else {
      dec->set_capture(false);
      std::vector<float> pix2;
      std::array<int, 4> sh2{};
      if (!dec->decode(lat2.data.data(), 2, 2, 2, &pix2, &sh2, &err)) {
        check(false, "multi-frame decode: " + err);
      } else {
        std::printf("       multi-frame -> [%d,%d,%d,%d]\n", sh2[0], sh2[1],
                    sh2[2], sh2[3]);
        check(sh2[1] == 9, "2 latent frames -> 9 output frames");
        check(pix2.size() == out2.data.size(), "the pixel count matches");
        if (pix2.size() == out2.data.size()) {
          const double r = npy::rel_l2(pix2, out2.data);
          std::printf("       multi-frame pixels  rel-L2 %.3e\n", r);
          check(r < 2e-5, "the multi-frame decode matches the reference");
        }
      }
    }
  }

  // ---- the METAL decoder --------------------------------------------
  //
  // Checked against the SAME goldens as the CPU reference, and against
  // the CPU reference itself. It runs bf16 (the checkpoint's own dtype)
  // where the golden is f32, so the bar is looser than the reference's
  // 1e-6 -- but it is set from the arithmetic, not from what passed:
  // bf16 has ~2^-8 = 3.9e-3 relative precision per rounding, and this is
  // ~40 convolutions deep.
  ltx25::MetalOps ops;
  if (!ops.init(&mc, &err)) {
    check(false, "metal ops: " + err);
  } else {
    auto gpu = ltx25::Ltx25VaeDecoder::load(vc, ws, ops, &err);
    if (!gpu) {
      check(false, "metal decoder load: " + err);
    } else {
      std::printf("       metal decoder holds %.1f MB\n",
                  (double)gpu->resident_bytes() / (1024.0 * 1024.0));
      std::vector<float> gp;
      std::array<int, 4> gs{};
      if (!gpu->decode(lat.data.data(), F, H, W, &gp, &gs, &err)) {
        check(false, "metal decode: " + err);
      } else {
        check(gs == shape, "the metal decode has the reference's shape");
        if (gp.size() == out.data.size()) {
          const double rg = npy::rel_l2(gp, out.data);
          const double rc = npy::rel_l2(gp, pix);
          std::printf("       metal vs f32 golden  rel-L2 %.3e\n", rg);
          std::printf("       metal vs CPU ref     rel-L2 %.3e\n", rc);
          check(rg < 2e-2, "the metal decode matches the reference");
          // Against the CPU reference the ONLY difference is bf16 vs
          // f32, so the two numbers must agree; a gap between them would
          // mean the two implementations disagree about semantics, not
          // about precision.
          check(std::abs(rg - rc) < 5e-3,
                "metal agrees with the CPU reference to the same degree");
        } else {
          check(false, "the metal pixel count matches");
        }

        // The MULTI-FRAME case on the GPU. The CPU reference gates this
        // behind an env var because it costs 3 minutes; the GPU does it
        // in well under a second, so it always runs -- and it is what
        // pins the TIME padding (a single frame cannot tell causal from
        // symmetric apart).
        npy::Array mlat = g("vae_latent");
        npy::Array mout = g("vae_out");
        if (mlat.ok && mout.ok) {
          std::vector<float> mp;
          std::array<int, 4> ms{};
          const auto t0 = std::chrono::steady_clock::now();
          if (!gpu->decode(mlat.data.data(), 2, 2, 2, &mp, &ms, &err)) {
            check(false, "multi-frame metal decode: " + err);
          } else {
            const double dt = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            std::printf("       multi-frame [%d,%d,%d,%d] in %.3f s\n", ms[0],
                        ms[1], ms[2], ms[3], dt);
            check(ms[1] == 9, "2 latent frames -> 9 output frames");
            if (mp.size() == mout.data.size()) {
              const double r = npy::rel_l2(mp, mout.data);
              std::printf("       multi-frame vs golden  rel-L2 %.3e\n", r);
              check(r < 2e-2, "the multi-frame metal decode matches");
            } else {
              check(false, "the multi-frame pixel count matches");
            }
          }
        }

        // A REALISTIC size, for the wall clock. 9 frames at 256x256 is
        // the smallest thing generate-video actually emits.
        if (!env_("VPIPE_LTX25_VAE_BENCH").empty()) {
          const int bf = 2, bh = 8, bw = 8;
          std::vector<float> bl((std::size_t)128 * bf * bh * bw, 0.1f);
          std::vector<float> bp;
          std::array<int, 4> bs{};
          const auto t0 = std::chrono::steady_clock::now();
          if (gpu->decode(bl.data(), bf, bh, bw, &bp, &bs, &err)) {
            const double dt = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            std::printf("       BENCH [%d,%d,%d,%d] in %.3f s\n", bs[0],
                        bs[1], bs[2], bs[3], dt);
          } else {
            check(false, "bench decode: " + err);
          }
        }

        // CHUNKING must not change the answer. Forcing a tiny im2col
        // ceiling splits every convolution into many chunks; a chunked
        // conv that indexed its output wrong would still produce a
        // plausible picture.
        gpu->set_max_im2col_elems(4096);
        std::vector<float> gp2;
        std::array<int, 4> gs2{};
        if (gpu->decode(lat.data.data(), F, H, W, &gp2, &gs2, &err)) {
          double d = 0.0;
          for (std::size_t i = 0; i < gp.size() && i < gp2.size(); ++i) {
            d = std::max(d, (double)std::fabs(gp[i] - gp2[i]));
          }
          std::printf("       chunked vs unchunked  max |delta| %.3e\n", d);
          check(d == 0.0, "chunking the convolution changes nothing");
        } else {
          check(false, "chunked decode: " + err);
        }
      }
    }
  }

  // ---- the ENCODER --------------------------------------------------
  //
  // Pixels to a whitened latent, which is what makes the image / video
  // reference paths producible at all. Its golden is 9 frames at 32x32
  // -- deliberately multi-frame, because every temporal thing the
  // encoder does differently from the decoder (causal padding, the
  // frame duplication before a time-halving block) is invisible on one
  // frame.
  {
    std::printf("  --- the encoder ---\n");
    npy::Array px = g("vaeenc_pixels");
    npy::Array want = g("vaeenc_latent");
    if (!px.ok || !want.ok) {
      std::printf("  [SKIP] no encoder goldens (run gen_goldens.py "
                  "vaeenc): %s%s\n", px.err.c_str(), want.err.c_str());
    } else {
      auto enc = ltx25::Ltx25VaeEncoderRef::load(vc, *ws, &mc, &err);
      if (!enc) {
        check(false, "encoder load: " + err);
      } else {
        check(true, "the encoder bound its weights");
        const int EF = (int)px.shape[1];
        const int EH = (int)px.shape[2], EW = (int)px.shape[3];
        enc->set_capture(true);
        std::vector<float> lat_out;
        std::array<int, 4> es{};
        const auto t0 = std::chrono::steady_clock::now();
        if (!enc->encode(px.data.data(), EF, EH, EW, &lat_out, &es, &err)) {
          check(false, "encode: " + err);
        } else {
          const double dt = std::chrono::duration<double>(
              std::chrono::steady_clock::now() - t0).count();
          std::printf("       [3,%d,%d,%d] -> [%d,%d,%d,%d] in %.2f s\n",
                      EF, EH, EW, es[0], es[1], es[2], es[3], dt);
          check(es[0] == vc.latent_channels, "128 latent channels");
          check(es[1] == 1 + (EF - 1) / vc.temporal_factor(),
                "1 + (F-1)/8 latent frames");
          check(es[2] == EH / vc.spatial_factor() &&
                es[3] == EW / vc.spatial_factor(), "32x spatial compression");

          // Intermediates in GRAPH ORDER: the first mismatch is the
          // block that broke, the rest are consequences.
          const char* etaps[] = {"conv_in", "down1", "down3", "down5",
                                 "down7", "down8", "head"};
          bool bad = false;
          for (const char* t : etaps) {
            npy::Array w = g(std::string("vaeenc_") + t);
            const std::vector<float>* got = enc->tap(t);
            if (!w.ok || got == nullptr) { continue; }
            std::array<int, 4> s{};
            enc->tap_shape(t, &s);
            if (got->size() != w.data.size()) {
              check(false, std::string(t) + ": " +
                    std::to_string(got->size()) + " values vs the golden's " +
                    std::to_string(w.data.size()));
              bad = true;
              break;
            }
            const double r = npy::rel_l2(*got, w.data);
            std::printf("       %-8s [%d,%d,%d,%d]  rel-L2 %.3e\n", t, s[0],
                        s[1], s[2], s[3], r);
            if (r >= 2e-5) {
              check(false, std::string("the tap after ") + t + " matches");
              bad = true;
              break;
            }
          }
          if (!bad) { check(true, "every encoder intermediate matches"); }

          check(lat_out.size() == want.data.size(),
                "the latent element count matches");
          if (lat_out.size() == want.data.size()) {
            const double r = npy::rel_l2(lat_out, want.data);
            std::printf("       whitened latent  rel-L2 %.3e\n", r);
            check(r < 2e-5, "the encoded latent matches the reference");
          }

          // The frame-count rule is REFUSED, not cropped. A stage that
          // quietly dropped frames would hand the DiT a reference for a
          // clip it is not generating.
          std::vector<float> junk((std::size_t)3 * 8 * EH * EW, 0.0f);
          std::vector<float> ignored;
          std::array<int, 4> is{};
          check(!enc->encode(junk.data(), 8, EH, EW, &ignored, &is, &err),
                "8 frames is refused, not cropped: " + err);

          // ---- the METAL encoder ------------------------------------
          //
          // Against BOTH the f32 golden and the CPU reference. bf16 over
          // ~40 sequential convolutions, so the bar is the dtype's, and
          // agreeing with the CPU path to the same degree is what says
          // the difference is precision rather than a second reading of
          // the layout.
          ltx25::MetalOps mops;
          std::string oerr;
          if (!mops.init(&mc, &oerr)) {
            check(false, "MetalOps init: " + oerr);
          } else {
            auto genc = ltx25::Ltx25VaeEncoder::load(vc, ws, mops, &err);
            if (!genc) {
              check(false, "metal encoder load: " + err);
            } else {
              std::printf("       metal encoder holds %.1f MB\n",
                          (double)genc->resident_bytes() / 1e6);
              std::vector<float> glat;
              std::array<int, 4> gs{};
              const auto t1 = std::chrono::steady_clock::now();
              if (!genc->encode(px.data.data(), EF, EH, EW, &glat, &gs,
                                &err)) {
                check(false, "metal encode: " + err);
              } else {
                const double dt2 = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t1).count();
                std::printf("       metal [%d,%d,%d,%d] in %.3f s\n", gs[0],
                            gs[1], gs[2], gs[3], dt2);
                check(gs == es, "the metal encode has the reference's shape");
                if (glat.size() == want.data.size()) {
                  const double r1 = npy::rel_l2(glat, want.data);
                  const double r2 = npy::rel_l2(glat, lat_out);
                  std::printf("       metal vs f32 golden  rel-L2 %.3e\n", r1);
                  std::printf("       metal vs CPU ref     rel-L2 %.3e\n", r2);
                  check(r1 < 2e-2, "the metal encode matches the reference");
                  check(r2 < 2e-2, "metal agrees with the CPU reference");
                } else {
                  check(false, "the metal latent element count matches");
                }

                // CHUNKING must not change the answer. A tiny im2col
                // ceiling splits every convolution into many chunks; a
                // chunked conv that indexed its output wrong would
                // still produce a plausible latent.
                genc->set_max_im2col_elems(4096);
                std::vector<float> g2;
                std::array<int, 4> gs2{};
                if (genc->encode(px.data.data(), EF, EH, EW, &g2, &gs2,
                                 &err)) {
                  double d = 0.0;
                  for (std::size_t i = 0; i < glat.size() && i < g2.size();
                       ++i) {
                    d = std::max(d, (double)std::fabs(glat[i] - g2[i]));
                  }
                  std::printf("       chunked vs unchunked  max |delta| "
                              "%.3e\n", d);
                  check(d == 0.0, "chunking changes nothing");
                } else {
                  check(false, "chunked encode: " + err);
                }
                genc->set_max_im2col_elems(64ull * 1024 * 1024);

                // A SECOND SIZE, 2x2 latent cells rather than 1x1.
                // With one cell per spatial axis every layout is the
                // same layout, so the tiny case cannot tell a
                // width-fastest regroup from a height-fastest one --
                // and a bf16 relative error has nothing to average
                // over. Metal only: the CPU reference at this size is
                // minutes for the same semantics.
                // ---- WHY THERE IS NO LATENT AGGREGATOR ----------------
                //
                // A clip must be encoded AS A CLIP. Encoding its frames
                // one at a time and concatenating the latents is a
                // different tensor, and not slightly:
                //
                //   * the temporal factor is 8, so 9 frames encode to 2
                //     latent frames -- concatenating 9 single-frame
                //     encodes gives 9, which RoPE then places across
                //     4.5x the seconds the clip actually spans;
                //   * the convolutions mix neighbouring frames, so
                //     latent frame 1 depends on pixel frames a
                //     single-frame encode never saw.
                //
                // What DOES hold is causality, and it is worth pinning
                // because it is the reason single-image anchoring works
                // at all: latent frame 0 of a clip sees only pixel frame
                // 0 (the causal padding replicates it, and the
                // time-halving blocks duplicate it), so it must equal
                // the single-frame encode EXACTLY. If that ever stops
                // being true, the causal padding has become symmetric
                // and every anchored generation is quietly wrong.
                {
                  const int EH2 = (int)px.shape[2], EW2 = (int)px.shape[3];
                  std::vector<float> one((std::size_t)3 * 1 * EH2 * EW2);
                  const std::size_t plane = (std::size_t)EH2 * EW2;
                  for (int c = 0; c < 3; ++c) {
                    std::memcpy(one.data() + (std::size_t)c * plane,
                                px.data.data() +
                                    (std::size_t)c * EF * plane,
                                plane * sizeof(float));
                  }
                  std::vector<float> l1;
                  std::array<int, 4> s1{};
                  if (!genc->encode(one.data(), 1, EH2, EW2, &l1, &s1,
                                    &err)) {
                    check(false, "single-frame encode: " + err);
                  } else {
                    check(s1[1] == 1 && gs[1] == 1 + (EF - 1) / 8,
                          "1 frame -> 1 latent frame, " +
                          std::to_string(EF) + " -> " +
                          std::to_string(gs[1]) + ": concatenating "
                          "single-frame latents would be the wrong "
                          "temporal scale");
                    const std::size_t cells =
                        (std::size_t)gs[0] * gs[2] * gs[3];
                    double worst_f0 = 0.0;
                    for (int c = 0; c < gs[0]; ++c) {
                      for (int k = 0; k < gs[2] * gs[3]; ++k) {
                        const std::size_t clip_i =
                            ((std::size_t)c * gs[1] + 0) *
                                (std::size_t)(gs[2] * gs[3]) + k;
                        const std::size_t one_i =
                            (std::size_t)c * (gs[2] * gs[3]) + k;
                        worst_f0 = std::max(
                            worst_f0,
                            (double)std::fabs(glat[clip_i] - l1[one_i]));
                      }
                    }
                    (void)cells;
                    std::printf("       clip frame 0 vs a 1-frame encode  "
                                "max |delta| %.3e\n", worst_f0);
                    check(worst_f0 == 0.0,
                          "the clip's FIRST latent frame is bit-identical "
                          "to encoding frame 0 alone -- the encoder is "
                          "causal");
                  }
                }

                npy::Array bp = g("vaeenc_pixels_big");
                npy::Array bw = g("vaeenc_latent_big");
                if (bp.ok && bw.ok) {
                  std::vector<float> bl;
                  std::array<int, 4> bs{};
                  const int BF = (int)bp.shape[1];
                  const int BH = (int)bp.shape[2], BW = (int)bp.shape[3];
                  const auto t2 = std::chrono::steady_clock::now();
                  if (!genc->encode(bp.data.data(), BF, BH, BW, &bl, &bs,
                                    &err)) {
                    check(false, "big metal encode: " + err);
                  } else {
                    const double dt3 = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t2).count();
                    std::printf("       big [3,%d,%d,%d] -> [%d,%d,%d,%d] "
                                "in %.3f s\n", BF, BH, BW, bs[0], bs[1],
                                bs[2], bs[3], dt3);
                    check(bl.size() == bw.data.size(),
                          "the big latent element count matches");
                    if (bl.size() == bw.data.size()) {
                      const double r = npy::rel_l2(bl, bw.data);
                      std::printf("       big vs f32 golden  rel-L2 %.3e\n",
                                  r);
                      check(r < 2e-2, "the 2x2 metal encode matches");
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  std::printf("%s\n", g_fail == 0 ? "ALL PASSED" : "FAILURES");
  return g_fail == 0 ? 0 : 1;
}
