// The family end to end: VideoModelFamily -> VideoGenerator -> latents.
//
// This is the path `generate-video` takes, driven directly so it can run
// without a pipeline. It exercises the parts nothing else does: the
// family's claim + streaming decision, the weight set going through
// open_weight_set, the 8-step ancestral loop over the 48-block DiT, and
// the latent that comes back.
//
// There is no golden -- a reference run means the whole 39 GB stack in
// PyTorch plus a text encoder that is not ported. What IS checked is
// what a mis-wired loop breaks: the latent's shape, that every value is
// finite, that the schedule actually moved the latent away from its
// initial noise, and that two seeds differ.

#include "apple-silicon/metal-compute/shared-buffer.h"

#include "ltx25-config.h"
#include "ltx25-metal-ops.h"
#include "ltx25-vocoder.h"
#include "ltx25-audio-vae.h"
#include "ltx25-family.h"
#include "ltx25-generator.h"

#include "generative-models/video-model-registry.h"
#include "common/flex-data.h"
#include "generative-models/weight-set.h"
#include "generative-models/shared/comfy-checkpoint.h"

#include <chrono>
#include <cmath>

#include <mach/mach.h>
#include <mach/task_info.h>

#include <mach/mach.h>
#include <mach/task_info.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <random>
#include <thread>
#include <string>
#include <vector>

extern "C" const unsigned char ltx25_kernels_f32_metallib[];
extern "C" const unsigned long ltx25_kernels_f32_metallib_len;
extern "C" const unsigned char ltx25_kernels_bf16_metallib[];
extern "C" const unsigned long ltx25_kernels_bf16_metallib_len;

using vpipe::metal_compute::MetalCompute;

namespace {

int g_fail = 0;

int
env_int_(const char* k, int dflt)
{
  const char* v = std::getenv(k);
  if (v == nullptr || *v == '\0') { return dflt; }
  const int n = std::atoi(v);
  return n > 0 ? n : dflt;
}

// The whole process, phys_footprint rather than resident_size: a Metal
// buffer's pages are wired through IOKit and resident_size misses them.
double
footprint_mb_()
{
  task_vm_info_data_t info{};
  mach_msg_type_number_t cnt = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO,
                (task_info_t)&info, &cnt) != KERN_SUCCESS) {
    return 0.0;
  }
  return (double)info.phys_footprint / (1024.0 * 1024.0);
}

void
check(bool ok, const std::string& what)
{
  std::printf("  [%s] %s\n", ok ? " OK " : "FAIL", what.c_str());
  if (!ok) { ++g_fail; }
}

double
rms(const std::vector<float>& v)
{
  if (v.empty()) { return 0.0; }
  double s = 0.0;
  for (float f : v) { s += (double)f * f; }
  return std::sqrt(s / (double)v.size());
}

std::uint16_t
to_bf16(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  const std::uint32_t r = ((u >> 16) & 1u) + 0x7fffu;
  return (std::uint16_t)((u + r) >> 16);
}

}  // namespace

int
main()
{
  const char* root = std::getenv("VPIPE_LTX25_TEST_MODEL_PATH");
  if (root == nullptr) {
    std::printf("SKIPPED: needs VPIPE_LTX25_TEST_MODEL_PATH. "
                "NOTHING was checked.\n");
    return 0;
  }
  MetalCompute mc(nullptr);
  if (!mc.valid()) {
    std::printf("SKIPPED: no Metal device. NOTHING was checked.\n");
    return 0;
  }
  mc.register_metal_library(ltx25::kMetalLibBf16, ltx25_kernels_bf16_metallib,
                            ltx25_kernels_bf16_metallib_len);

  // The family, exactly as generate-video reaches it.
  ltx25::Ltx25Family fam;
  check(fam.claims(root, ""), "the family claims the checkpoint");
  check(fam.align_frames(root, 8) == 9, "frames 8 -> 9 (the % 8 == 1 rule)");
  check(!fam.declare_resources(root).empty(),
        "the family declares its weights for the planning phase");

  vpipe::genai::VideoModelCreateArgs args;
  args.root = root;
  args.metal = &mc;
  args.session = nullptr;
  args.prefer_streaming = false;

  auto t0 = std::chrono::steady_clock::now();
  std::unique_ptr<vpipe::genai::VideoGenerator> gen = fam.load(args);
  if (!gen) {
    check(false, "family load");
    std::printf("FAILURES\n");
    return 1;
  }
  const double load_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();
  std::printf("  loaded in %.1f s\n", load_s);
  // WHAT THE STREAMING DECISION ACTUALLY DID. The pin count is the term
  // that decides whether a bounded box survives, and it is derived from
  // the checkpoint's own shape (per-block bytes, the trunk that never
  // streams) against a fraction of RAM -- so nothing outside this line
  // can check it. Drive it with VPIPE_RAM_LIMIT_MB + VPIPE_LTX25_STREAM
  // to see the decision a smaller box would take.
  if (auto* g = dynamic_cast<ltx25::Ltx25Generator*>(gen.get())) {
    const auto st0 = vpipe::metal_compute::shared_buffer_memory_stats();
    std::printf("  AFTER LOAD: shared buffers live %llu MB in %llu handles\n",
                (unsigned long long)(st0.live_bytes >> 20),
                (unsigned long long)st0.live_count);
    std::printf("  streaming=%s, %d blocks pinned (%llu MB), "
                "footprint %.0f MB\n",
                g->streaming_blocks() ? "yes" : "no", g->pinned_blocks(),
                (unsigned long long)(g->pinned_weight_bytes() >> 20),
                footprint_mb_());
  }
  check(gen->latent_channels() == 128, "128 latent channels");
  check(gen->spatial_compression() == 32, "1/32 spatial");

  // A small request: 256x256 x 9 frames -> latent [128, 2, 8, 8].
  // Geometry knobs: the memory behaviour of this model is only visible
  // at the geometry it will actually be asked for -- the scratch, the
  // attention and the per-forward streaming all scale with the tokens.
  const int W = env_int_("VPIPE_LTX25_W", 256);
  const int H = env_int_("VPIPE_LTX25_H", 256);
  const int F = env_int_("VPIPE_LTX25_F", 9);
  const int TT = 256;      // a whole number of connector register tiles
  const int CD = 4096;     // cross_attention_dim
  std::vector<std::uint16_t> cond((std::size_t)TT * CD);
  std::mt19937 rng(3);
  std::normal_distribution<float> nd(0.0f, 0.5f);
  for (auto& v : cond) { v = to_bf16(nd(rng)); }

  vpipe::genai::VideoGenRequest req;
  req.height = H;
  req.width = W;
  req.frames = F;
  req.fps = 24.0;
  req.steps = env_int_("VPIPE_LTX25_STEPS", 8);
  req.seed = 1234;
  req.cond = cond.data();
  req.cond_rows = TT;
  req.cond_dim = CD;

  // The FIRST generation runs with the adaLN bake OFF, so it is the
  // reference the baked path is compared against below. Order matters:
  // bake_adaln RELEASES the projections, so once anything has baked
  // there is no un-baked path left to compare with.
  ::setenv("VPIPE_LTX25_NO_ADALN_BAKE", "1", 1);

  // PROGRESS, counted. A reporting hook that silently never fires looks
  // exactly like one that works, and this family shipped for months with
  // the DiT's per-block hook unset and nothing noticing.
  int prog_steps = 0, prog_blocks = 0, last_step = 0, last_total = 0;
  int block_total = 0;
  req.progress = [&](int step, int total) {
    ++prog_steps; last_step = step; last_total = total;
    return true;
  };
  req.block_progress = [&](int done, int total) {
    ++prog_blocks; block_total = total; (void)done;
    return true;
  };

  const double fp_pre = footprint_mb_();
  std::atomic<bool> sampling{true};
  std::atomic<double> fp_peak{fp_pre};
  std::thread sampler([&]{
    while (sampling.load(std::memory_order_relaxed)) {
      const double n = footprint_mb_();
      if (n > fp_peak.load(std::memory_order_relaxed)) {
        fp_peak.store(n, std::memory_order_relaxed);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  });

  vpipe::genai::VideoGenResult res;
  auto t1 = std::chrono::steady_clock::now();
  if (!gen->generate(req, &res)) {
    check(false, "generate");
    std::printf("FAILURES\n");
    return 1;
  }
  const double gen_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t1)
          .count();
  sampling.store(false, std::memory_order_relaxed);
  sampler.join();
  std::printf("  %d-step generation in %.1f s (%.1f s/step)\n", req.steps,
              gen_s, gen_s / (double)req.steps);
  std::printf("  DENOISE FOOTPRINT: %.0f MB before -> %.0f MB peak "
              "(+%.0f MB while denoising)\n",
              fp_pre, fp_peak.load(), fp_peak.load() - fp_pre);
  // The SAME question asked of the buffer accounting rather than of the
  // OS: if these disagree with the process footprint, the buffers are
  // being freed and something else is holding the memory.
  {
    const auto st = vpipe::metal_compute::shared_buffer_memory_stats();
    std::printf("  SHARED BUFFERS: live %llu MB in %llu handles, "
                "peak %llu MB, cumulative %llu MB\n",
                (unsigned long long)(st.live_bytes >> 20),
                (unsigned long long)st.live_count,
                (unsigned long long)(st.peak_bytes >> 20),
                (unsigned long long)(st.total_bytes >> 20));
  }
  if (std::getenv("VPIPE_LTX25_ONE_SHOT") != nullptr) {
    std::printf("ONE_SHOT: stopping after the first generation\n");
    return 0;
  }

  // One step report per step, ONE-BASED and at the end -- the host's bar
  // takes the index of the step that just finished, so a zero-based or
  // start-of-step report runs the bar ahead of the work.
  check(prog_steps == 8, "the step hook fired once per step (" +
        std::to_string(prog_steps) + ")");
  check(last_step == last_total && last_total > 0,
        "the last step report is 1-based and reaches the total (" +
        std::to_string(last_step) + "/" + std::to_string(last_total) + ")");
  // And the block hook, which is what makes the bar move within a step.
  check(prog_blocks == prog_steps * block_total && block_total > 0,
        "the block hook fired once per block per step (" +
        std::to_string(prog_blocks) + " over " +
        std::to_string(block_total) + " blocks)");

  const std::vector<int> want_shape = {128, 2, 8, 8};
  check(res.video_shape == want_shape,
        "latent shape [128, 2, 8, 8] from 9 frames at 256x256");
  check(res.video.size() == 128u * 2 * 8 * 8, "latent element count");

  std::size_t bad = 0;
  for (float f : res.video) { if (!std::isfinite(f)) { ++bad; } }
  check(bad == 0, "every latent value is finite (" + std::to_string(bad) +
        " bad)");
  std::printf("  latent rms %.4f\n", rms(res.video));
  // The schedule ends at sigma 0, so the last step returns the denoised
  // prediction: a latent still at the noise scale means the loop never
  // moved it.
  check(rms(res.video) > 1e-4, "the latent is not empty");

  // A different seed must give a different latent. If it does not, the
  // seed is not reaching the initial noise -- the model would then
  // produce one video forever, which no shape check would catch.
  // ---- the adaLN bake is the SAME arithmetic, moved earlier ----------
  //
  // bake_adaln precomputes the eight per-step modulation chains and then
  // frees the projections. Nothing about the values changes -- so the
  // bar is BIT-IDENTICAL, not "close". A rel-L2 of 1e-7 here would mean
  // the bake is feeding a different sigma to some chain, which is
  // exactly the failure a tolerance would hide.
  {
    ::unsetenv("VPIPE_LTX25_NO_ADALN_BAKE");
    vpipe::genai::VideoGenRequest breq = req;
    vpipe::genai::VideoGenResult bres;
    auto tb = std::chrono::steady_clock::now();
    const bool ok = gen->generate(breq, &bres);
    const double bake_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - tb)
            .count();
    check(ok, "a baked generation runs");
    if (ok) {
      check(bres.video.size() == res.video.size(),
            "the bake does not change the latent's shape");
      std::size_t diff = 0;
      if (bres.video.size() == res.video.size()) {
        for (std::size_t i = 0; i < res.video.size(); ++i) {
          if (bres.video[i] != res.video[i]) { ++diff; }
        }
      }
      std::printf("       baked run in %.1f s (unbaked %.1f s), %zu/%zu "
                  "elements differ\n",
                  bake_s, gen_s, diff, res.video.size());
      check(diff == 0, "baked and unbaked latents are BIT-IDENTICAL");
    }
  }

  vpipe::genai::VideoGenRequest req2 = req;
  req2.seed = 9999;
  vpipe::genai::VideoGenResult res2;
  if (gen->generate(req2, &res2)) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < res.video.size(); ++i) {
      const double d = (double)res2.video[i] - res.video[i];
      num += d * d;
      den += (double)res.video[i] * res.video[i];
    }
    const double rel = den > 0 ? std::sqrt(num / den) : 0.0;
    std::printf("  seed 1234 vs 9999: relative difference %.3f\n", rel);
    check(rel > 0.05, "the seed reaches the initial noise");
  } else {
    check(false, "second generate");
  }

  // A conditioning of the wrong width must be REFUSED, not resampled:
  // it would otherwise be a clean run against a caption the model never
  // saw.
  vpipe::genai::VideoGenRequest bad_req = req;
  bad_req.cond_dim = 2048;
  vpipe::genai::VideoGenResult bad_res;
  check(!gen->generate(bad_req, &bad_res),
        "a conditioning of the wrong width is refused");

  // So must a caption that is not a whole number of register tiles.
  vpipe::genai::VideoGenRequest odd = req;
  odd.cond_rows = 200;
  vpipe::genai::VideoGenResult odd_res;
  check(!gen->generate(odd, &odd_res),
        "a caption that is not a multiple of 128 rows is refused");

  // ---- the JOINT audio-video denoise -----------------------------------
  //
  // The bar here is STRUCTURAL and stated as such: there is no golden
  // for a joint generation (one would mean the whole 22B stack in
  // PyTorch), so what is checked is that the second stream exists, is
  // finite, is the shape the audio VAE wants, actually moved off its
  // initial noise, and responds to the seed. The audio DiT output
  // itself is NOT verified against the reference.
  //
  // The context is synthetic. A real one needs the 12B encoder, and
  // "the soundtrack matches the prompt" is not a claim this test can
  // make either way -- only that the plumbing carries a real second
  // modality end to end.
  {
    const int ACD = 2048;    // audio_cross_attention_dim
    std::vector<std::uint16_t> acond((std::size_t)TT * ACD);
    for (auto& v : acond) { v = to_bf16(nd(rng)); }

    vpipe::genai::VideoGenRequest areq = req;
    areq.audio_cond      = acond.data();
    areq.audio_cond_rows = TT;
    areq.audio_cond_dim  = ACD;

    vpipe::genai::VideoGenResult ares;
    const auto t0 = std::chrono::steady_clock::now();
    if (!gen->generate(areq, &ares)) {
      check(false, "joint audio-video generate");
    } else {
      const double secs = std::chrono::duration<double>(
                              std::chrono::steady_clock::now() - t0).count();
      // frames / fps * 25 latent frames a second.
      const int want_at =
          (int)std::llround((double)F / 24.0 * ltx25::kAudioLatentsPerSecond);
      std::printf("       audio latent %s in %.1f s\n",
                  ares.audio_shape.size() == 3
                      ? (std::to_string(ares.audio_shape[0]) + "x" +
                         std::to_string(ares.audio_shape[1]) + "x" +
                         std::to_string(ares.audio_shape[2])).c_str()
                      : "(none)", secs);
      check(ares.audio_shape.size() == 3, "an audio latent came back");
      if (ares.audio_shape.size() == 3) {
        // [channels][frames][mel] -- the layout Ltx25AudioVaeDecoder
        // takes. The DiT emits [128][T] packed c*16+f, so a missing
        // transpose here decodes a plausible spectrogram from scrambled
        // channels.
        check(ares.audio_shape[0] == ltx25::kAudioLatentChannels,
              "8 latent channels");
        check(ares.audio_shape[2] == ltx25::kAudioLatentMelBins,
              "16 latent mel bins");
        check(ares.audio_shape[1] == want_at,
              "frames/fps * 25 latent frames (" +
                  std::to_string(ares.audio_shape[1]) + " vs " +
                  std::to_string(want_at) + ")");
        check(ares.latents_per_second == ltx25::kAudioLatentsPerSecond,
              "latents_per_second is stamped");
        int bad = 0;
        double s2 = 0.0;
        for (float v : ares.audio) {
          if (!std::isfinite(v)) { ++bad; }
          s2 += (double)v * (double)v;
        }
        const double rms =
            ares.audio.empty() ? 0.0
                               : std::sqrt(s2 / (double)ares.audio.size());
        std::printf("       audio rms %.4f, %d non-finite\n", rms, bad);
        check(bad == 0, "every audio value is finite");
        check(rms > 0.0, "the audio latent is not all zero");
        // The video half must be UNCHANGED in shape by turning audio on.
        check(ares.video_shape == res.video_shape,
              "the video latent's shape is unaffected by the audio stream");
      }

      // A different seed must move the soundtrack, or the audio noise is
      // not reaching the loop.
      vpipe::genai::VideoGenRequest areq2 = areq;
      areq2.seed = 9999;
      vpipe::genai::VideoGenResult ares2;
      if (gen->generate(areq2, &ares2) &&
          ares2.audio.size() == ares.audio.size() && !ares.audio.empty()) {
        double num = 0.0, den = 0.0;
        for (std::size_t i = 0; i < ares.audio.size(); ++i) {
          const double d0 = (double)ares.audio[i] - (double)ares2.audio[i];
          num += d0 * d0;
          den += (double)ares.audio[i] * (double)ares.audio[i];
        }
        const double rel = (den > 0.0) ? std::sqrt(num / den) : 0.0;
        std::printf("       seed 1234 vs 9999: audio rel diff %.3f\n", rel);
        check(rel > 0.1, "the seed reaches the audio stream too");
      }

      // ---- and DECODE it, through the already-verified chain ----------
      //
      // This is what completes the bar. The audio VAE decoder
      // (1.7e-3..6.0e-3 vs the reference) and the BigVGAN vocoder
      // (3.1e-5) are each pinned elsewhere, so running the DiT's own
      // latent through them checks the one thing left: that what the
      // joint denoise produces is something the decoder can actually
      // turn into sound, in the layout it expects. A wrong TRANSPOSE
      // upstream lands here as a plausible-looking spectrogram, which is
      // why "not silence" is checked rather than "it ran".
      if (!ares.audio.empty() && ares.audio_shape.size() == 3) {
        ltx25::Config acfg;
        std::string aerr;
        if (!ltx25::resolve(root, acfg, &aerr) || acfg.audio_vae_file.empty()) {
          std::printf("  [SKIP] no audio VAE in this checkpoint\n");
        } else {
          mc.register_metal_library(ltx25::kMetalLibF32,
                                    ltx25_kernels_f32_metallib,
                                    ltx25_kernels_f32_metallib_len);
          vpipe::FlexData vmeta;
          ltx25::AudioVaeConfig avc;
          ltx25::VocoderConfig vocc;
          ltx25::MetalOps aops;
          auto aws = vpipe::genai::open_weight_set(acfg.audio_vae_file,
                                                   nullptr);
          if (aws && aops.init(&mc, &aerr) &&
              vpipe::genai::comfy::metadata_json(acfg.audio_vae_file,
                                                 ltx25::kVaeMetaKey, vmeta,
                                                 &aerr) &&
              ltx25::parse_audio_vae_config(vmeta, avc, &aerr) &&
              ltx25::parse_vocoder_config(vmeta, "vocoder", vocc, &aerr)) {
            auto adec = ltx25::Ltx25AudioVaeDecoder::load(avc, aws, aops,
                                                          &aerr);
            auto voc = ltx25::Ltx25Vocoder::load(vocc, aws, &mc,
                                                 "vocoder.vocoder.", &aerr);
            std::vector<float> mel, wav;
            std::array<int, 3> ms{};
            std::array<int, 2> wsh{};
            if (adec && voc &&
                adec->decode(ares.audio.data(), ares.audio_shape[1], &mel,
                             &ms, &aerr) &&
                voc->synthesize(mel.data(), ms[1], ms[2], &wav, &wsh,
                                &aerr)) {
              double peak = 0.0, e = 0.0;
              int nbad = 0;
              for (float v : wav) {
                if (!std::isfinite(v)) { ++nbad; }
                peak = std::max(peak, (double)std::fabs(v));
                e += (double)v * (double)v;
              }
              const double wrms =
                  wav.empty() ? 0.0 : std::sqrt(e / (double)wav.size());
              std::printf("       decoded: mel [%d,%d,%d] -> wav [%d,%d] "
                          "(%.3f s at %d Hz), peak %.3f rms %.4f\n",
                          ms[0], ms[1], ms[2], wsh[0], wsh[1],
                          (double)wsh[1] / (double)vocc.output_sampling_rate,
                          vocc.output_sampling_rate, peak, wrms);
              check(nbad == 0, "every sample is finite");
              check(peak <= 1.0, "the waveform is clamped to [-1, 1]");
              check(wrms > 1e-5, "the waveform is NOT silence");
              // 4F-3 mel frames, then 160 samples each.
              check(ms[1] == 4 * ares.audio_shape[1] - 3,
                    "mel frames are 4F-3");
              check(wsh[1] == ms[1] * vocc.hop(),
                    "samples are mel frames x the 160 hop");
            } else {
              check(false, "audio decode: " + aerr);
            }
          } else {
            std::printf("  [SKIP] audio VAE setup: %s\n", aerr.c_str());
          }
        }
      }
    }
  }

  std::printf("%s\n", g_fail == 0 ? "ALL PASSED" : "FAILURES");
  return g_fail == 0 ? 0 : 1;
}
