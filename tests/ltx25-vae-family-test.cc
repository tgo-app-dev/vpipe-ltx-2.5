// The LTX-2.5 VAE family: the plugin half of the register_vae_family seam.
//
// Driven from a SYNTHETIC latent, so it costs one VAE load and a few
// seconds rather than the 42 GB DiT the real producer needs. The decoder
// underneath is verified against the reference elsewhere
// (ltx25-vae-ref-test, 3.8e-3); what is checked here is the SEAM:
//
//   * the family claims an LTX checkpoint and REFUSES a non-LTX one --
//     it is asked before the host's built-in `_class_name` chain, so a
//     loose claim would shadow a working built-in path;
//   * it declares the VAE file (not the root, which also holds the DiT);
//   * `idle_peers` names the Comfy dirs, without which the stage sizes
//     the box at zero and keeps the VAE resident beside a 39 GB DiT;
//   * the adapter's chunk is the contract the host quantises from:
//     f32 channel-first [C][F][H][W] in [-1,1], frames_total right, and
//     a false sink aborts;
//   * the frame count is the causal rule 8(T-1)+1.
//
// It stops at the family boundary ON PURPOSE. `Session`, `Pipeline` and
// `vae-decode-stage.h` are not in the plugin SDK -- and installing them
// to reach one test would grow the SDK for something no plugin needs at
// run time. The HOST stage's own plugin branch is covered instead by the
// full pipeline run (text-prompt -> conditioner -> generate-video ->
// vae-decode -> save-image), which is the only place it can be exercised
// against real weights anyway.

#include "ltx25-config.h"
#include "ltx25-vae-family.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/flex-data.h"
#include "generative-models/vae-model-registry.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <mach/mach.h>
#include <mach/task_info.h>

extern "C" const unsigned char ltx25_kernels_bf16_metallib[];
extern "C" const unsigned long ltx25_kernels_bf16_metallib_len;

using namespace vpipe;

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

int
env_int_(const char* k, int dflt)
{
  const std::string v = env_(k);
  if (v.empty()) { return dflt; }
  const int n = std::atoi(v.c_str());
  return n > 0 ? n : dflt;
}

// What the WHOLE PROCESS is holding, the way the tree measures memory --
// phys_footprint rather than resident_size, because a Metal buffer's
// pages are wired through IOKit and resident_size does not see all of
// them.
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

}  // namespace

int
main()
{
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const std::string root = env_("VPIPE_LTX25_TEST_MODEL_PATH");
  if (root.empty()) {
    std::printf("SKIPPED: needs VPIPE_LTX25_TEST_MODEL_PATH. NOTHING was "
                "checked.\n");
    return 0;
  }
  metal_compute::MetalCompute mc_obj(nullptr);
  if (!mc_obj.valid()) {
    std::printf("SKIPPED: no Metal device. NOTHING was checked.\n");
    return 0;
  }
  metal_compute::MetalCompute* mc = &mc_obj;
  // The plugin's kernels, which the decoder resolves by name. In a real
  // run the plugin's entry point registers these; here the test stands
  // in for it.
  mc->register_metal_library(ltx25::kMetalLibBf16,
                             ltx25_kernels_bf16_metallib,
                             ltx25_kernels_bf16_metallib_len);

  // ---- the claim -------------------------------------------------------
  ltx25::Ltx25VaeFamily fam;
  check(fam.tag() == ltx25::kFamily, "the family tags itself 'ltx-2.5'");
  check(fam.claims(root, root, ""), "it claims the LTX-2.5 checkpoint");
  // SURE, not merely plausible: this probe runs BEFORE the host's own
  // _class_name chain, so a loose claim would shadow a built-in.
  check(!fam.claims("/definitely/not/a/checkpoint", "", ""),
        "it refuses a directory that is not a checkpoint");
  check(!fam.declare_resources(root, root).empty(),
        "it declares the VAE file for the planning phase");
  // The COMFY spelling; the stage's built-in guess sums to zero here.
  const auto peers = fam.idle_peers(root);
  bool named_dit = false;
  for (const auto& p : peers) {
    if (p.find("diffusion_models") != std::string::npos) { named_dit = true; }
  }
  check(named_dit, "idle_peers names the DiT dir, so the box is sized right");

  // Register it, exactly as the plugin's entry point does.
  check(genai::VaeModelRegistry::get().add(
            std::make_unique<ltx25::Ltx25VaeFamily>()),
        "the registry accepts it");

  // ---- load and decode, directly ---------------------------------------
  genai::VaeModelCreateArgs args;
  args.root    = root;
  args.vae_dir = root;             // LTX has no vae/config.json
  args.metal   = mc;
  args.session = nullptr;          // open_weight_set falls back to a private set
  std::unique_ptr<genai::VaeDecoder> dec = fam.load_decoder(args);
  if (!dec) {
    check(false, "load_decoder built a decoder");
    std::printf("FAILURES\n");
    return 1;
  }
  check(dec->latent_channels() == 128, "128 latent channels");
  check(dec->spatial_compression() == 32, "32x spatial");
  check(dec->temporal_compression() == 8, "8x temporal");
  // The causal rule, and the reason a 2-frame latent is enough to pin
  // it: a decoder that dropped or duplicated the first frame lands on
  // 16 or 17, not 9.
  check(dec->decoded_frames(2) == 9, "8(T-1)+1 = 9 frames from T=2");
  std::printf("       decoder %.1f MB\n",
              (double)dec->resident_bytes() / (1024.0 * 1024.0));

  // THE GEOMETRY IS A KNOB, and its absence is why a 100x scaling in this
  // decoder's transient memory went unseen. The default is the smallest
  // latent that still pins the causal rule; a bigger one costs only time
  // here, and the footprint line below is what makes the cost of a real
  // clip's geometry visible before a 16 GB box finds it.
  const int Z = 128;
  const int T  = env_int_("VPIPE_LTX25_VAE_T",  2);
  const int LH = env_int_("VPIPE_LTX25_VAE_LH", 2);
  const int LW = env_int_("VPIPE_LTX25_VAE_LW", 2);
  const int want_frames = 8 * (T - 1) + 1;
  std::vector<float> lat((std::size_t)Z * T * LH * LW);
  for (std::size_t i = 0; i < lat.size(); ++i) {
    lat[i] = 0.7f * std::sin(0.031f * (float)i);
  }
  genai::VaeDecodeRequest req;
  req.latent = lat.data();
  req.shape  = {Z, T, LH, LW};
  req.fps    = 24.0;

  std::vector<int> seen;
  int total = -1, chan = 0, hh = 0, ww = 0;
  bool in_range = true, uniform = true;
  float first = 0.0f;
  std::string err;
  // The output picture, f32 -- the unit the transient budget is expressed
  // in below, and the only figure here that does not depend on how the
  // decoder is written.
  const double out_mb = (double)((std::size_t)3 * (std::size_t)want_frames *
                                 (std::size_t)(32 * LH) *
                                 (std::size_t)(32 * LW) * 4) /
                        (1024.0 * 1024.0);
  const double fp_before = footprint_mb_();

  // SAMPLED, not read at the end. The peak is inside the decode -- at
  // the widest block, with the picture allocated on top of pages the
  // kernel has not taken back yet -- and by the time the sink runs the
  // stack has already unwound past it. Reading the footprint at the sink
  // reports a decode that has already released most of what it held.
  std::atomic<bool> sampling{true};
  std::atomic<double> fp_peak{fp_before};
  std::thread sampler([&] {
    while (sampling.load(std::memory_order_relaxed)) {
      const double now = footprint_mb_();
      if (now > fp_peak.load(std::memory_order_relaxed)) {
        fp_peak.store(now, std::memory_order_relaxed);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  });

  double fp_in_sink = 0.0;
  const bool ok = dec->decode(req, [&](const genai::VaeFrameChunk& c) {
    fp_in_sink = footprint_mb_();
    total = c.frames_total;
    chan = c.channels; hh = c.height; ww = c.width;
    for (int k = 0; k < c.n; ++k) { seen.push_back(c.frame0 + k); }
    const std::size_t n = (std::size_t)c.channels * c.n * c.height * c.width;
    for (std::size_t i = 0; i < n; ++i) {
      const float v = c.rgb[i];
      if (!(v >= -1.5f && v <= 1.5f)) { in_range = false; }
      if (i == 0) { first = v; }
      else if (v != first) { uniform = false; }
    }
    return true;
  }, &err);
  sampling.store(false, std::memory_order_relaxed);
  sampler.join();

  check(ok, ok ? "decode succeeded" : "decode: " + err);
  std::printf("       latent [%d,%d,%d,%d] -> [%d,%d,%d,%d]\n", Z, T, LH, LW,
              chan, (int)seen.size(), hh, ww);
  const double measured = fp_peak.load(std::memory_order_relaxed) - fp_before;
  std::printf("       footprint %.0f MB before, %.0f MB peak during, "
              "%.0f MB at the sink -> %.0f MB of transients\n",
              fp_before, fp_peak.load(std::memory_order_relaxed), fp_in_sink,
              measured);
  std::printf("       that is %.1fx the %.0f MB output picture\n",
              out_mb > 0.0 ? measured / out_mb : 0.0, out_mb);
  // THE REGRESSION GUARD, in units of the output rather than in bytes, so
  // it holds at whatever geometry the knob above selects.
  //
  // This decoder used to encode its whole graph into one command buffer
  // and commit once at the end. An un-committed Metal command buffer
  // retains every resource it references, so every intermediate of all
  // nine levels stayed allocated: 20.3 GB of transients at 121 frames of
  // 960x544, which is ~27x the picture. Committing per block and reusing
  // three slots puts it at ~6x.
  //
  // 12x is the line: comfortably above what the per-block path measures
  // at every geometry tested, and less than half of what the accumulating
  // one did. A failure here means the commits or the slots have been
  // undone, which no correctness test would notice -- the pixels are
  // identical either way.
  check(measured <= 12.0 * out_mb + 512.0,
        "the decode's transients stay within 12x the output picture");
  check((int)seen.size() == want_frames, "the sink saw 8(T-1)+1 frames");
  check(total == want_frames, "frames_total agrees with what arrived");
  check(chan == 3, "3 channels");
  check(hh == 32 * LH && ww == 32 * LW, "32x spatial in the chunk");
  bool ordered = true;
  for (std::size_t i = 0; i < seen.size(); ++i) {
    if (seen[i] != (int)i) { ordered = false; }
  }
  check(ordered, "frames arrived in order, each exactly once");
  // [-1,1] IS the contract -- the host quantises (x+1)/2*255 straight
  // from this, so a decoder in another space would come out clipped.
  check(in_range, "the pixels are in [-1, 1]");
  check(!uniform, "the picture is not a flat colour");

  // A false sink aborts, which is how a Stop mid-decode gets out.
  int calls = 0;
  const bool aborted = dec->decode(req, [&](const genai::VaeFrameChunk&) {
    ++calls;
    return false;
  }, &err);
  check(!aborted && calls == 1, "a false sink aborts the decode");

  // `req.progress` reports, and reports something USABLE: monotonic,
  // bounded, and ending at the total. A bar fed counts that go backwards
  // or stop short is worse than no bar, so those are what is checked
  // rather than the mere fact of a call.
  std::vector<std::pair<int, int>> prog;
  genai::VaeDecodeRequest preq = req;
  preq.progress = [&prog](int done, int total) {
    prog.emplace_back(done, total);
    return true;
  };
  const bool pok = dec->decode(preq, [&](const genai::VaeFrameChunk&) {
    return true;
  }, &err);
  check(pok, "the decode with progress attached still succeeds");
  check(prog.size() >= 2, "progress was reported more than once (" +
        std::to_string(prog.size()) + " calls)");
  bool monotonic = !prog.empty();
  for (std::size_t i = 0; i < prog.size(); ++i) {
    if (prog[i].second <= 0 || prog[i].first < 0 ||
        prog[i].first > prog[i].second) { monotonic = false; }
    if (i > 0 && prog[i].first < prog[i - 1].first) { monotonic = false; }
    if (i > 0 && prog[i].second != prog[0].second) { monotonic = false; }
  }
  check(monotonic, "the counts are monotonic, in range, and one total");
  check(!prog.empty() && prog.back().first == prog.back().second,
        "the last report is complete");

  // And it is the cancel path the stage relies on: a false return stops
  // the decode instead of running it to the end.
  int pcalls = 0;
  genai::VaeDecodeRequest creq = req;
  creq.progress = [&pcalls](int, int) { ++pcalls; return false; };
  int csink = 0;
  const bool cok = dec->decode(creq, [&](const genai::VaeFrameChunk&) {
    ++csink;
    return true;
  }, &err);
  check(!cok && pcalls == 1 && csink == 0,
        "a false progress cancels the decode before any frame is emitted");

  std::printf("%s\n", g_fail == 0 ? "ALL PASSED" : "FAILURES");
  return g_fail == 0 ? 0 : 1;
}
