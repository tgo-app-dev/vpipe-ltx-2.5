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

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

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

  const int Z = 128, T = 2, LH = 2, LW = 2;
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
  const bool ok = dec->decode(req, [&](const genai::VaeFrameChunk& c) {
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
  check(ok, ok ? "decode succeeded" : "decode: " + err);
  std::printf("       latent [%d,%d,%d,%d] -> [%d,%d,%d,%d]\n", Z, T, LH, LW,
              chan, (int)seen.size(), hh, ww);
  check((int)seen.size() == 9, "the sink saw 9 frames");
  check(total == 9, "frames_total agrees with what arrived");
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

  std::printf("%s\n", g_fail == 0 ? "ALL PASSED" : "FAILURES");
  return g_fail == 0 ? 0 : 1;
}
