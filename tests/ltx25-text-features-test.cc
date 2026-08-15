// The text-encoder feature extractor, on the real projection weights.
//
// Checks the LAYOUT before the 188160-wide GEMM can hide it (the golden
// carries the normalised intermediate for exactly that), then both
// projections.

#include "ltx25-config.h"
#include "ltx25-metal-ops.h"
#include "ltx25-text-features.h"
#include "npy.h"

#include "generative-models/weight-set.h"

#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

extern "C" const unsigned char ltx25_kernels_bf16_metallib[];
extern "C" const unsigned long ltx25_kernels_bf16_metallib_len;

using vpipe::genai::WeightSet;
using vpipe::metal_compute::MetalCompute;

namespace {
int g_fail = 0;
std::string g_dir;
void check(bool ok, const std::string& w)
{
  std::printf("  [%s] %s\n", ok ? " OK " : "FAIL", w.c_str());
  if (!ok) { ++g_fail; }
}
npy::Array g(const std::string& n) { return npy::load(g_dir + "/" + n + ".npy"); }
constexpr int kT = 12, kValid = 8, kD = 3840, kL = 49;
}  // namespace

int
main()
{
  const char* root = std::getenv("VPIPE_LTX25_TEST_MODEL_PATH");
  const char* gdir = std::getenv("VPIPE_LTX25_GOLDENS");
  if (root == nullptr || gdir == nullptr) {
    std::printf("SKIPPED: needs VPIPE_LTX25_TEST_MODEL_PATH and "
                "VPIPE_LTX25_GOLDENS. NOTHING was checked.\n");
    return 0;
  }
  g_dir = gdir;
  MetalCompute mc(nullptr);
  if (!mc.valid()) {
    std::printf("SKIPPED: no Metal device. NOTHING was checked.\n");
    return 0;
  }
  mc.register_metal_library(ltx25::kMetalLibBf16, ltx25_kernels_bf16_metallib,
                            ltx25_kernels_bf16_metallib_len);

  ltx25::Config cfg;
  std::string err;
  if (!ltx25::resolve(root, cfg, &err, {})) {
    std::printf("SKIPPED: %s\n", err.c_str());
    return 0;
  }
  if (cfg.enc_file.empty()) {
    std::printf("SKIPPED: no text encoder in the checkpoint.\n");
    return 0;
  }
  std::shared_ptr<WeightSet> ws = WeightSet::open(cfg.enc_file, nullptr);
  if (!ws) { std::printf("SKIPPED: cannot open the text encoder\n"); return 0; }

  ltx25::MetalOps ops;
  if (!ops.init(&mc, &err)) {
    std::printf("FAILED ops: %s\n", err.c_str());
    return 1;
  }
  auto tf = ltx25::Ltx25TextFeatures::load(cfg.dit, *ws, ops, &err);
  if (!tf) { check(false, "load: " + err); std::printf("FAILURES\n"); return 1; }
  check(true, "both aggregate projections bound");
  // Derived from the projection's width, not assumed.
  check(tf->layers() == kL, "49 layers (48 + the embedding output), derived "
        "from the projection width");
  check(tf->flat_dim() == kD * kL, "flat dim 188160");

  npy::Array hidden = g("te_hidden");     // [1,T,D,L]
  npy::Array normed = g("te_normed");     // [1,T,D*L]
  npy::Array video  = g("te_video");      // [1,T,4096]
  npy::Array audio  = g("te_audio");      // [1,T,2048]
  if (!hidden.ok || !normed.ok || !video.ok || !audio.ok) {
    check(false, "goldens: " + hidden.err + normed.err + video.err + audio.err);
    std::printf("FAILURES\n");
    return 1;
  }

  if (!tf->reserve(kT, &err)) {
    check(false, "reserve: " + err);
    std::printf("FAILURES\n");
    return 1;
  }
  auto vout = ops.alloc((std::size_t)kT * cfg.dit.cross_attention_dim);
  auto aout = ops.alloc((std::size_t)kT * cfg.dit.audio_cross_attention_dim);
  if (!tf->compute(hidden.data.data(), kT, 0, kValid, vout, aout, &err)) {
    check(false, "compute: " + err);
    std::printf("FAILURES\n");
    return 1;
  }

  const std::vector<float> gv = ltx25::MetalOps::download_bf16(
      vout, (std::size_t)kT * cfg.dit.cross_attention_dim);
  const std::vector<float> ga = ltx25::MetalOps::download_bf16(
      aout, (std::size_t)kT * cfg.dit.audio_cross_attention_dim);

  // A 188160-deep bf16 GEMM: the accumulation alone is the floor here.
  const double bar = 6e-2;
  const double rv = npy::rel_l2(gv, video.data);
  const double ra = npy::rel_l2(ga, audio.data);
  std::printf("       video context  rel-L2 %.3e\n", rv);
  std::printf("       audio context  rel-L2 %.3e\n", ra);
  check(rv < bar, "the video context matches the reference");
  check(ra < bar, "the audio context matches the reference");

  // What the padding convention actually is, stated rather than assumed.
  //
  // The reference zeroes the NORMALISED rows -- before the projection,
  // not after -- and the projection has a BIAS, so a padded row comes
  // out as the bias, NOT as zero. Zeroing the output instead would be a
  // different (and plausible-looking) convention, so this compares the
  // padded rows against the golden's own rather than against zero.
  const int VD = cfg.dit.cross_attention_dim;
  std::vector<float> got_pad, want_pad;
  for (int t = kValid; t < kT; ++t) {
    for (int i = 0; i < VD; ++i) {
      got_pad.push_back(gv[(std::size_t)t * VD + i]);
      want_pad.push_back(video.data[(std::size_t)t * VD + i]);
    }
  }
  double e_got = 0.0, e_want = 0.0;
  for (float f : got_pad)  { e_got  += std::fabs((double)f); }
  for (float f : want_pad) { e_want += std::fabs((double)f); }
  std::printf("       padded rows: |ours| %.3f vs |reference| %.3f  "
              "(the bias, not zero)\n", e_got, e_want);
  check(e_want > 1.0, "the reference's padded rows carry the bias, "
        "confirming the zeroing happens BEFORE the projection");
  check(npy::rel_l2(got_pad, want_pad) < bar,
        "our padded rows match the reference's");

  // LEFT PADDING, which is what LTX-2.5 actually uses: the reference
  // tokenizer is PaddingSide.LEFT, so the caption is a SUFFIX. Move the
  // real rows to the end, ask for that range, and every projected row
  // must move with them. A prefix-only implementation passes everything
  // above and fails right here -- which is the whole point of checking
  // it, since a wrong padding side yields correctly-shaped output.
  const int off = kT - kValid;
  auto vout2 = ops.alloc((std::size_t)kT * VD);
  auto aout2 = ops.alloc((std::size_t)kT * cfg.dit.audio_cross_attention_dim);
  if (!tf->compute(hidden.data.data(), kT, off, kValid, vout2, aout2, &err)) {
    check(false, "left-padded compute: " + err);
  } else {
    const std::vector<float> gv2 =
        ltx25::MetalOps::download_bf16(vout2, (std::size_t)kT * VD);
    std::vector<float> moved, orig;
    for (int t = 0; t < kValid; ++t) {
      for (int i = 0; i < VD; ++i) {
        moved.push_back(gv2[(std::size_t)(off + t) * VD + i]);
        orig.push_back(gv[(std::size_t)t * VD + i]);
      }
    }
    check(npy::rel_l2(moved, orig) < 1e-6,
          "left-padded rows carry the same features, just moved");
    // And the LEADING rows are now the padded ones (the bias).
    std::vector<float> lead, want_lead;
    for (int t = 0; t < off; ++t) {
      for (int i = 0; i < VD; ++i) {
        lead.push_back(gv2[(std::size_t)t * VD + i]);
        want_lead.push_back(video.data[(std::size_t)(kValid) * VD + i]);
      }
    }
    check(npy::rel_l2(lead, want_lead) < bar,
          "the leading rows are the padding bias");
  }

  std::printf("%s\n", g_fail == 0 ? "ALL PASSED" : "FAILURES");
  return g_fail == 0 ? 0 : 1;
}
