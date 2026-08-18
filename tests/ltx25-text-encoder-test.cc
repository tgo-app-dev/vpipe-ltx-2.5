// Ltx25TextEncoder: the JOIN of the Gemma-4 12B forward (through the
// host's HiddenStateEncoder seam) and LTX's own projections.
//
// The two halves are already verified apart -- the projections against
// reference goldens at 2.3e-3, the tap against the host's own tests --
// so what is left to pin here is the JOIN, and specifically the things
// that are wrong-but-plausible across it:
//
//   * the whole 49-state stack actually arrives (a model with a
//     KV-shared tail serves part of it and the concatenation is
//     silently short);
//   * the tokenization matches the reference's rules (leading BOS,
//     stripped caption, truncation at 1024);
//   * the caption lands as a SUFFIX -- LTX pads LEFT, and a prefix
//     layout is the natural guess and produces the right shape.
//
// Needs VPIPE_LTX25_TEST_MODEL_PATH. This loads a 12B text encoder, so
// it is not cheap.

#include "ltx25-config.h"
#include "ltx25-metal-ops.h"
#include "ltx25-text-encoder.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "generative-models/hidden-state-encoder.h"
#include "generative-models/weight-set.h"
#include "npy.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <mach/mach.h>
#include <mach/task_info.h>

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

// The whole process, the way the tree measures memory -- phys_footprint
// and not resident_size, because a Metal buffer's pages are wired
// through IOKit and resident_size does not see all of them.
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
  const std::string root = env_("VPIPE_LTX25_TEST_MODEL_PATH");
  if (root.empty()) {
    std::printf("SKIPPED: needs VPIPE_LTX25_TEST_MODEL_PATH. "
                "NOTHING was checked.\n");
    return 0;
  }

  vpipe::metal_compute::MetalCompute metal(nullptr);
  if (!metal.valid()) {
    std::printf("SKIPPED: no Metal device. NOTHING was checked.\n");
    return 0;
  }
  metal.register_metal_library(ltx25::kMetalLibBf16,
                               ltx25_kernels_bf16_metallib,
                               ltx25_kernels_bf16_metallib_len);
  auto* mc = &metal;

  // WHICH ENCODER, because the answer decides whether the run fits.
  // Empty is the shipped default, which deliberately prefers the released
  // bf16 file over any quantized directory; VPIPE_LTX25_ENC_VARIANT is how
  // a bounded box asks for one of the packs beside it. The footprint
  // printed below is the reason this knob is here: the encoder is the
  // largest single thing a conditioning graph loads, and 24 GB, 15 GB and
  // 9.9 GB are three different machines.
  const std::string enc_variant = env_("VPIPE_LTX25_ENC_VARIANT");
  const double fp_start = footprint_mb_();

  ltx25::Config cfg;
  std::string err;
  if (!ltx25::resolve(root, cfg, &err, {}, enc_variant)) {
    check(false, "resolve: " + err);
    std::printf("FAILURES\n");
    return 1;
  }

  ltx25::MetalOps ops;
  if (!ops.init(mc, &err)) {
    check(false, "metal ops: " + err);
    std::printf("FAILURES\n");
    return 1;
  }

  auto ws = vpipe::genai::open_weight_set(cfg.enc_file, nullptr);
  if (!ws) {
    check(false, "could not open " + cfg.enc_file);
    std::printf("FAILURES\n");
    return 1;
  }

  auto enc = ltx25::Ltx25TextEncoder::load(cfg, ws, ops, mc, nullptr, &err);
  if (!enc) {
    check(false, "load: " + err);
    std::printf("FAILURES\n");
    return 1;
  }
  check(true, "the text encoder loaded");
  std::printf("       encoder '%s'\n", cfg.enc_file.c_str());
  std::printf("       footprint %.0f MB -> %.0f MB (+%.0f MB to load it)\n",
              fp_start, footprint_mb_(), footprint_mb_() - fp_start);
  check(enc->layers() == 49, "49 hidden states (48 layers + the embedding)");
  check(enc->hidden_dim() == 3840, "3840-wide hidden states");

  // ---- tokenization -------------------------------------------------
  const std::string prompt =
      "  A red fox leaps over a snowy log at dawn.  ";
  const auto ids = enc->tokenize(prompt, ltx25::Ltx25TextEncoder::kMaxTokens);
  check(!ids.empty(), "the caption tokenizes");
  std::printf("       %zu tokens, first id %d\n", ids.size(),
              ids.empty() ? -1 : (int)ids[0]);
  // Gemma-4 does not emit BOS itself; the reference prepends it and
  // refuses without one.
  check(ids.size() > 1 && ids[0] == 2, "a leading <bos> (id 2)");
  // Stripping is the reference's, not cosmetic: a leading space is a
  // real extra token.
  const auto ids_stripped =
      enc->tokenize("A red fox leaps over a snowy log at dawn.",
                    ltx25::Ltx25TextEncoder::kMaxTokens);
  check(ids == ids_stripped, "the caption is stripped before tokenizing");
  // Truncation keeps room for the BOS.
  std::string longp;
  for (int i = 0; i < 4000; ++i) { longp += "fox "; }
  const auto ids_long = enc->tokenize(longp, 64);
  check((int)ids_long.size() == 64, "truncated to the cap, BOS included");
  check(ids_long[0] == 2, "the BOS survives truncation");

  // ---- the full encode ----------------------------------------------
  const int pad_to = 256;      // a whole number of 128-register tiles
  const int VD = cfg.dit.cross_attention_dim;
  const int AD = cfg.dit.audio_cross_attention_dim;
  auto vctx = ops.alloc((std::size_t)pad_to * VD);
  auto actx = ops.alloc((std::size_t)pad_to * AD);
  int vb = -1, vc = -1;
  if (!enc->encode(prompt, pad_to, vctx, actx, &vb, &vc, &err)) {
    check(false, "encode: " + err);
    std::printf("FAILURES\n");
    return 1;
  }
  check(true, "the caption encoded end to end");
  check(vc == (int)ids.size(), "the reported valid count is the token count");
  // LEFT padding: the caption is a SUFFIX.
  check(vb == pad_to - vc, "the caption lands at the END of the context");
  std::printf("       valid rows [%d, %d) of %d\n", vb, vb + vc, pad_to);

  const std::vector<float> v =
      ltx25::MetalOps::download_bf16(vctx, (std::size_t)pad_to * VD);
  const std::vector<float> a =
      ltx25::MetalOps::download_bf16(actx, (std::size_t)pad_to * AD);

  int bad = 0;
  for (float f : v) { if (!std::isfinite(f)) { ++bad; } }
  for (float f : a) { if (!std::isfinite(f)) { ++bad; } }
  check(bad == 0, "every context value is finite");

  // The real rows must actually carry the caption, and the padded rows
  // must be the projection BIAS (the zeroing happens before the Linear).
  // Those two being INDISTINGUISHABLE is what a dead encoder looks like.
  auto row_energy = [&](const std::vector<float>& buf, int dim, int r) {
    double s = 0.0;
    for (int i = 0; i < dim; ++i) {
      const double x = buf[(std::size_t)r * dim + i];
      s += x * x;
    }
    return std::sqrt(s / dim);
  };
  const double pad_e  = row_energy(v, VD, 0);            // a padded row
  const double real_e = row_energy(v, VD, pad_to - 1);   // the last caption row
  std::printf("       padded row rms %.4f, caption row rms %.4f\n",
              pad_e, real_e);
  check(real_e > 1e-3, "the caption rows carry signal");
  check(std::fabs(real_e - pad_e) > 1e-4,
        "the caption rows differ from the padding bias");

  // Every padded row is the SAME row (the bias does not vary), which is
  // what confirms they were zeroed uniformly before the projection.
  double maxdiff = 0.0;
  for (int r = 1; r < vb; ++r) {
    for (int i = 0; i < VD; ++i) {
      maxdiff = std::max(maxdiff,
                         (double)std::fabs(v[(std::size_t)r * VD + i] -
                                           v[(std::size_t)i]));
    }
  }
  check(maxdiff < 1e-6, "all padded rows are identical (the bias)");

  // Determinism, and that the prompt actually reaches the output: a
  // different caption must give a different context.
  auto vctx2 = ops.alloc((std::size_t)pad_to * VD);
  auto actx2 = ops.alloc((std::size_t)pad_to * AD);
  if (enc->encode(prompt, pad_to, vctx2, actx2, nullptr, nullptr, &err)) {
    const std::vector<float> v2 =
        ltx25::MetalOps::download_bf16(vctx2, (std::size_t)pad_to * VD);
    double d = 0.0;
    for (std::size_t i = 0; i < v.size(); ++i) {
      d = std::max(d, (double)std::fabs(v[i] - v2[i]));
    }
    check(d == 0.0, "the same caption gives the same context");
  } else {
    check(false, "re-encode: " + err);
  }

  if (enc->encode("A blue whale breaches in a calm sea.", pad_to, vctx2,
                  actx2, nullptr, nullptr, &err)) {
    const std::vector<float> v3 =
        ltx25::MetalOps::download_bf16(vctx2, (std::size_t)pad_to * VD);
    double d = 0.0;
    for (std::size_t i = 0; i < v.size(); ++i) {
      d = std::max(d, (double)std::fabs(v[i] - v3[i]));
    }
    std::printf("       max |delta| across captions %.4f\n", d);
    check(d > 1e-3, "a different caption gives a different context");
  } else {
    check(false, "second caption: " + err);
  }

  // ---- the 12B forward, against a REFERENCE forward -----------------
  //
  // Everything above checks the encoder against itself. This is the one
  // check that says the Gemma-4 hidden states are the RIGHT ones:
  // gen_goldens.py `gemma` runs transformers' Gemma4TextModel on the
  // same caption, in the same bf16 the checkpoint ships, and saves all
  // 49 states.
  const std::string gdir = env_("VPIPE_LTX25_GOLDENS");
  if (gdir.empty()) {
    std::printf("  [SKIP] no VPIPE_LTX25_GOLDENS: the 12B forward is NOT "
                "checked against a reference\n");
  } else {
    npy::Array gid  = npy::load(gdir + "/te_gemma_ids.npy");
    npy::Array ghid = npy::load(gdir + "/te_gemma_hidden.npy");
    if (!gid.ok || !ghid.ok) {
      std::printf("  [SKIP] no gemma goldens (run gen_goldens.py gemma): "
                  "%s%s\n", gid.err.c_str(), ghid.err.c_str());
    } else {
      std::vector<std::int32_t> rid;
      for (float f : gid.data) { rid.push_back((std::int32_t)f); }
      // The tokenizations must agree BEFORE the states are compared --
      // otherwise a mismatch below is ambiguous between a bad forward
      // and a bad tokenizer, and the tokenizer is the cheaper suspect.
      const auto ours =
          enc->tokenize(prompt, ltx25::Ltx25TextEncoder::kMaxTokens);
      check(ours == rid, "our tokenization matches the reference's");

      vpipe::genai::HiddenTapResult tap;
      std::string terr;
      if (!enc->lm()->encode(rid, enc->lm()->available_indices(), &tap,
                             &terr)) {
        check(false, "reference-id encode: " + terr);
      } else {
        // golden is [tokens][D][L]; the tap is [L][tokens][D].
        const int T = tap.tokens, D = tap.hidden, L = tap.slots;
        check((int)ghid.data.size() == T * D * L,
              "the golden and the tap have the same element count");
        std::vector<float> got((std::size_t)T * D * L);
        const auto* src =
            static_cast<const std::uint16_t*>(tap.data.contents());
        const bool bf = (tap.dtype != "f16");
        for (int l = 0; l < L; ++l) {
          for (int t = 0; t < T; ++t) {
            for (int d = 0; d < D; ++d) {
              const std::uint16_t b =
                  src[((std::size_t)l * T + t) * D + d];
              std::uint32_t w;
              if (bf) {
                w = (std::uint32_t)b << 16;
              } else {
                const std::uint32_t sg = (std::uint32_t)(b >> 15) << 31;
                const std::uint32_t ex = (b >> 10) & 0x1f;
                const std::uint32_t mn = b & 0x3ff;
                w = (ex == 0) ? sg
                  : (ex == 0x1f) ? (sg | 0x7f800000u | (mn << 13))
                  : (sg | ((ex + 112) << 23) | (mn << 13));
              }
              float f; std::memcpy(&f, &w, 4);
              got[((std::size_t)t * D + d) * L + l] = f;
            }
          }
        }
        // A raw [T][D][L] f32 dump, so a divergence can be taken apart
        // against the reference offline instead of by re-reading code.
        const std::string dump = env_("VPIPE_LTX25_DUMP");
        if (!dump.empty()) {
          std::FILE* fp = std::fopen(dump.c_str(), "wb");
          if (fp != nullptr) {
            std::fwrite(got.data(), 4, got.size(), fp);
            std::fclose(fp);
            std::printf("       dumped %zu floats to %s\n", got.size(),
                        dump.c_str());
          }
        }
        const double r = npy::rel_l2(got, ghid.data);
        std::printf("       all %d hidden states  rel-L2 %.3e\n", L, r);
        // Layer 0 is a pure embedding lookup with no accumulation, so it
        // pins the table and the scaling exactly; the stack as a whole
        // accumulates 48 layers of bf16, which is where the reference's
        // own padded/unpadded runs already differ by 4.4e-3.
        std::vector<float> g0, w0;
        for (int t = 0; t < T; ++t) {
          for (int d = 0; d < D; ++d) {
            g0.push_back(got[((std::size_t)t * D + d) * L]);
            w0.push_back(ghid.data[((std::size_t)t * D + d) * L]);
          }
        }
        // Per-layer, so a divergence is LOCATED rather than just seen.
        const bool verbose = !env_("VPIPE_LTX25_VERBOSE").empty();
        for (int l = 0; verbose && l <= L; ++l) {
          std::vector<float> ga, wa;
          for (int t = 0; t < T; ++t) {
            for (int d = 0; d < D; ++d) {
              ga.push_back(got[((std::size_t)t * D + d) * L + l]);
              wa.push_back(ghid.data[((std::size_t)t * D + d) * L + l]);
            }
          }
          std::printf("         state %2d  rel-L2 %.3e\n", l,
                      npy::rel_l2(ga, wa));
        }
        const double r0 = npy::rel_l2(g0, w0);
        std::printf("       hidden state 0 (embeddings)  rel-L2 %.3e\n", r0);
        // BARS, and why they are where they are. Both sides run bf16, so
        // ~1 ulp is 2^-8 = 3.9e-3 relative. State 0 is a table lookup plus
        // the sqrt(hidden) scale -- one rounding, so 5e-3. For the stack,
        // the REFERENCE's own left-padded and unpadded runs differ by
        // 4.4e-3 (48 layers of bf16 accumulation), so anything at that
        // level is agreement and 1.5e-2 is the honest ceiling. A tighter
        // bar would just be re-tuned every time it tripped.
        check(r0 < 5e-3, "the embedding output matches the reference");
        check(r < 1.5e-2, "the whole 49-state stack matches the reference");
      }
    }
  }

  std::printf("%s\n", g_fail == 0 ? "ALL PASSED" : "FAILURES");
  return g_fail == 0 ? 0 : 1;
}
