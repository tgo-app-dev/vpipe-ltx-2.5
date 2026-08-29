// What an LTX-2.5 adapter turns into, checked against the REAL files.
//
// The risk here is not arithmetic -- `y += (x@A^T)@B^T` is not where a
// LoRA port goes wrong. It is the NAMES and the WIDTHS: an adapter
// written against the module tree carries one prefix level fewer than
// the checkpoint, the audio stream is an `audio_` infix the video
// modules do not have, and `to_gate_logits` writes one logit per head
// where everything beside it writes `inner`. Bind any of those to the
// wrong base linear and the run does not fail, it renders slightly
// wrong.
//
// So this loads the actual published adapters and asserts the shape of
// what came back. Gated on the model paths and SKIPS when they are
// unset -- read the output, not the exit code: a vacuous skip looks
// exactly like a pass, and this says which it did.

#include "ltx25-config.h"
#include "ltx25-lora.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "generative-models/weight-set.h"

#include <memory>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

int g_fail = 0;
int g_ran  = 0;

void
check(bool ok, const std::string& what)
{
  std::printf("  [%s] %s\n", ok ? " OK " : "FAIL", what.c_str());
  if (!ok) { ++g_fail; }
  ++g_ran;
}

const char*
env_(const char* k)
{
  const char* v = std::getenv(k);
  return (v != nullptr && *v != '\0') ? v : nullptr;
}

// The DiT config both adapters were trained against. Taken from the
// defaults rather than from a checkpoint so this test needs only the
// 327 MB adapter, not the 39 GB model beside it.
ltx25::DitConfig
cfg_()
{
  return ltx25::DitConfig{};
}

void
test_ic_lora(const char* file)
{
  std::printf("IC-LoRA pixel spatial upscaler: %s\n", file);
  // A PRIVATE set and a bare MetalCompute: no session here, which is
  // the offline-tool path WeightSet::open(dir, nullptr) exists for.
  vpipe::metal_compute::MetalCompute mc(nullptr);
  auto ws = vpipe::genai::WeightSet::open(file, nullptr);
  if (!ws) { check(false, "weight set opened"); return; }

  std::vector<std::string> unmapped;
  std::string err;
  auto a = ltx25::LoraAdapter::load(*ws, &mc, cfg_(), 1.0,
                                    file, &unmapped, &err);
  if (!a) {
    check(false, "loaded: " + err);
    for (std::size_t i = 0; i < unmapped.size() && i < 4; ++i) {
      std::printf("        unmapped: %s\n", unmapped[i].c_str());
    }
    return;
  }
  check(true, "loaded");
  // 48 blocks x 10 video-stream modules. Every one of them has to bind:
  // this adapter names nothing else, so a lower count is a name that
  // did not match and a higher one is a module bound twice.
  check(a->modules() == 480,
        "480 modules (got " + std::to_string(a->modules()) + ")");
  check(a->max_rank() == 32,
        "rank 32 (got " + std::to_string(a->max_rank()) + ")");
  // ff.net.0.proj widens to 4*inner_dim, and that is what the runtime
  // scratch has to be sized for.
  check(a->max_out() == 16384,
        "widest output 16384 (got " + std::to_string(a->max_out()) + ")");
  // The whole point of the adapter, and the value the conditioning
  // dilates by. Straight out of the file's __metadata__.
  check(a->reference_downscale_factor() == 2,
        "reference_downscale_factor 2 (got "
            + std::to_string(a->reference_downscale_factor()) + ")");
  check(a->wants_reference(), "declares itself an IC-LoRA");
  // Video stream only: no audio, no cross-attention, no trunk.
  check(a->trunk() == nullptr, "touches no trunk weight");
  const ltx25::LoraBlock* b0 = a->block(0);
  check(b0 != nullptr, "block 0 adapted");
  if (b0 != nullptr) {
    check(b0->video_attn1.q.valid() && b0->video_attn1.o.valid(),
          "block 0 video self-attention q/o bound");
    check(b0->video_ff_in.valid() && b0->video_ff_out.valid(),
          "block 0 video ff bound");
    check(b0->video_ff_in.n == 16384,
          "block 0 ff_in widens to 16384 (got "
              + std::to_string(b0->video_ff_in.n) + ")");
    check(b0->video_ff_out.k == 16384,
          "block 0 ff_out reads 16384 (got "
              + std::to_string(b0->video_ff_out.k) + ")");
    // The modules this adapter deliberately does NOT carry. Binding one
    // would mean a name matched something it should not.
    check(!b0->video_attn1.gate.valid(), "block 0 gate NOT adapted");
    check(!b0->audio_attn1.q.valid(), "block 0 audio NOT adapted");
    check(!b0->a2v.q.valid(), "block 0 cross-attention NOT adapted");
  }
  check(a->block(47) != nullptr, "block 47 adapted (all 48 present)");
  std::printf("        %d modules, %.1f MB resident\n", a->modules(),
              (double)a->bytes() / 1e6);
}

// The distilled adapter is the OTHER shape: 62 module kinds, mixed rank
// (450 for the projections, 32 for every gate), the audio stream, both
// cross-attentions and the whole trunk. It is what proves the loader
// reads an adapter rather than one adapter.
void
test_distilled(const char* file)
{
  std::printf("distilled-450 adapter: %s\n", file);
  // A PRIVATE set and a bare MetalCompute: no session here, which is
  // the offline-tool path WeightSet::open(dir, nullptr) exists for.
  vpipe::metal_compute::MetalCompute mc(nullptr);
  auto ws = vpipe::genai::WeightSet::open(file, nullptr);
  if (!ws) { check(false, "weight set opened"); return; }

  std::vector<std::string> unmapped;
  std::string err;
  auto a = ltx25::LoraAdapter::load(*ws, &mc, cfg_(), 1.0,
                                    file, &unmapped, &err);
  if (!a) {
    check(false, "loaded: " + err);
    for (std::size_t i = 0; i < unmapped.size() && i < 6; ++i) {
      std::printf("        unmapped: %s\n", unmapped[i].c_str());
    }
    return;
  }
  check(true, "loaded");
  // Mixed rank in ONE file is the property a single-rank loader gets
  // wrong, so it is asserted rather than assumed.
  check(a->max_rank() == 450,
        "max rank 450 (got " + std::to_string(a->max_rank()) + ")");
  check(a->trunk() != nullptr, "trunk adapted");
  const ltx25::LoraBlock* b0 = a->block(0);
  check(b0 != nullptr, "block 0 adapted");
  if (b0 != nullptr) {
    check(b0->audio_attn1.q.valid(), "audio stream adapted");
    check(b0->a2v.q.valid() && b0->v2a.q.valid(),
          "both cross-attentions adapted");
    check(b0->video_attn1.gate.valid(), "gate logits adapted");
    // The gate stays at rank 32 while its neighbours are 450 -- the
    // exact case a per-adapter rank would have mangled.
    check(b0->video_attn1.gate.rank == 32,
          "gate rank 32 beside rank-450 peers (got "
              + std::to_string(b0->video_attn1.gate.rank) + ")");
    check(b0->video_attn1.q.rank == 450,
          "projection rank 450 (got "
              + std::to_string(b0->video_attn1.q.rank) + ")");
  }
  const ltx25::LoraTrunk* t = a->trunk();
  if (t != nullptr) {
    check(t->patchify.valid() && t->proj_out.valid(),
          "patchify + output head adapted");
    check(t->video.out.valid() && t->video.emb1.valid(),
          "the video adaLN chain adapted");
    check(t->av_a2v_gate.out.valid(), "the cross-attention gate chain");
  }
  // A plain LoRA: no reference, nothing for the conditioning to dilate.
  check(!a->wants_reference(), "not an IC-LoRA");
  std::printf("        %d modules, %.1f MB resident\n", a->modules(),
              (double)a->bytes() / 1e6);
}

}  // namespace

int
main()
{
  const char* ic = env_("VPIPE_LTX25_IC_LORA_PATH");
  const char* di = env_("VPIPE_LTX25_LORA_PATH");
  if (ic != nullptr) { test_ic_lora(ic); }
  else {
    std::printf("SKIP: VPIPE_LTX25_IC_LORA_PATH unset "
                "(the IC-LoRA .safetensors)\n");
  }
  if (di != nullptr) { test_distilled(di); }
  else {
    std::printf("SKIP: VPIPE_LTX25_LORA_PATH unset "
                "(the distilled-450 .safetensors)\n");
  }
  if (g_ran == 0) {
    std::printf("ltx25-lora: nothing ran -- both adapters absent\n");
    return 0;
  }
  std::printf("ltx25-lora: %d checks, %d failed\n", g_ran, g_fail);
  return g_fail == 0 ? 0 : 1;
}
