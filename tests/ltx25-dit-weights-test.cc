// The streamed weight read, against the read it replaces.
//
// A streamed block is bound by allocating a destination and preading the
// bytes into it (generative-models/shared/streamed-refill.h) rather than
// by letting stream_tensor allocate and then memcpy out of the shard's
// mmap. That is a different ROUTE to the same bytes, so the only thing
// worth checking is that they really are the same bytes -- across every
// dtype an LTX-2.5 block actually carries, on a real checkpoint, because
// the dtype rule is where a route like this goes wrong.
//
// It also reports the SPLIT: how many of a block's tensors the raw read
// serves and how many fall back. That number is the reason the refill
// decides per tensor instead of per block -- LTX-2.5 blocks mix a few
// f32 tables in among many bf16 matrices and u32 code words, and an
// all-or-nothing rule would give up the whole block to avoid them.
//
// Env: VPIPE_LTX25_TEST_MODEL_PATH = the LTX-2.5 model directory.

#include "ltx25-config.h"
#include "ltx25-dit-weights.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/shared/streamed-refill.h"
#include "generative-models/weight-set.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using vpipe::genai::Refill;
using vpipe::genai::RefillDst;
using vpipe::genai::WeightSet;
using vpipe::genai::refill_streamed_tensor;
using vpipe::metal_compute::MetalCompute;
using vpipe::metal_compute::SharedBuffer;

namespace {

int g_fail = 0;

void
check(bool ok, const std::string& what)
{
  std::printf("  [%s] %s\n", ok ? " OK " : "FAIL", what.c_str());
  if (!ok) { ++g_fail; }
}

}  // namespace

int
main()
{
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const char* root = std::getenv("VPIPE_LTX25_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') {
    std::printf("SKIPPED: needs VPIPE_LTX25_TEST_MODEL_PATH. "
                "NOTHING was checked.\n");
    return 0;
  }
  MetalCompute mc(nullptr);
  if (!mc.valid()) {
    std::printf("SKIPPED: no Metal device. NOTHING was checked.\n");
    return 0;
  }
  ltx25::Config cfg;
  std::string err;
  if (!ltx25::resolve(root, cfg, &err, {})) {
    std::printf("SKIPPED: %s\n", err.c_str());
    return 0;
  }
  std::shared_ptr<WeightSet> ws = WeightSet::open(cfg.dit_file, nullptr);
  if (!ws) {
    std::printf("SKIPPED: cannot open the DiT\n");
    return 0;
  }

  // One block's worth of names. Block 0 is representative and is the
  // only one that has to be read to know the checkpoint's dtype mix.
  const std::string pre =
      std::string(ltx25::kDitPrefix) + "transformer_blocks.0.";
  std::vector<std::string> names;
  for (const std::string& n : ws->src().tensor_names()) {
    if (n.rfind(pre, 0) == 0) { names.push_back(n); }
  }
  check(!names.empty(), "block 0 has tensors (" +
        std::to_string(names.size()) + ")");
  if (names.empty()) {
    std::printf("FAILURES\n");
    return 1;
  }

  std::map<std::string, int> by_dtype;
  std::size_t filled = 0, unservable = 0, failed = 0;
  std::size_t filled_bytes = 0, other_bytes = 0, mismatched = 0;

  for (const std::string& n : names) {
    const auto* ti = ws->src().info(n);
    if (ti == nullptr) { continue; }
    ++by_dtype[ti->dtype];

    // The reference: the read this replaces, byte for byte.
    SharedBuffer want = ws->stream_tensor(n, &mc,
                                          WeightSet::Residency::Copied);
    if (want.empty()) { continue; }

    SharedBuffer dst = mc.make_shared_buffer(ti->nbytes);
    if (dst.empty()) { continue; }
    // kRaw, which is what the DiT's own fetch asks for: this compares
    // the two routes to the CHECKPOINT's bytes. The f16 scales a
    // quantized pack carries are converted by the plugin's own as-bf16
    // path and are therefore expected to come back unservable here.
    const Refill r = refill_streamed_tensor(*ws, n, dst, RefillDst::kRaw);
    if (r == Refill::kFilled) {
      ++filled;
      filled_bytes += ti->nbytes;
      // Nothing is converted under kRaw, so EVERY filled tensor must
      // match the read it replaces byte for byte -- no exemptions, which
      // is what makes this a real comparison rather than a sampled one.
      if (std::memcmp(dst.contents(), want.contents(), ti->nbytes) != 0) {
        ++mismatched;
      }
    } else if (r == Refill::kUnservable) {
      ++unservable;
      other_bytes += ti->nbytes;
    } else {
      ++failed;
      other_bytes += ti->nbytes;
    }
  }

  std::printf("  dtypes in block 0:");
  for (const auto& kv : by_dtype) {
    std::printf(" %s x%d", kv.first.c_str(), kv.second);
  }
  std::printf("\n");
  std::printf("  refill: %zu filled (%.1f MB), %zu unservable, %zu failed "
              "(%.1f MB on the old path)\n", filled,
              (double)filled_bytes / 1e6, unservable, failed,
              (double)other_bytes / 1e6);

  check(mismatched == 0,
        "every raw refill matches the read it replaces, byte for byte (" +
        std::to_string(mismatched) + " differ)");
  check(failed == 0, "nothing failed outright (" + std::to_string(failed) +
        ")");
  // The point of a per-tensor rule: the refill has to carry the bulk
  // even though the block is not uniform. A checkpoint that came out
  // all-unservable would mean the dtype rule does not fit this model
  // and the change is buying nothing.
  check(filled_bytes > 10 * other_bytes,
        "the raw path carries the bulk of the block");
  // The f32 tables AND the f16 scales fall back here, and that is the
  // arrangement being pinned: the plugin converts both by its own route,
  // so the refill must hand back neither.
  check(unservable > 0, "the tensors this route does not serve are named "
        "as unservable rather than failing (" +
        std::to_string(unservable) + ")");

  if (g_fail != 0) {
    std::printf("FAILURES: %d\n", g_fail);
    return 1;
  }
  std::printf("OK\n");
  return 0;
}
