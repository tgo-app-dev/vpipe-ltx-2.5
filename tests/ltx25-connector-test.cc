// The embeddings connectors on the real checkpoint.
//
// Three things here have no shape consequence and would each be silent:
//
//   * the connector's feed-forward HAS bias where the DiT's video
//     feed-forward does not;
//   * padded positions are REPLACED by learnable registers, tiled over
//     the sequence, and the attention mask is then discarded -- so the
//     attention is FULL, not masked;
//   * the RoPE is 1-D at the CONNECTOR's own max_pos (4096), not the
//     DiT's [20, 2048, 2048].
//
// The golden pads deliberately (100 valid of 256) so the substitution is
// exercised rather than skipped.

#include "ltx25-config.h"
#include "ltx25-connector.h"
#include "ltx25-metal-ops.h"
#include "npy.h"

#include "generative-models/weight-set.h"

#include <cstdio>
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

void
check(bool ok, const std::string& what)
{
  std::printf("  [%s] %s\n", ok ? " OK " : "FAIL", what.c_str());
  if (!ok) { ++g_fail; }
}

npy::Array
g(const std::string& n)
{
  return npy::load(g_dir + "/" + n + ".npy");
}

constexpr int kSeq = 256, kValid = 100;

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
  std::shared_ptr<WeightSet> ws = WeightSet::open(cfg.dit_file, nullptr);
  if (!ws) { std::printf("SKIPPED: cannot open the DiT\n"); return 0; }

  ltx25::MetalOps ops;
  if (!ops.init(&mc, &err)) {
    std::printf("FAILED ops: %s\n", err.c_str());
    return 1;
  }

  // The connector is ~8 blocks x (4 x 4096^2 + 2 x 4096 x 16384) -- about
  // 3.2 GB for the video one. Nothing like the DiT's 39 GB.
  for (int audio = 0; audio < 2; ++audio) {
    const char* tag = audio ? "audio" : "video";
    std::printf("%s connector\n", tag);
    auto conn = ltx25::Ltx25Connector::load(cfg.dit, *ws, ops, audio != 0,
                                            &err);
    if (!conn) { check(false, std::string(tag) + ": load: " + err); continue; }
    check(true, std::string(tag) + " connector loaded (" +
          std::to_string(conn->dim()) + " wide, " +
          std::to_string(conn->registers()) + " registers)");

    // A sequence that is not a whole number of register tiles has no
    // defined substitution -- the reference asserts it, so this refuses.
    std::string e2;
    check(!conn->reserve(kSeq - 1, &e2),
          "a non-multiple of the register count is refused");

    if (!conn->reserve(kSeq, &err)) {
      check(false, std::string(tag) + ": reserve: " + err);
      continue;
    }

    npy::Array in = g(std::string("conn_") + tag + "_in");
    npy::Array want = g(std::string("conn_") + tag + "_out");
    if (!in.ok || !want.ok) {
      check(false, std::string(tag) + " goldens: " + in.err + want.err);
      continue;
    }
    auto in_b = ops.upload_bf16(in.data);
    auto out_b = ops.alloc((std::size_t)kSeq * conn->dim());
    if (!conn->forward(in_b, out_b, kSeq, kValid, &err)) {
      check(false, std::string(tag) + ": forward: " + err);
      continue;
    }
    const std::vector<float> got =
        ltx25::MetalOps::download_bf16(out_b, (std::size_t)kSeq * conn->dim());
    // 8 blocks deep in bf16, each with a 4096-wide GEMM chain.
    const double bar = 5e-2;
    const double r = npy::rel_l2(got, want.data);
    std::printf("       rel-L2 %.3e (bar %.0e)\n", r, bar);
    check(r < bar, std::string(tag) + " connector matches the reference");

    // The registers really are used: the padded tail must differ from
    // what a run with no padding produces. If the substitution were a
    // no-op the two would be identical, and the golden alone would not
    // say so.
    auto out2 = ops.alloc((std::size_t)kSeq * conn->dim());
    if (conn->forward(in_b, out2, kSeq, kSeq, &err)) {
      const std::vector<float> all_valid = ltx25::MetalOps::download_bf16(
          out2, (std::size_t)kSeq * conn->dim());
      check(npy::rel_l2(got, all_valid) > 1e-3,
            "the learnable registers actually replace the padding");
    }
  }

  std::printf("%s\n", g_fail == 0 ? "ALL PASSED" : "FAILURES");
  return g_fail == 0 ? 0 : 1;
}
