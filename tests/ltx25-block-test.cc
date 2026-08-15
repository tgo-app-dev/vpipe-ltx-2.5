// One AV transformer block, against the reference.
//
// Run on a SMALL config (video 64/4 heads, audio 32/4) whose every weight
// fits in a golden, because what is being checked is STRUCTURE, not
// arithmetic: which of the nine scale_shift_table rows drives which
// modulation, that the text cross-attention modulates K/V as well as Q,
// that both audio<->video directions read the PRE-cross snapshot, and
// that gating is 2*sigmoid per head applied before to_out. Every one of
// those is invisible in the output shape.
//
// The three outputs matter together: a port that gets the cross-attention
// snapshot wrong still matches the video-only and audio-only runs, so a
// mismatch localises rather than just failing.

#include "ltx25-block-ref.h"
#include "ltx25-rope.h"
#include "npy.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

int g_fail = 0;

void
check(bool ok, const std::string& what)
{
  std::printf("  [%s] %s\n", ok ? " OK " : "FAIL", what.c_str());
  if (!ok) { ++g_fail; }
}

// gen_goldens.py's SMALL config.
constexpr int kVD = 64, kVH = 4, kAD = 32, kAH = 4;
constexpr int kTV = 6, kTA = 4, kTT = 3;

std::string g_dir;

npy::Array
g(const std::string& name)
{
  return npy::load(g_dir + "/" + name + ".npy");
}

// The goldens name weights as `small_w__<param path with . -> _>`.
bool
load_weights(ltx25::BlockWeights& w)
{
  static const char* kNames[] = {
    "attn1.to_q.weight", "attn1.to_q.bias", "attn1.to_k.weight",
    "attn1.to_k.bias", "attn1.to_v.weight", "attn1.to_v.bias",
    "attn1.to_out.0.weight", "attn1.to_out.0.bias", "attn1.q_norm.weight",
    "attn1.k_norm.weight", "attn1.to_gate_logits.weight",
    "attn1.to_gate_logits.bias",
    "attn2.to_q.weight", "attn2.to_q.bias", "attn2.to_k.weight",
    "attn2.to_k.bias", "attn2.to_v.weight", "attn2.to_v.bias",
    "attn2.to_out.0.weight", "attn2.to_out.0.bias", "attn2.q_norm.weight",
    "attn2.k_norm.weight", "attn2.to_gate_logits.weight",
    "attn2.to_gate_logits.bias",
    "ff.net.0.proj.weight", "ff.net.2.weight",
    "scale_shift_table", "prompt_scale_shift_table",
    "audio_attn1.to_q.weight", "audio_attn1.to_q.bias",
    "audio_attn1.to_k.weight", "audio_attn1.to_k.bias",
    "audio_attn1.to_v.weight", "audio_attn1.to_v.bias",
    "audio_attn1.to_out.0.weight", "audio_attn1.to_out.0.bias",
    "audio_attn1.q_norm.weight", "audio_attn1.k_norm.weight",
    "audio_attn1.to_gate_logits.weight", "audio_attn1.to_gate_logits.bias",
    "audio_attn2.to_q.weight", "audio_attn2.to_q.bias",
    "audio_attn2.to_k.weight", "audio_attn2.to_k.bias",
    "audio_attn2.to_v.weight", "audio_attn2.to_v.bias",
    "audio_attn2.to_out.0.weight", "audio_attn2.to_out.0.bias",
    "audio_attn2.q_norm.weight", "audio_attn2.k_norm.weight",
    "audio_attn2.to_gate_logits.weight", "audio_attn2.to_gate_logits.bias",
    "audio_ff.net.0.proj.weight", "audio_ff.net.0.proj.bias",
    "audio_ff.net.2.weight", "audio_ff.net.2.bias",
    "audio_scale_shift_table", "audio_prompt_scale_shift_table",
    "audio_to_video_attn.to_q.weight", "audio_to_video_attn.to_q.bias",
    "audio_to_video_attn.to_k.weight", "audio_to_video_attn.to_k.bias",
    "audio_to_video_attn.to_v.weight", "audio_to_video_attn.to_v.bias",
    "audio_to_video_attn.to_out.0.weight",
    "audio_to_video_attn.to_out.0.bias",
    "audio_to_video_attn.q_norm.weight", "audio_to_video_attn.k_norm.weight",
    "audio_to_video_attn.to_gate_logits.weight",
    "audio_to_video_attn.to_gate_logits.bias",
    "video_to_audio_attn.to_q.weight", "video_to_audio_attn.to_q.bias",
    "video_to_audio_attn.to_k.weight", "video_to_audio_attn.to_k.bias",
    "video_to_audio_attn.to_v.weight", "video_to_audio_attn.to_v.bias",
    "video_to_audio_attn.to_out.0.weight",
    "video_to_audio_attn.to_out.0.bias",
    "video_to_audio_attn.q_norm.weight", "video_to_audio_attn.k_norm.weight",
    "video_to_audio_attn.to_gate_logits.weight",
    "video_to_audio_attn.to_gate_logits.bias",
    "scale_shift_table_a2v_ca_audio", "scale_shift_table_a2v_ca_video",
  };
  std::unordered_map<std::string, ltx25::NamedTensor> t;
  for (const char* n : kNames) {
    std::string file = "small_w__";
    for (const char* p = n; *p; ++p) { file += (*p == '.') ? '_' : *p; }
    npy::Array a = g(file);
    if (!a.ok) {
      check(false, "load weight " + std::string(n) + ": " + a.err);
      return false;
    }
    t[n] = ltx25::NamedTensor{a.data, a.shape};
  }
  std::string err;
  if (!ltx25::load_block_weights(t, kVD, kVH, kAD, kAH, w, &err)) {
    check(false, "assemble block: " + err);
    return false;
  }
  return true;
}

ltx25::Mat
mat(const npy::Array& a, int rows, int cols)
{
  ltx25::Mat m;
  m.rows = rows;
  m.cols = cols;
  m.v = a.data;
  return m;
}

void
run(const ltx25::BlockWeights& w, bool with_video, bool with_audio,
    const ltx25::RopeTable& vpe, const ltx25::RopeTable& ape,
    const ltx25::RopeTable& vcpe, ltx25::Mat* vout, ltx25::Mat* aout)
{
  ltx25::StreamInput v, a;
  v.x       = mat(g("small_vx_in"), kTV, kVD);
  v.context = mat(g("small_vctx"),  kTT, kVD);
  v.timesteps         = g("small_v_ts").data;
  v.cross_scale_shift = g("small_v_css").data;
  v.cross_gate        = g("small_v_cg").data;
  v.prompt_timestep   = g("small_v_pts").data;
  v.pe = &vpe;
  v.cross_pe = &vcpe;
  v.present = with_video;

  a.x       = mat(g("small_ax_in"), kTA, kAD);
  a.context = mat(g("small_actx"),  kTT, kAD);
  a.timesteps         = g("small_a_ts").data;
  a.cross_scale_shift = g("small_a_css").data;
  a.cross_gate        = g("small_a_cg").data;
  a.prompt_timestep   = g("small_a_pts").data;
  a.pe = &ape;
  a.cross_pe = &ape;      // audio's cross table IS its self table here
  a.present = with_audio;

  std::string err;
  if (!ltx25::block_forward(w, v, a, &err)) {
    check(false, "block_forward: " + err);
    return;
  }
  if (vout != nullptr) { *vout = v.x; }
  if (aout != nullptr) { *aout = a.x; }
}

void
cmp(const char* what, const ltx25::Mat& got, const std::string& golden,
    double bar)
{
  npy::Array w = g(golden);
  if (!w.ok) { check(false, std::string(what) + ": " + w.err); return; }
  const double r = npy::rel_l2(got.v, w.data);
  std::printf("       %-22s rel-L2 %.3e (max abs %.3e)\n", what, r,
              npy::max_abs_diff(got.v, w.data));
  check(r < bar, std::string(what) + " matches the reference");
}

}  // namespace

int
main()
{
  const char* dir = std::getenv("VPIPE_LTX25_GOLDENS");
  if (dir == nullptr) {
    std::printf("SKIPPED: set VPIPE_LTX25_GOLDENS to the directory "
                "gen_goldens.py wrote (needs the `small` goldens).\n"
                "NOTHING was checked.\n");
    return 0;
  }
  g_dir = dir;

  ltx25::BlockWeights w;
  if (!load_weights(w)) {
    std::printf("FAILURES\n");
    return 1;
  }
  check(true, "block weights assembled from the reference's own names");

  // The same 1-D tables gen_goldens.py built. The cross tables are at the
  // AUDIO width for BOTH streams -- that is the trap this rebuilds rather
  // than reading from a golden, so a wrong width fails here.
  const ltx25::RopeTable vpe =
      ltx25::build_index_rope(kTV, {20}, kVD, kVH);
  const ltx25::RopeTable ape =
      ltx25::build_index_rope(kTA, {20}, kAD, kAH);
  const ltx25::RopeTable vcpe =
      ltx25::build_index_rope(kTV, {20}, kAD, kAH);

  // f32 accumulation over a 64-wide block: the reference sums in f32 on
  // MPS/CPU, this sums in f64 and narrows, so a few 1e-7s are expected.
  const double kBar = 1e-5;

  std::printf("joint (video + audio)\n");
  ltx25::Mat vj, aj;
  run(w, true, true, vpe, ape, vcpe, &vj, &aj);
  cmp("video out", vj, "small_vx_out", kBar);
  cmp("audio out", aj, "small_ax_out", kBar);

  // Single-stream runs. These exercise the same block with the
  // audio<->video cross-attention SKIPPED, so together with the joint run
  // above they separate "the block is wrong" from "the cross-attention is
  // wrong".
  std::printf("video only (audio absent)\n");
  ltx25::Mat v1, a1;
  run(w, true, false, vpe, ape, vcpe, &v1, &a1);
  cmp("video out", v1, "small_vx_out_noaudio", kBar);

  std::printf("audio only (video absent)\n");
  ltx25::Mat v2, a2;
  run(w, false, true, vpe, ape, vcpe, &v2, &a2);
  cmp("audio out", a2, "small_ax_out_novideo", kBar);

  std::printf("%s\n", g_fail == 0 ? "ALL PASSED" : "FAILURES");
  return g_fail == 0 ? 0 : 1;
}
