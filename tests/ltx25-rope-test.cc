// LTX-2.5's RoPE against the reference, tensor for tensor.
//
// This is the piece with the most ways to be silently wrong -- log-spaced
// frequencies, fractional signed positions, a midpoint grid, front
// padding, a head split, and a half rotation -- and every one of them
// produces a clean forward with scrambled attention. So it is checked
// against goldens produced by the reference's own
// `precompute_freqs_cis` / `apply_split_rotary_emb`, not against a
// re-derivation of the same formula.
//
// Goldens come from gen_goldens.py (`rope`), which needs no weights.
// Point VPIPE_LTX25_GOLDENS at the directory; without it this SKIPS and
// says so.

#include "ltx25-rope.h"
#include "npy.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

int g_fail = 0;

void
check(bool ok, const std::string& what)
{
  std::printf("  [%s] %s\n", ok ? " OK " : "FAIL", what.c_str());
  if (!ok) { ++g_fail; }
}

// The golden's geometry, fixed by gen_goldens.py: 2 frames x 3 x 4 = 24
// tokens, 32 heads x 128 head_dim.
constexpr int kF = 2, kH = 3, kW = 4;
constexpr int kTokens = kF * kH * kW;
constexpr int kHeads = 32, kHeadDim = 128, kInner = kHeads * kHeadDim;

// The reference emits cos/sin as [B, H, T, half]; build_rope stores
// [heads][tokens][half]. Same order, so a flat compare is the right one.
void
test_table(const std::string& dir)
{
  std::printf("frequency table vs the reference\n");
  npy::Array want_cos = npy::load(dir + "/rope_cos.npy");
  npy::Array want_sin = npy::load(dir + "/rope_sin.npy");
  if (!want_cos.ok || !want_sin.ok) {
    check(false, "load goldens: " + want_cos.err + want_sin.err);
    return;
  }

  const ltx25::RopeTable t = ltx25::build_video_rope(
      kF, kH, kW, {20, 2048, 2048}, kInner, kHeads,
      ltx25::RopeGeometry::identity(), 10000.0);
  check(t.tokens == kTokens && t.heads == kHeads && t.half == kHeadDim / 2,
        "table geometry [" + std::to_string(t.heads) + ", " +
        std::to_string(t.tokens) + ", " + std::to_string(t.half) + "]");
  check(t.cos.size() == want_cos.count(),
        "cos element count " + std::to_string(t.cos.size()) + " vs golden " +
        std::to_string(want_cos.count()));
  if (t.cos.size() != want_cos.count()) { return; }

  const double rc = npy::rel_l2(t.cos, want_cos.data);
  const double rs = npy::rel_l2(t.sin, want_sin.data);
  const double mc = npy::max_abs_diff(t.cos, want_cos.data);
  const double ms = npy::max_abs_diff(t.sin, want_sin.data);
  std::printf("       cos rel-L2 %.3e (max abs %.3e)\n", rc, mc);
  std::printf("       sin rel-L2 %.3e (max abs %.3e)\n", rs, ms);
  // The bar is f32 round-off, not "close enough". Both sides build the
  // ladder in f64 and narrow to f32 before forming the angles, so the
  // only difference left is the last rounding. Getting the narrowing
  // POINT wrong (staying in double) reads as 1.3e-4 here, and picking
  // the f32 ladder reads as 6e-4 -- both pass any loose threshold, which
  // is exactly why this one is tight.
  check(rc < 1e-6, "cos matches to f32 round-off");
  check(rs < 1e-6, "sin matches to f32 round-off");
}

void
test_apply(const std::string& dir)
{
  std::printf("rotation vs the reference\n");
  npy::Array qin  = npy::load(dir + "/rope_q_in.npy");
  npy::Array qout = npy::load(dir + "/rope_q_out.npy");
  if (!qin.ok || !qout.ok) {
    check(false, "load goldens: " + qin.err + qout.err);
    return;
  }
  const ltx25::RopeTable t = ltx25::build_video_rope(
      kF, kH, kW, {20, 2048, 2048}, kInner, kHeads,
      ltx25::RopeGeometry::identity(), 10000.0);

  std::vector<float> x = qin.data;      // [1, 24, 4096], row-major
  const bool applied = ltx25::apply_rope(t, x.data(), kTokens, kHeads,
                                         kHeadDim);
  check(applied, "apply_rope accepted the shapes");
  if (!applied) { return; }

  const double r = npy::rel_l2(x, qout.data);
  std::printf("       q rel-L2 %.3e (max abs %.3e)\n", r,
              npy::max_abs_diff(x, qout.data));
  check(r < 1e-6, "rotated q matches to f32 round-off");

  // A rotation is norm-preserving per (j, j+half) pair. Worth its own
  // check: a sign error in one of the four terms still lands close in
  // rel-L2 on random data but breaks this exactly.
  double n_in = 0.0, n_out = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    n_in  += (double)qin.data[i] * qin.data[i];
    n_out += (double)x[i] * x[i];
  }
  check(std::fabs(n_in - n_out) / n_in < 1e-5,
        "the rotation preserves the norm");
}

// Properties that hold with no golden at all, so they run everywhere and
// pin the parts a golden of ONE geometry cannot.
void
test_properties()
{
  std::printf("properties (no golden needed)\n");

  // The cross-attention table is built at the AUDIO width on the time
  // axis only -- 32 heads x 64 head_dim -> half 32, whatever the video
  // stream's width is.
  std::vector<double> t0;
  for (int f = 0; f < kF; ++f) {
    for (int i = 0; i < kH * kW; ++i) { t0.push_back(f); }
  }
  const ltx25::RopeTable cross =
      ltx25::build_cross_rope(t0, 20, /*audio_inner_dim=*/2048, 32);
  check(cross.half == 32, "cross table half is 32 (audio head_dim 64)");
  check(cross.tokens == kTokens, "one cross row per video token");
  // Every cell of one frame shares a time position, so their rows must
  // be identical -- that is what makes this a TIME-only table.
  bool same = true;
  for (int j = 0; j < cross.half; ++j) {
    if (cross.cos[(std::size_t)j] !=
        cross.cos[(std::size_t)(kH * kW - 1) * cross.half + j]) {
      same = false;
      break;
    }
  }
  check(same, "all cells of one frame share a cross-attention position");

  // A mismatched max_pos must be REFUSED, not silently rescaled: it is a
  // per-axis divisor, so a short list changes every angle it does cover.
  std::string err;
  const ltx25::RopeTable bad = ltx25::build_rope(
      {{0.0}, {0.0}, {0.0}}, {{1.0}, {1.0}, {1.0}}, {20, 2048}, kInner,
      kHeads, 10000.0, /*f64_precision=*/true, &err);
  check(bad.tokens == 0 && !err.empty(),
        "a short max_pos is refused (" + err + ")");

  // max_pos genuinely changes the angles -- the property that makes this
  // RoPE position-EXTENT dependent rather than index dependent.
  const ltx25::RopeTable a =
      ltx25::build_video_rope(kF, kH, kW, {20, 2048, 2048}, kInner, kHeads, ltx25::RopeGeometry::identity());
  const ltx25::RopeTable b =
      ltx25::build_video_rope(kF, kH, kW, {40, 2048, 2048}, kInner, kHeads, ltx25::RopeGeometry::identity());
  check(npy::rel_l2(a.cos, b.cos) > 1e-3,
        "changing max_pos changes the table (fractional positions)");
}

}  // namespace


void
test_positions(const std::string& dir)
{
  // ---- the POSITIONS, not just the math -------------------------------
  //
  // Every other rope check in this file pins the MATH on an arange grid.
  // That is exactly how this port shipped a severe bug: it fed raw
  // latent indices where the model wants PIXELS and SECONDS, and because
  // gen_goldens.py fed precompute_freqs_cis the same arange, the tables
  // agreed at 1.8e-8 while both sides shared one wrong assumption. The
  // goldens below come from the reference's OWN pipeline path
  // (get_patch_grid_bounds -> get_pixel_coords -> /fps), so they pin the
  // coordinates themselves.
  {
    npy::Array wpos = npy::load(dir + "/ropep_positions.npy");
    npy::Array wcos = npy::load(dir + "/ropep_cos.npy");
    npy::Array acos = npy::load(dir + "/ropep_audio_cos.npy");
    if (!wpos.ok || !wcos.ok) {
      std::printf("  [SKIP] no position goldens (gen_goldens.py ropep)\n");
    } else {
      const int F = 2, H = 3, W = 4, TV = F * H * W;
      ltx25::RopeGeometry geo;      // the real one: 8 / 32 / 32 at 24 fps
      geo.scale_t = 8; geo.scale_h = 32; geo.scale_w = 32;
      geo.fps = 24.0; geo.causal_fix = true;

      // The positions themselves, in the golden's [1,3,T,2] layout.
      const std::vector<double> tsec = ltx25::video_time_seconds(F, geo);
      std::vector<float> pos((std::size_t)3 * TV * 2, 0.0f);
      std::size_t k = 0;
      for (int f = 0; f < F; ++f) {
        for (int h = 0; h < H; ++h) {
          for (int w = 0; w < W; ++w, ++k) {
            double te = (double)((f + 1) * geo.scale_t) + 1.0 -
                        (double)geo.scale_t;
            if (te < 0.0) { te = 0.0; }
            pos[k * 2 + 0] = (float)tsec[(std::size_t)f];
            pos[k * 2 + 1] = (float)(te / geo.fps);
            pos[(std::size_t)TV * 2 + k * 2 + 0] = (float)(h * geo.scale_h);
            pos[(std::size_t)TV * 2 + k * 2 + 1] =
                (float)((h + 1) * geo.scale_h);
            pos[(std::size_t)2 * TV * 2 + k * 2 + 0] = (float)(w * geo.scale_w);
            pos[(std::size_t)2 * TV * 2 + k * 2 + 1] =
                (float)((w + 1) * geo.scale_w);
          }
        }
      }
      const double rp = npy::rel_l2(pos, wpos.data);
      std::printf("       positions  rel-L2 %.3e\n", rp);
      check(rp < 1e-6, "the RoPE positions are PIXELS and SECONDS");

      const ltx25::RopeTable t = ltx25::build_video_rope(
          F, H, W, {20, 2048, 2048}, 4096, 32, geo);
      const double rc = npy::rel_l2(t.cos, wcos.data);
      std::printf("       cos table  rel-L2 %.3e\n", rc);
      check(rc < 1e-5, "the video table matches the REAL-position golden");

      // And the identity geometry must NOT match -- otherwise this test
      // would pass on the very bug it exists to catch.
      const ltx25::RopeTable bad = ltx25::build_video_rope(
          F, H, W, {20, 2048, 2048}, 4096, 32,
          ltx25::RopeGeometry::identity());
      const double rbad = npy::rel_l2(bad.cos, wcos.data);
      std::printf("       latent-index table  rel-L2 %.3e (must be BIG)\n",
                  rbad);
      check(rbad > 1e-2,
            "latent indices are visibly wrong against the real positions");

      if (acos.ok) {
        const ltx25::RopeTable at =
            ltx25::build_audio_rope(8, {20}, 2048, 32);
        const double ra = npy::rel_l2(at.cos, acos.data);
        std::printf("       audio table  rel-L2 %.3e\n", ra);
        // The POSITIONS are exact (verified separately below); the table
        // is not, and the bar says so honestly rather than hiding it.
        //
        // The 1-axis ladder runs to theta^1 * pi/2 ~ 1.6e4, and f32 has
        // roughly 1e-3 absolute resolution there, so cos/sin of an angle
        // that large is only good to ~1e-3 unless both sides do the
        // arithmetic in the same ORDER. The video table happens to match
        // at 1.8e-8, the audio one at ~1e-4 -- so the orders differ
        // somewhere in the 1-axis path. It is not chased here because
        // the audio table is unused until the joint audio-video denoise
        // exists; it is recorded in that task instead.
        check(ra < 2e-4, "the audio table is in SECONDS too (see the note "
                         "on its ~1e-4 f32 ordering gap)");
        // What IS exact: the positions themselves.
        const std::vector<double> asec = ltx25::audio_time_seconds(8);
        npy::Array wap = npy::load(dir + "/ropep_audio_positions.npy");
        if (wap.ok && wap.data.size() == 16) {
          double worst = 0.0;
          for (int i = 0; i < 8; ++i) {
            worst = std::max(worst,
                             std::fabs(asec[(std::size_t)i] -
                                       (double)wap.data[(std::size_t)i * 2]));
          }
          std::printf("       audio positions  max abs %.3e\n", worst);
          // 1e-7, not 1e-9: the golden is stored as f32, so ~0.27 s
          // carries about 1e-8 of representable resolution and a
          // tighter bar would be testing the file format.
          check(worst < 1e-7, "the audio positions are exact SECONDS");
        }
      }
    }
  }

}

int
main()
{
  test_properties();
  if (const char* dir = std::getenv("VPIPE_LTX25_GOLDENS")) {
    test_table(dir);
    test_positions(dir);
    test_apply(dir);
  } else {
    std::printf("SKIPPED the golden comparisons: set VPIPE_LTX25_GOLDENS "
                "to the directory gen_goldens.py wrote.\n"
                "Only the properties above actually ran.\n");
  }
  std::printf("%s\n", g_fail == 0 ? "ALL PASSED" : "FAILURES");
  return g_fail == 0 ? 0 : 1;
}
