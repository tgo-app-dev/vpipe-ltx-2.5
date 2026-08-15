#include "ltx25-block-ref.h"

#include <cmath>
#include <string>
#include <vector>

namespace ltx25 {

namespace {

// out[r][o] = sum_i x[r][i] * w[o][i] + b[o]
void
linear_(const Mat& x, const Mat& w, const std::vector<float>& b, Mat& out)
{
  out.rows = x.rows;
  out.cols = w.rows;
  out.v.assign((std::size_t)out.rows * out.cols, 0.0f);
  for (int r = 0; r < x.rows; ++r) {
    const float* xr = x.at(r);
    float*       o  = out.at(r);
    for (int c = 0; c < w.rows; ++c) {
      const float* wr = w.at(c);
      double acc = b.empty() ? 0.0 : (double)b[(std::size_t)c];
      for (int i = 0; i < x.cols; ++i) { acc += (double)xr[i] * wr[i]; }
      o[c] = (float)acc;
    }
  }
}

void
rms_norm_gain_(Mat& m, const std::vector<float>& gain, double eps)
{
  for (int r = 0; r < m.rows; ++r) {
    float* row = m.at(r);
    double ss = 0.0;
    for (int i = 0; i < m.cols; ++i) { ss += (double)row[i] * row[i]; }
    const double inv = 1.0 / std::sqrt(ss / m.cols + eps);
    for (int i = 0; i < m.cols; ++i) {
      row[i] = (float)(row[i] * inv *
                       (gain.empty() ? 1.0 : (double)gain[(std::size_t)i]));
    }
  }
}

// Scaled dot-product attention, dense and per head. q is [tq][H*D], k/v
// are [tk][H*D]; out is [tq][H*D].
void
sdpa_(const Mat& q, const Mat& k, const Mat& v, int heads, int head_dim,
      Mat& out)
{
  out.rows = q.rows;
  out.cols = heads * head_dim;
  out.v.assign((std::size_t)out.rows * out.cols, 0.0f);
  const double scale = 1.0 / std::sqrt((double)head_dim);
  std::vector<double> logits((std::size_t)k.rows);
  for (int h = 0; h < heads; ++h) {
    const int off = h * head_dim;
    for (int i = 0; i < q.rows; ++i) {
      const float* qi = q.at(i) + off;
      double mx = -1e300;
      for (int j = 0; j < k.rows; ++j) {
        const float* kj = k.at(j) + off;
        double s = 0.0;
        for (int d = 0; d < head_dim; ++d) { s += (double)qi[d] * kj[d]; }
        s *= scale;
        logits[(std::size_t)j] = s;
        if (s > mx) { mx = s; }
      }
      double den = 0.0;
      for (int j = 0; j < k.rows; ++j) {
        logits[(std::size_t)j] = std::exp(logits[(std::size_t)j] - mx);
        den += logits[(std::size_t)j];
      }
      float* o = out.at(i) + off;
      for (int d = 0; d < head_dim; ++d) {
        double acc = 0.0;
        for (int j = 0; j < k.rows; ++j) {
          acc += logits[(std::size_t)j] * (double)v.at(j)[off + d];
        }
        o[d] = (float)(acc / den);
      }
    }
  }
}

// One attention, whole. `pe` rotates BOTH q and k unless `k_pe` is given
// -- which is the a2v / v2a case, where the two streams sit on different
// token grids and each side carries its own table.
void
attention_(const AttnWeights& w, const Mat& x, const Mat& ctx,
           const RopeTable* pe, const RopeTable* k_pe, Mat& out)
{
  Mat q, k, v;
  linear_(x,   w.q_w, w.q_b, q);
  linear_(ctx, w.k_w, w.k_b, k);
  linear_(ctx, w.v_w, w.v_b, v);

  // q_norm / k_norm are RMSNorm over the WHOLE inner dim, before the head
  // split -- not per head. Normalising per head is a different operator
  // and lands close enough to look right.
  rms_norm_gain_(q, w.q_norm, 1e-6);
  rms_norm_gain_(k, w.k_norm, 1e-6);

  if (pe != nullptr) {
    apply_rope(*pe, q.v.data(), q.rows, w.heads, w.head_dim);
    const RopeTable& kt = (k_pe != nullptr) ? *k_pe : *pe;
    apply_rope(kt, k.v.data(), k.rows, w.heads, w.head_dim);
  }

  Mat att;
  sdpa_(q, k, v, w.heads, w.head_dim, att);

  // Per-head gating: 2 * sigmoid(x @ gate_w^T + gate_b), one scalar per
  // (token, head), applied BEFORE to_out. Note the logits come from the
  // attention's INPUT x, not from its output.
  if (w.gate_w.rows > 0) {
    Mat g;
    linear_(x, w.gate_w, w.gate_b, g);
    for (int r = 0; r < att.rows; ++r) {
      float* a = att.at(r);
      const float* gr = g.at(r);
      for (int h = 0; h < w.heads; ++h) {
        const float gate = 2.0f / (1.0f + std::exp(-gr[h]));
        for (int d = 0; d < w.head_dim; ++d) { a[h * w.head_dim + d] *= gate; }
      }
    }
  }
  linear_(att, w.o_w, w.o_b, out);
}

// Broadcast one `dim`-wide modulation row across every token.
void
mod_(const float* x, float* out, int rows, int cols, const float* scale,
     const float* shift)
{
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      const std::size_t i = (std::size_t)r * cols + c;
      out[i] = x[i] * (1.0f + scale[c]) + shift[c];
    }
  }
}

}  // namespace

void
rms_norm(float* x, int rows, int cols, double eps)
{
  for (int r = 0; r < rows; ++r) {
    float* row = x + (std::size_t)r * cols;
    double ss = 0.0;
    for (int i = 0; i < cols; ++i) { ss += (double)row[i] * row[i]; }
    const double inv = 1.0 / std::sqrt(ss / cols + eps);
    for (int i = 0; i < cols; ++i) { row[i] = (float)(row[i] * inv); }
  }
}

void
ada_zero(const float* x, float* out, int rows, int cols, double eps,
         const float* scale, const float* shift)
{
  std::vector<float> tmp(x, x + (std::size_t)rows * cols);
  rms_norm(tmp.data(), rows, cols, eps);
  mod_(tmp.data(), out, rows, cols, scale, shift);
}

std::vector<std::vector<float>>
ada_values(const std::vector<float>& table, const std::vector<float>& timesteps,
           int n_rows, int dim, int lo, int hi)
{
  // `scale_shift_table[i] + timestep.reshape(B, T, n_rows, -1)[:, :, i, :]`
  // -- the table row and the timestep slice are ADDED, and the slice
  // index is the same i. Reading the timestep with a different row count
  // silently mixes neighbouring modulations.
  std::vector<std::vector<float>> out;
  (void)n_rows;
  for (int i = lo; i < hi; ++i) {
    std::vector<float> row((std::size_t)dim);
    for (int c = 0; c < dim; ++c) {
      const std::size_t k = (std::size_t)i * dim + c;
      const float a = k < table.size() ? table[k] : 0.0f;
      const float b = k < timesteps.size() ? timesteps[k] : 0.0f;
      row[(std::size_t)c] = a + b;
    }
    out.push_back(std::move(row));
  }
  return out;
}

namespace {

// One stream's self-attention + text cross-attention, up to (not
// including) the audio<->video cross-attentions.
void
stream_first_half_(const StreamWeights& w, StreamInput& s, double eps)
{
  const int n = s.x.rows, d = w.dim;
  // rows 0..3 of the 9: shift, scale, gate for the self-attention.
  auto msa = ada_values(w.scale_shift, s.timesteps, 9, d, 0, 3);
  Mat normed;
  normed.rows = n; normed.cols = d;
  normed.v.assign((std::size_t)n * d, 0.0f);
  ada_zero(s.x.v.data(), normed.v.data(), n, d, eps, msa[1].data(),
           msa[0].data());

  Mat sa;
  attention_(w.attn1, normed, normed, s.pe, nullptr, sa);
  // post_sa: x = x + out*gate, THEN x_normed = rms_norm(x). The second
  // normalisation feeds the cross-attention and is NOT re-derived from
  // the pre-residual x.
  for (std::size_t i = 0; i < s.x.v.size(); ++i) {
    s.x.v[i] += sa.v[i] * msa[2][(std::size_t)(i % (std::size_t)d)];
  }
  Mat xn = s.x;
  rms_norm(xn.v.data(), n, d, eps);

  // Text cross-attention with adaLN: rows 6..9 are (shift_q, scale_q,
  // gate) for the QUERY side; prompt_scale_shift_table is (shift, scale)
  // for the KEY/VALUE side. Modulating only the query is the easy
  // mistake and costs prompt adherence, not correctness of shape.
  auto ca = ada_values(w.scale_shift, s.timesteps, 9, d, 6, 9);
  Mat qin;
  qin.rows = n; qin.cols = d;
  qin.v.assign((std::size_t)n * d, 0.0f);
  mod_(xn.v.data(), qin.v.data(), n, d, ca[1].data(), ca[0].data());

  Mat kv = s.context;
  {
    // shift = table[0] (+ prompt_ts[0]), scale = table[1] (+ prompt_ts[1]).
    std::vector<float> shift(w.prompt_scale_shift.begin(),
                             w.prompt_scale_shift.begin() + d);
    std::vector<float> scale(w.prompt_scale_shift.begin() + d,
                             w.prompt_scale_shift.begin() + 2 * d);
    if ((int)s.prompt_timestep.size() >= 2 * d) {
      for (int i = 0; i < d; ++i) {
        shift[(std::size_t)i] += s.prompt_timestep[(std::size_t)i];
        scale[(std::size_t)i] += s.prompt_timestep[(std::size_t)(d + i)];
      }
    }
    mod_(s.context.v.data(), kv.v.data(), kv.rows, d, scale.data(),
         shift.data());
  }
  Mat co;
  attention_(w.attn2, qin, kv, nullptr, nullptr, co);
  for (int r = 0; r < n; ++r) {
    for (int c = 0; c < d; ++c) {
      s.x.at(r)[c] += co.at(r)[c] * ca[2][(std::size_t)c];
    }
  }
}

// The feed-forward tail: rows 3..6 of the 9.
void
stream_ff_(const StreamWeights& w, StreamInput& s, double eps)
{
  const int n = s.x.rows, d = w.dim;
  auto mlp = ada_values(w.scale_shift, s.timesteps, 9, d, 3, 6);
  Mat scaled;
  scaled.rows = n; scaled.cols = d;
  scaled.v.assign((std::size_t)n * d, 0.0f);
  ada_zero(s.x.v.data(), scaled.v.data(), n, d, eps, mlp[1].data(),
           mlp[0].data());
  Mat h, o;
  linear_(scaled, w.ff_in, w.ff_in_b, h);
  // GELU-approximate (tanh), which is what `activation_fn:
  // "gelu-approximate"` means -- not the erf form. The two differ by
  // ~1e-3 at the tails, which over 48 blocks is not nothing.
  for (auto& z : h.v) {
    const double x = z;
    const double t = 0.7978845608028654 * (x + 0.044715 * x * x * x);
    z = (float)(0.5 * x * (1.0 + std::tanh(t)));
  }
  linear_(h, w.ff_out, w.ff_out_b, o);
  for (int r = 0; r < n; ++r) {
    for (int c = 0; c < d; ++c) {
      s.x.at(r)[c] += o.at(r)[c] * mlp[2][(std::size_t)c];
    }
  }
}

// The a2v / v2a scale-shift+gate triple. The table is [5, dim]: rows
// 0..4 are the four scale/shift values (a2v pair then v2a pair) and row
// 4 is the gate -- and the two halves are driven by DIFFERENT timesteps
// (`cross_scale_shift` vs `cross_gate`).
void
av_ada_(const std::vector<float>& table, const std::vector<float>& css,
        const std::vector<float>& cg, int dim, int lo,
        std::vector<float>& scale, std::vector<float>& shift,
        std::vector<float>& gate)
{
  auto ss = ada_values(table, css, 4, dim, lo, lo + 2);
  scale = ss[0];
  shift = ss[1];
  // The gate is row 4 of the table against the GATE timestep, which is
  // one dim-wide row rather than four.
  gate.assign((std::size_t)dim, 0.0f);
  for (int c = 0; c < dim; ++c) {
    const std::size_t k = (std::size_t)4 * dim + c;
    gate[(std::size_t)c] = (k < table.size() ? table[k] : 0.0f) +
                           ((std::size_t)c < cg.size() ? cg[(std::size_t)c]
                                                       : 0.0f);
  }
}

}  // namespace

bool
block_forward(const BlockWeights& w, StreamInput& video, StreamInput& audio,
              std::string* err)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  const bool rv = video.present && video.x.rows > 0;
  const bool ra = audio.present && audio.x.rows > 0;
  if (!rv && !ra) { return fail("neither stream is present"); }
  if (rv && video.x.cols != w.video.dim) {
    return fail("video x is " + std::to_string(video.x.cols) + " wide, want " +
                std::to_string(w.video.dim));
  }
  if (ra && audio.x.cols != w.audio.dim) {
    return fail("audio x is " + std::to_string(audio.x.cols) + " wide, want " +
                std::to_string(w.audio.dim));
  }

  if (rv) { stream_first_half_(w.video, video, w.norm_eps); }
  if (ra) { stream_first_half_(w.audio, audio, w.norm_eps); }

  // Audio <-> video cross-attention.
  //
  // BOTH DIRECTIONS READ THE PRE-CROSS SNAPSHOT. a2v updates the video
  // stream, and if v2a then read that updated video as its keys, the
  // result would depend on which direction ran first -- so the reference
  // snapshots both and each direction attends to the other's snapshot.
  // Nothing in the shapes catches getting this wrong; it just quietly
  // couples the modalities in one direction more than the other.
  if (rv && ra) {
    const Mat vx_pre = video.x;
    const Mat ax_pre = audio.x;
    const int vd = w.video.dim, ad = w.audio.dim;

    {   // a2v: video queries, audio keys/values
      std::vector<float> vsc, vsh, vg, asc, ash, ag;
      av_ada_(w.video.cross_table, video.cross_scale_shift, video.cross_gate,
              vd, 0, vsc, vsh, vg);
      av_ada_(w.audio.cross_table, audio.cross_scale_shift, audio.cross_gate,
              ad, 0, asc, ash, ag);
      // ada_zero, i.e. rms_norm THEN modulate -- not a bare affine. The
      // pre-cross snapshot is what gets normalised, not the running x.
      Mat qv = vx_pre, kv = ax_pre;
      ada_zero(vx_pre.v.data(), qv.v.data(), vx_pre.rows, vd, w.norm_eps,
               vsc.data(), vsh.data());
      ada_zero(ax_pre.v.data(), kv.v.data(), ax_pre.rows, ad, w.norm_eps,
               asc.data(), ash.data());
      Mat o;
      attention_(w.a2v, qv, kv, video.cross_pe, audio.cross_pe, o);
      for (int r = 0; r < video.x.rows; ++r) {
        for (int c = 0; c < vd; ++c) {
          video.x.at(r)[c] += o.at(r)[c] * vg[(std::size_t)c];
        }
      }
    }
    {   // v2a: audio queries, video keys/values -- table rows 2..4
      std::vector<float> asc, ash, ag, vsc, vsh, vg;
      av_ada_(w.audio.cross_table, audio.cross_scale_shift, audio.cross_gate,
              ad, 2, asc, ash, ag);
      av_ada_(w.video.cross_table, video.cross_scale_shift, video.cross_gate,
              vd, 2, vsc, vsh, vg);
      Mat qa = ax_pre, kvv = vx_pre;
      ada_zero(ax_pre.v.data(), qa.v.data(), ax_pre.rows, ad, w.norm_eps,
               asc.data(), ash.data());
      ada_zero(vx_pre.v.data(), kvv.v.data(), vx_pre.rows, vd, w.norm_eps,
               vsc.data(), vsh.data());
      Mat o;
      attention_(w.v2a, qa, kvv, audio.cross_pe, video.cross_pe, o);
      for (int r = 0; r < audio.x.rows; ++r) {
        for (int c = 0; c < ad; ++c) {
          audio.x.at(r)[c] += o.at(r)[c] * ag[(std::size_t)c];
        }
      }
    }
  }

  if (rv) { stream_ff_(w.video, video, w.norm_eps); }
  if (ra) { stream_ff_(w.audio, audio, w.norm_eps); }
  return true;
}

namespace {

const NamedTensor*
find_(const std::unordered_map<std::string, NamedTensor>& t,
      const std::string& n)
{
  auto it = t.find(n);
  return it == t.end() ? nullptr : &it->second;
}

bool
take_mat_(const std::unordered_map<std::string, NamedTensor>& t,
          const std::string& n, Mat& out, std::string* miss)
{
  const NamedTensor* p = find_(t, n);
  if (p == nullptr || p->shape.size() != 2) {
    if (miss != nullptr) { *miss = n; }
    return false;
  }
  out.rows = p->shape[0];
  out.cols = p->shape[1];
  out.v = p->data;
  return true;
}

bool
take_vec_(const std::unordered_map<std::string, NamedTensor>& t,
          const std::string& n, std::vector<float>& out, std::string* miss,
          bool required = true)
{
  const NamedTensor* p = find_(t, n);
  if (p == nullptr) {
    if (required) {
      if (miss != nullptr) { *miss = n; }
      return false;
    }
    out.clear();
    return true;
  }
  out = p->data;
  return true;
}

bool
load_attn_(const std::unordered_map<std::string, NamedTensor>& t,
           const std::string& p, int heads, int head_dim, AttnWeights& a,
           std::string* miss)
{
  a.heads = heads;
  a.head_dim = head_dim;
  if (!take_mat_(t, p + ".to_q.weight", a.q_w, miss) ||
      !take_mat_(t, p + ".to_k.weight", a.k_w, miss) ||
      !take_mat_(t, p + ".to_v.weight", a.v_w, miss) ||
      // to_out is a Sequential, so the linear is index 0.
      !take_mat_(t, p + ".to_out.0.weight", a.o_w, miss)) {
    return false;
  }
  if (!take_vec_(t, p + ".to_q.bias", a.q_b, miss) ||
      !take_vec_(t, p + ".to_k.bias", a.k_b, miss) ||
      !take_vec_(t, p + ".to_v.bias", a.v_b, miss) ||
      !take_vec_(t, p + ".to_out.0.bias", a.o_b, miss) ||
      !take_vec_(t, p + ".q_norm.weight", a.q_norm, miss) ||
      !take_vec_(t, p + ".k_norm.weight", a.k_norm, miss)) {
    return false;
  }
  // Gating is optional at the type level but present throughout LTX-2.5.
  std::string ignored;
  if (!take_mat_(t, p + ".to_gate_logits.weight", a.gate_w, &ignored)) {
    a.gate_w = Mat{};
  } else {
    take_vec_(t, p + ".to_gate_logits.bias", a.gate_b, miss, false);
  }
  return true;
}

bool
load_stream_(const std::unordered_map<std::string, NamedTensor>& t,
             bool is_audio, int dim, int heads, StreamWeights& s,
             std::string* miss)
{
  const std::string pre = is_audio ? "audio_" : "";
  s.dim = dim;
  const int head_dim = dim / heads;
  if (!load_attn_(t, pre + "attn1", heads, head_dim, s.attn1, miss) ||
      !load_attn_(t, pre + "attn2", heads, head_dim, s.attn2, miss)) {
    return false;
  }
  // FeedForward is `net.0.proj` (in) then `net.2` (out) -- index 1 is the
  // activation and index 3 the dropout, so neither carries weights.
  if (!take_mat_(t, pre + "ff.net.0.proj.weight", s.ff_in, miss) ||
      !take_mat_(t, pre + "ff.net.2.weight", s.ff_out, miss)) {
    return false;
  }
  std::string ig;
  take_vec_(t, pre + "ff.net.0.proj.bias", s.ff_in_b, &ig, false);
  take_vec_(t, pre + "ff.net.2.bias", s.ff_out_b, &ig, false);
  if (!take_vec_(t, pre + "scale_shift_table", s.scale_shift, miss) ||
      !take_vec_(t, pre + "prompt_scale_shift_table", s.prompt_scale_shift,
                 miss)) {
    return false;
  }
  const std::string ct = is_audio ? "scale_shift_table_a2v_ca_audio"
                                  : "scale_shift_table_a2v_ca_video";
  return take_vec_(t, ct, s.cross_table, miss);
}

}  // namespace

bool
load_block_weights(const std::unordered_map<std::string, NamedTensor>& t,
                   int vdim, int vheads, int adim, int aheads,
                   BlockWeights& out, std::string* err)
{
  std::string miss;
  auto fail = [&]() {
    if (err != nullptr) { *err = "missing tensor '" + miss + "'"; }
    return false;
  };
  if (!load_stream_(t, false, vdim, vheads, out.video, &miss)) { return fail(); }
  if (!load_stream_(t, true, adim, aheads, out.audio, &miss)) { return fail(); }
  // BOTH cross-attentions use the AUDIO head count and head dim -- the
  // video query is projected down to the audio head size. Using the
  // video head_dim here is a shape error on the real model and, on a
  // config where the two happen to match, silently wrong attention.
  const int a_head = adim / aheads;
  if (!load_attn_(t, "audio_to_video_attn", aheads, a_head, out.a2v, &miss)) {
    return fail();
  }
  if (!load_attn_(t, "video_to_audio_attn", aheads, a_head, out.v2a, &miss)) {
    return fail();
  }
  return true;
}

}  // namespace ltx25
