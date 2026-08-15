#include "ltx25-audio-encoder.h"

#include "apple-silicon/metal-compute/command-stream.h"

#include <algorithm>
#include <cmath>
#include <cstring>

using vpipe::genai::WeightSet;
using vpipe::metal_compute::CommandStream;
using vpipe::metal_compute::SharedBuffer;

namespace ltx25 {

namespace {

// ---- the SLANEY mel scale, as librosa and torchaudio define it -------
//
// Linear below 1 kHz at 200/3 Hz per mel, logarithmic above. NOT the HTK
// formula (2595 * log10(1 + f/700)), which is the other common one and
// moves every filter centre.
constexpr double kFSp        = 200.0 / 3.0;
constexpr double kMinLogHz   = 1000.0;
constexpr double kMinLogMel  = kMinLogHz / kFSp;             // 15
const     double kLogStep    = std::log(6.4) / 27.0;

double
hz_to_mel_(double f)
{
  if (f < kMinLogHz) { return f / kFSp; }
  return kMinLogMel + std::log(f / kMinLogHz) / kLogStep;
}

double
mel_to_hz_(double m)
{
  if (m < kMinLogMel) { return m * kFSp; }
  return kMinLogHz * std::exp(kLogStep * (m - kMinLogMel));
}

}  // namespace

Ltx25AudioMel::Ltx25AudioMel(const Config& cfg) : _cfg(cfg)
{
  // torch.hann_window is PERIODIC by default: 0.5 - 0.5 cos(2 pi n / N)
  // over N points, not the symmetric N-1 denominator. The two differ by
  // one sample's worth of taper, which is small and systematic --
  // exactly the kind of difference a spectrogram hides.
  _window.resize((std::size_t)_cfg.win_length);
  for (int i = 0; i < _cfg.win_length; ++i) {
    _window[(std::size_t)i] =
        (float)(0.5 - 0.5 * std::cos(2.0 * M_PI * i / _cfg.win_length));
  }

  // The triangular filter bank over the FFT bin centres.
  const int nf = n_freqs();
  const int nm = _cfg.mel_bins;
  std::vector<double> all_freqs((std::size_t)nf);
  for (int i = 0; i < nf; ++i) {
    all_freqs[(std::size_t)i] =
        (double)i * _cfg.sample_rate / 2.0 / (nf - 1);
  }
  std::vector<double> f_pts((std::size_t)nm + 2);
  {
    const double m_min = hz_to_mel_(_cfg.f_min);
    const double m_max = hz_to_mel_(_cfg.f_max);
    for (int i = 0; i < nm + 2; ++i) {
      f_pts[(std::size_t)i] =
          mel_to_hz_(m_min + (m_max - m_min) * i / (nm + 1));
    }
  }
  _fb.assign((std::size_t)nm * nf, 0.0f);
  for (int m = 0; m < nm; ++m) {
    const double lo = f_pts[(std::size_t)m];
    const double ct = f_pts[(std::size_t)m + 1];
    const double hi = f_pts[(std::size_t)m + 2];
    // SLANEY normalisation: each filter is scaled so its area is
    // constant, 2 / (hi - lo). Without it the low filters -- which are
    // narrow -- come out weaker than the model was trained on.
    const double enorm = 2.0 / (hi - lo);
    for (int k = 0; k < nf; ++k) {
      const double f = all_freqs[(std::size_t)k];
      const double down = (f - lo) / (ct - lo);
      const double up   = (hi - f) / (hi - ct);
      const double v = std::min(down, up);
      if (v > 0.0) {
        _fb[(std::size_t)m * nf + k] = (float)(v * enorm);
      }
    }
  }
}

int
Ltx25AudioMel::frames_for(int n_samples) const
{
  if (n_samples <= 0 || _cfg.hop_length <= 0) { return 0; }
  return 1 + n_samples / _cfg.hop_length;
}

void
Ltx25AudioMel::compute(const float* pcm, int channels, int n_samples,
                       std::vector<float>* out,
                       std::array<int, 3>* shape) const
{
  if (out == nullptr || pcm == nullptr || channels <= 0 || n_samples <= 0) {
    return;
  }
  const int nf = n_freqs();
  const int nm = _cfg.mel_bins;
  const int frames = frames_for(n_samples);
  const int pad = _cfg.n_fft / 2;
  out->assign((std::size_t)channels * frames * nm, 0.0f);
  if (shape != nullptr) { *shape = {channels, frames, nm}; }

  // REFLECT padding, `pad` on each side -- centre=True. A reflect pad
  // mirrors WITHOUT repeating the edge sample (numpy's 'reflect', not
  // 'symmetric'), which is what torch does.
  std::vector<float> padded((std::size_t)n_samples + 2 * pad);
  std::vector<double> re((std::size_t)nf), im((std::size_t)nf);
  std::vector<double> mag((std::size_t)nf);
  for (int c = 0; c < channels; ++c) {
    const float* src = pcm + (std::size_t)c * n_samples;
    for (int i = 0; i < (int)padded.size(); ++i) {
      int j = i - pad;
      // Fold repeatedly: a clip shorter than the pad reflects more than
      // once, which torch also does.
      while (j < 0 || j >= n_samples) {
        if (j < 0) { j = -j; }
        if (j >= n_samples) { j = 2 * (n_samples - 1) - j; }
        if (n_samples == 1) { j = 0; break; }
      }
      padded[(std::size_t)i] = src[(std::size_t)j];
    }
    for (int t = 0; t < frames; ++t) {
      const float* w = padded.data() + (std::size_t)t * _cfg.hop_length;
      // A direct DFT. O(frames * n_freqs * win) and small at these
      // sizes -- ~0.3 GFLOP for a 5 s clip -- against a radix-2 FFT's
      // extra surface for an indexing bug in a place a golden would
      // only see through the mel bank.
      for (int k = 0; k < nf; ++k) {
        double sr = 0.0, si = 0.0;
        const double f = -2.0 * M_PI * k / _cfg.n_fft;
        for (int nsmp = 0; nsmp < _cfg.win_length; ++nsmp) {
          const double v = (double)w[(std::size_t)nsmp] *
                           _window[(std::size_t)nsmp];
          const double a = f * nsmp;
          sr += v * std::cos(a);
          si += v * std::sin(a);
        }
        re[(std::size_t)k] = sr;
        im[(std::size_t)k] = si;
        // POWER 1: the magnitude, not its square.
        mag[(std::size_t)k] = std::sqrt(sr * sr + si * si);
      }
      float* dst = out->data() +
                   ((std::size_t)c * frames + t) * (std::size_t)nm;
      for (int m = 0; m < nm; ++m) {
        double acc = 0.0;
        const float* fb = _fb.data() + (std::size_t)m * nf;
        for (int k = 0; k < nf; ++k) { acc += fb[k] * mag[(std::size_t)k]; }
        dst[m] = (float)std::log(std::max(acc, 1.0e-5));
      }
    }
  }
}

// ---------------------------------------------------------------------
// The encoder.
// ---------------------------------------------------------------------

namespace {

bool
shape_of_(WeightSet& ws, const std::string& name, std::vector<int>& out)
{
  const auto* info = ws.src().info(name);
  if (info == nullptr) { return false; }
  out.clear();
  for (std::int64_t d : info->shape) { out.push_back((int)d); }
  return true;
}

}  // namespace

bool
Ltx25AudioVaeEncoder::load_conv_(WeightSet& ws, const std::string& prefix,
                                 Conv& out, bool optional, std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  std::vector<int> shape;
  const std::string wn = prefix + ".conv.weight";
  if (!shape_of_(ws, wn, shape)) {
    if (optional) { out.cout = 0; return true; }
    return fail("missing tensor " + wn);
  }
  if (shape.size() != 4 || shape[2] != shape[3] ||
      (shape[2] != 3 && shape[2] != 1)) {
    return fail(wn + " is not a 3x3 or 1x1 kernel");
  }
  out.cout = shape[0];
  out.cin  = shape[1];
  out.k    = shape[2];
  out.w = ws.tensor(wn, _ops->mc(), WeightSet::Residency::Copied);
  out.b = ws.tensor(prefix + ".conv.bias", _ops->mc(),
                    WeightSet::Residency::Copied);
  if (out.w.empty() || out.b.empty()) {
    return fail("could not bind " + prefix);
  }
  _resident += out.w.byte_size() + out.b.byte_size();
  return true;
}

bool
Ltx25AudioVaeEncoder::load_res_(WeightSet& ws, const std::string& prefix,
                                ResBlock& out, std::string* err)
{
  return load_conv_(ws, prefix + ".conv1", out.conv1, false, err) &&
         load_conv_(ws, prefix + ".conv2", out.conv2, false, err) &&
         load_conv_(ws, prefix + ".nin_shortcut", out.nin, true, err);
}

std::unique_ptr<Ltx25AudioVaeEncoder>
Ltx25AudioVaeEncoder::load(const AudioVaeConfig& cfg,
                           std::shared_ptr<WeightSet> ws, const MetalOps& ops,
                           std::string* err)
{
  std::unique_ptr<Ltx25AudioVaeEncoder> m(new Ltx25AudioVaeEncoder());
  m->_cfg = cfg;
  m->_ws  = std::move(ws);
  m->_ops = &ops;
  // The checkpoint nests everything under `audio_vae.`, the same way the
  // decoder's loader does; the statistics sit beside it rather than
  // under it.
  const std::string P = "audio_vae.encoder.";
  if (!m->load_conv_(*m->_ws, P + "conv_in", m->_conv_in, false, err) ||
      !m->load_conv_(*m->_ws, P + "conv_out", m->_conv_out, false, err)) {
    return nullptr;
  }
  // The levels run LOW to HIGH here -- `down.0` is the full-resolution
  // one. The decoder walks `up` in the opposite order; reading either
  // list the other way binds channel counts that happen to line up for
  // the first block and then fail, which reads as a missing tensor.
  const int levels = cfg.num_levels();
  for (int i = 0; i < levels; ++i) {
    const std::string p = P + "down." + std::to_string(i);
    Level lv;
    for (int b = 0;; ++b) {
      const std::string rp = p + ".block." + std::to_string(b);
      if (m->_ws->src().info(rp + ".conv1.conv.weight") == nullptr) { break; }
      ResBlock rb;
      if (!m->load_res_(*m->_ws, rp, rb, err)) { return nullptr; }
      lv.res.push_back(std::move(rb));
    }
    if (lv.res.empty()) {
      if (err != nullptr) { *err = p + " has no blocks"; }
      return nullptr;
    }
    // Every level but the LAST downsamples; the reference builds it that
    // way, so an absent `downsample` on the last level is structure, not
    // a missing tensor.
    if (!m->load_conv_(*m->_ws, p + ".downsample", lv.down,
                       /*optional=*/i == levels - 1, err)) {
      return nullptr;
    }
    m->_levels.push_back(std::move(lv));
  }
  if (!m->load_res_(*m->_ws, P + "mid.block_1", m->_mid1, err) ||
      !m->load_res_(*m->_ws, P + "mid.block_2", m->_mid2, err)) {
    return nullptr;
  }
  // `mid_block_add_attention` is false in this checkpoint and there are
  // no `mid.attn_1` tensors. A checkpoint that HAS them would need an
  // attention block this port does not implement, so refuse rather than
  // silently skip it.
  if (m->_ws->src().info(P + "mid.attn_1.q.conv.weight") != nullptr ||
      m->_ws->src().info(P + "mid.attn_1.q.weight") != nullptr) {
    if (err != nullptr) {
      *err = "this audio VAE has a mid attention block, which is not ported";
    }
    return nullptr;
  }

  // The per-(channel, mel-bin) statistics. NOT optional here, unlike the
  // video VAE's: an unwhitened audio latent is silently in a different
  // space from the one the DiT conditions in.
  const int slen = cfg.stats_len();
  std::vector<float> sv((std::size_t)slen, 1.0f), mv((std::size_t)slen, 0.0f);
  bool have = false;
  auto load_stat = [&](const char* name, std::vector<float>& dst) {
    const auto* info = m->_ws->src().info(name);
    if (info == nullptr) { return; }
    const SharedBuffer b =
        m->_ws->read(name, ops.mc(), WeightSet::Residency::Copied);
    if (b.empty() || info->shape.size() != 1 || (int)info->shape[0] != slen) {
      return;
    }
    if (info->dtype == "F32") {
      std::memcpy(dst.data(), b.contents(), (std::size_t)slen * 4);
      have = true;
    } else if (info->dtype == "BF16") {
      const auto* p = static_cast<const std::uint16_t*>(b.contents());
      for (int i = 0; i < slen; ++i) {
        const std::uint32_t w = (std::uint32_t)p[i] << 16;
        std::memcpy(&dst[(std::size_t)i], &w, 4);
      }
      have = true;
    }
  };
  load_stat("audio_vae.per_channel_statistics.std-of-means", sv);
  load_stat("audio_vae.per_channel_statistics.mean-of-means", mv);
  if (!have) {
    if (err != nullptr) {
      *err = "this audio VAE carries no per_channel_statistics; an "
             "unwhitened latent is silently in the wrong space";
    }
    return nullptr;
  }
  m->_std_of_means  = ops.upload_f32(sv);
  m->_mean_of_means = ops.upload_f32(mv);
  return m;
}

void
Ltx25AudioVaeEncoder::conv_(CommandStream& stream, const Conv& c,
                            const SharedBuffer& x, int H, int W,
                            const SharedBuffer& y)
{
  const int cells = H * W;
  if (c.k == 1) {
    auto enc = stream.begin_compute();
    _ops->linear(enc, x, c.w, &c.b, y, cells, c.cin, c.cout);
    enc.end();
    return;
  }
  const std::size_t k = (std::size_t)c.cin * 9;
  int chunk = cells;
  if (k > 0) {
    const std::size_t max_rows = std::max<std::size_t>(1, _max_im2col / k);
    chunk = (int)std::min<std::size_t>((std::size_t)cells, max_rows);
  }
  const std::size_t need = (std::size_t)chunk * k;
  if (_im2col.empty() || _im2col.byte_size() < need * 2) {
    _im2col = _ops->alloc(need);
  }
  for (int c0 = 0; c0 < cells; c0 += chunk) {
    const int n = std::min(chunk, cells - c0);
    auto enc = stream.begin_compute();
    _ops->audio_im2col(enc, x, _im2col, c.cin, H, W, c0, n);
    _ops->linear(enc, _im2col, c.w, &c.b,
                 y.subview((std::size_t)c0 * (std::size_t)c.cout * 2,
                           (std::size_t)n * (std::size_t)c.cout * 2),
                 n, (int)k, c.cout);
    enc.end();
  }
}

void
Ltx25AudioVaeEncoder::res_(CommandStream& stream, const ResBlock& rb,
                           SharedBuffer& x, int& cc, int H, int W)
{
  const std::size_t cells = (std::size_t)H * W;
  SharedBuffer h = _ops->alloc(cells * (std::size_t)cc);
  {
    auto enc = stream.begin_compute();
    _ops->copy(enc, x, h, (int)(cells * (std::size_t)cc));
    _ops->vae_pixel_norm_silu(enc, h, cc, (int)cells);
    enc.end();
  }
  SharedBuffer t = _ops->alloc(cells * (std::size_t)rb.conv1.cout);
  conv_(stream, rb.conv1, h, H, W, t);
  {
    auto enc = stream.begin_compute();
    _ops->vae_pixel_norm_silu(enc, t, rb.conv1.cout, (int)cells);
    enc.end();
  }
  SharedBuffer t2 = _ops->alloc(cells * (std::size_t)rb.conv2.cout);
  conv_(stream, rb.conv2, t, H, W, t2);

  // The shortcut runs on the ORIGINAL x, not on the normalised copy.
  if (rb.nin.cout != 0) {
    SharedBuffer s = _ops->alloc(cells * (std::size_t)rb.nin.cout);
    conv_(stream, rb.nin, x, H, W, s);
    x = std::move(s);
  }
  {
    auto enc = stream.begin_compute();
    _ops->vae_add_into(enc, x, t2, cells * (std::size_t)rb.conv2.cout);
    enc.end();
  }
  cc = rb.conv2.cout;
}

void
Ltx25AudioVaeEncoder::grab_(CommandStream& stream, const std::string& name,
                            const SharedBuffer& x, int C, int H, int W)
{
  if (!_capture) { return; }
  SharedBuffer buf =
      _ops->mc()->make_shared_buffer((std::size_t)C * H * W * 4);
  if (buf.empty()) { return; }
  {
    auto enc = stream.begin_compute();
    _ops->audio_out(enc, x, buf, C, H, W);
    enc.end();
  }
  stream.commit().wait();
  stream = _ops->mc()->make_command_stream();
  std::vector<float> host((std::size_t)C * H * W);
  std::memcpy(host.data(), buf.contents(), host.size() * 4);
  _taps.emplace_back(name, std::move(host));
  _tap_shapes.emplace_back(name, std::array<int, 3>{C, H, W});
}

const std::vector<float>*
Ltx25AudioVaeEncoder::tap(const std::string& name) const
{
  for (const auto& t : _taps) {
    if (t.first == name) { return &t.second; }
  }
  return nullptr;
}

bool
Ltx25AudioVaeEncoder::tap_shape(const std::string& name,
                                std::array<int, 3>* shape) const
{
  for (const auto& t : _tap_shapes) {
    if (t.first == name) {
      if (shape != nullptr) { *shape = t.second; }
      return true;
    }
  }
  return false;
}

bool
Ltx25AudioVaeEncoder::encode(const float* mel, int frames,
                             std::vector<float>* rows,
                             std::array<int, 2>* shape, std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  if (mel == nullptr || rows == nullptr) { return fail("null argument"); }
  if (frames <= 0) { return fail("no mel frames"); }
  _taps.clear();
  _tap_shapes.clear();

  const int Cin = _conv_in.cin;              // 2, stereo
  int ch = frames, cw = _cfg.mel_bins;
  const std::size_t n_in = (std::size_t)Cin * ch * cw;
  std::vector<float> host(n_in);
  std::memcpy(host.data(), mel, n_in * sizeof(float));
  const SharedBuffer d_mel = _ops->upload_f32(host);

  CommandStream stream = _ops->mc()->make_command_stream();
  SharedBuffer x = _ops->alloc((std::size_t)ch * cw * Cin);
  {
    auto enc = stream.begin_compute();
    _ops->audio_mel_in(enc, d_mel, x, Cin, ch, cw);
    enc.end();
  }
  SharedBuffer y = _ops->alloc((std::size_t)ch * cw * _conv_in.cout);
  conv_(stream, _conv_in, x, ch, cw, y);
  x = std::move(y);
  int cc = _conv_in.cout;
  grab_(stream, "conv_in", x, cc, ch, cw);

  for (std::size_t i = 0; i < _levels.size(); ++i) {
    const Level& lv = _levels[i];
    for (const ResBlock& rb : lv.res) {
      res_(stream, rb, x, cc, ch, cw);
    }
    if (lv.down.cout != 0) {
      // The stride-2 convolution, whose padding is its own -- (0, 1, 2,
      // 0), so the output halves BOTH axes and stays causal in time.
      const int oh = (ch - 1) / 2 + 1;
      const int ow = (cw - 2) / 2 + 1;
      SharedBuffer o = _ops->alloc((std::size_t)oh * ow * lv.down.cout);
      const int cells = oh * ow;
      const std::size_t k = (std::size_t)lv.down.cin * 9;
      int chunk = cells;
      if (k > 0) {
        const std::size_t max_rows =
            std::max<std::size_t>(1, _max_im2col / k);
        chunk = (int)std::min<std::size_t>((std::size_t)cells, max_rows);
      }
      const std::size_t need = (std::size_t)chunk * k;
      if (_im2col.empty() || _im2col.byte_size() < need * 2) {
        _im2col = _ops->alloc(need);
      }
      for (int c0 = 0; c0 < cells; c0 += chunk) {
        const int n = std::min(chunk, cells - c0);
        auto enc = stream.begin_compute();
        _ops->audio_im2col_down(enc, x, _im2col, lv.down.cin, ch, cw, ow, c0,
                                n);
        _ops->linear(enc, _im2col, lv.down.w, &lv.down.b,
                     o.subview((std::size_t)c0 * (std::size_t)lv.down.cout * 2,
                               (std::size_t)n * (std::size_t)lv.down.cout * 2),
                     n, (int)k, lv.down.cout);
        enc.end();
      }
      x = std::move(o);
      cc = lv.down.cout;
      ch = oh; cw = ow;
    }
    grab_(stream, "down" + std::to_string(i), x, cc, ch, cw);
  }

  res_(stream, _mid1, x, cc, ch, cw);
  res_(stream, _mid2, x, cc, ch, cw);
  grab_(stream, "mid", x, cc, ch, cw);

  {
    auto enc = stream.begin_compute();
    _ops->vae_pixel_norm_silu(enc, x, cc, ch * cw);
    enc.end();
  }
  SharedBuffer head = _ops->alloc((std::size_t)ch * cw * _conv_out.cout);
  conv_(stream, _conv_out, x, ch, cw, head);
  grab_(stream, "head", head, _conv_out.cout, ch, cw);

  const int Z = _cfg.z_channels;
  if (_conv_out.cout < Z) {
    return fail("the head emits " + std::to_string(_conv_out.cout) +
                " channels, fewer than the " + std::to_string(Z) +
                " latent channels");
  }
  if (cw != _cfg.latent_mel_bins()) {
    return fail("the encoder produced " + std::to_string(cw) +
                " latent mel bins where the config says " +
                std::to_string(_cfg.latent_mel_bins()));
  }
  const int dim = Z * cw;
  const std::size_t nrows = (std::size_t)ch * dim;
  SharedBuffer out = _ops->mc()->make_shared_buffer(nrows * sizeof(float));
  if (out.empty()) { return fail("could not allocate the output rows"); }
  {
    auto enc = stream.begin_compute();
    _ops->audio_whiten_rows(enc, head, _std_of_means, _mean_of_means, out,
                            _conv_out.cout, Z, ch, cw);
    enc.end();
  }
  stream.commit().wait();

  rows->resize(nrows);
  std::memcpy(rows->data(), out.contents(), nrows * sizeof(float));
  if (shape != nullptr) { *shape = {ch, dim}; }
  return true;
}

}  // namespace ltx25
