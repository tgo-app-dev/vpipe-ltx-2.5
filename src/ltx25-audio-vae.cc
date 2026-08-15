#include "ltx25-audio-vae.h"

#include "apple-silicon/metal-compute/command-stream.h"

#include <algorithm>
#include <cstring>

using vpipe::FlexData;
using vpipe::genai::WeightSet;
using vpipe::metal_compute::CommandStream;
using vpipe::metal_compute::SharedBuffer;

namespace ltx25 {

int
AudioVaeConfig::latent_mel_bins() const
{
  int m = mel_bins;
  for (int i = 1; i < num_levels(); ++i) { m /= 2; }
  return std::max(1, m);
}

bool
parse_audio_vae_config(const FlexData& cfg, AudioVaeConfig& out,
                       std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  if (!cfg.is_object()) { return fail("the audio VAE config is not an object"); }
  // config -> audio_vae -> model -> params -> ddconfig, and any of the
  // wrappers may already have been peeled by the caller.
  FlexData cur = cfg;
  for (const char* k : {"audio_vae", "model", "params", "ddconfig"}) {
    if (!cur.is_object()) { break; }
    const auto o = cur.as_object();
    if (o.contains(k)) { cur = o.at(k); }
  }
  if (!cur.is_object()) { return fail("no ddconfig in the audio VAE config"); }
  const auto dd = cur.as_object();
  auto geti = [&](const char* k, int d) {
    return dd.contains(k) ? (int)dd.at(k).as_int(d) : d;
  };
  auto getb = [&](const char* k, bool d) {
    return dd.contains(k) ? dd.at(k).as_bool(d) : d;
  };
  auto gets = [&](const char* k, const char* d) {
    return std::string(dd.contains(k) ? dd.at(k).as_string(d) : d);
  };

  out.ch = geti("ch", 128);
  out.out_ch = geti("out_ch", 2);
  out.num_res_blocks = geti("num_res_blocks", 2);
  out.z_channels = geti("z_channels", 8);
  out.mel_bins = geti("mel_bins", 64);
  out.norm_type = gets("norm_type", "pixel");
  out.causality_axis = gets("causality_axis", "height");
  out.mid_block_add_attention = getb("mid_block_add_attention", false);
  if (dd.contains("ch_mult")) {
    const FlexData& m = dd.at("ch_mult");
    if (!m.is_array()) { return fail("`ch_mult` is not an array"); }
    const auto a = m.as_array();
    out.ch_mult.clear();
    for (std::size_t i = 0; i < a.size(); ++i) {
      out.ch_mult.push_back((int)a.at(i).as_int(1));
    }
  }
  if (out.ch_mult.empty()) { return fail("`ch_mult` is empty"); }

  // What this port implements, stated as a refusal rather than
  // discovered as a wrong spectrogram.
  if (out.norm_type != "pixel") {
    return fail("audio VAE norm_type '" + out.norm_type +
                "' is not supported; this decoder implements pixel_norm");
  }
  if (out.causality_axis != "height") {
    return fail("audio VAE causality_axis '" + out.causality_axis +
                "' is not supported; this decoder is causal in TIME "
                "(height), padding only at the top");
  }
  if (out.mid_block_add_attention) {
    return fail("this audio decoder's mid block has attention, which is "
                "not ported");
  }
  return true;
}

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
Ltx25AudioVaeDecoder::load_conv_(WeightSet& ws, const std::string& prefix,
                                 Conv& out, bool optional, std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  std::vector<int> shape;
  const std::string wn = prefix + ".conv.weight";
  if (!shape_of_(ws, wn, shape)) {
    // An absent nin_shortcut / upsample is the structure saying in ==
    // out, or that this is the last level. Only a REQUIRED tensor
    // missing is an error.
    if (optional) { out.cout = 0; return true; }
    return fail("missing tensor " + wn);
  }
  if (shape.size() != 4 || shape[2] != shape[3] ||
      (shape[2] != 3 && shape[2] != 1)) {
    return fail(wn + " is not a 3x3 or 1x1 kernel");
  }
  out.cout = shape[0];
  out.cin = shape[1];
  out.k = shape[2];
  // tensor(), not read(): the decoder KEEPS these. [C_out][C_in][k][k]
  // IS [C_out][C_in*k*k], the K-major weight the GEMM wants, so nothing
  // is permuted at load.
  out.w = ws.tensor(wn, _ops->mc(), WeightSet::Residency::Copied);
  out.b = ws.tensor(prefix + ".conv.bias", _ops->mc(),
                    WeightSet::Residency::Copied);
  if (out.w.empty() || out.b.empty()) { return fail("could not bind " + prefix); }
  _resident += out.w.byte_size() + out.b.byte_size();
  return true;
}

bool
Ltx25AudioVaeDecoder::load_res_(WeightSet& ws, const std::string& prefix,
                                ResBlock& out, std::string* err)
{
  return load_conv_(ws, prefix + ".conv1", out.conv1, false, err) &&
         load_conv_(ws, prefix + ".conv2", out.conv2, false, err) &&
         load_conv_(ws, prefix + ".nin_shortcut", out.nin, true, err);
}

std::unique_ptr<Ltx25AudioVaeDecoder>
Ltx25AudioVaeDecoder::load(const AudioVaeConfig& cfg,
                           std::shared_ptr<WeightSet> ws, const MetalOps& ops,
                           std::string* err)
{
  std::unique_ptr<Ltx25AudioVaeDecoder> m(new Ltx25AudioVaeDecoder());
  m->_cfg = cfg;
  m->_ws = std::move(ws);
  m->_ops = &ops;
  WeightSet& w = *m->_ws;

  // The checkpoint nests everything under `audio_vae.`; the decoder's
  // own tensors add `decoder.` on top, and the statistics sit BESIDE
  // the decoder rather than under it.
  const std::string P = "audio_vae.decoder.";
  if (!m->load_conv_(w, P + "conv_in", m->_conv_in, false, err) ||
      !m->load_conv_(w, P + "conv_out", m->_conv_out, false, err) ||
      !m->load_res_(w, P + "mid.block_1", m->_mid1, err) ||
      !m->load_res_(w, P + "mid.block_2", m->_mid2, err)) {
    return nullptr;
  }
  m->_levels.resize((std::size_t)cfg.num_levels());
  for (int lvl = 0; lvl < cfg.num_levels(); ++lvl) {
    Level& L = m->_levels[(std::size_t)lvl];
    const std::string lp = P + "up." + std::to_string(lvl);
    // num_res_blocks + 1 -- the LDM decoder's extra block per level.
    for (int b = 0; b < cfg.num_res_blocks + 1; ++b) {
      ResBlock rb;
      if (!m->load_res_(w, lp + ".block." + std::to_string(b), rb, err)) {
        return nullptr;
      }
      L.blocks.push_back(std::move(rb));
    }
    // Level 0 has no upsample; the others do.
    if (!m->load_conv_(w, lp + ".upsample.conv", L.upsample, true, err)) {
      return nullptr;
    }
  }

  // The statistics, as f32. z_channels * latent_mel_bins long and
  // indexed c*latent_mel_bins + f -- NOT per channel. Absent is not an
  // error (the reference's defaults are identity), so this falls back
  // rather than refusing a checkpoint that has none.
  const int S = cfg.stats_len();
  std::vector<float> sv((std::size_t)S, 1.0f), mv((std::size_t)S, 0.0f);
  auto load_stat = [&](const char* name, std::vector<float>& dst) {
    const auto* info = w.src().info(name);
    if (info == nullptr) { return; }
    const SharedBuffer b = w.read(name, ops.mc(), WeightSet::Residency::Copied);
    if (b.empty() || info->shape.size() != 1 || (int)info->shape[0] != S) {
      return;
    }
    if (info->dtype == "F32") {
      std::memcpy(dst.data(), b.contents(), (std::size_t)S * 4);
    } else if (info->dtype == "BF16") {
      const auto* p = static_cast<const std::uint16_t*>(b.contents());
      for (int i = 0; i < S; ++i) {
        const std::uint32_t u = (std::uint32_t)p[i] << 16;
        std::memcpy(&dst[(std::size_t)i], &u, 4);
      }
    }
  };
  load_stat("audio_vae.per_channel_statistics.std-of-means", sv);
  load_stat("audio_vae.per_channel_statistics.mean-of-means", mv);
  m->_std = ops.upload_f32(sv);
  m->_mean = ops.upload_f32(mv);
  return m;
}

void
Ltx25AudioVaeDecoder::conv_(CommandStream& stream, const Conv& c,
                            const SharedBuffer& x, int H, int W,
                            const SharedBuffer& y)
{
  const int cells = H * W;
  if (c.k == 1) {
    // A 1x1 CausalConv2d pads by nothing at all, so the shortcut is a
    // plain GEMM over the channel-last rows -- no im2col, no padding
    // convention to get wrong.
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
Ltx25AudioVaeDecoder::res_(CommandStream& stream, const ResBlock& rb,
                           SharedBuffer& x, int& cc, int H, int W)
{
  const std::size_t cells = (std::size_t)H * W;
  SharedBuffer h = _ops->alloc(cells * (std::size_t)cc);
  {
    // h = SiLU(PixelNorm(x)). Copy first: the norm is in place and `x`
    // is the residual this block adds back.
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
Ltx25AudioVaeDecoder::grab_(CommandStream& stream, const std::string& name,
                            const SharedBuffer& x, int C, int H, int W)
{
  if (!_capture) { return; }
  Tap t;
  t.name = name;
  t.shape = {C, H, W};
  t.buf = _ops->mc()->make_shared_buffer((std::size_t)C * H * W * 4);
  if (t.buf.empty()) { return; }
  auto enc = stream.begin_compute();
  _ops->audio_out(enc, x, t.buf, C, H, W);
  enc.end();
  _taps.push_back(std::move(t));
}

const std::vector<float>*
Ltx25AudioVaeDecoder::tap(const std::string& name) const
{
  for (const Tap& t : _taps) {
    if (t.name == name) { return &t.host; }
  }
  return nullptr;
}

bool
Ltx25AudioVaeDecoder::tap_shape(const std::string& name,
                                std::array<int, 3>* out) const
{
  for (const Tap& t : _taps) {
    if (t.name == name) {
      if (out != nullptr) { *out = t.shape; }
      return true;
    }
  }
  return false;
}

bool
Ltx25AudioVaeDecoder::decode(const float* latent, int F,
                             std::vector<float>* out,
                             std::array<int, 3>* shape, std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  if (latent == nullptr || out == nullptr) { return fail("null argument"); }
  if (F <= 0) { return fail("degenerate latent shape"); }
  _taps.clear();

  const int C = _cfg.z_channels, MEL = _cfg.latent_mel_bins();
  std::vector<float> lat((std::size_t)C * F * MEL);
  std::memcpy(lat.data(), latent, lat.size() * 4);
  const SharedBuffer d_lat = _ops->upload_f32(lat);

  CommandStream stream = _ops->mc()->make_command_stream();

  int ch = F, cw = MEL, cc = C;
  SharedBuffer x = _ops->alloc((std::size_t)ch * cw * cc);
  {
    auto enc = stream.begin_compute();
    _ops->audio_denorm_in(enc, d_lat, _std, _mean, x, C, ch, cw);
    enc.end();
  }
  grab_(stream, "denorm", x, cc, ch, cw);
  SharedBuffer y = _ops->alloc((std::size_t)ch * cw * _conv_in.cout);
  conv_(stream, _conv_in, x, ch, cw, y);
  x = std::move(y);
  cc = _conv_in.cout;
  grab_(stream, "conv_in", x, cc, ch, cw);

  res_(stream, _mid1, x, cc, ch, cw);
  res_(stream, _mid2, x, cc, ch, cw);
  grab_(stream, "mid", x, cc, ch, cw);

  // Levels run HIGH to LOW, which is the reverse of how they are stored.
  for (int lvl = _cfg.num_levels() - 1; lvl >= 0; --lvl) {
    const Level& L = _levels[(std::size_t)lvl];
    for (const ResBlock& rb : L.blocks) {
      res_(stream, rb, x, cc, ch, cw);
    }
    grab_(stream, "up" + std::to_string(lvl), x, cc, ch, cw);
    if (L.upsample.cout == 0) { continue; }
    // Nearest 2x on BOTH axes, then the conv, then drop the first row.
    const int uh = ch * 2, uw = cw * 2;
    SharedBuffer up = _ops->alloc((std::size_t)uh * uw * cc);
    {
      auto enc = stream.begin_compute();
      _ops->audio_up2x(enc, x, up, cc, ch, cw);
      enc.end();
    }
    SharedBuffer uc = _ops->alloc((std::size_t)uh * uw * L.upsample.cout);
    conv_(stream, L.upsample, up, uh, uw, uc);
    // The drop is FREE: channel-last rows are contiguous, so row 0 is
    // the first uw*cout elements and dropping it is a subview. What
    // makes this legal rather than a shortcut is that the conv is
    // causal in H -- output row 0 depends only on input row 0, so it
    // carries no information the surviving rows lack.
    x = uc.subview((std::size_t)uw * (std::size_t)L.upsample.cout * 2,
                   (std::size_t)(uh - 1) * uw * (std::size_t)L.upsample.cout
                       * 2);
    ch = uh - 1;
    cw = uw;
    cc = L.upsample.cout;
  }

  // norm_out -> SiLU -> conv_out. tanh_out is False.
  {
    auto enc = stream.begin_compute();
    _ops->vae_pixel_norm_silu(enc, x, cc, ch * cw);
    enc.end();
  }
  SharedBuffer mel = _ops->alloc((std::size_t)ch * cw * _conv_out.cout);
  conv_(stream, _conv_out, x, ch, cw, mel);

  const int oc = _conv_out.cout;
  const std::size_t n = (std::size_t)oc * ch * cw;
  SharedBuffer o = _ops->mc()->make_shared_buffer(n * sizeof(float));
  if (o.empty()) { return fail("could not allocate the output spectrogram"); }
  {
    auto enc = stream.begin_compute();
    _ops->audio_out(enc, mel, o, oc, ch, cw);
    enc.end();
  }
  stream.commit().wait();

  for (Tap& t : _taps) {
    const std::size_t tn =
        (std::size_t)t.shape[0] * t.shape[1] * t.shape[2];
    t.host.resize(tn);
    std::memcpy(t.host.data(), t.buf.contents(), tn * sizeof(float));
  }
  out->resize(n);
  std::memcpy(out->data(), o.contents(), n * sizeof(float));
  if (shape != nullptr) { *shape = {oc, ch, cw}; }
  return true;
}

}  // namespace ltx25
