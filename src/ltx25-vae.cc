#include "ltx25-vae.h"

#include "apple-silicon/metal-compute/command-stream.h"

#include <algorithm>
#include <cstring>

using vpipe::genai::WeightSet;
using vpipe::metal_compute::CommandStream;
using vpipe::metal_compute::ComputeEncoder;
using vpipe::metal_compute::SharedBuffer;

namespace ltx25 {

namespace {

// A checkpoint tensor's shape, without materialising it.
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
Ltx25VaeDecoder::load_conv_(WeightSet& ws, const std::string& prefix,
                            Conv& out, std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  std::vector<int> shape;
  const std::string wn = prefix + ".conv.weight";
  if (!shape_of_(ws, wn, shape)) { return fail("missing tensor " + wn); }
  if (shape.size() != 5 || shape[2] != 3 || shape[3] != 3 || shape[4] != 3) {
    return fail(wn + " is not a 3x3x3 kernel");
  }
  out.cout = shape[0];
  out.cin  = shape[1];
  // tensor(), not read(): the decoder KEEPS these for its lifetime, and
  // Copied because this loader converts nothing -- the bf16 bytes go to
  // the GEMM as they sit. [C_out][C_in][3][3][3] IS [C_out][C_in*27],
  // which is exactly the K-major weight the GEMM wants, so nothing is
  // permuted at load.
  out.w = ws.tensor(wn, _ops->mc(), WeightSet::Residency::Copied);
  out.b = ws.tensor(prefix + ".conv.bias", _ops->mc(),
                    WeightSet::Residency::Copied);
  if (out.w.empty() || out.b.empty()) {
    return fail("could not bind " + prefix);
  }
  _resident += out.w.byte_size() + out.b.byte_size();
  return true;
}

std::unique_ptr<Ltx25VaeDecoder>
Ltx25VaeDecoder::load(const VaeConfig& cfg, std::shared_ptr<WeightSet> ws,
                      const MetalOps& ops, std::string* err)
{
  std::unique_ptr<Ltx25VaeDecoder> m(new Ltx25VaeDecoder());
  m->_cfg = cfg;
  // Held for this decoder's lifetime: every bound tensor is an alias
  // into the set, and the checkpoint must unmap when the LAST holder
  // goes away.
  m->_ws  = std::move(ws);
  m->_ops = &ops;
  if (!m->load_conv_(*m->_ws, "decoder.conv_in", m->_conv_in, err) ||
      !m->load_conv_(*m->_ws, "decoder.conv_out", m->_conv_out, err)) {
    return nullptr;
  }
  const std::vector<VaeBlock> ups = cfg.up_blocks();
  for (std::size_t i = 0; i < ups.size(); ++i) {
    const VaeBlock& b = ups[i];
    const std::string p = "decoder.up_blocks." + std::to_string(i);
    UpBlock u;
    u.is_res = b.is_res();
    if (u.is_res) {
      for (int r = 0;; ++r) {
        const std::string rp = p + ".res_blocks." + std::to_string(r);
        if (m->_ws->src().info(rp + ".conv1.conv.weight") == nullptr) {
          break;
        }
        ResBlock rb;
        if (!m->load_conv_(*m->_ws, rp + ".conv1", rb.conv1, err) ||
            !m->load_conv_(*m->_ws, rp + ".conv2", rb.conv2, err)) {
          return nullptr;
        }
        u.res.push_back(std::move(rb));
      }
      if (u.res.empty()) {
        if (err != nullptr) { *err = p + " has no res_blocks"; }
        return nullptr;
      }
    } else {
      if (!m->load_conv_(*m->_ws, p + ".conv", u.conv, err)) {
        return nullptr;
      }
      b.stride(&u.st, &u.sh, &u.sw);
    }
    m->_ups.push_back(std::move(u));
  }

  // The latent statistics, as f32. Absent is not an error -- the
  // reference's defaults are identity -- so this falls back to that
  // rather than refusing a checkpoint that simply has none.
  const int C = cfg.latent_channels;
  std::vector<float> sv((std::size_t)C, 1.0f), mv((std::size_t)C, 0.0f);
  auto load_stat = [&](const char* name, std::vector<float>& dst) {
    const auto* info = m->_ws->src().info(name);
    if (info == nullptr) { return; }
    const SharedBuffer b =
        m->_ws->read(name, ops.mc(), WeightSet::Residency::Copied);
    if (b.empty() || info->shape.size() != 1 ||
        (int)info->shape[0] != C) {
      return;
    }
    if (info->dtype == "F32") {
      std::memcpy(dst.data(), b.contents(), (std::size_t)C * 4);
    } else if (info->dtype == "BF16") {
      const auto* p = static_cast<const std::uint16_t*>(b.contents());
      for (int i = 0; i < C; ++i) {
        const std::uint32_t w = (std::uint32_t)p[i] << 16;
        std::memcpy(&dst[(std::size_t)i], &w, 4);
      }
    }
  };
  load_stat("per_channel_statistics.std-of-means", sv);
  load_stat("per_channel_statistics.mean-of-means", mv);
  m->_std_of_means  = ops.upload_f32(sv);
  m->_mean_of_means = ops.upload_f32(mv);
  return m;
}

void
Ltx25VaeDecoder::conv3d_(CommandStream& stream, const Conv& c,
                         const SharedBuffer& x, int F, int H, int W,
                         const SharedBuffer& y)
{
  const int cells = F * H * W;
  const std::size_t k = (std::size_t)c.cin * 27;
  // CHUNKED over output cells so the im2col scratch is bounded by the
  // configured ceiling rather than by the picture. One chunk is one
  // im2col + one GEMM, so bigger is better until it does not fit.
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
    _ops->vae_im2col(enc, x, _im2col, c.cin, F, H, W, c0, n);
    // y[n][cout] = im2col[n][cin*27] @ w[cout][cin*27]^T + bias.
    _ops->linear(enc, _im2col, c.w, &c.b,
                 y.subview((std::size_t)c0 * (std::size_t)c.cout * 2,
                           (std::size_t)n * (std::size_t)c.cout * 2),
                 n, (int)k, c.cout);
    enc.end();
  }
}

bool
Ltx25VaeDecoder::decode(const float* latent, int F, int H, int W,
                        std::vector<float>* out, std::array<int, 4>* shape,
                        std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  if (latent == nullptr || out == nullptr) { return fail("null argument"); }
  if (F <= 0 || H <= 0 || W <= 0) { return fail("degenerate latent shape"); }

  const int C = _cfg.latent_channels;
  std::vector<float> lat((std::size_t)C * F * H * W);
  std::memcpy(lat.data(), latent, lat.size() * 4);
  const SharedBuffer d_lat = _ops->upload_f32(lat);

  CommandStream stream = _ops->mc()->make_command_stream();

  // ---- conv_in, into channel-last ------------------------------------
  SharedBuffer x = _ops->alloc((std::size_t)F * H * W * C);
  {
    auto enc = stream.begin_compute();
    _ops->vae_denorm_in(enc, d_lat, _std_of_means, _mean_of_means, x, C, F,
                        H, W);
    enc.end();
  }
  SharedBuffer y =
      _ops->alloc((std::size_t)F * H * W * _conv_in.cout);
  conv3d_(stream, _conv_in, x, F, H, W, y);
  x = std::move(y);
  int cc = _conv_in.cout, cf = F, ch = H, cw = W;

  // ---- the up blocks --------------------------------------------------
  for (const UpBlock& u : _ups) {
    if (u.is_res) {
      for (const ResBlock& rb : u.res) {
        const std::size_t n = (std::size_t)cf * ch * cw * cc;
        SharedBuffer h = _ops->alloc(n);
        {
          // h = SiLU(PixelNorm(x)) -- copy first, since the norm is in
          // place and `x` is the residual this block adds back.
          auto enc = stream.begin_compute();
          _ops->copy(enc, x, h, (int)n);
          _ops->vae_pixel_norm_silu(enc, h, cc, cf * ch * cw);
          enc.end();
        }
        SharedBuffer t = _ops->alloc((std::size_t)cf * ch * cw * rb.conv1.cout);
        conv3d_(stream, rb.conv1, h, cf, ch, cw, t);
        {
          auto enc = stream.begin_compute();
          _ops->vae_pixel_norm_silu(enc, t, rb.conv1.cout, cf * ch * cw);
          enc.end();
        }
        SharedBuffer t2 =
            _ops->alloc((std::size_t)cf * ch * cw * rb.conv2.cout);
        conv3d_(stream, rb.conv2, t, cf, ch, cw, t2);
        {
          // The shortcut is Identity (in == out everywhere), so the
          // residual is a plain add.
          auto enc = stream.begin_compute();
          _ops->vae_add_into(enc, x, t2, n);
          enc.end();
        }
      }
    } else {
      SharedBuffer t = _ops->alloc((std::size_t)cf * ch * cw * u.conv.cout);
      conv3d_(stream, u.conv, x, cf, ch, cw, t);
      const int drop = (u.st == 2) ? 1 : 0;
      const int of = cf * u.st - drop, oh = ch * u.sh, ow = cw * u.sw;
      const int oc = u.conv.cout / (u.st * u.sh * u.sw);
      SharedBuffer o = _ops->alloc((std::size_t)of * oh * ow * oc);
      {
        auto enc = stream.begin_compute();
        _ops->vae_d2s(enc, t, o, u.conv.cout, cf, ch, cw, u.st, u.sh, u.sw,
                      drop);
        enc.end();
      }
      x = std::move(o);
      cc = oc; cf = of; ch = oh; cw = ow;
    }
  }

  // ---- the output head ------------------------------------------------
  {
    auto enc = stream.begin_compute();
    _ops->vae_pixel_norm_silu(enc, x, cc, cf * ch * cw);
    enc.end();
  }
  SharedBuffer o48 = _ops->alloc((std::size_t)cf * ch * cw * _conv_out.cout);
  conv3d_(stream, _conv_out, x, cf, ch, cw, o48);

  const int patch = _cfg.patch_size;
  const int oc = _conv_out.cout / (patch * patch);
  const int oh = ch * patch, ow = cw * patch;
  const std::size_t npix = (std::size_t)oc * cf * oh * ow;
  SharedBuffer pix = _ops->mc()->make_shared_buffer(npix * sizeof(float));
  if (pix.empty()) { return fail("could not allocate the output picture"); }
  {
    auto enc = stream.begin_compute();
    _ops->vae_unpatchify(enc, o48, pix, _conv_out.cout, cf, ch, cw, patch);
    enc.end();
  }
  stream.commit().wait();

  out->resize(npix);
  std::memcpy(out->data(), pix.contents(), npix * sizeof(float));
  if (shape != nullptr) { *shape = {oc, cf, oh, ow}; }
  return true;
}

// ---------------------------------------------------------------------
// The ENCODER.
// ---------------------------------------------------------------------

bool
Ltx25VaeEncoder::load_conv_(WeightSet& ws, const std::string& prefix,
                            Conv& out, std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  std::vector<int> shape;
  const std::string wn = prefix + ".conv.weight";
  if (!shape_of_(ws, wn, shape)) { return fail("missing tensor " + wn); }
  if (shape.size() != 5 || shape[2] != 3 || shape[3] != 3 || shape[4] != 3) {
    return fail(wn + " is not a 3x3x3 kernel");
  }
  out.cout = shape[0];
  out.cin  = shape[1];
  out.w = ws.tensor(wn, _ops->mc(), WeightSet::Residency::Copied);
  out.b = ws.tensor(prefix + ".conv.bias", _ops->mc(),
                    WeightSet::Residency::Copied);
  if (out.w.empty() || out.b.empty()) {
    return fail("could not bind " + prefix);
  }
  _resident += out.w.byte_size() + out.b.byte_size();
  return true;
}

std::unique_ptr<Ltx25VaeEncoder>
Ltx25VaeEncoder::load(const VaeConfig& cfg, std::shared_ptr<WeightSet> ws,
                      const MetalOps& ops, std::string* err)
{
  std::unique_ptr<Ltx25VaeEncoder> m(new Ltx25VaeEncoder());
  m->_cfg = cfg;
  m->_ws  = std::move(ws);
  m->_ops = &ops;
  if (!m->load_conv_(*m->_ws, "encoder.conv_in", m->_conv_in, err) ||
      !m->load_conv_(*m->_ws, "encoder.conv_out", m->_conv_out, err)) {
    return nullptr;
  }
  // In LIST order: the decoder reverses the block list because it undoes
  // the encoder, and this is the encoder the list was written for.
  int ch = m->_conv_in.cout;
  for (std::size_t i = 0; i < cfg.encoder_blocks.size(); ++i) {
    const VaeBlock& b = cfg.encoder_blocks[i];
    const std::string p = "encoder.down_blocks." + std::to_string(i);
    DownBlock dn;
    dn.is_res = b.is_res();
    if (dn.is_res) {
      for (int r = 0;; ++r) {
        const std::string rp = p + ".res_blocks." + std::to_string(r);
        if (m->_ws->src().info(rp + ".conv1.conv.weight") == nullptr) {
          break;
        }
        ResBlock rb;
        if (!m->load_conv_(*m->_ws, rp + ".conv1", rb.conv1, err) ||
            !m->load_conv_(*m->_ws, rp + ".conv2", rb.conv2, err)) {
          return nullptr;
        }
        dn.res.push_back(std::move(rb));
      }
      if (dn.res.empty()) {
        if (err != nullptr) { *err = p + " has no res_blocks"; }
        return nullptr;
      }
    } else {
      if (!m->load_conv_(*m->_ws, p + ".conv", dn.conv, err)) {
        return nullptr;
      }
      b.stride(&dn.st, &dn.sh, &dn.sw);
      const int prod = dn.st * dn.sh * dn.sw;
      const int out_ch = dn.conv.cout * prod;
      if (out_ch == 0 || (ch * prod) % out_ch != 0) {
        if (err != nullptr) {
          *err = p + ": " + std::to_string(ch) + " channels through a " +
                 std::to_string(prod) + "x space-to-depth does not divide " +
                 std::to_string(out_ch);
        }
        return nullptr;
      }
      dn.group_size = (ch * prod) / out_ch;
      ch = out_ch;
    }
    m->_downs.push_back(std::move(dn));
  }

  const int C = cfg.latent_channels;
  std::vector<float> sv((std::size_t)C, 1.0f), mv((std::size_t)C, 0.0f);
  auto load_stat = [&](const char* name, std::vector<float>& dst) {
    const auto* info = m->_ws->src().info(name);
    if (info == nullptr) { return; }
    const SharedBuffer b =
        m->_ws->read(name, ops.mc(), WeightSet::Residency::Copied);
    if (b.empty() || info->shape.size() != 1 || (int)info->shape[0] != C) {
      return;
    }
    if (info->dtype == "F32") {
      std::memcpy(dst.data(), b.contents(), (std::size_t)C * 4);
    } else if (info->dtype == "BF16") {
      const auto* p = static_cast<const std::uint16_t*>(b.contents());
      for (int i = 0; i < C; ++i) {
        const std::uint32_t w = (std::uint32_t)p[i] << 16;
        std::memcpy(&dst[(std::size_t)i], &w, 4);
      }
    }
  };
  load_stat("per_channel_statistics.std-of-means", sv);
  load_stat("per_channel_statistics.mean-of-means", mv);
  m->_std_of_means  = ops.upload_f32(sv);
  m->_mean_of_means = ops.upload_f32(mv);
  return m;
}

void
Ltx25VaeEncoder::conv3d_(CommandStream& stream, const Conv& c,
                         const SharedBuffer& x, int F, int H, int W,
                         const SharedBuffer& y)
{
  const int cells = F * H * W;
  const std::size_t k = (std::size_t)c.cin * 27;
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
    // CAUSAL, unlike the decoder's: this is the one op in the encoder
    // that is not simply the decoder's run backwards.
    _ops->vae_im2col_causal(enc, x, _im2col, c.cin, F, H, W, c0, n);
    _ops->linear(enc, _im2col, c.w, &c.b,
                 y.subview((std::size_t)c0 * (std::size_t)c.cout * 2,
                           (std::size_t)n * (std::size_t)c.cout * 2),
                 n, (int)k, c.cout);
    enc.end();
  }
}

bool
Ltx25VaeEncoder::encode(const float* pixels, int F, int H, int W,
                        std::vector<float>* out, std::array<int, 4>* shape,
                        std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  if (pixels == nullptr || out == nullptr) { return fail("null argument"); }
  const int tf = _cfg.temporal_factor(), sf = _cfg.spatial_factor();
  if (F <= 0 || H <= 0 || W <= 0) { return fail("degenerate pixel shape"); }
  if ((F - 1) % tf != 0) {
    return fail("this VAE encodes 1 + " + std::to_string(tf) +
                "k frames; got " + std::to_string(F) +
                ". Cropping here would hand the DiT a reference for a clip "
                "it is not generating, so it is refused instead.");
  }
  if (H % sf != 0 || W % sf != 0) {
    return fail("this VAE compresses space by " + std::to_string(sf) + "; " +
                std::to_string(H) + "x" + std::to_string(W) +
                " is not a multiple");
  }

  const int Cin = _cfg.out_channels;   // the PIXEL channel count (3)
  std::vector<float> host((std::size_t)Cin * F * H * W);
  std::memcpy(host.data(), pixels, host.size() * 4);
  const SharedBuffer d_px = _ops->upload_f32(host);

  CommandStream stream = _ops->mc()->make_command_stream();

  // ---- patchify, into channel-last ------------------------------------
  const int patch = _cfg.patch_size;
  int cf = F, ch = H / patch, cw = W / patch;
  int cc = Cin * patch * patch;
  SharedBuffer x = _ops->alloc((std::size_t)cf * ch * cw * cc);
  {
    auto enc = stream.begin_compute();
    _ops->vae_patchify_in(enc, d_px, x, Cin, F, H, W, patch);
    enc.end();
  }
  SharedBuffer y = _ops->alloc((std::size_t)cf * ch * cw * _conv_in.cout);
  conv3d_(stream, _conv_in, x, cf, ch, cw, y);
  x = std::move(y);
  cc = _conv_in.cout;

  // ---- the down blocks -------------------------------------------------
  for (const DownBlock& dn : _downs) {
    if (dn.is_res) {
      for (const ResBlock& rb : dn.res) {
        const std::size_t n = (std::size_t)cf * ch * cw * cc;
        SharedBuffer h = _ops->alloc(n);
        {
          auto enc = stream.begin_compute();
          _ops->copy(enc, x, h, (int)n);
          _ops->vae_pixel_norm_silu(enc, h, cc, cf * ch * cw);
          enc.end();
        }
        SharedBuffer t =
            _ops->alloc((std::size_t)cf * ch * cw * rb.conv1.cout);
        conv3d_(stream, rb.conv1, h, cf, ch, cw, t);
        {
          auto enc = stream.begin_compute();
          _ops->vae_pixel_norm_silu(enc, t, rb.conv1.cout, cf * ch * cw);
          enc.end();
        }
        SharedBuffer t2 =
            _ops->alloc((std::size_t)cf * ch * cw * rb.conv2.cout);
        conv3d_(stream, rb.conv2, t, cf, ch, cw, t2);
        {
          auto enc = stream.begin_compute();
          _ops->vae_add_into(enc, x, t2, n);
          enc.end();
        }
      }
    } else {
      // A time-halving block sees ONE EXTRA FRAME: its own frame 0,
      // duplicated -- and BOTH its conv and its skip read the padded
      // volume. That is what keeps 1 + 8k -> 1 + k exact rather than
      // losing a frame at each of the three halvings.
      SharedBuffer in = std::move(x);
      int inf = cf;
      if (dn.st == 2) {
        SharedBuffer dup =
            _ops->alloc((std::size_t)(cf + 1) * ch * cw * cc);
        auto enc = stream.begin_compute();
        _ops->vae_dup_frame0(enc, in, dup, cc, cf, ch, cw);
        enc.end();
        in = std::move(dup);
        inf = cf + 1;
      }
      const int prod = dn.st * dn.sh * dn.sw;
      const int oc = dn.conv.cout * prod;
      const int of = inf / dn.st, oh = ch / dn.sh, ow = cw / dn.sw;

      // The SKIP: space-to-depth of the block's INPUT, with consecutive
      // channel groups averaged down to the output width. A mean, not a
      // projection -- there is nothing to bind, which is exactly why
      // leaving it out loads cleanly and encodes something plausible.
      SharedBuffer skip = _ops->alloc((std::size_t)of * oh * ow * oc);
      {
        auto enc = stream.begin_compute();
        _ops->vae_s2d(enc, in, skip, cc, inf, ch, cw, dn.st, dn.sh, dn.sw,
                      dn.group_size);
        enc.end();
      }
      // The CONV path: a stride-1 convolution, then the same regroup.
      SharedBuffer t = _ops->alloc((std::size_t)inf * ch * cw * dn.conv.cout);
      conv3d_(stream, dn.conv, in, inf, ch, cw, t);
      SharedBuffer o = _ops->alloc((std::size_t)of * oh * ow * oc);
      {
        auto enc = stream.begin_compute();
        _ops->vae_s2d(enc, t, o, dn.conv.cout, inf, ch, cw, dn.st, dn.sh,
                      dn.sw, /*group=*/1);
        _ops->vae_add_into(enc, o, skip, (std::size_t)of * oh * ow * oc);
        enc.end();
      }
      x = std::move(o);
      cc = oc; cf = of; ch = oh; cw = ow;
    }
  }

  // ---- the head --------------------------------------------------------
  {
    auto enc = stream.begin_compute();
    _ops->vae_pixel_norm_silu(enc, x, cc, cf * ch * cw);
    enc.end();
  }
  SharedBuffer head =
      _ops->alloc((std::size_t)cf * ch * cw * _conv_out.cout);
  conv3d_(stream, _conv_out, x, cf, ch, cw, head);

  const int C = _cfg.latent_channels;
  if (_conv_out.cout < C) {
    return fail("the head emits " + std::to_string(_conv_out.cout) +
                " channels, fewer than the " + std::to_string(C) +
                " latent channels");
  }
  const std::size_t nlat = (std::size_t)C * cf * ch * cw;
  SharedBuffer lat = _ops->mc()->make_shared_buffer(nlat * sizeof(float));
  if (lat.empty()) { return fail("could not allocate the output latent"); }
  {
    auto enc = stream.begin_compute();
    _ops->vae_whiten_out(enc, head, _std_of_means, _mean_of_means, lat,
                         _conv_out.cout, C, cf, ch, cw);
    enc.end();
  }
  stream.commit().wait();

  out->resize(nlat);
  std::memcpy(out->data(), lat.contents(), nlat * sizeof(float));
  if (shape != nullptr) { *shape = {C, cf, ch, cw}; }
  return true;
}

}  // namespace ltx25
