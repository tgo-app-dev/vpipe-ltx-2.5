#include "ltx25-upscaler.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

using vpipe::genai::WeightSet;
using vpipe::metal_compute::CommandStream;
using vpipe::metal_compute::SharedBuffer;

namespace ltx25 {

bool
Ltx25Upscaler::load_conv_(WeightSet& ws, const std::string& p, Conv& out,
                          std::string* err)
{
  const auto* info = ws.src().info(p + ".weight");
  if (info == nullptr) {
    if (err != nullptr) { *err = "missing " + p + ".weight"; }
    return false;
  }
  const auto& sh = info->shape;
  if (sh.size() == 5) {
    out.cout = (int)sh[0]; out.cin = (int)sh[1];
    out.kf = (int)sh[2]; out.kh = (int)sh[3]; out.kw = (int)sh[4];
  } else if (sh.size() == 4) {
    // A 4-D kernel is the SPATIAL upsampler's Conv2d, which the module
    // runs per frame. The rank is the only thing that says so.
    out.cout = (int)sh[0]; out.cin = (int)sh[1];
    out.kf = 1; out.kh = (int)sh[2]; out.kw = (int)sh[3];
  } else {
    if (err != nullptr) { *err = p + ".weight is not a 4-D or 5-D kernel"; }
    return false;
  }
  // Bound as-is: the checkpoint ships bf16 and every kernel here is
  // bf16, so there is no conversion and no second copy. Cached, because
  // the model KEEPS these.
  out.w = _ws->tensor(p + ".weight", _ops->mc(),
                      WeightSet::Residency::Copied);
  if (out.w.empty()) {
    if (err != nullptr) { *err = "could not bind " + p + ".weight"; }
    return false;
  }
  if (_ws->src().info(p + ".bias") != nullptr) {
    out.b = _ws->tensor(p + ".bias", _ops->mc(),
                        WeightSet::Residency::Copied);
  }
  _resident += out.w.byte_size() + out.b.byte_size();
  return true;
}

bool
Ltx25Upscaler::load_norm_(WeightSet& ws, const std::string& p, Norm& out,
                          std::string* err)
{
  if (ws.src().info(p + ".weight") == nullptr) {
    if (err != nullptr) { *err = "missing " + p + ".weight"; }
    return false;
  }
  // f32, not bf16: these are two vectors of `mid_channels` and they
  // multiply a normalised value, so the rounding would land directly on
  // the output. Cheap at 1024 floats.
  auto to_f32 = [&](const std::string& name, SharedBuffer& dst) {
    const auto* info = ws.src().info(name);
    if (info == nullptr) { return; }
    const SharedBuffer raw =
        ws.read(name, _ops->mc(), WeightSet::Residency::Copied);
    if (raw.empty()) { return; }
    std::size_t n = 1;
    for (std::int64_t d : info->shape) { n *= (std::size_t)d; }
    std::vector<float> v(n, 0.0f);
    if (info->dtype == "F32") {
      std::memcpy(v.data(), raw.contents(), n * 4);
    } else if (info->dtype == "BF16") {
      const auto* q = static_cast<const std::uint16_t*>(raw.contents());
      for (std::size_t i = 0; i < n; ++i) {
        const std::uint32_t u = (std::uint32_t)q[i] << 16;
        std::memcpy(&v[i], &u, 4);
      }
    }
    dst = _ops->upload_f32(v);
  };
  to_f32(p + ".weight", out.g);
  to_f32(p + ".bias", out.b);
  if (out.g.empty()) {
    if (err != nullptr) { *err = "could not bind " + p + ".weight"; }
    return false;
  }
  _resident += out.g.byte_size() + out.b.byte_size();
  return true;
}

std::unique_ptr<Ltx25Upscaler>
Ltx25Upscaler::load(const UpscalerConfig& cfg,
                    const std::vector<float>& std_of_means,
                    const std::vector<float>& mean_of_means,
                    std::shared_ptr<WeightSet> ws, const MetalOps& ops,
                    std::string* err)
{
  if (!ws) {
    if (err != nullptr) { *err = "no weight set"; }
    return nullptr;
  }
  std::unique_ptr<Ltx25Upscaler> m(new Ltx25Upscaler());
  m->_cfg = cfg;
  m->_ws = std::move(ws);
  m->_ops = &ops;

  if (!cfg.spatial_upsample && !cfg.temporal_upsample) {
    if (err != nullptr) {
      *err = "the config enables neither spatial nor temporal upsampling";
    }
    return nullptr;
  }
  if (cfg.spatial_upsample && cfg.temporal_upsample) {
    if (err != nullptr) {
      *err = "a combined spatial+temporal upscaler is not implemented; "
             "the two shipped checkpoints do one each";
    }
    return nullptr;
  }
  // rational_resampler is NOT consulted -- see the note in
  // ltx25-upscaler-ref.h. The shipped temporal checkpoint sets it true
  // and takes the plain Sequential branch anyway.
  if (cfg.spatial_upsample && cfg.rational_resampler) {
    if (err != nullptr) {
      *err = "a rational-resampler SPATIAL upscaler is not implemented";
    }
    return nullptr;
  }

  const int C = cfg.in_channels;
  std::vector<float> sv((std::size_t)C, 1.0f), mv((std::size_t)C, 0.0f);
  if ((int)std_of_means.size() == C) { sv = std_of_means; }
  if ((int)mean_of_means.size() == C) { mv = mean_of_means; }
  m->_std = ops.upload_f32(sv);
  m->_mean = ops.upload_f32(mv);

  auto blocks = [&](const std::string& p, std::vector<ResBlock>& out) {
    out.resize((std::size_t)cfg.num_blocks_per_stage);
    for (int i = 0; i < cfg.num_blocks_per_stage; ++i) {
      const std::string b = p + "." + std::to_string(i) + ".";
      ResBlock& r = out[(std::size_t)i];
      if (!m->load_conv_(*m->_ws, b + "conv1", r.conv1, err)
          || !m->load_norm_(*m->_ws, b + "norm1", r.norm1, err)
          || !m->load_conv_(*m->_ws, b + "conv2", r.conv2, err)
          || !m->load_norm_(*m->_ws, b + "norm2", r.norm2, err)) {
        return false;
      }
    }
    return true;
  };
  if (!m->load_conv_(*m->_ws, "initial_conv", m->_initial_conv, err)
      || !m->load_norm_(*m->_ws, "initial_norm", m->_initial_norm, err)
      || !blocks("res_blocks", m->_res)
      || !m->load_conv_(*m->_ws, "upsampler.0", m->_up_conv, err)
      || !blocks("post_upsample_res_blocks", m->_post)
      || !m->load_conv_(*m->_ws, "final_conv", m->_final_conv, err)) {
    return nullptr;
  }
  // The rank of the upsampler's kernel must agree with the config: a
  // 4-D kernel is a per-frame Conv2d (spatial), a 5-D one is a Conv3d
  // (temporal). Disagreeing means the config and the file describe
  // different modules, and the forward below would pick by config.
  const bool want_2d = cfg.spatial_upsample;
  if (m->_up_conv.two_d() != want_2d) {
    if (err != nullptr) {
      *err = std::string("upsampler.0.weight is ")
           + (m->_up_conv.two_d() ? "4-D (a per-frame Conv2d)"
                                  : "5-D (a Conv3d)")
           + " but the config asks for a "
           + (want_2d ? "SPATIAL" : "TEMPORAL") + " upscaler";
    }
    return nullptr;
  }
  return m;
}

const std::vector<float>*
Ltx25Upscaler::tap(const std::string& name) const
{
  for (const auto& t : _taps) {
    if (t.first == name) { return &t.second; }
  }
  return nullptr;
}

// Downloads channel-LAST bf16 and rewrites it channel-FIRST f32, which
// is the layout the reference's taps are in -- otherwise every
// comparison would be against a permutation.
void
Ltx25Upscaler::keep_(const char* name, const SharedBuffer& b, std::size_t n)
{
  if (!_capture || b.empty()) { return; }
  const std::vector<float> v = MetalOps::download_bf16(b, n);
  _taps.emplace_back(name, v);
}

void
Ltx25Upscaler::release_idle()
{
  _im2col = SharedBuffer();
  _a = SharedBuffer();
  _b = SharedBuffer();
}

void
Ltx25Upscaler::conv_(CommandStream& stream, const Conv& c,
                     const SharedBuffer& x, int F, int H, int W,
                     const SharedBuffer& y)
{
  const int cells = F * H * W;
  const int taps = c.two_d() ? 9 : 27;
  const std::size_t k = (std::size_t)c.cin * (std::size_t)taps;
  // CHUNKED over output cells so the im2col scratch is bounded by the
  // ceiling and not by the clip. One chunk is one gather + one GEMM.
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
    if (c.two_d()) {
      _ops->ups_im2col2d(enc, x, _im2col, c.cin, F, H, W, c0, n);
    } else {
      _ops->ups_im2col(enc, x, _im2col, c.cin, F, H, W, c0, n);
    }
    // y[n][cout] = im2col[n][cin*taps] @ w[cout][cin*taps]^T + bias.
    // The column order the gather writes IS the weight's own flattening,
    // so nothing is permuted at load.
    _ops->linear(enc, _im2col, c.w, c.b.empty() ? nullptr : &c.b,
                 y.subview((std::size_t)c0 * (std::size_t)c.cout * 2,
                           (std::size_t)n * (std::size_t)c.cout * 2),
                 n, (int)k, c.cout);
    enc.end();
  }
}

void
Ltx25Upscaler::block_(CommandStream& stream, const ResBlock& b,
                      SharedBuffer& x, int F, int H, int W)
{
  const int cells = F * H * W;
  const std::size_t n = (std::size_t)cells * (std::size_t)b.conv1.cout;
  if (_a.empty() || _a.byte_size() < n * 2) { _a = _ops->alloc(n); }
  if (_b.empty() || _b.byte_size() < n * 2) { _b = _ops->alloc(n); }

  conv_(stream, b.conv1, x, F, H, W, _a);
  {
    auto enc = stream.begin_compute();
    _ops->ups_group_norm(enc, _a, b.norm1.g, b.norm1.b, b.conv1.cout, cells);
    _ops->ups_silu(enc, _a, n);
    enc.end();
  }
  conv_(stream, b.conv2, _a, F, H, W, _b);
  {
    auto enc = stream.begin_compute();
    _ops->ups_group_norm(enc, _b, b.norm2.g, b.norm2.b, b.conv2.cout, cells);
    // THE RESIDUAL JOINS BEFORE THE ACTIVATION. `SiLU(x + residual)`,
    // not `SiLU(x) + residual` -- the unusual half of this block, and
    // both orders produce a correctly-shaped tensor.
    _ops->vae_add_into(enc, _b, x, n);
    _ops->ups_silu(enc, _b, n);
    enc.end();
  }
  std::swap(x, _b);
}

bool
Ltx25Upscaler::upscale(const float* latent, int F, int H, int W,
                       std::vector<float>* out, std::array<int, 4>* shape,
                       std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  if (latent == nullptr || out == nullptr) { return fail("null argument"); }
  if (F <= 0 || H <= 0 || W <= 0) { return fail("degenerate latent shape"); }

  const int C = _cfg.in_channels;
  const int mid = _initial_conv.cout;
  const int cells = F * H * W;
  _taps.clear();

  std::vector<float> lat((std::size_t)C * cells);
  std::memcpy(lat.data(), latent, lat.size() * 4);
  const SharedBuffer d_lat = _ops->upload_f32(lat);

  auto stream = _ops->mc()->make_command_stream();
  // COMMITTED PER STAGE, not as one buffer for the whole model.
  //
  // MEASURED, and it is not a preference: built as a single command
  // buffer this model returns a WRONG answer -- rel-L2 0.97 against the
  // reference where a per-stage commit gives 1.2e-2 -- deterministically,
  // with no GPU error reported. The DiT says the same thing at its own
  // block fence: an over-committed buffer fails silently under a bare
  // wait(). ~17 convolutions, each of which is a gather plus a GEMM, is
  // past whatever that limit is.
  //
  // So every stage below fences, and every fence is CHECKED.
  auto fence = [&](const char* what) {
    std::string e;
    const bool ok = stream.commit().wait_ok(&e);
    stream = _ops->mc()->make_command_stream();
    if (!ok && err != nullptr) {
      *err = std::string(what) + ": "
           + (e.empty() ? std::string("GPU error") : e);
    }
    return ok;
  };

  // UN-NORMALIZE and go channel-last in one op: the model works in the
  // VAE's raw latent space while the DiT generates in the whitened one.
  SharedBuffer x = _ops->alloc((std::size_t)cells * (std::size_t)C);
  {
    auto enc = stream.begin_compute();
    _ops->vae_denorm_in(enc, d_lat, _std, _mean, x, C, F, H, W);
    enc.end();
  }

  SharedBuffer h = _ops->alloc((std::size_t)cells * (std::size_t)mid);
  conv_(stream, _initial_conv, x, F, H, W, h);
  {
    auto enc = stream.begin_compute();
    _ops->ups_group_norm(enc, h, _initial_norm.g, _initial_norm.b, mid,
                         cells);
    _ops->ups_silu(enc, h, (std::size_t)cells * (std::size_t)mid);
    enc.end();
  }
  if (!fence("initial")) { return false; }
  keep_("initial", h, (std::size_t)cells * (std::size_t)mid);
  for (const ResBlock& b : _res) {
    block_(stream, b, h, F, H, W);
    if (!fence("res block")) { return false; }
  }
  keep_("res", h, (std::size_t)cells * (std::size_t)mid);

  // The upsampler, then the shuffle. `vae_d2s` is PixelShuffleND: it
  // splits (c p1 p2 p3) with WIDTH fastest and drops the first frame,
  // which is exactly what both branches want.
  int oF = F, oH = H, oW = W;
  SharedBuffer up;
  {
    const int mult = _cfg.spatial_upsample ? 4 : 2;
    SharedBuffer c =
        _ops->alloc((std::size_t)cells * (std::size_t)(mid * mult));
    conv_(stream, _up_conv, h, F, H, W, c);
    if (_cfg.spatial_upsample) {
      oH = H * 2; oW = W * 2;
      up = _ops->alloc((std::size_t)(oF * oH * oW) * (std::size_t)mid);
      auto enc = stream.begin_compute();
      _ops->vae_d2s(enc, c, up, mid * 4, F, H, W, 1, 2, 2, 0);
      enc.end();
    } else {
      // 2F frames from the shuffle, then FRAME 0 IS DROPPED: the first
      // latent frame encodes a single pixel frame, so the shuffle's
      // first output frame is not a real one. 2F-1 out.
      oF = 2 * F - 1;
      up = _ops->alloc((std::size_t)(oF * oH * oW) * (std::size_t)mid);
      auto enc = stream.begin_compute();
      _ops->vae_d2s(enc, c, up, mid * 2, F, H, W, 2, 1, 1, 1);
      enc.end();
    }
  }
  if (!fence("upsampler")) { return false; }
  keep_("upsampled", up, (std::size_t)(oF * oH * oW) * (std::size_t)mid);
  for (const ResBlock& b : _post) {
    block_(stream, b, up, oF, oH, oW);
    if (!fence("post block")) { return false; }
  }
  keep_("post", up, (std::size_t)(oF * oH * oW) * (std::size_t)mid);

  const int ocells = oF * oH * oW;
  SharedBuffer y = _ops->alloc((std::size_t)ocells * (std::size_t)C);
  conv_(stream, _final_conv, up, oF, oH, oW, y);

  // RE-NORMALIZE on the way out, back into the whitened space the
  // sampler and the DiT hold.
  SharedBuffer fout =
      _ops->mc()->make_shared_buffer((std::size_t)C * ocells * 4);
  {
    auto enc = stream.begin_compute();
    _ops->vae_whiten_out(enc, y, _std, _mean, fout, C, C, oF, oH, oW);
    enc.end();
  }
  if (!fence("output")) { return false; }

  out->assign((std::size_t)C * ocells, 0.0f);
  std::memcpy(out->data(), fout.contents(), out->size() * 4);
  if (shape != nullptr) { *shape = {C, oF, oH, oW}; }
  return true;
}

}  // namespace ltx25
