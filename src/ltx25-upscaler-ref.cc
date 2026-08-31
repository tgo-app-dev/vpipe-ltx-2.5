#include "ltx25-upscaler-ref.h"

#include "common/flex-data.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using vpipe::genai::WeightSet;

namespace ltx25 {

namespace {

float
bf16_to_f32_(std::uint16_t h)
{
  const std::uint32_t u = (std::uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

float
f16_to_f32_(std::uint16_t h)
{
  const std::uint32_t s = (std::uint32_t)(h >> 15) & 1u;
  std::int32_t e = (std::int32_t)((h >> 10) & 0x1fu);
  std::uint32_t m = (std::uint32_t)h & 0x3ffu;
  std::uint32_t u;
  if (e == 0) {
    if (m == 0) { u = s << 31; }
    else {
      e = -1;
      do { ++e; m <<= 1; } while ((m & 0x400u) == 0);
      m &= 0x3ffu;
      u = (s << 31) | ((std::uint32_t)(e + 127 - 15) << 23) | (m << 13);
    }
  } else if (e == 31) {
    u = (s << 31) | 0x7f800000u | (m << 13);
  } else {
    u = (s << 31) | ((std::uint32_t)(e + 127 - 15) << 23) | (m << 13);
  }
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// Read one tensor into f32, whatever the file stores it as.
//
// read(), not tensor(): these bytes are CONSUMED into the f32 vectors
// and dropped, so caching them would keep a redundant copy alive beside
// the product -- the WeightSet contract's "cache what the model KEEPS,
// read what it consumes".
bool
load_(WeightSet& ws, vpipe::metal_compute::MetalCompute* mc,
      const std::string& name, std::vector<float>& out,
      std::vector<int>* shape, std::string* err)
{
  const auto* info = ws.src().info(name);
  if (info == nullptr) {
    if (err != nullptr) { *err = "missing tensor " + name; }
    return false;
  }
  std::size_t n = 1;
  if (shape != nullptr) { shape->clear(); }
  for (std::int64_t d : info->shape) {
    n *= (std::size_t)d;
    if (shape != nullptr) { shape->push_back((int)d); }
  }
  out.resize(n);
  const vpipe::metal_compute::SharedBuffer b =
      ws.read(name, mc, WeightSet::Residency::Copied);
  const void* raw = b.empty() ? nullptr : b.contents();
  if (raw == nullptr) {
    if (err != nullptr) { *err = "could not read " + name; }
    return false;
  }
  if (info->dtype == "F32") {
    std::memcpy(out.data(), raw, n * 4);
  } else if (info->dtype == "BF16") {
    const auto* p = static_cast<const std::uint16_t*>(raw);
    for (std::size_t i = 0; i < n; ++i) { out[i] = bf16_to_f32_(p[i]); }
  } else if (info->dtype == "F16") {
    const auto* p = static_cast<const std::uint16_t*>(raw);
    for (std::size_t i = 0; i < n; ++i) { out[i] = f16_to_f32_(p[i]); }
  } else {
    if (err != nullptr) {
      *err = name + " has dtype " + info->dtype + ", which is not supported";
    }
    return false;
  }
  return true;
}

// A [C][F][H][W] volume, channel-major -- the layout the goldens are
// saved in and the layout every op below reads.
struct Vol {
  std::vector<float> v;
  int c = 0, f = 0, h = 0, w = 0;

  void resize(int C, int F, int H, int W)
  {
    c = C; f = F; h = H; w = W;
    v.assign((std::size_t)C * F * H * W, 0.0f);
  }
  std::size_t idx(int ci, int fi, int hi, int wi) const
  {
    return (((std::size_t)ci * f + fi) * h + hi) * (std::size_t)w + wi;
  }
  float at(int ci, int fi, int hi, int wi) const { return v[idx(ci,fi,hi,wi)]; }
};

// Conv3d, kernel 3x3x3, padding 1, stride 1 -- torch's Conv3d default
// padding mode, which is ZERO on every axis. (The VAE's convolutions
// replicate-pad in time; these do not, and using one for the other is a
// quiet edge-only error.)
void
conv3d_(const Vol& x, const std::vector<float>& w, const std::vector<float>& b,
        int cout, int cin, int kf, int kh, int kw, Vol& y)
{
  y.resize(cout, x.f, x.h, x.w);
  const int of = kf / 2, oh = kh / 2, ow = kw / 2;
  for (int co = 0; co < cout; ++co) {
    const float bias = b.empty() ? 0.0f : b[(std::size_t)co];
    for (int fi = 0; fi < x.f; ++fi) {
      for (int hi = 0; hi < x.h; ++hi) {
        for (int wi = 0; wi < x.w; ++wi) {
          double acc = bias;
          for (int ci = 0; ci < cin; ++ci) {
            for (int a = 0; a < kf; ++a) {
              const int sf = fi + a - of;
              if (sf < 0 || sf >= x.f) { continue; }
              for (int p = 0; p < kh; ++p) {
                const int sh = hi + p - oh;
                if (sh < 0 || sh >= x.h) { continue; }
                for (int q = 0; q < kw; ++q) {
                  const int sw = wi + q - ow;
                  if (sw < 0 || sw >= x.w) { continue; }
                  const std::size_t wi_ =
                      ((((std::size_t)co * cin + ci) * kf + a) * kh + p)
                          * (std::size_t)kw + q;
                  acc += (double)x.at(ci, sf, sh, sw) * w[wi_];
                }
              }
            }
          }
          y.v[y.idx(co, fi, hi, wi)] = (float)acc;
        }
      }
    }
  }
}

// Conv2d, kernel 3x3, padding 1 -- applied PER FRAME. The spatial
// upsampler's weight is 4-D, and that is the only thing that says the
// module reshapes to (b f) c h w first.
void
conv2d_per_frame_(const Vol& x, const std::vector<float>& w,
                  const std::vector<float>& b, int cout, int cin, int kh,
                  int kw, Vol& y)
{
  y.resize(cout, x.f, x.h, x.w);
  const int oh = kh / 2, ow = kw / 2;
  for (int fi = 0; fi < x.f; ++fi) {
    for (int co = 0; co < cout; ++co) {
      const float bias = b.empty() ? 0.0f : b[(std::size_t)co];
      for (int hi = 0; hi < x.h; ++hi) {
        for (int wi = 0; wi < x.w; ++wi) {
          double acc = bias;
          for (int ci = 0; ci < cin; ++ci) {
            for (int p = 0; p < kh; ++p) {
              const int sh = hi + p - oh;
              if (sh < 0 || sh >= x.h) { continue; }
              for (int q = 0; q < kw; ++q) {
                const int sw = wi + q - ow;
                if (sw < 0 || sw >= x.w) { continue; }
                const std::size_t wi_ =
                    (((std::size_t)co * cin + ci) * kh + p)
                        * (std::size_t)kw + q;
                acc += (double)x.at(ci, fi, sh, sw) * w[wi_];
              }
            }
          }
          y.v[y.idx(co, fi, hi, wi)] = (float)acc;
        }
      }
    }
  }
}

// GroupNorm: `groups` groups over the channel axis, reducing over
// (channels-in-group x F x H x W), then an affine PER CHANNEL.
//
// The reduction spans the whole group, not one channel -- that is what
// makes it a GROUP norm, and reducing per channel instead is an
// InstanceNorm that still produces a normalised tensor of the right
// shape. torch's eps default is 1e-5.
void
group_norm_(Vol& x, const std::vector<float>& gamma,
            const std::vector<float>& beta, int groups, double eps = 1e-5)
{
  if (groups <= 0 || x.c % groups != 0) { return; }
  const int per = x.c / groups;
  const std::size_t plane = (std::size_t)x.f * x.h * x.w;
  const std::size_t n = plane * (std::size_t)per;
  for (int g = 0; g < groups; ++g) {
    double mean = 0.0;
    for (int k = 0; k < per; ++k) {
      const float* p = x.v.data() + (std::size_t)(g * per + k) * plane;
      for (std::size_t i = 0; i < plane; ++i) { mean += (double)p[i]; }
    }
    mean /= (double)n;
    double var = 0.0;
    for (int k = 0; k < per; ++k) {
      const float* p = x.v.data() + (std::size_t)(g * per + k) * plane;
      for (std::size_t i = 0; i < plane; ++i) {
        const double d = (double)p[i] - mean;
        var += d * d;
      }
    }
    var /= (double)n;
    const double inv = 1.0 / std::sqrt(var + eps);
    for (int k = 0; k < per; ++k) {
      const int ci = g * per + k;
      float* p = x.v.data() + (std::size_t)ci * plane;
      const double sc = gamma.empty() ? 1.0 : (double)gamma[(std::size_t)ci];
      const double sh = beta.empty() ? 0.0 : (double)beta[(std::size_t)ci];
      for (std::size_t i = 0; i < plane; ++i) {
        p[i] = (float)((((double)p[i] - mean) * inv) * sc + sh);
      }
    }
  }
}

void
silu_(Vol& x)
{
  for (float& z : x.v) { z = z / (1.0f + std::exp(-z)); }
}

// PixelShuffleND(2) on a per-frame basis: (c p1 p2) h w -> c (h p1) (w p2).
//
// THE NESTING: p2 (WIDTH) is FASTEST, so the source channel is
// ((c*2)+p1)*2+p2. The VAE's DepthToSpace and its unpatchify nest in
// opposite orders in this same model family, so this is spelled out
// rather than shared.
void
pixel_shuffle_2d_(const Vol& x, int p1, int p2, Vol& y)
{
  const int co = x.c / (p1 * p2);
  y.resize(co, x.f, x.h * p1, x.w * p2);
  for (int c = 0; c < co; ++c) {
    for (int a = 0; a < p1; ++a) {
      for (int b = 0; b < p2; ++b) {
        const int src = (c * p1 + a) * p2 + b;
        for (int fi = 0; fi < x.f; ++fi) {
          for (int hi = 0; hi < x.h; ++hi) {
            for (int wi = 0; wi < x.w; ++wi) {
              y.v[y.idx(c, fi, hi * p1 + a, wi * p2 + b)] =
                  x.at(src, fi, hi, wi);
            }
          }
        }
      }
    }
  }
}

// PixelShuffleND(1): (c p1) f h w -> c (f p1) h w.
void
pixel_shuffle_1d_(const Vol& x, int p1, Vol& y)
{
  const int co = x.c / p1;
  y.resize(co, x.f * p1, x.h, x.w);
  for (int c = 0; c < co; ++c) {
    for (int a = 0; a < p1; ++a) {
      const int src = c * p1 + a;
      for (int fi = 0; fi < x.f; ++fi) {
        for (int hi = 0; hi < x.h; ++hi) {
          for (int wi = 0; wi < x.w; ++wi) {
            y.v[y.idx(c, fi * p1 + a, hi, wi)] = x.at(src, fi, hi, wi);
          }
        }
      }
    }
  }
}

}  // namespace

bool
UpscalerConfig::from_metadata(const std::string& file, UpscalerConfig* out,
                              std::string* err)
{
  if (out == nullptr) { return false; }
  std::FILE* f = std::fopen(file.c_str(), "rb");
  if (f == nullptr) {
    if (err != nullptr) { *err = "cannot open " + file; }
    return false;
  }
  std::uint64_t n = 0;
  if (std::fread(&n, 1, sizeof(n), f) != sizeof(n) || n == 0
      || n > (std::uint64_t{1} << 30)) {
    std::fclose(f);
    if (err != nullptr) { *err = file + " is not a safetensors file"; }
    return false;
  }
  std::string hdr((std::size_t)n, '\0');
  const bool ok = std::fread(hdr.data(), 1, (std::size_t)n, f)
                  == (std::size_t)n;
  std::fclose(f);
  if (!ok) {
    if (err != nullptr) { *err = "short header in " + file; }
    return false;
  }
  vpipe::FlexData doc;
  try { doc = vpipe::FlexData::from_json(hdr); }
  catch (...) { doc = vpipe::FlexData(); }
  if (!doc.is_object()) {
    if (err != nullptr) { *err = "unparsable header in " + file; }
    return false;
  }
  const auto o = doc.as_object();
  if (!o.contains("__metadata__")) {
    if (err != nullptr) { *err = file + " carries no __metadata__"; }
    return false;
  }
  const vpipe::FlexData meta = o.at("__metadata__");
  if (!meta.is_object()) { return false; }
  const auto mo = meta.as_object();
  if (!mo.contains("config")) {
    if (err != nullptr) { *err = file + " carries no config"; }
    return false;
  }
  // The config is a JSON STRING inside the metadata object, not a
  // nested object -- safetensors metadata values are all strings.
  const vpipe::FlexData cfg_s = mo.at("config");
  vpipe::FlexData cfg;
  try { cfg = vpipe::FlexData::from_json(std::string(cfg_s.as_string())); }
  catch (...) { cfg = vpipe::FlexData(); }
  if (!cfg.is_object()) {
    if (err != nullptr) { *err = file + "'s config is not JSON"; }
    return false;
  }
  const auto c = cfg.as_object();
  auto geti = [&](const char* k, int d) {
    return c.contains(k) ? (int)c.at(k).as_int(d) : d;
  };
  auto getb = [&](const char* k, bool d) {
    return c.contains(k) ? c.at(k).as_bool(d) : d;
  };
  out->in_channels = geti("in_channels", out->in_channels);
  out->mid_channels = geti("mid_channels", out->mid_channels);
  out->num_blocks_per_stage =
      geti("num_blocks_per_stage", out->num_blocks_per_stage);
  out->dims = geti("dims", out->dims);
  out->spatial_upsample = getb("spatial_upsample", out->spatial_upsample);
  out->temporal_upsample = getb("temporal_upsample", out->temporal_upsample);
  out->spatial_scale =
      c.contains("spatial_scale") ? c.at("spatial_scale").as_real(2.0) : 2.0;
  out->rational_resampler =
      getb("rational_resampler", out->rational_resampler);
  return true;
}

std::unique_ptr<Ltx25UpscalerRef>
Ltx25UpscalerRef::load(const UpscalerConfig& cfg, WeightSet& ws,
                       vpipe::metal_compute::MetalCompute* mc,
                       std::string* err)
{
  std::unique_ptr<Ltx25UpscalerRef> m(new Ltx25UpscalerRef());
  m->_cfg = cfg;
  if (!cfg.spatial_upsample && !cfg.temporal_upsample) {
    if (err != nullptr) {
      *err = "the config enables neither spatial nor temporal upsampling";
    }
    return nullptr;
  }
  // rational_resampler is deliberately NOT consulted: the module reads
  // it only on the spatial-without-temporal branch, and the shipped
  // temporal checkpoint sets it true while taking the plain Sequential
  // branch. Believing it would build a different module.
  if (cfg.spatial_upsample && cfg.rational_resampler) {
    if (err != nullptr) {
      *err = "a rational-resampler SPATIAL upscaler is not implemented; "
             "the shipped x2 spatial checkpoint sets it false";
    }
    return nullptr;
  }

  auto conv = [&](const std::string& p, Ltx25UpscalerRef::Conv& c) {
    std::vector<int> sh;
    std::string e;
    if (!load_(ws, mc, p + ".weight", c.w, &sh, &e)) {
      if (err != nullptr) { *err = e; }
      return false;
    }
    if (sh.size() == 5) {
      c.cout = sh[0]; c.cin = sh[1]; c.kf = sh[2]; c.kh = sh[3]; c.kw = sh[4];
    } else if (sh.size() == 4) {
      c.cout = sh[0]; c.cin = sh[1]; c.kf = 1; c.kh = sh[2]; c.kw = sh[3];
    } else {
      if (err != nullptr) { *err = p + ".weight is not a 4-D or 5-D kernel"; }
      return false;
    }
    load_(ws, mc, p + ".bias", c.b, nullptr, &e);   // bias is optional
    return true;
  };
  auto norm = [&](const std::string& p, Ltx25UpscalerRef::Norm& nn) {
    std::string e;
    if (!load_(ws, mc, p + ".weight", nn.w, nullptr, &e)) {
      if (err != nullptr) { *err = e; }
      return false;
    }
    load_(ws, mc, p + ".bias", nn.b, nullptr, &e);
    return true;
  };
  auto blocks = [&](const std::string& p,
                    std::vector<Ltx25UpscalerRef::ResBlock>& out) {
    out.resize((std::size_t)cfg.num_blocks_per_stage);
    for (int i = 0; i < cfg.num_blocks_per_stage; ++i) {
      const std::string b = p + "." + std::to_string(i) + ".";
      if (!conv(b + "conv1", out[(std::size_t)i].conv1)
          || !norm(b + "norm1", out[(std::size_t)i].norm1)
          || !conv(b + "conv2", out[(std::size_t)i].conv2)
          || !norm(b + "norm2", out[(std::size_t)i].norm2)) {
        return false;
      }
    }
    return true;
  };

  if (!conv("initial_conv", m->_initial_conv)
      || !norm("initial_norm", m->_initial_norm)
      || !blocks("res_blocks", m->_res)
      || !conv("upsampler.0", m->_up_conv)
      || !blocks("post_upsample_res_blocks", m->_post)
      || !conv("final_conv", m->_final_conv)) {
    return nullptr;
  }
  return m;
}

const std::vector<float>*
Ltx25UpscalerRef::tap(const std::string& name) const
{
  for (const auto& t : _taps) {
    if (t.first == name) { return &t.second; }
  }
  return nullptr;
}

bool
Ltx25UpscalerRef::tap_shape(const std::string& name,
                            std::array<int, 4>* shape) const
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
Ltx25UpscalerRef::forward(const float* latent, int F, int H, int W,
                          std::vector<float>* out,
                          std::array<int, 4>* shape, std::string* err)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  if (latent == nullptr || out == nullptr) { return fail("no input"); }
  if (F <= 0 || H <= 0 || W <= 0) { return fail("degenerate geometry"); }

  _taps.clear();
  _tap_shapes.clear();
  auto keep = [&](const char* name, const Vol& v) {
    if (!_capture) { return; }
    _taps.emplace_back(name, v.v);
    _tap_shapes.emplace_back(name, std::array<int, 4>{v.c, v.f, v.h, v.w});
  };

  Vol x;
  x.resize(_cfg.in_channels, F, H, W);
  std::memcpy(x.v.data(), latent,
              x.v.size() * sizeof(float));

  Vol h;
  conv3d_(x, _initial_conv.w, _initial_conv.b, _initial_conv.cout,
          _initial_conv.cin, _initial_conv.kf, _initial_conv.kh,
          _initial_conv.kw, h);
  group_norm_(h, _initial_norm.w, _initial_norm.b, 32);
  silu_(h);
  keep("initial", h);

  auto run_block = [&](const ResBlock& b, Vol& v) {
    const Vol residual = v;
    Vol t;
    conv3d_(v, b.conv1.w, b.conv1.b, b.conv1.cout, b.conv1.cin, b.conv1.kf,
            b.conv1.kh, b.conv1.kw, t);
    group_norm_(t, b.norm1.w, b.norm1.b, 32);
    silu_(t);
    Vol u;
    conv3d_(t, b.conv2.w, b.conv2.b, b.conv2.cout, b.conv2.cin, b.conv2.kf,
            b.conv2.kh, b.conv2.kw, u);
    group_norm_(u, b.norm2.w, b.norm2.b, 32);
    // The activation is on (x + residual), not on x -- the residual
    // joins BEFORE the SiLU, which is the unusual half of this block.
    for (std::size_t i = 0; i < u.v.size(); ++i) {
      u.v[i] += residual.v[i];
    }
    silu_(u);
    v = std::move(u);
  };
  for (const ResBlock& b : _res) { run_block(b, h); }
  keep("res", h);

  Vol up;
  if (_cfg.temporal_upsample && !_cfg.spatial_upsample) {
    Vol c;
    conv3d_(h, _up_conv.w, _up_conv.b, _up_conv.cout, _up_conv.cin,
            _up_conv.kf, _up_conv.kh, _up_conv.kw, c);
    Vol s;
    pixel_shuffle_1d_(c, 2, s);
    keep("shuffled", s);
    // DROP FRAME 0. The first latent frame encodes a single pixel
    // frame, so the shuffle's first output frame is not a real one.
    up.resize(s.c, s.f - 1, s.h, s.w);
    for (int ci = 0; ci < s.c; ++ci) {
      for (int fi = 1; fi < s.f; ++fi) {
        for (int hi = 0; hi < s.h; ++hi) {
          for (int wi = 0; wi < s.w; ++wi) {
            up.v[up.idx(ci, fi - 1, hi, wi)] = s.at(ci, fi, hi, wi);
          }
        }
      }
    }
  } else if (_cfg.spatial_upsample && !_cfg.temporal_upsample) {
    Vol c;
    conv2d_per_frame_(h, _up_conv.w, _up_conv.b, _up_conv.cout, _up_conv.cin,
                      _up_conv.kh, _up_conv.kw, c);
    pixel_shuffle_2d_(c, 2, 2, up);
  } else {
    return fail("a combined spatial+temporal upscaler is not implemented");
  }
  keep("upsampled", up);

  for (const ResBlock& b : _post) { run_block(b, up); }
  keep("post", up);

  Vol y;
  conv3d_(up, _final_conv.w, _final_conv.b, _final_conv.cout,
          _final_conv.cin, _final_conv.kf, _final_conv.kh, _final_conv.kw, y);
  if (shape != nullptr) { *shape = {y.c, y.f, y.h, y.w}; }
  *out = std::move(y.v);
  return true;
}

}  // namespace ltx25
