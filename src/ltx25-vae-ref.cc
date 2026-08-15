#include "ltx25-vae-ref.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include <cmath>
#include <cstring>

using vpipe::genai::WeightSet;

namespace ltx25 {

namespace {

inline float
bf16_to_f32_(std::uint16_t b)
{
  const std::uint32_t w = (std::uint32_t)b << 16;
  float f;
  std::memcpy(&f, &w, 4);
  return f;
}

inline float
f16_to_f32_(std::uint16_t h)
{
  const std::uint32_t sg = (std::uint32_t)(h >> 15) << 31;
  const std::uint32_t ex = (h >> 10) & 0x1f;
  const std::uint32_t mn = h & 0x3ff;
  std::uint32_t w = (ex == 0) ? sg
                  : (ex == 0x1f) ? (sg | 0x7f800000u | (mn << 13))
                  : (sg | ((ex + 112) << 23) | (mn << 13));
  float f;
  std::memcpy(&f, &w, 4);
  return f;
}

// A checkpoint tensor widened to f32. The conv weights are bf16 here,
// but reading the dtype rather than assuming it is what lets the same
// loader take an f16 or f32 repack.
bool
read_f32_(WeightSet& ws, vpipe::metal_compute::MetalCompute* mc,
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
  // read(), not tensor(): these bytes are CONSUMED into the f32 vectors
  // below and then dropped, so caching them would keep a redundant copy
  // alive next to the product (the WeightSet contract's "cache what the
  // model KEEPS, read what it consumes").
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

struct Vol {
  std::vector<float> v;
  int c = 0, f = 0, h = 0, w = 0;
  std::size_t idx(int ci, int fi, int hi, int wi) const
  {
    return (((std::size_t)ci * f + fi) * h + hi) * (std::size_t)w + wi;
  }
  void resize(int C, int F, int H, int W)
  {
    c = C; f = F; h = H; w = W;
    v.assign((std::size_t)C * F * H * W, 0.0f);
  }
};

// 3x3x3 convolution. REPLICATE in time (causal=false pads both ends with
// the edge frame), ZERO in space. Stride 1, so the output keeps the
// input's F/H/W.
void
conv3d_(const Vol& x, const std::vector<float>& wt, const std::vector<float>& b,
        int cout, Vol& y)
{
  y.resize(cout, x.f, x.h, x.w);
  const int Ci = x.c;
  for (int co = 0; co < cout; ++co) {
    const float bias = b[(std::size_t)co];
    const float* wc = wt.data() + (std::size_t)co * Ci * 27;
    for (int fi = 0; fi < x.f; ++fi) {
      for (int hi = 0; hi < x.h; ++hi) {
        for (int wi = 0; wi < x.w; ++wi) {
          float acc = bias;
          for (int ci = 0; ci < Ci; ++ci) {
            const float* wk = wc + (std::size_t)ci * 27;
            for (int kf = 0; kf < 3; ++kf) {
              // Time: clamp to the edge (replicate).
              int sf = fi + kf - 1;
              if (sf < 0) { sf = 0; }
              if (sf >= x.f) { sf = x.f - 1; }
              for (int kh = 0; kh < 3; ++kh) {
                const int sh = hi + kh - 1;
                if (sh < 0 || sh >= x.h) { continue; }   // zero pad
                for (int kw = 0; kw < 3; ++kw) {
                  const int sw = wi + kw - 1;
                  if (sw < 0 || sw >= x.w) { continue; } // zero pad
                  acc += wk[(kf * 3 + kh) * 3 + kw] * x.v[x.idx(ci, sf, sh, sw)];
                }
              }
            }
          }
          y.v[y.idx(co, fi, hi, wi)] = acc;
        }
      }
    }
  }
}

// PixelNorm: x / sqrt(mean(x^2 over CHANNELS) + eps), per (f,h,w).
// eps 1e-8 -- the PixelNorm class default, which is what the resnet
// blocks and conv_norm_out both construct.
void
pixel_norm_(Vol& x)
{
  constexpr float kEps = 1e-8f;
  for (int fi = 0; fi < x.f; ++fi) {
    for (int hi = 0; hi < x.h; ++hi) {
      for (int wi = 0; wi < x.w; ++wi) {
        double ss = 0.0;
        for (int ci = 0; ci < x.c; ++ci) {
          const double t = x.v[x.idx(ci, fi, hi, wi)];
          ss += t * t;
        }
        const float inv =
            1.0f / std::sqrt((float)(ss / (double)x.c) + kEps);
        for (int ci = 0; ci < x.c; ++ci) { x.v[x.idx(ci, fi, hi, wi)] *= inv; }
      }
    }
  }
}

void
silu_(Vol& x)
{
  for (float& t : x.v) { t = t / (1.0f + std::exp(-t)); }
}

// depth-to-space: `b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)`.
// p3 (WIDTH) is the FASTEST channel axis, then p2 (height), then p1
// (time). When p1 == 2 the FIRST time slice is dropped.
void
depth_to_space_(const Vol& x, int p1, int p2, int p3, Vol& y)
{
  const int c = x.c / (p1 * p2 * p3);
  const int fo = x.f * p1, ho = x.h * p2, wo = x.w * p3;
  Vol full;
  full.resize(c, fo, ho, wo);
  for (int ci = 0; ci < c; ++ci) {
    for (int a = 0; a < p1; ++a) {
      for (int bb = 0; bb < p2; ++bb) {
        for (int d = 0; d < p3; ++d) {
          const int src_c = ((ci * p1 + a) * p2 + bb) * p3 + d;
          for (int fi = 0; fi < x.f; ++fi) {
            for (int hi = 0; hi < x.h; ++hi) {
              for (int wi = 0; wi < x.w; ++wi) {
                full.v[full.idx(ci, fi * p1 + a, hi * p2 + bb, wi * p3 + d)] =
                    x.v[x.idx(src_c, fi, hi, wi)];
              }
            }
          }
        }
      }
    }
  }
  if (p1 == 2) {
    // Drop the first frame. This is what makes the decoder emit
    // 8*(F-1)+1 frames rather than 8F, and it happens at EVERY
    // time-doubling block, not once at the end.
    y.resize(c, fo - 1, ho, wo);
    for (int ci = 0; ci < c; ++ci) {
      for (int fi = 1; fi < fo; ++fi) {
        for (int hi = 0; hi < ho; ++hi) {
          for (int wi = 0; wi < wo; ++wi) {
            y.v[y.idx(ci, fi - 1, hi, wi)] = full.v[full.idx(ci, fi, hi, wi)];
          }
        }
      }
    }
  } else {
    y = std::move(full);
  }
}

// unpatchify: `b (c p r q) f h w -> b c f (h q) (w r)` with p == 1.
// q (HEIGHT) is the FASTEST channel axis and r is width -- the OPPOSITE
// nesting to depth_to_space_ above.
void
unpatchify_(const Vol& x, int patch, Vol& y)
{
  const int c = x.c / (patch * patch);
  y.resize(c, x.f, x.h * patch, x.w * patch);
  for (int ci = 0; ci < c; ++ci) {
    for (int r = 0; r < patch; ++r) {          // width offset
      for (int q = 0; q < patch; ++q) {        // height offset
        const int src_c = (ci * patch + r) * patch + q;
        for (int fi = 0; fi < x.f; ++fi) {
          for (int hi = 0; hi < x.h; ++hi) {
            for (int wi = 0; wi < x.w; ++wi) {
              y.v[y.idx(ci, fi, hi * patch + q, wi * patch + r)] =
                  x.v[x.idx(src_c, fi, hi, wi)];
            }
          }
        }
      }
    }
  }
}

// The ENCODER's convolution: 3x3x3, stride 1, CAUSAL in time.
//
// Causal padding is two copies of frame 0 at the FRONT and nothing at
// the back, so the time offset is `kf - 2` and only the low end clamps.
// The decoder's conv3d_ above pads BOTH ends with one frame each; using
// that here would let an early frame see a later one, which is exactly
// the thing "causal" is for and is invisible except at the clip's edges.
void
conv3d_causal_(const Vol& x, const std::vector<float>& wt,
               const std::vector<float>& b, int cout, Vol& y)
{
  y.resize(cout, x.f, x.h, x.w);
  const int Ci = x.c;
  for (int co = 0; co < cout; ++co) {
    const float bias = b[(std::size_t)co];
    const float* wc = wt.data() + (std::size_t)co * Ci * 27;
    for (int fi = 0; fi < x.f; ++fi) {
      for (int hi = 0; hi < x.h; ++hi) {
        for (int wi = 0; wi < x.w; ++wi) {
          float acc = bias;
          for (int ci = 0; ci < Ci; ++ci) {
            const float* wk = wc + (std::size_t)ci * 27;
            for (int kf = 0; kf < 3; ++kf) {
              int sf = fi + kf - 2;
              if (sf < 0) { sf = 0; }        // replicate frame 0
              for (int kh = 0; kh < 3; ++kh) {
                const int sh = hi + kh - 1;
                if (sh < 0 || sh >= x.h) { continue; }   // zero pad
                for (int kw = 0; kw < 3; ++kw) {
                  const int sw = wi + kw - 1;
                  if (sw < 0 || sw >= x.w) { continue; } // zero pad
                  acc += wk[(kf * 3 + kh) * 3 + kw] * x.v[x.idx(ci, sf, sh, sw)];
                }
              }
            }
          }
          y.v[y.idx(co, fi, hi, wi)] = acc;
        }
      }
    }
  }
}

// patchify: `b c f (h q) (w r) -> b (c p r q) f h w` with p == 1.
// q (HEIGHT) is the FASTEST channel axis -- the exact inverse of
// unpatchify_ above, and NOT the nesting space_to_depth_ uses.
void
patchify_(const Vol& x, int patch, Vol& y)
{
  const int c = x.c;
  y.resize(c * patch * patch, x.f, x.h / patch, x.w / patch);
  for (int ci = 0; ci < c; ++ci) {
    for (int r = 0; r < patch; ++r) {          // width offset
      for (int q = 0; q < patch; ++q) {        // height offset
        const int dst_c = (ci * patch + r) * patch + q;
        for (int fi = 0; fi < y.f; ++fi) {
          for (int hi = 0; hi < y.h; ++hi) {
            for (int wi = 0; wi < y.w; ++wi) {
              y.v[y.idx(dst_c, fi, hi, wi)] =
                  x.v[x.idx(ci, fi, hi * patch + q, wi * patch + r)];
            }
          }
        }
      }
    }
  }
}

// space-to-depth: `b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w`.
// p3 (WIDTH) is the fastest channel axis -- the inverse nesting of
// depth_to_space_, and the OPPOSITE of patchify_'s.
//
// The caller has already duplicated frame 0 when p1 == 2, so this is a
// pure regroup with no frame handling of its own.
void
space_to_depth_(const Vol& x, int p1, int p2, int p3, Vol& y)
{
  const int c = x.c;
  y.resize(c * p1 * p2 * p3, x.f / p1, x.h / p2, x.w / p3);
  for (int ci = 0; ci < c; ++ci) {
    for (int a = 0; a < p1; ++a) {
      for (int bb = 0; bb < p2; ++bb) {
        for (int d = 0; d < p3; ++d) {
          const int dst_c = ((ci * p1 + a) * p2 + bb) * p3 + d;
          for (int fi = 0; fi < y.f; ++fi) {
            for (int hi = 0; hi < y.h; ++hi) {
              for (int wi = 0; wi < y.w; ++wi) {
                y.v[y.idx(dst_c, fi, hi, wi)] =
                    x.v[x.idx(ci, fi * p1 + a, hi * p2 + bb, wi * p3 + d)];
              }
            }
          }
        }
      }
    }
  }
}

// Prepend a copy of frame 0. What a time-halving downsample does before
// it regroups, and the reason 1 + 8k frames survive three halvings as
// 1 + k rather than losing a frame at each.
void
duplicate_first_frame_(const Vol& x, Vol& y)
{
  y.resize(x.c, x.f + 1, x.h, x.w);
  for (int ci = 0; ci < x.c; ++ci) {
    for (int fi = 0; fi < y.f; ++fi) {
      const int src = (fi == 0) ? 0 : fi - 1;
      for (int hi = 0; hi < x.h; ++hi) {
        for (int wi = 0; wi < x.w; ++wi) {
          y.v[y.idx(ci, fi, hi, wi)] = x.v[x.idx(ci, src, hi, wi)];
        }
      }
    }
  }
}

// Average consecutive GROUPS of channels: `b (c g) d h w -> b c d h w`
// by mean over g. The downsample's skip path, after its space-to-depth.
void
mean_channel_groups_(Vol& x, int group)
{
  if (group <= 1) { return; }
  Vol y;
  y.resize(x.c / group, x.f, x.h, x.w);
  const std::size_t plane = (std::size_t)x.f * x.h * x.w;
  for (int ci = 0; ci < y.c; ++ci) {
    for (std::size_t i = 0; i < plane; ++i) {
      float acc = 0.0f;
      for (int g = 0; g < group; ++g) {
        acc += x.v[(std::size_t)(ci * group + g) * plane + i];
      }
      y.v[(std::size_t)ci * plane + i] = acc / (float)group;
    }
  }
  x = std::move(y);
}

}  // namespace

bool
Ltx25VaeRef::load_conv_(WeightSet& ws,
                        vpipe::metal_compute::MetalCompute* mc,
                        const std::string& prefix, Conv& out,
                        std::string* err)
{
  std::vector<int> shape;
  if (!read_f32_(ws, mc, prefix + ".conv.weight", out.w, &shape, err)) {
    return false;
  }
  if (shape.size() != 5 || shape[2] != 3 || shape[3] != 3 || shape[4] != 3) {
    if (err != nullptr) {
      *err = prefix + ".conv.weight is not a 3x3x3 kernel";
    }
    return false;
  }
  out.cout = shape[0];
  out.cin  = shape[1];
  return read_f32_(ws, mc, prefix + ".conv.bias", out.b, nullptr, err);
}

std::unique_ptr<Ltx25VaeRef>
Ltx25VaeRef::load(const VaeConfig& cfg, WeightSet& ws,
                  vpipe::metal_compute::MetalCompute* mc, std::string* err)
{
  std::unique_ptr<Ltx25VaeRef> m(new Ltx25VaeRef());
  m->_cfg = cfg;
  if (!m->load_conv_(ws, mc, "decoder.conv_in", m->_conv_in, err)) {
    return nullptr;
  }
  if (!m->load_conv_(ws, mc, "decoder.conv_out", m->_conv_out, err)) {
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
        if (ws.src().info(rp + ".conv1.conv.weight") == nullptr) { break; }
        ResBlock rb;
        if (!m->load_conv_(ws, mc, rp + ".conv1", rb.conv1, err) ||
            !m->load_conv_(ws, mc, rp + ".conv2", rb.conv2, err)) {
          return nullptr;
        }
        u.res.push_back(std::move(rb));
      }
      if (u.res.empty()) {
        if (err != nullptr) {
          *err = p + " is a res_x block with no res_blocks";
        }
        return nullptr;
      }
    } else {
      if (!m->load_conv_(ws, mc, p + ".conv", u.conv, err)) { return nullptr; }
      b.stride(&u.st, &u.sh, &u.sw);
    }
    m->_ups.push_back(std::move(u));
  }
  // The latent statistics. Absent is not an error in the reference (the
  // defaults are identity), so it is not one here either.
  read_f32_(ws, mc, "per_channel_statistics.std-of-means",
            m->_std_of_means, nullptr, nullptr);
  read_f32_(ws, mc, "per_channel_statistics.mean-of-means",
            m->_mean_of_means, nullptr, nullptr);
  return m;
}

const std::vector<float>*
Ltx25VaeRef::tap(const std::string& name) const
{
  for (const auto& t : _taps) {
    if (t.first == name) { return &t.second; }
  }
  return nullptr;
}

bool
Ltx25VaeRef::tap_shape(const std::string& name,
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
Ltx25VaeRef::decode(const float* latent, int F, int H, int W,
                    std::vector<float>* out, std::array<int, 4>* shape,
                    std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  if (latent == nullptr || out == nullptr) { return fail("null argument"); }
  if (F <= 0 || H <= 0 || W <= 0) { return fail("degenerate latent shape"); }
  _taps.clear();
  _tap_shapes.clear();

  auto capture = [&](const char* name, const Vol& v) {
    if (!_capture) { return; }
    _taps.emplace_back(name, v.v);
    _tap_shapes.emplace_back(name, std::array<int, 4>{v.c, v.f, v.h, v.w});
  };

  const int C = _cfg.latent_channels;
  Vol x;
  x.resize(C, F, H, W);
  std::memcpy(x.v.data(), latent, x.v.size() * sizeof(float));

  // Denormalize: x * std-of-means + mean-of-means, per channel.
  if ((int)_std_of_means.size() == C && (int)_mean_of_means.size() == C) {
    for (int ci = 0; ci < C; ++ci) {
      const float s = _std_of_means[(std::size_t)ci];
      const float m = _mean_of_means[(std::size_t)ci];
      for (int i = 0; i < F * H * W; ++i) {
        x.v[(std::size_t)ci * F * H * W + i] =
            x.v[(std::size_t)ci * F * H * W + i] * s + m;
      }
    }
  }

  Vol y;
  conv3d_(x, _conv_in.w, _conv_in.b, _conv_in.cout, y);
  x = std::move(y);
  capture("conv_in", x);

  for (std::size_t i = 0; i < _ups.size(); ++i) {
    const UpBlock& u = _ups[i];
    if (u.is_res) {
      for (const ResBlock& rb : u.res) {
        Vol h = x;
        pixel_norm_(h);
        silu_(h);
        Vol t;
        conv3d_(h, rb.conv1.w, rb.conv1.b, rb.conv1.cout, t);
        h = std::move(t);
        pixel_norm_(h);
        silu_(h);
        conv3d_(h, rb.conv2.w, rb.conv2.b, rb.conv2.cout, t);
        // in_channels == out_channels throughout, so the shortcut is
        // Identity (no conv_shortcut / norm3 tensors exist).
        for (std::size_t k = 0; k < x.v.size(); ++k) { x.v[k] += t.v[k]; }
      }
    } else {
      Vol t;
      conv3d_(x, u.conv.w, u.conv.b, u.conv.cout, t);
      depth_to_space_(t, u.st, u.sh, u.sw, x);
    }
    capture(("up" + std::to_string(i)).c_str(), x);
  }

  pixel_norm_(x);
  silu_(x);
  conv3d_(x, _conv_out.w, _conv_out.b, _conv_out.cout, y);
  Vol px;
  unpatchify_(y, _cfg.patch_size, px);

  *out = std::move(px.v);
  if (shape != nullptr) { *shape = {px.c, px.f, px.h, px.w}; }
  return true;
}

// ---------------------------------------------------------------------
// The ENCODER.
// ---------------------------------------------------------------------

bool
Ltx25VaeEncoderRef::load_conv_(WeightSet& ws,
                               vpipe::metal_compute::MetalCompute* mc,
                               const std::string& prefix, Conv& out,
                               std::string* err)
{
  std::vector<int> shape;
  if (!read_f32_(ws, mc, prefix + ".conv.weight", out.w, &shape, err)) {
    return false;
  }
  if (shape.size() != 5 || shape[2] != 3 || shape[3] != 3 || shape[4] != 3) {
    if (err != nullptr) {
      *err = prefix + ".conv.weight is not a 3x3x3 kernel";
    }
    return false;
  }
  out.cout = shape[0];
  out.cin  = shape[1];
  return read_f32_(ws, mc, prefix + ".conv.bias", out.b, nullptr, err);
}

std::unique_ptr<Ltx25VaeEncoderRef>
Ltx25VaeEncoderRef::load(const VaeConfig& cfg, WeightSet& ws,
                         vpipe::metal_compute::MetalCompute* mc,
                         std::string* err)
{
  std::unique_ptr<Ltx25VaeEncoderRef> m(new Ltx25VaeEncoderRef());
  m->_cfg = cfg;
  if (!m->load_conv_(ws, mc, "encoder.conv_in", m->_conv_in, err)) {
    return nullptr;
  }
  if (!m->load_conv_(ws, mc, "encoder.conv_out", m->_conv_out, err)) {
    return nullptr;
  }
  // In LIST order, not reversed: the decoder walks its block list
  // backwards because it undoes the encoder, and the encoder is the one
  // the list was written for.
  int ch = m->_conv_in.cout;
  for (std::size_t i = 0; i < cfg.encoder_blocks.size(); ++i) {
    const VaeBlock& b = cfg.encoder_blocks[i];
    const std::string p = "encoder.down_blocks." + std::to_string(i);
    DownBlock dn;
    dn.is_res = b.is_res();
    if (dn.is_res) {
      for (int r = 0;; ++r) {
        const std::string rp = p + ".res_blocks." + std::to_string(r);
        if (ws.src().info(rp + ".conv1.conv.weight") == nullptr) { break; }
        ResBlock rb;
        if (!m->load_conv_(ws, mc, rp + ".conv1", rb.conv1, err) ||
            !m->load_conv_(ws, mc, rp + ".conv2", rb.conv2, err)) {
          return nullptr;
        }
        dn.res.push_back(std::move(rb));
      }
      if (dn.res.empty()) {
        if (err != nullptr) {
          *err = p + " is a res_x block with no res_blocks";
        }
        return nullptr;
      }
    } else {
      if (!m->load_conv_(ws, mc, p + ".conv", dn.conv, err)) { return nullptr; }
      b.stride(&dn.st, &dn.sh, &dn.sw);
      // out_channels = in * multiplier; the conv itself emits
      // out / prod(stride), and the skip pools the input's own
      // space-to-depth down to out. Both numbers come from the
      // checkpoint's shapes rather than from the block name, so a
      // multiplier this port guessed wrong fails to bind instead of
      // encoding quietly.
      const int prod = dn.st * dn.sh * dn.sw;
      const int out_ch = dn.conv.cout * prod;
      dn.group_size = (ch * prod) / out_ch;
      if (dn.group_size < 1 || (ch * prod) % out_ch != 0) {
        if (err != nullptr) {
          *err = p + ": " + std::to_string(ch) + " channels through a " +
                 std::to_string(prod) + "x space-to-depth does not divide " +
                 std::to_string(out_ch);
        }
        return nullptr;
      }
      ch = out_ch;
    }
    m->_downs.push_back(std::move(dn));
  }
  read_f32_(ws, mc, "per_channel_statistics.std-of-means",
            m->_std_of_means, nullptr, nullptr);
  read_f32_(ws, mc, "per_channel_statistics.mean-of-means",
            m->_mean_of_means, nullptr, nullptr);
  return m;
}

const std::vector<float>*
Ltx25VaeEncoderRef::tap(const std::string& name) const
{
  for (const auto& t : _taps) {
    if (t.first == name) { return &t.second; }
  }
  return nullptr;
}

bool
Ltx25VaeEncoderRef::tap_shape(const std::string& name,
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
Ltx25VaeEncoderRef::encode(const float* pixels, int F, int H, int W,
                           std::vector<float>* out,
                           std::array<int, 4>* shape, std::string* err)
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
    return fail("this VAE compresses space by " + std::to_string(sf) +
                "; " + std::to_string(H) + "x" + std::to_string(W) +
                " is not a multiple");
  }
  _taps.clear();
  _tap_shapes.clear();

  auto capture = [&](const std::string& name, const Vol& v) {
    if (!_capture) { return; }
    _taps.emplace_back(name, v.v);
    _tap_shapes.emplace_back(name, std::array<int, 4>{v.c, v.f, v.h, v.w});
  };

  Vol px;
  px.resize(_cfg.out_channels, F, H, W);
  std::memcpy(px.v.data(), pixels, px.v.size() * sizeof(float));

  Vol x;
  patchify_(px, _cfg.patch_size, x);
  Vol y;
  conv3d_causal_(x, _conv_in.w, _conv_in.b, _conv_in.cout, y);
  x = std::move(y);
  capture("conv_in", x);

  for (std::size_t i = 0; i < _downs.size(); ++i) {
    const DownBlock& dn = _downs[i];
    if (dn.is_res) {
      for (const ResBlock& rb : dn.res) {
        Vol h = x;
        pixel_norm_(h);
        silu_(h);
        Vol t;
        conv3d_causal_(h, rb.conv1.w, rb.conv1.b, rb.conv1.cout, t);
        h = std::move(t);
        pixel_norm_(h);
        silu_(h);
        conv3d_causal_(h, rb.conv2.w, rb.conv2.b, rb.conv2.cout, t);
        for (std::size_t k = 0; k < x.v.size(); ++k) { x.v[k] += t.v[k]; }
      }
    } else {
      // A time-halving block sees one extra frame: its own frame 0,
      // duplicated. That is what keeps 1 + 8k -> 1 + k exact.
      Vol in = x;
      if (dn.st == 2) {
        Vol dup;
        duplicate_first_frame_(x, dup);
        in = std::move(dup);
      }
      // The SKIP: space-to-depth of the input, then channel groups
      // averaged down to the block's output width. Not a projection --
      // a mean, which is why it has no weights to bind and is easy to
      // leave out entirely.
      Vol skip;
      space_to_depth_(in, dn.st, dn.sh, dn.sw, skip);
      mean_channel_groups_(skip, dn.group_size);
      // The CONV path: stride-1 conv, then the same regroup.
      Vol t;
      conv3d_causal_(in, dn.conv.w, dn.conv.b, dn.conv.cout, t);
      space_to_depth_(t, dn.st, dn.sh, dn.sw, x);
      for (std::size_t k = 0; k < x.v.size(); ++k) { x.v[k] += skip.v[k]; }
    }
    capture("down" + std::to_string(i), x);
  }

  pixel_norm_(x);
  silu_(x);
  conv3d_causal_(x, _conv_out.w, _conv_out.b, _conv_out.cout, y);
  capture("head", y);

  // 129 channels out: 128 means and ONE shared log-variance, which the
  // encoder's `uniform` mode broadcasts and the caller then throws away.
  // Keeping the means is a prefix, not a chunk of a doubled tensor.
  const int C = _cfg.latent_channels;
  if (y.c < C) {
    return fail("the head emitted " + std::to_string(y.c) +
                " channels, fewer than the " + std::to_string(C) +
                " latent channels");
  }
  Vol lat;
  lat.resize(C, y.f, y.h, y.w);
  const std::size_t plane = (std::size_t)y.f * y.h * y.w;
  std::memcpy(lat.v.data(), y.v.data(), (std::size_t)C * plane * sizeof(float));

  // WHITEN: (x - mean) / std, per channel -- the exact inverse of the
  // decoder's denormalise, and what makes this latent the one the DiT
  // was trained on.
  if ((int)_std_of_means.size() == C && (int)_mean_of_means.size() == C) {
    for (int ci = 0; ci < C; ++ci) {
      const float s = _std_of_means[(std::size_t)ci];
      const float m = _mean_of_means[(std::size_t)ci];
      const float inv = (s != 0.0f) ? 1.0f / s : 1.0f;
      for (std::size_t k = 0; k < plane; ++k) {
        lat.v[(std::size_t)ci * plane + k] =
            (lat.v[(std::size_t)ci * plane + k] - m) * inv;
      }
    }
  }

  *out = std::move(lat.v);
  if (shape != nullptr) { *shape = {lat.c, lat.f, lat.h, lat.w}; }
  return true;
}

}  // namespace ltx25
