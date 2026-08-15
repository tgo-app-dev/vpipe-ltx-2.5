#include "ltx25-vocoder.h"

#include "ltx25-config.h"

#include <algorithm>
#include <cstring>

using vpipe::FlexData;
using vpipe::genai::WeightSet;
using vpipe::metal_compute::CommandStream;
using vpipe::metal_compute::ComputeEncoder;
using vpipe::metal_compute::MetalCompute;
using vpipe::metal_compute::SharedBuffer;

namespace ltx25 {

int
VocoderConfig::hop() const
{
  int h = 1;
  for (int r : upsample_rates) { h *= r; }
  return h;
}

namespace {

std::vector<int>
int_array_(const FlexData& fd, const std::vector<int>& dflt)
{
  if (!fd.is_array()) { return dflt; }
  const auto a = fd.as_array();
  std::vector<int> v;
  for (std::size_t i = 0; i < a.size(); ++i) {
    v.push_back((int)a.at(i).as_int(0));
  }
  return v.empty() ? dflt : v;
}

}  // namespace

bool
parse_vocoder_config(const FlexData& cfg, const std::string& which,
                     VocoderConfig& out, std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  if (!cfg.is_object()) { return fail("the vocoder config is not an object"); }
  // config -> vocoder -> <which>, with any wrapper possibly peeled.
  // `parent` is the level holding BOTH `vocoder` and `bwe`, which is
  // where the sample rates live.
  // as_object() returns a VIEW into the FlexData it was called on, so
  // every intermediate is bound to a NAMED local. Chaining
  // `a.as_object().at(k).as_object()` reads through a destroyed
  // temporary and throws bad_variant_access.
  FlexData parent = cfg;
  {
    const auto top = parent.as_object();
    if (top.contains("vocoder")) {
      const FlexData inner = top.at("vocoder");
      if (inner.is_object()) {
        const auto io = inner.as_object();
        if (io.contains("vocoder") || io.contains("bwe")) { parent = inner; }
      }
    }
  }
  FlexData cur = parent;
  if (cur.is_object()) {
    const auto po = cur.as_object();
    if (po.contains(which)) { cur = po.at(which); }
  }
  if (!cur.is_object()) {
    return fail("no `" + which + "` section in the vocoder config");
  }
  const auto v = cur.as_object();
  auto geti = [&](const char* k, int d) {
    return v.contains(k) ? (int)v.at(k).as_int(d) : d;
  };
  auto getb = [&](const char* k, bool d) {
    return v.contains(k) ? v.at(k).as_bool(d) : d;
  };
  auto gets = [&](const char* k, const char* d) {
    return std::string(v.contains(k) ? v.at(k).as_string(d) : d);
  };

  out.upsample_initial_channel = geti("upsample_initial_channel", 1024);
  out.resblock = gets("resblock", "1");
  out.activation = gets("activation", "snake");
  out.use_tanh_at_final = getb("use_tanh_at_final", true);
  out.use_bias_at_final = getb("use_bias_at_final", true);
  out.output_sampling_rate = geti("output_sampling_rate", 24000);
  // The MAIN vocoder's own section carries no sample rate: the
  // reference gives it the BWE's INPUT rate, because the main generator
  // feeds the bandwidth extender. Left at the 24 kHz default its audio
  // is correct but plays 1.5x fast, which no rel-L2 on the samples
  // would ever catch.
  if (which == "vocoder" && parent.is_object()) {
    const auto po = parent.as_object();
    if (po.contains("bwe")) {
      // Bound to a NAMED local: `po.at("bwe").as_object()` would be a
      // view into a temporary that dies at the end of the statement.
      const FlexData bwe = po.at("bwe");
      if (bwe.is_object()) {
        const auto b = bwe.as_object();
        if (b.contains("input_sampling_rate")) {
          out.output_sampling_rate = (int)b.at("input_sampling_rate")
                                         .as_int(out.output_sampling_rate);
        }
      }
    }
  }
  if (v.contains("upsample_rates")) {
    out.upsample_rates = int_array_(v.at("upsample_rates"), out.upsample_rates);
  }
  if (v.contains("upsample_kernel_sizes")) {
    out.upsample_kernel_sizes =
        int_array_(v.at("upsample_kernel_sizes"), out.upsample_kernel_sizes);
  }
  if (v.contains("resblock_kernel_sizes")) {
    out.resblock_kernel_sizes =
        int_array_(v.at("resblock_kernel_sizes"), out.resblock_kernel_sizes);
  }
  out.resblock_dilation_sizes.clear();
  if (v.contains("resblock_dilation_sizes")) {
    const FlexData& d = v.at("resblock_dilation_sizes");
    if (d.is_array()) {
      const auto a = d.as_array();
      for (std::size_t i = 0; i < a.size(); ++i) {
        out.resblock_dilation_sizes.push_back(int_array_(a.at(i), {1, 3, 5}));
      }
    }
  }
  if (out.resblock_dilation_sizes.empty()) {
    out.resblock_dilation_sizes.assign(out.resblock_kernel_sizes.size(),
                                       {1, 3, 5});
  }

  if (out.upsample_rates.size() != out.upsample_kernel_sizes.size()) {
    return fail("upsample_rates and upsample_kernel_sizes disagree in length");
  }
  if (out.resblock_kernel_sizes.size() != out.resblock_dilation_sizes.size()) {
    return fail("resblock_kernel_sizes and resblock_dilation_sizes disagree");
  }
  // What this port implements, as a refusal rather than a wrong sound.
  if (out.resblock != "AMP1") {
    return fail("vocoder resblock '" + out.resblock +
                "' is not supported; this port implements AMP1 (BigVGAN)");
  }
  if (out.activation != "snakebeta") {
    return fail("vocoder activation '" + out.activation +
                "' is not supported; this port implements snakebeta");
  }
  return true;
}

// --------------------------------------------------------------------
// Loading
// --------------------------------------------------------------------

SharedBuffer
Ltx25Vocoder::alloc_(std::size_t n) const
{
  return _mc->make_shared_buffer(n * sizeof(float));
}

SharedBuffer
Ltx25Vocoder::f32_(const std::string& name, std::string* err)
{
  const auto* info = _ws->src().info(name);
  if (info == nullptr) {
    if (err != nullptr) { *err = "missing tensor " + name; }
    return SharedBuffer();
  }
  std::size_t n = 1;
  for (std::int64_t d : info->shape) { n *= (std::size_t)d; }
  const std::string dtype = info->dtype;
  // derived(), not tensor(): the vocoder KEEPS a TRANSFORM of the
  // checkpoint bytes (bf16 widened to f32), and the key names both the
  // transform and the source. Caching the bf16 original as well would
  // hold a redundant copy next to the product.
  SharedBuffer out = _ws->derived(
      "ltx25-voc/f32/" + name,
      [&]() -> SharedBuffer {
        // read(), uncached: these bytes are CONSUMED by the widening.
        const SharedBuffer src =
            _ws->read(name, _mc, WeightSet::Residency::Copied);
        if (src.empty()) { return SharedBuffer(); }
        SharedBuffer dst = _mc->make_shared_buffer(n * sizeof(float));
        if (dst.empty()) { return SharedBuffer(); }
        float* d = static_cast<float*>(dst.contents());
        if (dtype == "F32") {
          std::memcpy(d, src.contents(), n * sizeof(float));
        } else if (dtype == "BF16") {
          const auto* p = static_cast<const std::uint16_t*>(src.contents());
          for (std::size_t i = 0; i < n; ++i) {
            const std::uint32_t u = (std::uint32_t)p[i] << 16;
            std::memcpy(&d[i], &u, 4);
          }
        } else {
          return SharedBuffer();      // F16 etc: not what these packs use
        }
        return dst;
      });
  if (out.empty() && err != nullptr) {
    *err = "could not widen " + name + " (dtype " + dtype + ")";
  }
  if (!out.empty()) { _resident += out.byte_size(); }
  return out;
}

bool
Ltx25Vocoder::load_conv_(const std::string& p, int k, int dil, int pad,
                         bool bias, Conv& out, std::string* err)
{
  const auto* info = _ws->src().info(p + ".weight");
  if (info == nullptr || info->shape.size() != 3) {
    if (err != nullptr) { *err = p + ".weight is not a Conv1d weight"; }
    return false;
  }
  out.cout = (int)info->shape[0];
  out.cin = (int)info->shape[1];
  out.k = (int)info->shape[2];
  out.dil = dil;
  out.pad = pad;
  if (k != 0 && out.k != k) {
    if (err != nullptr) {
      *err = p + " has kernel " + std::to_string(out.k) + ", expected " +
             std::to_string(k);
    }
    return false;
  }
  out.w = f32_(p + ".weight", err);
  if (out.w.empty()) { return false; }
  if (bias) {
    out.b = f32_(p + ".bias", err);
    if (out.b.empty()) { return false; }
  } else {
    // No bias in the checkpoint (use_bias_at_final=false). Bind ZEROS
    // rather than leaving the argument unbound -- an unbound buffer is
    // undefined, and the kernel adds unconditionally.
    out.b = alloc_((std::size_t)out.cout);
    if (out.b.empty()) { return false; }
    std::memset(out.b.contents(), 0, (std::size_t)out.cout * sizeof(float));
  }
  out.has_bias = true;
  return true;
}

bool
Ltx25Vocoder::load_convt_(const std::string& p, int k, int stride, int pad,
                          ConvT& out, std::string* err)
{
  const auto* info = _ws->src().info(p + ".weight");
  if (info == nullptr || info->shape.size() != 3) {
    if (err != nullptr) {
      *err = p + ".weight is not a ConvTranspose1d weight";
    }
    return false;
  }
  // PyTorch stores ConvTranspose1d as [C_IN][C_OUT][K] -- the OPPOSITE
  // of Conv1d. Reading it the Conv1d way binds a correctly-sized weight
  // that mixes the channels up.
  out.cin = (int)info->shape[0];
  out.cout = (int)info->shape[1];
  out.k = (int)info->shape[2];
  out.stride = stride;
  out.pad = pad;
  if (out.k != k) {
    if (err != nullptr) {
      *err = p + " has kernel " + std::to_string(out.k) + ", expected " +
             std::to_string(k);
    }
    return false;
  }
  out.w = f32_(p + ".weight", err);
  out.b = f32_(p + ".bias", err);
  return !out.w.empty() && !out.b.empty();
}

bool
Ltx25Vocoder::load_act_(const std::string& p, Act& out, std::string* err)
{
  const auto* a = _ws->src().info(p + ".act.alpha");
  if (a == nullptr || a->shape.empty()) {
    if (err != nullptr) { *err = "missing " + p + ".act.alpha"; }
    return false;
  }
  out.c = (int)a->shape[0];
  out.alpha = f32_(p + ".act.alpha", err);
  out.beta = f32_(p + ".act.beta", err);
  // The kaiser-sinc filters come FROM THE CHECKPOINT. Recomputing them
  // would mean reproducing the beta/cutoff/half-width conventions
  // exactly, and any drift is a filter that still looks like a lowpass.
  out.up_filt = f32_(p + ".upsample.filter", err);
  out.down_filt = f32_(p + ".downsample.lowpass.filter", err);
  if (out.alpha.empty() || out.beta.empty() || out.up_filt.empty() ||
      out.down_filt.empty()) {
    return false;
  }
  const auto* uf = _ws->src().info(p + ".upsample.filter");
  const auto* df = _ws->src().info(p + ".downsample.lowpass.filter");
  out.kup = (int)uf->shape[uf->shape.size() - 1];
  out.kdown = (int)df->shape[df->shape.size() - 1];
  return true;
}

bool
Ltx25Vocoder::init_ops_(std::string* err)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  _lib = _mc->load_library(kMetalLibF32);
  if (!_lib.valid()) {
    return fail(std::string("no '") + kMetalLibF32 +
                "' library -- the plugin's f32 kernels were not registered");
  }
  _fn_mel_in = _lib.function("ltx_voc_mel_in");
  _fn_im2col = _lib.function("ltx_voc_im2col");
  _fn_gemm   = _lib.function("ltx_voc_gemm");
  _fn_convt  = _lib.function("ltx_voc_convt");
  _fn_up2x   = _lib.function("ltx_voc_up2x");
  _fn_down2x = _lib.function("ltx_voc_down2x");
  _fn_snake  = _lib.function("ltx_voc_snakebeta");
  _fn_mean3  = _lib.function("ltx_voc_mean3");
  _fn_out    = _lib.function("ltx_voc_out");
  // VPIPE_ELT is float in this library, so the video decoder's add is
  // the f32 add here.
  _fn_add    = _lib.function("ltx_vae_add_into");
  _fn_copy   = _lib.function("ltx_copy");

  struct { const vpipe::metal_compute::ComputeFunction* f; const char* n; }
  need[] = {
      {&_fn_mel_in, "ltx_voc_mel_in"}, {&_fn_im2col, "ltx_voc_im2col"},
      {&_fn_gemm, "ltx_voc_gemm"},     {&_fn_convt, "ltx_voc_convt"},
      {&_fn_up2x, "ltx_voc_up2x"},     {&_fn_down2x, "ltx_voc_down2x"},
      {&_fn_snake, "ltx_voc_snakebeta"}, {&_fn_mean3, "ltx_voc_mean3"},
      {&_fn_out, "ltx_voc_out"},       {&_fn_add, "ltx_vae_add_into"},
      {&_fn_copy, "ltx_copy"},
  };
  for (const auto& e : need) {
    if (!e.f->valid()) {
      return fail(std::string("kernel '") + e.n + "' did not resolve");
    }
  }
  return true;
}

std::unique_ptr<Ltx25Vocoder>
Ltx25Vocoder::load(const VocoderConfig& cfg, std::shared_ptr<WeightSet> ws,
                   MetalCompute* mc, const std::string& prefix,
                   std::string* err)
{
  if (mc == nullptr || !mc->valid()) {
    if (err != nullptr) { *err = "no usable Metal device"; }
    return nullptr;
  }
  std::unique_ptr<Ltx25Vocoder> v(new Ltx25Vocoder());
  v->_cfg = cfg;
  v->_ws = std::move(ws);
  v->_mc = mc;
  v->_prefix = prefix;
  if (!v->init_ops_(err)) { return nullptr; }

  const std::string P = prefix;
  // conv_pre is k=7 pad=3; conv_post the same, and WITHOUT bias when
  // use_bias_at_final is false (these checkpoints carry only a weight).
  if (!v->load_conv_(P + "conv_pre", 7, 1, 3, true, v->_conv_pre, err) ||
      !v->load_conv_(P + "conv_post", 7, 1, 3, cfg.use_bias_at_final,
                     v->_conv_post, err)) {
    return nullptr;
  }
  for (int i = 0; i < cfg.num_upsamples(); ++i) {
    const int k = cfg.upsample_kernel_sizes[(std::size_t)i];
    const int s = cfg.upsample_rates[(std::size_t)i];
    ConvT t;
    if (!v->load_convt_(P + "ups." + std::to_string(i), k, s, (k - s) / 2, t,
                        err)) {
      return nullptr;
    }
    v->_ups.push_back(std::move(t));
  }
  for (int i = 0; i < cfg.num_upsamples(); ++i) {
    for (int j = 0; j < cfg.num_kernels(); ++j) {
      const int idx = i * cfg.num_kernels() + j;
      const int k = cfg.resblock_kernel_sizes[(std::size_t)j];
      const std::vector<int>& dil = cfg.resblock_dilation_sizes[(std::size_t)j];
      const std::string bp = P + "resblocks." + std::to_string(idx);
      AmpBlock blk;
      for (std::size_t d = 0; d < dil.size(); ++d) {
        Conv c1, c2;
        Act a1, a2;
        // get_padding(k, dilation) = dilation*(k-1)/2; convs2 always
        // run at dilation 1.
        if (!v->load_conv_(bp + ".convs1." + std::to_string(d), k, dil[d],
                           dil[d] * (k - 1) / 2, true, c1, err) ||
            !v->load_conv_(bp + ".convs2." + std::to_string(d), k, 1,
                           (k - 1) / 2, true, c2, err) ||
            !v->load_act_(bp + ".acts1." + std::to_string(d), a1, err) ||
            !v->load_act_(bp + ".acts2." + std::to_string(d), a2, err)) {
          return nullptr;
        }
        blk.convs1.push_back(std::move(c1));
        blk.convs2.push_back(std::move(c2));
        blk.acts1.push_back(std::move(a1));
        blk.acts2.push_back(std::move(a2));
      }
      v->_resblocks.push_back(std::move(blk));
    }
  }
  if (!v->load_act_(P + "act_post", v->_act_post, err)) { return nullptr; }
  return v;
}

// --------------------------------------------------------------------
// Forward
// --------------------------------------------------------------------

void
Ltx25Vocoder::conv1d_(Stream& s, const Conv& c, const SharedBuffer& x, int T,
                      const SharedBuffer& y)
{
  const std::size_t k = (std::size_t)c.cin * (std::size_t)c.k;
  int chunk = T;
  if (k > 0) {
    const std::size_t rows = std::max<std::size_t>(1, _max_im2col / k);
    chunk = (int)std::min<std::size_t>((std::size_t)T, rows);
  }
  const std::size_t need = (std::size_t)chunk * k;
  if (_im2col.empty() || _im2col.byte_size() < need * sizeof(float)) {
    _im2col = alloc_(need);
  }
  for (int t0 = 0; t0 < T; t0 += chunk) {
    const int n = std::min(chunk, T - t0);
    auto enc = s.begin_compute();
    enc.set_function(_fn_im2col);
    enc.set_buffer(0, x);
    enc.set_buffer(1, _im2col);
    enc.set_constant(2, c.cin);
    enc.set_constant(3, T);
    enc.set_constant(4, c.k);
    enc.set_constant(5, c.dil);
    enc.set_constant(6, c.pad);
    enc.set_constant(7, t0);
    enc.set_constant(8, n);
    enc.dispatch({(unsigned)c.cin, (unsigned)n, 1}, {32, 1, 1});

    const unsigned gx = (unsigned)((c.cout + 15) / 16 * 16);
    const unsigned gy = (unsigned)((n + 15) / 16 * 16);
    enc.set_function(_fn_gemm);
    enc.set_buffer(0, _im2col);
    enc.set_buffer(1, c.w);
    enc.set_buffer(2, c.b);
    enc.set_buffer(3, y.subview((std::size_t)t0 * (std::size_t)c.cout
                                    * sizeof(float),
                                (std::size_t)n * (std::size_t)c.cout
                                    * sizeof(float)));
    enc.set_constant(4, n);
    enc.set_constant(5, c.cout);
    enc.set_constant(6, (int)k);
    enc.set_constant(7, 1);
    enc.dispatch({gx, gy, 1}, {16, 16, 1});
    enc.end();
  }
}

void
Ltx25Vocoder::convt_(Stream& s, const ConvT& c, const SharedBuffer& x, int T,
                     const SharedBuffer& y)
{
  const int To = (T - 1) * c.stride - 2 * c.pad + c.k;
  auto enc = s.begin_compute();
  enc.set_function(_fn_convt);
  enc.set_buffer(0, x);
  enc.set_buffer(1, c.w);
  enc.set_buffer(2, c.b);
  enc.set_buffer(3, y);
  enc.set_constant(4, c.cin);
  enc.set_constant(5, c.cout);
  enc.set_constant(6, T);
  enc.set_constant(7, c.k);
  enc.set_constant(8, c.stride);
  enc.set_constant(9, c.pad);
  enc.dispatch({(unsigned)c.cout, (unsigned)To, 1}, {32, 1, 1});
  enc.end();
}

SharedBuffer
Ltx25Vocoder::activation_(Stream& s, const Act& a, const SharedBuffer& x,
                          int C, int T, const char* tag)
{
  // UpSample1d for ratio 2: pad = K/2 - 1, and the crop offset is
  // pad*2 + (K - 2)/2. Both come from the reference's arithmetic, not
  // from the filter values, so they are computed rather than stored.
  const int pad = a.kup / 2 - 1;
  const int pad_left = pad * 2 + (a.kup - 2) / 2;
  SharedBuffer up = alloc_((std::size_t)2 * T * C);
  SharedBuffer down = alloc_((std::size_t)T * C);
  auto enc = s.begin_compute();
  enc.set_function(_fn_up2x);
  enc.set_buffer(0, x);
  enc.set_buffer(1, a.up_filt);
  enc.set_buffer(2, up);
  enc.set_constant(3, C);
  enc.set_constant(4, T);
  enc.set_constant(5, a.kup);
  enc.set_constant(6, pad);
  enc.set_constant(7, pad_left);
  enc.dispatch({(unsigned)C, (unsigned)(2 * T), 1}, {32, 1, 1});
  enc.end();
  if (tag != nullptr) { grab_(s, std::string(tag) + "_up", up, C, 2 * T); }
  enc = s.begin_compute();

  enc.set_function(_fn_snake);
  enc.set_buffer(0, up);
  enc.set_buffer(1, a.alpha);
  enc.set_buffer(2, a.beta);
  enc.set_constant(3, C);
  enc.set_constant(4, 2 * T);
  enc.dispatch({(unsigned)C, (unsigned)(2 * T), 1}, {32, 1, 1});
  enc.end();
  if (tag != nullptr) {
    grab_(s, std::string(tag) + "_snake", up, C, 2 * T);
  }
  enc = s.begin_compute();

  // LowPassFilter1d with an EVEN kernel pads (K/2 - 1) on the left.
  const int dpad_left = a.kdown / 2 - 1;
  enc.set_function(_fn_down2x);
  enc.set_buffer(0, up);
  enc.set_buffer(1, a.down_filt);
  enc.set_buffer(2, down);
  enc.set_constant(3, C);
  enc.set_constant(4, 2 * T);
  enc.set_constant(5, a.kdown);
  enc.set_constant(6, dpad_left);
  enc.dispatch({(unsigned)C, (unsigned)T, 1}, {32, 1, 1});
  enc.end();
  return down;
}

SharedBuffer
Ltx25Vocoder::amp_(Stream& s, const AmpBlock& blk, const SharedBuffer& x,
                   int C, int T)
{
  const std::size_t n = (std::size_t)T * (std::size_t)C;
  // A COPY of the input, not the input: the block adds its residual in
  // place, and `x` is the same tensor the other two resblocks of this
  // stage are about to read (they all run on the SAME input).
  SharedBuffer cur = alloc_(n);
  {
    auto enc = s.begin_compute();
    enc.set_function(_fn_copy);
    enc.set_buffer(0, x);
    enc.set_buffer(1, cur);
    enc.set_constant(2, (int)n);
    enc.dispatch({(unsigned)n, 1, 1}, {64, 1, 1});
    enc.end();
  }
  for (std::size_t d = 0; d < blk.convs1.size(); ++d) {
    SharedBuffer h = activation_(s, blk.acts1[d], cur, C, T);
    SharedBuffer t1 = alloc_(n);
    conv1d_(s, blk.convs1[d], h, T, t1);
    SharedBuffer h2 = activation_(s, blk.acts2[d], t1, C, T);
    SharedBuffer t2 = alloc_(n);
    conv1d_(s, blk.convs2[d], h2, T, t2);
    auto enc = s.begin_compute();
    enc.set_function(_fn_add);            // cur += t2
    enc.set_buffer(0, cur);
    enc.set_buffer(1, t2);
    enc.set_constant(2, (int)n);
    enc.dispatch({(unsigned)n, 1, 1}, {64, 1, 1});
    enc.end();
  }
  return cur;
}

void
Ltx25Vocoder::grab_(Stream& s, const std::string& name, const SharedBuffer& x,
                    int C, int T)
{
  if (!_capture) { return; }
  Tap t;
  t.name = name;
  t.shape = {C, T};
  t.buf = alloc_((std::size_t)C * T);
  if (t.buf.empty()) { return; }
  auto enc = s.begin_compute();
  enc.set_function(_fn_out);
  enc.set_buffer(0, x);
  enc.set_buffer(1, t.buf);
  enc.set_constant(2, C);
  enc.set_constant(3, T);
  enc.set_constant(4, 0);               // taps are NOT clamped
  enc.dispatch({(unsigned)C, (unsigned)T, 1}, {32, 1, 1});
  enc.end();
  _taps.push_back(std::move(t));
}

bool
Ltx25Vocoder::probe_activation(const float* x, int C, int T,
                               std::vector<float>* up,
                               std::vector<float>* snake,
                               std::vector<float>* down, std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  if (_resblocks.empty()) { return fail("no resblocks are loaded"); }
  const Act& a = _resblocks[0].acts1[0];
  if (C != a.c) {
    return fail("this activation has " + std::to_string(a.c) +
                " channels, not " + std::to_string(C));
  }
  _taps.clear();
  const bool was = _capture;
  _capture = true;

  SharedBuffer d_x = alloc_((std::size_t)T * C);
  if (d_x.empty()) { return fail("could not allocate the probe input"); }
  // The caller hands CHANNEL-FIRST [C][T], as the golden is stored.
  float* p = static_cast<float*>(d_x.contents());
  for (int t = 0; t < T; ++t) {
    for (int c = 0; c < C; ++c) {
      p[(std::size_t)t * C + c] = x[(std::size_t)c * T + t];
    }
  }
  CommandStream s = _mc->make_command_stream();
  SharedBuffer d = activation_(s, a, d_x, C, T, "act");
  grab_(s, "act_out", d, C, T);
  s.commit().wait();
  for (Tap& t : _taps) {
    const std::size_t n = (std::size_t)t.shape[0] * t.shape[1];
    t.host.resize(n);
    std::memcpy(t.host.data(), t.buf.contents(), n * sizeof(float));
  }
  _capture = was;
  const std::vector<float>* u = tap("act_up");
  const std::vector<float>* sn = tap("act_snake");
  const std::vector<float>* o = tap("act_out");
  if (u == nullptr || sn == nullptr || o == nullptr) {
    return fail("the activation probe captured nothing");
  }
  if (up != nullptr) { *up = *u; }
  if (snake != nullptr) { *snake = *sn; }
  if (down != nullptr) { *down = *o; }
  return true;
}

const std::vector<float>*
Ltx25Vocoder::tap(const std::string& name) const
{
  for (const Tap& t : _taps) {
    if (t.name == name) { return &t.host; }
  }
  return nullptr;
}

bool
Ltx25Vocoder::tap_shape(const std::string& name, std::array<int, 2>* out) const
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
Ltx25Vocoder::synthesize(const float* mel, int T, int mel_bins,
                         std::vector<float>* out, std::array<int, 2>* shape,
                         std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  if (mel == nullptr || out == nullptr) { return fail("null argument"); }
  if (T <= 0 || mel_bins <= 0) { return fail("degenerate mel shape"); }
  if (2 * mel_bins != _conv_pre.cin) {
    return fail("conv_pre wants " + std::to_string(_conv_pre.cin) +
                " channels but the mel gives 2 x " + std::to_string(mel_bins));
  }
  _taps.clear();

  const std::size_t nmel = (std::size_t)2 * T * mel_bins;
  SharedBuffer d_mel = alloc_(nmel);
  if (d_mel.empty()) { return fail("could not upload the mel"); }
  std::memcpy(d_mel.contents(), mel, nmel * sizeof(float));

  CommandStream s = _mc->make_command_stream();

  int C = _conv_pre.cin, len = T;
  SharedBuffer x = alloc_((std::size_t)len * C);
  {
    auto enc = s.begin_compute();
    enc.set_function(_fn_mel_in);
    enc.set_buffer(0, d_mel);
    enc.set_buffer(1, x);
    enc.set_constant(2, T);
    enc.set_constant(3, mel_bins);
    enc.dispatch({(unsigned)C, (unsigned)T, 1}, {32, 1, 1});
    enc.end();
  }
  SharedBuffer y = alloc_((std::size_t)len * _conv_pre.cout);
  conv1d_(s, _conv_pre, x, len, y);
  x = std::move(y);
  C = _conv_pre.cout;
  grab_(s, "conv_pre", x, C, len);

  for (int i = 0; i < _cfg.num_upsamples(); ++i) {
    const ConvT& u = _ups[(std::size_t)i];
    const int To = (len - 1) * u.stride - 2 * u.pad + u.k;
    SharedBuffer up = alloc_((std::size_t)To * u.cout);
    convt_(s, u, x, len, up);
    x = std::move(up);
    len = To;
    C = u.cout;
    grab_(s, "ups" + std::to_string(i), x, C, len);

    // The THREE resblocks run on the SAME input and are MEAN-aggregated.
    // Chaining them is a working vocoder that is not this one.
    const int nk = _cfg.num_kernels();
    std::vector<SharedBuffer> outs;
    for (int j = 0; j < nk; ++j) {
      outs.push_back(
          amp_(s, _resblocks[(std::size_t)(i * nk + j)], x, C, len));
    }
    if (i == 0) { grab_(s, "rb0", outs[0], C, len); }
    SharedBuffer m = alloc_((std::size_t)len * C);
    {
      auto enc = s.begin_compute();
      enc.set_function(_fn_mean3);
      enc.set_buffer(0, m);
      enc.set_buffer(1, outs[0]);
      enc.set_buffer(2, outs[(std::size_t)std::min(1, nk - 1)]);
      enc.set_buffer(3, outs[(std::size_t)std::min(2, nk - 1)]);
      const int n = len * C;
      enc.set_constant(4, n);
      enc.dispatch({(unsigned)n, 1, 1}, {64, 1, 1});
      enc.end();
    }
    x = std::move(m);
  }

  SharedBuffer post = activation_(s, _act_post, x, C, len);
  grab_(s, "act_post", post, C, len);
  SharedBuffer wav = alloc_((std::size_t)len * _conv_post.cout);
  conv1d_(s, _conv_post, post, len, wav);

  const int oc = _conv_post.cout;
  SharedBuffer o = alloc_((std::size_t)oc * len);
  if (o.empty()) { return fail("could not allocate the waveform"); }
  {
    auto enc = s.begin_compute();
    enc.set_function(_fn_out);
    enc.set_buffer(0, wav);
    enc.set_buffer(1, o);
    enc.set_constant(2, oc);
    enc.set_constant(3, len);
    // use_tanh_at_final is false for these packs, so the final
    // activation is a CLAMP.
    const int docl = _cfg.apply_final_activation ? 1 : 0;
    enc.set_constant(4, docl);
    enc.dispatch({(unsigned)oc, (unsigned)len, 1}, {32, 1, 1});
    enc.end();
  }
  s.commit().wait();

  for (Tap& t : _taps) {
    const std::size_t tn = (std::size_t)t.shape[0] * t.shape[1];
    t.host.resize(tn);
    std::memcpy(t.host.data(), t.buf.contents(), tn * sizeof(float));
  }
  const std::size_t n = (std::size_t)oc * len;
  out->resize(n);
  std::memcpy(out->data(), o.contents(), n * sizeof(float));
  if (shape != nullptr) { *shape = {oc, len}; }
  return true;
}

}  // namespace ltx25
