#include "ltx25-bwe.h"

#include "ltx25-config.h"

#include <algorithm>
#include <cmath>
#include <cstring>

using vpipe::FlexData;
using vpipe::genai::WeightSet;
using vpipe::metal_compute::CommandStream;
using vpipe::metal_compute::MetalCompute;
using vpipe::metal_compute::SharedBuffer;

namespace ltx25 {

bool
parse_bwe_config(const FlexData& cfg, BweConfig& out, std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  if (!cfg.is_object()) { return fail("the vocoder config is not an object"); }
  // as_object() returns a VIEW, so every intermediate is a named local.
  FlexData parent = cfg;
  {
    const auto top = parent.as_object();
    if (top.contains("vocoder")) {
      const FlexData inner = top.at("vocoder");
      if (inner.is_object()) {
        const auto io = inner.as_object();
        if (io.contains("bwe")) { parent = inner; }
      }
    }
  }
  if (!parent.is_object()) { return fail("no vocoder section"); }
  const auto po = parent.as_object();
  if (!po.contains("bwe")) {
    return fail("this checkpoint has no `bwe` section");
  }
  const FlexData bwe = po.at("bwe");
  if (!bwe.is_object()) { return fail("`bwe` is not an object"); }
  const auto b = bwe.as_object();
  auto geti = [&](const char* k, int d) {
    return b.contains(k) ? (int)b.at(k).as_int(d) : d;
  };
  out.n_fft = geti("n_fft", 512);
  out.hop_length = geti("hop_length", 80);
  out.win_length = geti("win_size", out.n_fft);
  out.num_mels = geti("num_mels", 64);
  out.input_sampling_rate = geti("input_sampling_rate", 16000);
  out.output_sampling_rate = geti("output_sampling_rate", 48000);
  if (out.input_sampling_rate <= 0 ||
      out.output_sampling_rate % out.input_sampling_rate != 0) {
    return fail("the BWE sample rates are not an integer ratio");
  }
  return true;
}

namespace {

float
sinc_(double x)
{
  if (x == 0.0) { return 1.0f; }
  const double p = M_PI * x;
  return (float)(std::sin(p) / p);
}

}  // namespace

std::uint64_t
Ltx25Bwe::resident_bytes() const
{
  std::uint64_t n = 0;
  if (_voc) { n += _voc->resident_bytes(); }
  if (_gen) { n += _gen->resident_bytes(); }
  n += _fwd_basis.byte_size() + _mel_basis.byte_size();
  return n;
}

std::unique_ptr<Ltx25Bwe>
Ltx25Bwe::load(const FlexData& cfg, std::shared_ptr<WeightSet> ws,
               MetalCompute* mc, std::string* err)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return std::unique_ptr<Ltx25Bwe>();
  };
  if (mc == nullptr || !mc->valid()) { return fail("no usable Metal device"); }
  std::unique_ptr<Ltx25Bwe> b(new Ltx25Bwe());
  b->_ws = std::move(ws);
  b->_mc = mc;
  if (!parse_bwe_config(cfg, b->_bcfg, err)) { return nullptr; }
  if (!parse_vocoder_config(cfg, "vocoder", b->_vcfg, err) ||
      !parse_vocoder_config(cfg, "bwe", b->_gcfg, err)) {
    return nullptr;
  }
  // The BWE generator does NOT apply the final activation. This is a
  // CALL-SITE argument in the reference, not a config value -- its
  // `_vocoder_from_config` passes apply_final_activation=False and never
  // reads the key, so a checkpoint saying otherwise is still ignored.
  // The output is a residual, and clamping it before it is added would
  // clip a correction meant to be able to cancel the skip.
  b->_gcfg.apply_final_activation = false;

  b->_voc = Ltx25Vocoder::load(b->_vcfg, b->_ws, mc, "vocoder.vocoder.", err);
  if (!b->_voc) { return nullptr; }
  b->_gen = Ltx25Vocoder::load(b->_gcfg, b->_ws, mc, "vocoder.bwe_generator.",
                               err);
  if (!b->_gen) { return nullptr; }

  b->_lib = mc->load_library(kMetalLibF32);
  if (!b->_lib.valid()) {
    return fail(std::string("no '") + kMetalLibF32 + "' library");
  }
  b->_fn_stft = b->_lib.function("ltx_voc_stft_im2col");
  b->_fn_mag  = b->_lib.function("ltx_voc_magnitude");
  b->_fn_logc = b->_lib.function("ltx_voc_log_clamp");
  b->_fn_gemm = b->_lib.function("ltx_voc_gemm");
  b->_fn_upr  = b->_lib.function("ltx_voc_up_ratio");
  struct { const vpipe::metal_compute::ComputeFunction* f; const char* n; }
  need[] = {
      {&b->_fn_stft, "ltx_voc_stft_im2col"},
      {&b->_fn_mag, "ltx_voc_magnitude"},
      {&b->_fn_logc, "ltx_voc_log_clamp"},
      {&b->_fn_gemm, "ltx_voc_gemm"},
      {&b->_fn_upr, "ltx_voc_up_ratio"},
  };
  for (const auto& e : need) {
    if (!e.f->valid()) {
      return fail(std::string("kernel '") + e.n + "' did not resolve");
    }
  }

  // The STFT bases and the mel filterbank, widened to f32. The
  // reference loads the EXACT bases from training rather than
  // recomputing a DFT, so this reads them too.
  auto widen = [&](const char* name, Buf& dst, std::size_t* count) -> bool {
    const auto* info = b->_ws->src().info(name);
    if (info == nullptr) {
      if (err != nullptr) { *err = std::string("missing ") + name; }
      return false;
    }
    std::size_t n = 1;
    for (std::int64_t d : info->shape) { n *= (std::size_t)d; }
    const std::string dt = info->dtype;
    const SharedBuffer src =
        b->_ws->read(name, mc, WeightSet::Residency::Copied);
    if (src.empty()) {
      if (err != nullptr) { *err = std::string("could not read ") + name; }
      return false;
    }
    dst = mc->make_shared_buffer(n * sizeof(float));
    if (dst.empty()) { return false; }
    float* d = static_cast<float*>(dst.contents());
    if (dt == "F32") {
      std::memcpy(d, src.contents(), n * sizeof(float));
    } else if (dt == "BF16") {
      const auto* p = static_cast<const std::uint16_t*>(src.contents());
      for (std::size_t i = 0; i < n; ++i) {
        const std::uint32_t u = (std::uint32_t)p[i] << 16;
        std::memcpy(&d[i], &u, 4);
      }
    } else {
      if (err != nullptr) { *err = std::string(name) + " has dtype " + dt; }
      return false;
    }
    if (count != nullptr) { *count = n; }
    return true;
  };
  if (!widen("vocoder.mel_stft.stft_fn.forward_basis", b->_fwd_basis,
             nullptr) ||
      !widen("vocoder.mel_stft.mel_basis", b->_mel_basis, nullptr)) {
    return nullptr;
  }
  // The GEMM adds a bias unconditionally, so bind zeros for these two.
  const int nb = std::max(2 * b->_bcfg.n_freqs(), b->_bcfg.num_mels);
  b->_zero_bias = mc->make_shared_buffer((std::size_t)nb * sizeof(float));
  if (b->_zero_bias.empty()) { return fail("no room for the zero bias"); }
  std::memset(b->_zero_bias.contents(), 0, (std::size_t)nb * sizeof(float));

  // ---- the resampler's HANN-window sinc -----------------------------
  //
  // NOT in the checkpoint (persistent=False) and NOT the kaiser window
  // the activations use, so it is rebuilt here exactly as UpSample1d's
  // `window_type="hann"` branch does. Getting the window family wrong
  // still gives a lowpass and still gives audio.
  const int ratio = b->_bcfg.ratio();
  const double rolloff = 0.99;
  const int lpw = 6;
  const int width = (int)std::ceil((double)lpw / rolloff);
  b->_filt_k = 2 * width * ratio + 1;
  b->_filt_pad = width;
  b->_filt_pad_left = 2 * width * ratio;
  b->_filt_host.resize((std::size_t)b->_filt_k);
  for (int i = 0; i < b->_filt_k; ++i) {
    const double t = ((double)i / (double)ratio - (double)width) * rolloff;
    const double tc = std::max(-(double)lpw, std::min((double)lpw, t));
    const double w = std::pow(std::cos(tc * M_PI / (double)lpw / 2.0), 2.0);
    b->_filt_host[(std::size_t)i] =
        (float)((double)sinc_(t) * w * rolloff / (double)ratio);
  }
  b->_filt = mc->make_shared_buffer(b->_filt_host.size() * sizeof(float));
  if (b->_filt.empty()) { return fail("no room for the resampler filter"); }
  std::memcpy(b->_filt.contents(), b->_filt_host.data(),
              b->_filt_host.size() * sizeof(float));
  return b;
}

bool
Ltx25Bwe::mel_of_(const float* wav, int T, int frames,
                  std::vector<float>* out, std::string* err)
{
  const int win = _bcfg.win_length, hop = _bcfg.hop_length;
  const int nf = _bcfg.n_freqs(), nm = _bcfg.num_mels;
  // CAUSAL: win - hop samples of zero padding on the LEFT ONLY.
  const int left_pad = std::max(0, win - hop);

  SharedBuffer d_y = _mc->make_shared_buffer((std::size_t)T * sizeof(float));
  SharedBuffer col =
      _mc->make_shared_buffer((std::size_t)frames * win * sizeof(float));
  SharedBuffer spec =
      _mc->make_shared_buffer((std::size_t)frames * 2 * nf * sizeof(float));
  SharedBuffer mag =
      _mc->make_shared_buffer((std::size_t)frames * nf * sizeof(float));
  SharedBuffer mel =
      _mc->make_shared_buffer((std::size_t)frames * nm * sizeof(float));
  if (d_y.empty() || col.empty() || spec.empty() || mag.empty() ||
      mel.empty()) {
    if (err != nullptr) { *err = "could not allocate the mel scratch"; }
    return false;
  }
  std::memcpy(d_y.contents(), wav, (std::size_t)T * sizeof(float));

  CommandStream s = _mc->make_command_stream();
  {
    auto enc = s.begin_compute();
    enc.set_function(_fn_stft);
    enc.set_buffer(0, d_y);
    enc.set_buffer(1, col);
    enc.set_constant(2, T);
    enc.set_constant(3, win);
    enc.set_constant(4, hop);
    enc.set_constant(5, left_pad);
    enc.set_constant(6, frames);
    enc.dispatch({(unsigned)win, (unsigned)frames, 1}, {32, 1, 1});

    // spec[frames][2*nf] = col[frames][win] @ forward_basis[2*nf][win]^T
    const int N = 2 * nf;
    enc.set_function(_fn_gemm);
    enc.set_buffer(0, col);
    enc.set_buffer(1, _fwd_basis);
    enc.set_buffer(2, _zero_bias);
    enc.set_buffer(3, spec);
    enc.set_constant(4, frames);
    enc.set_constant(5, N);
    enc.set_constant(6, win);
    enc.set_constant(7, 0);
    enc.dispatch({(unsigned)((N + 15) / 16 * 16),
                  (unsigned)((frames + 15) / 16 * 16), 1}, {16, 16, 1});

    enc.set_function(_fn_mag);
    enc.set_buffer(0, spec);
    enc.set_buffer(1, mag);
    enc.set_constant(2, nf);
    enc.set_constant(3, frames);
    enc.dispatch({(unsigned)nf, (unsigned)frames, 1}, {32, 1, 1});

    // mel[frames][nm] = mag[frames][nf] @ mel_basis[nm][nf]^T
    enc.set_function(_fn_gemm);
    enc.set_buffer(0, mag);
    enc.set_buffer(1, _mel_basis);
    enc.set_buffer(2, _zero_bias);
    enc.set_buffer(3, mel);
    enc.set_constant(4, frames);
    enc.set_constant(5, nm);
    enc.set_constant(6, nf);
    enc.set_constant(7, 0);
    enc.dispatch({(unsigned)((nm + 15) / 16 * 16),
                  (unsigned)((frames + 15) / 16 * 16), 1}, {16, 16, 1});

    const int n = frames * nm;
    enc.set_function(_fn_logc);
    enc.set_buffer(0, mel);
    enc.set_constant(1, n);
    enc.dispatch({(unsigned)n, 1, 1}, {64, 1, 1});
    enc.end();
  }
  s.commit().wait();
  out->resize((std::size_t)frames * nm);
  std::memcpy(out->data(), mel.contents(),
              out->size() * sizeof(float));
  return true;
}

bool
Ltx25Bwe::resample_(const std::vector<float>& x, int T, int C,
                    std::vector<float>* out, std::string* err)
{
  const int ratio = _bcfg.ratio();
  SharedBuffer d_x =
      _mc->make_shared_buffer((std::size_t)T * C * sizeof(float));
  SharedBuffer d_o =
      _mc->make_shared_buffer((std::size_t)ratio * T * C * sizeof(float));
  if (d_x.empty() || d_o.empty()) {
    if (err != nullptr) { *err = "could not allocate the resampler scratch"; }
    return false;
  }
  // The kernel is channel-last, and `x` arrives channel-FIRST [C][T].
  float* p = static_cast<float*>(d_x.contents());
  for (int t = 0; t < T; ++t) {
    for (int c = 0; c < C; ++c) {
      p[(std::size_t)t * C + c] = x[(std::size_t)c * T + t];
    }
  }
  CommandStream s = _mc->make_command_stream();
  {
    auto enc = s.begin_compute();
    enc.set_function(_fn_upr);
    enc.set_buffer(0, d_x);
    enc.set_buffer(1, _filt);
    enc.set_buffer(2, d_o);
    enc.set_constant(3, C);
    enc.set_constant(4, T);
    enc.set_constant(5, _filt_k);
    enc.set_constant(6, _filt_pad);
    enc.set_constant(7, _filt_pad_left);
    enc.set_constant(8, ratio);
    enc.dispatch({(unsigned)C, (unsigned)(ratio * T), 1}, {32, 1, 1});
    enc.end();
  }
  s.commit().wait();
  const int To = ratio * T;
  out->resize((std::size_t)C * To);
  const float* q = static_cast<const float*>(d_o.contents());
  for (int c = 0; c < C; ++c) {
    for (int t = 0; t < To; ++t) {
      (*out)[(std::size_t)c * To + t] = q[(std::size_t)t * C + c];
    }
  }
  return true;
}

const std::vector<float>*
Ltx25Bwe::tap(const std::string& name) const
{
  for (const auto& t : _taps) {
    if (t.first == name) { return &t.second; }
  }
  return nullptr;
}

bool
Ltx25Bwe::synthesize(const float* mel, int T, int mel_bins,
                     std::vector<float>* out, std::array<int, 2>* shape,
                     std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  if (mel == nullptr || out == nullptr) { return fail("null argument"); }
  _taps.clear();

  // ---- the main vocoder, at 16 kHz -----------------------------------
  std::array<int, 2> lsh{};
  if (!_voc->synthesize(mel, T, mel_bins, &_low, &lsh, err)) { return false; }
  const int C = lsh[0];
  _low_len = lsh[1];
  const int ratio = _bcfg.ratio();
  // Computed from the UNPADDED length: the padding below exists only so
  // the mel comes out to a whole number of frames, and must not appear
  // in the output.
  const int out_len = _low_len * ratio;

  // Pad up to a whole number of hop-length frames. The SKIP is taken
  // from the padded signal too -- the reference reassigns x before
  // resampling -- so both paths see the same thing and the crop at the
  // end removes the extra.
  const int hop = _bcfg.hop_length;
  const int rem = (hop > 0) ? (_low_len % hop) : 0;
  const int padded = (rem != 0) ? (_low_len + hop - rem) : _low_len;
  std::vector<float> x((std::size_t)C * padded, 0.0f);
  for (int c = 0; c < C; ++c) {
    std::memcpy(&x[(std::size_t)c * padded], &_low[(std::size_t)c * _low_len],
                (std::size_t)_low_len * sizeof(float));
  }
  if (_capture) { _taps.emplace_back("low", x); }

  // ---- the causal log-mel of it --------------------------------------
  const int win = _bcfg.win_length;
  const int left_pad = std::max(0, win - hop);
  const int frames = (padded + left_pad - win) / hop + 1;
  if (frames <= 0) { return fail("the 16 kHz waveform is too short to mel"); }
  const int nm = _bcfg.num_mels;
  // The generator wants [2][frames][mel_bins], and mel_of_ produces one
  // channel's [frames][nm] -- which IS that layout per channel.
  std::vector<float> mel2((std::size_t)C * frames * nm);
  for (int c = 0; c < C; ++c) {
    std::vector<float> one;
    if (!mel_of_(&x[(std::size_t)c * padded], padded, frames, &one, err)) {
      return false;
    }
    std::memcpy(&mel2[(std::size_t)c * frames * nm], one.data(),
                one.size() * sizeof(float));
  }
  if (_capture) { _taps.emplace_back("mel2", mel2); }

  // ---- the residual and the skip -------------------------------------
  std::vector<float> residual;
  std::array<int, 2> rsh{};
  if (!_gen->synthesize(mel2.data(), frames, nm, &residual, &rsh, err)) {
    return false;
  }
  if (_capture) { _taps.emplace_back("residual", residual); }

  std::vector<float> skip;
  if (!resample_(x, padded, C, &skip, err)) { return false; }
  if (_capture) { _taps.emplace_back("skip", skip); }

  if (rsh[0] != C || (std::size_t)rsh[0] * rsh[1] != skip.size()) {
    return fail("the BWE residual is " + std::to_string(rsh[0]) + "x" +
                std::to_string(rsh[1]) + " but the skip is " +
                std::to_string(skip.size() / std::max(1, C)) + " long");
  }
  const int To = rsh[1];
  if (out_len > To) {
    return fail("the BWE produced " + std::to_string(To) +
                " samples, short of the " + std::to_string(out_len) +
                " the input calls for");
  }
  out->resize((std::size_t)C * out_len);
  for (int c = 0; c < C; ++c) {
    for (int t = 0; t < out_len; ++t) {
      const float v = residual[(std::size_t)c * To + t] +
                      skip[(std::size_t)c * To + t];
      (*out)[(std::size_t)c * out_len + t] = std::min(1.0f, std::max(-1.0f, v));
    }
  }
  if (shape != nullptr) { *shape = {C, out_len}; }
  return true;
}

}  // namespace ltx25
