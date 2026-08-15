#include "ltx25-vae-family.h"

#include "ltx25-config.h"
#include "ltx25-metal-ops.h"
#include "ltx25-vae.h"
#include "ltx25-vae-config.h"
#include "ltx25-audio-encoder.h"
#include "ltx25-audio-vae.h"
#include "ltx25-bwe.h"
#include "ltx25-vocoder.h"

#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "generative-models/shared/comfy-checkpoint.h"
#include "generative-models/weight-set.h"
#include "interfaces/session-context-intf.h"
#include "stages/model-memory.h"

#include <filesystem>
#include <utility>

using vpipe::FlexData;
using vpipe::ResourceClaim;
using vpipe::fmt;
using vpipe::genai::VaeDecodeRequest;
using vpipe::genai::VaeDecoder;
using vpipe::genai::VaeFrameChunk;
using vpipe::genai::VaeFrameSink;
using vpipe::genai::VaeModelCreateArgs;

namespace ltx25 {

namespace {

// One resident LTX conv video VAE.
//
// Owns its MetalOps because Ltx25VaeDecoder keeps a BORROWED pointer to
// one -- the ops object has to outlive the decoder, and the family is
// stateless, so this is the only place with the right lifetime.
class Ltx25VaeDecoderAdapter final : public VaeDecoder {
public:
  Ltx25VaeDecoderAdapter(VaeConfig cfg, std::unique_ptr<MetalOps> ops,
                         std::unique_ptr<Ltx25VaeDecoder> dec)
    : _cfg(std::move(cfg)), _ops(std::move(ops)), _dec(std::move(dec))
  {}

  int latent_channels() const override { return _cfg.latent_channels; }
  int spatial_compression() const override { return _cfg.spatial_factor(); }
  int temporal_compression() const override { return _cfg.temporal_factor(); }

  std::uint64_t resident_bytes() const override
  {
    return _dec ? _dec->resident_bytes() : 0;
  }

  bool decode(const VaeDecodeRequest& req, const VaeFrameSink& sink,
              std::string* err) override
  {
    auto fail = [&](std::string m) {
      if (err != nullptr) { *err = std::move(m); }
      return false;
    };
    if (_dec == nullptr) { return fail("the decoder is not loaded"); }
    if (req.latent == nullptr) { return fail("no latent"); }
    if (req.shape.size() != 4) {
      return fail("this is a VIDEO VAE and wants a [z, T, H/32, W/32] "
                  "latent; got a " + std::to_string(req.shape.size()) +
                  "-D tensor");
    }
    const int T = req.shape[1], lh = req.shape[2], lw = req.shape[3];
    if (T <= 0 || lh <= 0 || lw <= 0) {
      return fail("degenerate latent geometry");
    }

    std::vector<float> pix;
    std::array<int, 4> shape{};
    if (!_dec->decode(req.latent, T, lh, lw, &pix, &shape, err)) {
      return false;
    }
    // ONE chunk. Ltx25VaeDecoder returns the whole clip -- the sink's
    // chunking is permitted, never required -- and its layout is already
    // what VaeFrameChunk wants: f32 channel-first [C][F][H][W] in
    // [-1, 1], so nothing is copied or converted here.
    VaeFrameChunk c;
    c.rgb          = pix.data();
    c.channels     = shape[0];
    c.frame0       = 0;
    c.n            = shape[1];
    c.height       = shape[2];
    c.width        = shape[3];
    c.frames_total = shape[1];
    if (!sink(c)) { return fail("the decode was cancelled"); }
    return true;
  }

private:
  VaeConfig _cfg;
  std::unique_ptr<MetalOps> _ops;
  std::unique_ptr<Ltx25VaeDecoder> _dec;
};

// resolve() + the VAE's own config, shared by claims() and load_decoder().
// `err` is null on the claims() path, where a miss is an answer rather
// than a fault.
bool
resolve_vae_(const std::string& root, Config& cfg, VaeConfig& vcfg,
             std::string* err)
{
  if (!resolve(root, cfg, err)) { return false; }
  if (cfg.vae_file.empty()) {
    if (err != nullptr) { *err = "no conv video VAE under '" + root + "'"; }
    return false;
  }
  FlexData meta;
  if (!vpipe::genai::comfy::metadata_json(cfg.vae_file, kVaeMetaKey, meta,
                                          err)) {
    return false;
  }
  return parse_vae_config(meta, vcfg, err);
}

}  // namespace

namespace {

// The whole audio chain behind one decode: latent -> log-mel -> 16 kHz
// -> 48 kHz.
//
// The BWE is OPTIONAL and its absence is not an error: a checkpoint
// without a `bwe` section simply plays at 16 kHz, and saying which rate
// came out is the stage's job (it stamps sample_rate from us). Silently
// labelling 16 kHz audio as 48 would play it a third too slow.
class Ltx25AudioDecoderAdapter final : public vpipe::genai::AudioVaeDecoder {
public:
  Ltx25AudioDecoderAdapter(AudioVaeConfig acfg, std::unique_ptr<MetalOps> ops,
                           std::unique_ptr<Ltx25AudioVaeDecoder> dec,
                           std::unique_ptr<Ltx25Bwe> bwe,
                           std::unique_ptr<Ltx25Vocoder> voc, int rate)
    : _acfg(std::move(acfg)), _ops(std::move(ops)), _dec(std::move(dec)),
      _bwe(std::move(bwe)), _voc(std::move(voc)), _rate(rate)
  {}

  int sample_rate() const override { return _rate; }

  std::uint64_t resident_bytes() const override
  {
    std::uint64_t n = _dec ? _dec->resident_bytes() : 0;
    if (_bwe) { n += _bwe->resident_bytes(); }
    else if (_voc) { n += _voc->resident_bytes(); }
    return n;
  }

  bool decode(const vpipe::genai::AudioVaeDecodeRequest& req,
              std::vector<float>* pcm, std::vector<int>* shape,
              std::string* err) override
  {
    auto fail = [&](std::string m) {
      if (err != nullptr) { *err = std::move(m); }
      return false;
    };
    if (_dec == nullptr) { return fail("the audio decoder is not loaded"); }
    if (req.latent == nullptr || req.shape.size() != 3) {
      return fail("this family wants a 3-D [channels, frames, mel] latent");
    }
    // [channels][frames][mel] -- the shape the LTX generator emits, and
    // the one Ltx25AudioVaeDecoder takes. Checked rather than assumed,
    // because MiniMax-H3's latent is also 3-D and means something else.
    const int C = req.shape[0], F = req.shape[1], M = req.shape[2];
    if (C != _acfg.z_channels || M != _acfg.latent_mel_bins()) {
      return fail("the latent is [" + std::to_string(C) + "," +
                  std::to_string(F) + "," + std::to_string(M) +
                  "] but this audio VAE wants [" +
                  std::to_string(_acfg.z_channels) + ",F," +
                  std::to_string(_acfg.latent_mel_bins()) + "]");
    }
    if (F <= 0) { return fail("no audio latent frames"); }

    std::vector<float> mel;
    std::array<int, 3> ms{};
    if (!_dec->decode(req.latent, F, &mel, &ms, err)) { return false; }
    if (req.progress && !req.progress(1, 2)) {
      return fail("the decode was cancelled");
    }
    std::array<int, 2> wsh{};
    if (_bwe) {
      if (!_bwe->synthesize(mel.data(), ms[1], ms[2], pcm, &wsh, err)) {
        return false;
      }
    } else if (_voc) {
      if (!_voc->synthesize(mel.data(), ms[1], ms[2], pcm, &wsh, err)) {
        return false;
      }
    } else {
      return fail("no vocoder");
    }
    // The stage clamps nothing; both generators already do.
    shape->assign({wsh[0], wsh[1]});
    return true;
  }

private:
  AudioVaeConfig _acfg;
  std::unique_ptr<MetalOps> _ops;
  std::unique_ptr<Ltx25AudioVaeDecoder> _dec;
  std::unique_ptr<Ltx25Bwe> _bwe;
  std::unique_ptr<Ltx25Vocoder> _voc;
  int _rate = 16000;
};

}  // namespace

std::string_view
Ltx25VaeFamily::tag() const noexcept
{
  return kFamily;
}

bool
Ltx25VaeFamily::claims(const std::string& root, const std::string& vae_dir,
                       const std::string& model_type) const
{
  (void)vae_dir;      // LTX has no vae/config.json, so this is the root
  (void)model_type;
  Config cfg;
  VaeConfig vcfg;
  if (!resolve_vae_(root, cfg, vcfg, nullptr)) { return false; }
  // SURE, not merely plausible: this probe runs before the host's own
  // `_class_name` chain, so a loose claim would shadow a working
  // built-in path. parse_vae_config already refuses a norm_layer or a
  // padding mode this decoder does not implement, and the class name is
  // the last discriminator.
  return vcfg.class_name == "CausalVideoAutoencoder";
}

std::vector<ResourceClaim>
Ltx25VaeFamily::declare_resources(const std::string& root,
                                  const std::string& vae_dir) const
{
  (void)vae_dir;
  Config cfg;
  VaeConfig vcfg;
  if (!resolve_vae_(root, cfg, vcfg, nullptr)) { return {}; }
  // The VAE FILE, not the root: the root also holds the 39 GB DiT, which
  // generate-video declares.
  return vpipe::model_memory::weight_claims({cfg.vae_file});
}

std::vector<std::string>
Ltx25VaeFamily::idle_peers(const std::string& root) const
{
  namespace fs = std::filesystem;
  // The COMFY spelling. The stage's built-in guess is the diffusers one
  // (transformer/, text_encoder/, mllm/), which sums to ZERO on this
  // pack -- and a zero peer footprint reads as "roomy, keep the VAE
  // resident" beside a 39 GB DiT, which is the wrong call on any box.
  return {(fs::path(root) / "diffusion_models").string(),
          (fs::path(root) / "text_encoders").string()};
}

std::unique_ptr<VaeDecoder>
Ltx25VaeFamily::load_decoder(const VaeModelCreateArgs& args)
{
  auto warn = [&](const std::string& m) {
    if (args.session != nullptr) {
      args.session->warn(fmt("ltx-2.5 VAE: {}", m));
    }
  };
  Config cfg;
  VaeConfig vcfg;
  std::string err;
  if (!resolve_vae_(args.root, cfg, vcfg, &err)) {
    warn(err);
    return nullptr;
  }
  if (args.metal == nullptr || !args.metal->valid()) {
    warn("no usable Metal device");
    return nullptr;
  }
  auto ops = std::make_unique<MetalOps>();
  if (!ops->init(args.metal, &err)) {
    warn(err);
    return nullptr;
  }
  // The set is opened on the VAE FILE. open_weight_set on the root would
  // index the DiT sitting beside it.
  auto ws = vpipe::genai::open_weight_set(cfg.vae_file, args.session);
  if (!ws) {
    warn("could not open '" + cfg.vae_file + "'");
    return nullptr;
  }
  auto dec = Ltx25VaeDecoder::load(vcfg, ws, *ops, &err);
  if (!dec) {
    warn(err);
    return nullptr;
  }
  return std::make_unique<Ltx25VaeDecoderAdapter>(std::move(vcfg),
                                                  std::move(ops),
                                                  std::move(dec));
}

namespace {

// The encoder, as the stock `vae-encode` sees it.
class Ltx25VaeEncoderAdapter final : public vpipe::genai::VaeEncoder {
public:
  Ltx25VaeEncoderAdapter(VaeConfig cfg, std::unique_ptr<MetalOps> ops,
                         std::unique_ptr<Ltx25VaeEncoder> enc)
      : _cfg(std::move(cfg)), _ops(std::move(ops)), _enc(std::move(enc)) {}

  int latent_channels() const override { return _cfg.latent_channels; }
  int spatial_compression() const override { return _cfg.spatial_factor(); }
  int temporal_compression() const override { return _cfg.temporal_factor(); }

  bool encode(const vpipe::genai::VaeEncodeRequest& req,
              std::vector<float>* out, std::vector<int>* shape,
              std::string* err) override
  {
    if (_enc == nullptr) {
      if (err != nullptr) { *err = "the encoder was released"; }
      return false;
    }
    std::array<int, 4> s{};
    if (!_enc->encode(req.pixels, req.frames, req.height, req.width, out, &s,
                      err)) {
      return false;
    }
    // [z, T, h, w], the same 4-D shape a video vae-encode emits. NOT
    // collapsed to [z, h, w] for a single frame: generate-video reads
    // `ref_latent0` as a latent whose SECOND axis is time, and a
    // 3-D beat would be read as one with 128 frames.
    if (shape != nullptr) { shape->assign(s.begin(), s.end()); }
    return true;
  }

  void release_idle() override { _enc.reset(); }

  std::uint64_t resident_bytes() const override
  {
    return _enc != nullptr ? _enc->resident_bytes() : 0;
  }

private:
  VaeConfig _cfg;
  std::unique_ptr<MetalOps> _ops;
  std::unique_ptr<Ltx25VaeEncoder> _enc;
};

}  // namespace

std::unique_ptr<vpipe::genai::VaeEncoder>
Ltx25VaeFamily::load_encoder(const VaeModelCreateArgs& args)
{
  auto warn = [&](const std::string& m) {
    if (args.session != nullptr) {
      args.session->warn(fmt("ltx-2.5 VAE encoder: {}", m));
    }
  };
  Config cfg;
  VaeConfig vcfg;
  std::string err;
  if (!resolve_vae_(args.root, cfg, vcfg, &err)) {
    warn(err);
    return nullptr;
  }
  if (args.metal == nullptr || !args.metal->valid()) {
    warn("no usable Metal device");
    return nullptr;
  }
  auto ops = std::make_unique<MetalOps>();
  if (!ops->init(args.metal, &err)) {
    warn(err);
    return nullptr;
  }
  auto ws = vpipe::genai::open_weight_set(cfg.vae_file, args.session);
  if (!ws) {
    warn("could not open '" + cfg.vae_file + "'");
    return nullptr;
  }
  auto enc = Ltx25VaeEncoder::load(vcfg, ws, *ops, &err);
  if (!enc) {
    warn(err);
    return nullptr;
  }
  return std::make_unique<Ltx25VaeEncoderAdapter>(std::move(vcfg),
                                                  std::move(ops),
                                                  std::move(enc));
}

namespace {

// The reference soundtrack, as `audio-vae-encode` sees it: a log-mel
// front end chained onto the VAE's encoder half.
class Ltx25AudioEncoderAdapter final : public vpipe::genai::AudioVaeEncoder {
public:
  Ltx25AudioEncoderAdapter(std::unique_ptr<MetalOps> ops,
                           std::unique_ptr<Ltx25AudioMel> mel,
                           std::unique_ptr<Ltx25AudioVaeEncoder> enc,
                           int rate)
      : _ops(std::move(ops)), _mel(std::move(mel)), _enc(std::move(enc)),
        _rate(rate) {}

  int sample_rate() const override { return _rate; }
  // STEREO. A mono reference is duplicated rather than refused: the
  // model was trained on two channels, and one channel of silence in
  // the other is a different soundtrack from the same signal twice.
  int channels() const override { return 2; }

  bool encode(const vpipe::genai::AudioVaeEncodeRequest& req,
              std::vector<float>* out, std::vector<int>* shape,
              std::string* err) override
  {
    if (_enc == nullptr || _mel == nullptr) {
      if (err != nullptr) { *err = "the encoder was released"; }
      return false;
    }
    if (req.pcm == nullptr || req.n_samples <= 0) {
      if (err != nullptr) { *err = "no samples"; }
      return false;
    }
    if (req.sample_rate != _rate) {
      if (err != nullptr) {
        *err = "this encoder wants " + std::to_string(_rate) +
               " Hz and was handed " + std::to_string(req.sample_rate);
      }
      return false;
    }
    // To stereo. One channel is duplicated; more than two is refused,
    // because picking two of five is a choice the caller should make.
    std::vector<float> st;
    const float* pcm = req.pcm;
    if (req.channels == 1) {
      st.resize((std::size_t)2 * req.n_samples);
      std::memcpy(st.data(), req.pcm,
                  (std::size_t)req.n_samples * sizeof(float));
      std::memcpy(st.data() + req.n_samples, req.pcm,
                  (std::size_t)req.n_samples * sizeof(float));
      pcm = st.data();
    } else if (req.channels != 2) {
      if (err != nullptr) {
        *err = std::to_string(req.channels) +
               " channels: this VAE takes mono or stereo, and choosing two "
               "of more is the caller's decision, not this encoder's";
      }
      return false;
    }

    std::vector<float> mel;
    std::array<int, 3> mshape{};
    _mel->compute(pcm, 2, req.n_samples, &mel, &mshape);
    if (mel.empty()) {
      if (err != nullptr) { *err = "the mel front end produced nothing"; }
      return false;
    }
    std::array<int, 2> rshape{};
    if (!_enc->encode(mel.data(), mshape[1], out, &rshape, err)) {
      return false;
    }
    if (shape != nullptr) { shape->assign(rshape.begin(), rshape.end()); }
    return true;
  }

  void release_idle() override { _enc.reset(); }

  std::uint64_t resident_bytes() const override
  {
    return _enc != nullptr ? _enc->resident_bytes() : 0;
  }

private:
  std::unique_ptr<MetalOps> _ops;
  std::unique_ptr<Ltx25AudioMel> _mel;
  std::unique_ptr<Ltx25AudioVaeEncoder> _enc;
  int _rate = 16000;
};

}  // namespace

std::unique_ptr<vpipe::genai::AudioVaeEncoder>
Ltx25VaeFamily::load_audio_encoder(const VaeModelCreateArgs& args)
{
  auto warn = [&](const std::string& m) {
    if (args.session != nullptr) {
      args.session->warn(fmt("ltx-2.5 audio encoder: {}", m));
    }
  };
  Config cfg;
  std::string err;
  if (!resolve(args.root, cfg, &err)) { warn(err); return nullptr; }
  if (cfg.audio_vae_file.empty()) { return nullptr; }
  if (args.metal == nullptr || !args.metal->valid()) {
    warn("no usable Metal device");
    return nullptr;
  }
  FlexData meta;
  if (!vpipe::genai::comfy::metadata_json(cfg.audio_vae_file, kVaeMetaKey,
                                          meta, &err)) {
    warn(err);
    return nullptr;
  }
  AudioVaeConfig acfg;
  if (!parse_audio_vae_config(meta, acfg, &err)) { warn(err); return nullptr; }

  auto ops = std::make_unique<MetalOps>();
  if (!ops->init(args.metal, &err)) { warn(err); return nullptr; }
  auto ws = vpipe::genai::open_weight_set(cfg.audio_vae_file, args.session);
  if (!ws) {
    warn("could not open '" + cfg.audio_vae_file + "'");
    return nullptr;
  }
  auto enc = Ltx25AudioVaeEncoder::load(acfg, ws, *ops, &err);
  if (!enc) { warn(err); return nullptr; }

  // The front end's parameters live in the checkpoint's `preprocessing`
  // block, NOT in ddconfig -- and its filter bank is not in the
  // checkpoint at all, so the defaults here are the model's documented
  // values rather than a fallback.
  Ltx25AudioMel::Config mc;
  mc.mel_bins = acfg.mel_bins;
  mc.f_max = mc.sample_rate / 2.0;
  auto mel = std::make_unique<Ltx25AudioMel>(mc);
  return std::make_unique<Ltx25AudioEncoderAdapter>(
      std::move(ops), std::move(mel), std::move(enc), mc.sample_rate);
}

std::unique_ptr<vpipe::genai::AudioVaeDecoder>
Ltx25VaeFamily::load_audio_decoder(const VaeModelCreateArgs& args)
{
  auto warn = [&](const std::string& m) {
    if (args.session != nullptr) {
      args.session->warn(fmt("ltx-2.5 audio VAE: {}", m));
    }
  };
  Config cfg;
  std::string err;
  if (!resolve(args.root, cfg, &err)) { warn(err); return nullptr; }
  if (cfg.audio_vae_file.empty()) {
    // Not an error: a checkpoint without one generates no soundtrack.
    return nullptr;
  }
  if (args.metal == nullptr || !args.metal->valid()) {
    warn("no usable Metal device");
    return nullptr;
  }
  FlexData meta;
  if (!vpipe::genai::comfy::metadata_json(cfg.audio_vae_file, kVaeMetaKey,
                                          meta, &err)) {
    warn(err);
    return nullptr;
  }
  AudioVaeConfig acfg;
  if (!parse_audio_vae_config(meta, acfg, &err)) { warn(err); return nullptr; }

  auto ops = std::make_unique<MetalOps>();
  if (!ops->init(args.metal, &err)) { warn(err); return nullptr; }
  auto ws = vpipe::genai::open_weight_set(cfg.audio_vae_file, args.session);
  if (!ws) {
    warn("could not open '" + cfg.audio_vae_file + "'");
    return nullptr;
  }
  auto dec = Ltx25AudioVaeDecoder::load(acfg, ws, *ops, &err);
  if (!dec) { warn(err); return nullptr; }

  // Prefer the BWE (48 kHz). A checkpoint without a `bwe` section falls
  // back to the 16 kHz vocoder alone -- and the RATE we report changes
  // with it, because a sink told 48 kHz for 16 kHz audio plays it a
  // third too slow.
  std::unique_ptr<Ltx25Bwe> bwe;
  std::unique_ptr<Ltx25Vocoder> voc;
  int rate = 0;
  BweConfig bcfg;
  if (parse_bwe_config(meta, bcfg, nullptr)) {
    bwe = Ltx25Bwe::load(meta, ws, args.metal, &err);
    if (bwe) { rate = bwe->output_sampling_rate(); }
  }
  if (!bwe) {
    VocoderConfig vcfg;
    if (!parse_vocoder_config(meta, "vocoder", vcfg, &err)) {
      warn(err);
      return nullptr;
    }
    voc = Ltx25Vocoder::load(vcfg, ws, args.metal, "vocoder.vocoder.", &err);
    if (!voc) { warn(err); return nullptr; }
    rate = vcfg.output_sampling_rate;
  }
  return std::make_unique<Ltx25AudioDecoderAdapter>(
      std::move(acfg), std::move(ops), std::move(dec), std::move(bwe),
      std::move(voc), rate);
}

}  // namespace ltx25
