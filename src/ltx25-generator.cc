#include "ltx25-generator.h"
#include "ltx25-conditioning.h"
#include "ltx25-sampler.h"

#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

using vpipe::FlexData;
using vpipe::fmt;
using vpipe::genai::VideoGenRequest;
using vpipe::genai::VideoGenResult;
using vpipe::genai::WeightSet;
using vpipe::metal_compute::MetalCompute;

namespace ltx25 {

GenerationParams
GenerationParams::from_flex(const FlexData& fd, std::string* err)
{
  GenerationParams p;
  if (!fd.is_object()) { return p; }
  const auto o = fd.as_object();
  auto real = [&o](const char* k, double& dst) {
    if (o.contains(k)) { dst = o.at(k).as_real(dst); }
  };
  auto boolean = [&o](const char* k, bool& dst) {
    if (o.contains(k)) { dst = o.at(k).as_bool(dst); }
  };
  if (o.contains("variant")) {
    p.variant = std::string(o.at("variant").as_string(""));
  }
  real("guidance", p.guidance);
  real("audio_guidance", p.audio_guidance);
  real("stg_scale", p.stg);
  real("audio_stg_scale", p.audio_stg);
  real("modality_scale", p.modality);
  real("audio_modality_scale", p.audio_modality);
  real("audio_seconds", p.audio_seconds);
  real("ref_strength", p.ref_strength);
  real("ref_last_strength", p.ref_last_strength);
  real("ref_audio_strength", p.ref_audio_strength);
  boolean("audio", p.audio);
  boolean("duration_head", p.duration_head);
  if (o.contains("lora")) {
    p.lora = std::string(o.at("lora").as_string(""));
    real("lora_scale", p.lora_scale);
  }
  // `variant` is NOT validated here, and is not acted on here either: it
  // is a LOAD-TIME selector, consumed by Ltx25Family::load through
  // VideoModelCreateArgs::model_config before this generator exists. By
  // the time these params are parsed the DiT is already chosen, so the
  // only thing a check could do is complain about a value that already
  // worked.
  //
  // It did exactly that: the value is matched against what is under
  // diffusion_models/, so a quantized pack ('w8g64') is a perfectly good
  // variant -- and a closed distilled/dev test warned on every step of
  // every run that used one, about a selection that had already
  // succeeded. The vocabulary belongs to the resolver, not to this
  // parse.
  return p;
}

void
Ltx25Generator::log_(const std::string& m) const
{
  if (_session != nullptr) { _session->info(fmt("ltx-2.5: {}", m)); }
}

void
Ltx25Generator::warn_(const std::string& m) const
{
  if (_session != nullptr) { _session->warn(fmt("ltx-2.5: {}", m)); }
}

bool
Ltx25Generator::streaming_blocks() const noexcept
{
  return _dit && _dit->streaming();
}

int
Ltx25Generator::pinned_blocks() const noexcept
{
  return _dit ? _dit->pinned_blocks() : 0;
}

std::size_t
Ltx25Generator::pinned_weight_bytes() const noexcept
{
  return _dit ? _dit->pinned_bytes() : 0;
}

std::unique_ptr<Ltx25Generator>
Ltx25Generator::create(const Config& cfg, std::shared_ptr<WeightSet> ws,
                       MetalCompute* mc, bool stream_blocks,
                       double pin_frac, int plan_w, int plan_h,
                       int plan_frames,
                       const vpipe::SessionContextIntf* session,
                       std::string* err)
{
  std::unique_ptr<Ltx25Generator> g(new Ltx25Generator());
  g->_cfg = cfg;
  // The set is HELD for this generator's lifetime, per the WeightSet
  // contract: every bound tensor is an alias into it, and the checkpoint
  // must unmap when the LAST holder goes away, not the first.
  g->_ws = std::move(ws);
  g->_stream_blocks = stream_blocks;
  g->_pin_frac = pin_frac;
  g->_plan_w = plan_w;
  g->_plan_h = plan_h;
  g->_plan_frames = plan_frames;
  g->_session = session;
  if (!g->_ops.init(mc, err)) { return nullptr; }
  // The connectors come with the DiT: this generator takes a PRE-
  // connector caption, so it needs them.
  g->_dit = Ltx25Dit::load(cfg, g->_ws, g->_ops, stream_blocks,
                           pin_frac, plan_w, plan_h, plan_frames, err,
                           /*with_connectors=*/true);
  if (!g->_dit) { return nullptr; }
  return g;
}

void
Ltx25Generator::release_idle()
{
  // Drop the whole stack. There is nothing smaller worth releasing --
  // the trunk and the scratch are a rounding error against 48 blocks --
  // and the next request rebuilds from the weight set, which still has
  // the bytes mapped.
  _dit.reset();
  _lf = _lh = _lw = _at = _tt = 0;
  _v_tokens = _a_tokens = 0;
  _v_levels = {1.0};
  _a_levels = {1.0};
  _v_baked.clear();
  _a_baked.clear();
  _resident = 0;
}

bool
Ltx25Generator::generate(const VideoGenRequest& req, VideoGenResult* out)
{
  if (out == nullptr) { return false; }
  if (_dit == nullptr) {
    warn_("the DiT was released and this generator does not reload it "
          "in place; wire unload_when_idle to 'never' for a continuous "
          "graph");
    return false;
  }

  std::string perr;
  const GenerationParams p =
      GenerationParams::from_flex(req.model_config != nullptr
                                      ? *req.model_config : FlexData{},
                                  &perr);
  if (!perr.empty()) { warn_("model_config: " + perr); }

  // ---- the conditioning ----------------------------------------------
  const DitConfig& d = _cfg.dit;
  if (req.cond == nullptr || req.cond_rows <= 0) {
    warn_("no conditioning on iport0; nothing to generate from");
    return false;
  }
  if (req.cond_dim != d.cross_attention_dim) {
    warn_(fmt("the conditioning is {} wide but this DiT cross-attends at "
              "{}. iport0 must carry the caption PROJECTED to that width "
              "(the encoder's video_aggregate_embed output), pre-connector",
              req.cond_dim, d.cross_attention_dim)());
    return false;
  }
  // The connector's registers TILE, so the caption has to be a whole
  // number of tiles. Rounding here rather than refusing would change
  // what the model sees without saying so.
  const int reg = d.connector_num_learnable_registers;
  if (reg > 0 && req.cond_rows % reg != 0) {
    warn_(fmt("the conditioning is {} rows, which is not a multiple of the "
              "connector's {} registers. Pad the caption to a multiple "
              "(the reference pads to {})",
              req.cond_rows, reg, 2 * reg)());
    return false;
  }
  if (req.neg != nullptr) {
    // Not an error: a graph wired for Wan should still run here.
    log_("the distilled checkpoint is guidance-distilled (CFG 1); the "
         "negative conditioning on iport1 is ignored");
  }
  if (p.guidance > 0.0 && _cfg.variant == Variant::kDistilled) {
    warn_(fmt("guidance {:.2f} was set but this is the DISTILLED "
              "checkpoint, which has no unconditional pass to blend with; "
              "ignoring it", p.guidance)());
  }

  // ---- geometry -------------------------------------------------------
  const int lf = (req.frames - 1) / kTemporalCompression + 1;
  const int lh = req.height / kSpatialCompression;
  const int lw = req.width / kSpatialCompression;
  if (lf <= 0 || lh <= 0 || lw <= 0) {
    warn_(fmt("degenerate latent geometry {}x{}x{} from {} frames at {}x{}",
              lf, lh, lw, req.frames, req.width, req.height)());
    return false;
  }
  // AUDIO. The audio stream needs a context of its own at
  // audio_cross_attention_dim. `ltx-2.5-conditioner` now produces one
  // and generate-video carries it on its audio_conditioning port, so
  // the two REASONS to run video-only are distinct and worth telling
  // apart -- one is a graph the user can fix, the other is this port.
  const bool want_audio = p.audio && d.use_audio_video_cross_attention;
  const bool have_audio_ctx =
      req.audio_cond != nullptr && req.audio_cond_rows > 0;
  if (want_audio && !have_audio_ctx) {
    log_(fmt("no audio conditioning is wired, so this generates VIDEO "
             "ONLY; wire ltx-2.5-conditioner's audio_conditioning oport "
             "(the {}-wide context) to generate-video's audio_conditioning "
             "iport", d.audio_cross_attention_dim)());
  } else if (want_audio && have_audio_ctx) {
    if (req.audio_cond_dim != d.audio_cross_attention_dim) {
      warn_(fmt("the audio conditioning is {} wide but this checkpoint "
                "wants {}; generating VIDEO ONLY", req.audio_cond_dim,
                d.audio_cross_attention_dim)());
    }
  }
  // The audio token count, from the clip's PIXEL frames over fps -- the
  // reference's AudioLatentShape.from_video_pixel_shape. Deriving it
  // from the video LATENT frame count instead is wrong by the temporal
  // compression factor and produces a soundtrack 8x too short.
  //
  // `audio_seconds` overrides the duration when the graph asked for one;
  // otherwise the soundtrack is exactly as long as the clip.
  int at = 0;
  const bool run_audio =
      want_audio && have_audio_ctx &&
      req.audio_cond_dim == d.audio_cross_attention_dim;
  double audio_secs = 0.0;
  if (run_audio) {
    const double fps = (req.fps > 0.0) ? req.fps : 24.0;
    audio_secs = (p.audio_seconds > 0.0)
                     ? p.audio_seconds
                     : (double)req.frames / fps;
    at = (int)std::llround(audio_secs * kAudioLatentsPerSecond);
    if (at <= 0) {
      log_(fmt("{} s of audio rounds to no latent frames; generating "
               "VIDEO ONLY", audio_secs)());
      at = 0;
    }
  }

  // ---- the references -------------------------------------------------
  //
  // LTX has no separate image-to-video model: a reference is a per-token
  // denoise mask over the SAME sequence, so all three ports collapse
  // into one conditioning build. See ltx25-conditioning.h.
  std::vector<VideoAnchor> vrefs;
  std::vector<AudioAnchor> arefs;
  const int zc_in = d.in_channels;

  auto latent_ok = [&](const std::vector<int>& shape, const char* what,
                       int* frames) {
    if (shape.size() != 4) {
      warn_(fmt("{} is {}-dimensional; this family wants a vae-encode's "
                "[z, T, h, w]. Ignoring it", what, shape.size())());
      return false;
    }
    if (shape[0] != zc_in) {
      warn_(fmt("{} has {} channels but this DiT's VAE produces {}. It was "
                "encoded by a different VAE; ignoring it", what, shape[0],
                zc_in)());
      return false;
    }
    if (shape[2] != lh || shape[3] != lw) {
      warn_(fmt("{} is {}x{} in latent space but this clip is {}x{}. Encode "
                "the reference at the SAME resolution as the output; "
                "ignoring it", what, shape[2], shape[3], lh, lw)());
      return false;
    }
    *frames = shape[1];
    return true;
  };

  if (req.ref != nullptr) {
    int rf = 0;
    if (latent_ok(req.ref_shape, "the reference on ref_latent0", &rf)) {
      if (rf > lf) {
        warn_(fmt("the reference on ref_latent0 is {} latent frames but the "
                  "clip is only {}; ignoring it", rf, lf)());
      } else {
        VideoAnchor a;
        a.latent = req.ref;
        a.channels = zc_in;
        a.frames = rf; a.h = lh; a.w = lw;
        a.strength = p.ref_strength;
        // At the START of the clip, so it OVERWRITES the target grid --
        // those tokens exist and already carry the right positions.
        // One latent frame is image-to-video; several is a video prefix
        // the model continues from.
        a.placement = VideoAnchor::Placement::kLatentIndex;
        a.index = 0;
        vrefs.push_back(a);
        log_(fmt("{} on ref_latent0: {} latent frame(s) held at strength "
                 "{:.2f} from the start of the clip",
                 rf == 1 ? "an IMAGE reference" : "a VIDEO reference",
                 rf, p.ref_strength)());
      }
    }
  }
  if (req.ref_last != nullptr) {
    int rf = 0;
    if (latent_ok(req.ref_last_shape, "the anchor on ref_latent1", &rf)) {
      VideoAnchor a;
      a.latent = req.ref_last;
      a.channels = zc_in;
      a.frames = rf; a.h = lh; a.w = lw;
      a.strength = p.ref_last_strength;
      // At the END, so it CANNOT overwrite: the target grid's last
      // latent frame spans the VAE's whole temporal stride, and writing
      // a still image there would say the picture holds for eight
      // frames. It rides as appended tokens carrying the last pixel
      // frame's own one-frame span.
      a.placement = VideoAnchor::Placement::kKeyframe;
      a.index = req.frames - 1;
      vrefs.push_back(a);
      log_(fmt("a CLOSING anchor on ref_latent1 at pixel frame {} "
               "(strength {:.2f})", a.index, p.ref_last_strength)());
    }
  }
  if (req.ref_audio_rows != nullptr && req.n_ref_audio_rows > 0) {
    if (!run_audio) {
      warn_("a reference soundtrack is wired on ref_audio_rows but this "
            "generation has no audio stream (audio is off, or no audio "
            "conditioning reached iport10); ignoring it");
    } else if (d.audio_out_channels != zc_in) {
      // The audio LATENT is read at in_channels and the velocity is
      // emitted at audio_out_channels; this generator holds one array
      // for both, which is only sound while the two agree (they do:
      // 8 channels x 16 mel bins = 128 either way). A checkpoint where
      // they differ needs two arrays, so say so rather than index one
      // of them past its end.
      warn_(fmt("this checkpoint patchifies audio at {} but emits {}; the "
                "reference-soundtrack path assumes they agree. Ignoring "
                "ref_audio_rows", zc_in, d.audio_out_channels)());
    } else if (req.ref_audio_dim != zc_in) {
      warn_(fmt("the reference soundtrack is {} wide but this checkpoint "
                "patchifies audio at {} (channels x mel bins). It is not an "
                "LTX-2.5 audio latent; ignoring it",
                req.ref_audio_dim, zc_in)());
    } else {
      AudioAnchor a;
      a.tokens = req.ref_audio_rows;
      a.rows = req.n_ref_audio_rows;
      a.dim = req.ref_audio_dim;
      a.strength = p.ref_audio_strength;
      arefs.push_back(a);
      log_(fmt("a reference SOUNDTRACK on ref_audio_rows: {} latent frames "
               "({:.2f} s) held at strength {:.2f}, placed before the clip "
               "on the shared time axis", a.rows,
               a.rows / kAudioLatentsPerSecond, p.ref_audio_strength)());
    }
  }
  if (req.ref_video_rows != nullptr && req.n_ref_video_rows > 0) {
    // Deliberately not read. That port carries MiniMax-H3's Ref2VA rows
    // from a video-ref-encoder; LTX's own reference-video conditioning
    // is IC-LoRA, which needs a LoRA this family does not load and a
    // latent no stage in this tree emits. Wiring it would be untested
    // code claiming a feature. A VIDEO reference here goes on
    // ref_latent0 instead, as a multi-frame latent.
    warn_(fmt("ref_video_rows is wired ({} rows) but LTX-2.5 does not read "
              "it: that port carries another family's reference rows. For a "
              "video reference, encode the clip and wire it to ref_latent0",
              req.n_ref_video_rows)());
  }

  BuiltConditioning bc;
  {
    ltx25::RopeGeometry cgeo;
    cgeo.scale_t = kTemporalCompression;
    cgeo.scale_h = kSpatialCompression;
    cgeo.scale_w = kSpatialCompression;
    cgeo.fps = (req.fps > 0.0) ? req.fps : 24.0;
    cgeo.causal_fix = true;
    std::string cerr;
    if (!build_conditioning(lf, lh, lw, zc_in, at, zc_in, cgeo, vrefs, arefs,
                            &bc, &cerr)) {
      warn_("conditioning: " + cerr);
      return false;
    }
  }
  const int v_tokens = bc.v_tokens, a_tokens = bc.a_tokens;

  // A DiT whose adaLN was baked for OTHER denoise levels cannot be
  // remodulated -- the bake released the projections. Rebuilding is the
  // honest answer and is cheap: the weight set still holds the bytes, so
  // this rebinds rather than re-reads.
  if (_dit->adaln_baked() &&
      (bc.cond.v_levels != _v_baked || bc.cond.a_levels != _a_baked)) {
    log_("the reference strengths changed, so the baked adaLN schedule no "
         "longer covers this request; rebuilding the DiT");
    std::string lerr;
    _dit = Ltx25Dit::load(_cfg, _ws, _ops, _stream_blocks, _pin_frac,
                          _plan_w, _plan_h, _plan_frames, &lerr,
                          /*with_connectors=*/true);
    if (!_dit) {
      warn_("rebuilding the DiT: " + lerr);
      return false;
    }
    _lf = _lh = _lw = _at = _tt = 0;
    _v_baked.clear();
    _a_baked.clear();
  }

  if (lf != _lf || lh != _lh || lw != _lw || at != _at ||
      req.cond_rows != _tt || bc.cond.v_levels != _v_levels ||
      bc.cond.a_levels != _a_levels ||
      v_tokens != _v_tokens || a_tokens != _a_tokens) {
    std::string gerr;
    // The coordinates the model was trained on: PIXELS spatially and
    // SECONDS in time, from the VAE's scale factors and the clip's fps.
    // Feeding latent indices here is what made every earlier run come
    // out as structureless mush -- the spatial RoPE collapsed into a
    // band 32x too narrow. See RopeGeometry.
    ltx25::RopeGeometry geo;
    geo.scale_t = kTemporalCompression;
    geo.scale_h = kSpatialCompression;
    geo.scale_w = kSpatialCompression;
    geo.fps = (req.fps > 0.0) ? req.fps : 24.0;
    geo.causal_fix = true;
    if (!_dit->set_geometry(lf, lh, lw, at, req.cond_rows, geo, &gerr,
                            bc.cond)) {
      warn_("set_geometry: " + gerr);
      return false;
    }
    _lf = lf; _lh = lh; _lw = lw; _at = at; _tt = req.cond_rows;
    _v_tokens = v_tokens; _a_tokens = a_tokens;
    _v_levels = bc.cond.v_levels;
    _a_levels = bc.cond.a_levels;
    log_(fmt("block scratch {:.2f} GB, SHARED by all {} blocks (it was one "
             "arena each -- {:.1f} GB at this geometry)",
             (double)_dit->scratch_bytes() / 1e9, _dit->num_layers(),
             (double)_dit->scratch_bytes() * _dit->num_layers() / 1e9)());
    log_(fmt("attention kernel: {}", _ops.attn_kernel())());

    // WHAT MUST STAY CLEAR when the DiT decides to keep a streamed block,
    // and the honest answer here is NOTHING.
    //
    // BlockResidency::set_reserve says it plainly: zero "is the honest
    // answer for a caller that FREES this model before the next peer
    // runs -- reserving room for a coexistence that does not happen buys
    // nothing and costs the whole denoise". That is this graph.
    // generate-video destroys the DiT at its idle point, and only then
    // does vae-decode load anything, so the two never coexist.
    //
    // The flat 1 GB that was here protected nobody and cost real blocks.
    // admit() asks for the block, the reserve, and ONE MORE block as
    // hysteresis, so a 1 GB reserve turned a 393 MB question into a
    // 1.8 GB one. MEASURED on a 16 GB box whose Metal working set is
    // 12124 MB: the run settled at 2 resident blocks with the box under
    // 10 GB and half a gigabyte of swap -- it had the room and could not
    // ask for it. And the figure was never right in the other direction
    // either: the VAE decode at this geometry wants ~4.7 GB, so 1 GB was
    // not protection, it was a number.
    //
    // The arena is still declared, because note_reserve_allocated then
    // subtracts it straight back out -- set_geometry has already
    // allocated it. Saying it and retracting it is not a no-op: it is
    // what makes the reserve DECLARED, and growth stays off until a
    // caller declares one.
    //
    // A graph that keeps this DiT resident across the decode would need
    // a real figure here, not a token one. Nothing wires that today.
    _dit->set_residency_reserve((std::size_t)_dit->scratch_bytes());
    log_(fmt("latent {}x{}x{} ({} tokens{}) from {} frames at {}x{}, "
             "{} caption rows", lf, lh, lw, lf * lh * lw,
             v_tokens > bc.v_target
                 ? fmt(" + {} conditioning", v_tokens - bc.v_target)()
                 : std::string(),
             req.frames, req.width, req.height, req.cond_rows)());
  }

  // ---- the schedule ---------------------------------------------------
  //
  // The DISTILLED checkpoint's schedule is FIXED at 8 steps -- it is a
  // property of the distillation, not a knob. A `steps` that disagrees
  // is reported rather than honoured, because interpolating this list is
  // not the same model.
  std::vector<double> sigmas = distilled_sigmas();
  const int steps = (int)sigmas.size() - 1;

  // REPORTED WHATEVER THE VARIANT SAYS, because the schedule is fixed
  // whatever it says. This notice used to be gated on the probe having
  // returned kDistilled -- so the one case where the probe FAILS was
  // also the case where a caller who asked for 12 steps, got 8, and was
  // told nothing. A silent override is the thing worth reporting, and
  // it is least excusable exactly when the model is unsure what it is.
  if (req.steps > 0 && req.steps != steps) {
    log_(fmt("this checkpoint's schedule is fixed at {} steps; the "
             "configured {} is ignored -- interpolating the distilled "
             "sigma list is not the same model", steps, req.steps)());
  }
  // And there IS only the distilled ladder here. A `dev` checkpoint
  // needs its own schedule and real classifier-free guidance, neither of
  // which this port has; running it on this list samples a model that
  // was never distilled as though it had been. Loud, because the output
  // is plausible rather than obviously broken.
  if (_cfg.variant == Variant::kDev) {
    warn_(fmt("this is the DEV checkpoint, but only the distilled {}-step "
              "sigma schedule is implemented -- the result will be a dev "
              "model sampled on a distilled ladder, which is not what "
              "either was meant to do", steps)());
  }

  // ---- the denoise loop ------------------------------------------------
  const std::size_t n = (std::size_t)d.in_channels * v_tokens;
  std::vector<float> x(n);
  fill_normal(x, req.seed);
  // sigma[0] is 1.0, so the initial latent IS the noise -- except where
  // a reference gave content. The reference's GaussianNoiser is exactly
  // `lerp(clean, noise, denoise_mask)`: a token at mask 0 STARTS at the
  // content it was given rather than at noise, and stays there.
  for (std::size_t i = 0; i < n; ++i) { x[i] *= (float)sigmas[0]; }
  if (bc.any()) {
    hold_conditioned(x.data(), bc.v_clean.data(), bc.v_mask.data(),
                     d.in_channels, v_tokens);
  }

  std::vector<float> denoised(n), noise(n), next(n);

  // ---- the AUDIO stream ------------------------------------------------
  //
  // A second latent, [audio_out_channels][audio_tokens], stepped on the
  // SAME sigma schedule as the video: the reference hands one `sigmas`
  // tensor to both modalities and there is no per-modality shift. (The
  // per-modality asymmetry is in the GUIDANCE scales -- 3.0 video
  // against 7.0 audio -- which the distilled checkpoint does not use at
  // all.) Its noise is drawn from a different seed offset so the two
  // streams do not start life correlated.
  const std::size_t na =
      run_audio ? (std::size_t)d.audio_out_channels * a_tokens : 0;
  std::vector<float> xa(na), a_denoised(na), a_noise(na), a_next(na);
  if (na > 0) {
    fill_normal(xa, req.seed + 777u);
    for (std::size_t i = 0; i < na; ++i) { xa[i] *= (float)sigmas[0]; }
    if (bc.any()) {
      hold_conditioned(xa.data(), bc.a_clean.data(), bc.a_mask.data(),
                       d.audio_out_channels, a_tokens);
    }
    log_(fmt("audio: {} latent frames ({:.2f} s at {} per second) from a "
             "{} x {} context", at, audio_secs, kAudioLatentsPerSecond,
             req.audio_cond_rows, req.audio_cond_dim)());
  }

  // Bake the adaLN modulations for the whole schedule. The chains run
  // on the HOST over 852 MB of demand-paged weights, eight times a
  // step; this replaces that with a table lookup. It frees no RSS --
  // see Ltx25Dit::bake_adaln for why, and for why the number is 852 MB
  // and not MiniMax-H3's 12.91 GB.
  //
  // `VPIPE_LTX25_NO_ADALN_BAKE=1` keeps the per-step host chains, which is
  // how the two paths get A/B'd: the outputs must be bit-identical,
  // because it is the same arithmetic run at a different time.
  const char* nb = std::getenv("VPIPE_LTX25_NO_ADALN_BAKE");
  if (nb == nullptr || nb[0] == '0' || nb[0] == '\0') {
    std::string berr;
    if (!_dit->bake_adaln(sigmas, &berr)) {
      warn_("adaLN bake: " + berr + " -- computing per step instead");
    } else {
      // Remember WHICH levels it covers. A later request whose reference
      // strengths differ needs a different table, and the projections
      // that would compute one are gone by now.
      _v_baked = bc.cond.v_levels;
      _a_baked = bc.cond.a_levels;
      log_(fmt("adaLN baked for {} steps; {:.2f} GB of host GEMV weights "
               "no longer touched per step",
               sigmas.size(),
               (double)_dit->adaln_bytes_per_step() / 1e9)());
    }
  }

  Ltx25Dit::Input in;
  in.context = req.cond;
  in.n_valid_text = req.cond_rows;   // the caller padded; all rows are real
  in.audio = (na > 0) ? xa.data() : nullptr;
  in.audio_context = (na > 0) ? req.audio_cond : nullptr;

  // Wall clock across the denoise only -- not the load, not the VAE.
  // The step cost is what a kernel change moves, and at this model's
  // sizes the load dominates a total badly enough to hide a 2x.
  const auto t_denoise = std::chrono::steady_clock::now();

  // PER-BLOCK REPORTING, wired straight through. The DiT has always had
  // this hook and nothing set it, so the only thing the host could be
  // told was "a step finished" -- and a step here is one forward of a
  // 48-block stack that may be streaming ~19 GB of weights, so a
  // step-granular bar sits still for minutes at a time and a Stop waits
  // just as long. It costs a compare per block.
  if (req.block_progress) {
    in.progress = [&req](int done, int total) {
      return req.block_progress(done, total);
    };
  }

  for (int s = 0; s < steps; ++s) {
    in.video = x.data();
    in.audio = (na > 0) ? xa.data() : nullptr;
    in.step  = s;
    in.sigma = sigmas[(std::size_t)s];
    // Both sigmas are needed even video-only: the a2v gate is driven by
    // the AUDIO noise level. With no audio stream the gate is unused,
    // but keeping them equal is what a joint run would see.
    in.audio_sigma = sigmas[(std::size_t)s];

    Ltx25Dit::Output vel;
    std::string ferr;
    if (!_dit->forward(in, &vel, &ferr)) {
      warn_(fmt("step {}/{}: {}", s + 1, steps, ferr)());
      return false;
    }
    if (vel.video.size() != n) {
      warn_(fmt("step {}/{}: the DiT returned {} values, expected {}",
                s + 1, steps, vel.video.size(), n)());
      return false;
    }
    // velocity -> x0, then one ancestral Euler step. The renoise draw is
    // seeded from the request's seed plus the step, so a run repeats and
    // two steps never share a draw.
    //
    // With references the conversion is PER TOKEN (`timesteps =
    // denoise_mask * sigma`) and the result is blended back onto the
    // given content -- and then blended AGAIN after the step, because
    // the stepper takes a scalar sigma and would otherwise renoise the
    // anchor while the model is being told it is clean. Both blends are
    // the reference's, in the order its ancestral loop applies them.
    if (bc.any()) {
      to_denoised_masked(x.data(), vel.video.data(), sigmas[(std::size_t)s],
                         bc.v_mask.data(), d.in_channels, v_tokens,
                         denoised.data());
      hold_conditioned(denoised.data(), bc.v_clean.data(), bc.v_mask.data(),
                       d.in_channels, v_tokens);
    } else {
      to_denoised(x.data(), vel.video.data(), sigmas[(std::size_t)s],
                  denoised.data(), n);
    }
    fill_normal(noise, req.seed + 10000u + (std::uint64_t)s);
    euler_ancestral_step(x.data(), denoised.data(), noise.data(),
                         sigmas[(std::size_t)s], sigmas[(std::size_t)s + 1],
                         kAncestralEta, kAncestralSNoise, next.data(), n);
    x.swap(next);
    if (bc.any()) {
      hold_conditioned(x.data(), bc.v_clean.data(), bc.v_mask.data(),
                       d.in_channels, v_tokens);
    }

    if (na > 0) {
      if (vel.audio.size() != na) {
        warn_(fmt("step {}/{}: the DiT returned {} audio values, expected "
                  "{}", s + 1, steps, vel.audio.size(), na)());
        return false;
      }
      // The SAME sigma pair and the same stepper -- the samplers take a
      // pointer and a count, so nothing about them is video-shaped.
      if (bc.any()) {
        to_denoised_masked(xa.data(), vel.audio.data(),
                           sigmas[(std::size_t)s], bc.a_mask.data(),
                           d.audio_out_channels, a_tokens,
                           a_denoised.data());
        hold_conditioned(a_denoised.data(), bc.a_clean.data(),
                         bc.a_mask.data(), d.audio_out_channels, a_tokens);
      } else {
        to_denoised(xa.data(), vel.audio.data(), sigmas[(std::size_t)s],
                    a_denoised.data(), na);
      }
      fill_normal(a_noise, req.seed + 20000u + (std::uint64_t)s);
      euler_ancestral_step(xa.data(), a_denoised.data(), a_noise.data(),
                           sigmas[(std::size_t)s],
                           sigmas[(std::size_t)s + 1], kAncestralEta,
                           kAncestralSNoise, a_next.data(), na);
      xa.swap(a_next);
      if (bc.any()) {
        hold_conditioned(xa.data(), bc.a_clean.data(), bc.a_mask.data(),
                         d.audio_out_channels, a_tokens);
      }
    }

    // AT THE END, ONE-BASED. This used to fire at the START of the step
    // with a zero-based index, which is not what the host's bar re-syncs
    // on: it takes the step that just FINISHED, so the old call reported
    // step s as complete before it had run and the bar ran one step
    // ahead of the work for the whole denoise. The abort is unaffected --
    // the per-block hook above catches a Stop far sooner anyway.
    if (req.progress && !req.progress(s + 1, steps)) {
      log_("stopped mid-denoise");
      return false;
    }
  }

  // The APPENDED conditioning tokens are dropped here. They were input,
  // not output -- the model read them and the sampler held them at the
  // content they came with -- so a VAE handed the whole sequence would
  // decode a clip with the reference stapled to its end.
  if (v_tokens != bc.v_target) {
    std::vector<float> cropped((std::size_t)d.in_channels * bc.v_target);
    for (int c = 0; c < d.in_channels; ++c) {
      const float* src = x.data() + (std::size_t)c * v_tokens;
      float* dst = cropped.data() + (std::size_t)c * bc.v_target;
      for (int t = 0; t < bc.v_target; ++t) { dst[t] = src[t]; }
    }
    out->video = std::move(cropped);
  } else {
    out->video = std::move(x);
  }
  out->video_shape = {d.in_channels, lf, lh, lw};
  if (na > 0) {
    // TRANSPOSE. The DiT carries the audio latent as
    // [audio_out_channels][tokens] with the channel index packed
    // c*mel_bins + f -- the patchifier's `b c t f -> b t (c f)`. The
    // audio VAE decoder wants [channels][frames][mel_bins]. Handing it
    // the DiT's layout unchanged decodes a plausible spectrogram from
    // scrambled channels.
    const int ac = kAudioLatentChannels, am = kAudioLatentMelBins;
    // Only the TARGET tokens; an appended reference soundtrack is input
    // and stops here. The source row stride is the FULL token count,
    // which is what makes this a crop as well as a transpose.
    out->audio.assign((std::size_t)ac * at * am, 0.0f);
    for (int c = 0; c < ac; ++c) {
      for (int f = 0; f < am; ++f) {
        const std::size_t src_row = (std::size_t)(c * am + f) * a_tokens;
        for (int t = 0; t < at; ++t) {
          out->audio[((std::size_t)c * at + t) * am + f] =
              xa[src_row + (std::size_t)t];
        }
      }
    }
    out->audio_shape = {ac, at, am};
    out->latents_per_second = kAudioLatentsPerSecond;
    log_(fmt("audio latent [{}, {}, {}] at {} per second", ac, at, am,
             kAudioLatentsPerSecond)());
  } else {
    out->audio.clear();
    out->audio_shape.clear();
  }
  {
    const double dt = std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - t_denoise)
                          .count();
    log_(fmt("{} steps done in {:.1f} s ({:.2f} s/step) -> "
             "latent [{}, {}, {}, {}]", steps, dt, dt / (double)steps,
             d.in_channels, lf, lh, lw)());
  }
  return true;
}

}  // namespace ltx25
