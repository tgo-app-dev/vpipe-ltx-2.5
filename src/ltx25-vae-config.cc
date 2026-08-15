#include "ltx25-vae-config.h"

#include "common/flex-data.h"

#include <algorithm>

using vpipe::FlexData;

namespace ltx25 {

void
VaeBlock::stride(int* t, int* h, int* w) const
{
  int st = 1, sh = 1, sw = 1;
  if (name == "compress_time" || name == "compress_time_res") {
    st = 2;
  } else if (name == "compress_space" || name == "compress_space_res") {
    sh = sw = 2;
  } else if (name == "compress_all" || name == "compress_all_res") {
    st = sh = sw = 2;
  }
  if (t != nullptr) { *t = st; }
  if (h != nullptr) { *h = sh; }
  if (w != nullptr) { *w = sw; }
}

std::vector<VaeBlock>
VaeConfig::up_blocks() const
{
  std::vector<VaeBlock> v(decoder_blocks.rbegin(), decoder_blocks.rend());
  return v;
}

int
VaeConfig::bottleneck_channels() const
{
  // Each compress block divides the channel count by its MULTIPLIER --
  // and by nothing else. Its conv widens to prod(stride)*in/multiplier
  // and the depth-to-space then divides by exactly prod(stride), so the
  // stride cancels and only the multiplier survives:
  //
  //   1024 -/2-> 512 -/1-> 512 -/2-> 256 -/2-> 128 == base_channels
  //
  // So the bottleneck is base_channels times the product of the
  // multipliers. Folding prod(stride) in as well (the obvious reading of
  // "compress_all doubles everything") gives 8192 here, which is a
  // plausible number that binds nothing.
  int c = decoder_base_channels;
  for (const VaeBlock& b : decoder_blocks) {
    if (b.is_res()) { continue; }
    c *= std::max(1, b.multiplier);
  }
  return c;
}

int
VaeConfig::spatial_factor() const
{
  int f = patch_size;
  for (const VaeBlock& b : decoder_blocks) {
    int t = 1, h = 1, w = 1;
    b.stride(&t, &h, &w);
    f *= h;
  }
  return f;
}

int
VaeConfig::temporal_factor() const
{
  int f = 1;
  for (const VaeBlock& b : decoder_blocks) {
    int t = 1, h = 1, w = 1;
    b.stride(&t, &h, &w);
    f *= t;
  }
  return f;
}

namespace {

// `[["res_x", {"num_layers": 4}], ["compress_space", {"multiplier": 2}]]`
bool
parse_blocks_(const FlexData& fd, std::vector<VaeBlock>& out)
{
  if (!fd.is_array()) { return false; }
  const auto arr = fd.as_array();
  for (std::size_t i = 0; i < arr.size(); ++i) {
    const FlexData& e = arr.at(i);
    if (!e.is_array()) { return false; }
    const auto pair = e.as_array();
    if (pair.size() < 1) { return false; }
    VaeBlock b;
    b.name = std::string(pair.at(0).as_string(""));
    if (b.name.empty()) { return false; }
    if (pair.size() >= 2) {
      const FlexData& p = pair.at(1);
      if (p.is_object()) {
        const auto o = p.as_object();
        if (o.contains("num_layers")) {
          b.num_layers = (int)o.at("num_layers").as_int(0);
        }
        if (o.contains("multiplier")) {
          b.multiplier = (int)o.at("multiplier").as_int(1);
        }
      } else {
        // The reference accepts a bare int as num_layers.
        b.num_layers = (int)p.as_int(0);
      }
    }
    out.push_back(b);
  }
  return !out.empty();
}

}  // namespace

bool
parse_vae_config(const FlexData& cfg, VaeConfig& out, std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  if (!cfg.is_object()) { return fail("the VAE config is not an object"); }
  auto root = cfg.as_object();
  // The file's `config` metadata wraps the model under a `vae` key.
  FlexData inner;
  if (root.contains("vae")) {
    inner = root.at("vae");
    if (!inner.is_object()) { return fail("`config.vae` is not an object"); }
    root = inner.as_object();
  }

  out.class_name = std::string(root.contains("_class_name")
                                   ? root.at("_class_name").as_string("")
                                   : "");
  auto geti = [&](const char* k, int d) {
    return root.contains(k) ? (int)root.at(k).as_int(d) : d;
  };
  auto getb = [&](const char* k, bool d) {
    return root.contains(k) ? root.at(k).as_bool(d) : d;
  };
  auto gets = [&](const char* k, const char* d) {
    return std::string(root.contains(k) ? root.at(k).as_string(d) : d);
  };

  out.latent_channels = geti("latent_channels", 128);
  out.out_channels    = geti("out_channels", 3);
  out.patch_size      = geti("patch_size", 4);
  out.decoder_base_channels = geti("decoder_base_channels", 128);
  out.norm_layer      = gets("norm_layer", "pixel_norm");
  out.spatial_padding_mode = gets("spatial_padding_mode", "zeros");
  out.causal_decoder  = getb("causal_decoder", false);
  out.timestep_conditioning = getb("timestep_conditioning", true);

  if (root.contains("decoder_blocks")) {
    if (!parse_blocks_(root.at("decoder_blocks"), out.decoder_blocks)) {
      return fail("`decoder_blocks` is not a list of [name, params] pairs");
    }
  }
  if (root.contains("encoder_blocks")) {
    parse_blocks_(root.at("encoder_blocks"), out.encoder_blocks);
  }
  if (out.decoder_blocks.empty()) {
    return fail("the VAE config carries no decoder_blocks");
  }

  // What this port implements, stated as a refusal rather than
  // discovered as a wrong picture.
  if (out.norm_layer != "pixel_norm") {
    return fail("norm_layer '" + out.norm_layer + "' is not supported; "
                "this decoder implements pixel_norm");
  }
  if (out.spatial_padding_mode != "zeros") {
    return fail("spatial_padding_mode '" + out.spatial_padding_mode +
                "' is not supported; this decoder zero-pads in space");
  }
  for (const VaeBlock& b : out.decoder_blocks) {
    if (b.name == "attn") {
      return fail("this decoder has an `attn` block, which is not ported");
    }
  }
  return true;
}

}  // namespace ltx25
