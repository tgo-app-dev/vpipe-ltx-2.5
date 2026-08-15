#ifndef VPIPE_LTX25_VAE_CONFIG_H
#define VPIPE_LTX25_VAE_CONFIG_H

#include <string>
#include <vector>

namespace vpipe { class FlexData; }

namespace ltx25 {

// One entry of the VAE's block list.
//
// The two numbers are the SAME config field read two ways: a `res_x`
// carries `num_layers`, every `compress_*` carries `multiplier`. Keeping
// one struct rather than a variant is what lets the decoder walk the
// list without knowing which kind it is until it dispatches.
struct VaeBlock {
  std::string name;      // res_x | compress_time | compress_space |
                         // compress_all | compress_*_res | attn
  int num_layers = 0;    // res_x
  int multiplier = 1;    // compress_*

  bool is_res() const { return name == "res_x"; }
  // The (time, height, width) upsample this block applies. {1,1,1} for a
  // res block.
  void stride(int* t, int* h, int* w) const;
};

// The conv video VAE, as the checkpoint's `__metadata__` describes it.
struct VaeConfig {
  std::string class_name;          // CausalVideoAutoencoder
  int  latent_channels   = 128;
  int  out_channels      = 3;
  int  patch_size        = 4;
  int  decoder_base_channels = 128;
  std::string norm_layer = "pixel_norm";
  std::string spatial_padding_mode = "zeros";
  bool causal_decoder    = false;
  bool timestep_conditioning = false;
  std::vector<VaeBlock> decoder_blocks;
  std::vector<VaeBlock> encoder_blocks;

  // The decoder walks `decoder_blocks` REVERSED -- up_blocks[i] is
  // decoder_blocks[n-1-i]. Reading them in list order builds a decoder
  // whose channel counts happen to line up for the first block and then
  // fail to bind, which reads as a missing tensor rather than as a
  // reversed list.
  std::vector<VaeBlock> up_blocks() const;

  // The bottleneck width conv_in projects INTO: base_channels doubled
  // once per channel-changing block, which is what makes it 1024 here.
  int bottleneck_channels() const;

  // Total upsample factors, derived from the block list rather than
  // assumed -- the reference derives them the same way
  // (SpatioTemporalScaleFactors::from_blocks).
  int spatial_factor() const;    // 32 = 8 (blocks) * 4 (patch)
  int temporal_factor() const;   // 8
};

// Parse the `config.vae` object out of a VAE file's `__metadata__`.
// False (with `err`) when this is not a conv video VAE.
bool parse_vae_config(const vpipe::FlexData& cfg, VaeConfig& out,
                      std::string* err = nullptr);

}  // namespace ltx25

#endif
