# LTX-2.5 — what the checkpoint actually is

Everything here was read off the **released weights** (`Lightricks/LTX-2.5`,
safetensors headers + `__metadata__` over HTTP Range) and cross-checked against
the reference implementation (`github.com/Lightricks/LTX-2`, `ltx-core`
~28k lines). Where the two disagree, the checkpoint wins — that is the lesson
from the MiniMax-H3 bring-up, where a published conversion and the released
weights used different `qkv` groupings and the cross-check proved nothing.

LTX-2.5 is a **joint audio-video** world model: one DiT denoises a video latent
and an audio latent *together*, with cross-attention in both directions. That
makes it MiniMax-H3's closest relative in this tree, and the reason it targets
`generate-video` (video oport0 + audio oport1) rather than `generate-image`.

## Packaging

A **Comfy-aligned split pack**: one `.safetensors` per component, no
`config.json` anywhere. Each file carries its component's config as a JSON
string in the safetensors `__metadata__`. vpipe already reads exactly this
layout — `genai::comfy::` in `generative-models/shared/comfy-checkpoint.{h,cc}`
— so detection reads the file rather than trusting a directory name.

| role | file | size | `__metadata__` key |
|---|---|---|---|
| DiT (distilled) | `diffusion_models/ltx-2.5-22b-distilled-transformer-bf16` | 39 GB | `config` |
| DiT (dev) | `diffusion_models/ltx-2.5-22b-dev-transformer-bf16` | 39 GB | `config` |
| text encoder | `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16` | 24 GB | `gemma_config` |
| video VAE (conv) | `vae/ltx-2.5-video-vae-conv-bf16` | 1.4 GB | `config` |
| video VAE (diffusion) | `vae/ltx-2.5-video-vae-bf16` | 1.4 GB | `config` |
| audio VAE + vocoder | `vae/ltx-2.5-audio-vae-bf16` | 348 MB | `config` |
| duration head | `model_patches/ltx-2.5-duration-head-bf16` | 3.7 MB | `config` |

The `*-comfy-int8-convrot` and `*-nvfp4` variants use Comfy-Org's own packing;
`comfy::unsupported_quant()` already names both, so a resolve skips them rather
than loading garbage. **bf16 is the only readable precision**, which sets the
memory problem below.

Two DiTs ship: **distilled** (fixed 8-step schedule, CFG=1 — guidance-distilled,
so no negative pass at all, exactly like MiniMax-H3) and **dev** (trainable,
40 steps, real CFG). They are the same architecture; only the schedule and
guidance differ.

## Geometry

- Video latent: **128 channels**, spatial **1/32**, temporal **1/8**
  (`SpatioTemporalScaleFactors(time=8, height=32, width=32)`, derived in the
  reference from the VAE block list rather than hardcoded).
- `frames % 8 == 1` (1, 9, …, 121, …); width and height divisible by **32**.
  So 121 frames at 544×960 → latent `[128, 16, 17, 30]` = 8160 video tokens.
- The DiT's own patchify is **1×1×1** (`patchify_proj` is `[4096, 128]`), so one
  latent cell is one token. All the spatial packing is the VAE's `patch_size 4`.
- Audio latent: 128 channels into `audio_patchify_proj [2048, 128]`.

## The DiT — `AVTransformer3DModel`, 21.0 G params, 4349 tensors

48 blocks over **two streams** carried side by side the whole way down:

| | video | audio |
|---|---|---|
| width | 4096 (32 heads × 128) | 2048 (32 heads × 64) |
| FF | 16384, **no bias** | 8192, bias |
| cross-attn context | 4096 | 2048 |

Per block (`BasicAVTransformerBlock`, faithful to the reference `forward`):

1. **video self-attn** — adaLN from `scale_shift_table[0:3]` + timestep;
   `rms_norm(x)*(1+scale)+shift`; RoPE on q and k; per-head gating.
   Then `x = x + out*gate`, and `x_normed = rms_norm(x)` is carried forward.
2. **video cross-attn to text** — adaLN with `scale_shift_table[6:9]`
   (shift_q, scale_q, gate) on the query side and `prompt_scale_shift_table[2]`
   on the **key/value** side. `use_prompt_adaln_single` defaults **true** and
   LTX-2.5 does not override it, so the K/V modulation is
   `prompt_scale_shift_table + prompt_adaln_single(sigma·1000)` — *timestep
   dependent*, and not cacheable across steps. (A checkpoint that turns the
   prompt MLP off gets the static table alone; that is what the reference's
   comment about cacheability describes, and it is not this one.)
3. **audio self-attn / cross-attn** — the same two, on `audio_*` weights and
   `audio_scale_shift_table`.
4. **a2v and v2a cross-attention** — video queries over audio K/V and the
   reverse. Both read `scale_shift_table_a2v_ca_{video,audio}` `[5, dim]`:
   rows `[0:2]` are the a2v scale/shift, `[2:4]` the v2a, row `[4]` the gate.
   **Both directions read the PRE-cross snapshot** of the other stream, so
   direction order does not bias the result — a detail that is invisible in the
   output shape and would silently degrade quality if got wrong.
5. **FF** on each stream with `scale_shift_table[3:6]`.

`scale_shift_table` is `[9, dim]` = 3 (self) + 3 (FF) + 3 (cross).

**Every scale/shift table in the file is F32 while everything else is bf16** —
all 290 of them: six per block (`scale_shift_table`,
`prompt_scale_shift_table`, `scale_shift_table_a2v_ca_{video,audio}` and the
two audio twins) plus the two output heads. Reading one as bf16 is not a small
error: it reinterprets each f32's upper half as a whole value and its lower
half as the next, so the table comes out as alternating garbage and the block
produces **NaN** at the first modulation. That is how this port found it.

### Gated attention (`apply_gated_attention: true`)

Every attention has `to_gate_logits: [heads, query_dim]` and applies
`out_head *= 2 * sigmoid(logits)` per head, per token, before `to_out`. Cheap,
but omitting it scales every head by ~1 instead of [0,2] and quietly wrecks the
result.

### RoPE — `rope_type: "split"`, and it is not the usual one

Not the standard inverse-frequency RoPE. Frequencies are **log-spaced from 1 to
theta** over `inner_dim // (2 * n_pos_dims)` entries, scaled by `pi/2`; positions
are **fractional** (`pos / max_pos`, `max_pos = [20, 2048, 2048]`) and mapped to
`2*frac - 1`, i.e. **[-1, 1]**, not to an absolute index. `freqs` then covers
(f, h, w) concatenated, front-padded with `cos=1, sin=0` to fill `head_dim/2`.
"Split" is the half-rotation (first half / second half of the head dim), not
interleaved pairs.

`use_middle_indices_grid: true` means the position grid is `[..., 2]`
(start, end) per token and the **midpoint** is used — that is what lets the model
allocate different compression to different regions ("diffusion fidelity
rendering"). A port that assumes one integer position per token loses that.

**The a2v/v2a cross-attention RoPE is a different one.** Both streams use the
**audio** width (`audio_cross_attention_dim` 2048 → head_dim 64, because
`audio_to_video_attn` projects the 4096-wide video query down to the audio head
size) and **only the time axis** — `modality.positions[:, 0:1, :]` with
`max_pos = [cross_pe_max_pos]`. So a video token is positioned by its frame
index alone when it attends to audio, which is the only axis the two modalities
share. Feeding the 3-axis self-attention PE here is a shape error at best, and
at coincidentally-matching widths it is silently wrong positions.

### The embeddings connector

Between the text encoder and the DiT sits an 8-layer **perceiver-style
resampler** per stream (`{,audio_}embeddings_connector`), with **128 learnable
registers**, its own gated attention and its own 1-D RoPE
(`connector_positional_embedding_max_pos: [4096]`).
`caption_proj_before_connector: true`. Both connectors live in the **DiT**
file even though they belong to the encoder side of the pipeline.

Its block is much simpler than the DiT's — pre-norm, self-attention, plain
residual, pre-norm, feed-forward, plain residual. No adaLN, no timestep, no
gate on either residual. Three details still differ in ways nothing checks:

- the connector's feed-forward **has bias** (`connector_ff_bias` defaults true
  and LTX-2.5 does not set it) where the DiT's *video* feed-forward does not —
  same code shape, opposite answer;
- **padded positions are replaced by the learnable registers**, tiled over the
  sequence (position `i` takes register `i % 128`), and the attention mask is
  then **discarded**. So the attention is plain full attention; a port that
  instead masks the padding computes something the model never does. The
  sequence length must be a multiple of 128, which is why captions pad to 256;
- its RoPE is 1-D at the **connector's own** `max_pos` (4096), not the DiT's
  `[20, 2048, 2048]`.

*Implemented and verified: 7.7e-3 video / 4.5e-3 audio against the reference.*

### AdaLN sources

Six separate `AdaLayerNormSingle` MLPs (256 → dim → k·dim), all timestep-driven:
`adaln_single` (9·4096 = 36864), `audio_adaln_single` (18432),
`prompt_adaln_single`, `audio_prompt_adaln_single`,
`av_ca_{video,audio}_scale_shift_adaln_single`, `av_ca_{a2v,v2a}_gate_adaln_single`.

#### Baking them, and why the H3 number does not carry over

`Ltx25Dit::bake_adaln(sigmas)` runs all eight chains for every step of the
schedule up front and stores the ~few-kB-per-step result. `Input::step` selects
the row; a baked DiT refuses a request that does not name one.
`VPIPE_LTX25_NO_ADALN_BAKE=1` restores the per-step host chains.

MiniMax-H3 has the same bake and it is worth far more there. Measured off the
shard headers:

| | per-block adaLN | model-level adaLN | checkpoint |
|---|---|---|---|
| MiniMax-H3 | **12.91 GB (54.9%)** | — | 23.5 GB |
| LTX-2.5 | 0.019 GB (0.0%) | **0.852 GB (2.0%)** | 42.02 GB |

H3 puts an adaLN projection **in every one of its 50 blocks**, so a streaming
run re-reads 12.91 GB per step and baking removes 55% of the streamed model.
LTX puts its adaLN at the **model level** — computed once per step and shared
by all 48 blocks — so there is no 55% here and it would be wrong to claim one.

**And it is not a memory win either.** The chains bind through
`WeightSet::tensor(..., Residency::Mapped)` — views into shard pages the DiT
maps anyway, with the set holding its own cached alias. Dropping the model's
handles returns no RSS, so `bake_adaln` reports *bytes no longer touched per
step*, never *bytes freed*. (Making it a real 852 MB would mean binding them
`Copied` under a part and calling `release_part` — i.e. paying 852 MB of copies
up front to hand them back later, which is worse.)

What the bake is actually worth for LTX: `adaln_()` runs its GEMV chain **on
the host**, eight times per forward, **touching** those 852 MB of demand-paged
weights every step. In a streaming run — the only way a 65 GB stack fits —
those pages compete with the streamed blocks for page cache, so re-reading them
per step is I/O and not merely arithmetic. The bake replaces it with a table
lookup. Real, but it is a *work* saving of 2% of the checkpoint, not a 55%
residency saving.

The bar is **bit-identical**, not "close" — it is the same arithmetic moved
earlier, so `ltx25-generate-test` runs unbaked first (the bake clears the
projections, so that order is forced) and asserts zero differing elements. A
1e-7 rel-L2 would mean some chain is being fed a different sigma, which is
exactly the failure a tolerance would hide. **MEASURED: 0 / 16384 elements
differ.**

And the speedup, measured rather than assumed: **20.1 s baked vs 20.5 s
unbaked** over 8 steps at 256×256 with the weights resident — about 2%, close
enough to the noise that it should not be quoted as a win on its own. The case
for the bake is the *streaming* run, where the 852 MB contend with the streamed
blocks for page cache. That has not been measured separately and is not
claimed here.

## Text encoder — Gemma-4 12B unified, which vpipe already has

`architectures: ["Gemma4UnifiedForConditionalGeneration"]`, `hidden_size 3840`,
48 layers, `head_dim 256` / `global_head_dim 512`, sliding+full `layer_types`
(5 sliding : 1 full), `attention_k_eq_v: true` — **this is vpipe's
`gemma4_unified` 12B**, the same family as `VPIPE_GEMMA12B_*`. Only **40** of 48
layers carry a `v_proj`; the other 8 share K as V.

What LTX adds on top:

- `text_embedding_projection.video_aggregate_embed [4096, 188160]` and
  `.audio_aggregate_embed [2048, 188160]`, both **with bias**.
  **188160 = 49 × 3840** — the projection consumes **all 49 hidden states**
  (embedding + 48 layers), not a 3-layer tap like FLUX.2.

  The assembly (`FeatureExtractorV2`) has three details that are silent when
  wrong, and are now implemented and verified at **2.35e-3** against the
  reference:

  - the stack is `[T, D, L]` and the flatten to `[T, D·L]` makes the index
    `d·L + l` — **layer-fastest**. Concatenating layer-major (all of layer 0,
    then all of layer 1) is the natural guess, yields a correctly-shaped
    188160-wide vector, and scrambles the projection;
  - the RMS norm is over `D`, **per (token, layer)** — each layer's
    3840-vector normalised on its own, per token. That is what
    `text_encoder_norm_type: "PER_TOKEN_RMS"` names;
  - each projection applies its **own** rescale, `sqrt(out_dim / 3840)`, to
    the normalised vector — so the video and audio paths see differently
    scaled inputs from one normalisation.

  Padding is zeroed on the **normalised** rows, *before* the projection — and
  the projection has a bias, so padded rows come out as the **bias, not
  zero**. Zeroing the output instead is a different and plausible-looking
  convention.
- `multi_modal_projector`, `audio_projector` and a small `vision_model`
  (patch_dense / pos_embedding) for image and audio conditioning.
- The tokenizer rides **inside the file** as a `U8` tensor
  (`tokenizer_json`, 32 MB), with `hf_asset__*` siblings for the chat template
  and processor config. There is no `tokenizer.json` on disk to open.

## VAEs

**Video, conv** (`CausalVideoAutoencoder`, 170 tensors): patchify 4, then
`res_x`/`compress_space_res`/`compress_time_res`/`compress_all_res` blocks to
128 latent channels; `pixel_norm`; causal in time. This is the port target —
"faster, lighter".

**Video, diffusion** (`CausalDiffusionVAE`, 396 tensors): same encoder, but the
decoder is a `NADiffusionDecoder` (neighbourhood-attention transformer, stage
channels `[2048, 1024, 512, 512, 256]`). Higher quality, much heavier, and it is
a *diffusion* decode — a second sampling loop. Deferred.

**Audio** (1329 tensors): a **mel-spectrogram** VAE — stereo, 64 mel bins,
`z_channels 8`, `causality_axis: "height"` — plus a HiFiGAN-style **vocoder**
(`AMP1` resblocks, upsample rates `[5,2,2,2,2,2]` = hop 160, initial channel
1536). 16 kHz, 5.12 s windows, causal STFT (filter 1024 / hop 160 / win 1024).
So the audio path is STFT → mel → VAE → vocoder, not a waveform VAE.

### The video ENCODER, and the four ways it is not the decoder backwards

Ported (`Ltx25VaeEncoderRef` f32, `Ltx25VaeEncoder` on Metal) because the
image and video reference paths take a latent and **nothing else in the
vpipe tree can make an LTX one** — every family `vae-encode` knows is 8×
or 16× at 16 or 32 channels, and this VAE is 32× at 128.

It shares the channel-last convention and the chunked im2col + GEMM
convolution with the decoder. What it does NOT share:

- **every convolution is CAUSAL**: two copies of frame 0 padded at the
  FRONT and nothing at the back, so a token never reads a later frame.
  The decoder pads symmetrically in the same place, and reusing one
  routine for both is wrong only near the clip's start — where a
  single-frame test cannot tell;
- **`SpaceToDepthDownsample` is not a strided conv.** It is a stride-1
  conv, then space-to-depth, PLUS a mean-pooled space-to-depth of its own
  INPUT added as a skip. The skip is a mean, so there is nothing to bind
  and leaving it out loads cleanly and encodes something plausible;
- a **time-halving block duplicates frame 0 first**, and both its conv
  and its skip read the padded volume. That is what makes 1 + 8k frames
  encode to 1 + k rather than losing one at each of the three halvings;
- the head emits **129 channels** — 128 means and ONE shared log-variance
  that `latent_log_var: uniform` discards — and the latent is WHITENED on
  the way out, the exact inverse of the decoder's denormalise.

Also opposite-nesting, from the other side: `patchify` splits `(c r q)`
with q (HEIGHT) fastest while the space-to-depth splits `(c p1 p2 p3)`
with p3 (WIDTH) fastest. Getting the two the same way round is the single
most likely way to produce a correctly-shaped wrong latent.

VERIFIED: **2.2e-6** CPU-reference against the reference implementation
(every intermediate ≤ 1.4e-6), **1.2e-2** for the Metal path at 1×1
latent cells and **1.0e-2** at 2×2 — and the metal-vs-CPU-reference
number equals metal-vs-golden to four digits, which is what says the
difference is bf16 over ~48 convolutions rather than a second reading of
the layout. 0.035 s on the GPU against 34 s on the CPU reference for the
same 9-frame 32×32 case.

It reaches `vae-encode` through `VaeModelFamily::load_encoder`, an
additive virtual beside `load_decoder` — the header had already named it
as the shape a symmetric extension point would take.

### The audio ENCODER, and its front end's four silent knobs

Ported (`Ltx25AudioMel` on the host, `Ltx25AudioVaeEncoder` on Metal) so
a reference SOUNDTRACK can reach `generate-video`'s `ref_audio_rows`.

The conv half is an ordinary LDM 2D encoder and reuses the decoder's ops
wholesale — the stride-1 convolutions have the same causal-in-TIME
padding, PixelNorm is the same parameterless one. Only three things are
its own: the mel upload, the STRIDE-2 downsample (padding `(0,1,2,0)`,
not the stride-1 `(1,1,2,0)`), and a head that keeps the first 8 of its
16 channels, patchifies `b c t f -> b t (c f)` and whitens — emitting
directly the `[rows, 128]` the DiT's audio patchify assumes.

**The FRONT END is where the traps are**, because it is a
`torchaudio.transforms.MelSpectrogram` with four settings that each
produce a plausible spectrogram when wrong: `power=1` (magnitudes, not
power), the **slaney** mel scale AND slaney normalisation (not HTK),
`center=True` with reflect padding, and `log(clamp(mel, 1e-5))`. Its
filter bank is **not in the checkpoint** — unlike the BWE's, which reads
its bases from `mel_stft.stft_fn.*` — so it is built from the scale
definition and goldened as a matrix in its own right.

And it is **NOT causal**, even though `preprocessing.stft.causal` is true
and the decode side's vocoder mel is: `AudioProcessor` ignores
`is_causal` and always centres. That asymmetry is the reference's.

VERIFIED: filter bank **2.5e-6** and log-mel **4.2e-7** against
torchaudio; whitened rows **4.0e-2** against the reference. That last
number is bf16 depth, not a layout question — reconstructing the rows
from the GOLDEN head with this port's own `(channel, mel-bin)` indexing
reproduces the golden **exactly** (rel-L2 0.0), and the head's own means
are already at 3.1e-2. (The head tap's 6.3e-3 is flattering: 8 of its 16
channels are the log-variance, at ~5x the magnitude, and they are
discarded.) A ROUND TRIP — golden latent -> mel -> 16 kHz waveform -> mel
-> latent — correlates **0.90**, which is the property a reference
actually needs: both halves speak the latent space the DiT was
conditioned in.

### Audio path: latent -> waveform, DONE end to end

The decode chain is `latent -> AudioDecoder -> mel -> Vocoder -> waveform`
(`decode_audio`), and both halves are implemented and verified.

- **AudioDecoder** (`ltx25-audio-vae.{h,cc}` + 4 kernels, bf16): every stage
  tap and the spectrogram match at **1.7e-3 - 6.0e-3**, bf16's own rounding
  over ~20 convolutions.
- **Vocoder** (`ltx25-vocoder.{h,cc}` + 9 kernels, **f32**): the waveform
  matches at **3.1e-5** and the anti-aliased activation on its own at
  **1e-7**.

- **BWE** (`ltx25-bwe.{h,cc}` + 4 kernels, **f32**): 16 -> 48 kHz. The 48 kHz
  waveform matches at **2.7e-5**, every intermediate at 6.2e-6 - 5.3e-5, and
  the rebuilt resampler filter at 2.7e-7.

The whole chain turns a `[8, 128, 16]` latent into 5.09 s of stereo in
**~1.5 s** at 16 kHz or **~2.1 s** at 48 kHz on an M4 Pro.

#### Why the vocoder runs in f32 while everything else is bf16

The reference forces its whole vocoder pass to fp32 and says why: bf16
accumulation compounds through its **108 sequential convolutions** and
degrades spectral metrics (mel_l1, MRSTFT) by 40-90%. The WEIGHTS are bf16
there too -- it upcasts them per-op -- so what has to be wide is the
**activations and the accumulation**, not the checkpoint.

So the plugin builds a **third dtype twin of its metallib**
(`ltx25_kernels_f32`, `VPIPE_ELT=float`) and the vocoder widens its weights
once at load (439 MB). This buys correctness that no amount of care in bf16
could, and it also buys VERIFIABILITY: at bf16 the error would sit far above
anything that could separate a port bug from rounding. libvpipe's dense GEMM
is built for half and bfloat only, which is exactly the trap this avoids, so
the vocoder brings its own tiled f32 GEMM rather than borrowing one that
would undo the point.

#### What is not plain HiFiGAN

- **SnakeBeta**, `x + (1/(beta+1e-9)) * sin(x*alpha)^2`, with alpha and beta
  stored in **log scale** -- both are exponentiated before use. Using them
  raw is a smooth, plausible nonlinearity that is not this one.
- **Every activation is anti-aliased**: `upsample 2x -> snake -> lowpass
  downsample 2x`, with 12-tap kaiser-sinc filters that come **from the
  checkpoint** (`.upsample.filter`, `.downsample.lowpass.filter`), so no
  window is recomputed and the beta/cutoff conventions cannot drift.
- Each upsample stage runs its **three resblocks on the same input and takes
  the mean** -- they are not chained.
- `ConvTranspose1d` stores its weight as **[C_in][C_out][K]**, the opposite
  of `Conv1d`. Reading it the Conv1d way binds a correctly-sized weight that
  mixes the channels up.
- The main vocoder's config section carries **no sample rate**: the reference
  takes it from the BWE's `input_sampling_rate` (16 kHz). Left at the 24 kHz
  default the samples are correct and play 1.5x fast, which no rel-L2 on the
  waveform would ever catch.

#### The BWE stage

`vocoder(mel)` -> pad to a whole number of hop-80 frames -> causal log-mel ->
`bwe_generator` for a residual -> add a x3 sinc-resampled skip -> clamp -> crop
to `T_low * 3`. The generator is the same BigVGAN with different rates, so it
is an `Ltx25Vocoder` under a different prefix and needed no new code. What is
new is the front end, and three things there are easy to get wrong quietly:

- **The STFT is CAUSAL**: `win_length - hop_length` = 432 samples of zero
  padding on the **left only**, so a frame never looks ahead. A centred STFT
  is a perfectly good spectrogram that shifts the residual in time.
- **Its DFT bases come from the checkpoint** (`mel_stft.stft_fn.forward_basis`,
  real rows then imaginary). The reference does this deliberately -- it wants
  the exact bases from training -- so this reads them rather than
  reconstructing a DFT.
- **The resampler's filter is NOT in the checkpoint** (`persistent=False`) and
  is a **HANN**-window sinc, not the kaiser one the activations use. It is the
  one tensor in the chain that has to be rebuilt rather than read, so it gets
  a golden of its own (43 taps, matching at 2.7e-7).

Also: `apply_final_activation=False` for the BWE generator is a **call-site
argument** in the reference, not a config value -- `_vocoder_from_config`
passes it explicitly and never reads the key. Clamping the residual before it
is added would clip a correction meant to be able to cancel the skip.

The padding deserves a note: the skip is taken from the **padded** signal too
(the reference reassigns `x` before resampling), and the output is cropped
using a length computed from the **unpadded** one. Both halves therefore see
the same signal and the extra never reaches the output.

**AudioDecoder** (102 tensors) is a classic LDM 2D decoder over the mel
spectrogram, walked in reverse:

- `conv_in` 8 -> 512; `mid.block_1`, `mid.block_2` at 512 with **no attention**
  (`attn_resolutions: []`, `mid_block_add_attention: false`);
- `up.2`: 3 res blocks at 512 + upsample; `up.1`: 512->256 (with
  `nin_shortcut`) + 2 more + upsample; `up.0`: 256->128 (with `nin_shortcut`)
  + 2 more, **no upsample**; `conv_out` 128 -> 2 (stereo).
- `num_res_blocks: 2` but each level has **three** blocks -- the usual LDM
  `num_res_blocks + 1` on the decoder side.
- `norm_type: pixel`, so the same PixelNorm as the video VAE.

Two conventions that are its own, and both are silent when wrong:

- **`causality_axis: "height"`.** `CausalConv2d` pads asymmetrically on the
  causal axis: for HEIGHT the pad tuple is `(pad_w//2, pad_w-pad_w//2, pad_h, 0)`
  -- i.e. **all** of the height padding goes on TOP and none on the bottom,
  while width is padded symmetrically. **Height is TIME, not the mel axis**:
  the tensor is `(batch, channels, frames, mel_bins)`, so torch's H is frames
  and W is frequency, and causality is over time as one would expect.
  `_adjust_output_shape` says so directly ("pad_top/bottom = time"). An earlier
  note in this file had it backwards.
- **`Upsample` drops the FIRST row after its conv**, for the same reason the
  video decoder drops the first frame: nearest-2x then a causal conv makes the
  first two outputs depend only on the first input, so dropping one undoes the
  encoder's padding and keeps the length at `1 + 2n`.

**The patchifier packing, PINNED.** `_denormalize_latents` denormalizes the
*patchified* latent, and `AudioPatchifier.patchify` is a single einops line,
`b c t f -> b t (c f)`. So the 128 statistics are indexed **`c * 16 + f`** --
channel-major, latent mel bin FASTEST -- and `un_normalize` is a per-*(channel,
mel-bin)* affine, **not** a per-channel one. The 16 is the latent's own mel
axis: the encoder halves frequency once per level boundary, so 64 / 2^2 = 16,
and 8 x 16 = 128 closes exactly.

Three things agree on this and none of them is the tensor's shape: the einops
expression itself; the frame arithmetic (two upsamples take F -> 4F-3, which is
precisely the target `_denormalize_latents` computes, so its crop/pad step is
always a no-op); and a golden on the denormalized latent. That last one is what
makes it a verified fact rather than a reading -- the statistics span 2.76x, and
a transposed reading of the same 128 values differs from the correct one at
rel-L2 **0.35**, against the 1.7e-3 the port actually measures. The tap
discriminates by a factor of 200.

**Vocoder** (1227 tensors) is BigVGAN-style, not plain HiFiGAN: `snakebeta`
activation, `AMP1` resblocks, `upsample_rates [5,2,2,2,2,2]` (= hop 160),
`upsample_initial_channel 1536`, kernels `[3,7,11]` with dilations
`[[1,3,5],[1,3,5],[1,3,5]]`, stereo, no bias and no tanh at the final layer.
The tensor list carries `act_post.act.alpha` / `.beta` plus
`act_post.upsample.filter` and `act_post.downsample.lowpass.filter`, so the
activations are **anti-aliased** (upsample -> snake -> lowpass -> downsample)
rather than pointwise. There is also a second `bwe_generator` (band-width
extension, rates `[6,5,2,2,2]`) which the config carries separately.

Reference harness note: the audio module imports `torchaudio` (installed in the
venv now), and the state dict needs BOTH the `audio_vae.` and `decoder.`
prefixes stripped -- `AudioDecoderConfigurator.from_metadata` builds the module
but does not load weights.

## Scheduler and sampler

`RectifiedFlowScheduler`, `sampler: "LinearQuadratic"`, 1000 train timesteps.
The distilled checkpoint uses a **fixed 8-step sigma list**:

```
[1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
```

and the two-stage pipeline re-noises to `0.909375` for a 3-step stage 2 at 2×
resolution. Sigmas **descend** (1 = pure noise), the opposite of Boogu's
inverted convention.

**Which stepper is not the default one.** The reference ships a deterministic
second-order `res_2s` exponential-integrator step *and* an ancestral (SDE)
Euler step, and `should_use_ancestral_sampler` selects the **ancestral** one
for any checkpoint of generation ≥ 2.5 — so LTX-2.5 is sampled ancestrally at
`eta = 1.0`, `s_noise = 1.0`. Reaching for `res_2s` because it is the module
default would be wrong twice: a different trajectory, and **twice the cost**,
since `res_2s` also evaluates the model at a midpoint. On a 22B model that is
8 forwards against 16.

The step is the **rectified-flow** parameterisation (`alpha = 1 - sigma`), not
the DDIM/variance-exploding ancestral coefficients; the two agree only at
`eta = 0`. *Implemented and verified at ~5e-8 per step.*

One more precision note, the third of its kind here: the scalar coefficients
must be computed in **f32**, where the reference computes them. At `eta = 1`,
`sigma_down = sigma_next²/sigma`, so `alpha_down = 1 - sigma_down` is ~0.019
at the schedule's second step and the renoise coefficient is a difference of
two nearly-equal terms divided by it. Computing that in double is *more
accurate* and differs from the reference by ~1e-6; matching it is the point.

## Kernels: which ones the model runs on

Every GEMM in the DiT, both VAEs and the connector goes through
`MetalOps::linear`, and every attention through one `run_attention_`. That is
deliberate: it means the kernel choice is made in two places for the whole
model, and this section is the whole of it.

**Attention is steel flash attention.** The scalar `sdpa_full_f16` runs one
simdgroup per (head, query token) across the whole key sequence. For an LM
decode that is the right shape; for a video DiT it is the entire step.
`MetalOps::sdpa_steel` binds libvpipe's register-resident steel kernel over
the *same* head-major buffers, so it is a dispatch swap rather than a second
data path — the four `transpose_abd` calls around it are unchanged.

MEASURED at 960 × 544 × 49 (3570 video tokens, w8g64, M4 Pro, 8 steps):

| attention | s/step |
|---|---|
| `sdpa_full_f16` (`VPIPE_LTX25_NO_STEEL_ATTN=1`) | 97.50 |
| `attn_steel_h_bd{128,64}_bf16` | **23.98** |

**4.07x**, and the gap widens with resolution and length because the term it
removes is the quadratic one. The shipped 121-frame geometry is 8160 tokens —
5.2x this clip's attention work against 2.3x its GEMM work.

The fallback is kept because it is the correctness A/B, not because anything
should run on it. Both arms clear every golden: block 0 lands at 6.659e-3
video / 3.912e-3 audio on steel against 6.662e-3 / 3.938e-3 on the scalar
kernel — *closer*, which is what an f32 online softmax over f32 accumulators
should do. The 48-block stack ramps smoothly 6.7e-3 -> 1.8e-2 with depth,
which is bf16 accumulating, not a kernel disagreeing.

**Two things about the steel contract are LTX-specific.** Both come from this
being the first DiT in the family with genuine cross-attention:

1. **`qL` and `kL` differ.** Text cross-attention, a2v and v2a all attend a
   sequence that is not their own length. Every sibling port sets `kL = qL`
   because it never has to. Here the K/V strides and `NK`/`NK_aligned` come
   from `tkv` and the Q/O strides from `tq`; filling all four from one length
   is a bug no self-attention test can see.
2. **Function constant 201 comes from the KEY length.** 200 says the last
   *query* tile is full, 201 the last *key* tile. Since the edge handling is a
   function constant rather than an argument, a plan is per SHAPE — a
   specialised pipeline plus its `AttnParams` buffer. All 48 blocks meet the
   same four shapes, so the plans are cached in the shared `BlockScratch`
   alongside the working buffers, and the stack builds four of them, not 192.

`tests/ltx25-attn-test.cc` pins both, without the checkpoint: eleven shapes
at head_dim 128 and 64, mostly NON-square, with tails chosen to fall off the
query tile (32) and the key tile (16) independently. Steel agrees with
`sdpa_full` at 5e-6 .. 9e-5 rel-L2 against an 8e-3 bar, and the two shapes
small enough to fit one tile come out bit-identical.

It carries a **negative control**, and the control is the reason the test is
worth its lines. Mutating `steel_attn_plan` to the `kL = qL` form every
sibling port uses leaves the two square-aligned cases at 7.6e-6 and 4.7e-6 —
passing, unchanged — and blows up all nine others, from 0.24 to 6.5e+16. A
suite of square cases would have signed that port off.

**Everything else is already on a fast path.** Dense GEMMs take libvpipe's
steel `dense_gemm_t_bm64_f16`; quantized ones take `affine_qmm_steel_w{4,8}g{32,64}`
with the `_bm64` arm above 64 rows; both VAEs are im2col + those same GEMMs
rather than direct convolution. (On a matrix-core GPU both of those move to
`dense_gemm_mma_*` — see the M5 section below.) The one hand-written GEMM left is
`ltx_voc_gemm` in the vocoder and BWE, which is a tiled threadgroup kernel
rather than a scalar one — it stays because those two run **f32** (see the
precision note in the vocoder section) and libvpipe ships no f32 steel GEMM.
They are a small fraction of a run.

### The M5 matrix-core paths

Written blind on the M4 Pro, which has no matrix cores, and since **measured
on an M5**. All of it is switched on by `supports_matrix_cores()` alone —
never by the library loading, because on a pre-M5 GPU the nax metallib's
entry points are build-time stubs that write a single zero. A `valid()`
library there would bind a kernel that computes nothing.

- **`attn_steel_nax_h_bd{128,64}_bf16`** replaces the ALU steel kernel. Same
  `AttnParams`, same function constants, same threadgroup contract; only the
  tiles change (bq 32/bk 16 -> bq 64/bk 32), which is why it is a function
  swap inside `steel_attn_plan` and not a branch anywhere else.
  `VPIPE_LTX25_NO_ATTN_NAX=1` forces the ALU kernel for an A/B.

  MEASURED on M5: every shape in `ltx25-attn-test` passes, at **1.2e-3**
  rel-L2 against `sdpa_full` where the ALU kernel scores 5e-6 .. 9e-5 on the
  same box. That gap is the kernel, not the wiring — the two shapes small
  enough to fit one tile are bit-identical on the ALU kernel and 9.4e-4 on
  the nax one, so it is structural rather than K-length accumulation, and it
  sits right alongside the 1.66e-3 the in-tree FLUX.2 port records for the
  same bf16 nax entry point. It clears the 8e-3 bar with room, but a golden
  cut on the ALU path should be re-cut rather than assumed.

- **`dense_gemm_mma_*`** replaces the steel dense GEMM at `M >= 64`,
  `N >= 16`, in **three** tiles — N-regions of 128, 256 and the TN=2 tile's
  512. The kernel voids its bias argument, so the bias is a second pass
  exactly as on the quantized path. `VPIPE_LTX25_NO_MMA=1`,
  `VPIPE_LTX25_MMA_MIN_M` and `VPIPE_LTX25_MMA_TILE=128|256|512` are the
  knobs.

  **The tile is picked from M, K and N together, and this is the one place
  a sibling DiT's rule could not be borrowed.** Those ports choose from K
  alone, which works when N is never below 4096 — and here the same kernel
  also serves a VAE whose N is the channel count, 1024 down to 48. MEASURED
  on M5, the K rule mispicks the two largest VAE levels AND the two largest
  DiT GEMMs, in opposite directions. No K threshold can fix it, because the
  winner is not a function of K: `(M=8160, K=4096, N=4096)` wants the 512
  tile and `(M=1024, K=4096, N=4096)` wants the 128 one.

  Mean ms over 9 rounds, every arm run ONCE per round with the round's
  fastest voting — the method `shared/kernel-autotune.h` uses, and not
  negotiable here: a first pass ran each arm to completion before starting
  the next, which is the exact error that file's sibling `shared/mma-tile.h`
  records as having turned a 7% regression into an apparent 1.07-1.22x win.
  Votes were 9/9 or 8/9 for every row below; re-measuring properly did not
  change any routing decision, but it did remove two apparent near-ties.

  | shape | M | K | N | n128 | n256 | tn2 | win |
  |---|---|---|---|---|---|---|---|
  | DiT self-attn | 8160 | 4096 | 4096 | 24.92 | 24.79 | **22.19** | +12% |
  | DiT ff_in | 8160 | 4096 | 16384 | 97.42 | 102.72 | **90.59** | +8% |
  | DiT ff_out | 8160 | 16384 | 4096 | 218.23 | **109.28** | 222.53 | +100% |
  | DiT cross k,v | 1024 | 4096 | 4096 | **2.46** | 3.28 | 3.05 | +24% |
  | DiT audio ff_out | 512 | 8192 | 2048 | **1.42** | 1.90 | 1.53 | +8% |
  | VAE L0 1024ch | 2427 | 27648 | 1024 | 32.51 | 14.12 | **11.57** | +22% |
  | VAE L0 small | 800 | 27648 | 1024 | 8.71 | 5.98 | **5.04** | +19% |
  | VAE L1 512ch | 4854 | 13824 | 512 | 10.59 | 6.35 | **5.46** | +16% |
  | VAE L2 256ch | 9709 | 6912 | 256 | 3.43 | 3.22 | **2.87** | +12% |
  | VAE L3 128ch | 19418 | 3456 | 128 | **1.73** | 2.85 | 2.69 | +55% |
  | VAE conv_out | 19418 | 3456 | 48 | **1.86** | 3.06 | 2.97 | +60% |

  The rule fitted to that, with a mechanism for each branch: narrow N takes
  the 128 region because a wider one is mostly padding (at N=48 the 512 tile
  computes ten times the columns that exist); deep K with wide N takes 256,
  where the 512 tile is **2x slower**; deep K otherwise takes 512, because
  each threadgroup already does enough K work to saturate; and below that,
  512 when there are enough row-tiles to fill the GPU (`M >= 2048`), 128
  when there are not.

  It was CHECKED on shapes held out of the fit, and the one miss was
  instructive: fitting the last branch on M alone cost **1.70x** at
  `(M=800, K=27648, N=1024)`, a real VAE level at a small chunk, which is
  what added the deep-K arm. Under the voting method every shape now
  resolves decisively (+8% at the narrowest), so a static rule is adequate
  here — the earlier claim that two shapes were irreducible ties was an
  artifact of the sequential-arm measurement, not a property of the kernels.

  **This does not contradict `shared/mma-tile.h`**, which records that an N
  term made an LM prefill rule *worse*. That negative result is about
  routing LARGE N (16384-43008) to the WIDE tile at shallow K; it was
  measured over two tiles, both narrower than the TN=2 one, on a model whose
  N never drops below 4096. Neither N term here is that: one sends TINY N
  (<= 128, which no LM has) to the narrow tile, and the other chooses
  between 256 and 512 at deep K. The shared rule's `mma_use_wide_tile` would
  still be the right thing to call if this plugin could reach it —
  `generative-models/shared/` is not in the installed SDK.

- **A quantized weight reaches the same kernel** by being expanded once into
  a shared `[N][K]` bf16 scratch by `affine_dequant_w{4,8}g{32,64}`, rather
  than staying on the fused dequant-in-loop affine qmm. This is the single
  largest M5 lever in the model, because the DiT ships quantized — bf16 is
  39 GB — so without it the matrix cores would serve only the VAEs while the
  dominant term stayed on the ALU.

  MEASURED on M5 through `MetalOps::linear` itself, arms interleaved
  (the GPU clock tracks the power budget here, and a sequential A-then-B can
  invert a result), at the shipped 960x544x121 geometry, w8g64:

  | per-block GEMM | x | qmm ms | mma ms | |
  |---|---|---|---|---|
  | self q,k,v,o | 4 | 76.76 | 24.04 | 3.19x |
  | cross q,o | 2 | 76.76 | 23.93 | 3.21x |
  | cross k,v (text) | 2 | 9.59 | 2.77 | 3.47x |
  | ff_in | 1 | 307.15 | 89.34 | 3.44x |
  | ff_out | 1 | 328.87 | 108.59 | 3.03x |
  | **per block** | | **1115.6** | **334.5** | **3.33x** |

  which is **53.5 s -> 16.1 s** per denoise step across 48 blocks, video
  stream only. w4g64 lands in the same place (2.7–3.5x). The two routes
  agree at 4.6e-5 .. 8.4e-5 rel-L2. `VPIPE_LTX25_NO_DEQUANT_MMA=1` keeps
  quantized weights on the qmm, which is the A/B above.

  (3.15x of that is the dequant-once route; the rest is the tile rule
  below, which is worth 1.06x on top of it.)

  The scratch is shared by every projection, which is safe **only** because
  encoders are serial (`DispatchType::Serial`), so Metal's hazard tracking
  orders each dequant against the matmul that reads it. Anything that ran
  two GEMMs concurrently would need one scratch each — the same constraint,
  for the same reason, as the shared `BlockScratch` above.

- **The 2 GB operand limit.** matmul2d addresses a tensor operand with a
  signed 32-bit BYTE offset, so an operand of 2 GB or more silently computes
  garbage from the 128-row tile that straddles the boundary onward — no
  error, no fault. MEASURED against the steel GEMM: at K=3456 the first
  wrong row is 310784 and at K=864 it is 1242880, in both cases exactly
  `2^30 / K` rounded up to a 128-row tile, with everything below correct to
  5e-5. `mma_eligible_` therefore refuses any shape whose operands cross it
  and falls back to steel, which is int64-safe.

  Nothing in the model reaches this today: the DiT is bounded by its token
  count, and the VAEs chunk im2col to 64 M **elements** = 128 MB, 16x under.
  The guard is there because that VAE bound is a public setter whose own
  comment invites raising it, and what it would buy is a corrupt lower half
  of the picture rather than an error.

The generator logs `attention kernel: steel-nax | steel | scalar` at the top
of every denoise, so which arm ran is in the log rather than inferred.

## The memory problem

bf16 is the only readable precision, so a minimal t2v stack is
**39 GB DiT + 24 GB text encoder + 1.4 GB VAE ≈ 65 GB**. That does not sit
resident on the 64 GB M4 Pro, let alone a 16 GB box. Three things make it
tractable, in the order they should be tried:

1. **Never hold the encoder and the DiT at once.** The graph already splits
   them: `diffusion-conditioner` encodes, `generate-video` denoises. Wiring
   `unload_when_idle` is the cheap half of the answer.
2. **Block streaming** for the DiT, as MiniMax-H3 does for its 33B stack
   (`model_memory::plan_streaming`, `shared/block-residency.h`). 48 blocks at
   ~800 MB each stream comfortably.
3. **Quantize.** vpipe's own `model-quantize` produces w4/w8 the metal kernels
   read; the vendor int8/nvfp4 packings are not readable here. A w8 DiT is
   ~20 GB, w4 ~10 GB — this is what makes the model routine rather than
   heroic, and it is the same path Krea-2 and QIE took.
4. **One scratch arena for all 48 blocks.** Weights are the term everyone
   reaches for first; at the shipped geometry the ACTIVATIONS were bigger.
   `BlockScratch` was a member of `MetalBlock`, so each block allocated its
   own — and at 960 × 544 × 121 (8160 tokens) that is **1.18 GB x 48 =
   56.8 GB**, on top of a 24 GB checkpoint. Blocks run strictly sequentially
   (one command stream each, `commit().wait()` between) and nothing in the
   arena survives a forward — the residual stream is the caller's buffer —
   so one arena serves the stack: **56.8 GB -> 1.18 GB**, verified
   bit-identical against 48 private arenas. It went unnoticed because the
   arena is sized `max(video, audio, TEXT)` tokens and every clip brought up
   before this had fewer than the 1024 text rows, so memory looked flat in
   the geometry.

   Sharing is safe ONLY because of that sequencing. Anything that ran two
   blocks concurrently would have to give them separate arenas again.

### The quantized path, and the four things that are not obvious

The plugin reads a quantized pack through the same `MetalOps::linear` every
dense block already uses — `QWeight` holds either a dense `w` or the
`codes`/`scales`/`biases` triple, and the branch is one `if` inside the op
rather than at all seven block call sites.

Four facts, each of which is a **silent wrong answer** rather than a load
failure, and each pinned by `tests/ltx25-qlinear-test.cc`:

1. **`scales`/`biases` are F16 in the checkpoint and BFLOAT to the kernel.**
   Every writer in the tree emits F16; the qmm kernels used here are the
   `_bf16` twin, whose buffers 1 and 2 are `VPIPE_ELT` = bfloat. Handing the
   raw bytes over is not an approximation — different exponent bias, different
   mantissa width — so `get_as_bf16_` converts at load. The element *count* is
   identical, so no shape check catches it.
2. **The qmm kernel has no bias slot.** Its buffer 2 is the quantization
   zero-point, a different tensor with a confusingly similar name. A linear's
   bias is a second pass (`ltx_bias_add`).
3. **`_bm64` exists only at group 64.** There is no
   `affine_qmm_steel_w4g32_bm64`. Requiring the wide tile for every
   (bits, group) turned quantization off wholesale the first time this was
   written — and it failed *quietly*, as "this host has no affine kernels".
4. **Bits and group are recovered from the SHAPES, not a config.** Other
   loaders read `quantization.{bits,group_size}` from config.json. Here `K` is
   known from the architecture, and the packing pins the rest:
   `group = K / scales_cols`, `bits = codes_cols * 32 / K`. Two shapes, two
   unknowns. That keeps the Comfy single file and the quantized directory
   loading through the same code.

**What must be EXCLUDED from the quantizer.** `quant_all_in_scope` takes every
2D floating-point tensor whose leaf is not a norm or an embedding. Over
`transformer_blocks.` that is right for the 28 linears and wrong for two
kinds of tensor that happen to be 2D:

| tensor | shape | why it must stay dense |
|---|---|---|
| `*scale_shift_table*` (4 per block) | F32 `[9,4096]`, `[2,4096]`, `[5,4096]`, … | modulation TABLES, not matrices; f32 on purpose, and read through `get_f32_as_bf16_` |
| `*to_gate_logits.weight*` | `[32, dim]` | 32 rows feeding a per-head sigmoid; a group-affine over that is all error and no saving |

So the pass needs
`quant_exclude: "scale_shift,to_gate_logits"` — which is why the host's
`model-quantize` stage grew a `quant_exclude` attribute. Neither exclusion
shows up as a load failure: the checkpoint quantizes, loads, and generates
the wrong thing.

**MEASURED, on the 22B distilled DiT** (`transformer_blocks.` scope, 1344
tensors quantized = exactly 48 blocks x 28 linears, 3005 passed through):

| pack | size | residency verdict | PSNR vs bf16, same seed |
|---|---|---|---|
| bf16 (single file) | 42.0 GB | 65 GB → **STREAM**, ~67 s/step | — |
| w8g64 | 23 GB | 49 GB → **PRELOAD** | **33.29 dB** (32.3–34.5) |
| w4g64 | **14 GB** | 40 GB → **PRELOAD** | 25.87 dB (24.8–26.6) |

**Read the residency column carefully — the verdict is not what bought the
speed.** LTX binds its blocks `Mapped` in both arms, so `STREAM` vs `PRELOAD`
selects only whether the WeightSet keeps a cached alias; the same shard is
mapped either way. What removed the ~550 MB/s of per-step re-faulting is that
14 GB (or 23 GB) of demand-paged weights FIT beside everything else resident,
where 42 GB did not. Quantizing changed the working set; it did not change a
policy.

The same is true in the other direction: on this box a bf16 DiT can be made to
stop thrashing without quantizing anything, by freeing the 24 GB text encoder
before the denoise — see "The encoder is the lever" below.

**w8 is the sensible default on a 64 GB box** — 7.4 dB better than w4 for 9 GB
more, and it still preloads. w4 is for a tighter box, and it is not free: the
scene and composition survive (same seed, same layout) but the subject is
visibly softer. For scale, the QIE conditioning fix moved PSNR 20.4 → 45.6 dB,
so 45+ is indistinguishable, 33 is close, and 26 is a real degradation.

Perf note, not yet taken: the qmm dispatch here picks BM=32 or BM=64 on the
token count. The kernels also ship a `bm128` arm at group 64, which the
in-tree DiTs select above a size threshold. At LTX's token counts (672 at
768x448, 1024 caption rows) that arm is plausibly faster and is untried.

**Choosing between packs is EXPLICIT.** A quantized directory and the bf16
Comfy file can sit side by side under `diffusion_models/`, and they differ by
3x in footprint. That is too big a difference to decide implicitly in either
direction — silently taking w4 surprises anyone keeping bf16 for quality;
silently taking bf16 makes a deliberate quantize look like it did nothing. So
`VPIPE_LTX25_VARIANT` selects (the host's `VideoModelCreateArgs` carries no
variant field, which is why it is an env var and not stage config), and the
family **always logs which file it loaded and whether it is a quantized pack**.

**The producer side also had to change.** `comfy_output_config` — which
synthesizes the config.json for a pass that reads one Comfy file and writes a
directory — was hardcoded to `minimax_h3` and refused everything else. It now
passes an architecture that names itself (`_class_name`) through **verbatim**,
including the sibling `scheduler` block; lifting only `transformer`, as the H3
branch does, would have written a config that loads with a silently defaulted
sampler.

## Memory residency: the encoder is the lever

LTX-2.5 is unusual among the DiTs in this tree, and reading
`docs/MODEL-MEMORY.md` straight onto it produces two wrong conclusions. Both
follow from one fact:

> **LTX asks for `Mapped` in BOTH arms — and gets `Copied` in both.**
> `stream` selects only whether the WeightSet keeps a cached alias
> (`stream_tensor` vs `tensor`); `bind_block` runs once, at load. But the
> mapping never happens: `MetalLlamaWeights::load_mapped` requires a
> 16-byte-aligned offset within the shard, and this file's data section
> begins at **677624 ≡ 8 (mod 16)**. Safetensors packs tensors contiguously,
> so **all 4349 tensors — 39.1 GB — inherit that and fall back to a copy.**
> The fallback is silent.
>
> CORRECTED: this section previously claimed the shard was mapped either way
> and that the pages were therefore clean and free to drop. It is not, and
> they are not. What proved it was not the code but the box: killing a
> thrashing run released **36 GB of compressor-occupied pages**, and clean
> file-backed pages are never compressed.

So:

- **The `STREAM` / `PRELOAD` verdict barely changes residency.** It changes
  the *accounting*; both arms produce the same anonymous copies.
- **The DiT is ~39 GB of anonymous memory**, reclaimable only through the
  compressor and swap. That is why bf16 on a 64 GB box is marginal rather
  than comfortable, and why the quantized packs help so much more than a
  "policy" framing suggests — they shrink a real allocation.
- **`revise_declaration()` still does not apply** — the model does not pin
  less when streaming, so revising down would report a shrink that did not
  happen.
- **`BlockResidency` now arguably DOES apply**, which it did not when this
  was written: with the blocks held as owned copies there is something real
  to evict. It is still not wired, and that is now a gap rather than a
  deliberate omission. See the note in `ltx25-family.cc`.

### What actually thrashes, and why it is worse on a BIG box

The competition is not DiT-vs-itself, it is **mapped weights vs anonymous
weights**. The Gemma-4 12B text encoder is ~24 GB read `Copied` — anonymous
memory, reclaimable only through the compressor and swap. Beside it, 42 GB of
mapped DiT cannot stay resident on a 64 GB machine, so the kernel evicts DiT
pages every step. MEASURED: ~550 MB/s of sustained re-faulting, ~67 s/step at
768x448.

And the encoder is **idle for all of it**. It encodes one caption in seconds;
the denoise then runs for minutes with 24 GB sitting there.

Worse, the cost compounds through the sizing path. `generate-video` loads its
DiT lazily, on the first conditioning beat, and that is where it takes the
**irreversible** streaming decision. Declarations persist for the whole run at
`max(held, estimate)` — deliberately — so an encoder that is merely dropped is
*invisible*: the DiT still sizes against 65 GB and concludes it must stream.

Hence the two fixes, which are ordering and bookkeeping rather than new
machinery:

1. **`ltx-2.5-conditioner` defaults to `destroy`** (not `keep`, which is what
   a bool defaulting to false meant), and it **revises its declaration to 0**
   when it does. Freeing without revising is invisible to the peer that needs
   to know.
2. **It releases BEFORE publishing the beats.** The DiT starts loading as soon
   as a beat lands, so releasing afterwards leaves it racing a free it cannot
   see — winning on a fast box and losing on a slow one, from one graph. This
   is the same ordering fix `e856fd9` made for the in-tree conditioner.

> **Why the in-tree conditioner does not need the revision, and this one
> does.** `DiffusionConditionerStage::unload_encoder_()` frees without
> revising either -- and that is harmless *there*, because `generate-image`
> loads its DiT in `initialize()`, concurrently with the conditioner's own
> load. Its streaming decision is already taken before a single prompt is
> encoded, which is exactly why pre-barrier `declare_resources()` exists.
> `generate-video` instead loads the DiT **lazily, on the first beat** --
> after the conditioner has encoded and released. That ordering is what turns
> a stale declaration from a harmless leftover into the thing that decides an
> irreversible verdict. Same code shape, opposite consequence; worth checking
> which one a new stage is before copying either.

### Measured, on the 64 GB M4 Pro (bf16, 768x448x9, 8 steps, same seed)

| | before | after |
|---|---|---|
| footprint the DiT sizes against | 65 GB | **41 GB** |
| streaming verdict | `STREAM blocks` | **`PRELOAD`** |
| disk I/O during the denoise | ~550 MB/s sustained | **0.41 MB/s** |
| end to end | ~25 min | **338.6 s** |

**4.4x, on the unquantized model, by freeing an idle encoder and saying so.**
Peak footprint 55.2 GB, max RSS 62.2 GB — high, but no longer re-reading the
model every step.

And the output is **9/9 frames byte-identical** to the pre-fix run. That is
the bar for a memory change: it may cost or save time, and it may not move a
single pixel.

`park` is accepted but reports **0 bytes** for this encoder: parking walks a
weight set's *cached* tensors, and a Gemma LM reads uncached, holding its
weights in its own members. That is logged, not hidden — a park that frees
nothing looks exactly like a park that worked until someone measures the box.

### Quantizing the text encoder

The encoder, not the DiT, is what a 16 GB box runs out of room on.
MEASURED as physical footprint (`vmmap`, not max-RSS — RSS counts shared
file pages and reads ~20 GB high here):

| phase | bf16 encoder | w4g64 encoder |
|---|---|---|
| conditioning peak | **27.1 GB** | **10.6 GB** |
| DiT denoise (w8g64) | 2.6–2.8 GB | 2.6–2.8 GB |

`model-quantize` produces it directly from the released single file:

```json
{"type": "model-quantize", "config": {
  "src_model": ".../text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
  "output_name": ".../text_encoders/gemma4-12b-with-proj-ltx-2.5-w4g64",
  "target": "model.layers.", "bits": 4, "group_size": 64}}
```

`target: "model.layers."` is the whole recipe: it quantizes the 20.3 GB
Gemma backbone and leaves the 1.88 GB embedding table and LTX's own
2.16 GB projection dense. Then set `encoder_variant: "w4g64"` on
`ltx-2.5-conditioner` — a quantized encoder is never picked implicitly,
because every conditioning token comes out of it.

FIDELITY, measured on the emitted cross-attention context against the
bf16 encoder (cosine over the real caption rows):

| | cosine | rel-L2 | size |
|---|---|---|---|
| w8g64 | **0.99996** | 0.0088 | 15 GB |
| w4g64 | **0.99387** | 0.1105 | 9.9 GB |

That grading is the check that matters. Pixels cannot make it: a
diffusion sample moves to a different mode on any conditioning change,
so the w4 run scores 20.9 dB against the bf16 run — about what two
different frames of the same clip score against each other. Use
`VPIPE_LTX25_COND_DUMP=<prefix>` to compare contexts numerically.

Two host changes were needed, and the second was a silent total failure:

- `comfy_output_config` now writes a config.json for a pack whose
  `__metadata__` carries a whole LM config (`gemma_config`), at the top
  level, so the quantized output is an ordinary HF directory checkpoint.
- `MetalGemmaModel` accepts a **dense embedding table over a quantized
  backbone**. Without it the load failed outright; with it half-wired —
  the decode step patched and the PREFILL not — the affine branch read
  three buffers the load never filled and wrote zeros, and zeros survive
  every layer. The conditioning came out at 6e-5 of its norm and
  byte-identical between a 4-bit and an 8-bit backbone, which is what a
  result that does not depend on the weights looks like.

### Weight prefetch, and why it is off by default

The host's MiniMax-H3 hides its streamed block read under the previous
block's GPU work (`VPIPE_H3_NO_PREFETCH` to A/B). The same lever exists
here — `VPIPE_LTX25_PREFETCH=1` warms block *i+1*'s pages between block
*i*'s `commit()` and its `wait()`, depth one, skipping blocks that are
already resident — but it is **opt-in**, because on this box there is
nothing for it to hide.

H3 *reads* block L on every forward, so it always has a read to move off
the critical path. This model does not: `bind_block` runs **once**, at
load, and holds the blocks for the model's lifetime. The only per-forward
cost is pages having been evicted between steps.

`VPIPE_LTX25_BLOCK_PROFILE=1` reports, per block and before it is
encoded, how much of it is still in RAM. MEASURED at 512x320x9, 8 steps,
both packs:

| | step 1 | steps 2-8 |
|---|---|---|
| w8g64 (23 GB, mapped) | 96.2% resident, 17.0 s | **100% resident**, 2.7 s |
| bf16 (39 GB, copied) | 1.0-17% resident, 5.1-6.0 s | **100% resident**, 2.9 s |

So it fires 3-19 times on step 1 and **zero** times after, and the wall
clock moves by less than the run-to-run spread (a genuinely cold-cache
pair went 5962 -> 5036 ms on step 1; a warmer pair went 5066 -> 5129 ms).
Where it should earn its keep is a box the checkpoint does not fit —
which is where H3 measured its own 4.1% — and that case is **not
measured here**: creating it means over-committing the machine during a
DiT run, which is what panicked this box's kernel once already.

Two things the probe found that the prefetch does **not** fix, and which
are the larger numbers:

- **The load is ~50 s per clip**, reading 39 GB at ~1 GB/s, and
  `unload_when_idle: always` means every clip pays it. The real analogue
  of H3's prefetch would be to overlap *that* with the denoise — i.e.
  give this model H3's per-block load/free loop instead of binding all 48
  up front. That is an architecture change, not a scheduling one.
- **Step 1 costs 5-17 s against a 2.7-2.9 s steady step even at 96-98%
  residency.** Whatever that is, it is not weights coming back, so no
  amount of page warming addresses it.

The two packs also differ in kind, which the probe makes visible: the
sharded quantized packs are 16-byte aligned throughout and really are
mapped, while the single-file bf16 is not (see above), so "warming" it is
a decompress rather than a file read.

## The generator

`Ltx25Generator` (`ltx25-generator.{h,cc}`) is the `genai::VideoGenerator`
`generate-video` drives: it owns the schedule, the noise and the denoise loop,
and emits **latents** — `vae-decode` makes the frames, which is why this is
useful before the VAE is ported.

It takes the caption **projected to `cross_attention_dim` (4096) and
pre-connector** on iport0, and runs the DiT's own connector over it
(`caption_proj_before_connector`). Two things it refuses rather than
approximates: a conditioning of the wrong width, and a caption that is not a
whole number of the connector's 128-register tiles.

The **audio stream needs its own 2048-wide context** from the encoder's
separate `audio_aggregate_embed`. Nothing produces one yet, so a request runs
video-only and says so; the block supports that (the reference's `run_a2v` is
conditional on audio being present), and a zero audio context would be
fabricating conditioning rather than declining it.

`GenerationParams::from_flex` parses the `ltx-2.5-model-config` beat **in the
model layer**, which is the contract `stages/model-config-source.h` states — a
knob added later needs a field there and no change in any stage.

## The joint audio-video denoise

The DiT carries both modalities in ONE forward -- there is no "video pass then
audio pass" -- and its audio half was implemented and goldened long before the
generator used it. What was missing was only the loop:

- **The audio token count comes from PIXEL frames**, not latent frames:
  `round(frames / fps * 25)`, where 25 = `sample_rate 16000 / hop 160 /
  latent downsample 4`. Using the video *latent* frame count instead is wrong
  by the temporal compression factor and yields a soundtrack 8x too short.
- **One sigma schedule for both streams.** The reference hands a single
  `sigmas` tensor to `DiffusionStage` and steps both modalities with it; there
  is no per-modality shift anywhere in it. (An earlier note here claimed
  otherwise, citing a "video shift 12" -- that belongs to MiniMax-H3.) The real
  per-modality asymmetry is the GUIDANCE scale, 3.0 video against 7.0 audio,
  which the distilled checkpoint does not use at all.
- **The samplers are reused verbatim.** `to_denoised` and
  `euler_ancestral_step` take a pointer and a count, so nothing about them is
  video-shaped. The audio noise is drawn from a different seed offset so the
  two streams do not start life correlated.
- **The output is TRANSPOSED.** The DiT carries the audio latent as
  `[audio_out_channels][tokens]` with the channel packed `c*mel + f` -- the
  patchifier's `b c t f -> b t (c f)` -- while `Ltx25AudioVaeDecoder` wants
  `[channels][frames][mel]`. Handing over the DiT's layout unchanged decodes a
  perfectly plausible spectrogram from scrambled channels.

**What is and is not verified.** Structural only, and deliberately: shape,
finiteness, a non-zero latent, the video half unaffected, the seed reaching the
audio stream, and the latent decoding through the already-verified audio VAE
(1.7e-3) and vocoder (3.1e-5) to a waveform that is not silence. The audio DiT
output itself is **not** checked against the reference -- a golden would mean
the whole 22B stack in PyTorch. The test's context is synthetic, so "the
soundtrack matches the prompt" is not a claim it makes either.

The prerequisite was the RoPE fix: until positions became seconds, a video
latent frame sat at 0.33 s and an audio token at 0.04 s, so the a2v/v2a
coupling attended to mismatched offsets.

## The stages this plugin ships

### `ltx-2.5-conditioner` — the caption, and why the host's will not do

The host `diffusion-conditioner` serves the image families and MiniMax-H3,
and every one of them makes ONE context from an encoder it knows how to
build. LTX matches neither half: its encoder is Gemma-4 12B with **all 49
hidden states** tapped, then LTX's per-token RMS, layer-fastest flatten and
a 188160-wide projection; and it produces **two** contexts, 4096 for video
and 2048 for audio, from the same hidden states through two projections. A
one-context stage cannot emit the second, and that is why `generate-video`
ran LTX video-only. Adding a fourth family branch to the host conditioner
would have put LTX's projections in the host, which is what this plugin
exists to avoid.

```
iport0 prompt      oport0 conditioning            video 4096
iport1 negative    oport1 neg_conditioning        video 4096
iport2 model       oport2 audio_conditioning      audio 2048
                   oport3 neg_audio_conditioning  audio 2048
```

**The 0/1 ordering is positional with `generate-video`'s iports**, matching
`diffusion-conditioner`. The audio context started on oport1, which made the
obvious wiring (0→0, 1→1) feed a 2048-wide audio context into the port that
wants a 4096-wide negative.

**The negatives are written BEFORE the positives, and that is load-bearing.**
`generate-video` reads its positive with a blocking read and then POLLS its
negative (`backlog(1) > 0`), because a graph with no negative would otherwise
deadlock; its own comment records that this is safe only because the
conditioner emits the negative first. Emitting the positive first makes the
poll a race it usually loses — silently, since the run then proceeds unguided
at CFG 1, which reads as a weak negative prompt rather than a dropped one.

A negative is **inert on the distilled checkpoint** (guidance-distilled: no
unconditional pass to blend with), so it is reported once and not encoded —
each negative costs a second full 12B forward.

**The sideband is not optional.** LTX pads LEFT, so the real tokens are a
SUFFIX; every beat carries `valid_begin`/`valid_count`/`padding_side`,
because every other conditioner in this tree emits a prefix and a consumer
that assumes one keeps precisely the empty rows.

### The VAE, through a host registry rather than a stage

The host `vae-decode` picks its decoder from a **hardcoded family chain**
(wan / minimax-h3 / flux2 / mage) keyed on the VAE config's `_class_name`,
and each branch constructs a decoder the host has the code for. There is no
seam an out-of-tree family can join.

The first version of this plugin worked around that with a stage of its own.
That was replaced by a **host extension point**, `register_vae_family`
(`generative-models/vae-model-registry.h`), the counterpart to
`register_video_family`: a plugin adding a video model registers **both** --
one family makes the latent, the other decodes it -- and needs a stage for
neither. LTX's graphs use the stock `vae-decode`.

The seam is drawn at **one decode**: given a latent and its sideband, produce
RGB frames. Un-whitening, tiling, colour space and residency live inside the
family, because each built-in already does them differently. The stage keeps
the ports, the U8 quantisation, the per-frame beat and the idle-unload policy.

Decode is **sink-callback** based, not return-a-buffer, and that is the one
real design choice here. A sink is a strict superset: a decoder that can only
produce a whole clip calls it once, while one that can stream bounds its own
peak. Return-a-buffer forces peak = whole clip on everyone -- 121 frames at
768×512 f32 is 570 MB. It also degenerates cleanly to an image VAE (one call,
`n = 1`) and gives cancellation for free, since the sink returns `bool`.

Two things a naive port of the built-in flow would get wrong, both of which
the interface has to expose:

- **`claims` is given the root AND `resolve_vae_dir(root)`.** That resolver
  knows two layouts and returns the root unchanged for anything else, which
  is every Comfy pack. LTX's VAE is `vae/*video-vae-conv*.safetensors` with
  its config in `__metadata__`, so the claim uses the root — and the probe
  has to run *before* the stage's `open_weight_set`, which would otherwise
  index the 39 GB DiT sitting in the same tree.
- **`idle_peers` names the Comfy directories.** The stage's own guess is the
  diffusers spelling (`transformer/`, `text_encoder/`), which sums to **zero**
  on this pack — and a zero peer footprint reads as "the box is roomy, keep
  the VAE resident", beside a 39 GB DiT.

### Three things the stage contract wants that the spec alone does not give

Each of these cost a launch failure, and none is visible in the StageSpec:

- `allocate_oports(kSpec.oports.size())` must be called in the constructor.
  The spec is documentation; skip the call and the stage reports **zero**
  oports, so every downstream edge is refused as "oport 0 out of range".
- Port **tags** must match the consumer's. `generate-video`'s iport0 is
  tagged `conditioning`; `tensor` is refused at launch with a tag mismatch.
- Whether an optional iport is WIRED is a runtime-graph fact
  (`ctx.num_iports()` / `ctx.iport_connected(p)`), not a constructor one, so
  a "nothing to work with" refusal belongs at the first beat rather than in
  `fail_config`.

## Port order

Bottom-up, each verified before the next — the order that worked for
MiniMax-H3 and Wan:

1. config + loader off the comfy `__metadata__` (**done**), and the RoPE
   (**done**, 1.8e-8 vs the reference)
2. video VAE (conv) **decode** — rel-L2 against a reference latent→pixels
3. text encoder — reuse `gemma4_unified`, add the 49-layer aggregate projection
4. DiT block, one layer, against a captured reference activation
5. full DiT + the 8-step distilled schedule → latent velocity in **latent
   space** (the QIE lesson: image-space comparisons are artifacts)

**One whole AV block now runs on the GPU**, at 8.8e-3 video / 5.2e-3 audio
against both the CPU reference and the golden — bf16 accumulated over ~40
dependent ops. The two comparisons landing on the same figure is the useful
part: it says the Metal path and the reference agree, so what is left is
precision, not a transcription bug.

The plugin's own Metal kernels are **done and GPU-verified** at 1.7e-3
(bf16 round-off): `ltx_rope_half_perhead`, `ltx_gate_heads`,
`ltx_rms_norm_gain`. Only those three are needed — `load_library()` reaches
libvpipe's built-in metallibs from a plugin, so the GEMMs (`dense_gemm_bf16`),
flash attention (`attn_steel`), RMSNorm, and `llm_elementwise_bf16`'s adaLN
modulation / gated residual / gelu-tanh / bias-add / im2col are all reused.

The **48-block stack** is written (`ltx25-dit.{h,cc}`): the six adaLN MLP
chains, patchify, the block loop and the output head. Three things about it
are deliberate and worth knowing:

- the **adaLN MLPs run on the HOST in f32**. They act on ONE row — a single
  timestep — so the whole chain is ~0.2 GFLOP against the stack's ~40 TFLOP,
  and a bf16 GEMV there would round the value that drives every modulation in
  every block;
- the block loop **commits per block**, not as one command buffer: 48 × ~60
  dispatches is ~2900 encodes, and one buffer that large both delays the first
  work and makes a mid-stack abort impossible;
- the output head uses **LayerNorm** (`elementwise_affine=False`), not
  RMSNorm — the only mean-subtracting normalisation in the model.

The connectors are implemented and **optionally owned** by the DiT: loading
with them makes `forward` take the raw projected caption, loading without
makes it take the context already connected. That is a memory choice, not a
style one — the video connector is ~3.2 GB on top of 39 GB, and a pipeline
that connects once then denoises for 8 steps should not hold it throughout.

`keyframes_abs_pos_embedding` is applied, and it is NOT an i2v-only
detail — that was this port's reading of it and it was wrong. The
reference marks the target's FIRST LATENT FRAME unconditionally
(`LatentTools._first_frame_keyframes_mask`), because a causal video VAE
makes that frame span one pixel frame while every later one spans eight,
and the learned marker is added to those tokens right after
`patchify_proj`. The checkpoint's copy is [1, 4096] and **fully
non-zero** (4096/4096), so every generation this port ran before the
conditioning work was missing a trained embedding. The marked set is
contiguous at the front of the sequence, so it is applied as a row
PREFIX rather than a mask; appended conditioning tokens are never marked
(only generated keyframe SLOTS are, which this port has none of).

## Conditioning: the image / video / audio reference paths

LTX-2.5 has **no separate image-to-video model and no extra weights**. A
reference is a per-token `denoise_mask`: the timestep the adaLN chain
sees is `denoise_mask * sigma`, so a token holding given content is
modulated as clean while its neighbours are modulated at the schedule's
sigma, and the sampler keeps forcing those tokens back to the content
they were given. The same mechanism serves an image anchor, a video
prefix, a closing keyframe and a reference soundtrack.

**Levels, not a mask per token.** The mask takes a handful of distinct
values — 1.0 for what is being generated, `1 - strength` per conditioning
item — so the port carries the distinct values plus one index per token,
and runs the adaLN chain once per LEVEL. A per-token chain would be a
4096×36864 projection thousands of times over instead of two or three.
The per-token modulation is checked BIT-IDENTICAL against the broadcast
path in `ltx25-block-metal-test` (every token on level 0, every token on
level 1 with the other row poisoned, and a mixed index over two identical
levels).

**Two placements, and why a closing keyframe cannot use the first.** A
latent belonging at the START of the clip overwrites the target grid in
place — its tokens exist and already carry the right positions. A latent
belonging LATER cannot: the grid's later frames each span the VAE's whole
temporal stride, so writing a still image into latent frame k says the
picture holds for eight frames. It rides as APPENDED tokens carrying the
positions of the pixel frame it belongs to, and its velocity is thrown
away. A reference soundtrack appends the same way, at NEGATIVE seconds,
so it cannot collide with the soundtrack being generated on the one axis
the two streams share.

All of it is pinned by `tests/ltx25-conditioning-test.cc` against goldens
taken from the reference's own `VideoConditionByLatentIndex`,
`VideoConditionByKeyframeIndex` and `AudioConditionByReferenceLatent`
(gen_goldens.py `cond`) — at EXACT equality, because the mechanism is
bookkeeping over latents and needs no weights and no GPU to check.

The block itself (step 4) is **done as a CPU reference** — `ltx25-block-ref`,
verified at 3e-7 rel-L2 against the reference on the `small` golden, joint and
both single-stream. That is deliberately the first half of step 4: it settles
every ORDERING and INDEXING question (which of the nine adaLN rows drives
what, that the AV cross tables are scale-then-shift where the self-attention
triple is shift-then-scale, that both cross directions read the pre-cross
snapshot) with no Metal involved. The Metal forward is then checked against
this locally, so a kernel bug and a semantics bug can never be debugged at the
same time through 48 blocks of a 22B model.
6. audio VAE + vocoder
7. i2v / keyframe conditioning, then the diffusion decoder and the upscalers

## Goldens

`~/dock/dump/vpipe-test/ltx25/gen_goldens.py` runs the reference against the
released checkpoint and writes `.npy` + a `manifest.json` naming every shape
and the inputs it came from. It loads tensors lazily out of the 39 GB DiT, so
a one-block golden costs a few hundred MB, not the whole file.

```sh
~/dock/dump/hf-venv/bin/python gen_goldens.py all
```

Everything is generated with `generate_freq_grid_np`, the f64 ladder this
checkpoint's `frequencies_precision` selects — not the reference's default.

Present today:

- **`rope`** — the frequency table and the rotation. Pure, no weights.
- **`block`** — transformer block 0 with the checkpoint's own layer-0
  weights: 24 video tokens x 8 audio x 7 text. The loader refuses to write a
  golden if any parameter name is missing, because `strict=False` plus random
  weights produces a golden that looks fine and matches nothing.
- **`small`** — a whole AV block at dim 64/32, 4 heads, random seeded
  weights, with **every weight dumped** (98 tensors, a few hundred KB). This
  is the one a from-scratch port diffs against: block 0's weights are 1.2 GB,
  so they cannot serve that purpose. It also emits video-only and audio-only
  runs of the same block, which is what separates "the block is wrong" from
  "the cross-attention is wrong" — a port that mishandles the pre-cross
  snapshot still matches the single-stream runs.

Still to add, in port order: VAE decode, the encoder's 49-layer aggregate
projection, and a full 8-step latent trajectory.

## Traps already identified

- **All-49-layer aggregation**, not a sparse tap. Wrong order = wrong
  conditioning, silently.
- **The pre-cross snapshot** in step 4 of the block. Reading the post-a2v `vx`
  for v2a is a one-line mistake with no shape consequence. *(Pinned by the
  `small` golden's three runs.)*
- **The AV cross tables unpack SCALE FIRST, then shift** — `scale, shift =
  get_av_ca_ada_values(...)` — where the self-attention triple is `shift,
  scale, gate`. Two adjacent conventions in one block, and swapping them is a
  plausible-looking image.
- **Fractional [-1, 1] RoPE positions with a midpoint grid** — nothing like the
  RoPE in any other model in this tree. *(Implemented and verified: 1.8e-8
  rel-L2 vs the reference.)*
- **`frequencies_precision` is part of the model, not a quality setting.** It
  selects between two different generators in the reference, and LTX-2.5 asks
  for `float64`: build the ladder in double, then **narrow to f32 before
  forming the angles**. The frequencies run up to `theta*pi/2` ≈ 15708, so a
  relative slip of 1e-7 is an *absolute* angle error of ~1e-3 rad, and cos/sin
  of an angle that large are pure phase. Measured against the reference:
  correct = 1.8e-8, staying in double past the narrowing point = 1.3e-4, using
  the f32 ladder = 6e-4. All three look fine under a loose threshold. My first
  golden used the wrong generator and had to be regenerated.
- **The tokenizer is a tensor**, not a file.
- **`ff.net` has no bias on the video stream** but does on audio
  (`ff_bias: false`, `audio_ff_bias` unset → true).
- Only **40 of 48** `v_proj` tensors exist in the text encoder.
- **The 290 scale/shift tables are F32, everything else bf16.** Convert at
  bind (a `derived()` transform, ~19 MB across all 48 blocks) — see above.
- **The prompt adaLN MLP is ON.** Omitting `prompt_timestep` leaves the text
  cross-attention's K/V modulation constant across the schedule. It costs
  nothing in shape and produces a self-consistent golden — this port
  generated one before catching it, the second time a golden had to be
  regenerated for a default that was not the model's.
- **The sampler is the ancestral one, not `res_2s`** — and precision
  placement decides the last digit in the RoPE ladder, the K/V modulation
  default, and the sampler's scalars. Three separate places where "compute it
  more accurately" means "compute a different model".
- **`AdaLayerNormSingle` has THREE linears, not two**:
  `emb.timestep_embedder.linear_1` (256→dim), `.linear_2` (dim→dim), then
  `linear` (dim→k·dim) at the adaLN *root*, outside `.emb`.
