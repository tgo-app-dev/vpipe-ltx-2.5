# vpipe-ltx-2.5

[LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) — Lightricks' 22B joint
**audio-video** world model — as a **vpipe plugin**: a separate `.so` built
against an installed vpipe, not a fork of the tree.

It is a plugin because the weights are under the **LTX-2.x Community
License**, which is not vpipe's Apache-2.0. Keeping the model out of the
vpipe tree keeps each under its own terms.

## Start here

**[docs/LTX-2.5.md](docs/LTX-2.5.md)** — what the model is, how to fetch it,
and how to generate a clip with sound, from a picture or from a reference
soundtrack. Six ready-to-run pipelines live in
[docs/pipelines/](docs/pipelines/): two that prepare a checkpoint and four
that generate. Read that first; the rest of this file is how the port is
built and what it is made of.

## Status

**It generates.** Prompt to frames and a soundtrack, end to end, through the
stock `generate-video` / `vae-decode` / `audio-vae-decode` stages on a 64 GB
M4 Pro: 9 frames at 768x448 of "a red fox walking through snow at dawn", plus
0.53 s of 48 kHz stereo, in 338 s on the bf16 checkpoint.

**At the geometry the model is built for** — 960 x 544, 121 frames (5.0 s at
24 fps) with its 5.04 s soundtrack, on the w8g64 pack — the same box takes
**532 s end to end** through the shipped text-to-video pipeline: 410.9 s of
denoise (51.4 s/step over the distilled checkpoint's 8 steps), plus the model
load, both VAE decodes and the mux. Out comes a **1.1 MB mp4 -- h264
960x544 at 24 fps, AAC 48 kHz stereo, 5.04 s** -- from `save-video` with
`enable_audio`, not a directory of frames.

> The **8 steps are the checkpoint's**, not a setting. The distilled DiT
> ships a fixed schedule, so a `steps` of 6 is reported and ignored rather
> than silently honoured.

**It also generates FROM an image.** `load-image -> vae-encode ->
generate-video` anchors the clip to a reference picture: at 512x320x9,
seed 1234, the anchored frame 0 comes back at **31.1 dB** against the
letterboxed reference and decays to 26.3 dB by frame 8 as the model
animates — where the same seed and prompt with the anchor UNWIRED scores
**6.8 dB**, i.e. a different picture entirely. See
[docs/pipelines/ltx-2.5-image-to-video.vpipeline](docs/pipelines/ltx-2.5-image-to-video.vpipeline).

**And from a reference soundtrack.** `load-audio -> audio-to-pcm ->
audio-vae-encode` turns any audio file into the latent rows `generate-video`
takes on `ref_audio_rows`. Run end to end at 512x320x9: a 0.40 s reference
became 10 appended audio tokens, the DiT denoised 19 audio tokens where an
unreferenced run has 9, and the clip came out coherent with its soundtrack.
That is also the first exercise of the APPENDED-token conditioning path on the
real 22B model -- the same machinery the closing-frame anchor uses.

**And it upscales in LATENT space.** `ltx-2.5-latent-upscale` doubles a
generated latent before it is decoded -- no VAE round trip, one convolutional
forward pass -- so a 960x544 generation is saved at 1920x1088. Measured at
**26.9 dB** against a Lanczos 2x of the same clip, at that geometry and at
1280x704 -> 2560x1408. `mode` picks the axis; chain two stages to double both.
See
[docs/pipelines/ltx-2.5-spatial-upscale.vpipeline](docs/pipelines/ltx-2.5-spatial-upscale.vpipeline).

**And it takes the host's three acceleration tiers**, all off by default and
all read off `generate-video`'s acceleration bag rather than spelled again
here — so a graph asks every family for them the same way, and a tier added
to the host later needs no rebuild of this binary.

| tier | what it changes | on this port |
|---|---|---|
| `sol_attn` | which key blocks the video self-attention READS | **1.20x denoise, 1.14x end to end** at 960x544x121 on an M4 Pro w8g64, keeping 23% of key blocks exact over 47 routed blocks |
| `sage_attn` | how the blocks that are read are COMPUTED — QK^T in int8 | applies to EVERY attention here, self, text-cross and both audio/video directions. Needs matrix cores; on a box without them it declines and the model runs dense |
| `i8_gemm` | the block's big matmuls in int8 | same condition |

Sol is lossy and the two clips are **different clips, not one degraded** — a
diffusion sampler is chaotic, so a perturbation lands on another sample from
the same distribution. 17.9 dB against the dense clip, both coherent. Judge
it by looking, not by PSNR against a dense run.

Verified against the reference implementation:

| piece | bar | measured |
|---|---|---|
| RoPE table + rotation | f32 round-off | **1.8e-8** |
| whole AV block (CPU ref), joint + both single-stream | f32 round-off | **3.0e-7** |
| block 0 of the REAL 22B checkpoint | bf16 at 4096/2048 width | **6.4e-3 video / 3.9e-3 audio** |
| the full 48-block DiT stack | bf16 over 48 blocks | **6.7e-3 -> 1.8e-2**, smooth in depth |
| conv video VAE decoder (CPU ref / Metal) | f32 / bf16 | **1.0e-6 / 3.8e-3** |
| audio VAE decoder | bf16 | **1.7e-3 – 6.0e-3** |
| BigVGAN vocoder (f32) | f32 round-off | **3.1e-5** |
| BWE 16 -> 48 kHz | f32 round-off | **2.7e-5** |
| conv video VAE **encoder** (CPU ref / Metal) | f32 / bf16 | **2.2e-6 / 1.2e-2** |
| audio mel front end (slaney filter bank / log-mel) | f32 round-off | **2.5e-6 / 4.2e-7** |
| audio VAE **encoder** (whitened rows) | bf16 over ~20 convs | **4.0e-2**; round trip through the decoder correlates **0.90** |
| the conditioning bookkeeping (mask, clean latent, appended positions) | exact | **0.000e+00** |
| quantized linear vs dense GEMM | bit-exact | **0.000e+00** (6 configs) |

Quantization, block streaming, the joint audio-video denoise, the two VAE
families and the adaLN bake are all in and measured — see ARCHITECTURE.md for
the numbers and, more usefully, for the traps each one hid.

### What is NOT done

Stated plainly, because everything above invites the assumption that the port
is finished. It is not:

- **A multi-frame VIDEO reference has no producer.** The generator
  takes any number of latent frames on `ref_latent0` and treats them as a
  prefix the clip continues from; the stock `vae-encode` supplies exactly
  one image per beat, for every family, so today that path is reachable only
  with a latent from somewhere else.
- **Classifier-free guidance is written and has never run.** Only the
  `distilled` checkpoint is on disk, and it is guidance-distilled (CFG 1), so
  the negative-prompt path — the conditioner's two negative oports and the
  generator's CFG branch — compiles and is unexercised. It needs the `dev`
  checkpoint.
- **The audio DiT output is unverified against the reference.** The bar taken
  was structural (shape, finite, the schedule moves it, two seeds differ,
  decodes to non-silent audio). A real golden means the whole 22B stack in
  PyTorch.
- **The closing-frame anchor is written and has never run.** `ref_latent1`
  appends a keyframe at the clip's last pixel frame, sharing every line of
  machinery with the first-frame path and pinned by the same golden, but no
  end-to-end run has wired two encoded images.
- **The diffusion VAE decoder is not ported** — only the conv one. It is a
  second sampling loop.
- Refused rather than approximated, and so also "not done": audio VAE `attn`
  blocks, and any `rope_type` other than `split`.
- **The int8 tiers have never run on matrix hardware.** `sage_attn` and
  `i8_gemm` both need a matrix-core GPU; the box this port was written on
  has none, so on it both tiers DECLINE and the model runs dense. Their
  tests pass by exercising the decline. What is unproven is the arm that
  does the work.
- `duration_head` is a config passthrough, not an implementation. It is
  parsed, carried and read by nothing.
- Perf lever untried: the `bm128` qmm arm at LTX's token counts.

## Building

Needs an installed vpipe with **plugin ABI 5** — the host loads a plugin on
STRICT equality, so this is an exact requirement, not a minimum. Rebuild
against the vpipe you deploy with; a mismatch is refused with a clear
message rather than crashed.

ABI 4 is what `VideoModelFamily::denoise_scratch_bytes` needs, and it also
carries the `MetalCompute::MemoryBudget` layout read by value. ABI 3 was
the acceleration bag. The tiers below arrive as one
open `FlexData` rather than as typed fields on the request, which is the
change that stops the NEXT tier invalidating this binary — but the bag
itself was a version bump, and a host older than that cannot describe it.

```sh
# in the vpipe tree
cmake --build ../vpipe-build -j
cmake --install ../vpipe-build --prefix /path/to/vpipe-install

# here
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/vpipe-install
cmake --build build -j
```

## Running

```sh
vpipe --plugin build/vpipe-ltx-2.5.so --list-models          # entries appear
vpipe --plugin build/vpipe-ltx-2.5.so --launch-stage ltx-2.5-model-config \
      --stage-cfg variant='"dev"' --stage-cfg guidance=3.5
vpipe --plugin build/vpipe-ltx-2.5.so --launch-stage generate-video \
      --stage-cfg hf_dir='"/path/to/Lightricks/LTX-2.5"' \
      --stage-cfg height=544 --stage-cfg width=960 --stage-cfg frames=121
```

### Ask for the picture you want

**Size and frame count round UP to what the model can tile.** LTX-2.5's VAE
compresses space by 32 and time in chunks of 8 (`8k + 1` frames), so a
literal reading of the checkpoint would have you computing legal geometries
by hand. You do not have to:

```
frames 120 -> 121, the nearest count at or above it that the ltx-2.5 VAE can chunk
953x550 -> 960x576, the nearest size at or above it that the ltx-2.5 VAE and DiT patch can tile
```

Both are **reported, not silent** — the clip that comes back is a different
shape from the one asked for, and a graph downstream would otherwise discover
that as a surprise. Rounding up rather than rejecting is deliberate too: a
graph can be pointed at a different model family without being re-authored.

## Shrinking the DiT

The bf16 DiT is 42 GB, which on a 64 GB box means block STREAMING — 37 GB
re-read from disk *every step*, about 67 s/step at 768x448. Quantizing it
removes that entirely, and the way to do it is the shipped pipeline:

```sh
vpipe --plugin build/vpipe-ltx-2.5.so \
      --launch docs/pipelines/prepare-ltx-2.5-8bit.vpipeline
```

That fetches the checkpoint and packs **both** components into one
self-contained model, `local/LTX-2.5-distilled-8bit`. It is idempotent, so
re-running after an interruption resumes rather than repeating a finished
half.

**The recipe is the plugin's, not the command line's**, and
`register_quantize_family` is what supplies it. `target: "dit"` selects
both block stacks — the DiT's
own `transformer_blocks.` and the two text connectors' `transformer_1d_blocks.`
— and the exclusions come with it. Hand-rolling the scope is how the
connectors used to be missed, which shipped 3.75 GB of dense bf16 inside a
pack whose whole point was to be small, and which never streams: the trunk is
resident for the entire denoise.

The exclusions are worth knowing even though you no longer type them. The
wholesale rule takes every 2-D floating-point tensor whose leaf is not a norm
or an embedding, which over a block stack also catches the f32
`scale_shift_table`s (modulation tables, not matrices) and the `[32, dim]`
gate logits. Quantizing either produces a checkpoint that loads fine and
generates the wrong thing.

Measured on the 22B distilled DiT alone (same seed, 768x448x9):

| pack | size | residency | PSNR vs bf16 |
|---|---|---|---|
| bf16 | 42.0 GB | STREAM, ~67 s/step | — |
| **w8g64** | 23 GB | PRELOAD | **33.29 dB** |
| w4g64 | 14 GB | PRELOAD | 25.87 dB |

### How the blocks are held

A box that can hold the pack **preloads** it and reads the blocks `Mapped` —
clean file pages, which the kernel drops for free and re-reads cheaply, and
which cost the process almost nothing to keep.

A box that cannot **streams**: a leading prefix of blocks stays resident, the
rest are read per forward and dropped, and each read is issued under the
previous block's GPU work so it is not on the critical path. Those blocks are
read `Copied`, because a pinned prefix means nothing if the kernel can evict
it — and because a forward is a cyclic scan, the one access pattern an LRU
page cache handles worst: every block dropped exactly before it comes round
again. As free RAM allows, streamed blocks are promoted back to resident, and
if the measurement says the resident set is being squeezed out of RAM they
are handed back one at a time.

The two are not interchangeable, and the split is measured rather than
argued. At 960x544x121 on the 64 GB box, in one paired run, mapping the
preloaded pack takes **409.7 s** of denoise where owning the same bytes takes
**473.5 s** — the
copies turn into 36 GB of compressor traffic, and the model's own weights
drop to 2-3% resident. On a box that cannot hold the pack the choice inverts,
which is what the streaming arm is for.

Both widths escape streaming, so the bit width is a quality/size trade with
the throughput win already banked. **w8 is the sensible default on a 64 GB
box**; w4 is for a tighter one and visibly softens the subject.

Where a pack sits beside the bf16 file rather than in a model of its own,
pick it explicitly — a bf16 file and a quantized directory can sit side by
side, and the choice is never implicit:

```sh
VPIPE_LTX25_VARIANT=w8g64 vpipe --plugin build/vpipe-ltx-2.5.so --launch graph.json
```

The family logs which pack it loaded on every run.

## What it plugs into

| extension point | what this contributes |
|---|---|
| `register_video_family` | the `ltx-2.5` family for `generate-video` — detection, the frame rule, the resource declaration and the whole joint audio-video denoise |
| `register_vae_family` | the `ltx-2.5` VAE for the stock `vae-decode`, `vae-encode`, `audio-vae-decode` and `audio-vae-encode` — the conv video codec, and the audio codec with its vocoder and bandwidth extension behind it |
| `register_quantize_family` | the recipe `model-quantize` packages this checkpoint with: which component lives where, and which of its tensors are matrices |
| `register_stage` | `ltx-2.5-model-config` (this family's own knobs), `ltx-2.5-conditioner` (the caption, and the audio context beside it), `ltx-2.5-latent-upscale` (the x2 latent upscalers) |
| `register_metal_library` | three metallibs — the same kernels compiled for bf16, f16 and f32, because the vocoder runs f32 where the DiT runs bf16 |
| `register_catalog_entries` | five entries: distilled, dev, the distilled LoRA, the x2 pixel-spatial IC-LoRA, the x2 latent upscalers |

Six of them, which is the point worth taking from the table: **no stage of
this plugin's own touches a pixel or a sample.** Generation goes through the
host's `generate-video`, both codecs through the host's four VAE stages, and
packaging through the host's `model-quantize`. The three stages here carry
knobs and the latent upscaler, nothing else.

The knobs are a **stage** rather than keys on `generate-video` for the
reason `stages/model-config-source.h` gives: a stage serving several
families accumulates the union of their knobs, and each is inert —
*silently* — on whichever family is not resident. Nothing in
`ltx-2.5-model-config` means anything to Wan or MiniMax-H3.

## Adapters

`ltx-2.5-model-config`'s `lora` key takes a registered model key, a
directory holding one `.safetensors`, or a path to one. Both published
adapters load: the **distilled-450** (1660 modules, mixed rank -- 450
for the projections and 32 for every gate in one file) and the **x2
pixel spatial upscaler** (480 modules, rank 32).

**Applied at RUNTIME, never fused.** The delta rides on each adapted
linear's output as `y += (x @ A^T) @ B^T`, so the base weights are
untouched. That is not a preference:

* the DiT **streams** its 48 blocks, and a fused weight is a new tensor
  that must be cached to be worth anything -- caching the fused set is
  ~25.8 GB, which would turn a model that runs on a 16 GB box into one
  that does not;
* a **quantized** pack holds u32 codes, so there is no bf16 weight to
  add a delta to at all.

The cost is `2 * rank / K` of the linear it rides on: about 1.6% at rank
32, ~22% at the distilled adapter's rank 450.

### IC-LoRA: the reference clip

An **in-context** LoRA is conditioned on a reference clip carried in the
sequence beside the target. With one loaded, `generate-video`'s
`ref_latent0` changes meaning -- it stops being a first-frame anchor that
overwrites the clip and becomes a whole reference clip appended next to
it, encoded at `1/reference_downscale_factor` of the output's
resolution. The factor comes from the adapter's own metadata, which is
what makes the two impossible to confuse.

Each reference token then stands for `factor x factor` of the target's
cells, so its spatial position spans are multiplied by the factor. Get
that wrong and the run is clean and the whole reference sits in the
top-left corner of the frame.

Three things are refused rather than warned about, because each of them
otherwise finishes and returns a plausible clip that ignored its input:

* an IC-LoRA with **no reference wired**;
* a reference at the **wrong resolution** for its factor, or an output
  whose latent grid the factor does not divide;
* an adapter declaring `reference_temporal_scale_factor > 1`, whose
  reference runs at a lower frame rate than the target -- the temporal
  re-spacing that needs is not implemented here.

## Layout

```
the DiT
  src/ltx25-config.{h,cc}            comfy __metadata__ -> LtxConfig; frame rule
  src/ltx25-rope.{h,cc}              LTX's rotary embedding (verified 1.8e-8)
  src/ltx25-block-ref.{h,cc}         CPU reference for one AV block (3.0e-7)
  src/ltx25-metal-ops.{h,cc}         the op vocabulary the block is written in
  src/ltx25-block-metal.{h,cc}       one AV block on the GPU
  src/ltx25-dit-weights.{h,cc}       bind the checkpoint straight to the GPU
  src/ltx25-dit.{h,cc}               the 48-block stack, adaLN chains, head
  src/ltx25-connector.{h,cc}         the two 8-layer caption resamplers
  src/ltx25-lora.{h,cc}              adapters, bound at load and applied at RUN
  src/ltx25-sampler.{h,cc}           the ancestral Euler step + distilled sigmas
  src/ltx25-conditioning.{h,cc}      denoise mask, clean latent, appended rows
  src/ltx25-generator.{h,cc}         the joint audio-video denoise loop
conditioning
  src/ltx25-text-encoder.{h,cc}      the Gemma caption encoder
  src/ltx25-text-features.{h,cc}     49 hidden states -> the two contexts
the codecs
  src/ltx25-vae-config.{h,cc}        both VAEs' geometry out of the checkpoint
  src/ltx25-vae-ref.{h,cc}           CPU ref for the conv video VAE (1.0e-6)
  src/ltx25-vae.{h,cc}               the conv video VAE on the GPU
  src/ltx25-audio-vae.{h,cc}         the audio VAE decoder -> log-mel
  src/ltx25-audio-encoder.{h,cc}     its encoder, and the mel front end
  src/ltx25-vocoder.{h,cc}           BigVGAN, mel -> 16 kHz (f32, 3.1e-5)
  src/ltx25-bwe.{h,cc}               bandwidth extension, 16 -> 48 kHz stereo
  src/ltx25-upscaler-ref.{h,cc}      CPU reference for the x2 latent upscalers
  src/ltx25-upscaler.{h,cc}          them on the GPU
the seams
  src/ltx25-family.{h,cc}            VideoModelFamily: claims/align/declare/load
  src/ltx25-vae-family.{h,cc}        VaeModelFamily: all four codec roles
  src/ltx25-quant-family.{h,cc}      QuantizableFamily: the quantize recipe
  src/ltx25-model-config-stage.{h,cc}  the ltx-2.5-model-config stage
  src/ltx25-conditioner-stage.{h,cc}   the ltx-2.5-conditioner stage
  src/ltx25-upscale-stage.{h,cc}       the ltx-2.5-latent-upscale stage
  src/ltx25-catalog.{h,cc}           the catalogue entries
  src/ltx25-plugin.cc                the three-symbol handshake
src/ltx25-kernels.metal              the 47 kernels libvpipe does NOT have
tests/                               32 ctest targets
ARCHITECTURE.md                      what the checkpoint IS, and the port order
```

**The CPU reference is not a fallback** and must never run on a hot path.
It exists so that ordering and indexing questions — which of the nine adaLN
rows drives what, whether both cross directions see the pre-cross snapshot —
are settled with no kernels involved. The Metal path is then checked against
it locally, so a kernel bug and a semantics bug are never debugged together
through 48 blocks of a 22B model.

**47 kernels are the plugin's own**, and the split by area says where the
work went: 13 for the BigVGAN vocoder, 11 for the video VAE, 7 for the audio
VAE, 4 for the latent upscalers, and 12 for the DiT block itself. That last
group is the one the port started with — `ltx_rope_half_perhead`,
`ltx_gate_heads`, `ltx_rms_norm_gain`, the fused `ltx_ada_zero` (the block's
most-used op; composing libvpipe's `rms_norm` and `adaln_modulate` would
write a full `[tokens][dim]` intermediate six times per block), and the
handful of copies and gated adds beside them.

Everything else — GEMMs, quantized GEMMs, full and flash attention, adaLN
modulation, gated residual, gelu-tanh, transpose, im2col, group norm — is
reached by NAME out of libvpipe's embedded libraries, which is why the
plugin is a few megabytes rather than a second copy of the kernel tree. The
47 are compiled three times, for bf16, f16 and f32: the DiT runs bf16 and
the vocoder runs f32, and the reference documents the f32 as load-bearing
rather than cautious.

## Licence

This plugin's source is Apache-2.0 — see [`LICENSE`](LICENSE), and
[`NOTICE`](NOTICE) for the attribution a redistribution carries with it.
The **LTX-2.5 weights are not** — they are under the
[LTX-2.x Community License](https://github.com/Lightricks/LTX-2/blob/main/LICENSE.md)
(free for commercial use under $10M annual revenue; a paid agreement
above that). Loading the plugin logs both.
