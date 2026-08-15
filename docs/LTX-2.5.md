# LTX-2.5 on Apple Silicon

**LTX-2.5** is a 22-billion-parameter video model that generates a clip **and
its soundtrack together**, from a single prompt. vpipe runs it on-device
through its **metal-compute** backend — its own Metal kernels, no Python and
no third-party tensor runtime in the forward pass.

It is also the first model vpipe runs **entirely out of tree**. Everything
below — the transformer, both VAEs, the vocoder, the text conditioning, the
catalogue entries — ships in a plugin `.so`, loaded by a host that has never
heard of LTX. Nothing in vpipe was special-cased for it.

The *together* is the same idea MiniMax-H3 has, arrived at differently: video
and audio are two latent streams stepped on **one shared sigma schedule**,
coupled block by block through audio→video and video→audio cross-attention.
The sound is generated with knowledge of the picture rather than dubbed onto
it.

The shipped checkpoint is **guidance-distilled**: no second unconditional
pass per step, and output in **8 steps**.

## What you need

| | |
|---|---|
| **Machine** | Apple Silicon Mac (M-series). |
| **Memory** | **64 GB** for the bf16 checkpoint as shipped. Less needs the quantized pack — see [Memory](#memory). |
| **Disk** | **~65 GB** to fetch; **+23 GB** if you quantize. |
| **Build** | An Apple Silicon build of vpipe, plus this plugin built against it. See the [README](../README.md). |

### Disk space

The published pack is split by component, and the catalogue pins exactly the
five files this port reads:

| | |
|---|---|
| DiT (`diffusion_models/`, bf16) | **42 GB** |
| Text encoder (Gemma-4 12B with LTX projections) | **~24 GB** |
| Video VAE + audio VAE + duration head | **~1.5 GB** |
| **Fetched total** | **~65 GB** |
| w8g64 DiT, kept alongside the bf16 one | **+23 GB** |

> **Do not fetch the whole repo.** It is **174 GB**, most of it `int8-convrot`
> and `nvfp4` packings that are ComfyUI-only and deliberately not read here.
> The prepare pipeline pins the file list, so `model-fetch` gets this right
> without being told.

## The pipelines

- **[`prepare-ltx-2.5.vpipeline`](pipelines/prepare-ltx-2.5.vpipeline)** —
  download the checkpoint. Run once.
- **[`prepare-ltx-2.5-w8.vpipeline`](pipelines/prepare-ltx-2.5-w8.vpipeline)**
  — quantize the DiT to 8-bit. What the text-to-video pipeline below is
  configured for, and what makes its 5-second clip comfortable; see
  [Memory](#memory).
- **[`ltx-2.5-text-to-video.vpipeline`](pipelines/ltx-2.5-text-to-video.vpipeline)**
  — prompt in, five seconds of `.mp4` with sound out.
- **[`ltx-2.5-image-to-video.vpipeline`](pipelines/ltx-2.5-image-to-video.vpipeline)**
  — a picture and a prompt in, frames out: the clip starts AT your image.
- **[`ltx-2.5-audio-reference.vpipeline`](pipelines/ltx-2.5-audio-reference.vpipeline)**
  — an audio file and a prompt in: the soundtrack is guided by the reference.

They are plain JSON — read them, edit them, keep them in version control.

**Every one of them needs the plugin**, because the stages and the catalogue
entries come from it:

```sh
vpipe --plugin build/vpipe-ltx-2.5.so --launch <file>
```

Put the plugin in your work directory's `plugins/` and the web UI's **Plugins**
panel will load it for you instead — see
[docs/PLUGINS.md](https://github.com/tgo-app-dev/vpipe/blob/main/docs/PLUGINS.md).
Without it, launching any of these fails with `unknown stage type
'ltx-2.5-conditioner'`.

## Step 1 — fetch the model

### First, choose a work directory

vpipe treats **the directory you launch it from** as its workspace and creates
its state there:

| | |
|---|---|
| `models/` | every model you download or quantize |
| `plugins/` | where the web UI looks for plugin dylibs |
| `data.mdb`, `lock.mdb` | the LMDB database — model registry, logs, stage output |
| `sandbox/` | created by **`vpipe-web-ui`** only: the directory it confines stage file I/O to |

Pick one on the volume with the ~65 GB, and use the **same** directory in
step 2 — the model you are about to fetch is recorded in that directory's
registry.

### Then run it

```sh
cd ~/vpipe-work
cp ~/src/vpipe-ltx-2.5/docs/pipelines/prepare-ltx-2.5.vpipeline .
~/src/vpipe/build/apps/vpipe/vpipe \
    --plugin ~/src/vpipe-ltx-2.5/build/vpipe-ltx-2.5.so \
    --launch prepare-ltx-2.5.vpipeline
```

One `model-fetch` stage. `skip_existing_files` is on, so an interrupted fetch
resumes rather than starting over.

**`model_variant: LTX-2.5-distilled` is required, not decorative.** That repo
publishes two DiTs — `distilled` and `dev` — at 39 GB each, over the same
peers. A fetch that does not say which is refused with both listed, rather
than quietly taking the first.

> **Gated repo.** Accept the licence on the model's Hugging Face page and put
> a token in the stage's `hf_token` if the fetch reports an authorization
> failure. The **weights** are under the LTX-2.x Community License; this
> plugin is Apache-2.0.

## Step 2 — text to video **and audio**

```sh
cd ~/vpipe-work
~/src/vpipe/build/apps/vpipe/vpipe \
    --plugin ~/src/vpipe-ltx-2.5/build/vpipe-ltx-2.5.so \
    --launch ltx-2.5-text-to-video.vpipeline
```

The shipped prompt asks for both halves at once:

> *Cinematic video of a young Asian female pianist passionately playing a
> grand piano.*

There is no separate audio prompt — both streams are conditioned from the same
text, through two projections of the same Gemma hidden states — so **the
subject carries the soundtrack**. "Playing a grand piano" is doing audio work;
a prompt that says only what a scene looks like gets whatever the model thinks
it sounds like. If the sound matters, name it: *its paws crunching through the
crust*, *canvas snapping in the breeze*.

### This one is 5 seconds, and that is not free

The shipped geometry is **960 × 544 × 121 frames** — five seconds at 24 fps,
matching the MiniMax-H3 pipeline so the two can be compared. That is **8160
video tokens**, and self-attention is quadratic in them: against the 9-frame
768 × 448 clip this port was first brought up on (672 tokens), it is 12× the
GEMM work and ~144× the attention.

That quadratic term is why the DiT runs **steel flash attention** rather than
the straightforward scalar kernel — measured at 960 × 544 × 49, it is the
difference between 24 and 98 seconds a step. Nothing needs setting; it is the
default wherever the head width has an entry point, and the log line
`attention kernel: steel` at the start of each denoise says which arm ran.
On an M5 the same attention moves to the GPU's matrix cores automatically.

So the pipeline ships with `variant: "w8g64"` on `ltx-2.5-model-config`, and
you want that pack: 24 GB rather than 42 leaves the room the activations at
this size need. Run
[`prepare-ltx-2.5-w8.vpipeline`](pipelines/prepare-ltx-2.5-w8.vpipeline)
first — see [Quantizing](#quantizing-for-a-smaller-machine).

**Without it the run still works**, because an unavailable variant falls back
to the shipped `distilled` bf16 checkpoint and says so in the log — but at
42 GB on a 64 GB box, and slower. For a first run, drop to 512 × 320 and keep
`frames: 121`: the soundtrack is the same five seconds either way, since audio
length is `frames / fps`, not resolution.

The graph is eight stages:

```
text-prompt ──> ltx-2.5-conditioner ─┬─0──> generate-video ─┬─0─> vae-decode ──> rgb-to-video ─┐
                                     └─2──────────^ (10)    │                                  ├─> save-video
                                                            └─1─> audio-vae-decode ────────────┘
ltx-2.5-model-config ─────────────────────────────^ (9)
```

Two edges are easy to get wrong:

- **The conditioner emits two contexts, and they go to different ports.**
  Video conditioning (4096-wide) is oport 0 → `generate-video` iport 0; audio
  conditioning (2048-wide) is oport 2 → `generate-video` **iport 10**. Miss
  the second and the run is video-only — it says so and keeps going, rather
  than inventing a zero audio context.
- **`generate-video` emits two latents**: video on port 0, audio on port 1.
  They decode through two different VAEs and meet again at `save-video`.

The shipped `output_url` is **relative** — `ltx-2.5-text-to-video.mp4` — so it
lands in the work directory from the CLI and under `sandbox/` in the web UI,
and the same file works both ways.

> Resist making it absolute out of habit. A leading `/` is the **sandbox**
> root under the web UI but the real filesystem root from the CLI, which on
> macOS is read-only: the run then denoises for minutes and fails in
> `save-video` at the very end, having thrown the clip away. Ask for that
> path only when you mean it.

### The settings worth knowing

From `generate-video`:

| key | shipped | notes |
|---|---|---|
| `width` / `height` | 960 × 544 | Rounded **up** per family to what the VAE can patch. The stage logs any change. |
| `frames` | 121 | Rounded **up** to `8n + 1` — the VAE's temporal compression is 8×, so only those counts have a latent form. 120 becomes 121. |
| `fps` | 24 | Also sets the audio length: tokens are `frames / fps × 25`. |
| `steps` | 8 | The distilled schedule is **fixed at 8**. Asking for another number is ignored with a warning — it is baked into the checkpoint, not a preference. |
| `seed` | 6 | Same seed + same settings ⇒ same clip. |
| `unload_when_idle` | `always` | Drop the DiT between runs. |

And from **`ltx-2.5-model-config`**, wired to `generate-video`'s `model_config`
iport (port 9):

| key | shipped | notes |
|---|---|---|
| `audio_seconds` | 0 | Audio length follows `frames` and `fps`; set this only to override. |
| `guidance` / `audio_guidance` | unset | Real CFG, **`dev` checkpoint only**. On distilled they are ignored with a warning — there is no unconditional pass to blend with. |
| `lora` / `lora_scale` | unset | The distilled LoRA-450 adapter, for the `dev` DiT. |

| `variant` | unset | **Which DiT to load** when the root holds more than one — `distilled`, `dev`, or the name of a quantized pack such as `w8g64`. See [Choosing the pack](#choosing-the-pack). |

## Step 3 — start from a picture

`ltx-2.5-image-to-video.vpipeline` adds two stages ahead of the text-to-video
graph — `load-image` and `vae-encode` — and wires the latent to
`generate-video`'s **`ref_latent0`** port (iport 5):

```
load-image -> vae-encode ---------------------\
text-prompt -> ltx-2.5-conditioner ------------> generate-video -> vae-decode
                       ltx-2.5-model-config --/
```

```sh
vpipe --plugin build/vpipe-ltx-2.5.so --launch ltx-2.5-image-to-video.vpipeline
```

The clip then **starts at your picture** and animates from there. It ships
pointing at `./reference.png` in your work directory — put an image there, or
change `load-image`'s `url`. Set `vae-encode`'s `target_width` /
`target_height` to the SAME size as `generate-video`'s `width` / `height`, and
the picture is letterbox-fitted into it (aspect ratio kept, the rest padded —
`pad_color` sets the colour).

Two rules the stage will tell you about if you break them:

- the size must be a multiple of **32**, this VAE's spatial compression. 512 ×
  320 works, 500 × 300 does not;
- `vae-encode` and `generate-video` must agree on the size. They are separate
  stages with separate config, so nothing but you keeps them in step.

How hard the picture is held is `ref_strength` on `ltx-2.5-model-config`:

| value | effect |
|---|---|
| `1.0` (default) | the first frame IS the picture; the model animates away from it |
| `0.5` | the model may reinterpret it |
| `0.0` | ignored entirely — the same as not wiring it |

Wiring a SECOND encoded image to **`ref_latent1`** (iport 6) anchors the
clip's LAST frame as well, with `ref_last_strength` for its own hold. That
path is written and shares all its machinery with the first-frame one, but it
has not been run end to end.

**Measured** (512 × 320 × 9 frames, seed 1234, the distilled checkpoint):
frame 0 comes back at **31.1 dB** against the letterboxed input — the residual
is the VAE's own round-trip at 32× compression, not the anchoring — decaying
to 26.3 dB by frame 8 as the model animates. The same seed and prompt with the
anchor unwired scores **6.8 dB**: a different picture entirely, which is what
says the anchor is doing the work rather than the prompt.

## Step 4 — a reference soundtrack

`audio-vae-encode` turns PCM into the latent rows `generate-video` takes on
its **`ref_audio_rows`** port (iport 8), so the model keeps an existing track
instead of inventing one:

```
<some PCM source> -> audio-vae-encode -> generate-video (iport 8)
```

Three things about it are unlike the other stages:

- **it accumulates and encodes ONCE, at the end of the stream.** There is no
  per-beat output. An audio VAE is causal and compresses time 4×, so chunks
  encoded separately and concatenated are a different tensor — wrong in length
  and wrong at every seam. `max_seconds` (default 30) bounds the buffer;
- **it does not resample.** The LTX encoder wants **16 kHz**, and a stream at
  another rate is refused with a message naming `audio-to-pcm`'s
  `output_sample_rate`. Encoding 48 kHz samples as 16 kHz would give a
  reference of the right length at the wrong pitch, which nothing downstream
  can see;
- **mono is duplicated to stereo**, because the model was trained on two
  channels and one channel of silence is a different soundtrack.

`ref_audio_strength` on `ltx-2.5-model-config` controls how hard it is held,
the same way `ref_strength` does for the picture. The reference is placed at
**negative seconds** — before the clip on the one axis the two streams share —
so it guides without overlapping the soundtrack being generated.

`ltx-2.5-audio-reference.vpipeline` is the whole graph:

```
load-audio -> audio-to-pcm -> audio-vae-encode ---------\
text-prompt -> ltx-2.5-conditioner -----------------------> generate-video -> ...
                          ltx-2.5-model-config ----------/
```

```sh
vpipe --plugin build/vpipe-ltx-2.5.so --launch ltx-2.5-audio-reference.vpipeline
```

It ships pointing at `./reference-audio.m4a`; put any audio file there, or any
video file — `load-audio` takes the audio track of an mp4 as happily as an mp3.
`audio-to-pcm`'s `output_sample_rate` **must be 16000**, because that is what
the encoder asks for and it refuses rather than resampling.

Measured on one run at 512x320x9: a 0.40 s reference became **10 appended
audio tokens**, so the DiT denoised 19 audio tokens where an unreferenced run
has 9, and the clip came out coherent with its own soundtrack. Note what the
generator logs — *"placed before the clip on the shared time axis"* — the
reference sits at negative seconds, so it guides without overlapping the
soundtrack being generated.

## Memory

**The bf16 checkpoint wants a 64 GB machine, it is marginal even there, and
the reason is not the DiT alone.**

The DiT asks for `Mapped` weights and **gets copies**: this checkpoint's
safetensors data section starts at an offset ≡ 8 (mod 16), and vpipe's
zero-copy path needs 16-byte alignment, so all 39.1 GB silently falls back to
`Copied`. The text encoder is another ~24 GB, also copied. Both are anonymous
memory — reclaimable only through the compressor and swap — so on a 64 GB box
they cannot coexist, and the machine compresses its way through the denoise.

Measured at 768 × 448 × 9 frames, 8 steps, before and after the encoder is
released:

| | encoder kept | encoder released |
|---|---|---|
| footprint the DiT sizes against | 65 GB | **41 GB** |
| residency verdict | `STREAM blocks` | **`PRELOAD`** |
| disk I/O during the denoise | ~550 MB/s sustained | **0.41 MB/s** |
| end to end | ~25 min | **338 s** |

**4.4×, and byte-identical output.** The encoder is idle for the whole
denoise — it encodes one caption in seconds — so the shipped pipeline leaves
`unload_when_idle` at `auto`, which drops it and, crucially, **revises its
declaration** so `generate-video` sizes against a box that actually has the
room back. That last part is why this is a memory *policy* fix and not just a
free: `generate-video` loads its DiT lazily on the first conditioning beat,
which is where it takes an **irreversible** streaming decision.

### Quantizing, for a smaller machine

```sh
vpipe --plugin build/vpipe-ltx-2.5.so --launch prepare-ltx-2.5-w8.vpipeline
```

Measured on the 22B distilled DiT, same seed:

Measured end to end on the 64 GB M4 Pro, same pipeline, 768 × 448 × 9, 8 steps:

| pack | size | peak footprint | wall clock | zero-copy mapped | PSNR vs bf16 |
|---|---|---|---|---|---|
| bf16 | 42.0 GB | 55.2 GB | 338 s | **no** — 39.1 GB copied | — |
| **w8g64** | 23 GB | **27.4 GB** | **210 s** | **yes** | 33.29 dB |
| w4g64 | 14 GB | — | — | yes | 25.87 dB |

The w8 row is **1.6× faster on half the peak footprint**, and the mapping
column is most of why: quantized packs are written by vpipe's own
`SafetensorsWriter`, which pads its header so the data section is 16-byte
aligned, so their tensors are handed to Metal as zero-copy views. The
published bf16 file is not aligned, so all 39.1 GB of it is copied into
anonymous memory — the loader now says so, once per shard.

**w8 is the sensible default at any size, not just below 64 GB.** It is
smaller, it is faster, and — because vpipe wrote it — it is the only one of
the two that is actually mappable. w4 halves it again but visibly softens the
subject. w4 halves it again but visibly
softens the subject — the scene and composition survive, the detail does not.
For scale, a conditioning bug elsewhere in this tree moved PSNR 20.4 → 45.6
dB, so 45+ is indistinguishable, 33 is close, and 26 is a real degradation.

**`quant_exclude` is not optional**, and the prepare pipeline sets it. The
wholesale scope rule takes every 2D floating-point tensor whose leaf is not a
norm or an embedding — which over `transformer_blocks.` also catches the f32
`scale_shift_table`s (modulation tables, not matrices) and the `[32, dim]`
gate logits. Quantizing either produces a checkpoint that loads, runs, and
generates the wrong thing.

> **The declaration follows the choice.** `declare_resources()` runs before
> any beat exists, so it cannot know which pack `variant` will name — it
> declares the **largest** candidate, because over-declaring costs a peer some
> caution while under-declaring tells the graph there is room that is not
> there. Once the beat has been read the family corrects the books to the
> pack it actually loaded, *before* the streaming decision is taken. You can
> see both numbers in the log: a w8 run reports `footprint 24 GB`, not the
> 41 GB of the bf16 file it declined.

### Choosing the pack

A bf16 file and a quantized directory can sit side by side under
`diffusion_models/`, as can the `distilled` and `dev` DiTs. The choice is
never implicit — a 3× footprint difference decided by resolution order is not
a default anyone should get by accident. Name it on the config stage:

```json
{
  "id": "ltx-config", "type": "ltx-2.5-model-config", "iports": [],
  "config": { "variant": "w8g64" }
}
```

The value is matched against what is actually under `diffusion_models/`, not
against a fixed list, so it names the two released DiTs *and* any pack
`model-quantize` wrote there. A value that matches nothing falls back to the
shipped preference rather than failing — which is safe only because the family
**logs the file it loaded on every run**:

```
ltx-2.5: DiT weights from .../ltx-2.5-22b-distilled-w8g64 (quantized pack)
```

`VPIPE_LTX25_VARIANT` overrides the config for a one-off run without editing
the pipeline, and **warns when it does** — an env var silently beating a
checked-in pipeline is the same trap as a config key that selects nothing,
pointed the other way.

## What this port does not do

Stated plainly, because everything above invites the assumption that it is
finished:

- **The audio reference has no PCM source in a graph.** `audio-vae-encode` and
  the LTX audio encoder behind it are done and verified, but nothing in vpipe
  turns an audio FILE into PCM beats: the stage's input comes from live
  capture (`rtsp-capture -> audio-to-pcm`) or from another model's
  `audio-vae-decode`. A file loader is a small stage that does not exist yet.
- **A multi-frame video reference has no producer either.** `ref_latent0`
  takes any number of latent frames and treats them as a prefix the clip
  continues from; `vae-encode` supplies exactly one image per beat, for every
  family. Single-image anchoring is what runs today.
- **Classifier-free guidance is written and has never run.** Only the
  distilled checkpoint is on disk here, and it is guidance-distilled, so the
  negative-prompt path is unexercised. It needs the `dev` checkpoint.
- **The audio DiT output is unverified against the reference.** The bar taken
  was structural — shape, all-finite, the schedule moves it, two seeds differ,
  it decodes to non-silent audio. A real golden means the whole 22B stack in
  PyTorch.
- **Only short clips have been run**: 9 frames, at 256² and 768×448. The model
  targets 121 frames, where both time and memory scale.
- The **diffusion** VAE decoder is not ported (only the conv one); audio VAE
  `attn` blocks and any `rope_type` other than `split` are refused rather than
  approximated.

## Troubleshooting

**`unknown stage type 'ltx-2.5-conditioner'`.** The plugin is not loaded. Pass
`--plugin`, or put it in `<work-dir>/plugins/` and load it from the web UI's
Plugins panel.

**The video has no sound.** Check `save-video` has `enable_audio: true`, that
`audio-vae-decode` is wired to `generate-video` **port 1** (port 0 is video),
and — the one that catches people — that the conditioner's **oport 2** reaches
`generate-video`'s **iport 10**. Without it the run is video-only and says so
in the log.

**`frames` isn't what I asked for.** Expected: rounded up to `8n + 1`.

**`steps` is ignored.** Expected on the distilled checkpoint: the 8-step
schedule is part of the weights.

**It thrashes on a 64 GB machine.** Check the conditioner's
`unload_when_idle` is not `keep`, and look for `-> PRELOAD` rather than
`-> STREAM blocks` in the log. See [Memory](#memory).

## Under the hood

- 48 blocks, video 4096-wide / 32 heads beside audio 2048-wide / 32 heads,
  coupled by a2v and v2a cross-attention with per-stream AdaLN gates.
- **One** sigma schedule for both streams — there is no per-modality shift in
  this model; that belongs to MiniMax-H3.
- RoPE positions are **seconds and pixels**, not latent indices. Video frames
  sit on a 24 fps clock, audio tokens at 25 per second, so the two streams'
  cross-attention agrees about time.
- An 8-layer connector with 128 learnable registers projects the caption
  before the DiT sees it; padded rows are replaced by those registers.
- The video VAE is 128-channel at 1/32 spatial and 8× temporal; audio decodes
  through a second VAE to log-mel, a BigVGAN vocoder to 16 kHz, and a
  bandwidth extension to **48 kHz stereo**.
