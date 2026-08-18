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
| **Disk** | **~65 GB** to fetch; **+38 GB** for the 8-bit model. |
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
| the w8g64 model, built alongside (DiT 23 + encoder 15) | **+38 GB** |

> **Do not fetch the whole repo.** It is **174 GB**, most of it `int8-convrot`
> and `nvfp4` packings that are ComfyUI-only and deliberately not read here.
> The prepare pipeline pins the file list, so `model-fetch` gets this right
> without being told.

## The pipelines

- **[`prepare-ltx-2.5.vpipeline`](pipelines/prepare-ltx-2.5.vpipeline)** —
  download the checkpoint. Run once.
- **[`prepare-ltx-2.5-8bit.vpipeline`](pipelines/prepare-ltx-2.5-8bit.vpipeline)**
  — **the whole preparation in one run**: fetch, then quantize the DiT (its
  block stack AND both text connectors, in one step) and the text encoder to
  w8g64 into **one self-contained model**,
  `local/LTX-2.5-distilled-8bit`. Start here on a fresh machine. Idempotent
  (`skip_existing_files` on the fetch, `skip_existing` on both quantize
  steps), so re-running after an interruption resumes rather than repeating
  a finished half.
- **[`ltx-2.5-text-to-video.vpipeline`](pipelines/ltx-2.5-text-to-video.vpipeline)**
  — prompt in, five seconds of `.mp4` with sound out.
- **[`ltx-2.5-image-to-video.vpipeline`](pipelines/ltx-2.5-image-to-video.vpipeline)**
  — a picture and a prompt in, frames out: the clip starts AT your image.
- **[`ltx-2.5-audio-reference.vpipeline`](pipelines/ltx-2.5-audio-reference.vpipeline)**
  — an audio file and a prompt in: the soundtrack is guided by the reference.

They are plain JSON — read them, edit them, keep them in version control.

> **What the 8-bit pipeline produces.** Two chained passes and one
> directory: a quantized DiT, a quantized encoder, the VAEs untouched — a
> model that can be named, moved and deleted as one thing, exactly like a
> built-in family's quantized output. Everything it does not quantize is
> **hard-linked**, so the second pass and the passthrough components cost no
> extra bytes. Point a pipeline at `local/LTX-2.5-distilled-8bit` and both
> precisions come with it: no `variant` and no `encoder_variant`, because a
> self-contained model holds exactly one of each. Those two keys are for the
> other arrangement — several DiTs, or a quantized pack sitting beside a
> bf16 file, in one repo — and are covered in
> [Choosing the pack](#choosing-the-pack).
>
> **The VAEs, vocoder, BWE, connector and audio encoder stay dense**, and
> that is not an oversight: the `QWeight` path that can read a quantized
> tensor at all exists only in the DiT blocks and (through the host's Gemma)
> the encoder backbone. It is also where the bytes are not — both VAEs
> together are 1.4 GB against the DiT's 42 GB and the encoder's 24.5 GB.

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

## Step 1 — fetch and prepare the model

### First, choose a work directory

vpipe treats **the directory you launch it from** as its workspace and creates
its state there:

| | |
|---|---|
| `models/` | every model you download or quantize |
| `plugins/` | where the web UI looks for plugin dylibs |
| `data.mdb`, `lock.mdb` | the LMDB database — model registry, logs, stage output |
| `sandbox/` | created by **`vpipe-web-ui`** only: the directory it confines stage file I/O to |

Pick one on the volume with the ~103 GB (65 fetched + 38 quantized), and use
the **same** directory in step 2 — the models you are about to build are
recorded in that directory's registry, and step 2 names one of them.

### Then run it

```sh
cd ~/vpipe-work
cp ~/src/vpipe-ltx-2.5/docs/pipelines/prepare-ltx-2.5-8bit.vpipeline .
~/src/vpipe/build/apps/vpipe/vpipe \
    --plugin ~/src/vpipe-ltx-2.5/build/vpipe-ltx-2.5.so \
    --launch prepare-ltx-2.5-8bit.vpipeline
```

Fetch, then two quantize passes, then the intermediate is removed — ending at
`models/local/LTX-2.5-distilled-8bit`, which is what step 2 loads. Every step
is resumable (`skip_existing_files` on the fetch, `skip_existing` on both
quantizes), so an interrupted run continues rather than starting over.

**Use [`prepare-ltx-2.5.vpipeline`](pipelines/prepare-ltx-2.5.vpipeline)
instead if you want the bf16 checkpoint and nothing else** — it is the fetch
on its own. The shipped text-to-video pipeline expects the 8-bit model, and
names it in ONE place: point its `model-select` stage at
`./models/Lightricks/LTX-2.5` if you go that way.

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

So the pipeline points at the **8-bit model** `local/LTX-2.5-distilled-8bit`,
and you want it: a 21 GB DiT rather than 42 leaves the room the activations
at this size need, and the encoder comes down with it. It is named once, on
the `model-select` stage, and read from there by the conditioner, the
generator and both VAE decoders — so changing which checkpoint a run uses is
one edit, not five, and the five cannot drift apart. That model is what
[`prepare-ltx-2.5-8bit.vpipeline`](pipelines/prepare-ltx-2.5-8bit.vpipeline)
builds — see [Quantizing](#quantizing-for-a-smaller-machine).

**Run that first**, because unlike a missing `variant` — which falls back to
bf16 and says so — a missing MODEL is not something the graph can substitute
for. It fails naming the directory it wanted, which is the better error: a
silent fall back to 42 GB on a 64 GB box is the outcome this geometry cannot
afford. For a first run, drop to 512 × 320 and keep `frames: 121`: the
soundtrack is the same five seconds either way, since audio length is
`frames / fps`, not resolution.

**What it costs, measured.** On the 64 GB M4 Pro at 960 × 544 × 121 with the
w8g64 pack: **532 s end to end**, of which 410.9 s is the denoise
(51.4 s/step across 8 steps) and the rest is the model load, the two VAE
decodes and the mux.

What lands is one file — `save-video` with `enable_audio` muxes the frames
and the soundtrack together:

```
h264 960x544 @ 24/1 fps + aac 48000 Hz stereo, 5.041667 s, 1.1 MB
```

The audio is not an afterthought bolted on at the end: it was denoised
jointly with the video by the same DiT, on the same schedule, and
`audio-vae-decode` turns those latents into 2 x 240480 samples that the muxer
puts alongside the frames.

The **8 steps are the checkpoint's own**. The distilled DiT ships a fixed
schedule, so a `steps` of 6 is reported and ignored:

```
ltx-2.5: the distilled checkpoint's schedule is fixed at 8 steps; the configured 6 is ignored
```

### You do not have to compute legal geometries

The VAE compresses space by 32 and time in chunks of 8 (`8k + 1` frames), so
`960 × 544 × 121` is a legal shape and `953 × 550 × 120` is not. Ask for
you want anyway — the stage rounds **up** and says so:

```
frames 120 -> 121, the nearest count at or above it that the ltx-2.5 VAE can chunk
953x550 -> 960x576, the nearest size at or above it that the ltx-2.5 VAE and DiT patch can tile
```

Up rather than down, so you never get less than you asked for, and reported
rather than silent, because the clip that comes back is a different shape from
the one requested and a downstream stage would otherwise meet that as a
surprise. Rounding rather than rejecting is what lets one graph be pointed at
a different model family without being re-authored.

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

The bf16 DiT asks for `Mapped` weights and **gets copies**: this checkpoint's
safetensors data section starts at an offset ≡ 8 (mod 16), and vpipe's
zero-copy path needs 16-byte alignment, so all 39.1 GB silently falls back to
`Copied`. (The quantized packs are written by vpipe and are aligned, so they
do map.) The text encoder is another ~24 GB, also copied. Anonymous memory is
reclaimable only through the compressor and swap, so on a 64 GB box the two
cannot coexist, and the machine compresses its way through the denoise.

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

From nothing — fetches first, then quantizes:

```sh
vpipe --plugin build/vpipe-ltx-2.5.so --launch prepare-ltx-2.5-8bit.vpipeline
```

It assembles `local/LTX-2.5-distilled-8bit`: a whole model, ready to name as
an `hf_dir` with nothing else to configure. The fetch it chains from is
skipped if the checkpoint is already on disk, so this is also the command to
run when only the quantize is wanted.

Disk:

| | fetched | + w8g64 |
|---|---|---|
| DiT | 42 GB | 21 GB |
| Text encoder | 24.5 GB | 15 GB |
| VAEs + duration head (stay dense) | ~1.5 GB | hard-linked, 0 |
| **added by preparing** | — | **~38 GB** |

The self-contained model costs the same as the two packs, because
everything it does not quantize is hard-linked rather than copied — its
`vae/` shares inodes with the fetched repo's. That also means deleting the
fetched repo afterwards frees only what the packs replaced.

Measured on the 22B distilled DiT, same seed:

Measured end to end on the 64 GB M4 Pro, same pipeline, 768 × 448 × 9, 8 steps:

| pack | size | peak footprint | wall clock | zero-copy mapped | PSNR vs bf16 |
|---|---|---|---|---|---|
| bf16 | 42.0 GB | 55.2 GB | 338 s | **no** — 39.1 GB copied | — |
| **w8g64** | 21 GB | **27.4 GB** | **210 s** | **yes** | 33.29 dB |
| w4g64 | 14 GB | — | — | yes | 25.87 dB |

The peak and wall clock in that row were measured when the pack was
23 GB, before the text connectors came into scope (below); at 21 GB they are
if anything pessimistic. The quality figures are unaffected — the connectors
were bf16 in the measured pack and are w8 now, which is the one number that
moved, and `ltx25-connector-test` puts it at rel-L2 1.170e-02 against the
reference where bf16 reads 7.654e-03, both far inside the 5e-2 bar.

The w8 row is **1.6× faster on half the peak footprint**. Two things
contribute and it is worth keeping them apart: it is half the bytes to read,
AND it maps where bf16 cannot. Quantized packs are written by vpipe's own
`SafetensorsWriter`, which pads the data section to 16 bytes and writes
odd-sized tensors last, so their tensors are handed to Metal as zero-copy
views; the published bf16 file is not aligned, so all 39.1 GB of it is copied
into anonymous memory. The loader says which case it is, once per shard.

### Holding the blocks: preload vs stream

Whether the blocks are mapped or owned is **not a fixed property of this
port** — it follows the residency verdict, and the two arms want opposite
things:

| | preload (box holds the pack) | stream (it does not) |
|---|---|---|
| blocks held | all of them | a leading prefix, sized to fit |
| read as | `Mapped` — clean file pages | `Copied` — owned |
| the rest | — | read per forward, dropped after |
| the read | — | issued under the previous block's GPU work |
| growth | — | streamed blocks promoted back as RAM allows |

Owning the bytes is what makes a pinned prefix mean anything: a prefix the
kernel can evict is not a prefix. And a forward is a **cyclic scan**, which
is the one access pattern an LRU page cache handles worst — each block
dropped exactly before it comes round again, so the cache stays full of data
that is never the data wanted next. A fixed resident subset gives exactly its
share of hits instead of none.

The reverse is just as true, which is why preloading maps. MEASURED at
960 × 544 × 121 on the 64 GB box, same seed, w8g64, both arms run together:
**409.7 s** of denoise mapped against **473.5 s** owning the same bytes — the copies became 36 GB of
compressor traffic and the model's own weights fell to 2-3% resident. Free-to-
drop file pages are strictly better when everything fits.

**w8 is the sensible default at any size, not just below 64 GB.** It is
smaller, it is faster, and — because vpipe wrote it — it is the only one of
the two that is actually mappable. w4 halves it again but visibly softens the
subject. w4 halves it again but visibly
softens the subject — the scene and composition survive, the detail does not.
For scale, a conditioning bug elsewhere in this tree moved PSNR 20.4 → 45.6
dB, so 45+ is indistinguishable, 33 is close, and 26 is a real degradation.

**The exclusions are not optional**, and the FAMILY sets them — not the
pipeline, which names only a target, bits and a group size. The wholesale
scope rule takes every 2D floating-point tensor whose leaf is not a norm or
an embedding, which inside a block also catches the f32 `scale_shift_table`s
(modulation tables, not matrices) and the `[32, dim]` gate logits. Quantizing
either produces a checkpoint that loads, runs, and generates the wrong thing.
Keeping that list with the scope rather than in the pipeline is what stops
the two drifting apart.

#### The text connectors are quantized too, by the same step

There is no separate stage for them. `target: "dit"` covers the DiT's own
`transformer_blocks.` **and** the two text connectors'
`transformer_1d_blocks.` — the scope is a substring and `_blocks.` is the one
both share. The quantizer says so when it runs: `1440 quantized, 2909
passthrough`, which is 1344 block tensors plus 96 connector ones, leaving the
connectors' 162 norms, biases and learnable registers dense.

It matters more than its 2 GB suggests. The connectors are **trunk**: unlike
the blocks they never stream, so they are resident for the whole denoise. On
a 16 GB box they were 3.75 GB off the top before a single block could be
pinned.

> **A pack built before this has dense connectors, and re-running the prepare
> pipeline will not fix it.** `skip_existing` on the second quantize step sees
> `local/LTX-2.5-distilled-8bit` already there and skips — so the run
> re-quantizes the DiT and then throws the result away. Delete the model
> first:
>
> ```sh
> vpipe --plugin build/vpipe-ltx-2.5.so \
>     --launch-stage model-remove \
>     --stage-cfg model='"local/LTX-2.5-distilled-8bit"' \
>     --stage-cfg delete_files=true --stage-cfg missing_ok=true
> ```
>
> then run the prepare pipeline again. Nothing else needs changing: the
> reader was taught the quantized layout in the same change, so an old pack
> still loads — it is simply 2 GB larger and 2 GB more resident.

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

The **text encoder is chosen separately**, on the conditioner rather than the
config stage, because it is a different component with a different tradeoff:

```json
{
  "id": "cond", "type": "ltx-2.5-conditioner",
  "config": { "encoder_variant": "w8g64" }
}
```

Same rule — a name substring matched against the directories under
`text_encoders/`, and **empty by default**, which takes the released bf16
file. This one is deliberately conservative: every conditioning token in the
clip comes out of this model, so a precision change nobody requested reads as
a port bug rather than as a setting. What it buys is the peak that decides
whether a small box runs at all — the bf16 encoder is 24.5 GB and peaks at
**27.1 GB** of footprint against the DiT denoise's 2.6–2.8 GB, so the encoder,
not the DiT, is what a 32 GB machine runs out of room on.

Graded on the cross-attention context it emits, cosine against the bf16
encoder over the real caption rows:

| pack | cosine | rel-L2 | size |
|---|---|---|---|
| **w8g64** | **0.99996** | 0.0088 | 15 GB |
| w4g64 | 0.99387 | 0.1105 | 9.9 GB |

**w8g64 is the safe default** — 0.99996 is the same caption for practical
purposes. Judge either by that grading and not by comparing frames: a
diffusion sample moves to a different mode on *any* conditioning change, so a
w4 run scores about 21 dB against a bf16 run, which is roughly what two
different frames of the same clip score against each other. Reading that as
"4-bit is lossy" is a mistake this port already made once.
`VPIPE_LTX25_COND_DUMP=<prefix>` writes both contexts so they can be compared
directly instead of inferred from pixels.

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
