# Feature candidates

Ideas for where RustingBrain could go after 2.0. Nothing here is committed,
scheduled, or promised; the README's Scope section is still the authoritative
statement of what the library does today. This file exists so the options are
written down with their real costs instead of being re-argued from scratch.

Effort is a rough size for one person: **S** is days, **M** is a week or two,
**L** is a month or more of focused work.

## Where the code stands

The constraints below shape almost every estimate in this document, so they
are worth stating once.

- `Matrix` (`src/matrix.rs`) is a two-dimensional, row-major, `f32` buffer.
  There is no N-dimensional tensor type and no shape metadata beyond rows and
  columns. Anything that wants NCHW or NHWC data has to either flatten into
  this type or introduce a new one.
- There is no autograd tape. Every module hand-writes its own `backward`
  against a cache struct produced by its forward pass — see
  `src/attention.rs:286`, `src/ffn.rs:87`, `src/moe.rs:349`. A new layer type
  means a new hand-written, finite-difference-tested gradient.
- `Dataset` (`src/dataset.rs`) holds `Vec<Vec<f32>>` for inputs and targets,
  fully in memory. `TokenBatch` (`src/batch.rs`) likewise takes token ids that
  the caller has already materialized. Nothing streams.
- `to_cuda(device, memory_budget_mib)` (`src/transformer.rs:726`) binds a model
  to exactly one device. The GPU path in `src/gpu_model.rs` assumes a single
  context throughout.
- The CPU and CUDA paths are written twice, once each, in parallel module
  trees. Every new layer that needs to run on a GPU is two implementations and
  two sets of gradients.

That last point is the structural tax on most of this list. It is worth
considering whether some form of shared kernel description would pay for itself
before three or four more layer types are added, though that is a refactor
proposal rather than a feature and it is not scoped here.

## Data loading

Today a training run must build its entire corpus in memory before the first
step. That caps the practical dataset size at whatever fits in RAM and makes
resuming a run mid-epoch impossible.

*Landed since this was written:* `TokenFile` (`src/token_file.rs`) reads
windows out of a flat file of little-endian `u32` ids, so a text corpus no
longer has to fit in memory and a resumed run replays the same batches. That is
the text half of this section without the trait; the items below are still open
for everything else.

### Streaming dataset trait

**What:** An iterator-shaped trait that yields batches, with `Dataset` becoming
one implementation of it. Training loops take the trait instead of the concrete
type.

**Why:** Every other item in this section depends on it. Without a streaming
boundary, each new format is another way to fill a `Vec` that still has to fit
in memory.

**Effort:** S for the trait, M including migrating `fit`, `TrainingSession`,
and the examples onto it.

**Risks:** The trait's shape is hard to change later. Getting the borrow
pattern right for zero-copy batches matters — `DatasetBatch<'a>` already
borrows, and a streaming source has no stable buffer to borrow from, so this
likely needs an owned batch type or an internal reusable buffer.

**Depends on:** Nothing.

*Landed.* `BatchSource` yields `DatasetBatch<'_>` borrowed from the source, so an
in-memory `Dataset` hands out slices of its own rows and a streaming source hands
out slices of a buffer it overwrites between calls: no owned batch type and no
copy on either path. `Dataset::stream` is the in-memory implementation, taking
the batch size, the shuffling and the seed from `TrainConfig`, and
`Network::fit_stream` and `fit_stream_with` train from any implementation.
`fit` and `fit_with` are unchanged from the outside and now run through the
trait. The device sessions still take a `&Dataset`, which they upload whole.

### Format readers: Parquet, Arrow, WebDataset, JSONL

*Landed since this was written:* `TokenFile::write_jsonl` (text),
`Dataset::from_npy` / `Dataset::from_idx` (arrays and MNIST-shaped files), and
now `JsonlStream` and `Dataset::from_jsonl` for numeric JSONL, the first reader
behind the streaming trait. All of them without a dependency, `serde_json`
having already been one. Parquet, Arrow and WebDataset are still open, and the
question this section raises about `arrow-rs` is still the right one to answer
first: a `to_parquet`-shaped conversion step on the Python side is one line for
the user and eighty crates for this one.

**What:** Readers for the formats that public text and image corpora actually
ship in. JSONL is trivial and needs no dependency; Parquet and Arrow mean
pulling in `arrow-rs`; WebDataset is tar shards and needs only `tar`.

**Why:** The friction the user named. Getting a Hugging Face dataset into
RustingBrain today means writing a conversion script.

**Effort:** S each behind the streaming trait, M for Parquet if column type
coverage is taken seriously.

**Risks:** `arrow-rs` is a large dependency tree for a crate that currently has
eight. Worth gating behind a feature flag, and worth asking whether pointing
users at a conversion step is the cheaper answer.

**Depends on:** Streaming dataset trait.

### On-the-fly tokenization

**What:** A hook that turns text into `u32` ids inside the loader, rather than
requiring a pre-tokenized corpus.

**Why:** Pre-tokenizing a large corpus is a separate pipeline the user has to
build and store. Doing it in the loader removes a step and a copy of the data.

**Effort:** S. The `tokenizers` crate already does the work; this is plumbing
plus a worker pool so tokenization overlaps the training step.

**Risks:** Adds `tokenizers` as a dependency, which the README currently and
deliberately leaves to the caller. Feature-gate it.

**Depends on:** Streaming dataset trait.

*Landed.* `TokenStream::text` and `TokenStream::jsonl` read a corpus, tokenize a
document at a time and cut the ids into windows, so `batch(sequences, seq_len)`
trains without a `TokenFile` on disk and without a second copy of the corpus.
Tokenization stays the caller's closure, the same convention
`TokenFile::write_jsonl` uses, so no dependency was added and none is
feature-gated. It yields windows in file order where `TokenFile::batch` draws
them from random offsets, which is the one thing the pre-tokenized path still
does better; a corpus sorted by source or by date wants a shuffle buffer here,
and nothing needs one yet. The worker pool is also skipped: tokenizing a batch
is a fraction of the step that trains on it. `restart` begins the next epoch.
A test asserts the streamed ids are the ids `write_jsonl` writes for the same
corpus.

### Resumable iteration

**What:** A serializable cursor so a run that dies at step 40,000 resumes at
the same position in the same shard order, not at the start of the epoch.

**Why:** Checkpoints already survive a restart (`save_optimizer_state`); the
data position does not, so a resumed run silently retrains on data it has
already seen.

**Effort:** S, if the streaming trait is designed with it in mind from the
start. Retrofitting is worse.

**Risks:** Interacts with shuffling. A seeded shuffle over a known-length
dataset is easy to resume; a streaming shuffle buffer is not, and needs its
state serialized too.

**Depends on:** Streaming dataset trait.

*Landed.* `BatchCursor` is the serializable position, `{ epoch, batch }`, and
`Network::fit_stream_resuming` both reports it after every batch and starts from
one. `BatchSource::skip_batches` returns a source to it, with a default that
reads and discards and an override on `DatasetStream` that is arithmetic. The
shuffling risk was real: `Dataset::shuffle` permutes the order it finds, so
every epoch's order depended on all the orders before it and a resumed run saw
rows the run it continued never would. `DatasetStream` now tracks which original
row sits at each position and rebuilds the order from the epoch's seed alone. A
test trains a run straight through and a second one interrupted mid-epoch and
asserts the two networks are equal; it fails if that fix is reverted. Sources
without a seeded, epoch-determined order, including an unseeded `DatasetStream`,
resume approximately, which is documented on `BatchCursor`.

## Multi-GPU

The README currently says this is not planned. It is the largest item on the
list and the one with the most structural impact.

### Data-parallel training

**What:** Replicate the model on N devices, split each batch across them, and
all-reduce gradients before the optimizer step.

**Why:** It is the only form of parallelism most users need, and it is the one
that turns a two-card desktop into a usefully faster machine. It also composes
with the existing gradient-accumulation path, which already separates
accumulation from the step.

**Effort:** L. The gradient buffers are per-model and the GPU context is
single-device; both assumptions run through `src/gpu_model.rs`.

**Risks:** NCCL means a new system dependency and a new failure surface, and
the crate's fail-closed error discipline would have to extend to collective
ops. A hand-rolled peer-to-peer all-reduce over `cudaMemcpyPeer` avoids the
dependency and is tractable for two to four cards on one host, at the cost of
being slower and hand-maintained. Two cards on one host is the case worth
targeting first; it is also the only case that can be tested here.

**Depends on:** Nothing strictly, but it is much more useful with a streaming
loader that can shard across ranks.

### Tensor and pipeline parallelism

**What:** Split individual layers (tensor) or contiguous blocks of layers
(pipeline) across devices so a model larger than one card's memory can train.

**Why:** The only way past the single-card parameter ceiling.

**Effort:** L, and larger than data-parallel. Tensor parallelism means
splitting the attention and FFN GEMMs and inserting collectives inside the
forward and backward passes of every block.

**Risks:** High complexity for a use case that a single-desktop library may not
have. Probably not worth starting unless data-parallel has landed, been used,
and the memory ceiling is the thing users actually hit.

**Depends on:** Data-parallel training, for the collective primitives.

### Multi-node

**What:** Extend data-parallel across machines over the network.

**Why:** Completeness.

**Effort:** L.

**Risks:** This is where a single-machine library stops being a
single-machine library, and where PyTorch's ecosystem advantage is largest.
Listed for completeness; recommend against.

**Depends on:** Data-parallel training.

## Computer vision

Also currently listed as out of scope. Two fairly independent halves: the
convolutional path, which needs new machinery, and the vision-transformer path,
which mostly reuses what already exists.

### Vision transformers

*Landed:* `src/vision.rs` — `VisionTransformer` and `VitConfig`, with
`patchify`, `forward_train`/`backward`, `train_step`, `fit` over a `Dataset`,
`predict`, `accuracy` and a JSON checkpoint. It was as cheap as this section
guessed: the patch embedding is a `Linear` over the pixels of a patch, and the
blocks, the norm and the optimizer are the language model's unchanged.

Two things were left out rather than built. Positions are the existing rotary
embedding over patches in reading order, not a learned table — the table is a
`Param` and an add if accuracy asks for it. Pooling is the mean over patches
rather than a class token, which is a parameter and a special-cased row fewer.

The kernel risk below did not have to be solved: the CUDA path refuses a
bidirectional model, so a vision transformer trains on the CPU. That is the
gap worth closing next for this item, and it is the same non-causal flash
kernel the section names.

**What:** Patch embedding, learned or sinusoidal position embeddings, and a
non-causal attention mask, feeding the existing transformer block stack.

**Why:** It is by far the cheapest way into vision, because the expensive parts
— attention, RMSNorm, SwiGLU, the fused kernel, mixed precision — are already
written and tested. A patch embedding is a strided linear projection, which
`Linear` already does.

**Effort:** M.

**Risks:** The flash-attention kernel in `src/cuda_flash.rs` is causal. A
bidirectional variant is either a new kernel or a flag threaded through the
existing one, and that kernel is the most performance-sensitive code in the
crate. Image decoding needs a dependency (`image`), feature-gated.

**Depends on:** An image dataset loader, so it has something to train on.

### Convolutions

**What:** `Conv2d`, pooling, batch norm, and the CPU and CUDA gradients for
each.

**Why:** CNNs are still the right tool for small-image and
embedded-scale vision, and they are what most people mean by "add CV support".

**Effort:** L. This is the item where the two-dimensional `Matrix` bites
hardest: convolution wants a four-dimensional tensor, and either an im2col
transform that flattens into the existing type or a genuine N-D tensor type
alongside it. im2col is the smaller diff and reuses the tuned GEMM path; a new
tensor type is the better long-term answer and a much larger change.

**Risks:** Each new layer is a hand-written backward on two backends. Batch
norm in particular has an awkward gradient and running-statistics state that
must survive save/load. Recommend im2col plus the existing GEMM first, and only
consider a tensor type if the layer count grows past what im2col can carry
cleanly.

**Depends on:** An image dataset loader.

### Image datasets

*Landed since this was written:* `Dataset::from_image_folder` (`src/dataset.rs`,
behind `--features images`) — directory-per-class, PNG/JPEG decode in parallel,
resize, grayscale or RGB, flattened into a `Dataset` with the class names. That
is everything below except the augmentations, of which `Dataset::flip_horizontal`
and `Dataset::standardize` have since landed; random crops and rotations still
want a batch-yielding loader rather than a one-shot decode into memory.

**What:** A directory-per-class loader, decoding, resize, and the standard
augmentations (crop, flip, normalize).

**Why:** Neither vision path is testable without it.

**Effort:** M.

**Risks:** `image` as a dependency, feature-gated. Augmentation is CPU work
that will bottleneck a GPU step unless it is threaded — rayon is already a
dependency and covers this.

**Depends on:** Streaming dataset trait.

## Other model families

### Encoder-only (BERT-style)

*Half landed:* bidirectional attention, the part both this and the
vision-transformer item wait on. `TransformerBuilder::bidirectional(true)` drops
the causal mask from every block, and `MultiHeadAttention::set_causal` does it
for one layer. The change is a flag and one softmax slice: the backward pass
already treated a masked weight as a zero probability rather than as a special
case, so it needed no edit, and a finite-difference check on the bidirectional
layer confirms that. The ONNX export drops the mask node with it.

The non-causal kernel risk below turned out to be avoidable rather than solved:
`to_cuda` refuses a bidirectional model, because the flash kernel in
`src/cuda_flash.rs` masks every key after the query and would otherwise train a
different model than the host. The same goes for the KV cache — `forward_cached`
refuses, since an appended token changes the tokens before it — so `generate`
and `decoder` report it through that.

*The rest landed too:* `MaskedBatch` and `masked_lm_loss` in `src/masked_lm.rs`,
with `TransformerLm::train_step_masked` and `evaluate_masked`.
`MaskedBatch::corrupt` does BERT's 80/10/10 substitution and guarantees one
masked position per sequence; `MaskedBatch::new` takes a caller's own corruption,
which is what a whole-word or span objective needs. There is no new head: the
existing unembedding already maps a hidden state to the vocabulary, and the loss
is the same cross-entropy reading a row's own token rather than the next one —
`causal_lm_loss` was split so both share one softmax pass.

So this item is done, except that a real encoder also wants a `[CLS]`-style
pooled output and a classification head on top of it, which is a linear layer
over one row and can wait for someone who needs it.

**What:** Bidirectional attention plus a masked-language-model head.

**Effort:** M, and it overlaps almost entirely with the vision-transformer
work — both need the same bidirectional mask.

**Risks:** Same non-causal kernel problem. If both this and ViT are wanted,
do the bidirectional mask once and let them share it.

### Encoder-decoder (seq2seq)

**What:** A separate encoder stack plus cross-attention in the decoder.

**Effort:** L.

**Risks:** Cross-attention is a new attention variant with its own hand-written
backward and its own KV-cache semantics during generation. Decoder-only models
have largely displaced this shape for the library's stated use cases.

### Diffusion

**What:** U-Net or DiT backbone, noise schedules, samplers.

**Effort:** L.

**Risks:** Needs the convolution work (U-Net) or the ViT work (DiT) as a
prerequisite, and then a substantial amount of new training machinery on top.
Listed for completeness; not a near-term candidate.

## Cross-cutting

### LoRA and adapters

*Landed.* `TransformerLm::add_lora`, `merge_lora`, `save_lora` and `load_lora`,
on the CPU and on CUDA. `Param::freeze` releases a base parameter's gradient
and both Adam moments, which is the memory saving, and `DeviceParam` never
allocates them in the first place for a frozen parameter. The adapters ride
inside `Linear`, so the module tree, `params_mut`, `save_bin`, `save_json` and
`to_cuda` all carry them without a special case.

The device path folds `scale * up . down` into the packed weight `Gpu::pack`
already builds, which is why the projection GEMMs, the flash-attention kernels
and the MoE gather and scatter needed no changes: every reader of a packed
weight sees the adapted one. Only the weight-gradient sites are adapter-aware,
and the routed expert's down projection moved onto the packed path so that it
is one of them. Attaching and merging remain host operations.

**What:** Low-rank adapters on the attention and FFN projections, with only
the adapter weights trainable and saved.

**Why:** Turns "I have one 12 GB card" from a training-from-scratch constraint
into a fine-tuning workflow, which is what most users with one card actually
want to do. Probably the highest value-to-effort ratio on this list.

**Effort:** M on the CPU, L including CUDA. `Linear` looks like the one place
every projection lives, but GPU *training* does not go through it:
`train_step`, `forward_batch` and `backward` dispatch to `crate::gpu_model`
(`src/transformer.rs:596`, `:838`), which keeps activations device-resident and
sequences its own GEMMs. `Linear::forward` is the decode path only. A GPU
adapter means extra GEMMs and gradients inside `gpu_model.rs`, which is 3,765
lines.

**Risks:** The CPU-only version is cheap and almost useless, because
fine-tuning on one card is the entire point. Interacts with mixed precision and
with the checkpoint format, which would need to distinguish a base model from
an adapter.

**Depends on:** Nothing.

### Quantization-aware training

**What:** Fake-quantize in the forward pass so the int8 checkpoint path
(`Precision::Q8`) stops costing accuracy.

**Why:** The README already warns that Q8 rounding shows up as a visible step
in a resumed loss curve. QAT is the principled fix.

**Effort:** M.

**Risks:** Straight-through estimators interact with the existing BF16 GEMM
path in ways that need careful numerical testing. The finite-difference tests
that guard every other backward pass do not straightforwardly apply here.

**Depends on:** Nothing.

*Landed.* `TransformerLm::quantization_aware(bool)` sets a `fake_quantize` flag
on every weight `quantize` replaces — the embedding table, every projection in
every block, and an untied head. `Linear::base_forward` and
`Embedding::unembed` then run the *same* `Quantized::matmul_rhs_transposed`
inference will run, rather than multiplying a rounded copy in full precision,
so what training sees is the arithmetic the checkpoint does. The embedding
gather rounds only the row it read, through a new `Quantized::round_row`: an
int8 scale covers one row and is read off that row alone.

The backward pass needed no straight-through code. It already differentiates
the stored `f32` weight, and the stored weight is what the optimizer moves, so
the estimator is what the existing code does with the flag on. The BF16 GEMM
risk did not materialize either: that path is `gpu_model`, which never calls a
`Linear` method, and a model on a device now reports that it cannot train
quantization-aware rather than ignoring the flag.

The risk about testing was real. A test that trains twice and compares what
`quantize` then costs measures about 1e-5 of loss on a model small enough for
`cargo test`, which is noise — it passed or failed on the step count. What is
tested instead is exact: a fake-quantized forward pass and the quantized
model's forward pass produce the same loss, and a fake-quantized weight still
moves by less than one grid step per update, which is what says the gradients
are flowing through the rounding rather than dying in it.

### More optimizers

**What:** Lion, Muon, Shampoo, or whatever has survived contact with reality by
the time this is picked up.

**Why:** Cheap, self-contained, and `Optimizer` is a small enum
(`src/optimizers.rs`, 50 lines) that is re-read every step.

**Effort:** S each.

**Risks:** Each new optimizer's state has to be added to
`save_optimizer_state` / `load_optimizer_state`, and the format is not
versioned in a way that tolerates unknown optimizer states gracefully. Worth
fixing that first if more than one is added.

**Depends on:** Nothing.

*Landed:* Lion, `Optimizer::lion` and `lion_with_weight_decay`. The state format
needed no change and no versioning after all: `Param` allocates both moment
buffers whatever the optimizer is, because the optimizer is a value that can be
swapped between steps, so Lion writes `moment1` and leaves `moment2` at zero and
`RBOPT001` still round-trips. That also means Lion does not save the memory it
saves elsewhere. An optimizer with state of a different shape, Shampoo being the
obvious one, is where the format question actually arrives.

Lion is CPU only. The CUDA and Metal paths report `UnsupportedCuda` and
`UnsupportedMetal` rather than quietly taking an Adam step; each needs an
elementwise kernel of one line, which is cheap to write and was not worth
writing untested.

### ONNX export

*Landed.* `Network::save_onnx` — dense networks, one `Gemm` plus one activation
node per layer — and `TransformerLm::save_onnx(path, seq_len)`, the transformer
half. Both write protobuf by hand (`src/onnx_export.rs`) and are verified by
running the file back through `tract` and comparing against the forward pass.

The dependency question answered itself: the encoding side is a hundred lines
of varints, and the graph construction is the work whatever writes the bytes.
The operator coverage the risk named was smaller than it looked — RMSNorm is
six opset-13 nodes, rotary positions nine, SwiGLU four, and grouped-query
attention is `Expand` around a reshape. A block comes to about forty-five
nodes.

What is refused rather than approximated: MoE layers, whose routing is
data-dependent control flow rather than a graph of tensor ops; an unmerged LoRA
adapter, which `merge_lora` folds in first; and a quantized or device-resident
model. The graph is fixed at one sequence length, because the rotary tables and
the causal mask are initializers, and it carries no KV cache, so it is a seam
for scoring and short prompts rather than a fast decode loop.

**What:** The other direction of the existing ONNX support — write a trained
RustingBrain model out for another runtime to serve.

**Why:** Currently a model trained here can only be run here. Export makes the
library usable as a training step in a larger pipeline.

**Effort:** M.

**Risks:** `tract-onnx` is a reader, not a writer, so this means constructing
protobuf by hand or taking another dependency. Operator coverage for
SwiGLU, RMSNorm, GQA and MoE routing is the real work, and MoE may not have a
clean ONNX representation at all.

**Depends on:** Nothing.

## Image generation hosting

The goal is to run any published image-generation model — FLUX.2-klein-4B is
the one that prompted this, but the design target is the family, not the file:
a rectified-flow transformer, a UNet diffusion model, and whatever ships next
should all load and run without the crate learning a fourth architecture from
scratch. No new dependency: the weight reader and the tokenizer are written
here rather than pulled in.

The reference hardware is one RTX 3060 with 12 GB, which sets the memory
budget. A 4B model at bf16 is about 8 GB of weights before activations, so
everything below assumes that the text encoder, the denoiser and the decoder
take turns on the card rather than sharing it.

The work splits into stages that are each useful on their own.

### Stage 1 — the weight reader — **landed** (`src/safetensors.rs`)

Every model published since 2023 ships `.safetensors`, usually sharded behind
a `model.safetensors.index.json`. `SafeTensors` parses the header and reads one
tensor at a time by seeking, so an 8 GB checkpoint is never held whole, and
`ShardedSafeTensors` opens shards lazily through the index. Every dtype in the
wild converts to `f32` on read, including bf16, fp16 and both fp8 encodings,
with hand-written conversions covering subnormals.

**Left out:** writing the format, and memory mapping. Reading is what hosting
needs; `save_bin` already covers writing.

### Stage 2 — the sampling loop — **landed** (`src/diffusion.rs`)

`Denoiser` is the seam a model plugs into, `Scheduler` is the noise schedule,
and `sample` is the loop. Flow matching with a resolution shift covers FLUX,
FLUX.2 and Stable Diffusion 3; DDIM over the scaled-linear betas covers the
Stable Diffusion 1.x and XL line. Classifier-free guidance is optional and off
by default, which is what a distilled model wants. The loop touches latents
only elementwise, so it does not care whether a latent is a patch sequence or a
convolutional feature map.

`Solver` says how a step is taken: Euler, Euler ancestral, or DPM++ 2M, which
is second order from the previous step's answer and so costs no extra forward
pass. A checkpoint's `scheduler_config.json` names the one it was published
with, and the loader reads it.

**Left out:** the higher-order single-step solvers (DPM++ 2S, Heun), which read
the model twice a step for what 2M gets from the step before.

### Stage 3 — the decoder — **landed** (`src/conv.rs`, `src/vae.rs`)

A latent is not an image. Every family ends in a VAE decoder, and the decoder
is the one part that is pure convolution. `Conv2d` lowers to the matrix
multiply the crate already has through `im2col`, so it inherits the tuned
kernel and the threading rather than being a seven-deep loop nest;
`GroupNorm`, SiLU, nearest-neighbour upsampling and pixel shuffle sit beside
it, and `FeatureMap` carries the channel-height-width shape between them.

`VaeDecoder` assembles those into the `AutoencoderKL` decoder every checkpoint
in this line ships, and `VaeDecoder::load` reads one straight from
`.safetensors` under the `diffusers` key names — stem, a middle pair of
residual blocks around one self-attention, a ladder of residual blocks that
doubles the resolution per rung, and a final convolution to three channels.
`VaeConfig::flux` carries FLUX's sixteen latent channels and its scale and
shift. `to_rgb8` and, with `--features images`, `save_png` finish the job.

Shapes stayed two-dimensional: a convolution weight is four-dimensional, but
it is stored as the `[out, in * kh * kw]` matrix it flattens to, with the
kernel size beside it. That avoided an N-D tensor type that every existing
layer would then have had to learn.

`VaeEncoder` is the other half, and it is what image-to-image needs: the same
ladder run the other way, halving the resolution per rung with the asymmetric
padding `diffusers` uses, and ending in the mean of the distribution the model
was trained to predict. `ImagePipeline` attaches one when the checkpoint holds
it, and `generate_from_image` encodes a picture, noises it as far up the
schedule as `strength` asks for, and denoises from there.

**Left out:** backward passes and CUDA. Hosting is a forward pass.

### Stage 4 — the denoiser blocks — **landed** (`src/mmdit.rs`)

`Dit` is the MMDiT denoiser: double-stream blocks where image and text keep
separate weights and meet only inside one joint attention, single-stream
blocks over the concatenated sequence with a fused attention-and-feed-forward
projection, adaptive layer norm carrying the timestep into every block,
per-head query and key normalization, and rotary positions in two dimensions
over the patch grid. `Dit::load` reads a checkpoint under the key names FLUX
ships with, and `Dit` implements `Denoiser`, so stage 2's sampler drives it.

### Stage 4b — the UNet denoiser — **landed** (`src/unet.rs`)

`Unet` is the other denoiser: a ladder of residual convolution blocks, with
cross-attention to the prompt on the rungs that have it, the noise level added
to every block through a projection, and every rung on the way up handed the
matching rung from the way down. It is what Stable Diffusion 1.x, 2.x and XL
are. Almost nothing about the shape comes from the configuration — how many
blocks there are, which of them attend, and whether the projections into
attention are linear layers or one-by-one convolutions are read from which keys
the checkpoint holds — so a model with a ladder nobody wrote down still loads.
The sampler's continuous noise level is turned back into the step index the
network was trained to count in, and XL's pooled prompt and crop size go
through the extra projection those checkpoints carry.

**Left out:** image conditioning — ControlNet and the inpainting shapes. Those
are more blocks of the same kind rather than a different model.

### Stage 5 — text conditioning — **landed** (`src/tokenizer.rs`,
`src/text_encoder.rs`)

`Bpe` reads a `tokenizer.json` and tokenizes the way every byte-level model in
this field does, with the GPT-2 pre-tokenizer written as a state machine
because the crate carries no regular-expression engine. `TextEncoder` runs the
LLaMA-family decoder — Qwen2, Qwen3, LLaMA, Mistral — from a Hugging Face
checkpoint, including the per-head query normalization Qwen3 added, and hands
back per-token hidden states and a pooled vector.

`ClipTextEncoder` runs CLIP's text tower — learned positions, LayerNorm,
quick-GELU, causal attention — which is what Stable Diffusion and FLUX.1 pool
for their conditioning vector. The pooled state is read at the end-of-text
token, not the last position, because those models pad every prompt to the
length of the position table. `ImagePipeline` takes it as a second encoder for
the pooled vector alone, and `ImagePipeline::load` picks it up from a published
directory that has a `text_encoder_2`.

`T5Encoder` runs the T5 encoder FLUX.1 and Stable Diffusion 3 take their
per-token states from — bidirectional attention with a learned relative-position
bias, unscaled queries, gated feed-forward — and `Unigram` is the sentencepiece
tokenizer it reads prompts with, one Viterbi pass over the scored pieces.
`PromptEncoder` is the seam: a pipeline conditions through a LLaMA-family
decoder or through T5, and `ImagePipeline::load` reads which from the
encoder's own configuration.

**Left out:** the precompiled character map in the sentencepiece normalizer,
which folds Unicode before the split. Whitespace folding and the space marker
are what a typed prompt actually meets.

### Stage 6 — making it fit and making it fast — **part landed**

Weights at one byte each, which is what decides whether a model runs at all: a
4B-parameter denoiser is 16 GB in `f32` and 4 GB quantized. `ImagePipeline::
load_at`, `Dit::load_at` and `TextEncoder::load_at` take a `Precision`, and each
tensor is quantized as it is read, so loading never holds the whole model in
`f32`.

Measured on a 283M-parameter denoiser over 1536 tokens on twelve cores, before
anything was changed: 1291 MiB peak and 4.0 s per step in `f32`. At one byte per
weight, 430 MiB and 5.0 s — and 4.4 s once the quantized matmul was forked
across tokens rather than across the weight's rows, which is the right axis for
a sequence and the wrong one for the single token of a language-model decode.

The third part of the stage, loading and dropping one component at a time, needs
no code: `condition`, `sample_latent` and `decode` are separate for that reason,
and a caller drops each part as it finishes.

**Still open:** CUDA for these blocks. The existing kernels are written for a
causal language model, and a diffusion step is a bidirectional attention over a
sequence with no cache — the fused attention kernel needs a non-causal path
before any of this runs on the card. `bf16` weights are a second option worth
measuring against the byte path once that exists.

### Stage 7 — the hosting API — **landed** (`src/pipeline.rs`)

`ImagePipeline` holds the four parts and runs them in order: prompt to ids, ids
to hidden states, noise to latent, latent to pixels, with the patch packing and
unpacking in between. `generate` is the whole run; `condition`, `sample_latent`
and `decode` are the same run split so a caller can drop each part as it
finishes, which is what makes a 4B model fit a 12 GB card. The seam checks that
the parts agree about channels rather than producing a wrong image.

`ImagePipeline::load` takes the directory a model was published in and reads
the rest: each component's `config.json` states its own shape, the patch size
is whatever squares the latent's channels up to the width the denoiser reads,
and the schedule's shift comes from the scheduler's configuration. Both
published spellings of a transformer checkpoint are read, the reference one and
the diffusers one, so a repacked model loads without conversion.
`examples/text_to_image.rs` is the whole thing as a command.

The loader reads which architecture a directory holds rather than being told:
a `unet` beside a `transformer`, a T5 or CLIP or LLaMA-family prompt encoder,
one CLIP tower or two. A UNet directory is sampled with DDIM at the betas its
scheduler names and with classifier-free guidance, which is what that line was
trained for; a transformer directory keeps the rectified-flow path.

A directory with a `text_encoder_3` is Stable Diffusion 3: two CLIP towers
side by side, padded out to T5's width, with the T5 states stacked underneath
them in one sequence and the pooled vector taken from both towers together.

**Left out:** ControlNet and the inpainting shapes, which condition on a second
image rather than on a prompt.

## Recommended first moves

If any of this gets built, the order that unblocks the most for the least
work:

1. **Streaming dataset trait** — S to M, and four other items are waiting
   behind it. It is also the item that most directly addresses the original
   complaint about loading datasets.
2. **LoRA** — M on the CPU but L with CUDA, and only the CUDA version is worth
   having. It matches what a one-card user is most likely to want to do, but it
   is not the cheap item it first looks like.
3. **Data-parallel training** — L, but it is the item whose absence the README
   currently apologizes for, and two cards on one host is a testable target.

Everything else should wait for evidence that someone needs it.

## Not on this list

Kept out deliberately, not by oversight:

- **Reinforcement learning, RLHF, DPO.** A different training loop, a different
  data shape, and a large surface area. If it happens it is a separate library.
- **Serving, batching servers, HTTP APIs.** Out of scope for a library; ONNX
  export is the better seam.
- **A Python binding.** The README's first claim is that there is no Python in
  the loop. Adding one would need a much better reason than convenience.
- **An autograd engine.** A rewrite, not a feature. Worth revisiting only if
  the hand-written-backward tax on new layers becomes the thing actually
  blocking work.
