# RustingBrain / RustingLLM Training Throughput Plan

> **For agentic workers:** Use `superpowers:subagent-driven-development` or
> `superpowers:executing-plans` to work through this task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise training throughput and remove the memory and storage ceilings
that stop a 55M-parameter model from being trained on tens of billions of
tokens on a single RTX 3060.

**Architecture:** Two repositories. `RustingBrain` is the library (model,
CUDA kernels, optimizer, serialization). `RustingLLM` is the driver
(tokenizer, corpus, training loop, checkpoints) and depends on it by path.
Most of the cheap wins are in `RustingLLM`; the wins that need kernel work are
in `RustingBrain`. Tasks are ordered so the cheap ones land first and the
expensive ones are justified by measurement rather than assumption.

**Tech Stack:** Rust 2024, `cudarc` 0.19 (cuBLAS + NVRTC), `rayon`,
`tokenizers` 0.20, `clap` 4.

**Hardware:** RTX 3060 12 GB (GA106, 28 SMs, 360 GB/s, ~25 TFLOPS BF16 tensor
with FP32 accumulate), Ryzen 5 7600X, 32 GB DDR5, one NVMe with 116 GB free.

---

## Global Constraints

- Weights, gradients and Adam moments stay FP32. Reduced precision is for
  GEMM inputs and the LM head only. This is already true and must remain true.
- No new crate dependencies unless a task names one explicitly.
- Every task ends with a measurement or a test, run and recorded, before the
  commit. Performance claims without a before/after number are not done.
- `cargo test --lib` must pass at the end of every task. It is 107 tests today.
- Benchmarks are only valid on an otherwise idle GPU. See Task 0.

---

## Measured Baseline

Superseded. `docs/baseline.md` holds the clean-GPU table; the numbers that
were here were taken while a second training job held 5754 MiB and 100% of
the device, and the same configurations run **2.14x faster** without it.

The contaminated figures are worth keeping only as the reason Task 0 exists:
they put MFU at ~15%, which made Tasks 6 and 7 look necessary. Clean MFU is
30.7-37.0%, and both tasks are closed.

| Configuration | contended | clean | ratio |
|---|---|---|---|
| v32000 d512 L8 16x512 moe | 16989 | 36302 | 2.14x |
| v16384 d512 L8 16x512 moe | 18651 | 40115 | 2.15x |
| v16384 d512 L8 16x512 dense | 21295 | 43704 | 2.05x |
| v16384 d512 L8 16x1024 dense | 14453 | 34089 | 2.36x |

Every OOM recorded during the investigation (batch 24, 12 layers, seq 1024 at
batch 16) was that job's 5.7 GB. None of them reproduce on an idle card.

---

## Task 0: Establish a clean baseline — DONE

**Files:**
- Use: `RustingBrain/examples/sweep_arch.rs` (already written)
- Use: `RustingBrain/examples/vram_probe.rs` (already written)
- Create: `RustingBrain/docs/baseline.md`

**Interfaces:**
- Produces: a recorded table of `tok/s` and peak VRAM per configuration on an
  idle GPU. Tasks 3, 6 and 7 are accepted or rejected against these numbers.

- [x] **Step 1: Stop the contending training run**

Your current run is at epoch 3 of a 368M-token corpus at `--seq-len 128`. It
checkpoints every 500 steps to `models/rusting_ogre_2_50m.json`, so stopping
it loses at most 500 steps, and `--resume` picks it back up.

```bash
kill -TERM 859200          # or Ctrl-C in its terminal
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv
```

Expected: no `./target/release/train` row. Desktop apps should leave roughly
600 MiB used.

- [x] **Step 2: Re-run the architecture sweep**

```bash
cd ~/Rusting/RustingBrain
cargo build --release --features cuda --example sweep_arch
B=target/release/examples/sweep_arch
for cfg in "32000 512 8 16 512 1" "16384 512 8 16 512 1" "16384 512 8 16 512 0" \
           "16384 512 8 32 512 0" "16384 512 8 16 1024 0" "16384 512 12 16 512 0" \
           "16384 768 12 16 512 0" "32000 512 8 128 128 1"; do
  $B ${=cfg}
done
```

Expected: every configuration that OOM'd before now runs. Record the table.

- [x] **Step 3: Record peak VRAM for the configuration you intend to train**

```bash
target/release/examples/sweep_arch 16384 512 8 32 512 0 &
while kill -0 $! 2>/dev/null; do
  nvidia-smi --query-gpu=memory.used --format=csv,noheader
done | sort -n | tail -1
```

- [x] **Step 4: Compute MFU and write it down**

```
FLOPs/token  = 6 * active_params + 12 * n_layers * seq_len * d_model
TFLOPS       = tok/s * FLOPs/token / 1e12
MFU          = TFLOPS / 25.0
```

Under contention this was 3.9 TFLOPS, ~15% MFU. If the clean number is above
30% MFU, Tasks 6 and 7 are not worth their complexity and you should skip
straight to Task 8 and accept the wall-clock. **This is the decision this task
exists to inform — do not skip it.**

- [x] **Step 5: Commit the baseline**

```bash
git add docs/baseline.md examples/sweep_arch.rs examples/vram_probe.rs
git commit -m "docs: record clean-GPU training throughput baseline"
```

---

## Task 1: Binary checkpoints in the training loop — DONE

`save_bin` and `load_bin` already exist in `RustingBrain`. `RustingLLM` still
calls `save_json`. Five checkpoint files in `models/` hold 2.1 GB today, on a
disk with 116 GB free that Task 2 wants 50 GB of.

**Files:**
- Modify: `RustingLLM/src/bin/train.rs:210-219` (`checkpoint_model`)
- Modify: `RustingLLM/src/bin/train.rs:66-67` (the `--checkpoint` default)
- Modify: `RustingLLM/src/bin/train.rs:133` (the `--resume` load)
- Modify: `RustingLLM/src/bin/generate.rs` (whichever line loads the model)

**Interfaces:**
- Consumes: `rusting_brain::{Precision, TransformerLm}`;
  `TransformerLm::save_bin(&mut self, path, Precision) -> Result<(), NetworkError>`
  and `TransformerLm::load_bin(path) -> Result<Self, NetworkError>`.
- Produces: checkpoints at `models/*.rbw` instead of `models/*.json`.

- [x] **Step 1: Change the checkpoint default and the save call**

In `train.rs`, change the `--checkpoint` default:

```rust
    #[arg(long, default_value = "models/checkpoint.rbw")]
    checkpoint: String,
```

and the body of `checkpoint_model`:

```rust
    let checkpoint_model = |model: &mut TransformerLm, epochs_done: usize| -> Result<()> {
        #[cfg(feature = "cuda")]
        if model.on_device() {
            model.sync_from_device()?;
        }
        // F32, not Q8: Q8 rounding costs about 0.4% per weight, which a
        // resumed Adam run turns into a visible step in the loss curve.
        model.save_bin(&args.checkpoint, Precision::F32)?;
        model.save_optimizer_state(&optimizer_state_path)?;
        std::fs::write(&progress_path, epochs_done.to_string())?;
        Ok(())
    };
```

- [x] **Step 2: Change the resume path**

```rust
        let mut model = TransformerLm::load_bin(&args.checkpoint)?;
```

- [x] **Step 3: Update the import**

```rust
use rusting_brain::{Precision, TransformerLm};
```

Delete the old `use rusting_brain::transformer::TransformerLm;` line.

- [x] **Step 4: Verify a save/resume round trip**

```bash
cd ~/Rusting/RustingLLM
cargo run --release --features cuda --bin train -- \
  --gpu --mixed-precision --epochs 1 --seq-len 128 --batch-size 32 \
  --checkpoint models/roundtrip.rbw --checkpoint-every 20
# let it pass step 20, then Ctrl-C
ls -la models/roundtrip.rbw
cargo run --release --features cuda --bin train -- \
  --gpu --mixed-precision --epochs 1 --seq-len 128 --batch-size 32 \
  --checkpoint models/roundtrip.rbw --resume
```

Expected: `roundtrip.rbw` is ~221 MB, not ~670 MB. The resumed run prints
`restored optimizer state ... at step N` and its first reported `lm_loss` is
within a few percent of the loss printed just before the interrupt. A first
loss that jumps by more than ~20% means the optimizer state did not restore.

- [x] **Step 5: Convert the checkpoint you care about, then reclaim the disk**

```bash
cd ~/Rusting/RustingLLM
cat > /tmp/convert.rs <<'EOF'
fn main() {
    let mut args = std::env::args().skip(1);
    let (input, output) = (args.next().unwrap(), args.next().unwrap());
    let mut model = rusting_brain::TransformerLm::load_json(&input).unwrap();
    model.save_bin(&output, rusting_brain::Precision::F32).unwrap();
    println!("{input} -> {output}");
}
EOF
mkdir -p src/bin && cp /tmp/convert.rs src/bin/convert_checkpoint.rs
cargo run --release --bin convert_checkpoint -- \
  models/rusting_ogre_2_50m.json models/rusting_ogre_2_50m.rbw
```

Verify the `.rbw` loads and generates before deleting anything:

```bash
cargo run --release --bin generate -- --checkpoint models/rusting_ogre_2_50m.rbw
```

Only once that prints sensible output:

```bash
rm models/checkpoint_50m.json models/rusting_ogre_1_50m.json models/rusting_ogre_2_50m.json
```

- [x] **Step 6: Commit**

```bash
git add src/bin/train.rs src/bin/generate.rs src/bin/convert_checkpoint.rs
git commit -m "feat: checkpoint to the binary format instead of JSON"
```

**Honest expected gain:** 1.33s saved per checkpoint, every 500 steps — under
1% of wall-clock. The real wins are 3x less disk per checkpoint and a 7.5x
faster resume. Do it because Task 2 needs the disk, not because it is fast.

---

## Task 2: Memory-mapped token stream — DONE

This is the task that makes 50B tokens possible at all. Today
`train.rs:106-117` reads the whole corpus into a `String`, tokenizes it, and
holds the result as `Vec<Vec<u32>>` — 2.87M separate heap allocations for
368M tokens, measured at 2.78 GB RSS. At 50B tokens that is ~380 GB of RAM.
It cannot work, and no amount of GPU tuning changes that.

The fix is to tokenize once, ahead of time, into a flat `u16` file, then
`mmap` it and index into it. A 32000-entry vocabulary fits in `u16`, so 50B
tokens is 100 GB on disk and zero bytes of resident heap.

**Files:**
- Create: `RustingLLM/src/bin/tokenize_corpus.rs`
- Create: `RustingLLM/src/tokens.rs`
- Modify: `RustingLLM/src/lib.rs` (add `pub mod tokens;`)
- Modify: `RustingLLM/src/bin/train.rs:105-119` (replace corpus loading)
- Modify: `RustingLLM/Cargo.toml` (add `memmap2`)

**Interfaces:**
- Produces: `tokens::TokenFile` with
  - `TokenFile::open(path: &Path) -> Result<TokenFile>`
  - `TokenFile::len(&self) -> usize` — total tokens
  - `TokenFile::sequences(&self, seq_len: usize) -> usize`
  - `TokenFile::sequence(&self, index: usize, seq_len: usize, out: &mut Vec<u32>)`
- Consumed by: Task 4, which batches over `sequence` indices.

- [x] **Step 1: Add the dependency**

`memmap2` is the one new crate this plan allows. Writing a correct mmap
wrapper by hand is more unsafe code than it is worth.

```toml
memmap2 = "0.9"
```

- [x] **Step 2: Write the failing test**

Create `RustingLLM/src/tokens.rs`:

```rust
//! A pre-tokenized corpus held on disk and read through the page cache.
//!
//! Token ids are `u16`. A 32000-entry vocabulary fits, the file is half the
//! size of a `u32` one, and nothing is resident: the kernel pages in the
//! windows the training loop actually touches.

use anyhow::{Context, Result, bail};
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

pub struct TokenFile {
    map: Mmap,
}

impl TokenFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let length = file.metadata()?.len();
        if length % 2 != 0 {
            bail!("{} holds {length} bytes, which is not a whole number of u16 ids", path.display());
        }
        // SAFETY: the file is opened read-only and is not written while a
        // training run holds it. A concurrent truncation would be undefined,
        // which is why `tokenize_corpus` writes to a temporary path and
        // renames into place.
        let map = unsafe { Mmap::map(&file)? };
        Ok(Self { map })
    }

    pub fn len(&self) -> usize {
        self.map.len() / 2
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// How many non-overlapping windows of `seq_len` the file holds.
    pub fn sequences(&self, seq_len: usize) -> usize {
        if seq_len < 2 { 0 } else { self.len() / seq_len }
    }

    /// Writes window `index` into `out`, replacing whatever it held.
    pub fn sequence(&self, index: usize, seq_len: usize, out: &mut Vec<u32>) {
        out.clear();
        let start = index * seq_len * 2;
        let end = start + seq_len * 2;
        out.extend(
            self.map[start..end]
                .chunks_exact(2)
                .map(|pair| u16::from_le_bytes([pair[0], pair[1]]) as u32),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_ids(ids: &[u16]) -> tempfile::NamedTempFile {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        for id in ids {
            file.write_all(&id.to_le_bytes()).unwrap();
        }
        file.flush().unwrap();
        file
    }

    #[test]
    fn windows_do_not_overlap_and_cover_the_file() {
        let file = write_ids(&[1, 2, 3, 4, 5, 6, 7]);
        let tokens = TokenFile::open(file.path()).unwrap();

        assert_eq!(tokens.len(), 7);
        // Seven ids hold three windows of two; the odd id at the end is dropped.
        assert_eq!(tokens.sequences(2), 3);

        let mut out = Vec::new();
        tokens.sequence(0, 2, &mut out);
        assert_eq!(out, vec![1, 2]);
        tokens.sequence(2, 2, &mut out);
        assert_eq!(out, vec![5, 6]);
    }

    #[test]
    fn an_odd_byte_count_is_not_a_token_file() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&[0u8, 1, 2]).unwrap();
        file.flush().unwrap();
        assert!(TokenFile::open(file.path()).is_err());
    }
}
```

Add to `Cargo.toml`:

```toml
[dev-dependencies]
tempfile = "3"
```

- [x] **Step 3: Run the tests to verify they fail**

```bash
cd ~/Rusting/RustingLLM
cargo test --lib tokens
```

Expected: FAIL — `tokens.rs` is not declared as a module yet.

- [x] **Step 4: Declare the module**

Add to `RustingLLM/src/lib.rs`:

```rust
pub mod tokens;
```

- [x] **Step 5: Run the tests to verify they pass**

```bash
cargo test --lib tokens
```

Expected: 2 passed.

- [x] **Step 6: Write the offline tokenizer**

Create `RustingLLM/src/bin/tokenize_corpus.rs`:

```rust
//! Turns text files into one flat `u16` token file.
//!
//! Run once per corpus. The training loop then memory-maps the result
//! instead of tokenizing 1.2 GB of text at every start.

use anyhow::{Result, bail};
use clap::Parser;
use rusting_llm::load_tokenizer;
use std::io::{BufWriter, Write};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "tokenizer.json")]
    tokenizer: String,

    /// Text files to encode, in order.
    #[arg(long, num_args = 1..)]
    input: Vec<String>,

    #[arg(long, default_value = "data/corpus.u16")]
    output: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let tokenizer = load_tokenizer(&args.tokenizer)?;
    let vocab = tokenizer.get_vocab_size(true);
    if vocab > u16::MAX as usize + 1 {
        bail!("vocabulary of {vocab} does not fit in u16; widen the token file to u32 first");
    }

    // Written to a temporary path and renamed, so a training run that has the
    // old file mapped never sees a half-written one.
    let temporary = format!("{}.partial", args.output);
    let mut writer = BufWriter::with_capacity(1 << 20, std::fs::File::create(&temporary)?);
    let mut total = 0usize;

    for path in &args.input {
        let text = std::fs::read_to_string(path)?;
        // Same blocking as `tokenize_and_chunk`: one `encode` over hundreds of
        // megabytes allocates offsets and word ids sized to the token count
        // and OOM-kills the process.
        const BLOCK_CHARS: usize = 4_000_000;
        let bytes = text.as_bytes();
        let mut start = 0;
        let mut file_tokens = 0usize;
        while start < bytes.len() {
            let mut end = (start + BLOCK_CHARS).min(bytes.len());
            while end < bytes.len() && !text.is_char_boundary(end) {
                end += 1;
            }
            let encoding = tokenizer
                .encode(&text[start..end], false)
                .map_err(|e| anyhow::anyhow!("tokenizing {path}: {e}"))?;
            for id in encoding.get_ids() {
                writer.write_all(&(*id as u16).to_le_bytes())?;
            }
            file_tokens += encoding.get_ids().len();
            start = end;
        }
        println!("{path}: {file_tokens} tokens");
        total += file_tokens;
    }

    writer.flush()?;
    drop(writer);
    std::fs::rename(&temporary, &args.output)?;
    println!(
        "{}: {total} tokens, {:.1} GB",
        args.output,
        total as f64 * 2.0 / 1e9
    );
    Ok(())
}
```

- [x] **Step 7: Tokenize the corpus you already have**

```bash
cd ~/Rusting/RustingLLM
cargo run --release --bin tokenize_corpus -- --input data/corpus.txt --output data/corpus.u16
ls -la data/corpus.u16
```

Expected: ~368M tokens, ~0.74 GB. Compare the token count to the
`2877837 sequences x 128` your current run reports — they should agree to
within a rounding of the final partial chunk.

- [x] **Step 8: Replace corpus loading in the training loop**

In `train.rs`, delete the `let mut sequences = Vec::new(); ... ` block at
lines 105-119 and replace it with:

```rust
    let tokens = rusting_llm::tokens::TokenFile::open(&args.tokens)?;
    let sequence_count = tokens.sequences(args.seq_len);
    if sequence_count == 0 {
        anyhow::bail!(
            "{} holds {} tokens, fewer than one sequence of {}",
            args.tokens,
            tokens.len(),
            args.seq_len
        );
    }
    println!(
        "{}: {} tokens, {sequence_count} sequences of {}",
        args.tokens,
        tokens.len(),
        args.seq_len
    );
```

Replace the `--corpus` argument with:

```rust
    #[arg(long, default_value = "data/corpus.u16")]
    tokens: String,
```

and replace the batching loop's source. The shuffle now runs over indices,
not over 2.87M heap-allocated vectors:

```rust
        let mut order: Vec<u32> = (0..sequence_count as u32).collect();
        order.shuffle(&mut rng);

        let mut scratch: Vec<Vec<u32>> = vec![Vec::with_capacity(args.seq_len); args.batch_size];
        for indices in order.chunks(args.batch_size) {
            for (slot, index) in scratch.iter_mut().zip(indices) {
                tokens.sequence(*index as usize, args.seq_len, slot);
            }
            let refs: Vec<&[u32]> = scratch[..indices.len()].iter().map(|s| &s[..]).collect();
            let batch_tokens = indices.len() * args.seq_len;
            let total = model.train_step(&refs)?;
            // ... the rest of the loop body is unchanged
```

- [x] **Step 9: Verify RAM dropped and the loss curve did not move**

```bash
cargo run --release --features cuda --bin train -- \
  --gpu --mixed-precision --epochs 1 --seq-len 128 --batch-size 128 \
  --checkpoint models/mmap_check.rbw &
sleep 120
ps -o rss= -p $!
```

Expected: RSS under 400 MB, down from 2.78 GB. The first few reported
`lm_loss` values should sit in the same range a fresh run reported before
this change — the data is identical, only its storage moved.

- [x] **Step 10: Commit**

```bash
git add Cargo.toml src/lib.rs src/tokens.rs src/bin/tokenize_corpus.rs src/bin/train.rs
git commit -m "feat: stream training tokens from a memory-mapped u16 file"
```

**Expected gain:** 2.78 GB RSS to under 400 MB, ~2 minutes of startup
tokenization removed from every run, and no upper bound on corpus size except
disk. This is the prerequisite for every token-count target above ~2B.

---

## Task 3: Drop the mixture of experts — DONE

Measured, under contention but on identical footing:

```
v16384 d512 L8 16x512 moe     27.7M active   18651 tok/s
v16384 d512 L8 16x512 dense   30.9M active   21295 tok/s
```

Dense is 14% faster **with 12% more active parameters**. The reason is in
`src/moe.rs` and `gpu_model.rs:1376` (`forward_moe`): eight experts at
`moe_d_ff` 352 turn one large GEMM into many small ones, and a 28-SM card
cannot fill itself with them. `forward_moe` also does a
`download_signed`/`download` of the routing assignment at
`gpu_model.rs:1429-1430`, which drains the stream once per MoE layer per step.

MoE earns its complexity when the expert GEMMs are large enough to saturate
the device and when memory, not compute, is what you are trading. Neither
holds here.

**Files:**
- Modify: `RustingLLM/src/bin/train.rs:159` (the `moe_layers` call)

- [x] **Step 1: Add a flag rather than deleting the capability**

The library keeps MoE; this run stops using it.

```rust
    /// Use a mixture-of-experts feed-forward from this layer on. Off by
    /// default: measured on an RTX 3060, eight experts at moe_d_ff 352 run
    /// 14% slower than a dense SwiGLU with more active parameters, because
    /// the per-expert GEMMs are too small to fill 28 SMs.
    #[arg(long)]
    moe: bool,
```

and at the builder:

```rust
            .moe_layers(if args.moe { args.n_layers / 4..args.n_layers } else { 0..0 })
```

- [x] **Step 2: Verify both paths still build a model and take a step**

```bash
cd ~/Rusting/RustingLLM
cargo run --release --features cuda --bin train -- --gpu --mixed-precision \
  --epochs 1 --seq-len 128 --batch-size 64 --checkpoint models/dense_check.rbw 2>&1 | head -20
cargo run --release --features cuda --bin train -- --gpu --mixed-precision --moe \
  --epochs 1 --seq-len 128 --batch-size 64 --checkpoint models/moe_check.rbw 2>&1 | head -20
```

Expected: the dense run reports a higher `tok/s` and the printed parameter
counts show `1.00x` sparsity for dense against `1.54x` for MoE.

- [x] **Step 3: Commit**

```bash
git add src/bin/train.rs
git commit -m "feat: make the mixture-of-experts feed-forward opt-in"
```

**Expected gain:** +14% throughput, and a simpler model to debug over a
multi-week run.

---

## Task 4: Gradient accumulation — DONE

Batch size is currently bounded by VRAM, and the optimizer takes a step for
every batch. That couples two things that should be independent: how many
tokens fit on the device, and how many tokens one optimizer step should see.
A 55M model wants an effective batch in the hundreds of thousands of tokens;
the device holds tens of thousands.

`TransformerLm::train_step_batch` at `transformer.rs:830` fuses
`zero_grad`, forward, backward and `step(1.0)`. The three pieces are already
public and `step` already takes a scale. What is missing is an entry point
that runs forward and backward without the zeroing and the step.

**Files:**
- Modify: `RustingBrain/src/transformer.rs:830-855`
- Modify: `RustingLLM/src/bin/train.rs` (the batching loop)

**Interfaces:**
- Produces: `TransformerLm::accumulate_step(&mut self, batch: &TokenBatch) -> Result<TotalLoss, NetworkError>`
  — runs forward and backward, adding into the existing gradients. The caller
  is responsible for `zero_grad()` before the first accumulation and
  `step(1.0 / accumulated as f32)` after the last.

- [x] **Step 1: Write the failing test**

Add to the `tests` module in `RustingBrain/src/transformer.rs`:

```rust
    #[test]
    fn accumulating_two_half_batches_matches_one_whole_batch() {
        let ids: [&[u32]; 4] = [&[1, 2, 3, 4], &[5, 6, 7, 8], &[9, 10, 11, 12], &[13, 14, 15, 16]];

        let mut whole = tiny().seed(3).build().unwrap();
        whole.train_step(&ids).unwrap();

        let mut split = tiny().seed(3).build().unwrap();
        split.zero_grad();
        split.accumulate_step(&TokenBatch::new(&ids[..2]).unwrap()).unwrap();
        split.accumulate_step(&TokenBatch::new(&ids[2..]).unwrap()).unwrap();
        split.step(0.5);

        // Two half batches averaged are the whole batch, up to the order the
        // two sums were taken in.
        let (whole_logits, _) = whole.forward_train(&[[2u32, 4, 6]]).unwrap();
        let (split_logits, _) = split.forward_train(&[[2u32, 4, 6]]).unwrap();
        for (a, b) in split_logits.data.iter().zip(&whole_logits.data) {
            assert!(
                (a - b).abs() < 1e-4,
                "accumulated {a} against whole-batch {b}"
            );
        }
    }
```

- [x] **Step 2: Run it to verify it fails**

```bash
cd ~/Rusting/RustingBrain
cargo test --lib accumulating_two_half_batches
```

Expected: FAIL — `no method named accumulate_step`.

- [x] **Step 3: Add the method**

In `RustingBrain/src/transformer.rs`, directly after `train_step_batch`:

```rust
    /// Forward and backward over one batch, adding into the gradients that
    /// are already there.
    ///
    /// Unlike [`TransformerLm::train_step_batch`] this neither zeroes the
    /// gradients first nor steps the optimizer after, which is what lets a
    /// caller build an effective batch larger than the device holds:
    ///
    /// ```ignore
    /// model.zero_grad();
    /// for part in parts {
    ///     model.accumulate_step(part)?;
    /// }
    /// model.step(1.0 / parts.len() as f32);
    /// ```
    ///
    /// The scale belongs on the step rather than on each backward pass
    /// because Adam normalizes by the gradient's own second moment: scaling
    /// every accumulation identically would cancel out, and the averaging
    /// has to happen once, on the sum.
    pub fn accumulate_step(&mut self, batch: &TokenBatch) -> Result<TotalLoss, NetworkError> {
        #[cfg(feature = "cuda")]
        if let Some(context) = self.device.clone() {
            self.check_length(batch.seq_len(), 0)?;
            let (lm_loss, auxiliary_loss) = crate::gpu_model::train_step(self, &context, batch)?;
            self.check_device()?;
            return Ok(TotalLoss {
                lm_loss,
                auxiliary_loss,
            });
        }

        let (logits, cache) = self.forward_batch(batch)?;
        let loss = causal_lm_loss_batch(&logits, batch)?;
        self.backward(&cache, &loss.grad_logits)?;
        Ok(TotalLoss {
            lm_loss: loss.loss,
            auxiliary_loss: cache.auxiliary_loss(),
        })
    }
```

Check `gpu_model::train_step` before trusting this: if it zeroes gradients
internally rather than relying on `transformer.rs:837`, move that zeroing out
into `train_step_batch` so `accumulate_step` accumulates. The test in Step 1
is what catches it.

- [x] **Step 4: Run the test to verify it passes**

```bash
cargo test --lib accumulating_two_half_batches
cargo test --lib
```

Expected: the new test passes and all 107 existing tests still pass.

- [x] **Step 5: Commit the library change**

```bash
git add src/transformer.rs
git commit -m "feat: add accumulate_step for gradient accumulation"
```

- [x] **Step 6: Use it from the training loop**

Add the argument to `train.rs`:

```rust
    /// Device batches to accumulate before one optimizer step. Effective
    /// batch is `batch_size * accumulate * seq_len` tokens.
    #[arg(long, default_value_t = 1)]
    accumulate: usize,
```

and restructure the inner loop:

```rust
        for group in order.chunks(args.batch_size * args.accumulate) {
            model.zero_grad();
            let mut group_loss = 0.0f32;
            let mut group_aux = 0.0f32;
            let mut parts = 0usize;
            for indices in group.chunks(args.batch_size) {
                for (slot, index) in scratch.iter_mut().zip(indices) {
                    tokens.sequence(*index as usize, args.seq_len, slot);
                }
                let refs: Vec<&[u32]> = scratch[..indices.len()].iter().map(|s| &s[..]).collect();
                let total = model.accumulate_step(&TokenBatch::new(&refs)?)?;
                group_loss += total.lm_loss;
                group_aux += total.auxiliary_loss;
                parts += 1;
            }
            model.step(1.0 / parts as f32);
            running_loss += group_loss / parts as f32;
            running_aux += group_aux / parts as f32;
            epoch_tokens += group.len() * args.seq_len;
            step += 1;
            // ... reporting and checkpointing unchanged
        }
```

`TokenBatch` needs importing: `use rusting_brain::TokenBatch;`

- [x] **Step 7: Verify the loss curve is unchanged at equal effective batch**

```bash
cd ~/Rusting/RustingLLM
# 64 sequences per step, taken two ways
cargo run --release --features cuda --bin train -- --gpu --mixed-precision \
  --seq-len 512 --batch-size 64 --accumulate 1 --epochs 1 \
  --checkpoint models/acc_a.rbw 2>&1 | head -8
cargo run --release --features cuda --bin train -- --gpu --mixed-precision \
  --seq-len 512 --batch-size 16 --accumulate 4 --epochs 1 \
  --checkpoint models/acc_b.rbw 2>&1 | head -8
```

Expected: the two runs report `lm_loss` within about 1% of each other at the
same step count. The second uses roughly a quarter of the activation memory.

- [x] **Step 8: Commit**

```bash
git add src/bin/train.rs
git commit -m "feat: accumulate gradients across device batches"
```

**Expected gain:** no throughput change by itself. It buys the freedom to
raise `--seq-len` to 1024 (Task 5) and to keep a sane effective batch while
doing it, without OOM.

---

## Task 5: Sequence length, learning-rate schedule, and the run configuration — DONE

Two things in the current command line will cost more model quality than
every throughput item in this plan will buy back.

**`--seq-len 128`.** The model is being trained to predict inside a 128-token
window. A Rust function with its imports and signature does not fit. Worse,
`train.rs:161` sets `max_seq_len(args.seq_len * 2)` = 256, so the checkpoint
cannot later be run at a longer context without retraining. Attention cost per
token grows with sequence length, so this is a real trade: at `d_model` 512
and 8 layers, going from 128 to 1024 adds `12 * 8 * 896 * 512` ≈ 44 MFLOPs per
token against a parameter cost of `6 * 30.9e6` ≈ 185 MFLOPs, so roughly a 24%
slowdown for an 8x longer context. Take it.

**A constant learning rate of 4e-4 with no warmup.** Adam's second moment
estimate is near-zero for the first few hundred steps, so the bias correction
divides by a small number and the first updates are far larger than intended.
The standard fix is a linear warmup over the first ~2000 steps followed by a
cosine decay to about a tenth of the peak. `model.optimizer` is a public
field and `TransformerLm::step` re-reads it every call
(`transformer.rs:869`), so this needs no library change.

**Files:**
- Modify: `RustingLLM/src/bin/train.rs`

- [x] **Step 1: Add the schedule arguments**

```rust
    /// Linear warmup over this many optimizer steps, then cosine decay.
    #[arg(long, default_value_t = 2000)]
    warmup_steps: usize,

    /// Floor of the cosine decay, as a fraction of --learning-rate.
    #[arg(long, default_value_t = 0.1)]
    min_lr_ratio: f32,

    /// Total optimizer steps the cosine decays over. Defaults to the whole run.
    #[arg(long, default_value_t = 0)]
    total_steps: usize,
```

- [x] **Step 2: Write the schedule**

Add above `fn main`:

```rust
/// Linear warmup then cosine decay, the schedule almost every small language
/// model is trained with.
///
/// Warmup exists because Adam's second-moment estimate starts at zero: for
/// the first few hundred steps the bias correction divides by a very small
/// number, and a peak learning rate applied there moves the weights much
/// further than the same rate does later.
fn learning_rate_at(step: usize, peak: f32, warmup: usize, total: usize, floor: f32) -> f32 {
    if step < warmup {
        return peak * (step + 1) as f32 / warmup as f32;
    }
    if total <= warmup {
        return peak;
    }
    let progress = ((step - warmup) as f32 / (total - warmup) as f32).clamp(0.0, 1.0);
    let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
    peak * (floor + (1.0 - floor) * cosine)
}

#[cfg(test)]
mod tests {
    use super::learning_rate_at;

    #[test]
    fn the_schedule_warms_up_then_decays_to_the_floor() {
        let (peak, warmup, total, floor) = (4e-4, 100, 1100, 0.1);
        assert!(learning_rate_at(0, peak, warmup, total, floor) < peak / 50.0);
        assert!((learning_rate_at(99, peak, warmup, total, floor) - peak).abs() < 1e-9);
        let end = learning_rate_at(1100, peak, warmup, total, floor);
        assert!((end - peak * floor).abs() < 1e-7, "decayed to {end}, wanted {}", peak * floor);
        // Monotone after warmup.
        for step in warmup..total {
            assert!(
                learning_rate_at(step, peak, warmup, total, floor)
                    >= learning_rate_at(step + 1, peak, warmup, total, floor) - 1e-9
            );
        }
    }
}
```

- [x] **Step 3: Run the test to verify it fails, then passes**

```bash
cd ~/Rusting/RustingLLM
cargo test --bin train the_schedule_warms_up
```

Expected: FAIL first (function missing), PASS after Step 2 is in place.

- [x] **Step 4: Apply the schedule each step**

Immediately before `model.step(1.0 / parts as f32);`:

```rust
            let rate = learning_rate_at(
                step,
                args.learning_rate,
                args.warmup_steps,
                if args.total_steps > 0 { args.total_steps } else { planned_steps },
                args.min_lr_ratio,
            );
            model.optimizer = Optimizer::adam_with_weight_decay(rate, 0.01);
```

and compute `planned_steps` once, before the epoch loop:

```rust
    let steps_per_epoch = sequence_count.div_ceil(args.batch_size * args.accumulate);
    let planned_steps = steps_per_epoch * args.epochs;
    println!("{steps_per_epoch} steps per epoch, {planned_steps} planned");
```

Add `rate` to the reporting line so the schedule is visible in the log:

```rust
                println!(
                    "epoch {epoch} step {step}/{planned_steps}: lm_loss={avg_loss:.4} \
                     ppl={:.2} aux={avg_aux:.4} lr={rate:.2e} tok/s={tok_per_sec:.0}",
                    avg_loss.exp()
                );
```

- [x] **Step 5: Verify on a short run**

```bash
cargo run --release --features cuda --bin train -- --gpu --mixed-precision \
  --seq-len 1024 --batch-size 8 --accumulate 8 --epochs 1 --warmup-steps 50 \
  --checkpoint models/sched_check.rbw 2>&1 | head -12
```

Expected: `lr=` climbs from near zero to 4.00e-4 over the first 50 steps and
then falls. The loss should drop faster over the first few hundred steps than
the constant-rate run did.

- [x] **Step 6: Commit**

```bash
git add src/bin/train.rs
git commit -m "feat: warm up and cosine-decay the learning rate"
```

**Expected gain:** none in throughput. A meaningful gain in final loss for the
same number of tokens, which is the only reason to spend the tokens.

---

## Task 6: BF16 activation storage — CLOSED, see docs/baseline.md

**Do not start this task until Task 0 says MFU is below about 25%.** If the
clean baseline shows the device already well fed, this is a large change for
a small return.

Every GEMM already runs its multiplies on the tensor cores
(`gpu_transformer.rs:678`, `CUBLAS_COMPUTE_32F_FAST_16BF`), but every buffer
they read and write is FP32 (`gpu_model.rs:232`, `fn uninit`). On a 360 GB/s
card the traffic, not the arithmetic, is often the limit. The cached
activations per block are listed in `BlockCache` at the end of `forward_block`
(`gpu_model.rs:1330-1345`): `input`, `attention_normed`, `qkv`, `merged`,
`residual`, `feed_forward_normed` are all `rows x d_model`-ish FP32 buffers.

The machinery is already there: `cast_to_bf16` (`gpu_model.rs:244`),
`cast_from_bf16` (`gpu_model.rs:264`), `accumulate_bf16` (`gpu_model.rs:284`)
and `uninit_bf16` (`gpu_model.rs:238`) exist for the LM head.

**Files:**
- Modify: `RustingBrain/src/gpu_model.rs` — `BlockCache`, `forward_block`,
  `backward_block`, `backward_attention`

- [ ] **Step 1: Measure before changing anything**

```bash
cd ~/Rusting/RustingBrain
target/release/examples/sweep_arch 16384 512 8 16 512 0
# and peak VRAM, per Task 0 Step 3
```

Record both numbers in `docs/baseline.md` under a "before bf16 activations"
heading.

- [ ] **Step 2: Convert one buffer and prove parity**

Start with `merged`, the attention output. It is written by one GEMM and read
by one GEMM, so it has the smallest blast radius of the six.

The existing parity test is the gate:

```bash
cargo test --lib gpu_backward_matches_cpu_or_skips_without_device
cargo test --lib a_batched_gpu_train_step_matches_the_host_or_skips_without_device
cargo test --lib repeated_gpu_batched_steps_stay_in_step_with_the_host_or_skip_without_device
```

Expected: all pass. Their tolerance is `1e-3` relative, and BF16's `2^-8`
storage error is about `4e-3`, so **these tests will need their tolerance
widened for the buffers you convert**. Widen it deliberately, in the same
commit, with a comment naming the new error budget — do not widen it silently
to make a failure go away.

- [ ] **Step 3: Re-measure after each buffer**

Convert one buffer, run the three parity tests, run `sweep_arch`, record.
Stop converting when a buffer costs more accuracy than it returns in
throughput. Six buffers, six measurements.

- [ ] **Step 4: Run the loss-curve check end to end**

```bash
cargo run --release --features cuda --example bf16_loss_check
```

Expected: the BF16 and FP32 loss curves still track to the precision that
example asserts. If they diverge, the last buffer you converted is the one
that mattered; revert it.

- [ ] **Step 5: Commit**

```bash
git add src/gpu_model.rs docs/baseline.md
git commit -m "perf: hold block activations in bf16"
```

**Expected gain:** roughly half the activation memory and half the activation
bandwidth. If the baseline shows a bandwidth-bound step, 1.3-1.6x. If it
shows a compute-bound step, close to nothing. Task 0 tells you which.

---

## Task 7: One batched GEMM across all heads — CLOSED, see docs/baseline.md

`forward_block` runs `for head in 0..heads` around
`gemm_rhs_transposed_batched` (`gpu_model.rs:1230`) and again around
`gemm_plain_batched` (`gpu_model.rs:1257`). At 8 heads and 8 layers that is
128 cuBLAS calls per forward pass, each with a batch count of only
`sequences`. The backward pass has the same shape.

They cannot be merged with a strided-batched call, because the stride between
heads (`head_dim` within a row of width `qkv_width`) differs from the stride
between sequences (`seq_len * qkv_width`), and cuBLAS strided-batched takes
one stride. The pointer-array form, `cublasGemmBatchedEx`, takes an explicit
device array of pointers and handles both — and it also handles the
grouped-query `kv_base = (head / group) * head_dim` mapping for free.

**Do this only if Task 0 shows low MFU and Task 6 did not fix it.** It is the
most invasive change in this plan.

**Files:**
- Modify: `RustingBrain/src/gpu_transformer.rs` — add a pointer-array batched
  GEMM beside `gemm_strided_batched_dispatch` at line 764
- Modify: `RustingBrain/src/gpu_model.rs:1226-1278` and the matching backward

**Interfaces:**
- Produces: `gemm_batched_pointers(context, a_ptrs, b_ptrs, c_ptrs, m, n, k, alpha, beta, count)`
  where the three pointer arrays are device-resident `CudaSlice<u64>` built
  once per block shape and reused across steps.

- [ ] **Step 1: Measure the launch overhead you are trying to remove**

```bash
cd ~/Rusting/RustingBrain
cargo run --release --features cuda --example profile_step
```

If per-launch overhead is under ~3% of step time, **stop — close this task as
not worth doing** and record why in `docs/baseline.md`. The occupancy gain
from a batch count of `heads * sequences` instead of `sequences` may still
justify it, but the plan should say which of the two it is buying.

- [ ] **Step 2: Build the pointer arrays once per shape**

Cache them on `GpuContext` keyed by `(rows, heads, seq_len, head_dim)` so a
steady-state training loop builds them on the first step and never again.
Rebuilding a pointer array per step trades a cheap kernel launch for an
expensive host-to-device copy and loses.

- [ ] **Step 3: Gate on the parity tests**

```bash
cargo test --lib gpu_forward_matches_cpu_or_skips_without_device
cargo test --lib gpu_backward_matches_cpu_or_skips_without_device
cargo test --lib a_padded_gpu_batch_matches_the_host_or_skips_without_device
cargo test --lib a_cached_decode_on_the_device_matches_the_host_or_skips_without_device
```

Expected: all pass at their existing `1e-3` tolerance. A pointer-array GEMM
computes exactly what the strided one did, so unlike Task 6 this must not
need any tolerance change. If it does, the pointers are wrong.

- [ ] **Step 4: Re-measure and commit**

```bash
target/release/examples/sweep_arch 16384 512 8 16 512 0
git add src/gpu_transformer.rs src/gpu_model.rs docs/baseline.md
git commit -m "perf: run attention as one batched GEMM across heads"
```

**Expected gain:** 128 launches per forward pass down to 16, and a batch count
8x larger per call. Somewhere between nothing and 15%, and Step 1 is what
tells you which before you spend the day.

---

## Task 8: Data and the run itself — NOT STARTED, needs your decisions

Throughput work is only worth doing if there are tokens to spend it on. Your
corpus is 1.2 GB of cloned Rust repositories — 368M tokens. A 50B-token target
needs about 136x more data than you have, and 100 GB of disk to hold it as
`u16` against 116 GB free.

**Sources, in the order worth taking them:**

| Dataset | Tokens | Note |
|---|---|---|
| `bigcode/the-stack-v2-dedup` | 600B+ | The main one. Permissive licences, 600+ languages. Gated — accept the terms on Hugging Face first |
| `OpenCoder-LLM/opc-fineweb-code-corpus` | ~150B | Quality-filtered code and code-adjacent web text |
| `HuggingFaceTB/smollm-corpus`, `python-edu` split | ~4B | The highest quality-per-token in this list |
| `HuggingFaceTB/fineweb-edu` | 1.3T | Mix in 20-30%. Code-only models reason badly and follow instructions worse |
| `EleutherAI/proof-pile-2` | ~55B | Mathematics. Helps code more than it looks like it should |

A mix of roughly 70% code, 20% `fineweb-edu`, 10% mathematics is the
conventional starting point and there is no reason to deviate on a first run.

- [ ] **Step 1: Decide the token budget against the disk you have**

116 GB free, 2 bytes per token, and you want room for checkpoints:

```
25B tokens = 50 GB    <- fits with room to spare
50B tokens = 100 GB   <- does not fit alongside checkpoints
```

Take 25B unless you free space. `/mnt/ollama` is on the same filesystem, so
the Ollama models stored there are the obvious thing to prune if you want the
full 50B.

- [ ] **Step 2: Stream, tokenize, and delete as you go**

Never land the raw text. For each shard: download, tokenize with
`tokenize_corpus`, append to the `.u16` file, delete the shard. A shell loop
around `huggingface-cli download` with `--include` per shard is enough; this
does not need a program.

- [ ] **Step 3: Retrain the tokenizer on the new mix**

`tokenizer.json` was trained by `train_tokenizer.rs` on Rust source alone. A
mixed corpus needs a tokenizer trained on that mixture, or the English and
mathematics fall apart into single characters. Vocabulary 16384 rather than
32000: it halves the embedding from 16.4M to 8.4M parameters, and the measured
throughput gain was 10%.

Changing the vocabulary invalidates every existing checkpoint. Do it before
the long run starts, not during.

- [ ] **Step 4: Prove the recipe on 2B tokens before spending a month**

```bash
cargo run --release --features cuda --bin train -- \
  --gpu --mixed-precision --tokens data/mix.u16 \
  --seq-len 1024 --batch-size 8 --accumulate 16 \
  --learning-rate 4e-4 --warmup-steps 2000 --epochs 1 \
  --checkpoint models/pilot.rbw --checkpoint-every 1000
```

Expect roughly a day and a half. What you are checking:

- The loss curve is smooth and still falling at the end. A plateau means the
  learning rate is wrong; a spike means warmup is too short.
- `generate` produces syntactically plausible code, not repeated tokens.
- `tok/s` in the log matches what `sweep_arch` predicted. A large gap means
  the data loader is now the bottleneck, not the GPU.

**Do not start the long run until all three hold.** A bug found on day twelve
of a thirty-day run costs twelve days, and this is the single most expensive
mistake available in this plan.

- [ ] **Step 5: Start the long run under a supervisor**

```bash
systemd-run --user --unit=rusting-train --working-directory=$HOME/Rusting/RustingLLM \
  ./target/release/train --gpu --mixed-precision --tokens data/mix.u16 \
  --seq-len 1024 --batch-size 8 --accumulate 16 --learning-rate 4e-4 \
  --warmup-steps 2000 --epochs 1 --checkpoint models/run3.rbw --checkpoint-every 1000
journalctl --user -u rusting-train -f
```

A month-long foreground process in a terminal will not survive a logout, an
X restart, or a stray Ctrl-C.

---

## What This Added Up To

Implemented and measured, all on an idle RTX 3060:

| Change | Before | After |
|---|---|---|
| Idle GPU instead of a shared one | 21295 tok/s | 43704 tok/s |
| Checkpoint size (55.2M model) | 668 MB JSON | 221 MB binary |
| Checkpoint save / load | 1.23s / 1.58s | 0.05s / 0.21s |
| Trainer anonymous RSS | 2.78 GB | 849 MB |
| Startup to first training step | ~3.5 min | 36 s |
| Dense feed-forward instead of MoE | 40115 tok/s | 43704 tok/s |
| Corpus ceiling | RAM-bound | disk-bound |

Gradient accumulation reproduces the un-accumulated loss curve exactly for a
dense model: batch 64 x accumulate 1 and batch 16 x accumulate 4 print the
same `lm_loss` at every reported step, at 105778 and 94118 tok/s. The cost of
splitting is 11% throughput; what it buys is an effective batch that no longer
has to fit in 12 GB.

Not implemented, deliberately: Tasks 6 and 7. The measurement they were gated
on came back at 30.7-37.0% MFU across every shape in the sweep, including the
128-sequence batch that issues the fewest launches per token. A device that
well fed does not have 1.3x sitting in its activation traffic.

Still open: Task 8. It needs decisions only you can make -- how much of the
115 GB free disk to spend, whether to retrain the tokenizer at vocabulary
16384 (which invalidates every existing checkpoint), and whether the target is
a model trained from scratch or a usable coding assistant. Those are not the
same project: 55M parameters on 25B tokens lands near SmolLM2-135M, which
writes plausible short functions and is not an assistant. Fine-tuning
Qwen3-0.6B on the same corpus is a few hours on this card.
