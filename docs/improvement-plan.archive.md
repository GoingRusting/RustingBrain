# Performance Improvement Plan

Ranked list of optimizations for training and inference throughput, with what
has already landed and what is left.

All measurements taken on the development machine: NVIDIA RTX 3060 (Ampere,
sm_86), AMD Ryzen 5 7600X (6C/12T, AVX-512), `--release` with LTO.

Benchmarks used:

```sh
# CPU / CUDA training, phase breakdown
RB_PHASES=1 cargo run --release --example profile_step -- dense 16 128 cpu
cargo run --release --features cuda --example profile_step -- moe 32 128 cuda

# Reduced precision is the default; `nomp` is the opt-out
cargo run --release --features cuda --example profile_step -- moe 32 128 cuda nomp

# Cached single-token decode
cargo run --release --example bench_decode -- 1024 128
```

---

## Landed

| Benchmark | Before | After |
| --- | --- | --- |
| CPU training, dense 5.2M, batch 16 × seq 128 | 1889 tok/s | **6000 tok/s** (3.2×) |
| CPU training, dense 5.2M, batch 16 × seq 512 | 2737 tok/s | **5390 tok/s** (2.0×) |
| CPU decode, 1024-token history | 202 tok/s | **354 tok/s** (1.75×) |
| CUDA training, dense 5.2M, batch 128 × seq 128 | 61439 tok/s | **114000 tok/s** (1.9×) |
| CUDA training, MoE 55M/36M, batch 32 × seq 128 | 14826 tok/s | **19000 tok/s** (1.3×) |

Test suite green: 105 passing without CUDA, 129 with `--features cuda`.

1. **Threaded, AVX-512 GEMM.** `Cargo.toml`: `matrixmultiply` 0.3.10 →
   `{ version = "0.3.11", features = ["threading", "avx512"] }`.
   CPU training previously used one core out of twelve — every `sgemm` call was
   single-threaded. Largest single win: 2239 → 3563 tok/s on its own. Thread
   count defaults to all cores; `MATMUL_NUM_THREADS` overrides it.

2. **Native target features.** New `.cargo/config.toml` sets
   `-C target-cpu=native`, enabling Zen 4 AVX-512 for the hand-written
   element-wise loops (SwiGLU, RMSNorm, Adam, softmax). Worth ~9%.

3. **Parallel language-modelling loss.** `src/causal_lm_loss.rs`: the 32k-wide
   softmax per row ran serially and had grown to 30% of a CPU step. Now
   `par_chunks_mut` over the gradient rows, with an ordered `collect` rather
   than a parallel reduction so the summed loss stays independent of how rayon
   splits the work — matching the convention already used in `network.rs`.
   176 ms → 26 ms.

4. **In-place key/value cache reads.** `src/attention.rs`: `forward_cached`
   called `cache.keys().to_vec()` and `Matrix::from_vec` per layer per decoded
   token, copying roughly 6 MB per token at a 1024-token history. The scoring
   helpers now take `&[f32]` plus a stride instead of a `&Matrix`, so the cache
   is read where it lives. The CUDA branch still materializes the two matrices,
   since `gpu_transformer::attention_heads` needs them.

5. **GEMV fast path.** `src/matrix.rs`: `dot_rhs_transposed` takes a direct
   dot-product path when `rows == 1`. Blocked `sgemm` with `m == 1` spends all
   its time packing panels for a kernel that never sees a second row. Affects
   decode only; 1.75× there.

6. **Decode benchmark.** New `examples/bench_decode.rs`, so inference
   throughput is measurable rather than guessed at.

7. **Mixed precision on by default** (G1). `TransformerBuilder::new` now sets
   `mixed_precision: true`. The flag was already implemented and already
   documented; it was simply off. Worth 1.8× on the dense CUDA preset.

   The four FP32 device-against-host parity tests in `gpu_transformer.rs` pin
   their fixture back to `mixed_precision(false)`. They compare parameters after
   an Adam step at `lr = 1e-2`, and TF32/BF16 rounding on a parameter whose true
   gradient is near zero moves it by a full learning-rate step — which is what
   the older `mixed_precision_tracks_the_fp32_device_path_or_skips_without_device`
   test already documents and tolerates.

8. **Attention as strided GEMMs** (C1). `src/attention.rs`:
   `head_scores_batched`, `accumulate_head_output_batched` and
   `MultiHeadAttention::backward` were scalar triple loops. Each is now one
   `matrixmultiply::sgemm` per sequence, with a head addressed as a strided view
   of the packed `[rows, heads * head_dim]` buffer — no copy, and the threaded
   AVX-512 kernel for free. The backward pass reuses a single `seq_len * seq_len`
   scratch and keeps `beta = 1.0` on the query, key and value gradients, which is
   what grouped-query attention needs since several heads write the same
   key/value rows.

   A/B with the rest of the tree held constant: seq 128, 4936 → 5796 tok/s
   (+17%); seq 512, 2737 → 4601 tok/s (+68%). The gain grows with sequence
   length, as it should for the O(T²) part of the step.

9. **Parallel element-wise passes.** `rayon` over the point-wise work that was
   left serial once GEMM was threaded: `RmsNorm::forward`/`backward`,
   `SwiGlu`'s three passes and `GeluMlp`'s two, `Rope::rotate`, and
   `transformer_block::add_in_place`. `RmsNorm::backward` sums the scale
   gradient over fixed 32-row tiles and adds the partials in order, so the
   result does not depend on how rayon split the work — the convention
   `network.rs` and `causal_lm_loss.rs` already use.

   seq 128, 5796 → ~6000 tok/s; seq 512, 4601 → 5390 tok/s.

10. **BF16 compute for the FP32 GEMMs** (G2). `src/gpu_transformer.rs` gained
    `gemm_dispatch`, which every non-batched GEMM helper now routes through.
    With `mixed_precision` set and FP32 operands it calls `cublasGemmEx` with
    `CUBLAS_COMPUTE_32F_FAST_16BF` instead of `cublasSgemm`; A, B and C stay
    FP32 in memory and cuBLAS still accumulates in FP32, and the only change is
    that the multiplier inputs are rounded to BF16 inside the tensor cores.
    Ampere runs BF16 tensor operations at twice the TF32 rate.

    No BF16 weight mirrors, no cast kernels, no extra buffers — which is why
    this is a fifty-line change rather than the per-call-site rework the
    original G2 entry assumed.

    MoE preset, batch 32 × seq 128: 14826 → 19000 tok/s (+28%). Dense preset:
    unchanged, because its GEMM time is already dominated by the BF16
    language-model head.

11. **One exponential per element in the cross-entropy softmax.**
    `src/cuda_training.rs`, both `ce_loss_grad` and `ce_loss_grad_bf16`.

    The online softmax scan computed `acc*expf(m-nm)+expf(v-nm)` for every
    element, so every one of the 131 million logits in a step paid for two
    exponentials even though the running maximum changes only a handful of
    times per row. Rescaling the accumulator only on a new maximum leaves the
    common element with one exponential, and `__expf` replaces `expf` in the
    scan and in the output pass — ample precision for a result stored as
    bfloat16.

    `ce_loss_grad_bf16` went from 7.39 ms to 2.54 ms per call, which is 2.9×.
    At the 786 MB the kernel moves that is 310 GB/s of the card's 360 GB/s, so
    it is now bandwidth bound and the vectorized loads the original entry
    suggested are no longer worth writing. End-to-end this is 3.2 ms off a
    216 ms step, which the GPU contention on this machine hides entirely.

12. **Fused attention forward, probabilities rebuilt in the backward pass**
    (G3). `src/cuda_training.rs` gains `flash_attention_fwd` and
    `causal_probs_from_lse`; `src/gpu_model.rs` replaces `BlockCache`'s
    probability matrix with a per-query log-sum-exp.

    The forward pass is now one kernel. Each warp owns one query row and walks
    the keys in tiles of 32 held in shared memory, keeping a running maximum
    and sum, so the score matrix is never written to global memory. The tiles
    are staged transposed with a stride of 33 floats, which both removes bank
    conflicts and lets one lane own one key, so the whole tile costs a single
    cross-lane reduction instead of one per score.

    The backward pass rebuilds what it needs: one batched GEMM per head for the
    scores, then one exponential per element against the stored log-sum-exp,
    and from there the four gradient GEMMs are the ones that were already
    there. `causal_softmax` and its forward kernel are gone.

    `BlockCache` now holds `heads * rows` floats where it held
    `heads * rows * seq_len`. For the MoE preset at batch 32 × seq 128 that is
    131 KB per layer instead of 134 MB, and the peak is one layer's rebuilt
    matrix in the backward pass rather than every layer's at once.

    Throughput is unchanged: 18747 tok/s against the 19000 tok/s of item 10,
    which is inside the run-to-run spread on a contended card. Total kernel
    time per step fell from 84 ms to 75 ms, and the fused forward is 4.6 ms of
    it.


13. **A decode GEMV that the compiler can actually vectorize, and a parallel
    unembedding.** `src/matrix.rs`: a new `dot` helper plus a rayon split in
    `dot_rhs_transposed`; `src/attention.rs` uses the same helper for the
    cached attention scores.

    The single-row fast path summed each dot product into one accumulator.
    Floating-point addition is not associative, so the compiler was not allowed
    to reorder it: every multiply-add waited on the four-cycle latency of the
    previous one, and the loop ran at roughly a quarter of a multiply-add per
    cycle where AVX can issue sixteen. Eight independent accumulators give it a
    chain to pipeline. The same one-accumulator loop scored cached attention,
    so it got the same helper.

    Separately, one decoded token against a 32000-row unembedding streams 16 MB
    of weights, which is several times what one core can pull from memory in
    the time the rest of the token takes. That one GEMV now runs on the rayon
    pool; the projections inside a block are below the threshold and stay on
    the calling thread, because the fork costs more than the work.

    Measured by varying one dimension at a time: the unembedding was 85% of a
    decode step before the split and is now at the memory roofline, and the
    per-layer cost fell by about half again on top of that.

    `bench_decode 256 128`: 2.46 → 0.46 ms/token (5.3×).
    `bench_decode 1024 128`: 3.30 → 0.91 ms/token (3.6×).

14. **One GEMM for the gate and up projections.** The two SwiGLU projections
    read the same input and have the same shape, so `Gpu::pack` copies their
    weight matrices end to end and one GEMM at twice the width replaces both.
    The backward pass does the same: one weight-gradient GEMM at twice the
    width, split back into the two parameters by two offset adds over a
    weight-sized buffer. It covers the dense feed-forward, the MoE shared
    expert and all eight routed experts, and it changes neither the host model
    structure nor the checkpoint format.

    cuBLAS is close to twice as fast on the wider shape, because a weight
    gradient whose output is only a few tiles wide cannot fill the device.

    `profile_step moe 128 128 cuda`: 48574 → 49932 tok/s.

15. **One GEMM for the query, key and value projections.** The same trick on
    attention. The three weights pack into one `[n_heads * head_dim + 2 *
    n_kv_heads * head_dim, d_model]` matrix, one GEMM produces all three, and
    every later reader takes a slice of the fused row instead of a buffer of
    its own. RoPE gained a row stride and a start offset so it can rotate the
    query and key slices in place, and the backward pass is one input-gradient
    GEMM and one weight-gradient GEMM split three ways.

    `profile_step moe 128 128 cuda`: 49932 → 50415 tok/s.

16. **Vectorized SwiGLU and gradient-split kernels.** `swiglu_fwd`,
    `swiglu_bwd` and `add_inplace` read and write four floats at a time when
    the width allows it, with the scalar form kept for an odd width. The
    forward kernel fell from 10.46 to 6.13 ms per step; the backward kernel was
    already at its roofline and did not move, and neither did `add_inplace`,
    which turns out to be bound by launch count rather than bandwidth.

17. **BF16 compute for the batched attention GEMMs.** `gpu_transformer.rs`
    gained `gemm_strided_batched_dispatch`, the batched twin of
    `gemm_dispatch`: it routes the three strided-batched helpers through
    `cublasGemmStridedBatchedEx` with `CUBLAS_COMPUTE_32F_FAST_16BF` when
    mixed precision is on. Those were the last GEMMs in the model still running
    TF32 — 27.9 ms per step of `s1688gemm`, now 26.9 ms of `s1688bf16gemm`.
    The gain is inside measurement noise, because these shapes turn out to be
    bound by the probability matrix's memory traffic rather than by tensor
    throughput, but it makes the precision policy uniform: every f32 GEMM in
    the model now honours the `mixed_precision` flag the same way.

18. **Two bfloat16 values per access in the cross-entropy kernel.** Both passes
    of `ce_loss_grad_bf16` over a vocabulary row now load and store a pair at a
    time, with a scalar tail for an odd vocabulary. A warp of single bfloat16
    loads asks for only 64 bytes; a pair doubles that. 10.17 to 9.76 ms per
    step over the largest tensor a training step touches.

---

## GPU work remaining

### G1, G2 — landed

See items 7 and 10 above.

### G3 — landed, but not as a full flash attention

See item 12 above. The forward pass is fused; the backward pass is not, and the
measurement that decided it is worth recording.

A full flash backward was written and thrown away. It recomputed the scores
inside two hand-written kernels, one for the query gradient and one for the key
and value gradients, and it was correct — but it cost 36 ms per step against
roughly 18 ms for the cuBLAS path it replaced. The reason is arithmetic
intensity, not a fixable bug: the inner loops read four floats of shared memory
for every two fused multiply-adds, which caps the kernel at about an eighth of
the card's FP32 rate, and the GEMMs it replaced run on BF16 tensor cores at
four times that rate again. Closing that gap means register-tiling several
queries and keys per thread and then moving to `mma.sync` — that is a tensor
core GEMM written by hand, and cuBLAS already ships one.

So the backward pass keeps the GEMMs and pays one extra score GEMM plus one
exponential per element to rebuild what the forward pass no longer stores. The
memory win survives in full; the speed is a wash.

### G4. Per-head loop of batched GEMMs

`forward_block` runs `for head in 0..heads` twice, issuing one
`gemmStridedBatched` per head with small `m`/`n`. Launch overhead and poor
occupancy.

- Plain multi-head attention (`group == 1`): collapse into a single call with
  `batch = heads * sequences`.
- Grouped-query attention: use the pointer-array batched form so the shared
  key/value heads can be addressed per batch entry.

**Effort:** small-medium. **Payoff:** fewer launches, better SM occupancy;
measure before committing.

### G5, G6 — measured, not worth doing

**G5 (device allocator).** `cuMemAllocAsync` costs 7.4 ms across 910 calls per
profiled run, because cudarc already allocates from a CUDA memory pool rather
than calling `cuMemAlloc` per buffer. An arena would remove an overhead that is
under 1% of a step.

It would still be worth doing for a different reason: the MoE preset runs out of
device memory at batch 128, and a pool with constant shapes would not fragment.
That is the same ceiling G3 addresses, and G3 addresses it by making the
allocation unnecessary rather than cheaper, so G3 comes first.

**G6 (resident norm weights).** The two uploads per block are 512 bytes each,
twelve per step on the default preset: about 0.08% of a step. The transfer was
never on the critical path.

### G7. Single stream, no overlap

Token-id and target uploads are serialized against compute on one stream. Low
priority at current model sizes, but it is the obvious next step once G5 lands
and allocation stops dominating.

---

## CPU work remaining

### C1 — landed

See items 8 and 9 above.

### C2, C4 — measured, not worth doing

**C2 (`Arc<Matrix>` in the caches).** The clones are about 30 MB per step on the
dense preset, 0.4% of the step. `Arc` would have to be threaded through
`transformer_block` into `attention`, `ffn` and `moe`, changing five signatures
for an unmeasurable gain.

**C4 (redundant `zero_grad`).** 0.2% of a step. The `step(scale)` callers that
accumulate gradients over several backward passes are the reason the extra reset
exists; removing it to save a fifth of a percent is not a trade worth making.

### C3. MoE allocates per expert per layer per step

`MoeLayer::forward_train` builds a `Matrix` per expert through `gather_rows`;
`MoeLayer::backward` builds a second one (`scaled`) per expert. Both are
reallocated every step with constant shapes for a fixed routing load.

Reuse scratch buffers held on the layer.

Note: rayon over experts nested inside threaded GEMM was checked for
oversubscription and is **not** a problem — twelve matmul threads still beat one
inside `par_iter` (486 → 887 tok/s on the MoE preset). No action needed there.

**Effort:** small-medium. **Payoff:** allocator pressure only; measure first.

### C5 — landed

See item 13 above. The profile the entry asked for was a dimension sweep rather
than a sampling profiler, since `perf` is not installed and `ptrace` is
restricted on this machine: hold everything fixed and double the vocabulary,
the layer count, `d_ff` and `d_model` in turn, and read the cost off the
differences. It put 85% of a decode step in the unembedding GEMV and the rest
in the per-layer projections, and both turned out to be the same
one-accumulator dot product.

What is left is genuinely memory bound: the unembedding moves 16 MB per token
at roughly 56 GB/s of the machine's ~60 GB/s. Cutting it further means not
reading all 32000 rows — a smaller head, or sampling against a candidate set.

### C6. No batched decode

`TransformerLm::forward_cached` handles one sequence. Batching N sequences
turns every GEMV back into a GEMM and raises inference throughput close to
linearly until it becomes compute-bound.

**Effort:** medium-large — `KvCache` needs a batch dimension.
**Payoff:** large for any serving workload.

### C7. JSON checkpoints

`save_json` / `load_json` push 55M parameters through `serde_json`. A raw f32
dump or `bincode` is roughly 10× faster and 3× smaller. Does not affect step
time, but it does affect any loop that checkpoints often.

**Effort:** small. **Payoff:** checkpoint latency only.

---

## Where the default MoE preset spends its time now

Retaken on an idle card, with `nsys profile -t cuda` on
`profile_step moe 128 128 cuda`, which is the shape a real training run uses
(batch 128, sequence 128, 16384 rows per step). The earlier table in this
section was taken while a separate `train --gpu` job held the card, so every
absolute number in it was low and its 19% idle figure was that other process's
time slices. Those numbers are gone; what follows replaces them.

The step is 0.325 s. The GPU is busy for 98.6% of it, so there is no host-side
win left to take: everything below has to come out of kernel time.

| Kernel group | ms/step | Share |
| --- | --- | --- |
| Language-model head GEMMs | 68 | 21% |
| Transformer-block GEMMs (projections, feed-forwards, experts) | 117 | 36% |
| Attention batched GEMMs | 28 | 9% |
| Attention softmax kernels | 11 | 3% |
| SwiGLU forward and backward | 15 | 5% |
| RMSNorm forward and backward | 12 | 4% |
| Everything else elementwise | 49 | 15% |
| Device-to-device copies and zeroing | 8 | 2% |

The step does 3630 GFLOP of GEMM work. Six times the active parameter count
times the token count is 3510 GFLOP, so the arithmetic is already the minimum
the architecture asks for — there is no redundant work to delete, only work to
run faster.

Two of those rows are already finished. The language-model head is 1610 GFLOP,
44% of the whole step, and it runs at 23.6 TFLOP/s, which is the card's
measured BF16 peak. Elementwise kernels sit at 70-91% of the 360 GB/s memory
roofline. Neither has anything left in it.

The one row that is far from its ceiling is attention: 39 ms for about 103
GFLOP, or 3.8 TFLOP/s. Two things cause it. The score GEMMs have K = 64 and
N = 64, which is too little reuse for the tensor cores, and the 128x128
probability matrix for every head and every sequence is written to device
memory, read back by the softmax, written again, and read once more by the
value GEMM — 67 MB per layer in the forward pass alone. The backward pass
rebuilds all of it. Half of every score matrix is then thrown away by the
causal mask, so half of those 103 GFLOP is wasted outright.

### Measured and rejected

- **FP16 compute type** on the slow GEMM shapes: up to 2x on the narrow ones,
  but worth only about 4% of the step, and FP16 accumulation risks silent
  training instability. Not taken.
- **TF32 compute type**: uniformly slower than BF16 on every shape tried.
- **A 16-way sweep of `cublasGemmAlgo_t`**: the default is already the best
  choice on every shape in the model.
- **A head-major attention layout**, which would need one batched GEMM instead
  of one per head: 7% on the isolated shape. An earlier draft of this document
  claimed 1.6x; that measurement was taken without warming the clock. The card
  idles at 210 MHz and ramps to 1.78 GHz, so a short standalone benchmark reads
  up to 12x low. Every GPU measurement here now burns a second of work first.
- **Split-k on the weight-gradient GEMMs**, via `cublasGemmStridedBatchedEx`
  with a per-split output and a reduction. cuBLAS already reaches 19-24
  TFLOP/s on nearly all of these shapes unaided. Only `768x512x16384` (16.9
  against 21.6 with eight splits) and `512x352x4096` (9.8 against 12.5 with
  two) improve at all, and together they are worth under 1% of the step.
- **Batching the eight routed experts into one strided GEMM**, which needs
  every expert padded to the same row count. The batched call is 14-32% faster
  than eight separate ones at these shapes, but padding to the largest expert
  adds roughly as much wasted arithmetic as the batching saves.

- **A hand-written SIMT fused attention forward**, tried in full and removed.
  One block per 64-query tile, head and sequence; queries, keys, values and
  probabilities in dynamic shared memory; the causal mask applied as the scores
  are built so the masked tiles are never visited; an online softmax so a key
  tile can be discarded as soon as it is folded in; and four query rows by four
  key columns per thread so eight shared loads feed sixteen multiply-adds.
  Keys and values share one tile, which buys a second resident block per
  multiprocessor and a quarter of the kernel's time. It passes every parity
  test and lands at 0.325 s per step against 0.324 s for the GEMM path: exactly
  even. The traffic it removes is real, and the arithmetic it adds costs just
  as much. The reason is arithmetic rate, not the kernel: skipping the masked
  half leaves about 25.8 GFLOP per step, but a SIMT kernel runs in FP32 at a
  peak of 12.75 TFLOP/s, and shared-memory bandwidth caps a four-by-four
  register tile at about half of that. The batched GEMMs it replaces run on
  BF16 tensor cores whose peak is 25.5 TFLOP/s. A fused attention only wins
  here if it uses the tensor cores too, which means `mma.sync` fragments
  through inline PTX — the one remaining item that is worth more than a few
  percent, and much the largest piece of work left.
- **Row and column from `blockIdx.y` and `blockIdx.x`** in the MoE gather and
  scatter kernels, instead of dividing a flat thread index by the runtime row
  width. The division is not what those kernels are spending their time on:
  the two-dimensional grid measured 0.325 s per step against 0.321 s for the
  flat index.

## Suggested order

Revised after the measurements above.

1. **Tensor-core fused attention.** The plain SIMT version of this was built,
   measured and removed — see "Measured and rejected" — and the lesson from it
   is that the fusion only pays if the fused kernel keeps the tensor cores.
   That means `mma.sync.m16n8k16` fragments written as inline PTX, since NVRTC
   has no `mma.h`, for both the forward and the backward, with a fallback to
   the present path for shapes the fragment layout does not fit. The prize is
   the 36 ms per step that attention now spends moving a probability matrix it
   does not need to keep: about 12% of the step, of which perhaps half is
   recoverable. It is also by far the largest remaining piece of work.

2. **C6** — batched decode, for any serving workload. `KvCache` needs a batch
   dimension.

3. **G4**, **C3**, **C7**, **G7** — as needed.

Everything else measured above is either already at its roofline or worth less
than 1% of the step.
