---
title: Layer 3 — Model Compilation
---

# Layer 3 — Model compilation (`hematite-codegen` + `hematite-memory`)

The developer experience. `#[model("path.tflite")]` turns a TFLite
flatbuffer into **typed, straight-line Rust** at compile time. There is
no inference interpreter, no op-dispatch loop, no runtime flatbuffers
dependency.

## `#[model]` — the entry point

```rust
use hematite_codegen::model;

#[model("models/zoo/keyword_spotting/kws_micro_speech_int8.tflite")]
pub struct KeywordSpotting;
```

At **compile time** the proc-macro:

1. **Parses** the flatbuffer with a hand-rolled byte-offset walker
   (`flatbuffer.rs`) — deliberately no `flatbuffers` runtime dep.
2. **Validates** the graph (supported op set, shapes).
3. **Plans memory** — a USMP-style liveness arena over the op schedule
   (`hematite-memory`), producing `ARENA_LEN`.
4. **Fuses** — a selector finds adjacent ops that compose into one
   `FusedKernelBackend` call (conv + requant + activation + residual-add;
   elementwise chains; pool input-folds), with a host-side SIMD
   eligibility mirror guaranteeing the fused call is SIMD-eligible on s3.
5. **Emits** straight-line Rust: `INPUT_LEN`, `OUTPUT_LEN`,
   `SCRATCH_LEN` (macro-time max per-op kernel scratch), `ARENA_LEN`,
   and the `Model<B>` wrapper with `new`/`input_len`/`output_len`/
   `predict`/`predict_with_scratch`.

The emitted code is **generic over `B: FusedKernelBackend`** — the same
bytes run on `RefBackend` (host/scalar) and `S3Backend` (device/SIMD).

## What the macro emits

See [Quickstart](../quickstart.md) for the full generated API. In
summary:

```rust
pub const INPUT_LEN: usize;
pub const OUTPUT_LEN: usize;
pub const SCRATCH_LEN: usize;   // max per-op kernel scratch (macro-time)
pub const ARENA_LEN: usize;     // liveness-planned intermediates peak

impl<B: FusedKernelBackend> Model<B> {
    pub const fn new(backend: B) -> Self;
    pub fn predict(&mut self, input: &[i8; INPUT_LEN]) -> [i8; OUTPUT_LEN];
    pub fn predict_with_scratch(&mut self, input, output, scratch) -> Result<(), KernelError>;
}
```

## Test-support variants

The macro exposes four additional attribute forms — all test arms of the
correctness gates, not user-facing:

| Attribute | Purpose |
|---|---|
| `#[model(..)]` | fused schedule + arena + staging (production) |
| `model_unfused` | unfused per-op sequence — the fused==unfused equivalence arm |
| `model_stack` | arena disabled — per-tensor stack, the arena-vs-stack arm |
| `model_unstaged` | graph-input 16B staging disabled — the staged-vs-unstaged arm |
| `model_force_t2` | T2 group gate forced open — the auto-unfuse-path prover |

## `hematite-memory`: the arena planner

- **Zero runtime allocation.** The planner runs at macro time; the
  emitted code is stack arrays + one arena local.
- `liveness_plan` → `ArenaPlan` (tensor offsets, peak) over the op
  schedule. `ARENA_LEN` is the peak.
- Scratch (`SCRATCH_LEN`) is separate from the arena: arena = tensor
  intermediates; scratch = per-op kernel working memory (padded copies
  for SIMD).
- Models whose peak exceeds the planner budget (mobilenet_v2's 224×224×32
  activation ≈ 1.6 MiB) fall back to **per-tensor stack emission** —
  bit-exact, just larger; see [memory model](memory-model.md).

## The fusion selector

The composed-kernel workstream (phase 19) is an *additive* trait
mechanism: `FusedKernelBackend` sits beside `KernelBackend`, and each
fused method's decomposition is documented — conv + residual + activation
as exactly the three per-op calls, etc. The selector:

- finds groups that can compose (structural rules),
- mirrors the s3 SIMD gates on the host (`eligibility.rs`) so composed
  groups are SIMD-eligible,
- emits composed calls when the gates pass, otherwise per-op,
- is pinned by tests (fused==unfused bit-exact over all zoo models).

## Correctness gates owned by this layer

- Fused vs unfused equivalence (all 6 zoo models, host)
- Arena vs stack bit-exactness
- Staged vs unstaged bit-exactness
- Scratch-parity: codegen `SCRATCH_LEN` mirrors s3 `*_scratch_need`

Next: [Layer 4 — Validation](layer-4-validation.md).