---
title: How-To — Scratch & Memory
---

# How to size scratch and use memory correctly

Hematite is zero-allocation: every buffer is either generated-constant,
stack, or caller-provided. This guide covers the two numbers that matter
when you call a model or a kernel directly: **`SCRATCH_LEN`** and
**`ARENA_LEN`** — and what to do when a model doesn't fit.

## The two sizes, in one sentence

- **`ARENA_LEN`** — compile-time-planned peak of the *intermediate
  tensor* arena (the activations between ops). Lives on the stack (or as
  per-tensor locals when the planner rejects the model).
- **`SCRATCH_LEN`** — macro-time *max over ops* of each op's kernel
  scratch (padded SIMD working memory). This is what
  `predict_with_scratch`'s `scratch` argument must satisfy.

See [memory model](../architecture/memory-model.md) for the full picture.

## Pattern 1 — Let the model allocate (`predict`)

```rust
// predict() internally allocates [u8; SCRATCH_LEN] + [i8; ARENA_LEN]
// on the stack — no heap, no caller buffers needed.
let mut model = MyModel::<S3Backend>::new(S3Backend);
let out = model.predict(&input);
```

Simple, but the stack usage is the model's (arena + scratch). For tiny
models this is ideal; for arena-heavy models prefer Pattern 2.

## Pattern 2 — caller-provided scratch (`predict_with_scratch`)

```rust
let mut model = MyModel::<S3Backend>::new(S3Backend);
let mut out = [0i8; MyModel::<S3Backend>::output_len()];
let mut scratch = [0u8; MyModel::<S3Backend>::SCRATCH_LEN];
model.predict_with_scratch(&input, &mut out, &mut scratch)
    .expect("scratch sized correctly");
```

- The method returns `Err(KernelError::ScratchTooSmall)` if
  `scratch.len() < SCRATCH_LEN` — a hard, honest error, not a silent
  fallback.
- The intermediate arena stays internal (`[i8; ARENA_LEN]` stack
  local): reusing caller scratch for tensors would need a byte-type
  cast the generated code must never contain. `scratch` covers only
  per-op kernel working memory.

## Pattern 3 — direct kernel calls (no model)

If you build the call chain yourself, size scratch via the backend's
associated functions and call kernels directly:

```rust
use hematite_core::{KernelBackend, KernelError};

let scratch_need = S3Backend::conv2d_scratch_size(&conv_params);
let mut scratch = [0u8; 4096]; // sized >= scratch_need for this op

S3Backend.conv2d(&input, &weights, &bias, &conv_params, &mut output, &mut scratch)?;
```

`RefBackend`'s scratch sizes default to `0`; `S3Backend` overrides them
for the ops with real SIMD staging needs (conv, fc, depthwise, softmax).

## Stack budget guidance

On the device, the model's stack usage is
`ARENA_LEN + SCRATCH_LEN` (+ the frame). Reference numbers:

| Model | arena behavior |
|---|---|
| sine / hello_world / kws / anomaly | fit the arena — fine on-device |
| person_detect_vww | generated predict allocas ~232 KB — exceeds the ~65 KB device stack → `SKIP reason=stack` |

If a generated model exceeds your stack budget, options are:

1. **PSRAM**: run the arena in PSRAM via a PSRAM-backed stack/arena
   (the bench firmware's `run_on_psram_stack` pattern).
2. **Bigger stack**: raise the stack region in the linker script.
3. **Fix the emitter backlog**: per-tensor stack emission only affects
   models the arena planner rejects (single tensor > 512 KiB); that's a
   codegen follow-up, not a runtime knob.

## 16-byte alignment

SIMD paths require 16-byte-aligned buffers. Generated code handles this
for its own locals (\(repr(C, align(16))\) arena/tensors, aligned graph
input staging, re-aligned scratch carves). If you pass *your own* input
or scratch, they are read byte-wise (no alignment requirement); only
kernel-internal carves need alignment, and the dispatch owns that.

## Sanity check

When in doubt, run the scratch-parity test:

```sh
cargo test -p hematite-codegen scratch
# scratch-mirror: codegen SCRATCH_LEN == s3 *_scratch_need over the
# spec corpus + widened grids
```