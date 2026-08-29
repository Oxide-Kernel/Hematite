---
title: Reference — API
---

# API reference

The full, item-level API documentation lives in **rustdoc** per crate —
published on docs.rs when the crates are released. This page is the
index.

## Crates

| Crate | Layer | docs.rs (when published) |
|---|---|---|
| `hematite-core` | L0 — semantics | [`hematite-core`](https://docs.rs/hematite-core) — `KernelBackend`, `FusedKernelBackend`, `KernelError`, all `*Params` |
| `hematite-int8` | L0 — math | [`hematite-int8`](https://docs.rs/hematite-int8) — requantize, saturating cast, quantize_multiplier |
| `hematite-ref` | L1 — reference | [`hematite-ref`](https://docs.rs/hematite-ref) — `RefBackend`, scalar kernels |
| `hematite-s3` | L2 — acceleration | [`hematite-s3`](https://docs.rs/hematite-s3) — `S3Backend`, kernels, scratch sizes |
| `hematite-codegen` | L3 — macro | [`hematite-codegen`](https://docs.rs/hematite-codegen) — `#[model]` + test variants |
| `hematite-memory` | L3 — planner | [`hematite-memory`](https://docs.rs/hematite-memory) — `liveness_plan`, `ArenaPlan`, `ScratchLayout` |
| `hematite-tests` | L4 | golden corpus (test-only) |
| `hematite-benchmarks` | L4 | device firmware (bin) |

## Build the docs locally

```sh
cargo doc --workspace --no-deps --open
```

## The key types at a glance

```rust
// L0 — the contract
pub trait KernelBackend { /* ~35 op methods + scratch sizes */ }
pub trait FusedKernelBackend: KernelBackend { /* 3 composed methods */ }
pub enum KernelError { ShapeMismatch, ScratchTooSmall, Unsupported }

// L2 — the accelerated backend
pub struct S3Backend;                       // zero-sized, stateless
impl S3Backend {
    pub fn conv2d_scratch_size(&Self, ...)  // scratch formula override
    // ...
}

// L3 — the macro
#[model("path.tflite")]                     // generates Model<B>
pub struct MyModel;                          // + INPUT_LEN/OUTPUT_LEN/SCRATCH_LEN/ARENA_LEN

// generated per model:
impl<B: FusedKernelBackend> MyModel<B> {
    pub const fn new(backend: B) -> Self;
    pub fn predict(&mut self, input: &[i8; INPUT_LEN]) -> [i8; OUTPUT_LEN];
    pub fn predict_with_scratch(&mut self, input, output, scratch)
        -> Result<(), KernelError>;
}
```

## Feature flags

| Crate | Feature | Meaning |
|---|---|---|
| `hematite-benchmarks` / workspace | `qemu` | compile out weighted-op SIMD paths, UART0 logging, PSRAM probe — for QEMU runs (see [run-under-qemu](../how-to/run-under-qemu.md)) |
| `hematite-benchmarks` | `model-validation` | enable the model-validation + zoo-runner sections of the firmware |
| `hematite-s3` | `qemu` | propagates the same gating into the s3 SIMD dispatch |

## Versioning policy

- `version 0.1.0` workspace-wide.
- Semver: the `KernelBackend` trait is the compatibility surface — adding
  a method is a breaking change; the `*_scratch_size` defaults exist so
  backends stop compiling loudly, never quietly wrong.

Next: [Op support matrix](op-support-matrix.md).