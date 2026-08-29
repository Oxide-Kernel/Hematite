---
title: Overview
---

# Overview

Hematite is a **pure-Rust, `no_std` int8 neural-network inference engine**
for the **ESP32-S3** (Xtensa LX7 + TIE728 SIMD), built bit-exact against
TensorFlow Lite Micro semantics.

It occupies a specific niche: **sizes where a small, deterministic,
bit-exact engine beats a heavyweight framework**, running entirely in
static memory with zero runtime allocation and zero C in the device build
path.

## The two halves of Hematite

Hematite is really two components that fit together:

### 1. A compile-time model compiler — `hematite-codegen`

```rust
use hematite_codegen::model;

#[model("path/to/model.tflite")]
pub struct MyModel;
```

At **compile time** the `#[model]` proc-macro:

1. reads the TFLite flatbuffer with a hand-rolled byte-offset walker (no
   runtime flatbuffers dependency),
2. plans memory (USMP-style liveness arena) and op fusion,
3. emits **straight-line Rust** — one function call chain, no interpreter
   loop, no dynamic dispatch at inference time.

The generated `Model<B>` is generic over the backend `B`:

```rust
// Host (scalar reference — golden oracle):
let out = MyModel::<hematite_ref::RefBackend>::new(hematite_ref::RefBackend)
    .predict(&input);

// Device (TIE728 SIMD accelerated):
let out = MyModel::<hematite_s3::S3Backend>::new(hematite_s3::S3Backend)
    .predict(&input);
```

The **same generated code** runs on both — output is bit-identical.

### 2. Kernel backends — `hematite-core` trait + implementations

The `KernelBackend` trait in `hematite-core` is the contract every backend
honors: conv2d, depthwise, fully-connected, pooling, softmax, activations,
elementwise, data movement, recurrent, reductions — each mirroring the
exact TFLite Micro int8 semantics.

| Backend | Crate | Purpose |
|---|---|---|
| `RefBackend` | `hematite-ref` | Scalar reference — the golden oracle |
| `S3Backend` | `hematite-s3` | ESP32-S3 TIE728 SIMD — the fast path |

## Design pillars

- **Bit-exactness by construction.** The accumulator scheme
  (`EE.VMULAS.S8.ACCX`, 32-bit GPR accumulators, full 16-bit products)
  is bit-exact vs. TFLM reference semantics. The bespoke kernels never
  saturate lanes at 8 bits the way the vendored `dl_tie728_s8_*` kernels
  do.
- **Zero runtime allocation.** No `alloc`, no `Vec`, no `Box`. Everything
  is stack arrays plus a compile-time-planned arena; `no_std` in the
  device path.
- **A compiler, not a library.** You declare the model; the macro emits
  the call chain. There is no inference loop and no op dispatch table to
  get wrong.
- **Honest engineering.** Benchmarks carry full ledger rows (timestamp +
  commit + both stacks' full cycles, never deltas-only); unsupported ops
  return `KernelError::Unsupported` instead of silently producing wrong
  answers; QEMU vs. silicon numbers are clearly separated.

## The 5-layer API surface

Hematite's crates form five layers. Each layer is a documented unit with
a clear responsibility — see [Architecture](architecture/index.md) for the
full layer-by-layer walk:

| Layer | Crates | Responsibility |
|---|---|---|
| **L0 — Semantics** | `hematite-core`, `hematite-int8` | The `KernelBackend`/`FusedKernelBackend` contract + int8 math (TFLM-exact requant) |
| **L1 — Reference oracle** | `hematite-ref` | Scalar implementation of every trait method — the golden answer |
| **L2 — Accelerated device backend** | `hematite-s3` | Bespoke TIE728 SIMD kernels + dispatch gates |
| **L3 — Model compilation** | `hematite-codegen`, `hematite-memory` | `#[model]` macro → typed `Model<B>`; arena/scratch planning |
| **L4 — Validation & measurement** | `hematite-tests`, `hematite-benchmarks` | Golden-corpus tests + on-device benchmark firmware |

## Scope

Hematite targets the **int8 quantized** inference path (per-channel
quantization, TFLM requant semantics). It is not a float engine, and it is
not a general ML framework — it is a focused, provably-correct engine for
deploying int8 models to the ESP32-S3 with maximum performance.

Next: [Installation](installation.md).