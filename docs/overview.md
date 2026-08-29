---
title: Overview
---

# Overview

Hematite is a **pure-Rust, `no_std` int8 neural-network inference
library**, built bit-exact against TensorFlow Lite Micro semantics. Its
core — the `KernelBackend` contract, the `#[model]` compiler, the memory
planner, the int8 math — is **platform-independent**: it runs anywhere
Rust runs. The **ESP32-S3 (Xtensa LX7 + TIE728 SIMD)** is the first
high-speed backend, and new backends plug in behind the same trait.

It occupies a specific niche: **sizes where a small, deterministic,
bit-exact library beats a heavyweight framework**, running entirely in
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

| Backend | Crate | Platform | Purpose |
|---|---|---|---|
| `RefBackend` | `hematite-ref` | any (host, `no_std`) | Scalar reference — the golden oracle |
| `S3Backend` | `hematite-s3` | ESP32-S3 (Xtensa TIE728) | SIMD-accelerated — the current speed backend |

Adding a backend for another target means implementing `KernelBackend`
(and optionally `FusedKernelBackend` for composed kernels) — the
generated model code, the compiler, and the correctness harness work
unchanged. See the [custom-backend tutorial](tutorials/custom-backend.md).

## Design pillars

- **Bit-exactness by construction.** The accumulator scheme
  (`EE.VMULAS.S8.ACCX`, 32-bit GPR accumulators, full 16-bit products)
  is bit-exact vs. TFLM reference semantics. The bespoke kernels never
  saturate lanes at 8 bits the way the vendored `dl_tie728_s8_*` kernels
  do.
- **Zero runtime allocation.** No `alloc`, no `Vec`, no `Box`. Everything
  is stack arrays plus a compile-time-planned arena; `no_std` in the
  device path.
- **A compiler-driven library.** You declare the model; the macro emits
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
| **L0 — Semantics** | `hematite-core`, `hematite-int8` | The `KernelBackend`/`FusedKernelBackend` contract + int8 math (TFLM-exact requant) — platform-independent |
| **L1 — Reference oracle** | `hematite-ref` | Scalar implementation of every trait method — the golden answer (platform-independent) |
| **L2 — Accelerated backends** | `hematite-s3` | `S3Backend` — bespoke TIE728 SIMD kernels + dispatch gates (ESP32-S3; the pattern for future backends) |
| **L3 — Model compilation** | `hematite-codegen`, `hematite-memory` | `#[model]` macro → typed `Model<B>`; arena/scratch planning (platform-independent) |
| **L4 — Validation & measurement** | `hematite-tests`, `hematite-benchmarks` | Golden-corpus tests + benchmark firmware |

## Scope

Hematite targets the **int8 quantized** inference path (per-channel
quantization, TFLM requant semantics). It is not a float engine, and it is
not a general ML framework — it is a focused, provably-correct *library*
for deploying int8 models, with maximum performance on the ESP32-S3 today
and a backend architecture ready for more targets tomorrow.

Next: [Installation](installation.md).