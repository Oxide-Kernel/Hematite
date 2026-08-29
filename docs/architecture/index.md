---
title: Architecture
---

# Architecture

Hematite is a **library of five clean layers**. Each layer is a crate (or
a pair of crates) with a narrow responsibility, and the dependency arrow
always points downward — higher layers depend on lower layers, never the
reverse. Layers 0, 1, 3, and 4 are **platform-independent**: they form
the NN *library* itself. Layer 2 is the backend slot — **hematite-s3**
(ESP32-S3 TIE728 SIMD) is the current occupant, and any future
accelerated backend joins by implementing the same traits.

```text
                    ┌───────────────────────────────────────────────┐
  L4  Validation    │  hematite-tests   hematite-benchmarks         │
  & measurement    │  goldens · simd sweep · zoo bench · firmware  │
                    └───────────────▲───────────────────────────────┘
                                    │ uses / proves
                    ┌───────────────┴───────────────────────────────┐
  L3  Compilation   │  hematite-codegen   hematite-memory          │
                    │  #[model] · flatbuffer · fusion · arena      │
                    └───────────────▲───────────────────────────────┘
                                    │ emits calls through traits
                    ┌───────────────┴───────────────────────────────┐
  L2  Backends      │  hematite-s3 (current)  ·  (your backend)    │
                    │  S3Backend · TIE728 SIMD · dispatch · traits │
                    └───────────────▲───────────────────────────────┘
                                    │ implements
                    ┌───────────────┴───────────────────────────────┐
  L1  Reference     │  hematite-ref                                 │
                    │  RefBackend · scalar oracle                   │
                    └───────────────▲───────────────────────────────┘
                                    │ implements
                    ┌───────────────┴───────────────────────────────┐
  L0  Semantics     │  hematite-core    hematite-int8               │
                    │  KernelBackend · Params · int8 math          │
                    └────────────────────────────────────────────────┘
```

## Why five layers?

- **L0 owns the contract.** All slice layouts, quantization semantics,
  and error behavior are defined once, in types. Backends implement;
  codegen emits against; tests assert against.
- **L1 owns the truth.** The scalar reference is the golden oracle —
  everything else is validated against it. Because it implements the
  *same trait*, "is this backend correct?" is a runtime equality check.
- **L2 owns the speed — and the pluggability.** Every accelerated
  backend lives behind the same trait, so it can never change the
  model-level behavior contract — only the per-op implementation.
  `hematite-s3` is the reference example; a new backend implements
  `KernelBackend` (+ `FusedKernelBackend`) and slots in unchanged
  (see the [custom-backend tutorial](../tutorials/custom-backend.md)).
- **L3 owns the ergonomics.** `#[model]` turns a flatbuffer into typed,
  straight-line Rust. It is the only layer that reads user model files.
- **L4 owns the evidence.** The test suite and benchmark firmware prove
  the claims: bit-exactness, power, cycles, methodology.

## Layer guides

| Layer | Guide |
|---|---|
| 0 — Semantics (the trait contract + int8 math) | [Layer 0](layer-0-semantics.md) |
| 1 — Reference oracle | [Layer 1](layer-1-reference.md) |
| 2 — Accelerated backends (s3 today; the pattern for more) | [Layer 2](layer-2-s3-backend.md) |
| 3 — Model compilation | [Layer 3](layer-3-codegen.md) |
| 4 — Validation & measurement | [Layer 4](layer-4-validation.md) |
| Cross-cutting — memory model | [Memory model](memory-model.md) |

## Data flow at a glance

```text
model.tflite
    │   (compile time)
    ▼
#[model] ── flatbuffer walk ── op schedule ── fusion selector
    │                                        │ arena planner
    │                             ┌──────────┴──────────┐
    ▼                             ▼                     ▼
typed Model<B>           FusedKernelBackend     ARENA_LEN / SCRATCH_LEN
(straight-line calls)    (composed groups)      (static sizes)
    │
    ▼ (runtime, generic over B)
RefBackend (host/scalar)  ·  S3Backend (device/SIMD, scalar fallback)
    │
    ▼
bit-identical bytes ── verified by ── L4 goldens + FNV-1a + device sweep
```