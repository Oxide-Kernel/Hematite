---
title: Architecture
---

# Architecture

Hematite is designed as **five clean layers**. Each layer is a crate (or
a pair of crates) with a narrow responsibility, and the dependency arrow
always points downward — higher layers depend on lower layers, never the
reverse.

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
  L2  Acceleration  │  hematite-s3                                 │
                    │  S3Backend · TIE728 SIMD asm · dispatch      │
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
- **L2 owns the speed.** The SIMD backend lives behind the same trait, so
  it can never change the model-level behavior contract — only the
  per-op implementation.
- **L3 owns the ergonomics.** `#[model]` turns a flatbuffer into typed,
  straight-line Rust. It is the only layer that reads user model files.
- **L4 owns the evidence.** The test suite and benchmark firmware prove
  the claims: bit-exactness, power, cycles, methodology.

## Layer guides

| Layer | Guide |
|---|---|
| 0 — Semantics (the trait contract + int8 math) | [Layer 0](layer-0-semantics.md) |
| 1 — Reference oracle | [Layer 1](layer-1-reference.md) |
| 2 — ESP32-S3 accelerated backend | [Layer 2](layer-2-s3-backend.md) |
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