---
title: Layer 2 — Accelerated Backends
---

# Layer 2 — The backend layer (`hematite-s3` today)

The speed — and the **extensibility point of the whole library**. Everything
above this layer (contract, compiler, tests) is backend-agnostic; Layer 2 is
where hardware-specific acceleration lives, behind the `KernelBackend` (+
`FusedKernelBackend`) traits.

The current backend is **`S3Backend`** — the ESP32-S3 (Xtensa TIE728)
implementation, with **bespoke SIMD kernels** written from scratch in Rust +
inline assembly (`s8_accx_*.S` files) — no vendored Espressif kernels.
Future backends (other chips, other SIMD ISAs, …) follow the same shape:
implement the trait, override the scratch-size functions, reuse the
dispatch-gate pattern.

## The dispatch contract

Every op follows the same three-tier dispatch:

```text
call S3Backend::conv2d(...)
   │
   ├─ SIMD eligibility gate (shape query) ── NO ──► scalar fallback (bit-exact)
   │        │ YES
   │        ▼
   ├─ scratch need check ────────────────────── NO ──► scalar fallback
   │        │
   │        ▼
   └─ ACCX/TIE728 SIMD kernel (bespoke asm) ──► requantize epilogue
```

- **Gates are honest.** A shape the gate rejects runs the scalar fallback
  *in the same backend* — bit-exact, never a wrong answer.
- **SIMD is `cfg`-gated** to real silicon
  (`cfg(all(target_arch = "xtensa", not(feature = "qemu")))`): on the
  host, `S3Backend` is entirely scalar — and bit-identical to the device.
- **Scratch-size overrides** exist so `predict_with_scratch` callers size
  correctly and the SIMD path can actually engage.

## The kernel inventory

| Kernel | File | Notes |
|---|---|---|
| `conv2d_1x1` | `conv1x1.rs` + `s8_accx_conv1x1.S` | incl. channel-padded variant for in_c < 16 / non-multiple-of-16 |
| `conv2d_3x3` | `conv3x3.rs` + `s8_accx_conv3x3.S` | incl. SAME/stride-2 paths |
| `depthwise_conv2d` | `depthwise.rs` + `s8_accx_depthwise.S` (+ `anytap` + `anytap_bc1`) | incl. `depth_multiplier > 1`, arbitrary filters (10×8 KWS), single-channel broadcast |
| `fully_connected` | `gemm.rs` + `s8_accx_gemm.S` | incl. tiny-fc fast path and inline-requantize dispatch |
| `softmax` | `softmax.rs` + `s8_softmax.S` | TFLM int8 convention (out scale = input scale, zp = i8::MIN) |
| pool / elementwise / activations / reductions / data movement | `pool.rs`, `elementwise.rs`, `activations.rs`, `reductions.rs`, `data_movement.rs` (+ `.S` where SIMD) | generic pool (any filter/stride/pad), relu6/hard-swish lanes, extended mean, bc1 broadcast |

## The TIE728/ACCX story (why the kernels are bit-exact)

The assembler kernels use `EE.*` TIE728 instructions. Two properties make
them *correct* where the vendor libraries are not:

1. **32-bit GPR accumulators.** The bespoke kernels accumulate with
   `EE.VMULAS.S8.ACCX` into the GPR file — a full 32-bit accumulator
   holding genuine 16-bit products. The vendored
   `dl_tie728_s8_conv`-style kernels saturate their **8-bit QACC lanes**:
   a single `127×127` product already reads back `0x7f`, so they cannot
   represent a real int8 convolution.
2. **A reverse-engineered QACC read-back.** For depthwise, the
   wide-accumulator (40-byte) read-back was reverse-engineered from
   silicon rather than copied — documented in
   [the engineering history](../benchmarks/methodology.md).

The requantize epilogue is the shared `hematite-int8` math — TFLM-exact
single-rounding. See
[comparing-to-esp-nn](../comparison/vs-esp-nn.md) for the full
correctness comparison.

## Scratch sizes

The SIMD paths stage padded copies in scratch. `S3Backend` overrides the
trait's `*_scratch_size` fns:

| Op | Scratch need formula |
|---|---|
| conv1x1 / fc | padded input (`pad16`), padded weights, i32 accum buffer, optional wsum |
| conv3x3 | padded input + weights (when padded), accum buffer |
| depthwise | staged padded input (`padded_h·w·padded_c`, or single-channel for bc1), padded filter, partials |
| softmax | scratch for the TFLM-deviation-free path |

`hematite-codegen` mirrors these formulas at macro time (`SCRATCH_LEN`,
scratch-parity tested) — see [Layer 3](layer-3-codegen.md) and the
[memory model](memory-model.md).

## Unsupported — honest failure

Ops with no s3 kernel (`matmul`, `sigmoid`, `tanh`, `leaky_relu`,
`prelu`, `quantize`, `dequantize`, recurrent, most reductions) return
`KernelError::Unsupported` — never a silently wrong result.

## Host-compilable by design

The same sources compile on the host (SIMD `cfg`-out, scalar fallback
in), so every kernel is host-testable and the full test suite runs
without hardware — and the s3 host behavior is bit-equal to the device.

Next: [Layer 3 — codegen](layer-3-codegen.md).