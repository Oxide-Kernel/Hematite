---
title: Comparison — Hematite vs ESP-NN
---

# Hematite vs. the standard ESP-NN stack

Hematite competes with (and is often confused with) the **ESP-NN /
ESP-DL** acceleration stacks for Espressif SoCs. The honest differences
are architectural, not just "faster":

## 1. A compiler vs. a kernel library

| | ESP-NN / ESP-DL | Hematite |
|---|---|---|
| What it is | A **kernel library** you wire by hand (`esp_nn_conv_s8(...)` calls, manual tensor plumbing) | A **compile-time model compiler**: `#[model]` → typed straight-line Rust |
| Op dispatch | You write the graph/loop | The macro emits the exact call chain |
| Error surface | Mostly your responsibility | `KernelError`, slice-length validation per op |
| Backends | C kernels | `RefBackend` (host/scalar) and `S3Backend` (device/SIMD) interchangeable behind one trait |

With ESP-NN you build the plumbing; with Hematite you declare the model
and get typed inference code. The kernels are behind a trait, so the
"which backend runs this model" question is a one-line choice.

## 2. Bit-exactness is a stated invariant, not an accident

The correctness gap between the two stacks is real and was measured:

- The vendored Espressif `dl_tie728_s8_*` conv kernels (used by ESP-DL
  and pre-ACCX Hematite) saturate **8-bit QACC lanes**: a single
  `127×127` product already reads back `0x7f`. They **cannot represent a
  real int8 convolution**.
- Standard `esp-nn`'s esp32s3 asm uses a different accumulator scheme
  and happens to be bit-exact for the A/B/C benchmark models — confirmed
  empirically.
- Hematite's bespoke `EE.VMULAS.S8.ACCX` GPR-accumulator kernels are
  bit-exact **by construction**: 32-bit accumulators, full 16-bit
  products, TFLM-exact requantize.

### The on-device proof

From `models_benchmark.md` (real silicon, real weights, real goldens —
`ai-edge-litert` 2.1.6 executed-TFLM host reference):

| Model | Hematite = host golden? | ESP-NN = host golden? |
|---|---|---|
| hello_world | ✅ `0x010c56d3` | ✅ `0x010c56d3` |
| kws_micro_speech | ✅ `0x897c5015` | ✅ `0x897c5015` |
| anomaly_detect | ✅ `0xa83d07d6` | ❌ **`0x16213cfa`** — esp_nn's own output **diverges from its own golden** |

The anomaly divergence was isolated: esp_nn's fc requantize differs from
TFLM `MultiplyByQuantizedMultiplier` by **exactly ±1** on 5 of 9 layers
(its sign-dependent double-nudge rounding is not gemmlowp-identical for
negative products; its s16 asm fallback is a third, again different,
rounding). **Hematite matches the executed-TFLM reference exactly.**

## 3. Zero runtime allocation, zero C, pure Rust

| | ESP-NN | Hematite |
|---|---|---|
| Language | C + assembly | Pure Rust (`no_std`) |
| Runtime alloc | caller-managed | none — stack + macro-planned arena |
| Dynamic shapes | caller responsibility | rejected at compile time (`shape4`/flat_len) |
| Static analysis | n/a | zero `Vec`/`Box`/`alloc`, zero `unsafe` in generated code |

## 4. Honest measurement methodology

Every Hematite benchmark row carries **ISO timestamp + commit of the
measured code + FULL cycles on both stacks + speedup + config** — never
deltas-only. Rows that cannot run on a given board are reported as
honest SKIP with reason + rerun condition. QEMU numbers are labeled
emulated; silicon is the source of truth. See
[benchmark methodology](../benchmarks/methodology.md).

## 5. The performance picture (verified on silicon)

Synthetic A/B/C models (real device, 240 MHz, ESP32-S3 rev v0.2):

| Model | ESP-NN | **Hematite** | Hematite wins |
|---|---|---|---|
| A — 4-layer CNN | 2,630,401 cyc | **1,686,922 cyc** | 1.56× |
| B — MobileNetV2-style 7-layer | 994,782 cyc | **763,105 cyc** | 1.30× |
| C — real MobileNetV2 (SAME + stride-2) | 655,303 cyc | **650,773 cyc** | 1.01× |

Zoo models (real weights, post-Phase-20, device): sine **800**, hello
**6,240**, kws **1,787,766**, anomaly **16,986,217** cycles. The
remaining per-model gaps vs ESP-NN on this board (hello 1.32×, kws
2.3×, anomaly 1.21×) are **flash-weight streaming**, not kernel cost:
generated weights are DROM consts, and an 80 KiB DROM stream is ~96×
slower than the same data in SRAM. Staging weights into SRAM once
(demonstrated 145× device speedup on the fit-model bench; the design
note for a future `PreparedModel::load`) closes it. Both stacks are
flash-bound on this board — the comparison stays fair.

Full tables: [Benchmarks](../benchmarks/index.md).

## 6. What Hematite is not

- Not an ESP-IDF component (no IDF dependency; works with the esp-hal
  no-std stack).
- Not a float engine (int8 quantized TFLM-semantics only).
- Not a general ML framework — a focused, provably-correct *library*
  for int8 models, accelerated on the ESP32-S3 today and portable to
  future backends through the same trait.