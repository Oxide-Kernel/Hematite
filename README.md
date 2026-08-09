# Hematite

A pure-Rust, `no_std` int8 neural-network inference engine for the
**ESP32-S3**, built from scratch and validated bit-exact against a scalar
reference on real hardware.

Hematite runs the same TensorFlow Lite–style int8 kernels (conv2d, depthwise
conv, fully connected, pooling, softmax, elementwise, activations) with
**100% bespoke Xtensa TIE728 SIMD assembly** — no vendored Espressif conv
kernels — and beats the standard `esp-nn` stack end-to-end on every model
benchmarked.

## Highlights

- **Bit-exact by construction.** Every SIMD kernel's output equals the scalar
  reference exactly (same `out_fnv` FNV-1a checksum) on the real ESP32-S3 —
  verified on-device for every kernel row and every layer of three models.
- **Bespoke from scratch.** `s8_accx_conv1x1.S`, `s8_accx_conv3x3.S`,
  `s8_accx_depthwise.S`, `s8_requantize.S`, `s8_softmax.S` implement the
  accumulation, the wide-accumulator read-back (reverse-engineered from
  silicon, not copied), and the requantize epilogue ourselves.
- **Full MobileNetV2 parity.** SAME padding, stride-2, non-zero input
  offsets, softmax, and non-multiple-of-16 depthwise channels all run SIMD
  (no scalar fallback) — the shapes a stock MobileNetV2 uses.
- **`no_std` + host tests.** Kernels are host-compilable; the 34-test suite
  (plus 3 kernel-level tests) runs on the host and every model/row is
  re-verified bit-exact on the device.

## Head-to-head vs the standard ESP-NN stack

Three identical models, deterministic fill, same FNV-1a checksums, run
end-to-end on both stacks at 240 MHz on an ESP32-S3 (rev v0.2). Full detail
in [`benchmarks/ESPRESSIF_VS_HEMATITE.md`](benchmarks/ESPRESSIF_VS_HEMATITE.md).

| Model | ESP-NN optimized | **Hematite** | Hematite wins |
|---|---|---|---|
| A — 4-layer CNN | 2,630,401 cyc | **1,708,383 cyc** | 1.54× |
| B — MobileNetV2-style 7-layer | 994,782 cyc | **770,986 cyc** | 1.29× |
| C — real MobileNetV2 (SAME + stride-2) | 655,303 cyc | **654,407 cyc** | 1.01× |

Every number is bit-exact with the scalar reference on every layer of every
model — the comparison is apples-to-apples.

## Workspace layout

| Crate | Role |
|---|---|
| `hematite-core` | op-parameter types (`Conv2DParams`, `DepthwiseConv2DParams`, …) |
| `hematite-int8` | `multiply_by_quantized_multiplier`, `saturating_cast` (TFLite semantics) |
| `hematite-ref` | scalar reference backend (the golden oracle) |
| `hematite-s3` | ESP32-S3 kernels — bespoke TIE728 SIMD asm + Rust dispatchers |
| `hematite-memory` | arena helpers |
| `hematite-codegen` | `#[model]` proc macro |
| `hematite-tests` | golden-corpus tests |
| `hematite-benchmarks` | on-device benchmark firmware + host test suite |

`benchmarks/` holds the comparison harnesses: `espnn-baseline` (standard
ESP-NN, vendored `esp-nn` v1.2.5), `espdl-baseline` (C microbenchmark /
QACC probes), and `qemu-baseline`.

## Building & running on hardware

Requires the `espup`-installed `esp` Xtensa toolchain (`source
~/export-esp.sh`), an ESP32-S3, and `espflash`:

```sh
# host tests
cargo test -p hematite-benchmarks --lib

# device firmware (release only)
cargo build --release -Zbuild-std=core,alloc \
  --target xtensa-esp32s3-none-elf -p hematite-benchmarks
```

See `benchmarks/ESPRESSIF_VS_HEMATITE.md` and `PROJECT_LOG.md` for the full
flash/capture pipeline, the engineering history (Phases 0–17), and the
hardware findings (including the permanent-flash-encryption flashing rule).

## Engineering history

`PROJECT_LOG.md` documents the full journey: hardware bring-up, the C-SIMD
bit-exact cross-language match, the bespoke ACCX GPR-accumulator kernels, the
from-silicon QACC depthwise read-back, the fast-path optimizations, and the
ESP-NN head-to-head.
