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
  verified on-device for every kernel row, every layer of three models, and
  the fused-path model-validation rows (sine / hello_world / kws pass
  bit-exact through `Model::<S3Backend>` on real silicon, real-silicon run 1,
  2026-08-11).
- **Bespoke from scratch.** `s8_accx_conv1x1.S`, `s8_accx_conv3x3.S`,
  `s8_accx_depthwise.S` (incl. the chunked anytap variant for arbitrary
  filters and `depth_multiplier > 1`), `s8_requantize.S`, `s8_softmax.S`
  implement the accumulation, the wide-accumulator read-back
  (reverse-engineered from silicon, not copied), and the requantize epilogue
  ourselves.
- **Composed kernels (Phase 19).** The code generator fuses adjacent ops
  into composed-kernel calls — conv + per-step requantize + activation +
  residual-add in one SIMD pass, register-held elementwise chains, and
  T2-gated pool/softmax input-folds — via an additive `FusedKernelBackend`
  trait. Fused output == unfused output **bit-exact** for all 6 zoo models
  (host harness); intermediates live in one stack-local 16-byte-aligned
  arena, no `unsafe`, no `static mut`.
- **Shape-flex SIMD coverage.** Beyond the classic shapes: generic pool
  (any filter/stride/pad + clamp), non-identity elementwise offsets/
  multipliers, relu6 + hard_swish lane models, channel-padded conv1x1/fc
  (input channels below / non-multiple of 16), extended mean (in_c > 256 via
  looped accumulation), depthwise with `depth_multiplier > 1` and
  non-3×3 filters (the KWS 10×8 anytap path), and small-shape FC — all
  bit-exact vs the scalar reference and QEMU-gated (`cfg(all(xtensa,
  not(qemu)))`).
- **`no_std` + host tests.** Kernels are host-compilable; the suite runs on
  the host and every model/row is re-verified bit-exact on the device.
  `S3Backend` implements the full `KernelBackend` trait so `#[model]`-
  generated zoo models run accelerated (conv, depthwise, fc, pooling,
  softmax, elementwise, mean, data movement) on real silicon.

## Head-to-head vs the standard ESP-NN stack

Three identical models, deterministic fill, same FNV-1a checksums, run
end-to-end on both stacks at 240 MHz on an ESP32-S3 (rev v0.2). Full detail
and the benchmark ledger in
[`benchmarks/ESPRESSIF_VS_HEMATITE.md`](benchmarks/ESPRESSIF_VS_HEMATITE.md).

| Model | ESP-NN optimized | **Hematite** | Hematite wins |
|---|---|---|---|
| A — 4-layer CNN | 2,630,401 cyc | **1,686,922 cyc** | 1.56× |
| B — MobileNetV2-style 7-layer | 994,782 cyc | **763,105 cyc** | 1.30× |
| C — real MobileNetV2 (SAME + stride-2) | 655,303 cyc | **650,773 cyc** | 1.01× |

Every number is bit-exact with the scalar reference on every layer of every
model — the comparison is apples-to-apples. Benchmark ledger rule: every
measured row carries ISO timestamp + git commit of the measured code + FULL
Hematite cycles + FULL C-stack cycles + speedup ratio + config (never
deltas-only) — see `benchmarks/ESPRESSIF_VS_HEMATITE.md` and the
`benchmarks/zoo-results/` readme.

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
QACC probes), and `qemu-baseline`. `hematite-benchmarks` carries the
on-device SIMD-correctness sweep (40 checks vs the scalar oracle), the zoo
model-bench runners (fused and unfused arms), the A/B/C graph benches, and
the per-kernel spec rows — see `hematite-benchmarks/src/simd_validation.rs`
and `spec.rs`.

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

Flashing this repo's board (ESP32-S3 rev v0.2, permanently flash-encrypted,
`SPI_BOOT_CRYPT_CNT=0x7`) requires the encrypted path — **`esptool.py
write_flash --encrypt`** (`espflash` has no encryption support and a
plaintext write will not boot). See `benchmarks/ESPRESSIF_VS_HEMATITE.md`
and `PROJECT_LOG.md` for the full flash/capture pipeline and the engineering
history (Phases 0–19).

## Engineering history

`PROJECT_LOG.md` documents the full journey: hardware bring-up, the C-SIMD
bit-exact cross-language match, the bespoke ACCX GPR-accumulator kernels, the
from-silicon QACC depthwise read-back, the fast-path optimizations, the
ESP-NN head-to-head, and Phase 19 (composed kernels + shape-flex SIMD +
selector — with the real-silicon run-1 record and its known panic
follow-up).
