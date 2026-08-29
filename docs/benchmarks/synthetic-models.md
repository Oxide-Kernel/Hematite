---
title: Benchmarks — Synthetic Models
---

# Synthetic A/B/C models

Three synthetic models with **deterministic fill** weights (same FNV-1a
checksums on both stacks), run end-to-end at 240 MHz on the reference
ESP32-S3. Full provenance: device, 2026-08-10, Hematite HEAD `33d498a`;
ESP-NN espnn-baseline (T0.3-era vendored `esp-nn` v1.2.5).

## The models

| Model | Layers |
|---|---|
| **A — 4-layer CNN** | conv3x3 (input_c=16), maxpool, conv1x1, FC |
| **B — MobileNetV2-style (mv2mini, 7-layer)** | 3-ch first conv, depthwise ×2, conv1x1 ×2, pool, FC |
| **C — real MobileNetV2 (mv2real, 6-layer)** | SAME + stride-2 conv3x3, stride-2 depthwise, conv1x1, FC |

## End-to-end results

| Model | ESP-NN optimized | **Hematite** | Hematite wins |
|---|---|---|---|
| A — 4-layer CNN | 2,630,401 cyc | **1,686,922 cyc** | **1.56×** |
| B — mv2mini 7-layer | 994,782 cyc | **763,105 cyc** | **1.30×** |
| C — real MobileNetV2 (SAME + stride-2) | 655,303 cyc | **650,773 cyc** | **1.01×** (floor-limited) |

Every layer of every model is **bit-exact vs the scalar reference**
(identical `out_fnv` on both stacks).

## Why the win

- **A:** conv3x3 (input_c=16) runs the **fast16 unrolled ACCX path**; FC
  benefits from the **hardware-loop** path and the **asm requantize
  epilogue**.
- **B:** the 3-channel first conv runs **zero-padded SIMD**; depthwise
  layers run the bespoke **QACC per-lane kernel** with the
  reverse-engineered 40-byte accumulator read-back; FC gets the same wins
  as A.
- **C:** SAME/stride-2 conv3x3, stride-2 depthwise, conv1x1, and FC all
  run bespoke SIMD — all six layers are SIMD-engaged.

Per-kernel cost (64-channel shapes): the ACCX kernels sit at the
**TIE728 MAC-issue floor** (~0.1–0.6 cyc/MAC), and the fused asm
requantize removes the last per-pixel wrapper cost.

## Per-operation reference table

Single-kernel microbenchmarks (same fill pattern). ⚠️ The Espressif column
is the **raw vendored asm entry** (kernel only, no requantize, one pixel
per call for conv3x3) — reference only, not apples-to-apples.

| Operation | Espressif raw cyc | Hematite cyc | Espressif csum | Hematite csum | ref csum |
|---|---|---|---|---|---|
| conv1x1 64×1×1×64 | 472 | 4266/4267 | `0x5eee898e` | `0x0bea8225` | `0x0bea8225` |
| conv3x3 32×32 64×3×3×64 VALID ⚠️ | 2824 | 8869776 | `0xd1a9b601` | `0x0a181085` | `0x0a181085` |
| conv3x3 16×16 SAME 32×3×3×32 | — | 881096 | — | `0xc53ebbc5` | `0xc53ebbc5` |
| depthwise 3×3 7×7×32 | — | 91411 | — | `0xea4d8cb0` | `0xea4d8cb0` |
| depthwise 3×3 12×12 S2 SAME 16ch | — | 35717 | — | `0x5159710e` | `0x5159710e` |
| depthwise 3×3 12×12 non-%16 12ch | — | 104047 | — | `0x8da1a066` | `0x8da1a066` |
| fc 256×64 | 1288 | 7161/7188 | — | `0x32e35185` | `0x32e35185` |
| max_pool 2×2×16 | 1396 | 14046 | — | `0x651bfdc5` | `0x651bfdc5` |
| avg_pool 2×2×16 | 7181 | 19913 | — | `0xb8a6ddc5` | `0xb8a6ddc5` |
| relu 256 | 175 | 358 | — | `0x6c620b3d` | `0x6c620b3d` |
| add 256 | 167 | 477/490 | — | `0x14834bbb` | `0x14834bbb` |
| mul 256 | 539 | 851/879 | — | `0xd3c0a7f1` | `0xd3c0a7f1` |
| sub 256 | 265 | 555/556 | — | `0x62d74671` | `0x62d74671` |
| softmax 1×1000 | — | 476499 | — | `0xaf0d15aa` | `0xaf0d15aa` |

## Honest caveats

- The residual gap between Hematite's Rust **public-API** rows and the C
  raw-asm entries is the **wrapper cost** (slice validation + eligibility
  gate + args build + requantize epilogue) — the prepared-path handles
  (Phase 18) close most of it: conv1x1 1.42×, avg_pool 1.02×, fc 1.22×.
- Model C's 1.01× is **floor-limited**: all six layers are already
  SIMD-engaged; the residual is the bit-exact ACCX contract vs the
  8-bit-saturating QACC asm (~2.5× issue-rate trade for correctness).
- Full ledger rows: `benchmarks/ESPRESSIF_VS_HEMATITE.md` in the repo.

Next: [Zoo models](zoo-models.md).