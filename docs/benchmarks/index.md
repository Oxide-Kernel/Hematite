---
title: Benchmarks
---

# Benchmarks

Hematite's performance claims are **measured on real silicon**, with
full provenance on every row. This section collects the numbers and the
methodology behind them.

## The two benchmark corpora

| Corpus | Models | What it measures |
|---|---|---|
| [Synthetic A/B/C](synthetic-models.md) | 3 deterministic-fill models (4-layer CNN, MobileNetV2-style, real MobileNetV2 shapes) | End-to-end + per-op cycle comparison vs the ESP-NN C stack |
| [Zoo models](zoo-models.md) | 6 real int8 `.tflite` models (sine, hello_world, kws, anomaly, person_detect, mobilenet_v2) | Real-model inference, bit-exactness vs executed-TFLM goldens, per-model cycles |

## Headline numbers (device, ESP32-S3 rev v0.2 @ 240 MHz)

### Synthetic models vs ESP-NN

| Model | ESP-NN | **Hematite** | Hematite wins |
|---|---|---|---|
| A — 4-layer CNN | 2,630,401 cyc | **1,686,922 cyc** | **1.56×** |
| B — mv2mini 7-layer | 994,782 cyc | **763,105 cyc** | **1.30×** |
| C — real MobileNetV2 (SAME + stride-2) | 655,303 cyc | **650,773 cyc** | **1.01×** (floor-limited) |

### Zoo models (Hematite on-device, post-Phase-20)

| Model | Hematite cycles | Bit-exact vs golden |
|---|---|---|
| sine_regression | 800 | ✅ |
| hello_world | 6,240 | ✅ |
| kws_micro_speech | 1,787,766 | ✅ |
| anomaly_detect | 16,986,217 | ✅ (ESP-NN diverges from its own golden — see [vs. ESP-NN](../comparison/vs-esp-nn.md)) |
| person_detect_vww | SKIP — needs PSRAM | — |
| mobilenet_v2 224×224 | SKIP — needs PSRAM | — |

## How to read these

- All rows follow the **ledger rule**: ISO timestamp + commit of the
  measured code + full cycles on both stacks + config (never
  deltas-only).
- **Same-conditions rule**: identical model, input bytes, memory tier,
  CPU frequency on both stacks.
- **FNV-1a checksums** machine-check bit-exactness on every row.
- QEMU-emulated numbers are labeled separately; silicon is the source of
  truth (see [run-under-qemu](../how-to/run-under-qemu.md)).

## Sections

- [Synthetic models](synthetic-models.md) — the A/B/C end-to-end and
  per-op tables
- [Zoo models](zoo-models.md) — real-model rows, SKIP rationale, bit-exactness
- [Methodology](methodology.md) — board, harness, flashing pipeline, ledger rules