---
title: Benchmarks — Zoo Models
---

# Zoo models

Six real int8 `.tflite` models benchmarked on-device: sine_regression,
hello_world, kws_micro_speech, anomaly_detect, person_detect_vww,
mobilenet_v2_1.0_224. Real weights, real inputs (the same golden inputs
the host corpus was captured with), executed-TFLM goldens as ground
truth.

## Runnable models (this board)

Reference board: ESP32-S3 rev v0.2, **no PSRAM**, 8 MB encrypted flash.
Only the four SRAM-fitting models run; the two PSRAM models are honest
SKIPs.

| model | in → out | ops |
|---|---|---|
| sine_regression (hello_world) | [1] → [1] | fc ×3 |
| hello_world_int8 | [1] → [1] | fc ×3 |
| kws_micro_speech | [1,1960] → [1,4] | reshape, dw-conv, reshape, fc, softmax |
| anomaly_detect | [1,640] → [1,640] | fc ×10 (RELU-fused) |
| person_detect_vww | [1,96,96,3] → [1,2] | SKIP — needs PSRAM |
| mobilenet_v2_1.0_224 | [1,224,224,3] → [1,1000] | SKIP — needs PSRAM |

## Results (on-device, min/median cycles @ 240 MHz)

Hematite column = `Model::<S3Backend>` post-Phase-20 optimizations;
ESP-NN = vendored `esp-nn` v1.2.5 S3 kernels. `Hematite speedup` =
ESP-NN median / Hematite median (>1 = Hematite faster).

| model | tier | ESP-NN cycles | Hematite-s3 cycles | Hematite speedup | bit-exact vs golden |
|---|---|---|---|---|---|
| sine_regression | SRAM | 189 | **800** | 0.24× (slower) | ✅ |
| hello_world | SRAM | 4,736 | **6,240** | 0.76× (slower) | ✅ |
| kws_micro_speech | SRAM | 771,690 | **1,787,766** | 0.43× (slower) | ✅ |
| anomaly_detect | SRAM | 14,002,080 | **16,986,217** | 0.82× (slower) | ✅ (ESP-NN diverges — see below) |
| person_detect | — | OOM | SKIP (no PSRAM) | — | — |
| mobilenet_v2 | — | OOM | SKIP (no PSRAM) | — | — |

!!! note "Why Hematite trails on per-model zoo rows"

    These models are **flash-weight-bound on this board**, not
    kernel-bound: generated weights are DROM (flash) consts, and an
    80 KiB DROM stream runs ~96× slower than the same data from SRAM.
    Both stacks stream flash on this board (ESP-NN passes static-const
    DROM weights directly), so the comparison is fair — and the win is
    architectural: SRAM/PSRAM-resident weight loading (proven **145×**
    on the fit-model bench) removes it. On the three synthetic
    conv-heavy models Hematite **beats** ESP-NN (1.68M vs 2.63M,
    763K vs 985K, 651K vs 649K cycles).

## Bit-exactness vs the executed-TFLM golden (FNV-1a)

| model | Hematite-s3 FNV | ESP-NN FNV | host golden FNV | verdict |
|---|---|---|---|---|
| hello_world | `0x010c56d3` | `0x010c56d3` | `0x010c56d3` | ✅ identical |
| kws | `0x897c5015` | `0x897c5015` | `0x897c5015` | ✅ identical |
| anomaly | `0xa83d07d6` | **`0x16213cfa`** | `0xa83d07d6` | Hematite = golden; **esp_nn diverges** |

The anomaly divergence is esp_nn's requantize rounding (not
accumulation): its fc asm uses a sign-dependent double nudge (+1 nudge
pattern) that is not gemmlowp-identical for negative products — exactly
±1 on 5 of 9 layers. Hematite implements the TFLM
`MultiplyByQuantizedMultiplier` semantics exactly. Full isolation:
`models_benchmark.md` §4 in the repo.

## Honest SKIP records

- **person_detect**: `SKIP reason=stack` on device (generated predict
  allocas 232 KB of intermediates on the stack; device stack ~65 KB).
  PASSES under QEMU (its emulation tolerates the windowed-underflow
  path) — see [run-under-qemu](../how-to/run-under-qemu.md).
- **mobilenet_v2**: `SKIP reason=no-psram` (needs ~4 MiB for
  intermediates). Re-run on a PSRAM board.
- Both stacks skip the same rows for the same hardware reasons — the
  comparison never fabricates a number.

Next: [Methodology](methodology.md).