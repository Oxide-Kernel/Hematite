# espnn-baseline — standard ESP-NN end-to-end model baseline for the ESP32-S3

> 📊 **TL;DR comparison:** see
> [`benchmarks/ESPRESSIF_VS_HEMATITE.md`](../ESPRESSIF_VS_HEMATITE.md) — the
> one-page "Espressif NN Stack vs Hematite Stack" summary.

C harness that runs quantized int8 CNN models **end-to-end through the
standard ESP-NN stack** (`espressif/esp-nn` v1.2.5, `CONFIG_NN_OPTIMIZED` →
the esp32s3 hand-written asm kernels) on real hardware (ESP32-S3 @ 240 MHz),
printing CCOUNT cycle totals and sign-extending FNV-1a output checksums — the
C side of the head-to-head against the pure-Rust `hematite-s3` kernels.

- **Output match**: per-layer and final FNV-1a checksums, byte-identical to
  the Rust firmware's `out_fnv(ref/s3)` report (sign-extending convention —
  `h ^= (uint32_t)(int8_t)b`, NOT raw `uint8_t` bytes).
- **Cycle match**: CCOUNT deltas around a thin C wrapper, same protocol as
  the Rust side: 1 untimed warm-up + N=10 timed runs, min + median.

> ⚠️ **Hardware measurements** (real ESP32-S3 @ 240 MHz, NOT QEMU). Requires
> the physical board and the encrypted-flash pipeline below. The host that
> owns this workspace currently has **no ESP-IDF toolchain**
> (`idf.py` absent, `$IDF_PATH` empty) — the C side of the todo-20 zoo rows
> cannot be built/measured here; see [Zoo head-to-head status](#zoo-head-to-head-status).

## What it currently runs

Three synthetic end-to-end models, driven by a deterministic fill pattern
(`input[i]=(i*7+3)&0xFF`, `weights[i]=(i*13+11)&0xFF`, `bias[i]=i*17-8`),
identical to `hematite-benchmarks/src/spec.rs` / `model_*.rs`:

| Model | Shape language | esp-nn path | Documented ESP-NN cycles (min/med) |
|---|---|---|---|
| **Model A** (4-layer CNN) | conv3x3 32x32x16 → maxpool → conv1x1 → fc 7200→16 | `esp_nn_conv_s8` + `esp_nn_max_pool_s8` + `esp_nn_conv_s8` + `esp_nn_fully_connected_s8` | 2,630,401 / 2,630,423 |
| **Model B** (mv2mini, 7-layer) | conv3x3 16x16x3 → maxpool → dw → conv1x1 → dw → conv1x1 → fc 1152→16 | `esp_nn_conv_s8` + pools + `esp_nn_depthwise_conv_s8` + `esp_nn_fully_connected_s8` | 994,782 / 994,782 |
| **Model C** (mv2real, 6-layer SAME/stride-2) | conv3x3 s2 SAME → dw SAME → conv1x1 → dw s2 SAME → conv1x1 → fc 2048→16 | same op set as B, with SAME/stride-2 params | 655,194 / 655,303 |

Each model also runs through a scalar C reference (TFLite semantics) so the
harness verifies on-device that the esp_nn optimized kernels are bit-exact
(`MATCH`/`DIFFER` verdict per model).

## Zoo-model head-to-head status

Plan `simd-zoo-hardening` todo 20: measure the same **real zoo models** the
Rust `model_bench` timers run, through ESP-NN, so a per-model head-to-head
table exists. Status per model (enumerated 2026-08-10):

| Zoo model | tflite (size) | ESP-NN C runner | On this board | Hematite row (todo 19, device) |
|---|---|---|---|---|
| KWS `kws_micro_speech_int8` | `models/zoo/keyword_spotting/` (18.8 KB) | **(b) needs new harness code** — no runner in `main.c`; needs tflite weight extraction + layer wiring | feasible when toolchain returns (18.8 KB embed ≪ ~200 KB adapter flash ceiling; SRAM fine) | 13,091,330 / 13,091,344 cyc (54/54 ms) |
| sine `models/sine.tflite` | 656 B | **(b) needs new harness code** | feasible | 536 / 536 cyc |
| hello_world `hello_world_int8` | `models/zoo/sine_regression/` (2.7 KB) | **(b) needs new harness code** | feasible | 11,329 / 11,329 cyc |
| anomaly_detect `anomaly_detect_int8` | `models/zoo/anomaly_detect/` (277 KB) | **(b) needs new harness code** | risky: 277 KB weights embed is above the ~200 KB app-write ceiling this USB adapter sustained (T16 finding); needs a working flash path | 19,669,640 / 19,669,640 cyc (81/81 ms) |
| person_detect `person_detect_int8` | `models/zoo/person_detect_vww/` (333 KB) | **(c) not feasible on this board** | 333 KB weights + intermediate tensors exceed SRAM/DRAM budget; no PSRAM to spill into; also above the adapter flash ceiling. Hematite side is itself `SKIP reason=stack` | SKIP (reason=stack) |
| mobilenet_v2 `mobilenet_v2_1.0_224_int8` | `models/zoo/mobilenetv2_cls/` (3.98 MB) | **(c) impossible on this board** | board probe says `PSRAM: 0 bytes`; 3.98 MB of weights cannot be embedded or held in the 416 KB DRAM | SKIP (reason=no-psram) |

Key facts:

- **Only Model C / mv2real is a confirmed ESP-NN C baseline** (synthetic
  shape-language model, 655,303 cyc — already in the A/B/C table). None of
  the 6 real zoo models has a C runner today (`main.c` has no per-model
  runners to reuse; the A/B/C runners are hand-wired kernels with generated
  fill-pattern weights, not tflite-backed models).
- **Baseline-building is a separate scope decision** (plan Metis F8): writing
  a tflite weight-extraction pipeline + per-model esp-nn wiring is not
  invented here. Every un-built zoo row renders **`—`** with its reason —
  never a fabricated number.
- **The C side is currently unmeasurable on this host**: no ESP-IDF
  (`idf.py` not found, `$IDF_PATH` empty). The head-to-head table in
  `ESPRESSIF_VS_HEMATITE.md` therefore carries Hematite's measured cycles
  (todo-19 device run) with `—` C cells, exactly like todo 15 did for the
  per-op table.

## Todo 20 — zoo-model rows to add (when the ESP-IDF toolchain is back)

Add to `main/main.c` (patterns: the A/B/C runners + `run_bench`; the scalar
refs for the synthetic models are NOT reusable — zoo weights come from the
tflite files, not the fill pattern):

| Row to add | Model / inputs | How | Expected Hematite out_fnv (todo-19 device run, same ramp input) |
|---|---|---|---|
| KWS | `kws_micro_speech_int8.tflite`, input 1×1960 (ramp `fill_input_pattern`), output 4 | Extract weights + per-layer quant params from the tflite (the `tools/tflm-goldens` harness or `tools/generate_goldens` can dump them); wire the TinyConv layers as `esp_nn_conv_s8` + `esp_nn_depthwise_conv_s8` + `esp_nn_fully_connected_s8` (RESHAPE is free data movement — no esp-nn call). `run_bench` row like Model A. | `0x2131fda5` |
| sine | `sine.tflite`, input 1×1, output 1 | `esp_nn_fully_connected_s8` per dense layer (2 layers). | `0x040c5b8c` |
| hello_world | `hello_world_int8.tflite`, input 1×1, output 1 | same as sine. | `0xfaf3a2e1` |
| anomaly_detect | `anomaly_detect_int8.tflite`, input 640, output 640 | `esp_nn_conv_s8` (1×1) + `esp_nn_fully_connected_s8` chain; 277 KB embed — verify the app still flashes via the encrypted esptool path before trusting the row. | `0xe8f86342` |

person_detect and mobilenet_v2 have **no rows to add** — un-runnable on this
board for both stacks (person_detect: SRAM/DRAM budget; mobilenet_v2: no
PSRAM; board probe `PSRAM: 0 bytes`), they stay `—`/SKIP.

Same-conditions rule (Metis F8) for every added row: **identical tflite file,
identical input bytes (the same ramp `fill_input_pattern` the Rust
`model_bench` uses — NOT the golden INPUT_DATA), identical memory tier
(SRAM), identical CPU frequency (240 MHz; `CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ_240=y`
is already in `sdkconfig.defaults`), identical cache config.** A C row whose
output fnv does not match the Hematite out_fnv must be reported with the
discrepancy, never silently compared.

## Build + run on hardware

```sh
cd benchmarks/espnn-baseline
source ~/esp/esp-idf/export.sh
export IDF_TARGET=esp32s3
idf.py build
```

Merge + encrypted flash (the board's flash encryption is PERMANENT — all
writes MUST use `esptool write_flash --encrypt`, never plaintext/espflash):

```sh
~/.cargo/bin/espflash save-image --chip esp32s3 --flash-size 8mb --flash-mode dio \
  --flash-freq 80mhz build/espnn_baseline.elf /tmp/espnn.bin
/tmp/espenv/bin/esptool.py --chip esp32s3 --port /dev/cu.usbserial-10 --baud 115200 \
  write_flash --encrypt 0x0 /tmp/bl.bin 0x8000 /tmp/pt.bin 0x10000 /tmp/espnn.bin
```

Capture the UART report (115200 baud) verbatim into
`local-notes/evidence/simd-zoo-hardening/task-20-espnn-zoo.log`.

## Files

| File | Role |
|---|---|
| `CMakeLists.txt` | ESP-IDF project (`project(espnn_baseline)`); sources the main component only |
| `sdkconfig.defaults` | `CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ_240=y` — match the Rust firmware's 240 MHz benchmark clock |
| `main/CMakeLists.txt` | Registers `main.c` |
| `main/main.c` | Harness: Models A/B/C esp_nn + scalar-ref runners, `run_bench` (1 warm-up + 10 timed, min+median), sign-extending FNV-1a, UART report |
| `components/esp-nn/` | Vendored `espressif/esp-nn` v1.2.5 (never modified) |
| `.gitignore` | `build/`, `sdkconfig*`, `managed_components/`, `dependencies.lock` |

## Notes

- The vendored esp-nn source is **never modified** — only the harness
  (`main.c` / README) changes.
- Cycle counts are frequency-independent (CCOUNT is CPU clocks); `ms @ 240MHz`
  figures assume 240 MHz.
- ESP-IDF's toolchain is `xtensa-esp-elf` (GCC); the Rust side uses the espup
  `esp` toolchain (GCC 15.2.0). The two stacks never share compiled code —
  only model files, inputs, and the cycle protocol.
