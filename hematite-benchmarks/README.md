# hematite-benchmarks — ESP32-S3 hardware benchmark suite

Plan **T5.3** (hardware benchmark suite) + **T5.3a** (benchmark-firmware
methodology guardrails, C3).  Per-kernel and model-level benchmarks for the
hematite NN engine on ESP32-S3, timed with the Xtensa **CCOUNT** cycle counter
at a locked 240 MHz.

> ⚠️ **No measurements exist on this host.**  This crate is the *deliverable*:
> benchmark definitions, the timing methodology, the report format and the
> device firmware guardrails.  Running it requires an ESP32-S3 and the
> esp-rs/rust fork toolchain.  The host binary only prints the report template
> and reference bars — it never fabricates numbers.

## Workspace status

- `cargo check --workspace` / `cargo test --workspace` are green on the host.
- The device firmware (`src/firmware.rs`, `cfg(target_arch = "xtensa")`) is
  NOT compiled on the host.  It is structurally reviewed and its guardrail
  logic is host-tested; the esp-hal calls carry `BRING-UP:` markers to
  validate on hardware.

## Running on hardware

```sh
# Normal build (watchdog stays ARMED — a hung benchmark resets the chip):
cargo xtensa-build -p hematite-benchmarks --release

# Bench-mode: disable the watchdog for long runs — REQUIRES the explicit flag:
RUSTFLAGS='--cfg bench_watchdog_disabled' cargo xtensa-build -p hematite-benchmarks --release

# Flash + monitor (probe-rs / espflash); results stream over RTT (defmt):
espflash flash --monitor target/xtensa-esp32s3-none-elf/release/hematite-benchmarks
```

The firmware refuses to run (panics with a clear message) if the boot profile
drifts from the locked configuration: **240 MHz CPU, QPI 80 MHz PSRAM, 64 KB
data cache / 64-byte line**, and if CCOUNT fails its calibration assert
(measured rate vs the independent wall clock, 1000 ppm tolerance, integer math
only).

## Reference bars (plan T5.3, B2 — single-core)

| Model | Bar | Source |
|-------|-----|--------|
| MobileNetV2 224×224 | **1294.5 ms** (single-core) | ESP-DL single-core reference. The **856 ms** figure is the DUAL-core number — never the bar. |
| KWS keyword_spotting_v1 (1×1960) | **7 ms** | edge-ml-model-zoo ESP-DL (plan T5.3). |

Per-kernel column-2 bar: beat ember-esp-nn's **15.57×** on `conv 1×1 64×1×1×64`
(plan T5.3 line 309).  Column-1 internal bar: conv SIMD ≥ **10×** vs our
scalar-Rust ref (T3.0).  ESP-DL's ANSI-C **26–77×** range is a C-vs-C number —
reported in column 3, never conflated with column 1.

## Report format

Every row is labeled **SRAM** or **PSRAM** (its working-set tier) and carries
three raw columns (min / median over **N ≥ 10** runs after **one untimed
warm-up**):

1. **CPU cycles** — CCOUNT deltas.
2. **ms @ 240 MHz** — integer `cycles * 1000 / 240_000_000`.
3. **wall-clock ms** — independent wall-clock deltas.

plus three speedup columns:

1. **vs scalar-Rust ref** — measured on device (the `hematite-ref` kernel of
   the same shape); T3.0 ≥ 10× bar.
2. **vs ember-esp-nn optimized-C** — needs the competitor's absolute cycle
   counts, sourced from its public benchmark tables at device bring-up;
   renders `—` until then.
3. **vs ESP-DL ANSI-C** — C-vs-C, reported separately.

**No comparison number is pre-filled.**  Columns 2/3 are `—` until sourced
(MUST-NOT-invent-numbers rule); only plan-attributed targets (15.57×, 10×,
1294.5 ms, 7 ms) appear in the docs and report footer.

## Per-kernel shapes (`src/spec.rs`)

ember-esp-nn shapes from plan T5.3: `conv_s8 8×8,64×3×3×3`,
`depthwise_conv_s8 18×18,1×3×3×16`, `fc_s8 271→3`,
`conv 1×1 64×1×1×64` (15.57× bar).  Plus ESP-DL / MobileNetV2-style rows
(first 224×224×3 conv, depthwise block, 1×1 projection, 1000-way head,
softmax, global average pool).

The bench entry points call the **exact public s3 free functions**
(`conv1x1::conv2d_1x1`, `conv3x3::conv2d_3x3`, `depthwise::depthwise_conv2d`,
`gemm::fully_connected`, `softmax::softmax`, `pool::average_pool_2d`) — no
invented ABI.  A host test runs every spec through both the s3 scalar path and
`hematite-ref` and asserts **bit-identical output**, proving the shapes are
valid and the calling convention is correct before any hardware is involved.

## Model-level benchmarks (`src/model_bench.rs`)

The model path is a **parameter**.  The registry carries MobileNetV2, KWS and
a sine smoke model with their documented bars; the harness is generic over
`ModelRunner`, whose shape mirrors the `#[model]`-generated `Model<B>` API
(`input_len` / `output_len` / `predict`).  The zoo `.tflite` files land with
T5.2 — wiring is then one `#[model("models/x.tflite")]` annotation plus a tiny
adapter implementing `ModelRunner`; the harness and report run unchanged.
Until then the firmware lists each model row with its bar and a
`NOT-WIRED (T5.2)` marker — no fabricated timings.

## Methodology guardrails (`src/guardrails.rs`, `src/firmware.rs` — C3)

- **Boot profile assert** — 240 MHz CPU / QPI 80 MHz PSRAM / 64 KB × 64 B
  cache; firmware panics on drift.
- **CCOUNT calibration assert** — measured rate must match 240 MHz within
  1000 ppm (integer math).
- **No f32 timing drift** — every cycle/ns → ms conversion is integer.
- **Stack canary + stack-depth budget** — canary verified after every run;
  SP-based depth check.
- **Watchdog policy** — disable only behind the explicit
  `--cfg bench_watchdog_disabled` flag; default keeps it armed.
- **Warm-up + N ≥ 10 + min/median** — enforced by `run_repeated` (floors N at
  10, never uses the first run as a data point).

## Host-side unit tests

`cargo test -p hematite-benchmarks` (29 tests) covers: shape validity +
bit-exact s3-vs-ref for every spec, arena carve alignment/disjointness, run
methodology (warm-up, N floor, min/median, CCOUNT wrap), integer timing math,
calibration tolerance, canary, stack depth, watchdog policy, report rendering
(three raw columns + tier labels + bars), model registry sanity, and bar
comparison.
