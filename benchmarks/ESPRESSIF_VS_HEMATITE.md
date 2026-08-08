# Espressif NN Stack vs Hematite Stack

Real-hardware benchmark comparison (ESP32-S3 rev v0.2 @ 240 MHz) of the two
int8 inference stacks, driven by **two identical models run end-to-end on
both stacks**: a 4-layer CNN (Model A) and a MobileNetV2-style 7-layer
convnet (Model B).

- **Espressif NN stack** = the standard `espressif/esp-nn` library (v1.2.5,
  `CONFIG_NN_OPTIMIZED` → the esp32s3 hand-written asm kernels), wired to a C
  harness at `benchmarks/espnn-baseline`.
- **Hematite stack** = the Rust `hematite-s3` kernels — **100% bespoke from
  scratch** (Phase 16 op-sweep): S8-ACCX GPR-accumulator conv1x1/conv3x3/fc,
  a QACC per-lane depthwise kernel with a from-silicon 40-byte accumulator
  read-back, and a fused asm requantize epilogue; pool/elementwise run the
  vendored TIE728 asm — driven through the Rust public API by the
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

Weight layout matches on both sides (conv OHWI `[oc][kh][kw][ic]`, depthwise
HWCN `[ky][kx][oc]` with depth-multiplier 1, FC `[oc][ic]`). Both stacks run
each model 10× (after 1 warmup) and report min/median cycles and the FNV-1a
of the final 16-byte output.

## The headline — end-to-end model time

### Model A (4-layer)

| Stack | End-to-end cycles (min/med) | Speedup vs scalar | Output fnv1a | Bit-exact vs scalar? |
|---|---|---|---|---|
| **Hematite (bespoke ACCX + SIMD + asm requantize)** | **1,707,746 / 1,707,746** | 45.9× | `0x75eb32f5` | ✅ |
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
| **Hematite (bespoke ACCX + SIMD + asm requantize)** | **770,827 / 770,840** | 18.8× | `0x7f23eb05` | ✅ |
| ESP-NN optimized (esp32s3 asm) | 994,782 / 994,782 | 14.6× | `0x7f23eb05` | ✅ |
| Scalar reference (TFLite semantics) | 14,519,278 / 14,519,291 | 1.0× | `0x7f23eb05` | — |

**Both stacks are again bit-exact on every layer** (L1 `0x86d550e4`,
L2 `0x05a45f0e`, L3 `0xcf2e7213`, L4 `0x84f685fa`, L5 `0xd9648eb2`,
L6 `0x25f5a385`, final `0x7f23eb05`) — fair comparison. On this
MobileNetV2-style model **Hematite is ~1.29× faster than ESP-NN**.

Hematite's Model B journey (all bit-exact) shows the op-sweep covering the
previously-scalar layers: 9,970,333 (pre-depthwise, depthwise scalar + first
conv in_c=3 scalar) → 9,074,866 (Phase 2: bespoke depthwise SIMD) →
1,928,931 (Phase 3: first-conv in_c<16 zero-pad) → 914,239 (Phase 4: conv3x3
fast16/fast32) → 887,689 (Phase 5: fc hardware loop) → **770,827 (Phase 6:
requantize-in-asm)**.

The Model B gap is **coverage, not kernel speed**: L1's first conv has
`input_c = 3` (below the ACCX `%16` gate) and L3/L5 are depthwise
(`hematite-s3` depthwise is scalar-only by design), so those three layers run
scalar in Hematite while ESP-NN has dedicated s8 asm for them (`im2col` for
the 3-channel conv, `mult1_3x3_padded` depthwise). The ACCX-eligible layers
(L4/L6 conv1x1, L7 FC) run bit-exact SIMD.

### Layer breakdown (checksums only; both stacks identical)

| Model A layer | fnv1a | Model B layer | fnv1a |
|---|---|---|---|
| L1 conv3x3 out | `0xa18d9741` | L1 conv3x3 out | `0x86d550e4` |
| L2 maxpool out | `0xbd989bf4` | L2 maxpool out | `0x05a45f0e` |
| L3 conv1x1 out | `0xb26f62c3` | L3 depthwise out | `0xcf2e7213` |
| L4 FC out | `0x75eb32f5` | L4 conv1x1 out | `0x84f685fa` |
| | | L5 depthwise out | `0xd9648eb2` |
| | | L6 conv1x1 out | `0x25f5a385` |
| | | L7 FC out | `0x7f23eb05` |

## Why the win

The 16-phase op-sweep closed every coverage gap the original comparison
exposed and then pushed past ESP-NN:

- **Model A (4-layer):** L1 conv3x3 (input_c=16) now runs the **fast16**
  unrolled ACCX path (Phase 4); L4 FC benefits from the **hardware-loop**
  general path (Phase 5) and the **asm requantize epilogue** (Phase 6).
- **Model B (mv2mini):** L1's 3-channel first conv runs **zero-padded** SIMD
  (Phase 3); L3/L5 depthwise run the bespoke **QACC per-lane depthwise
  kernel** with a from-silicon 40-byte accumulator read-back (Phases 0–2);
  L7 FC gets the same fc/requantize wins as Model A's L4.

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
| conv1x1 64x1x1x64 | 472 | 4379/4392 | `0x5eee898e` | `0x0bea8225` | `0x0bea8225` |
| conv3x3 32x32 64x3x3x64 VALID ⚠️ | 2824 | 8868886/8868886 | `0xd1a9b601` | `0x0a181085` | `0x0a181085` |
| depthwise 3x3 7x7x32 | — | 91411/91411 (C bespoke) | — | `0xea4d8cb0` | `0xea4d8cb0` |
| FC 256→64 | 1288 | 8393/8394 | `0x16542aba` | `0x32e35185` | `0x32e35185` |
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

## Where to look

- ESP-NN model harness: `benchmarks/espnn-baseline/` (`main.c`, vendored
  `components/esp-nn/`).
- Hematite model runners: `hematite-benchmarks/src/model_cnn.rs` (Model A),
  `hematite-benchmarks/src/model_mv2.rs` (Model B); `bench_cnn_model` /
  `bench_mv2_model` in `firmware.rs`.
- C microbenchmark harness: `benchmarks/espdl-baseline/`.
- Kernels: `hematite-s3/src/asm/s8_accx_conv1x1.S`,
  `hematite-s3/src/asm/s8_accx_conv3x3.S`,
  `hematite-s3/src/asm/s8_accx_depthwise.S`,
  `hematite-s3/src/asm/s8_requantize.S` (all bespoke); `dl_tie728_s8*.S`
  (vendored Espressif, pool/elementwise only).
- Full engineering history: `PROJECT_LOG.md` Phases 9–16.
