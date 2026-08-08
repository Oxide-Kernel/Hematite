# Espressif NN Stack vs Hematite Stack

Real-hardware benchmark comparison (ESP32-S3 rev v0.2 @ 240 MHz) of the two
int8 inference stacks, operation by operation.

- **Espressif NN stack** = the vendored Espressif **TIE728 SIMD assembly**
  (`dl_tie728_s8_*.S`, from ESP-DL) called directly from a thin C wrapper —
  the same asm `hematite-s3` inlines via `global_asm!`. Measured by
  `benchmarks/espdl-baseline` on the physical board.
- **Hematite stack** = the Rust `hematite-s3` kernels: the **bespoke
  S8-ACCX GPR-accumulator kernels** for the weighted ops (bit-exact by
  construction), the vendored TIE728 asm for pool/elementwise, all driven
  through the Rust public API (validation + dispatch + requantize). Measured
  by the `hematite-benchmarks` firmware report.

Same fill pattern on both sides (`input i*7+3`, `weights i*13+11`,
`bias i*17-8`, `output 0`); same sign-extending FNV-1a checksums.

## The headline

| | Espressif NN stack | Hematite stack |
|---|---|---|
| Weighted ops (conv1x1, conv3x3, FC) | **fast but wrong**: QACC lanes saturate at 8-bit (16-bit for S16), so output diverges from the scalar reference for any realistic accumulator | **correct**: bit-exact vs the scalar reference (32-bit ACCX accumulation, full 16-bit products) |
| Pool ops | bit-exact vs **C-SIMD** (matches Hematite's) — differs from scalar `round_half_away_zero` by design | identical asm, same checksum |
| Elementwise (relu / add / sub / mul) | bit-exact | bit-exact (same asm) |
| Depthwise | **not available** (scalar-only) | **not available** (scalar-only) |
| Kernel speed | raw asm entry (no wrapper) | full public API (validation + dispatch + requantize) |

**One-line takeaway:** Espressif's TIE728 per-layer conv is *faster on raw
cycles but produces wrong output* — its 8-bit saturating QACC lanes cannot
represent a real int8 convolution (even one `127×127` product saturates).
Hematite's bespoke ACCX kernels are ~2.5–10x slower on the raw kernel but
**bit-exact**; for the elementwise ops the two stacks use the same asm and
are identical.

## Per-operation table

`Espressif raw` = vendored asm entry, thin C wrapper (kernel only).
`Hematite` = Rust public API, all cycles min/med from the `bench37` report.
Checksums: `ref` = scalar reference, both sides identical.

| Operation | Espressif raw cycles | Hematite cycles | Espressif checksum | Hematite checksum | ref checksum | Bit-exact vs scalar ref? |
|---|---|---|---|---|---|---|
| conv1x1 64x1x1x64 | 472 | **5041** / 5041 | `0x5eee898e` | `0x0bea8225` | `0x0bea8225` | ❌ Espressif / ✅ Hematite |
| conv3x3 32x32 64x3x3x64 VALID | 2824 ⚠️ | **9511784** / 9511785 | `0xd1a9b601` ⚠️ | `0x0a181085` | `0x0a181085` | ❌ / ✅ |
| FC 256→64 | 1288 | **15293** / 15294 | `0x16542aba` | `0x32e35185` | `0x32e35185` | ❌ / ✅ |
| max-pool 2x2, 32x32x16 | 1396 | **1892** / 1920 | `0x50d8f9c5` | `0x50d8f9c5` | `0x651bfdc5` | no (pool fixed-point semantics vs scalar `round_half_away_zero`; identical on both stacks) |
| avg-pool 2x2, 32x32x16 | 7181 | **7342** / 7343 | `0xdedd2dc5` | `0xdedd2dc5` | `0xb8a6ddc5` | no (same; identical on both stacks) |
| relu 256 | 175 | **357** / 358 | `0x6c620b3d` | `0x6c620b3d` | `0x6c620b3d` | ✅ both |
| add 256 | 167 | **410** / 438 | `0x14834bbb` | `0x14834bbb` | `0x14834bbb` | ✅ both |
| sub 256 | 265 | **490** / 518 | `0x62d74671` | `0x62d74671` | `0x62d74671` | ✅ both |
| mul 256 | 539 | **774** / 787 | `0xd3c0a7f1` | `0xd3c0a7f1` | `0xd3c0a7f1` | ✅ both |

⚠️ The Espressif conv3x3 entry computes **one output pixel per call**; the raw
number is for that single call and does not represent a full image pass.

## Correctness detail — why Espressif's weighted ops are wrong

On-device probes (`probe_qacc` / `probe_s8accx` / `probe_accx` in
`benchmarks/espdl-baseline`) established:

- `EE.VSMULAS.S8.QACC` saturates each of its 16 lanes at **8 bits** — a single
  `127×127` product reads back `0x7f`. 16/32/64 MACs of `127²` all return
  `0x7f`.
- `EE.VSMULAS.S16.QACC` saturates at **16 bits** (`0x7FFF` for 129032).
- The data is lost *inside* QACC — no post-processing epilogue can recover it.
- `EE.VMULAS.S8.ACCX` is a 16-wide element-wise **dot-product reduction** into
  a 32-bit GPR accumulator with **full 16-bit products** (`127×127=16129`
  preserved). `EE.SRS.ACCX gpr, 0, 0` extracts it exactly.

That is why the Espressif checksums diverge from the scalar reference on every
weighted op while Hematite's match bit-for-bit.

## Speed detail — what the cycle gap is

- Espressif numbers are the **raw asm entry** (args struct built on the stack,
  no validation, no requantize). Hematite numbers are the **full public API**
  (slice validation, SIMD eligibility gate, dispatch, 32-bit ACCX kernel,
  per-channel requantize in Rust).
- The Hematite *kernel* (PURE, no requantize) is much closer: conv1x1 996,
  conv3x3 7353504, FC 10334 cycles (from the C harness A/B rows). The gap to
  the public-API numbers is the Rust wrapper + requantize.
- The raw Espressif conv speed is real but unusable: it returns the wrong
  numbers, so the only meaningful speed baseline for correctness is the scalar
  reference, against which Hematite's ACCX kernels are ~14–66x faster.

## Where to look

- C harness: `benchmarks/espdl-baseline/` (README has the full per-op story,
  probes, arg layouts).
- Rust firmware + report: `hematite-benchmarks/` (`spec.rs` `kernel_specs()`,
  `firmware.rs` bench loop).
- Kernels: `hematite-s3/src/asm/s8_accx_conv1x1.S`,
  `hematite-s3/src/asm/s8_accx_conv3x3.S` (bespoke); `dl_tie728_s8*.S`
  (vendored Espressif).
- Full engineering history: `PROJECT_LOG.md` Phases 9–15.
