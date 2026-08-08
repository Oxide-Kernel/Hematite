# espdl-baseline — hardware C-SIMD baseline for the ESP32-S3 TIE728 kernels

> 📊 **TL;DR comparison:** see
> [`benchmarks/ESPRESSIF_VS_HEMATITE.md`](../ESPRESSIF_VS_HEMATITE.md) — the
> one-page "Espressif NN Stack vs Hematite Stack" summary, operation by
> operation.

Plain-C harness that calls the **vendored Espressif TIE728 SIMD assembly**
(the same `dl_tie728_s8_*.S` files `hematite-s3` inlines via `global_asm!`)
on **real hardware**, so the C-side result can be matched bit-for-bit
against the Rust `hematite-s3` SIMD kernels and their cycle counts compared.

It is the C side of the *"match output and cycles against C-SIMD"* work:
- **Output match**: FNV-1a checksums of the SIMD kernel output (and of the
  scalar reference) printed on device, byte-identical to the Rust firmware's
  `out_fnv(ref/s3)` report.
- **Cycle match**: CCOUNT deltas around a thin C wrapper that builds the
  asm args struct and calls the entry, vs the Rust firmware's
  `run_repeated` measurement of the full public API.

> ⚠️ **Hardware measurements** (real ESP32-S3 @ 240 MHz, NOT QEMU). Requires
> the physical board and the encrypted-flash pipeline below.

## What it measures

**All nine SIMD-capable operations** — one row each, sized so every TIE728
SIMD eligibility gate fires (16-byte aligned pointers, channels/dims
multiples of 16, offsets 0, identity requantize `mult=1<<30`):
conv1x1 (`_11cn`), conv3x3 (`_33cn`, VALID/pad-0), fully-connected (`_11cn`),
max-pool 2x2 (`_22c1`), avg-pool 2x2 (`_22c1`), ReLU (`_relu_11c`), and
add / sub / mul elementwise (`_w1_16_w2_16`). These are exactly the rows
added to the Rust firmware as `SIMD_*` specs in `kernel_specs()`
(`hematite-benchmarks/src/spec.rs`), so each row has a Rust
`out_fnv(ref/s3)` counterpart from the `bench10` device report.

Depthwise conv is **scalar-only by design** in `hematite-s3` (no SIMD
entry exists) — it is not benchmarked here.

Fill pattern (identical to `hematite-benchmarks/src/spec.rs`):
`input[i]=(i*7+3)&0xFF`, `weights[i]=(i*13+11)&0xFF`, `bias[i]=i*17-8`,
`output=0`.

## Results (hardware, ESP32-S3 rev v0.2 @ 240 MHz)

`C-SIMD` = raw vendored-asm entry called from a thin C wrapper (args struct
built on the stack each call). `Rust s3` = `hematite-s3` public kernel
(`bench10` firmware report). All FNV-1a checksums are sign-extending, the
same convention as the Rust firmware.

| Operation (row) | C-SIMD cycles min/med | Rust s3 cycles min/med | C-SIMD checksum | Rust s3 checksum | Bit-exact? | Scalar checksum (C == Rust) | SIMD == scalar? |
|---|---|---|---|---|---|---|---|
| conv1x1 64x1x1x64 | 472 / 472 | 2627 / 2628 | `0x5eee898e` | `0x5eee898e` | ✅ | `0x0bea8225` | no (filter layout) |
| conv3x3 32x32 64x3x3x64 VALID | 2824 / 2824 | 4849 / 4876 | `0xd1a9b601` | `0xd1a9b601` | ✅ | `0x0a181085` | no (filter layout) |
| fc 256→64 | 1288 / 1288 | 3187 / 3214 | `0x16542aba` | `0x16542aba` | ✅ | `0x32e35185` | no (filter layout) |
| max-pool 2x2, 32x32x16 | 1396 / 1396 | 1978 / 1992 | `0x50d8f9c5` | `0x50d8f9c5` | ✅ | `0x651bfdc5` | no |
| avg-pool 2x2, 32x32x16 | 7181 / 7181 | 7378 / 7405 | `0xdedd2dc5` | `0xdedd2dc5` | ✅ | `0xb8a6ddc5` | no |
| relu 256 | 175 / 175 | 425 / 426 | `0x6c620b3d` | `0x6c620b3d` | ✅ | `0x6c620b3d` | yes |
| add 256 | 167 / 167 | 467 / 481 | `0x14834bbb` | `0x14834bbb` | ✅ | `0x14834bbb` | yes |
| sub 256 | 265 / 265 | 547 / 574 | `0x62d74671` | `0x62d74671` | ✅ | `0x62d74671` | yes |
| mul 256 | 539 / 539 | 876 / 876 | `0xd3c0a7f1` | `0xd3c0a7f1` | ✅ | `0xd3c0a7f1` | yes |

Key facts established:

1. **Output match (bit-exact), for every operation.**  All 9 C-SIMD
   checksums equal the Rust s3 `out_fnv` from `bench10` — the same vendored
   asm fed the same data produces identical bytes whether driven from C
   (ESP-IDF) or Rust (`global_asm!`).  All 9 C scalar-refs equal the Rust
   `ref` checksums too.
2. **The FNV convention matters**: the Rust firmware's `fnv1a(&[i8])` does
   `h ^= b as u32`, which **sign-extends** negative bytes (e.g. `0x80` →
   `0xffffff80`).  The C harness must do the same (`h ^= (uint32_t)(int8_t)b`),
   not XOR raw `uint8_t` bytes, or the checksums diverge.
3. **Cycle gap is wrapper overhead, not kernel cost.**  For every row the
   raw asm (C) is substantially faster than Rust's measured number, which
   includes the full public API (slice-length validation, SIMD eligibility
   gate, `Tie728*Args` construction, dispatch).  E.g. conv1x1: 472 vs
   2627 cycles; relu: 175 vs 425.  The *kernel itself* is identical between
   the two languages.
4. **SIMD output ≠ scalar reference for the weighted/positional ops** —
   conv1x1, conv3x3, fc (filter-layout: the asm indexes weights as
   `[g][ic][lane]` instead of the scalar `[oc][ic]`), and **also for
   max-pool and avg-pool** (pooling semantics in the asm differ from the
   scalar ref — e.g. avg-pool's `shift/area_inv` fixed-point rounding vs
   `round_half_away_zero`).  The C-SIMD and Rust-s3 checksums agree with
   each other in all cases; only the *reference* differs.  These are
   deterministic properties of the vendored ESP-DL asm, not checksum bugs.
   ReLU / add / sub / mul are **bit-exact vs scalar** (elementwise identity
   contracts — and the ReLU match also validates the off-by-16 trip-count
   fix in `hematite-s3/src/activations.rs`, which reserves the asm's
   trailing 16-element block via `c_rs1_1=(c-16)/32`, `c_rs2_1=((c-16)%32)/16`).
5. **Cycle reference points** (per-op, C raw asm): conv1x1 472, conv3x3
   2824, fc 1288, max-pool 1396, avg-pool 7181, relu 175, add 167, sub 265,
   mul 539 cycles — the raw TIE728 kernel costs on this chip.

## Rust prepared-path vs C raw-asm (the wrapper-gap closure)

The Rust `hematite-s3` **prepared handles** (`PreparedConv1x1`, `PreparedFc`,
etc.) run the SIMD eligibility gate **once at construction** (`Prepared*::new`)
instead of on every call, then `run` only re-checks pointer alignment and
dispatches. `hematite-benchmarks` (`firmware.rs bench_kernel`) measures this
path (construct once outside the timed window) as a `prepared:` line next to
the public-API `s3` row. Results from the `bench11b` device report — every
prepared checksum is bit-exact equal to the public s3 checksum:

| Operation (row) | C raw-asm | Rust public s3 | Rust prepared | prepared ≈ C raw |
|---|---|---|---|---|
| conv1x1 64x1x1x64 | 472 | 2509 / 2521 | **669 / 669** | 1.42x |
| conv3x3 32x32 VALID | 2824 | 4662 / 4662 | **3075 / 3102** | 1.10x |
| fc 256→64 | 1288 | 3335 / 3335 | **1547 / 1574** | 1.22x |
| max-pool 2x2, 32x32x16 | 1396 | 1896 / 1896 | **1829 / 1856** | 1.33x |
| avg-pool 2x2, 32x32x16 | 7181 | 7361 / 7388 | **7274 / 7301** | 1.02x |
| relu 256 | 175 | 361 / 388 | **354 / 354** | 2.02x |
| add 256 | 167 | 414 / 441 | **363 / 363** | 2.17x |
| sub 256 | 265 | 494 / 494 | **442 / 442** | 1.67x |
| mul 256 | 539 | 777 / 781 | **743 / 743** | 1.38x |

The prepared path lands within ~1.0–2.2x of the raw asm — down from the
~2.5–5.5x gap of the legacy public API (e.g. conv1x1 went 2509 → 669 cycles).
Remaining overhead is the per-call pointer-alignment check plus the args
struct build.

Engineering notes from this work:

1. **MaybeUninit args build**: all `Tie728*Args` constructions now write only
   the fields the asm actually reads (via `core::mem::MaybeUninit` +
   byte-offset `write`s), eliminating the `memset`/dead-pad-store the struct
   literal with `..Default::default()` emitted.
2. **`#[inline(never)]` on `dispatch_fc`**: inlining the args-building
   dispatch into `fully_connected` caused the Xtensa backend to source args
   from wrong registers (a real miscompile — the fc row's checksum diverged
   until the dispatch was forced out-of-line). Kept as a standalone fn.
3. **ReLU off-by-16 fix validated on-device**: `c_rs1_1=(c-16)/32`,
   `c_rs2_1=((c-16)%32)/16` reserves the asm's trailing 16-element block;
   the relu row is bit-exact vs scalar (0x6c620b3d) on both the C and Rust
   SIMD paths.

## Bespoke ACCX kernels — bit-exact SIMD for the weighted ops

On-device probes (this harness's `probe_qacc`/`probe_s8accx`/`probe_accx`)
established that **no QACC-based accumulation can be bit-exact** for the
weighted ops: `EE.VSMULAS.S8.QACC` saturates each of its 16 lanes at 8 bits
(even one `127×127` product reads back `0x7f`), and `EE.VSMULAS.S16.QACC`
saturates at 16 bits — so the vendored `dl_tie728_s8_conv2d_*` per-layer
requantize is fundamentally inexact for any realistic accumulator.  All the
pre-ACCX `bit-exact` C-SIMD == Rust-s3 matches in the table above were the two
sides producing *the same* 8-bit-saturated garbage.

The fix is a small **bespoke GPR-accumulator kernel** (in
`hematite-s3/src/asm/s8_accx_conv1x1.S` / `s8_accx_conv3x3.S`, also used from
this harness):

* `EE.VMULAS.S8.ACCX` is a **16-wide element-wise dot-product reduction** into
  a 32-bit GPR accumulator with **full 16-bit products** (`127×127=16129`
  preserved, no lane saturation) — the exact bit-exact int8 conv primitive.
* It works on the **raw `[oc][ic]` weight layout directly** (element-wise
  `F[lane]·I[lane]`), so no `[g][ic][lane]` weight transform is needed.
* `EE.SRS.ACCX gpr, 0, 0` extracts the exact 32-bit accumulator; the
  TFLite requantize (`(acc·mult + round) >> total_shift`, clamp, saturating
  cast) runs in Rust — bit-exact vs the scalar reference by construction.

Device results (`bench34` firmware report, ESP32-S3 @ 240 MHz) — **SIMD now
equals the scalar reference bit-exact** on every weighted row, and also
matches the scalar reference where the vendored path could never match:

| Operation (row) | ref scalar cycles | s3 SIMD cycles | out_fnv(ref) | out_fnv(s3) | SIMD == scalar ref? |
|---|---|---|---|---|---|
| conv1x1 64x1x1x64 | — | 12595 / 12622 | `0x0bea8225` | `0x0bea8225` | ✅ |
| conv3x3 32x32 64x3x3x64 VALID | — | 35743534 / 35743561 | `0x0a181085` | `0x0a181085` | ✅ (full 30×30 image; the vendored path only ever wrote pixel (0,0)) |
| fc 256→64 | — | 20068 / 20081 | `0x32e35185` | `0x32e35185` | ✅ |
| max-pool 2x2, 32x32x16 | — | 1892 / 1920 | `0x651bfdc5` | `0x50d8f9c5` | no (pool fixed-point semantics vs scalar `round_half_away_zero` — same as the C-SIMD row) |
| avg-pool 2x2, 32x32x16 | — | 7342 / 7343 | `0xb8a6ddc5` | `0xdedd2dc5` | no (same; s3 == C-SIMD `0xdedd2dc5`) |

`conv1x1` went from 2627 cycles (vendored asm, wrong output) to 12595 cycles
(ACCX, bit-exact) — the ACCX path is ~2.5x the raw asm (element-wise
reduction is one MAC per lane per input element rather than the QACC
broadcast trick) but is *correct*.

Three Xtensa-LLVM backend miscompiles were found and fixed while wiring the
ACCX kernel into `hematite-s3` (each surfaces as a scrambled register at an
inline-asm or high-arg-count call site):

1. **`clobber_abi("C")` does not mark the caller's `a15` clobbered** across a
   `call8` — the kernel's `a7` (oc loop counter, = caller's `a15` after the
   window rotation) corrupted the caller's output pointer → `out.ptr=0x40`
   garbage. Fix: `out("a15") _` on the `accx_conv1x1` asm.
2. **`in("a12")` value is clobbered by the kernel's `a4` increment** (callee
   `a4` = caller `a12`) — the caller re-used the stale `a12` as the accs
   pointer. Fix: `inout("a12") acc_out => _`.
3. **`avg_pool_2d_simd_ctx`** (a `&mut`-ctx wrapper for the 8-arg vendored avg
   pool) is miscompiled both inlined (scrambled `or a10,a7,a7`) and
   out-of-line (the MaybeUninit args build's 16-byte array copy gets
   field-swapped, and `{args}`/`{target}` operands get clobbered by the
   template's `mov a10/a11`). Fix: build the args as a plain struct literal
   (no MaybeUninit pointer-cast writes) and pin every asm operand to an
   explicit register (`in("a10")`..`in("a13")`, `callx8 a13`).

## Optimized ACCX kernels (Phase 14 + 15)

The bespoke S8-ACCX kernels got a **fast path for `input_c == 64`** (`bench36`
device report, ESP32-S3 @ 240 MHz, all checksums still bit-exact), and the
Rust requantize epilogue got **uniform-scale fast paths** (`bench37`):

| Operation (row) | kernel path | C PURE cycles | Rust s3 full (bench36) | Rust s3 full (bench37) |
|---|---|---|---|---|
| conv1x1 64x1x1x64 | fast64 (input resident q0..q3, `loop a6`) | **996** | 9937 / 9939 | **5041 / 5041** |
| conv3x3 32x32 64x3x3x64 VALID | fast64 (9 taps unrolled, 8 VLD + 4 VMULAS per tap) | **7353504** | 15009334 / 15009361 | **9511784 / 9511785** |
| fc 256→64 | general path (in_c=256, 16 groups — no fast path) | 10334 | 20133 / 20159 | **15293 / 15294** |

- `conv1x1` kernel PURE cost dropped **3.4x** (3422 → 996 cyc); full-API
  (kernel + Rust requantize) went 12593 → 9939 (bench36) → **5041** (bench37).
- `conv3x3` full-API dropped **2.4x** (35743533 → **15009334** cyc) with the
  fast64 path, then a further **1.58x** (→ **9511784** cyc) from the requantize
  fast paths. The PURE kernel (7.35M cyc = 6.2M instructions over 900 px × 64
  oc × 9 taps) is near the issue-rate floor given the chip exposes only **8
  TIE728 Q registers** — the 36-vector 3×3 input window cannot stay resident,
  so each tap re-loads 4 input + 4 filter vectors per 4 MACs.
- Phase 15 (`hematite-s3/src/accx.rs`): `uniform_scale()` scans the per-channel
  arrays once (outside the pixel loop), and `requantize_1x1` takes
  `uniform_mult`/`uniform_shift` hints — identity `(1<<30,1)`, half-round
  `(1<<30,0)` → `(acc+1)>>1`, hoisted general-uniform, or per-channel with one
  upfront length assert + unchecked indexing instead of four per-iteration
  bounds checks. All fast paths verified bit-identical to the i64 reference
  (unit tests `requantize_fast_paths_match_reference`,
  `uniform_scale_detects_uniformity`).
- The two languages measure the **same kernel** (shared `.S` files); the
  remaining Rust-vs-C full gap is the Rust dispatch + the (now much lighter)
  requantize epilogue.

Engineering notes:

1. **`include_str!` staleness**: cargo does not rebuild a crate when a file
   pulled in via `include_str!` changes — `cargo clean -p hematite-s3` (or
   touching the source) is required, otherwise the firmware silently links the
   stale kernel.
2. **`.Lfast64` label collision**: both `.S` files define `.Lfast64`, and
   `global_asm!` concatenates both files into one assembly stream. Renamed the
   conv3x3 labels (`.Lc3fast64`/`.Lc3f64done`).
3. **Xtensa hardware `loop` under LLVM-MC**: the ~350-byte unrolled 9-tap body
   exceeds the `loop` fixup range (`loop fixup value out of range`) and the
   8-bit branch range (`bge` too far) — the conv3x3 fast path uses a
   short-branch + long-jump pattern (`blt …; j …`) instead. GNU-as (ESP-IDF)
   accepts both; LLVM-MC (Rust `global_asm!`) requires the branch version.

## Files

| File | Role |
|---|---|
| `CMakeLists.txt` | ESP-IDF project (`project(espdl_baseline)`); sources the main component only |
| `sdkconfig.defaults` | `CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ_240=y` — match the Rust firmware's 240 MHz benchmark clock |
| `main/CMakeLists.txt` | Registers `main.c` + **all 8** vendored asm files from `hematite-s3/src/asm/` (compiled with `-x assembler-with-cpp`) + the bespoke probes/kernels |
| `main/main.c` | Harness: per-op fill_pattern + scalar refs, all `Tie728*Args` C structs, one SIMD wrapper + `run_bench` (1 warm-up + 10 timed, min+median) per op, sign-extending FNV-1a, UART report |
| `main/probe_qacc.S` / `probe_s16.S` / `probe_s8accx.S` / `probe_accx.S` | On-device TIE728 primitive probes: QACC lane width, S16 QACC, S8-ACCX reduction, ACCX semantics |
| `main/s8_accx_conv1x1.S` | Bespoke 16-wide S8-ACCX dot-product conv1x1 kernel (bit-exact, raw `[oc][ic]` weights, fast64 path for `in_c==64`) — synced from `hematite-s3/src/asm/` |
| `main/s8_accx_conv1x1_orig.S` | Original branchy conv1x1 kernel (A/B baseline) |
| `main/s8_accx_conv3x3.S` | Bespoke S8-ACCX conv3x3 kernel (9-tap unrolled fast64, branch-loop) — synced from `hematite-s3/src/asm/` |
| `.gitignore` | `build/`, `sdkconfig*`, `managed_components/`, `dependencies.lock` |

## Build + run on hardware

```sh
cd benchmarks/espdl-baseline
source ~/esp/esp-idf/export.sh
export IDF_TARGET=esp32s3
idf.py build
```

Merge + encrypted flash (the board's flash encryption is permanent — all
writes MUST use `--encrypt`):

```sh
ESPTOOL=…/esptool-venv/bin/esptool      # venv containing esptool v5.3.1
~/.cargo/bin/espflash save-image --chip esp32s3 --merge \
  --flash-size 4mb build/espdl_baseline.elf /tmp/espdl.bin
$ESPTOOL --port /dev/cu.usbserial-1110 --baud 921600 \
  write-flash --encrypt 0x0 /tmp/espdl.bin
```

Capture the UART report (115200 baud) — the reference output is the
9-row table above.

## Notes

- The vendored asm files are NOT copied here — `main/CMakeLists.txt`
  references `hematite-s3/src/asm/` directly (single source of truth).
- ESP-IDF's toolchain is `xtensa-esp-elf` (GCC 14.2.0 via `install.sh
  esp32s3`); the Rust side uses the espup `esp` toolchain (GCC 15.2.0).
  The asm assembles identically under both.
- Cycle counts are frequency-independent (CCOUNT is CPU clocks); the
  `..us @240MHz` figures assume 240 MHz.
