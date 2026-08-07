# espdl-baseline — hardware C-SIMD baseline for the ESP32-S3 TIE728 kernels

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

## Files

| File | Role |
|---|---|
| `CMakeLists.txt` | ESP-IDF project (`project(espdl_baseline)`); sources the main component only |
| `sdkconfig.defaults` | `CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ_240=y` — match the Rust firmware's 240 MHz benchmark clock |
| `main/CMakeLists.txt` | Registers `main.c` + **all 8** vendored asm files from `hematite-s3/src/asm/` (compiled with `-x assembler-with-cpp`) |
| `main/main.c` | Harness: per-op fill_pattern + scalar refs, all `Tie728*Args` C structs, one SIMD wrapper + `run_bench` (1 warm-up + 10 timed, min+median) per op, sign-extending FNV-1a, UART report |
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
