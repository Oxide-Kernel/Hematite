# espdl-baseline — hardware C-SIMD baseline for the ESP32-S3 TIE728 kernels

Plain-C harness that calls the **vendored Espressif TIE728 SIMD assembly**
(the same `dl_tie728_s8_*.S` files `hematite-s3` inlines via `global_asm!`)
on **real hardware**, so the C-side result can be matched bit-for-bit
against the Rust `hematite-s3` SIMD kernels and their cycle counts compared.

It is the C side of the *"match output and cycles against C-SIMD"* work:
- **Output match**: FNV-1a checksums of the SIMD kernel output (and of the
  scalar reference) printed on device, byte-identical to the Rust firmware's
  `out_fnv(ref/s3)` report.
- **Cycle match**: CCOUNT deltas around the raw asm entry and around a C
  mirror of the Rust `conv2d_1x1` public API, vs the Rust firmware's
  `run_repeated` measurement.

> ⚠️ **Hardware measurements** (real ESP32-S3 @ 240 MHz, NOT QEMU). Requires
> the physical board and the encrypted-flash pipeline below.

## What it measures

One row: `conv1x1_s8 64x1x1x64` (the ember-esp-nn row — the only benchmark
row where the TIE728 SIMD gate fires: `input_c=64%16==0`,
`out_channels=64%16==0`, 16-aligned pointers, offsets 0, activation
`-128/127`, `mult=[1<<30;64]`, `shift=[0;64]`).

Fill pattern (identical to `hematite-benchmarks/src/spec.rs`):
`input[i]=(i*7+3)&0xFF`, `weights[i]=(i*13+11)&0xFF`, `bias[i]=i*17-8`,
`output=0`.

## Results (hardware, ESP32-S3 rev v0.2 @ 240 MHz)

| Measurement | Cycles (min/median) | FNV-1a checksum | Notes |
|---|---|---|---|
| C raw-asm (`dl_tie728_s8_conv2d_11cn` direct call) | **380 / 380** | `0x5eee898e` | Pure asm kernel; == Rust s3 SIMD output (bit-exact) |
| C full-API mirror (gate + Tie728ConvArgs build + call) | **1767 / 1767** | `0x5eee898e` | Mirrors Rust `conv2d_1x1` wrapper in C |
| Rust `hematite-s3` SIMD (bench9 firmware) | **2626 / 2628** | `0x5eee898e` | Full Rust public API incl. validation + dispatch |
| Rust `hematite-ref` scalar (bench9 firmware) | — | `0x0bea8225` | == C scalar-ref checksum (bit-exact) |
| C scalar-ref (same loop as `hematite-ref`) | — | `0x0bea8225` | == Rust ref checksum (bit-exact) |

Key facts established:

1. **Output match (bit-exact)**.  C-SIMD checksum `0x5eee898e` == Rust s3
   SIMD `0x5eee898e`; C scalar-ref `0x0bea8225` == Rust ref `0x0bea8225`.
   The same vendored asm fed the same data produces identical bytes on
   device, whether driven from C (ESP-IDF) or Rust (`global_asm!`).
2. **The FNV convention matters**: the Rust firmware's `fnv1a(&[i8])` does
   `h ^= b as u32`, which **sign-extends** negative bytes (e.g. `0x80` →
   `0xffffff80`).  The C harness must do the same (`h ^= (uint32_t)(int8_t)b`),
   not XOR raw `uint8_t` bytes, or the checksums diverge.
3. **Cycle gap is wrapper overhead, not kernel cost**.  The raw asm call is
   380 cycles (0.09 cyc/MAC for 4096 MACs — 16-wide TIE728).  Rust's
   measured 2628 cycles includes the full `conv2d_1x1` public API
   (slice-length validation, SIMD eligibility gate over 64 channels,
   `Tie728ConvArgs` construction, call).  A thin C mirror of that wrapper
   measures 1767 cycles; Rust's full implementation lands at 2628.  The
   *kernel itself* is identical between the two languages.
4. **SIMD output ≠ scalar reference** (filter-layout question, confirmed on
   hardware).  `0x5eee898e` vs `0x0bea8225`.  Both Rust and C feed the raw
   `[oc][ic]` weights straight to the asm; the asm indexes them as
   `[g][ic][lane]`, so the SIMD result is a different-but-deterministic
   layout transform of the scalar result.  This is a known property of the
   vendored ESP-DL asm calling convention, not a checksum bug.

## Files

| File | Role |
|---|---|
| `CMakeLists.txt` | ESP-IDF project (`project(espdl_baseline)`); sources the main component only |
| `sdkconfig.defaults` | `CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ_240=y` — match the Rust firmware's 240 MHz benchmark clock |
| `main/CMakeLists.txt` | Registers `main.c` + the two vendored asm files from `hematite-s3/src/asm/` (compiled with `-x assembler-with-cpp`) |
| `main/main.c` | Harness: fill_pattern, `Tie728ConvArgs` C struct, `conv2d_1x1_full` wrapper mirror, `scalar_conv1x1` ref, FNV-1a (sign-extending), CCOUNT timing (1 warm-up + 10 timed, min+median), UART report |
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

Capture the UART report (115200 baud) — see the board log in the repo for
the reference output.

## Notes

- The vendored asm files are NOT copied here — `main/CMakeLists.txt`
  references `hematite-s3/src/asm/` directly (single source of truth).
- ESP-IDF's toolchain is `xtensa-esp-elf` (GCC 14.2.0 via `install.sh
  esp32s3`); the Rust side uses the espup `esp` toolchain (GCC 15.2.0).
  The asm assembles identically under both.
- Cycle counts are frequency-independent (CCOUNT is CPU clocks); the
  `..us @240MHz` figures assume 240 MHz.
