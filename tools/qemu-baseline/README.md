# qemu-baseline — freestanding C benchmark for ESP32-S3 under QEMU

Plain-C int8 reference kernels for the four ember-esp-nn benchmark rows from
`hematite-benchmarks/src/spec.rs`, running **freestanding** (no ESP-IDF) on
the Espressif QEMU fork.  This is the C side of the C-vs-Rust performance
comparison: the Rust suite times its `hematite-ref` (scalar) and
`hematite-s3` (SIMD) kernels on hardware; this tool times the equivalent
plain-C scalar kernels in the same emulator the Rust firmware will later be
smoke-tested in.

> ⚠️ **QEMU emulation smoke — NOT hardware measurements.**  Every number
> printed by this binary is an emulated Xtensa CCOUNT value under QEMU
> `-icount 3`.  It is reproducible and useful for *relative* C-vs-Rust
> comparison inside the same emulator, but it is NOT a hardware cycle count
> and must never be presented as one.

## Files

| File | Role |
|---|---|
| `main.c`   | Benchmark harness: buffer fill, CCOUNT timing (mirror `timing.rs` run_repeated: 1 untimed warm-up + N≥10, min+median, integer ms@240MHz), UART0 report |
| `kernels.c`| Plain-C int8 kernels mirroring `hematite-ref` loop-for-loop: `conv2d` (3×3 and 1×1), `depthwise_conv2d`, `fully_connected`; i32 accumulate, CMSIS/TFLM single-rounding requantize |
| `appdesc.c`| ESP-IDF `esp_app_desc_t` placed at the start of segment 0 (required by the embedded IDF v5.5 bootloader, see below) |
| `uart.c`   | UART0 putc/puts/hex/dec driver (see register notes in the file header — the S3 base differs from the C3!) |
| `startup.S`| `_start` entry, call0 ABI, sets stack, zeroes .bss, calls `main` |
| `linker.ld`| Memory regions mirroring esp-hal 1.1.1 (irom/drom/dram windows) |
| `Makefile` | Build (gcc → espflash save-image --merge) + QEMU run targets |
| `run1.log` | Captured boot + benchmark output (evidence) |

## Toolchain (already installed, no ESP-IDF)

```sh
export PATH="$HOME/.cargo/bin:$PATH" && source ~/export-esp.sh >/dev/null 2>&1
XT_GCC=$(find ~/.rustup/toolchains/esp -name 'xtensa-esp32s3-elf-gcc' | head -1)
~/.cargo/bin/espflash --version            # 4.5.0
~/.esp-qemu/qemu/bin/qemu-system-xtensa --version   # 9.2.2 (esp_develop fork)
```

## Build

```sh
cd tools/qemu-baseline
make all
```

This runs:

```sh
# freestanding call0-ABI compile (no libc, no IDF):
$XT_GCC -O2 -mabi=call0 -mtext-section-literals -ffreestanding -nostdlib \
        -fno-builtin -Wall -Wextra -ffunction-sections -fdata-sections \
        -c startup.S appdesc.c uart.c kernels.c main.c

# link (no --relax; see linker.ld notes):
$XT_GCC -nostdlib -T linker.ld -Wl,--gc-sections \
        -Wl,--no-warn-rwx-segments -lgcc -o baseline.elf *.o

# merged flash image (bootloader @0x0, partition table @0x8000, app @0x10000):
~/.cargo/bin/espflash save-image --chip esp32s3 --merge baseline.elf baseline.bin -S
```

## Run

```sh
make run-log          # writes run1.log (kills QEMU after RUN_SECONDS=30)
# or manually:
~/.esp-qemu/qemu/bin/qemu-system-xtensa -nographic -machine esp32s3 \
  -drive file=baseline.bin,if=mtd,format=raw \
  -monitor none -serial file:run1.log -icount 3 &
sleep 30; kill -INT %1
```

Notes:
- `-nographic` + `-serial stdio` conflicts — use `-monitor none -serial file:run1.log`.
- The embedded bootloader is ESP-IDF **v5.5.1**; it boots the app only after
  the segment-0 `esp_app_desc` checks pass (efuse block revision, MMU page
  size).  `appdesc.c` satisfies them.
- QEMU never exits on its own; the benchmark ends in a halt loop
  (`0x42000048: j .`), so the driver kills QEMU after the run.

## Kernel rows (spec.rs ember-esp-nn shapes)

| Row | input | filter | output |
|---|---|---|---|
| conv_s8 8×8 | [1,8,8,3] | [64,3,3,3] | [1,8,8,64] |
| depthwise_conv_s8 18×18 | [1,18,18,16] | [1,3,3,16] | [1,18,18,16] |
| fc_s8 271→3 | 271 | 3×271 | 3 |
| conv1x1_s8 64×1×1×64 | [1,1,1,64] | [64,1,1,64] | [1,1,1,64] |

All: SAME padding (3×3/1×1), stride 1, dilation 1, input/output offset 0,
per-channel multiplier `1<<30`, shift 0, activation [-128,127] — identical
to the spec rows.  Buffers are filled with the spec's deterministic pattern;
each kernel is timed with `fill once → 1 untimed warm-up → 10 timed runs`
(min + median, integer `cycles*1000/240_000_000`), mirroring
`hematite-benchmarks/src/timing.rs`.

## Results (QEMU-EMULATION, `run1.log`, N=10 + 1 warm-up)

```
conv_s8 8x8, 64x3x3x3 (ember-esp-nn)
  min    cycles=0x000b4007  us@240MHz=3072  ms@240MHz=3
  median cycles=0x000f1097  us@240MHz=4113  ms@240MHz=4

depthwise_conv_s8 18x18, 1x3x3x16 (ember-esp-nn)
  min    cycles=0x0008bb70  us@240MHz=2384  ms@240MHz=2
  median cycles=0x0008bb70  us@240MHz=2384  ms@240MHz=2

fc_s8 271->3 (ember-esp-nn)
  min    cycles=0x00000979  us@240MHz=10    ms@240MHz=0
  median cycles=0x00000979  us@240MHz=10    ms@240MHz=0

conv1x1_s8 64x1x1x64 (ember-esp-nn 15.57x bar)
  min    cycles=0x000039ca  us@240MHz=61    ms@240MHz=0
  median cycles=0x000039ca  us@240MHz=61    ms@240MHz=0
```

Reproducibility: two consecutive runs gave identical min values and
identical FNV checksums for all four kernels; the depthwise *median*
occasionally jumps ~0x40000 cycles when a QEMU timer interrupt lands inside
one of the 10 timed runs (the min — the robust stat — is stable).

`boot_marker (.data copy check)=0x0badc0de` proves the bootloader copied
`.data` into DRAM correctly.

## Boot journey (what had to be fixed to get here)

1. **App descriptor missing** — the IDF v5.5 bootloader casts segment 0's
   data to `esp_app_desc_t` and read garbage → `Image requires efuse blk rev
   >= v303.18`.  Fixed with a real descriptor (`appdesc.c`) whose min/max
   efuse rev fields are 0.
2. **All-RAM layout aborts** — the S3 bootloader's `set_cache_and_start_app`
   expects DROM+IROM flash segments (it programs the flash MMU from them).
   An all-RAM app made it fault.  Fixed by moving `.text` to irom
   (0x42000020) and the descriptor to drom (0x3C000020) — the esp-hal layout.
3. **Empty DRAM segment** — a `.data` with no initialized content made
   espflash emit a bogus segment (vaddr 0, 64 KB); the app hung.  Fixed by
   keeping a real 4-byte `.data` (the `boot_marker` self-test).
4. **UART register map wrong** — the task's original `uart.c` spec
   (base 0x60013000, STATUS 0x18, TXFIFO_CNT [23:16]) is the ESP32-C3 map.
   The S3's UART0 is at **0x60000000**, STATUS at **0x1C**, TXFIFO_CNT at
   bits **[25:16]** (IDF v5.5 `esp32s3/register/soc/uart_reg.h` + QEMU
   `esp32s3_reg.h` agree).  Writes to 0x60013000 went to unmapped memory and
   produced zero output.
5. **DROM flash-cache reads land one 64 KB page off under QEMU** — the app
   read its own irom code bytes where the string literals should be.  Fixed
   by keeping *all* runtime-read data (`.rodata`) in DRAM and leaving only
   the app descriptor in drom (which the bootloader reads via direct flash
   reads, not the cache).

The remaining `segment 2: vaddr=00000000` in the boot log is espflash's
64 KB-alignment gap filler for the irom segment (zeros written to address 0,
benign; the bootloader prints it without a load/map tag).

## Rust firmware run (`rust_run1.log`) — QEMU-EMULATION

The Rust bench firmware (`hematite-benchmarks` ELF, `--features qemu`) boots
under the same Espressif QEMU fork with the same methodology (1 untimed
warm-up + N≥10 timed runs, min+median, integer `cycles*1000/240_000_000`).
Serial log: `rust_run1.log`.

```sh
cd <repo root>
export PATH="$HOME/.cargo/bin:$PATH" && source ~/export-esp.sh >/dev/null 2>&1
cargo xtensa-build -p hematite-benchmarks --features qemu   # debug profile
~/.cargo/bin/espflash save-image --chip esp32s3 --merge \
  target/xtensa-esp32s3-none-elf/debug/hematite-benchmarks tools/qemu-baseline/rust.bin
~/.esp-qemu/qemu/bin/qemu-system-xtensa -nographic -machine esp32s3 \
  -drive file=tools/qemu-baseline/rust.bin,if=mtd,format=raw \
  -monitor none -serial file:tools/qemu-baseline/rust_run1.log -icount 3 &
sleep 60; kill -INT %1
```

Captured output (min cycles):

```
| conv_s8 8x8,64x3x3x3 (ember-esp-nn)          | SRAM | 22720033/23720033 | ...
| depthwise_conv_s8 18x18,1x3x3x16 (ember-esp-nn) | SRAM | 16057231/16057231 | ...
| fc_s8 271row,3out (ember-esp-nn)             | SRAM | 55483/55483 | ...
| conv1x1_s8 64x1x1x64 (ember-esp-nn 15.57x bar) | SRAM | 389918/389919 | ...
PANIC: panicked at hematite-benchmarks/src/firmware.rs:373:19:
arena too small for spec 'conv_s8 224x224x3->32 (ESP-DL/MobileNetV2 first layer)'
```

The firmware boots, runs all four SRAM rows, then panics honestly on the
first PSRAM row (QEMU has no PSRAM; the arena is empty). That panic is the
expected no-PSRAM behavior, not a defect.

Determinism: three consecutive runs gave byte-identical min values for all
four kernels.

### QEMU-EMULATION comparison — C (-O2) vs Rust (debug)

⚠️ **QEMU emulation smoke — NOT hardware measurements.** The Rust firmware
was built in the **debug profile (`opt-level=0`)**; the C baseline is `-O2`.
The absolute gap is dominated by the optimization-level difference, not by
language. Use this table for *relative* C-vs-Rust behavior under identical
emulation only; the true hardware comparison requires a release Rust build.

| Kernel | C min cycles (-O2) | Rust min cycles (debug) | Rust/C | Delta % |
|---|---|---|---|---|
| conv_s8 8x8, 64×3×3×3 | 737,287 | 22,720,033 | 30.8× | +2981.6% |
| depthwise_conv_s8 18×18, 1×3×3×16 | 572,272 | 16,057,231 | 28.1× | +2705.9% |
| fc_s8 271→3 | 2,425 | 55,483 | 22.9× | +2188.0% |
| conv1x1_s8 64×1×1×64 | 14,794 | 389,918 | 26.4× | +2535.6% |

Notes:
- Both columns are CCOUNT min cycles under `-icount 3` on the same QEMU fork.
- The Rust `col1` (scalar-ref vs s3) reports `Some(100)` = 1.00×: under QEMU
  the s3 kernels run the scalar fallback (no SIMD in the emulator), so the
  two paths are the same code — expected, not a measurement error.
- Wall-ms column is 0 under the `qemu` feature (esp-hal systimer is not
  initialized; CCOUNT columns are unaffected).
- The Rust run bypasses the CCOUNT-calibration guardrail and PSRAM init via
  the documented `qemu` feature (see hematite-benchmarks/Cargo.toml).

## Boot journey — Rust firmware (what had to be fixed)

The Rust firmware needed the same boot-descriptor fix as the C baseline plus
two QEMU-specific firmware changes:

1. **`.flash.appdesc` app descriptor (256-byte espflash layout)** — esp-hal's
   linker script places `.flash.appdesc` first in DROM; espflash 4.5.0
   auto-detects that section and parses it as its 256-byte `AppDescriptor`
   (a different layout than the IDF struct the C baseline hand-rolled).
   Mismatched size → `save-image` panics. The Rust `EspAppDesc` in
   `firmware.rs` matches espflash exactly (see its doc comment for offsets).
2. **`esp_hal::init` hangs under QEMU** — its PLL reconfiguration polls
   `bbpll_cal_done`, which QEMU never asserts. The `qemu` feature bypasses
   `esp_hal::init` entirely (freestanding like the C baseline): CCOUNT, SRAM
   and UART0 work without it; PSRAM and the wall clock are unavailable.
3. **Link rooting** — with `esp_hal::init` bypassed, cargo would prune the
   esp-hal dependency (dropping the rt startup hooks xtensa-lx-rt's reset
   vector needs). Referencing `CpuClock::max()` (a const fn, no PLL) under
   `qemu` keeps esp-hal linked; `#[xtensa_lx_rt::entry]` on `main` roots the
   Reset→main chain. Full detail in learnings.md.
