---
title: How-To — Run Under QEMU
---

# How to run Hematite under QEMU

The Espressif QEMU fork can boot and run the Hematite firmware with
**correct results for the SIMD instruction set** — including a
documented QEMU-specific discovery: the emulator hangs on *exception
paths*, not on the TIE ops themselves.

## What works under QEMU

- The full **TIE728 SIMD instruction set** decodes and executes.
- All 9 tested ops (add, sub, max, mul-as, smul-as, fused load forms,
  MAC16 reads) are verified **correct** via isolated call0 probes.
- Model validation runs (sine / hello_world / kws bit-exact; person_detect
  **PASSES under QEMU** though it SKIPs on the device).
- Where the `qemu` feature compiles the weighted-op SIMD path out, the
  scalar fallback gives bit-identical output.

## The two QEMU gotchas (documented, worked around)

1. **Window exception hang (pre-fix):** QEMU's esp32s3 model double-faults
   (infinite exception loop) when a window overflow/underflow exception is
   taken during the deep `call8` chain into a SIMD kernel. The `qemu`
   Cargo feature gating stays **required** as a workaround — but its
   rationale is QEMU's exception path, **not** a broken instruction set.
2. **`rur.fcr` hang (fixed):** with esp-hal's default features,
   `xtensa-lx-rt`'s esp32s3 config falsely claims `XCHAL_HAVE_FP=1`, so
   `save_context` emits `rur.fcr`, which QEMU does not emulate
   (double-exception). The fix: `esp-hal` with `default-features = false`
   (see [installation](../installation.md)). This is why the workspace
   sets esp-hal's features explicitly.

## Running the firmware under QEMU

Use the runner harness:

```sh
cd tools/qemu-runner
./run_all.sh            # builds + runs baseline / rust / simd variants
```

Or manually, with the Espressif QEMU fork:

```sh
qemu-system-xtensa -machine esp32s3 \
  -drive file=path/to/rust.bin,if=mtd,format=raw \
  -icount 3 \
  -nographic        # or -serial mon:stdio for UART0 log lines
```

The firmware logs to UART0 (via the `qemu` feature's print path), so
`-nographic`/serial capture shows model validation + bench rows.

## Building the QEMU firmware

```sh
# the `qemu` feature is propagated by hematite-benchmarks:
cargo xtensa-build -p hematite-benchmarks --release --features qemu
```

The `qemu` feature:

- gates the weighted-op SIMD paths out (`cfg(all(target_arch = "xtensa",
  not(feature = "qemu")))`) so the ACCX kernels never run in the
  emulator,
- routes logging to UART0 (context-print path, no defmt-RTT),
- skips/bypasses the PLL calibration that hangs the emulator,
- probes the emulated PSRAM device (`-m 8M` attaches `ssi_psram` at
  SPI1 CS1) — a real esp-hal PSRAM init against the emulator.

## What QEMU numbers mean

QEMU rows are **labeled emulated** (e.g. the `-icount 3` HOST-EMULATED
ledger rows). They validate *correctness* and give a relative cycle
picture, but **silicon is the source of truth** for performance. The
docs/repo always separate the two.

## Reference documents

- `benchmarks/QEMU_VALIDATION.md` — 4-baseline cycle-count validation,
  PSRAM probe findings (8/16 MB images), the 
  `PSRAM arena re-asserted` mechanism
- `benchmarks/QEMU_SIMD_VALIDATION.md` — the full TIE728 bisection story
  and why the `qemu` gating exists
- `tools/qemu-runner/README.md` — the runner harness

## Related

- [Installation](../installation.md) — esp-hal feature flags
- [Methodology](../benchmarks/methodology.md) — emulated vs silicon labels