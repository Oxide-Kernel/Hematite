# qemu-runner — unified QEMU test runner for Hematite

`run_all.sh` builds BOTH QEMU-runnable benchmark suites of the Hematite repo,
merges each at a configurable flash size, runs each under the Espressif QEMU
fork with `-icount 3` (or free-running with `--no-icount`), captures serial
logs into `logs/`, and validates the output against the documented golden
checksums (see
[`benchmarks/QEMU_VALIDATION.md`](../../benchmarks/QEMU_VALIDATION.md)).
It prints a unified PASS/FAIL table and exits non-zero on any unexpected
failure.

| Suite | Source | Features | What it validates |
|---|---|---|---|
| **c-baseline** | `benchmarks/qemu-baseline` (freestanding C, `make`) | — | 4 kernel checksums + boot marker + completion marker |
| **rust(qemu)** | `hematite-benchmarks` firmware | `qemu` | 4 kernel fnv + Model A + Model B |
| **rust(qemu,model-validation)** | `hematite-benchmarks` firmware | `qemu,model-validation` (add `--models`) | same 6 + `MODEL VALIDATION DONE` marker; person_detect FAIL tolerated |

## Quickstart

```sh
# everything at the default 4mb flash size (C ~30s + Rust ~2 min of QEMU
# at icount 3; with --no-icount the full suite is done in seconds)
tools/qemu-runner/run_all.sh

# Rust only, 16mb, with emulated PSRAM
tools/qemu-runner/run_all.sh --skip-c --flash-size 16mb --psram

# fast mode: free-running, full C + Rust suites in ~1 min total (cycles
# become informational; checksums still validate bit-exact)
tools/qemu-runner/run_all.sh --no-icount

# add the model-validation suite (needs 8mb or 16mb — see below)
tools/qemu-runner/run_all.sh --models --flash-size 8mb

# reuse fresh artifacts (no rebuild when nothing changed)
tools/qemu-runner/run_all.sh --fast
```

Output ends in `ALL PASS (N suite(s))` with `exit 0`, or `FAILURES: ...` with
`exit 1`. Serial logs live in `tools/qemu-runner/logs/<timestamp>_<suite>_<size>.log`;
merged images in `tools/qemu-runner/images/`.

## Prerequisites

| Tool | Default location | Override with |
|---|---|---|
| Espressif QEMU fork (9.2.2, **not on PATH**) | `~/.esp-qemu/qemu/bin/qemu-system-xtensa` | `QEMU_BIN` |
| espflash 4.5.0 | `~/.cargo/bin/espflash` | `ESPFLASH` |
| Xtensa GCC (espup esp-15.2.0_20250920) | `~/.rustup/toolchains/esp/xtensa-esp-elf/esp-15.2.0_20250920/xtensa-esp-elf/bin/xtensa-esp32s3-elf-gcc` | `XT_GCC` / `ESP_TOOLCHAIN_DIR` |
| esp Rust toolchain | `rustup` channel `esp` (pinned by `rust-toolchain.toml`) + `~/export-esp.sh` | — |

The script checks all tools at startup and fails fast with a clear message.
`~/export-esp.sh` must exist for the Rust build (`cargo xtensa-build` needs
the esp fork on `PATH`).

## What the runner does (per suite)

1. **Build**
   - C: `make baseline.elf XT_GCC=...` in `benchmarks/qemu-baseline` (only the
     ELF — the sized merge is done below, not by the Makefile's `-S` target).
   - Rust: `export PATH=~/.cargo/bin:$PATH && source ~/export-esp.sh && cargo
     xtensa-build --release -p hematite-benchmarks --features qemu`
     (`qemu,model-validation` with `--models`).
2. **Merge at the configured flash size**
   `espflash save-image --chip esp32s3 --merge --flash-size <SIZE> <elf> <image>`
   — `--flash-size` pads the merged image to exactly 2/4/8/16 MB. `--skip-padding`
   is deliberately NOT used: QEMU rejects any image whose size does not map to
   one of its flash models (2MB→w25x16, 4MB→gd25q32, 8MB→gd25q64, 16MB→is25lp128).
3. **Run**
   `qemu-system-xtensa -nographic -machine esp32s3 [-m 8M] -drive file=IMG,if=mtd,format=raw
   -monitor none -serial file:LOG [-icount 3] &` then `sleep N; kill -INT`.
   `-icount 3` is the default (deterministic CCOUNT); `--no-icount` drops it
   for free-running (see "Speed" below).
   QEMU never exits on its own (the benchmark ends in a halt loop), so the
   runner kills it after the configured duration.
4. **Validate** by grepping the serial log (bit-exact checksums + completion
   marker; cycle minima are informational with a 5% tolerance and never gate
   the verdict).

## Golden checksums

### c-baseline (`benchmarks/QEMU_VALIDATION.md` §1)

| Check | Golden | Notes |
|---|---|---|
| boot marker | `boot_marker (.data copy check)=0x0badc0de` | proves `.data` copy |
| conv_s8 8x8 | checksum `0xcc7be479` | min `0x000b4007` (737,287) |
| depthwise 18x18 | checksum `0x49f763d2` | min `0x0008bb70` (572,272) |
| fc 271→3 | checksum `0xc87e9c19` | min `0x00000979` (2,425) |
| conv1x1 64x1x1x64 | checksum `0x272a7025` | min `0x000039c9` (14,793; ±1 cyc jitter documented) |
| completion | `=== benchmark complete - QEMU halt ===` | |

### rust(qemu) (`benchmarks/QEMU_VALIDATION.md` §3)

| Check | Golden | Notes |
|---|---|---|
| conv_s8 8x8 | fnv `0xa6d4f279` | release min 1,045,880 |
| depthwise 18x18 | fnv `0x56e836d2` | release min 744,236 |
| fc 271→3 | fnv `0x7d803f19` | release min 3,498 (drift to 3,521 documented) |
| conv1x1 64x1x1x64 | fnv `0x0bea8225` | release min 25,981 (drift to 26,030 documented) |
| Model A (cnn_model) | out_fnv `0x75eb32f5` (ref == s3) | matches hardware headline |
| Model B (mv2mini) | out_fnv `0x7f23eb05` (ref == s3) | matches hardware headline |

> **No completion marker is required for the plain Rust suite at `-icount 3`.**
> After the four ember-esp-nn rows the kernel loop continues into
> `conv3x3_s8 32x32` (SIMD row), which runs for **hours** in scalar fallback
> under QEMU at `-icount 3` (`QEMU_VALIDATION.md` §4) - the firmware's
> `benchmarks complete; reference bars:` line (firmware.rs:1556) is
> unreachable in a practical icount-3 run. The six bit-exact fnv values ARE
> the completion proof at icount 3. **This is NOT true free-running:**
> with `--no-icount` the conv3x3_s8 32x32 row completes in seconds and the
> whole suite (including the PSRAM rows under `--psram`) finishes in ~16s
> with the completion marker printed (`QEMU_VALIDATION.md` §8, verified
> 2026-08-11). The "hours" behavior is deterministic-CCOUNT mode, not the
> firmware.

### rust(qemu,model-validation) — `--models`

Validates the same 6 fnv values **plus** the `=== MODEL VALIDATION DONE ===`
marker. The zoo-model rows print `model X: PASS (fnv=...)`; one documented
divergence exists:

- `model person_detect_int8: FAIL at idx 0: got=127 want=120` — **matches
  real hardware** and is tolerated by default (`--tolerate-divergences` is
  auto-ON for this suite). Pass `--no-tolerate-divergences` to make it a hard
  failure.

> **Flash size:** the model-validation firmware (includes the
> mobilenet_v2_1.0_224 golden — a 150 KB input + 1000-way output) produces an
> app image of ~5.3 MB, which **exceeds the 4mb builtin factory partition**
> (4,128,768 B). Use `--flash-size 8mb` or `16mb`. The runner's merge step
> fails with a hint if the image does not fit. The old 4MB
> `model_validation.log` evidence predates the mobilenet golden.

## Flash-size capability

`--flash-size` accepts **2mb | 4mb | 8mb | 16mb** only (default 4mb). The
value is passed to `espflash save-image --flash-size` (which pads the merged
image to that byte size) **and** selects QEMU's flash model from the resulting
image size:

| Image size | QEMU flash model |
|---|---|
| 2 MB | w25x16 |
| 4 MB | gd25q32 |
| 8 MB | gd25q64 |
| 16 MB | is25lp128 |

Any other size → QEMU error (`drive file=...,if=mtd` rejects it), which is
why the runner never uses `--skip-padding`.

**16 MB capability is exercised**: `benchmarks/qemu-baseline/baseline_16mb.log`
(C, bit-exact goldens incl. `SPI Flash Size : 16MB` in the boot log) and
`/tmp/hematite_qemu_16mb.log` (Rust firmware, 16 MB — all 6 fnv goldens
bit-exact) are the recorded proofs.

## PSRAM (`--psram`)

`--psram` adds `-m 8M` to the QEMU command line, which attaches 8 MB of
**emulated PSRAM** (the `ssi_psram` device at SPI1 CS1; without `-m` the
machine still attaches it, sized at 2 MB).

Since **2026-08-11** the firmware's `qemu` feature no longer skips PSRAM
init: it runs a real esp-hal `Psram::new` probe. With `-m 8M` the firmware
logs `PSRAM probe: 8388608 bytes` (2,097,152 bytes without `-m`) and
populates the PSRAM bench arena (verified, all golden checksums bit-exact -
see `benchmarks/QEMU_VALIDATION.md` §7 for the probe/control/mv6 logs).

> **WARNING - do NOT combine `--psram` with `--models` for validation runs.**
> With the PSRAM arena populated, the mobilenet_v2 224x224 `[bench]` row
> actually executes (scalar, MANY minutes under QEMU) instead of SKIPping.
> It runs before the per-kernel rows whose fnv checksums the validator
> greps, so the model-validation suite can fail validation or require a much
> longer `--durations` third value. `--psram` on the plain Rust suite is
> safe: without the `model-validation` feature the model rows are bar-only
> and never execute.

## Flags

| Flag | Meaning |
|---|---|
| `--flash-size SIZE` | 2mb/4mb/8mb/16mb (default 4mb) |
| `--models` | also run the model-validation suite (features `qemu,model-validation`) |
| `--psram` | add `-m 8M` to QEMU |
| `--no-icount` | drop `-icount 3` (free-running; full suite in seconds). Cycle minima no longer match the documented goldens and always report drift; checksums still validate bit-exact. Default: icount 3 |
| `--skip-c` | skip the C baseline suite |
| `--skip-rust` | skip the plain Rust suite (MV suite still runs with `--models`) |
| `--durations C R MV` | QEMU run seconds: default `30 120 180` |
| `--tolerate-divergences` | allow documented-divergent rows (default: ON for MV, OFF otherwise) |
| `--no-tolerate-divergences` | strict validation everywhere |
| `--fast` | skip rebuilds when artifacts are up to date |
| `--no-rebuild` | never rebuild; fail if artifacts are missing |
| `--help` | usage |

### Durations

The documented values from the evidence runs are ~30s (C) / ~90s (Rust) /
~150s (MV) **at `-icount 3`**. Those land **at the edge**: the Rust suite
finishes its last printed row right around 90s, and a kill at exactly that
moment can miss nothing important (the six fnv are already printed). The
defaults add headroom (120s / 180s) so the run always captures the full
reachable output; shorten them with `--durations` when you only care about
the checksums.

**With `--no-icount` the durations are just a kill-after floor**: the C
suite completes in a few seconds and the entire plain Rust suite (including
the PSRAM rows under `--psram`) in ~16s, so the 30s/120s defaults are
generous. The one exception is `--no-icount` + `--psram` + `--models`: the
mobilenet_v2 224x224 bench row executes and takes minutes even free-running,
so keep a large third duration there.

### Speed: `--no-icount` vs `-icount 3`

`-icount 3` forces deterministic-CCOUNT mode, which is the dominant QEMU
slowdown (~30x+ wall-clock vs free-running); TCG options such as
`-accel tcg,thread=multi` and `-accel tcg,tb-size=2048` measured **zero**
effect (`QEMU_VALIDATION.md` §8). At icount 3 the plain Rust suite's
conv3x3_s8 32x32 SIMD row grinds for hours, so a full-suite run is
impractical; **free-running completes the whole plain suite in ~7-16s with
bit-exact fnv checksums** (checksums are icount-independent; only the cycle
minima depend on icount, and they match the documented goldens exclusively
at `-icount 3`). Recommended: `--no-icount` for any full-suite validation
run; keep icount 3 (the default) only when reproducing the documented cycle
goldens, where the plain Rust suite is limited to the six-fnv reachable
subset.

### `--fast` / `--no-rebuild`

- `--fast`: C is skipped when `make -q baseline.elf` says up to date. Rust is
  skipped when the release ELF exists, the feature stamp
  (`images/.rust_features`) matches the requested features, and no
  `.rs`/`Cargo.toml`/`Cargo.lock`/`build.rs` under the repo (excluding
  `target/`, `.git/`) is newer than the ELF.
- `--no-rebuild`: skips all build steps entirely and reuses existing
  artifacts; fails with a clear message if an ELF is missing or if the
  feature stamp does not match the requested feature set (`cargo
  xtensa-build` writes the SAME ELF path for `qemu` and
  `qemu,model-validation` — reusing the wrong build would silently run the
  wrong firmware). Merging still runs (it is fast and size-specific).

## Failure diagnosis

| Symptom | Likely cause / fix |
|---|---|
| `MISSING boot_marker` / all checksums missing | QEMU did not boot the image — check the log for the ESP-ROM banner; image/partition problem (see `save-image` hint) |
| Only the completion marker missing (C) | run cut short — increase `--durations` first value |
| Rust: all 6 fnv present, log ends at `conv1x1` | expected at `-icount 3` - the kernel loop continues into the hours-long SIMD conv3x3 row (see golden table note); free-running (`--no-icount`) reaches the completion marker instead |
| `MISSING MODEL VALIDATION DONE` | MV run cut short — increase `--durations` third value |
| `save-image failed ... image too big` | ELF exceeds the builtin factory partition for this flash size — use a bigger `--flash-size` |
| `MISSING ... fnv 0x...` | bit-exactness regression — compare against `benchmarks/QEMU_VALIDATION.md` and the tracked logs |
| QEMU: `terminating on signal 2` | normal — the runner's `kill -INT` |
| `--fast` rebuilds every time | `images/.rust_features` stamp does not match the requested feature string (or a source was touched) |

Cycle minima are reported as `ok` or `drift(kernel: got vs want)` and never
fail a row: the documented evidence itself shows small build-to-build drift
(fc 3,521 vs 3,498; conv1x1 26,030 vs 25,981) — checksums are the stable
validation key.

## Notes

- The runner never modifies anything outside `tools/qemu-runner/` (plus the
  C build's own object/ELF artifacts in `benchmarks/qemu-baseline`, which
  `make` requires).
- Logs are timestamped per run; the runner prints the `logs/` prefix at
  startup so you can find the exact files for a given run.
- Every suite runs with `-icount 3` for determinism (unless `--no-icount`):
  CCOUNT values under the emulator are reproducible and good for relative
  C-vs-Rust comparison, but they are **emulation smoke numbers, never
  hardware measurements** (`QEMU_VALIDATION.md` is explicit about this).
  Free-running cycles drift and are informational only; fnv checksums are
  bit-exact in both modes.
