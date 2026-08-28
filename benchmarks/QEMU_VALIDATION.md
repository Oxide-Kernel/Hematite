# QEMU Validation of the Four Benchmark Baselines

Verification of the benchmark numbers claimed across the repo by re-running
every QEMU-runnable baseline under the Espressif QEMU fork (`-icount 3`) and
comparing against the documented logs. The four baselines:

| # | Baseline | Harness | QEMU-runnable? |
|---|---|---|---|
| 1 | **Scalar C ops** | `benchmarks/qemu-baseline` (freestanding C, `-O2`) | ✅ yes |
| 2 | **SIMD C ops** | `benchmarks/espdl-baseline` (ESP-IDF C-SIMD asm) | ❌ hardware + ESP-IDF only |
| 3 | **Scalar hematite ops** | `hematite-benchmarks` firmware `ref` column | ✅ yes (release) |
| 4 | **SIMD hematite kernels** | `hematite-benchmarks` firmware `s3` column | ⚠️ falls back to scalar under QEMU |

> ⚠️ Every number in this document is an **emulated** Xtensa CCOUNT value
> under QEMU `-icount 3`, except where explicitly marked "hardware". QEMU
> reproduces *correctness* and *relative* scalar behavior. The TIE728 SIMD
> instructions themselves execute **correctly** under QEMU (proven by isolated
> probes — see `QEMU_SIMD_VALIDATION.md`); what QEMU gets wrong is the
> **window-overflow/underflow exception path** (and, before the
> `float-save-restore` config fix, `rur.fcr`), so SIMD kernels called through
> the deep windowed call chain hang the emulator. See §4 and
> `QEMU_SIMD_VALIDATION.md`.

---

## §1 Baseline 1 — Scalar C ops (QEMU)

Rebuilt `benchmarks/qemu-baseline` (`make clean && make all`), ran under QEMU
(`-icount 3`, serial → `validate_run1.log`, N=10 + 1 warm-up). Re-run vs
tracked `run1.log`:

| Kernel | run1.log min | re-run min | re-run median | csum | match |
|---|---|---|---|---|---|
| conv_s8 8x8 | 0xb4007 (737,287) | 0xb4007 (737,287) | 0xf1097 (986,263) | `0xcc7be479` | ✅ |
| depthwise 18x18 | 0x8bb70 (572,272) | 0x8bb70 (572,272) | 0x8bb70 | `0x49f763d2` | ✅ |
| fc 271→3 | 0x979 (2,425) | 0x979 (2,425) | 0x979 | `0xc87e9c19` | ✅ |
| conv1x1 64x1x1x64 | 0x39ca (14,794) | 0x39c9 (14,793) | 0x39ca | `0x272a7025` | ✅ (1-cyc jitter) |

Verdict: **exact**. The only delta is a 1-cycle conv1x1 *min* (14,793 vs
14,794) — same median, same checksum; that row's min is at the edge of the
1-cycle CCOUNT granularity. FNV-1a checksums identical on all four kernels.

Evidence: `benchmarks/qemu-baseline/validate_run1.log` (re-run), `run1.log`
(documented).

---

## §2 Baseline 2 — SIMD C ops (hardware only)

`benchmarks/espdl-baseline` is an ESP-IDF project (needs `~/esp/esp-idf`,
**not installed on this machine**) driving the bespoke C + TIE728 asm kernels
against a live ESP32-S3. It cannot be built or run here; its numbers are
**hardware claims, not QEMU-validatable**. From its README (hardware table,
min/med cycles):

| Op | C-SIMD cyc | note |
|---|---|---|
| conv1x1 64x1x1x64 | 472/472 | bespoke `s8_accx_conv1x1.S` |
| conv3x3 (1 px/call) | 2824/2824 | one pixel per call — NOT a full pass |
| fc 256→64 | 1288/1288 | |
| maxpool 2x2 | 1396 | |
| avgpool 2x2 | 7181 | |
| relu 256 | 175 | |
| add 256 | 167 | |
| sub 256 | 265 | |
| mul 256 | 539 | |

These are referenced in `ESPRESSIF_VS_HEMATITE.md` §"Per-operation reference
table" as the Espressif raw column. **Cannot be independently reproduced on
this machine** — flagged as unverified-under-QEMU, hardware-only.

---

## §3 Baseline 3 — Scalar hematite ops (QEMU)

`cargo xtensa-build --release -p hematite-benchmarks --features qemu`, flashed
`/tmp/hematite_release.bin`, ran under QEMU. The `ref` column times the scalar
`hematite-ref` kernels. Re-run vs tracked `rust_release.log`:

| Kernel | rust_release.log min | re-run min | re-run median | csum (ref/s3) | match |
|---|---|---|---|---|---|
| conv_s8 8x8 | 1,045,880 | 1,045,880 | 1,545,881 | `0xa6d4f279` | ✅ |
| depthwise 18x18 | 744,236 | 744,236 | 994,237 | `0x56e836d2` | ✅ |
| fc 271→3 | 3,498 | 3,498 | 3,499 | `0x7d803f19` | ✅ |
| conv1x1 64x1x1x64 | 25,981 | 25,982 | 25,983 | `0x0bea8225` | ✅ |

Verdict: **exact** on min cycles (conv1x1 +1 cycle, at granularity edge).
`prepared:` rows also bit-exact vs `s3` on every kernel. The `col1` column
(reported speedup ref-vs-s3) is `Some(100)`/`Some(83)`/`Some(74)` — QEMU
jitter on the *median*, min matches the documented values.

> **Debug profile note:** `rust_run1.log` (debug, opt-level=0: conv
> 22,720,033 / dw 16,057,231 / fc 55,483 / conv1x1 389,918) cannot be
> reproduced on the **current** firmware — post-merge `run_benchmarks` runs the
> end-to-end model benches (`bench_cnn_model`, `bench_mv2_model`) *before* the
> per-kernel rows, and at opt-level=0 under QEMU those benches take many
> minutes (abandoned after 90 s with only the report header printed). The
> documented debug numbers came from pre-merge firmware (old panic at
> `firmware.rs:373`, no model benches). The release numbers — the ones the
> repo's conclusions rely on — validate cleanly.

### End-to-end model rows (QEMU) — correctness only

Under QEMU both models run the **scalar fallback** (no SIMD in emulator), so
cycle counts are emulator-virtual and ~13–26× above hardware. What QEMU does
prove is **bit-exactness**:

- **Model A** (4-layer): final fnv1a `ref==s3==0x75eb32f5` — matches the
  hardware headline. Layer csums identical on both paths (L1 `0xa18d9741`,
  L2 `0xbd989bf4`, L3 `0xb26f62c3`). s3 min 22,808,079 cyc.
- **Model B** (mv2mini 7-layer): final fnv1a `ref==s3==0x7f23eb05` — matches
  hardware. All 6 layer csums identical (`0x86d550e4` … `0x25f5a385`). s3 min
  4,411,590 cyc.

Both match the hardware `ESPRESSIF_VS_HEMATITE.md` outputs exactly.

Evidence: `/tmp/rust_validate_release.log`, `/tmp/rust_validate_release_long.log`,
tracked `rust_release.log`.

---

## §4 Baseline 4 — SIMD hematite kernels (QEMU = scalar fallback)

Under the `qemu` feature, `hematite-s3` compiles out the TIE728 weighted-op
SIMD paths (`#[cfg(all(target_arch="xtensa", not(feature="qemu")))]` on the
asm dispatch in `accx.rs`, `conv1x1.rs`, `conv3x3.rs`, `depthwise.rs`,
`gemm.rs`) and the s3 kernels run their **scalar fallback**. That is why:

- `col1 = Some(100)` on every kernel row — s3 is byte-identical to ref, by
  design, and the QEMU run confirms it.
- The elementwise/pool SIMD rows (relu/add/mul/sub/maxpool/avgpool) remain
  enabled under QEMU, but `simd_validation.rs` documents them as
  **non-terminating on this QEMU build** — the first check hangs (confirmed:
  the model-validation build printed the SIMD-CORRECTNESS header and hung).
  Root cause: a window-overflow exception taken during the deep `call8` chain
  into a SIMD kernel double-faults QEMU's esp32s3 model (the window
  vectors / `save_context` window-spill path), producing the infinite
  exception loop — **not** the TIE728 instructions, which QEMU executes
  correctly (see `QEMU_SIMD_VALIDATION.md`). The firmware's own SIMD bench
  rows are therefore also unreachable under QEMU: the row *before* them,
  `conv3x3_s8 32x32,64x3x3x64 VALID (SIMD)`, runs its 33M-MAC kernel in scalar
  fallback (~300M+ cycles/run × 33 runs ≈ **hours of emulation** at QEMU's
  ~1.4M cycles/sec). A background run was killed after 400+ s still stuck on
  it.

So **no SIMD speedup number is QEMU-validatable**. The claimed hardware SIMD
numbers (`ESPRESSIF_VS_HEMATITE.md` Model A 1,707,746 cyc, Model B 770,827
cyc; `espdl-baseline` README microbench table) rest on on-device runs only,
and remain **hardware claims awaiting a device**.

---

## §5 Cross-baseline comparison (QEMU-validated subset)

Scalar-only, all `-icount 3`, same inputs/checksums:

| Kernel | C min (B1) | Rust ref min (B3) | Rust/C | comment |
|---|---|---|---|---|
| conv_s8 8x8 | 737,287 | 1,045,880 | 1.42× | Rust slower — scalar conv2d |
| depthwise 18x18 | 572,272 | 744,236 | 1.30× | |
| fc 271→3 | 2,425 | 3,498 | 1.44× | |
| conv1x1 64x1x1x64 | 14,793 | 25,982 | 1.76× | |

Matches the README release table (1.30–1.76×). Both are scalar algorithms; the
gap is compiler/loop-structure, and it is *within emulation* only.

The full 4-way comparison (adding B2 SIMD-C and B4 SIMD-hematite) is only
meaningful on hardware:

| Op | Scalar C (hw) | SIMD C (hw, B2) | Scalar hematite (hw) | SIMD hematite (hw, B4) |
|---|---|---|---|---|
| conv1x1 64×1×1×64 | (scalar ref) | 472 | — | 4379/4392 (Hematite public API, includes wrappers) |
| conv3x3 1px | (scalar ref) | 2824 | — | — (full-pass 8.87M cyc) |
| fc 256→64 | (scalar ref) | 1288 | — | 8393/8394 |
| Model A end-to-end | 78.27M | — | — | **1,707,746** |
| Model B end-to-end | 14.52M | — | — | **770,827** |

(B4 numbers are Hardware microbench from `ESPRESSIF_VS_HEMATITE.md`; B2 from
`espdl-baseline` README.)

---

## §6 Verdict

1. **Baseline 1 (Scalar C)** — VALIDATED under QEMU: exact cycle + checksum
   match (`validate_run1.log` vs `run1.log`).
2. **Baseline 3 (Scalar hematite, release)** — VALIDATED under QEMU: exact min
   cycles + bit-exact fnv on all 4 kernels and both end-to-end models.
   Debug-profile numbers are no longer reproducible on current firmware (model
   benches precede kernel rows; documented debug log was pre-merge).
3. **Baseline 2 (SIMD C)** — NOT QEMU-validatable (ESP-IDF + hardware only).
4. **Baseline 4 (SIMD hematite)** — NOT QEMU-validatable: weighted-op SIMD is
   compiled out under the `qemu` feature (scalar fallback, bit-exact by
   design); elementwise/pool SIMD hangs on this QEMU (window-exception path
   emulation bug — the TIE ops themselves are proven correct, see
   `QEMU_SIMD_VALIDATION.md`); the SIMD conv3x3 row is too slow to reach in
   scalar fallback. The claimed SIMD speedups are **hardware-only claims** and
   remain unverified on this machine.

Bottom line: everything the QEMU *can* validate validates cleanly (scalar C,
scalar hematite, bit-exactness of both models). The headline SIMD wins
(45.9×/18.8×, espdl C-SIMD microbench) are on-device measurements that QEMU
cannot reproduce and that need a real ESP32-S3 to confirm.

---

## §7 PSRAM emulation under QEMU — VERIFIED WORKING (2026-08-11)

**Answer: PSRAM is emulated AND usable under the Espressif QEMU fork.**
esp-hal 1.1.1 `Psram::new` initializes cleanly against QEMU's `ssi_psram`
device. The esp32s3 machine attaches `ssi_psram` (hw/misc/ssi_psram.c) at
SPI1 CS1 whenever the machine is created; `-m <size>` only SIZES it (the size
maps to the 2/4/8/16/32 MB density set). With no `-m` at all the device comes
up at 2 MB (density 0x0). In octal mode it answers vendor ID 0x0d (AP), MR0
read-latency code 2, and density 0x03 for 8 MB, exactly what esp-hal's octal
init expects, so esp-hal runs init straight through and skips timing tuning.

QEMU invocation (background + sleep + `kill -INT`):

```sh
~/.esp-qemu/qemu/bin/qemu-system-xtensa -nographic -machine esp32s3 -m 8M \
  -drive file=IMAGE.bin,if=mtd,format=raw -monitor none -serial file:LOG -icount 3
```

Evidence (probe artifacts in /tmp):

| Run | log | `-m` | PSRAM probe | Golden csums | Notes |
|---|---|---|---|---|---|
| plain qemu | `/tmp/hematite_psram_probe.log` | 8M | `8388608 bytes`, BEFORE the boot-profile line | conv_s8 `0xa6d4f279` · depthwise `0x56e836d2` · fc_s8 `0x7d803f19` · conv1x1 `0x0bea8225`, bit-exact | full octal init, MMU-mapped arena |
| control (no `-m`) | `/tmp/hematite_psram_control.log` | none | `2097152 bytes` (2 MB, density 0x0) | identical, bit-exact | machine ALWAYS attaches ssi_psram; `-m` only sizes it |
| qemu + model-validation | `/tmp/hematite_psram_mv6.log` | 8M | `8388608 bytes` | identical, bit-exact | arena re-asserted; mv2 row un-SKIPs and starts |

Findings:

- **PSRAM init does not perturb the flash/cache path.** All four golden
  checksums are bit-exact across every run (conv_s8 `0xa6d4f279`, depthwise
  `0x56e836d2`, fc_s8 `0x7d803f19`, conv1x1 `0x0bea8225`).
- **Model-validation rows unchanged vs the pre-probe baseline.** 4/6 PASS +
  1 divergent FAIL + 1 SKIP. anomaly FAIL idx1 got=41 want=42 (fnv
  `0xf2a76cd6`) is the known-divergent case; person_detect PASS (fnv
  `0x6962079d`).
- **mobilenet_v2 [bench] row un-SKIPs** (arena non-empty) and starts
  executing. Killed at ~320 s wall while still running: 224×224 scalar on
  QEMU is many-minutes slow, NOT a hang.

> ⚠️ **Caveat: pre-existing bug exposed.** The person_detect stack probe in
> `model_validation.rs` (`set_sp(&SRAM_ARENA + 0x40000)`) lands exactly on the
> linker-placed `PSRAM_ARENA.1` / `MAPPED_PSRAM_START/END` words and zeroes
> them (masked on real hardware where the arena is empty). Workaround: the
> firmware.rs qemu branch re-asserts the arena from probe-captured locals
> after the validation sections (`PSRAM arena re-asserted: 8388608 bytes`).
> Root-cause fix (probe SP target / linker layout) is a known open item.

**Upstream references:** espressif/qemu issue #129 (PSRAM works on esp-develop
builds; the IDF-shipped pre-built binary is older and does NOT work, so an
esp-develop build is required) and issue #139 (log shows full octal init plus
"Adding pool of 8192K"). Open draft PRs QEMU-298 (CS0_DIS/CS1_DIS defaults)
and QEMU-306 (ssi_psram address bounds) are worth watching for edge cases;
probe results were clean.

> **Note:** the `qemu` firmware feature previously SKIPPED PSRAM init
> entirely ("no PSRAM in emulator"). That assumption is now FALSE; the skip
> was replaced by a real probe (cfg-gated).

---

## §8 QEMU speed: root cause and fast mode (2026-08-11)

**Root cause of the "hours" behavior: `-icount 3` (deterministic-CCOUNT mode)
is the dominant slowdown, ~30x+ wall-clock vs free-running.** TCG options
(`-accel tcg,thread=multi`, `-accel tcg,tb-size=2048`) measured **zero**
effect. Dropping `-icount` entirely ("free-running") completes the whole
plain Rust suite in seconds with bit-exact fnv checksums (checksums are
icount-independent); only the CCOUNT cycle minima depend on icount, and they
match the documented goldens exclusively at `-icount 3` (the goldens were
captured there).

### TCG probe (2026-08-11) - `/tmp/probe_time_all.log`

Four QEMU variants, each running the complete plain Rust suite
(`qemu` feature, 4mb, no `-m`); all four finished in **7s** with identical
output. TCG tuning changes nothing.

| Variant | QEMU opts added | wall to PANIC(end) | rows | result |
|---|---|---|---|---|
| control | (none; free-running) | 7s | all SRAM rows | identical |
| thread_multi | `-accel tcg,thread=multi` | 7s | all SRAM rows | identical |
| tb2048 | `-accel tcg,tb-size=2048` | 7s | all SRAM rows | identical |
| multi_tb2048 | both | 7s | all SRAM rows | identical |

Landmarks (identical across variants): cnn_model 2s, mv2mini 2s, conv1x1 row
2s, conv3x3_s8 32x32 6s (the row that grinds for hours at icount 3),
softmax 7s, PANIC(end) 7s (row 65 "arena too small": no `-m 8M` so the
PSRAM arena is empty).

### The ~30x multiplier - `/tmp/hematite_qemu_16mb.log`

At `-icount 3` the emulator runs at roughly 1.4M cycles/sec of guest time
regardless of host speed (deterministic mode bounds the instruction rate);
free-running is bounded only by the host. A 110s icount-3 run reached only
row 4 (conv1x1) of the plain suite; the same ground at free-running takes
~2s. Measured ~30x+ end-to-end, which is why the conv3x3_s8 32x32 row
(~33M MACs x 33 runs in scalar fallback) grinds for hours at icount 3.

### icount shift controls CCOUNT - `/tmp/icount_*.log`

Varying the icount shift moves CCOUNT: `-icount none` → 45,160 cycles,
`icount=1` → 184,321, `icount=5` → 4,447,101 for the C conv row. Only
`-icount 3` reproduces the documented golden minima (C conv min
`0xb4007` = 737,287). So cycle minima are **informational** at free-running
(always drift), while checksums stay bit-exact.

### Full-suite verification run (free-running + `-m 8M`) - `/tmp/qemu_fullsuite_free.log`

Plain Rust suite (`qemu` feature, 8mb merged image, `-m 8M`, no icount),
polled every 2s, killed after completion. **The ENTIRE suite completed in
16s**, all 83 table rows printed (1 header + 3 model benches + 6 bar-only
registry rows + **all 73 kernel rows**), ending at the firmware's own
completion marker (firmware.rs:1556). With the PSRAM arena populated
(`PSRAM probe: 8388608 bytes`) the PSRAM-tier rows 65-68 execute instead of
the row-65 "arena too small" panic.

| Landmark | wall |
|---|---|
| PSRAM probe 8388608 bytes + cnn_model + mv2mini + mv2real benches | T+2s |
| softmax_s8 1x1000 (last SRAM kernel row) | T+6s |
| conv_s8 224x224x3->32 (PSRAM row 65) + depthwise_s8 112x112x32 (row 66) | T+10s |
| conv1x1_s8 112x112 (row 67) + conv1x1_s8 1280->1000 (row 68) + avg_pool 7x7x1280 (row 69) + mean rows 70-73 | T+16s |
| `benchmarks complete; reference bars:` (firmware.rs:1556) | T+16s |

All six runner-validated fnv values bit-exact (`0xa6d4f279`, `0x56e836d2`,
`0x7d803f19`, `0x0bea8225`, `0x75eb32f5`, `0x7f23eb05`); every row prints
`out_fnv(ref/s3)` with ref == s3 (conv 224x224 row: `0x6392575a`).
Zero FAIL rows, zero PANIC. The 224x224 row (43M MACs scalar) took ~4-6s,
not the 1-3 min predicted from the icount-3 rate.

### C baseline free-running - `/tmp/qemu_c_free.log`

C suite (benchmarks/qemu-baseline) free-running: boot marker + all 4
checksums + `=== benchmark complete - QEMU halt ===` all present; cycle
minima drift as expected (conv 45,600 vs 737,287 golden). PASS 6/6 via the
runner's `--no-icount` flag.

### Practical guidance

- `tools/qemu-runner/run_all.sh --no-icount` runs the full C + Rust suites
  in ~1 minute total (mostly the 30s C duration floor); at icount 3 the
  plain Rust suite alone is impractical (hours to reach the SIMD conv3x3
  row). Use `--no-icount` for full-suite validation; use icount 3 (default)
  only when reproducing the documented cycle goldens.
- `-smp 1` breaks boot (the ESP-IDF multicore bootloader requires both
  CPUs); speed must come from dropping `-icount`, not from SMP config.
- QEMU never exits on its own: after the firmware's PANIC handler loops
  forever (and after normal completion the firmware halts in a loop), so
  runs must always be killed by PID after a timeout or after the completion
  marker appears.

---

## §9 Full zoo under QEMU + PSRAM, and the two hard emulator boundaries (2026-08-12)

With the PSRAM probe (§7) and the speed work (§8), the full benchmark zoo
is now runnable under QEMU except for two hard boundaries. Verified with
the `qemu,model-validation` firmware at `-m 8M`:

| Suite | Rows | Under QEMU | Evidence |
|---|---|---|---|
| C baseline (qemu-baseline) | 4 kernels + boot marker | ALL run, 6/6 PASS (icount 3 or free-running) | §1, §8 |
| Rust kernel zoo | 73 rows (ember, SIMD-class, pool, activations, fused, softmax, PSRAM-tier, mean) | ALL run; fnv ref==s3 bit-exact; PSRAM rows 65-69 execute with `-m 8M` | §3, §8, `/tmp/qemu_fullsuite_free.log` (83 rows, 16 s) |
| Model benches A/B/C | cnn_model, mv2mini, mv2real | ALL run; out_fnv bit-exact (`0x75eb32f5` / `0x7f23eb05` / `0x75eb32f5`), per-layer csums ref==s3 | `/tmp/hematite_qemu_guarded.log`, `/tmp/hematite_qemu_zoo_full.log` |
| Model validation (6 zoo models) | sine, hello_world, kws, anomaly, person_detect, mobilenet | 5 run (sine/hello/kws/person PASS, anomaly ±1 documented divergence); mobilenet SKIP reason=drom-map-cap (see below) | `/tmp/hematite_qemu_guarded.log` |
| SIMD C ops (espdl-baseline, B2) | 9 rows + 4 todo-15 rows | NOT runnable — ESP-IDF + real hardware only | §2 |
| Rust SIMD 40-check suite | simd_validation::validate_all | NOT runnable — cfg-gated `not(feature = "qemu")`; TIE728 window-exception emulation gap | §4, QEMU_SIMD_VALIDATION.md |

### Boundary 1: SIMD (TIE728) — window-exception double-fault

All SIMD dispatch is qemu-gated to scalar fallback (§4). The TIE728
instructions themselves execute correctly under QEMU in isolation; the
hang is QEMU's window-overflow/underflow double-fault in deep call8 chains
(QEMU_SIMD_VALIDATION.md). Both SIMD suites (C + Rust) are hardware-only
claims.

### Boundary 2: drom map caps at 8 MiB — mobilenet validation is hardware-only

Espressif's QEMU esp32s3 model maps drom (flash-mapped `.rodata`) only up
to `0x3C800000` (8 MiB). The mobilenet VALIDATION rows reference a second
copy of the full weights (the `[bench]` runner already links its own copy);
once both are retained the drom grows to 8,731,444 bytes and the model
dispatch tables land past the map, where reads return garbage and the run
dies with `Guru Meditation Error: Core 1 panic'ed (IllegalInstruction)`
before any model row (PC in a data-region address; identical at icount 3 —
a layout issue, not timing — documented in the QEMU validation notes).

Fix in place (2026-08-12): the qemu arms of `validate_mobilenet` /
`validate_s3_mobilenet` now log an honest `SKIP reason=drom-map-cap` and
must NOT instantiate the model (a reference-free arm lets the linker strip
the duplicate weights — drom back to 5,153,892 B, app 5,327,472 B fits the
8 MB partition). The hardware (`not(qemu)`) arms keep the full
run-on-PSRAM-stack un-SKIP implementation — real S3 MMUs have 512 entries,
no 8 MiB cap — so mobilenet VALIDATION rows are **hardware-only** until the
drom is shrunk (weight compression with runtime decompression into PSRAM is
the only real lever) or the QEMU fork lifts the map cap.

Remaining QEMU limitation: the mobilenet `[bench]` row (full 224x224
scalar inference on the PSRAM arena) executes but takes many minutes even
free-running — it runs after the model benches and blocks the kernel rows
that follow, so a validation run should kill QEMU after the model bench
section (`mv2real per-layer` output) rather than waiting for completion.
The kernel zoo and all other rows are unaffected (they precede or are
independent of it).

