# QEMU SIMD Validation — why the emulator "hangs" and how to run it

Ground-truth investigation into running the TIE728 SIMD kernels and the SIMD
correctness suite under the Espressif QEMU fork. Read this before trusting any
claim that "QEMU can't emulate the SIMD ops" — **it can, and we proved it.**
The hang you hit is QEMU's exception-path emulation, not the instruction set.

Sibling document: `QEMU_VALIDATION.md` (the 4-baseline cycle-count validation).

---

## TL;DR

| Question | Answer |
|---|---|
| Does QEMU have the TIE728 SIMD instruction set? | Yes — every opcode decodes and executes. |
| Do the SIMD ops produce correct results under QEMU? | **Yes** — all 9 tested ops (add, sub, max, mul-as, smul-as, + fused `.LD.INCP`/`.LD.IP` load forms, MAC16 reads) verified via isolated call0 probes. |
| Then why does the firmware hang at the SIMD CORRECTNESS header? | QEMU's esp32s3 model double-faults (infinite exception loop) when a **window overflow/underflow exception** is taken during the deep `call8` chain into a SIMD kernel. Also (pre-fix) `rur.fcr` (FPU control reg) hung it — QEMU does not emulate FPU user registers. |
| Is the `qemu` feature's SIMD gating correct? | It stays **required** as a QEMU workaround, but its rationale is **not** "the TIE ops are broken". It's a workaround for QEMU's window-exception path. |
| Would SIMD validation pass on a real ESP32-S3? | Almost certainly yes — window exceptions are handled there by the standard vectors (they're a normal, expected Xtensa event). |

---

## 1. The full investigation history

### 1.1 The documented claim

`PROJECT_LOG.md` and commit `6f82c35` claimed the TIE728 opcode
`EE.VSMULAS.S8.QACC.LD.INCP` (used by the conv1x1/conv3x3/gemm kernels) is
"decoded by Espressif's QEMU xtensa/esp32s3 fork but not correctly executed,
causing a silent infinite exception loop". A 7-round UART-print bisection
narrowed it to that opcode, and a
`qemu` Cargo feature was added to compile out the weighted-op SIMD paths under
QEMU. Elementwise/pool SIMD was left enabled but `simd_validation.rs`
documented the suite as "non-terminating on this QEMU build".

### 1.2 Static analysis — the helpers exist

`strings`/`nm` on `~/.esp-qemu/qemu/bin/qemu-system-xtensa`
(build `esp_develop_9.2.2_20260417`):
- All TIE728 opcode names are present (decoder knows them).
- Real execution helpers exist: `_helper_vadds_s3`, `_helper_vsubs_s3`,
  `_helper_vmax_s3`, `_helper_vmulas_qacc_s3`, `_helper_vsmulas_s3`, etc.
  `objdump` of `_helper_vadds_s3` shows a complete, correct ARM64 NEON
  implementation (saturating S8/S16/S32/S64 adds).

### 1.3 Isolated probes — every op works

A freestanding, call0-ABI probe harness was built under
`/var/folders/bt/n7jptqxx1sg7h3ms59nq_6lw0000gn/T/opencode/qemu_tie/`
(copies of `startup.S`/`appdesc.c`/`uart.c`/`uart.h`/`linker.ld` from
`benchmarks/qemu-baseline`, own `Makefile`, `tie.S` with the SIMD probes,
`main.c` driving them). All results read from the UART serial log:

| Probe | Instructions | Result |
|---|---|---|
| vadds | `vld.128.ip` ×2, `vadds.s8`, `vst.128.ip` | out = 2..17 ✓ |
| vsmulas | `zero.qacc`, `vsmulas.s8.qacc` ×3 (slide 0/1/2), `srcmb.s8.qacc` | all lanes = 6 ✓ |
| vsmulas ldincp | fused `vsmulas.s8.qacc.ld.incp` ×3 (filters F0/F1/F2) | all lanes = 6 ✓ (post-increment works) |
| vsubs | `vsubs.s8` | 0..15 ✓ |
| vmax | `vmax.s8` | 1..16 ✓ |
| vmax ldincp | `vmax.s8.ld.incp` + 2× `vmax.s8` | 1..16 ✓ |
| vmulas | `vmulas.s8.qacc` (2-operand) | all 1 ✓ |
| vmulas ldip | `vmulas.s8.qacc.ld.ip` ×3 | all 6 ✓ |
| MAC16 | `rsr.m0..m3` | reads fine ✓ |

**Conclusion: commit `6f82c35`'s "broad TIE728 emulation gap (5/5 checks
fail)" is not reproducible with a correct minimal probe. The SIMD instruction
set executes correctly under QEMU.**

---

## 2. The real causes of the hang

### 2.1 First bug found: `rur.fcr` (FPU control-register read) — FIXED

- `xtensa-lx-rt 0.22.0`'s exception `save_context` unconditionally saves the
  FPU registers (`rur a3, fcr` / `rur a3, fsr` / `ssi f0..f15`) when built
  with feature `float-save-restore` and `XCHAL_HAVE_FP`.
- `esp-hal 1.1.1` enables `float-save-restore` **by default**.
- `xtensa-lx-rt`'s `config/esp32s3.rs` claims `XCHAL_HAVE_FP = 1` even though
  the physical ESP32-S3 has **no FPU** — a known config quirk. Both gates
  therefore pass and `rur.fcr` lands in every exception handler.
- Isolated probe: `rur.fcr` alone **hangs QEMU** (output stops at "FCR only:").
- **Fix:** disable the `float-save-restore` feature (see §4). After the fix,
  `rur.fcr`/`rur.fsr`/`ssi f0` count in the firmware ELF dropped to **0**.

### 2.2 Second bug (the actual blocker): window-exception path double-fault

Even with `rur.fcr` removed, the SIMD correctness suite still hangs at the
same header. `-d int` trace shows an infinite
`cause 11 (window overflow) → cause 7 (window underflow) → cause 12 (double
exception) → ...` loop:

- The SIMD kernels are called via windowed `call8` (`clobber_abi("C")`, args
  in a10–a13) into `entry sp,128` / `retw` asm. The deep Rust call chain
  exhausts the 64-register window → **window overflow (cause 11)** — a normal,
  expected Xtensa event, handled by the standard `_WindowOverflow*` vectors.
- QEMU's esp32s3 model mishandles this path: the double-exception (cause 12)
  fires at the window vectors (`4037808f` `s32e a4,a0,-32`,
  `40378100`) and inside `save_context`'s window-spill `rotw 3/4` sequence
  (`40378fa6`/`40378fb2`), where a window **underflow (cause 7)** occurs while
  EXCM is already set. Result: `__default_naked_double_exception` re-enters
  `save_context` infinitely (a0 decrements by `0x100` per iteration, CCOUNT
  frozen).

So: **the TIE728 SIMD instructions are correct; QEMU's window
overflow/underflow exception emulation is broken** (specifically when the
exception is taken in a deep windowed context). `rur.fcr` was merely the first
double-faulting instruction in the old binary.

On real ESP32-S3 hardware neither problem exists: window exceptions are
handled by the standard vectors (this is core Xtensa functionality), and the
SIMD kernels run fine on-chip (proven by the hardware benchmark numbers).

---

## 3. What the `qemu` feature actually does

In `hematite-s3`, `#[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]`
compiles out the TIE728 weighted-op SIMD dispatch in:
`accx.rs` (191/258/376), `conv1x1.rs` (154/175/301/391),
`conv3x3.rs` (156/175/370/479), `depthwise.rs` (102/205/224/348),
`gemm.rs` (85/103/196/258), `activations.rs` (56/75).

Under `qemu`, those kernels run their **scalar fallback** — which is why every
`col1` value reads `Some(100)` (byte-identical to `ref`, by design). This is a
**workaround** for the window-exception hang (§2.2), *not* because the SIMD
ops fail. The rationale in older docs ("emulator bug in EE.VSMULAS…") should
be read as: "QEMU's window-exception path hangs; the TIE ops themselves are
correct (see this document)".

---

## 4. The `float-save-restore` config fix

**Files changed:**

- Workspace root `Cargo.toml:21` — *the* decisive edit:
  ```toml
  esp-hal = { version = "1.1", default-features = false }
  ```
  (`cargo metadata` showed `uses_default_features: False` only after this
  edit. A `default-features = false` set *only* in the member manifest did
  **not** take effect.)
- `hematite-benchmarks/Cargo.toml:47`:
  ```toml
  esp-hal = { workspace = true, default-features = false, features = ["esp32s3", "defmt", "unstable", "rt", "exception-handler"] }
  ```
  (explicitly re-enables everything except `float-save-restore`).

**Verify the fix:** rebuild and count FPU-register instructions:
```bash
objdump -d target/xtensa-esp32s3-none-elf/release/hematite-benchmarks \
  | grep -cE 'rur\.(fcr|fsr)|ssi f[0-9]'
# expect: 0
```

---

## 5. Building and running the QEMU validation

### Toolchain prerequisites (verified on this machine)

- Espressif QEMU fork: `~/.esp-qemu/qemu/bin/qemu-system-xtensa`
  (build `esp_develop_9.2.2_20260417`). **Not on PATH — use the full path.**
- `espflash` (≥ 4.5.0), on PATH.
- Xtensa GCC: `$HOME/.rustup/toolchains/esp/xtensa-esp-elf/esp-15.2.0_20250920/xtensa-esp-elf/bin`
  (also contains the assembler for the SIMD probes).
- Rust `esp` channel (rustup, pinned by `rust-toolchain.toml`).
- No ESP-IDF required for anything in this document (`espdl-baseline` /
  `espnn-baseline` are hardware-only and need it).

Always export the toolchain PATH first:
```bash
export PATH="$HOME/.rustup/toolchains/esp/xtensa-esp-elf/esp-15.2.0_20250920/xtensa-esp-elf/bin:$PATH"
```

### 5.1 Baseline 1 — scalar C reference kernels

```bash
cd benchmarks/qemu-baseline
make clean && make all          # gcc -O2 freestanding → baseline.elf → baseline.bin
qemu-system-xtensa -nographic -machine esp32s3 \
  -drive file=baseline.bin,if=mtd,format=raw \
  -monitor none -serial file:run1.log -icount 3 &
sleep 30 && kill -INT %1
```

Expect (documented + re-validated):
`conv_s8 8x8` min `0xb4007` csum `0xcc7be479`; `depthwise 18x18` min `0x8bb70`
csum `0x49f763d2`; `fc 271→3` min `0x979` csum `0xc87e9c19`;
`conv1x1 64x1x1x64` min `0x39c9`/`0x39ca` csum `0x272a7025`.

### 5.2 Baselines 3+4 — Rust firmware (release, scalar-hematite + s3)

```bash
cargo xtensa-build --release -p hematite-benchmarks --features qemu
espflash save-image --chip esp32s3 --merge \
  target/xtensa-esp32s3-none-elf/release/hematite-benchmarks \
  /tmp/hematite_release.bin
qemu-system-xtensa -nographic -machine esp32s3 \
  -drive file=/tmp/hematite_release.bin,if=mtd,format=raw \
  -monitor none -serial file:/tmp/rust_validate_release.log -icount 3 &
sleep 90 && kill -INT %1
```

Expect: Model A + Model B end-to-end rows (bit-exact fnv `0x75eb32f5` /
`0x7f23eb05`), then kernel rows `conv 1,045,880 / dw 744,236 / fc 3,498 /
conv1x1 25,982` with `col1=Some(100)` (scalar fallback under QEMU). The run
stops after the `conv1x1` row: the next SIMD rows either hang (window-exception
path) or are too slow in scalar fallback (see §2.2, §6).

### 5.3 Model + SIMD correctness validation

```bash
cargo xtensa-build --release -p hematite-benchmarks --features qemu,model-validation
espflash save-image --chip esp32s3 --merge \
  target/xtensa-esp32s3-none-elf/release/hematite-benchmarks \
  /tmp/hematite_modelval_rel.bin
qemu-system-xtensa -nographic -machine esp32s3 \
  -drive file=/tmp/hematite_modelval_rel.bin,if=mtd,format=raw \
  -monitor none -serial file:/tmp/rust_validate_modelval_rel.log -icount 3 &
sleep 70 && kill -INT %1
```

Expect: MODEL VALIDATION — `sine`, `hello_world_int8`, `kws_micro_speech_int8`,
`anomaly_detect_int8` **PASS**; `person_detect_int8` **FAIL** at idx 0
(got=127 want=120, fnv `0xef5a8bca` — matches documented hardware divergence);
`mobilenet_v2_1.0_224_int8` **SKIP** (needs PSRAM, absent under QEMU). Then the
`=== SIMD CORRECTNESS ... ===` header prints and the run **hangs** — this is
the window-exception path bug (§2.2), *not* the SIMD ops. The suite is meant
for hardware; it cannot complete under this QEMU build.

### 5.4 Isolated SIMD-op probes (the proof)

```bash
cd /var/folders/bt/n7jptqxx1sg7h3ms59nq_6lw0000gn/T/opencode/qemu_tie
make all     # needs PATH above; call0-ABI tie.S + main.c → probe.bin
qemu-system-xtensa -nographic -machine esp32s3 \
  -drive file=probe.bin,if=mtd,format=raw \
  -monitor none -serial file:serial.log -icount 3 &
sleep 14 && kill -INT %1
```

Expect every op to print its correct result (table in §1.3) and end with
`TIE probe done`. This is the ground truth that the TIE728 ops execute
correctly under QEMU.

---

## 6. What QEMU can and cannot validate (recap)

| | QEMU | Hardware |
|---|---|---|
| Scalar C kernels — cycles + csums | ✅ exact | ✅ |
| Scalar hematite (ref) — cycles + csums | ✅ exact (release) | ✅ |
| Model bit-exactness (both models) | ✅ exact fnv/csums | ✅ |
| TIE728 SIMD op correctness | ✅ exact (probes) | ✅ |
| SIMD **cycle counts** | ❌ not meaningful (scalar fallback / hangs / too slow) | ✅ measured |
| SIMD correctness suite | ❌ hangs (window-exception path) | ✅ |

The headline hardware SIMD wins (Model A 1,707,746 cyc = 45.9×, Model B
770,827 = 18.8×, espdl C-SIMD microbench) remain on-device measurements —
QEMU reproduces the *correctness* of the SIMD ops, not their *speed*.

---

## 7. Logs and evidence

- `benchmarks/qemu-baseline/run1.log` — documented scalar-C reference
- `benchmarks/qemu-baseline/run1.validated.log`, `validate_run1.log` — re-runs
- `benchmarks/qemu-baseline/rust_release.log`, `rust_run1.log` (debug),
  `rust_simd_*.log`, `simd_correctness.log`, `model_validation.log` — tracked
  firmware logs
- `/tmp/rust_validate_release.log`, `/tmp/rust_validate_modelval_rel.log` —
  re-runs from this investigation
- `/tmp/simd_repro_trace.log`, `/tmp/hematite_nofsr_trace.log` — `-d int`
  exception traces proving the window-exception double-fault loop
- Probe dir `/var/folders/bt/n7jptqxx1sg7h3ms59nq_6lw0000gn/T/opencode/qemu_tie/`
  with `tie.S`/`main.c`/`serial.log`
- `/tmp/serial_fcr.log` — `rur.fcr` hang proof
