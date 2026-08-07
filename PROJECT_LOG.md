# Hematite — Project Log

A chronological technical record of what has been built, verified, and
discovered on Hematite so far. Hematite is a pure-Rust, `no_std`, int8
neural-network inference engine for the ESP32-S3 (Xtensa LX7 + TIE728 SIMD),
built bit-exact against TensorFlow Lite Micro semantics with zero runtime
allocation and zero C in the device build path.

This log complements `local-notes/plans/hematite-nn.md` (the working task plan,
kept out of git) and `local-notes/notepads/hematite-nn/{learnings,issues,problems}.md`
(the append-only engineering notepad). Those are the day-to-day scratch
record; this file is the durable, committed history.

---

## 1. Architecture

Eight workspace crates:

| Crate | Role |
|---|---|
| `hematite-core` | `KernelBackend` trait + op-parameter structs — the device-safe API contract every backend implements |
| `hematite-int8` | Fixed-point quantization primitives (`multiply_by_quantized_multiplier`, `quantize_multiplier`, requantize) |
| `hematite-memory` | USMP-style compile-time arena allocator — liveness-based buffer coalescing, zero runtime allocation |
| `hematite-ref` | Host-and-device scalar reference kernels + `RefBackend` (the golden-oracle implementation of `KernelBackend`) |
| `hematite-s3` | ESP32-S3 backend: scalar fallback kernels + `cfg(target_arch = "xtensa")`-gated TIE728 SIMD assembly glue (vendored from `espressif/esp-dl`, MIT license) |
| `hematite-codegen` | Proc-macro crate: hand-rolled TFLite flatbuffer parser (host-only, compile-time) + straight-line Rust code emitter (`#[model("x.tflite")]` → `Model<B>`) + fusion/arena/layout optimization passes |
| `hematite-tests` | Golden fixture corpus (`goldens/`) + per-op and per-model TDD test suite |
| `hematite-benchmarks` | ESP32-S3 hardware/QEMU benchmark firmware — per-kernel + model-level CCOUNT timing, methodology guardrails, model validation |

Plus `tools/generate_goldens/` (host-side fixture generator) and the
`benchmarks/` directory of C cross-language comparison baselines:
`benchmarks/qemu-baseline/` (freestanding C benchmark under QEMU) and
`benchmarks/espdl-baseline/` (ESP-IDF v5.5.1 C harness calling the same
vendored TIE728 SIMD assembly, for on-hardware output/cycle matching).

---

## 2. Phase-by-phase build history (commits `f982b06` → `e4c1e8f` → HEAD)

### Phase 0 — Workspace scaffold + CI
`a11e688` workspace manifest + crate stubs, xtensa target config.
`eb4800b` CI pipeline (host clippy/test + `cargo vet` supply-chain config).
`f09483b` golden fixture generator bootstrap — TFLM-faithful int8 reference
math, closes the Phase-2 TDD dependency gap.

### Phase 1 — Core trait + int8 math
`77bf449` `KernelBackend` trait (one method per in-scope op), all op
parameter structs, int8 quantization math primitives (CMSIS single-rounding
`MultiplyByQuantizedMultiplier`), USMP arena allocator (`hematite-memory`).

### Phase 2 — Scalar reference backend (golden oracle)
`f280a42` Conv2D scalar kernel — first kernel, validates the TDD pipeline
(fixture → failing test → implementation → bit-exact pass).
`3e2fb7d` depthwise_conv, fully_connected, pooling, activations, softmax,
elementwise, resize.
`3630fdc` data-movement (concat/split/pad/slice/transpose/reshape) +
reductions (mean/sum/argmax/argmin/l2_norm).
`687d5ff` recurrent ops — LSTM/SVDF/GRU, including hand-rolled fixed-point
sigmoid/tanh gate math (TFLM has no GRU kernel to reference at all; GRU
fixtures came from a from-scratch gemmlowp-style derivation).

### Phase 3 — ESP32-S3 SIMD backend
`95d80be` conv2d 1×1 — scalar fallback (bit-exact, host-tested) +
vendored TIE728 assembly SIMD backend (esp-dl @ `12c0616d`, MIT), cfg-gated
behind `target_arch = "xtensa"` so the host build never touches it.
`35c89ca` conv3x3/depthwise/gemm/elementwise/pool/softmax/activations/mean
— same scalar+SIMD-glue pattern across the remaining kernel families.

### Phase 4 — Codegen
`afe6946` TFLite flatbuffer parser — hand-rolled byte-offset walker (no
bindings crate), pinned IR API, vendored schema.
`775ea23` straight-line code emitter (17-op dispatch at the time) +
fusion/arena/layout optimization passes (fusion and arena were built and
unit-tested as standalone passes but never wired into the emitter — a
deliberate, documented scope cut, not a bug).

### Phase 5 — Golden corpus, TDD crate, benchmark suite, model zoo
`2ba2572` extended the golden corpus to the full 36-op `KernelBackend`
surface + executed-TFLite model goldens (captured from a real
`ai-edge-litert` interpreter run, not hand-computed), `RefBackend` trait
wiring, per-op TDD crate, and the first cut of the ESP32-S3 benchmark
firmware (`hematite-benchmarks`).
`ad0c113` model-level zoo compilation: 5 real public int8 `.tflite` models
compiled end-to-end via `#[model(...)]` (the plan's originally-named 18
zoo models turned out to be unobtainable — 15 are proprietary `.espdl`,
not `.tflite`; documented in `models/zoo/DEFERRED_MODELS.md`). Closed
several emitter gaps found by real models (legacy TFLite opcode encoding,
PAD, TRANSPOSE, multi-axis MEAN) and wired 5 previously-unsupported ops
(matmul, sigmoid, tanh, reduce_max, reduce_min) into `RefBackend`.
`158bd47` final clippy cleanup.

**Plan closure**: all 37 plan tasks reached `[x]` (one, T0.2 — the
esp-rs-fork toolchain pin — was initially recorded `[~]`/blocked because no
Xtensa toolchain was installed in the environment; see Phase 6 below for how
it was unblocked). The Final Verification Wave (4 independent reviewers —
goal/constraint compliance, code quality, bit-exactness/test-integrity,
benchmark-claim honesty) all returned **APPROVE**.

### Phase 6 — Toolchain install, device compilation, QEMU bring-up
`5001bac` **T0.2 unblocked**: installed the esp-rs fork toolchain via
`espup` (custom rustup channel `esp`, rustc 1.95.0-nightly), pinned it in
`rust-toolchain.toml`.
`672f415` made the **entire device tree actually compile** for
`xtensa-esp32s3-none-elf` for the first time — this surfaced real defects
that had been invisible with no toolchain to compile against: the
`#![feature(asm_experimental_arch)]` gate the esp-rs fork requires for
Xtensa inline/global asm, `Default` impls needed for padded FFI-style
structs, visibility fixes for the SIMD entry points, and several esp-hal
1.1.1 API corrections in the benchmark firmware (`Instant::now()`,
`CpuClock` discriminants, defmt-rtt linkage).
`4c8e9f0` **device bring-up**: esp-hal linker script integration
(`linkall.x`/`defmt.x`/`-nostartfiles`), PSRAM arena rebuilt as a runtime
slice (esp-hal 1.1.1 has no `.dram1.psram` static section), **74
GNU-as→LLVM-MC assembler rewrites** across the vendored `.S` files (the
esp-rs fork's LLVM integrated assembler is stricter than the GNU assembler
the original esp-dl code targeted — branch-relaxation range errors,
`.macro` visibility, missing struct pad fields), and the first freestanding
C benchmark (`benchmarks/qemu-baseline/`) booting under Espressif's QEMU fork
for a cross-language comparison baseline.
`0a42b2f` **Rust firmware boots under QEMU** — a `qemu` Cargo feature
routes benchmark output through direct UART0 register writes (defmt-RTT has
no sink under QEMU) and bypasses PLL/PSRAM/CCOUNT-calibration steps that
hang or drift under emulation; `#[xtensa_lx_rt::entry]` roots the reset
vector.
`b7784d7` fixed a **release-profile-only** assembler bug: `rustc` merges
every `global_asm!` unit of a crate into one LLVM-MC assembly stream at
`opt-level ≥ 1` (debug keeps them separate), so shared vendored `.S` files
referenced from multiple kernel modules collided on macro redefinition.
Fixed with `.ifndef`/`.set`/`.endif` include-guards.
`56fb530` first real release-profile Rust-vs-C comparison under QEMU:
Rust was 1.30–1.76× slower than C `-O2` on 4 isolated kernel shapes — at
the time unexplained (a debug-vs-`-O2` mismeasurement was ruled out first,
since it dropped a 22–31× debug-profile gap down to this).

### Phase 7 — Full zoo-model device validation
`e7824aa` wired `#[model]`-generated inference into the benchmark firmware
(`model-validation` Cargo feature) and ran the 5 runnable zoo models
end-to-end **on the device target, inside QEMU**, comparing output against
the deterministic executed-TFLite golden byte-for-byte:

| Model | Result |
|---|---|
| `sine` | ✅ bit-exact PASS |
| `hello_world_int8` | ✅ bit-exact PASS |
| `kws_micro_speech_int8` | ✅ bit-exact PASS |
| `anomaly_detect_int8` | ✅ bit-exact PASS |
| `person_detect_int8` | ⚠️ FAIL at idx 0 (got=127, want=120) — known, documented divergence (see §3) |
| `mobilenet_v2_1.0_224_int8` | honest SKIP — needs PSRAM, QEMU has none |

Host suite stayed at 80 test suites green throughout. Two real, small bugs
were found and fixed during review of the (partially aborted) helper work
that produced this: a lifetime-annotation error and a module-privacy error
in the new validation code — both one-line fixes.

### Phase 8 — TIE728 SIMD dispatch wiring + QEMU emulation-gap discovery
`e4c1e8f` the single most important finding of this project's benchmark
work: **the TIE728 SIMD backend, though it compiled and assembled
correctly for Xtensa since Phase 3, had never been called from anywhere** —
every public kernel dispatch function called only its scalar fallback. The
1.30–1.76× "Rust is slower than C" numbers from `56fb530` were a
scalar-vs-scalar comparison, not SIMD-vs-scalar; the SIMD assembly was
dead code from a call-graph perspective despite being correctly compiled,
assembled, and bit-exactness-tested in isolation.

A full audit of all 9 `hematite-s3` kernel files found:
- **5 kernels have real, correct SIMD backends** — wired this pass:
  `conv2d_1x1`, `conv2d_3x3`, `fully_connected`, elementwise
  `add`/`mul`/`sub`, and `avg_pool_2d`/`max_pool_2d`. Each dispatch is
  gated behind `cfg(target_arch = "xtensa")` plus the exact runtime
  preconditions the vendored assembly requires (16-byte pointer alignment,
  16-channel-multiple counts, zero padding for 1×1/3×3, uniform per-tensor
  quantization, etc.) — verified correct by hand-tracing the assembly, not
  assumed.
- **3 kernels do *not* have a safe backend and were correctly left
  scalar-only, with the reason documented in code and in the notepad**:
  `depthwise_conv2d` (the vendored `.S` file only has shared per-channel
  requantize macros, no actual depthwise MAC entry point — dispatching to
  it would silently compute the wrong thing); `mean`/reductions (the SIMD
  variant is a literal no-op stub that writes nothing); `relu` (the
  existing, never-executed SIMD glue has a **provable off-by-16-element
  buffer-undercounting bug** — mathematically verified across several
  input sizes — logged for a future fix, not touched this pass since
  `relu` isn't in the timed benchmark table).

Wiring `avg_pool_2d`'s SIMD path was its **first ever real invocation**,
which immediately exposed a genuine Xtensa ISA constraint: the vendored
`call8` direct-call instruction has an 18-bit signed PC-relative range
(~±512 KB), and unlike GCC, LLVM's Xtensa backend has no automatic
long-call relaxation — the final binary's layout pushed the call site out
of range. Fixed with a register-indirect `callx8` (no range limit),
touching only the Rust-side call site.

Re-running the release-profile QEMU benchmark then surfaced a **genuine
QEMU emulator bug**: the newly-live `EE.VSMULAS.S8.QACC.LD.INCP` fused
multiply-accumulate instruction (used by conv1x1/conv3x3/fully_connected)
is decoded by Espressif's QEMU xtensa/esp32s3 fork but not correctly
executed, causing a silent infinite exception loop. This was root-caused —
not guessed — via seven rounds of UART-print bisection narrowing the fault
from "eligibility check" down to a hand-unrolled replica of the exact
macro body, then confirmed with `qemu -d int` interrupt tracing. Fixed by
adding a `qemu` Cargo feature that disables just the three affected SIMD
dispatch paths under QEMU while leaving them enabled for real hardware
builds.

**Honest conclusion**: after this fix, the QEMU benchmark's cycle counts
are numerically identical to the pre-wiring scalar baseline. This is
expected, not a sign of an incomplete fix — every SIMD path relevant to
the benchmark's specific shapes is *either* QEMU-gated off because of the
emulator bug above, *or* was already structurally ineligible for reasons
unrelated to QEMU (depthwise has no backend at all; the benchmark's
avg-pool row uses a 7×7 filter but the SIMD path only supports 2×2).
**Demonstrating a real SIMD speedup now requires physical ESP32-S3
hardware** — QEMU has been proven incapable of it for this benchmark, for
a documented, verified reason, not an assumption.

### Phase 9 — Physical ESP32-S3 hardware bring-up
Physical bring-up on a real ESP32-S3 dev board (Silicon Labs CP2102N
USB-UART bridge; chip rev v0.2; **4 MB flash with flash encryption ENABLED
and PERMANENT** — `SPI_BOOT_CRYPT_CNT=0x7`, irreversible, so *all* flash
writes must go through `esptool write-flash --encrypt`, which encrypts
on-chip with the eFuse key, never plaintext).

Four real hardware root causes were found and fixed while getting a
freestanding C harness to boot (each verified by capture on-device):
1. **Inline-literal corruption** (`-mtext-section-literals` + `movi`
   immediates like `0x60008000`) embeds raw data in the code stream →
   `IllegalInstruction`; fix = declared `.literal` symbols + `l32r` for all
   large immediates.
2. `.align 4` placed before `entry` makes the assembler drop the `entry`
   instruction (garbage emitted); `entry` must be the very first
   instruction of the aligned section.
3. **RTC WDT write-protect offset**: `wdtwprotect` is at RTC_CNTL +`0xb0`
   (not `0xa0`, which is WDTCONFIG2), so the unlock never worked and every
   build silently reset every ~9 s.
4. **UART0 TX FIFO requires 32-bit stores** — byte (`s8i`) stores to the
   FIFO are silently dropped on ESP32-S3; 32-bit stores work. (The Rust
   firmware's `write_volatile(fifo, u32)` had worked all along.)

The bare-metal C path was abandoned when its flash-resident `main` faulted
with `IllegalInstruction` on its first `entry` (root cause never resolved —
the four bugs above were all found *before* that). Pivoted to **Option B:
ESP-IDF v5.5.1 + the vendored TIE728 assembly** as a proper C-SIMD harness
(`benchmarks/espdl-baseline/`), which boots through ESP-IDF's official
startup and runs the flash code normally.

Simultaneously, the first *real-hardware* execution of Hematite's SIMD
kernels exposed a **windowed-ABI `call8` argument-placement bug** in all 9
`hematite-s3` SIMD call sites: for a `call8`-target (callee uses
`entry sp,128`), the window rotates by 8, so arguments must go in the
*caller's* `a10/a11/a12` (not `a2/a3/a4`). Fixed at every call site
(conv1x1, conv3x3, gemm, elementwise, activations, pool); verified by GCC
disassembly. This was why QEMU never caught it — the `qemu` feature gates
off all SIMD paths due to the `VSMULAS` emulator bug.

Final clean on-device benchmark (`hematite-benchmarks` release build, 240
MHz calibrated, 4 SRAM rows + expected no-PSRAM panic for the PSRAM-tier
rows):

| Row | Cycles (min/med) | col1 speedup |
|---|---|---|
| `conv_s8 8x8,64x3x3x3` | 4893951/4893951 | 1.03× (pad gated, scalar) |
| `depthwise_conv_s8 18x18` | 3538988/3538988 | 0.99× (input_c=1 gated, scalar) |
| `fc_s8 271row,3out` | 17936/17936 | 0.99× (out=3 gated, scalar) |
| `conv1x1_s8 64x1x1x64` | **2626/2628** | **51.37×** (SIMD fires) |

The conv1x1 SIMD path ran the real TIE728 asm for the first time: 4096
MACs in 2628 cycles (0.64 cyc/MAC), with `out_fnv(ref/s3) =
0x0bea8225/0x5eee898e` — SIMD output *differs* from the scalar reference
(see §3).

### Phase 10 — C-SIMD bit-exact output match (cross-language benchmark)
`benchmarks/espdl-baseline/` (ESP-IDF v5.5.1 C harness) calls the *same*
vendored `dl_tie728_s8_conv2d_11cn` asm entry directly, with the same
`fill_pattern` (input `i*7+3`, weights `i*13+11`, bias `i*17-8`, out 0),
same `Tie728ConvArgs` (offsets verified byte-identical to the Rust struct),
and the same quantization constants (`mult=1<<30`, `shift=0`, act
`-128/127`).

One root cause surfaced by the comparison: **FNV-1a sign-extension
convention.** The Rust firmware's `fnv1a` does `h ^= b as u32` on `i8`
bytes, which *sign-extends* negatives (`0x80 → 0xFFFFFF80`); the first C
version XORed raw bytes, producing a different checksum for the same
output. Fixed in C (`h ^= (uint32_t)(int8_t)b`), after which **both
checksums matched bit-exact on-device**:

| Path | Cycles (min/med) | out_checksum |
|---|---|---|
| C raw TIE728 asm call | 380/380 | `0x5eee898e` |
| C full-API mirror (gate+dispatch+asm) | 1767/1767 | `0x5eee898e` |
| Rust `hematite-s3` public `conv2d_1x1` | 2626/2628 | `0x5eee898e` |
| C scalar reference | — | `0x0bea8225` |
| Rust scalar reference (`hematite-ref`) | 141055 | `0x0bea8225` |

**Findings:** (a) the C↔Rust SIMD output is *bit-exact* — Rust executes the
same vendored asm as C, so there is no Rust-specific SIMD slowdown; (b) the
380 → 1767 → 2628 cycle ladder is pure wrapper overhead (validation +
dispatch + arg build), not kernel cost; the kernel itself is 0.09 cyc/MAC
for 16-wide TIE728; (c) SIMD output (`0x5eee898e`) ≠ scalar reference
(`0x0bea8225`) because the asm consumes filter in `[g][ic][lane]` layout
while the Rust/C wrappers feed raw `[oc][ic]` weights — a deterministic
filter-layout transform, not a bug, and real headroom for a faster kernel.

---

## 3. Known divergences and open technical debt

- **`person_detect_int8` / `mobilenet_v2` output divergence** (idx 0:
  got=127, want=120 on `person_detect`): two independent, well-understood
  causes, not a bug in Hematite's kernels. (1) TFLM's
  `MultiplyByQuantizedMultiplier` uses single-rounding fixed-point
  requantization; the executed-TFLite golden for these two models was
  captured from `ai-edge-litert` (used because no full TFLM C++ build
  environment was available on this host), which uses gemmlowp-style
  double-rounding — the two agree almost everywhere but disagree by ±1 at
  specific rounding-boundary values. (2) TFLM's and LiteRT's int8 softmax
  implementations diverge on wide-dynamic-range logits; `person_detect`'s
  final-layer logits `[115, -122]` saturate to `[127, -128]` under TFLM's
  algorithm (exactly what Hematite produces, confirming Hematite correctly
  implements its pinned TFLM spec) but LiteRT's differently-scaled softmax
  produces `[120, -120]` (the golden's answer). The smaller models
  (`sine`, `hello_world`, `kws`, `anomaly_detect`) are immune because they
  either have no softmax stage or too few quantized-multiply operations to
  statistically hit a rounding-boundary case. Documented in
  `models/zoo/DEFERRED_MODELS.md`. Reproduced identically on-device inside
  QEMU, which is itself useful confirmation that the Xtensa-compiled
  kernels behave identically to the host `RefBackend`.
- **`relu_simd` off-by-16 buffer bug** — logged, not fixed (not in any
  timed benchmark path; low risk since it's still dead/unwired).
- **`FusedSchedule`/arena-plan passes not integrated into the emitter** —
  built and unit-tested standalone since Phase 4, deliberately not wired
  into `#[model]`'s straight-line output; every intermediate tensor is
  currently a stack local rather than an arena-allocated slot. Works fine
  for small models; the largest model tested on-device
  (`person_detect_int8`) is borderline on stack usage, and `mobilenet_v2`
  needs real PSRAM.
- **Elementwise/pool SIMD under QEMU** — left *enabled* under the `qemu`
  feature (unlike conv1x1/conv3x3/fc) because they use different TIE
  instructions (`VADDS`/`VSUBS`/non-`VSMULAS` `VMULAS`/`VMAX`) with no
  evidence of the same emulator bug, but this has not been exercised by
  the current benchmark table's specific shapes and is therefore an open,
  documented, low-confidence item — not proven safe under QEMU, just not
  proven broken either.
- **SIMD output ≠ scalar reference on conv1x1** (`out_fnv` `0x5eee898e` vs
  `0x0bea8225`, bit-exact across C↔Rust): the vendored TIE728 asm consumes
  the filter in `[g][ic][lane]` layout (per 16-output-channel group,
  input-channel-major, 16 lanes), while both the C and Rust wrappers feed
  it raw `[oc][ic]` row-major weights. The result is a *deterministic*
  rearrangement of the same math — not a bug, and it is why `conv1x1` SIMD
  "speedup" numbers (col1 ≈ 51×) are computed against the scalar reference
  even though the outputs differ. Fixing it (transposing the filter, or
  feeding the asm's expected layout) is the largest single performance and
  correctness headroom item left.

---

## 4. Toolchain / environment reference

- **Rust toolchain**: `espup`-installed esp-rs fork, custom rustup channel
  `esp` (rustc 1.95.0-nightly). `rust-toolchain.toml` pins `channel = "esp"`
  only — this is a *custom* rustup toolchain, so `rustup component add` /
  `rustup target add` do not work against it; its components and Xtensa
  targets are baked in by `espup`.
- **QEMU**: Espressif's fork (`qemu-system-xtensa`, machine `esp32s3`),
  installed at `~/.esp-qemu`. Boot images are produced with `espflash
  save-image --chip esp32s3 --merge <elf> <bin>`, which auto-embeds the
  ESP-IDF v5.5.1 second-stage bootloader. Correct invocation:
  `qemu-system-xtensa -nographic -machine esp32s3 -monitor none -serial
  file:<log> -drive file=<bin>,if=mtd,format=raw -icount 3` (QEMU does not
  exit on its own unless it crashes; run in the background, sleep, then
  kill).
- **Git identity for this repository**: `Yatendra Singh
  <singh.0.yatendra@gmail.com>`, set as this repo's *local* git config
  (no `-c` flags needed for any commit here). This repository's entire
  history was retroactively rewritten to this identity (previously a mix
  of a scaffolding-era GitHub identity, this machine's global git config,
  and — for one working session — an unrelated organization's identity
  mistakenly inherited from a global agent-configuration file). Recorded
  as a standing rule in this repository's own `repo-guidelines.md`. **The rewritten
  history has not been force-pushed to `origin/main`** — that remains an
  explicit, separate decision pending user confirmation.

---

## 5. Current status

All 37 original plan tasks are complete and the Final Verification Wave
approved. Hardware bring-up (Phase 9) and the C-SIMD bit-exact output match
(Phase 10) are committed. Host test suite: 80 suites, 0 failures,
maintained throughout every change in this log.

On-device benchmark state: conv1x1 `64x1x1x64` SIMD runs the real TIE728
asm at 2626/2628 cycles (51× vs scalar), with output verified bit-exact
against an independent ESP-IDF C harness (`benchmarks/espdl-baseline`,
380-cycle raw asm / 1767-cycle full-API). Other SRAM rows are
legitimately scalar (pad/input_c/out-channel gates); PSRAM-tier rows
require a PSRAM-equipped board.

**Open decisions awaiting explicit direction:**
1. Force-push the rewritten git history to `origin/main`?
2. The `person_detect`/`mobilenet_v2` rounding divergence's practical
   impact on hardware (now that real timing exists).
3. The conv1x1 SIMD filter-layout gap (see §3): transpose the weights to
   match the asm's `[g][ic][lane]` layout to make SIMD output equal the
   scalar reference, and to recover the remaining wrapper overhead
   (2628 → 1767+ cycles).
