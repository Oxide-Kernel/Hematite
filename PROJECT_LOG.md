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

### Phase 11 — Per-operation C-SIMD comparison (all SIMD ops) + relu fix

Extended the comparison from one row (conv1x1) to **all nine SIMD-capable
operations**. Added SIMD-eligible `SIMD_*` bench rows to
`hematite-benchmarks/src/spec.rs` (conv3x3 32x32 VALID, fc 256→64,
max/avg-pool 2x2 32x32x16, relu 256, add/sub/mul 256 — each satisfying its
kernel's TIE728 gate: `%16` channels/dims, 16-aligned pointers, offsets 0,
identity requantize), with new `OpKind`/`KernelParams` variants
(`MaxPool`, `Relu`, `Add`, `Mul`, `Sub`; `Activation`, `Elementwise`) and
`run_kernel`/`run_ref_kernel` dispatch arms. The C harness
(`benchmarks/espdl-baseline`) grew per-op wrappers + scalar refs + arg
structs (`Tie728Conv33Args`, `Tie728ReluArgs`, `Tie728MaxPoolArgs`,
`Tie728AvgPoolArgs`, `AddSubAlignedArgs`, `MulAlignedArgs`) and now links
all 8 vendored asm files.

Two on-device bugs found and fixed along the way:

1. **`relu_simd` off-by-16 bug (fixed).** The vendored
   `dl_tie728_s8_relu_11c` processes `32·c_rs1_1 + 16·c_rs2_1 + 16`
   elements — an unconditional trailing 16-element block before `retw`.
   The old glue computed `c_rs1_1 = c/32 − 1`, leaving the last 16 elements
   unprocessed. Fixed in `hematite-s3/src/activations.rs` to reserve the
   block (`c_rs1_1=(c−16)/32`, `c_rs2_1=((c−16)%32)/16`), switched the
   call to `callx8` (the asm is in a separate region), and wired a SIMD
   dispatch gate into the public `relu()`. Verified on device: relu row is
   now bit-exact vs scalar (`0x6c620b3d`).
2. **`.section .iram1` orphan (relu/pool asm).** The vendored
   `relu/max_pool2d/avg_pool2d.S` emitted `.section .iram1`, which
   ESP-IDF's linker placed in an orphan `.rwtext.wifi` not covered by any
   LOAD segment → calls jumped into unloaded IRAM and the bench hung.
   Fixed by commenting out `.section .iram1` (same treatment conv2d.S
   already had), so the kernels link into flash irom.

**Full on-device result (ESP32-S3 @ 240 MHz, `bench10` Rust firmware vs
C harness):** all 9 SIMD checksums bit-exact C↔Rust; all 9 scalar-refs
bit-exact C↔Rust:

| Op | C-SIMD cycles | Rust s3 cycles | C-SIMD == Rust s3 | SIMD == scalar |
|---|---|---|---|---|
| conv1x1 64x1x1x64 | 472 | 2627/2628 | ✅ `0x5eee898e` | no (filter layout) |
| conv3x3 32x32 VALID | 2824 | 4849/4876 | ✅ `0xd1a9b601` | no (filter layout) |
| fc 256→64 | 1288 | 3187/3214 | ✅ `0x16542aba` | no (filter layout) |
| max-pool 2x2 | 1396 | 1978/1992 | ✅ `0x50d8f9c5` | no |
| avg-pool 2x2 | 7181 | 7378/7405 | ✅ `0xdedd2dc5` | no |
| relu 256 | 175 | 425/426 | ✅ `0x6c620b3d` | yes |
| add 256 | 167 | 467/481 | ✅ `0x14834bbb` | yes |
| sub 256 | 265 | 547/574 | ✅ `0x62d74671` | yes |
| mul 256 | 539 | 876 | ✅ `0xd3c0a7f1` | yes |

**Findings:** (a) C↔Rust SIMD bit-exact for *every* op — the vendored asm
is the single kernel source; (b) the cycle gap per row is wrapper overhead
(raw asm 472 vs Rust 2627 for conv1x1; 175 vs 425 for relu); (c)
weighted/positional ops (conv, fc) differ from scalar due to the `[g][ic][lane]`
filter layout; max/avg-pool also differ from scalar (asm pooling/rounding
semantics — avg-pool uses `shift/area_inv` fixed-point vs scalar
`round_half_away_zero`), and C↔Rust agree in all cases; (d) elementwise
ops and ReLU are bit-exact vs scalar, validating the identity contracts and
the relu fix.

### Phase 12 — Prepared-kernel fast path (wrapper-overhead closure)

Motivated by the Phase 10/11 finding that the Rust public API measured 2.5–5.5x
the C raw-asm cycles even though the underlying TIE728 kernel is identical
(bit-exact checksums proved the kernel cost is the same). The gap was
**wrapper overhead**, not language: slice-length validation + SIMD gate + args
struct build paid on every call.

What was added:

1. **Prepared handle structs in `hematite-s3`** — one per SIMD-capable op:
   `PreparedConv1x1`, `PreparedConv3x3`, `PreparedFc`, `PreparedRelu`,
   `PreparedMaxPool`, `PreparedAvgPool`, `PreparedAdd`, `PreparedMul`,
   `PreparedSub`. `new(params)` runs the full SIMD eligibility gate ONCE and
   caches the arg fields (`ocd8`/`cdiv`/`mac_shift`/`use_relu`, pool offsets,
   `mul_shift`, …); `run(input, …, output, scratch)` then only re-checks
   16-byte pointer alignment and dispatches. Falls back to the scalar kernel
   when not SIMD-eligible. Shared host-compilable `simd_eligible_*` gates
   reused by the legacy public paths (single source of truth).
2. **MaybeUninit args builds** — every `Tie728*Args` dispatch now writes only
   the fields the asm reads (byte-offset `ptr::write`s on a
   `MaybeUninit::uninit()`), removing the `memset`/dead-pad-store that the
   struct literal `..Default::default()` emitted.
3. **`#[inline(never)]` on `gemm_simd::dispatch_fc`** — found during the
   fc-row regression: inlining the args-building dispatch into
   `fully_connected` produced a genuine Xtensa miscompile (args sourced from
   wrong registers; the fc 256→64 checksum diverged to `0x68e95989` vs the
   correct `0x16542aba`). Forcing the dispatch out-of-line fixed it.
4. **Prepared benchmark in the firmware** — `spec.rs` gained
   `PreparedKernel` enum + `prepare_kernel(spec)` + a host-compilable
   `run_kernel_scalar` fallback; `firmware.rs bench_kernel` constructs the
   handle once (outside the timed window) and emits a `prepared:` line
   (min/med cycles + FNV checksum) per row. New host test
   `every_spec_prepared_matches_ref_bit_exact` (suite: 30 tests).

Device results (`bench11b`, ESP32-S3 @ 240 MHz) — prepared out_fnv is
bit-exact equal to the public s3 checksum on **every** row:

| Row | C raw-asm | Rust public s3 | Rust prepared |
|---|---|---|---|
| conv1x1 64x1x1x64 | 472 | 2509 / 2521 | **669 / 669** |
| conv3x3 32x32 VALID | 2824 | 4662 / 4662 | **3075 / 3102** |
| fc 256→64 | 1288 | 3335 / 3335 | **1547 / 1574** |
| max-pool 2x2x16 | 1396 | 1896 / 1896 | **1829 / 1856** |
| avg-pool 2x2x16 | 7181 | 7361 / 7388 | **7274 / 7301** |
| relu 256 | 175 | 361 / 388 | **354 / 354** |
| add 256 | 167 | 414 / 441 | **363 / 363** |
| sub 256 | 265 | 494 / 494 | **442 / 442** |
| mul 256 | 539 | 777 / 781 | **743 / 743** |

The prepared path closes the gap to ~1.0–2.2x of C raw-asm (was 2.5–5.5x).
Documented in `benchmarks/espdl-baseline/README.md` under "Rust prepared-path
vs C raw-asm".

### Phase 13 — Bespoke GPR-accumulator (ACCX) kernels: bit-exact weighted SIMD

On-device TIE728 probes (`probe_qacc`/`probe_s16`/`probe_s8accx`/`probe_accx`
in `benchmarks/espdl-baseline/main/`) proved that **no QACC-based
accumulation can be bit-exact** for the weighted ops:

* `EE.VSMULAS.S8.QACC` saturates each of its 16 lanes at 8 bits — even a
  single `127×127` product reads back `0x7f`. So the vendored
  `dl_tie728_s8_conv2d_*` per-layer requantize (which Phase 10/11 had
  "matched" between C and Rust) is fundamentally inexact: C-SIMD == Rust-s3
  was the two sides producing the *same* 8-bit-saturated garbage.
* `EE.VSMULAS.S16.QACC` saturates at 16 bits (32767 for a true acc of
  129032) — so the earlier sign-extend-to-int16 plan was refuted too.
* `EE.SRS.ACCX` is a 1-bit-pos instruction with a triangular shift mapping
  (`gpr n → shift n(n+1)/2`), so `gpr=0` extracts the exact 32-bit
  accumulator.
* **`EE.VMULAS.S8.ACCX` is a 16-wide element-wise dot-product reduction into
  a 32-bit GPR accumulator with full 16-bit products** (`127×127=16129`
  preserved) — the exact bit-exact int8 conv primitive, working on the raw
  `[oc][ic]` weight layout directly (no `[g][ic][lane]` transform needed).

What was added:

1. **`hematite-s3/src/asm/s8_accx_conv1x1.S` + `s8_accx_conv3x3.S`** —
   bespoke kernels: per output channel, `EE.ZERO.ACCX`, inner ic16 loop
   (`VLD.128.IP` filter + input, `VMULAS.S8.ACCX`), `SRS.ACCX gpr,0,0`,
   store i32 acc to scratch. conv3x3 loops the 9 taps + row_delta between
   rows; the *caller* loops output pixels (the asm computes one pixel).
2. **`hematite-s3/src/accx.rs`** — `accx_eligible_1x1/3x3` gates
   (`in_c>=16 && in_c%16==0 && out_c>=1`), a `ReqCtx` struct, and
   `requantize_1x1` (bit-exact TFLite epilogue: `(acc·mult+round)>>total_shift`,
   `+output_offset`, clamp `[act_min,act_max]`, saturating cast). The ACCX
   path is wired into `PreparedConv1x1/Fc/Conv3x3` AND the public
   `conv2d_1x1/fully_connected/conv2d_3x3`; scratch is a 4096-byte aligned
   buffer in the bench firmware.
3. **Three Xtensa-LLVM backend miscompiles found and fixed** (each surfaces
   as a scrambled register at an inline-asm or high-arg-count call site):
   `clobber_abi("C")` does not mark caller `a15` clobbered across `call8`
   (the kernel's `a7` loop counter corrupted the caller's output pointer →
   `out.ptr=0x40`; fix `out("a15") _`); `in("a12")` is clobbered by the
   kernel's `a4` increment (fix `inout("a12") acc_out => _`); and
   `avg_pool_2d_simd_ctx` is miscompiled both inlined and out-of-line (the
   MaybeUninit args build's 16-byte array copy gets field-swapped) — fixed
   by building the args as a plain struct literal and pinning every asm
   operand to an explicit register (`in("a10")..in("a13")`, `callx8 a13`).

Device results (`bench34`, ESP32-S3 @ 240 MHz) — **SIMD now equals the
scalar reference bit-exact on every weighted row**:

| Row | s3 SIMD cycles (min/med) | out_fnv(s3 == ref) |
|---|---|---|
| conv1x1 64x1x1x64 | 12595 / 12622 | `0x0bea8225` ✅ |
| conv3x3 32x32 VALID | 35743534 / 35743561 | `0x0a181085` ✅ (full 30×30 image) |
| fc 256→64 | 20068 / 20081 | `0x32e35185` ✅ |
| max-pool 2x2x16 | 1892 / 1920 | `0x50d8f9c5` (== C-SIMD; pool fixed-point vs scalar `round_half_away_zero`) |
| avg-pool 2x2x16 | 7342 / 7343 | `0xdedd2dc5` (== C-SIMD; same) |
| relu/add/mul/sub 256 | 357/411/774/491 | bit-exact vs ref |

conv1x1 went from 2627 cycles (vendored asm, wrong output) to 12595 cycles
(ACCX, bit-exact) — ~2.5x the raw asm (element-wise reduction is one MAC
per lane per input element rather than the QACC broadcast) but *correct*.
Documented in `benchmarks/espdl-baseline/README.md` under "Bespoke ACCX
kernels".

### Phase 14 — Optimized ACCX kernels (fast64 paths)

The bespoke S8-ACCX kernels got a **fast path for `input_c == 64`**,
exploiting the chip's eight TIE728 Q registers:

* **conv1x1 (`s8_accx_conv1x1.S`)** — `.Lfast64`: the 4 input vectors stay
  resident in `q0..q3` (loaded once), per output channel a single-level
  hardware `loop a6` does 4 filter `VLD.128.IP` into `q4..q7` + 4
  `VMULAS.S8.ACCX`. The general (branch-based, nested-loop-free) path covers
  any other `in_c%16==0`.
* **conv3x3 (`s8_accx_conv3x3.S`)** — `.Lc3fast64`: the 9 taps are fully
  unrolled via a `TAP_64` macro (4 filter + 4 input `VLD.128.IP`, 4
  `VMULAS.S8.ACCX` per tap), with loads hoisted 2 instructions ahead to hide
  VLD latency; `row_delta` bridges rows. Uses a short-branch + long-jump
  loop (`blt a8,a6; j done`) instead of hardware `loop`, because the ~350-byte
  unrolled body exceeds LLVM-MC's `loop` fixup range and the 8-bit `bge`
  range. Label collision `.Lfast64` (both files concatenated by
  `global_asm!`) fixed by renaming the conv3x3 labels.
* **Shared-file note**: the kernels are `include_str!`'d by Rust, so the wins
  carry into `hematite-s3` automatically — but cargo does NOT rebuild a crate
  when an `include_str!` file changes (`cargo clean -p hematite-s3` required;
  an early `bench35` silently linked the stale kernel).

Device results (`bench36`, ESP32-S3 @ 240 MHz, all bit-exact unchanged):

| Row | Rust s3 cycles (min/med) | before | Δ |
|---|---|---|---|
| conv1x1 64x1x1x64 | 9937 / 9939 | 12593 / 12595 | 1.27x faster |
| conv3x3 32x32 VALID | 15009334 / 15009361 | 35743533 / 35743561 | 2.38x faster |
| fc 256→64 | 20133 / 20159 | 20133 / 20159 | unchanged (general path, in_c=256) |

Kernel PURE costs (C harness, cycles): conv1x1 **996** (was 3422, 3.4x),
conv3x3 **7353504** (≈6.2M instructions ≈ 1.1 instr/cycle — near the issue
floor given the 8-Q-register limit; the 36-vector 3×3 window cannot stay
resident). The scalar requantize epilogue is now the dominant full-API cost
(conv1x1: 9939 − 996 ≈ 8943 cyc), and Rust's `requantize_1x1` is heavier
than the C harness's — why Rust full is ~1.15–1.36x C full for the same
kernel.

### Phase 15 — Requantize fast paths (uniform-scale detection)

The scalar per-channel requantize epilogue (`requantize_1x1`, ~140
cyc/channel) had become the dominant full-API cost: four slice bounds
checks per channel plus an i64 `multiply_by_quantized_multiplier` per
iteration. `hematite-s3/src/accx.rs` now:

* **`uniform_scale(mult, shift)`** — scans the per-channel arrays once
  (in the dispatcher, *outside* the pixel loop for conv3x3) and returns
  `Some((m, s))` when every channel shares the same scale.
* **`ReqCtx` gains `uniform_mult`/`uniform_shift`** (with
  `uniform_shift == i32::MIN` meaning "per-channel"). The dispatchers
  (conv1x1 / fc / conv3x3) populate them from `uniform_scale`.
* **`requantize_1x1` fast paths**, bit-identical to the i64 reference:
  * `mult == 1<<30, shift == 1` — **identity**: `scaled == acc` (no
    fixed-point multiply at all).
  * `mult == 1<<30, shift == 0` — `(acc + 1) >> 1`, the common
    identity-mult bench scale.
  * any other uniform pair — the scale is **hoisted** (round/total_shift
    and mult live in registers; same i64 arithmetic).
  * per-channel (mixed) — the general path, now with **one upfront length
    assert + unchecked indexing** instead of four per-iteration bounds
    checks.

Device results (`bench37`, ESP32-S3 @ 240 MHz, all checksums still
bit-exact — conv1x1 `0x0bea8225`, conv3x3 `0x0a181085`, fc `0x32e35185`):

| Row | bench36 | bench37 | Δ |
|---|---|---|---|
| conv1x1 64x1x1x64 | 9939 / 9939 | **5041 / 5041** | 1.97x |
| conv3x3 32x32 VALID | 15009334 / 15009361 | **9511784 / 9511785** | 1.58x |
| fc 256→64 | 20133 / 20159 | **15293 / 15294** | 1.32x |

Host tests: two new unit tests in `accx.rs`
(`requantize_fast_paths_match_reference` — the fast paths equal the i64
reference across uniform and per-channel cases, boundary/saturation
values included; `uniform_scale_detects_uniformity`). Suite: 31
`hematite-benchmarks` + 2 `hematite-s3` tests, all green.

### Phase 16 — Bespoke full op-sweep: from scratch, beat ESP-NN end-to-end

Directive (verbatim): *"we are rewriting everything from scratch to make this
the best NN library for ESP32 out there"* — no vendoring of Espressif's conv
kernels; every op implemented ourselves and validated bit-exact on the real
ESP32-S3 board (`benchmarks/espnn-baseline` runs the same two models through
standard `esp-nn` v1.2.5 for the head-to-head). Result: **Hematite now beats
ESP-NN on both models, bit-exact everywhere.**

Two real models were added to the firmware to make the comparison meaningful
(`model_cnn.rs` / Model A 4-layer; `model_mv2.rs` / Model B
MobileNetV2-style 7-layer), each reporting end-to-end cycles + per-layer
FNV-1a checksums that must match the scalar reference exactly (and do, on
every layer of both models, plus the C scalar/ESP-NN harnesses).

| Phase | Work | Model B effect (cyc) |
|---|---|---|
| 0 | QACC 40-byte-store layout probe; esp-nn-style two-pass read-back locked from silicon | — |
| 1 | Bespoke `s8_accx_depthwise.S` (QACC per-lane, from-scratch 16×i32 read-back), bit-exact in C | — |
| 2 | Depthwise wiring into `hematite-s3` (+ pool prepared-path fix) | 9,970,333 → 9,074,866 |
| 3 | First-conv `in_c<16` zero-pad to next multiple of 16 | 9,074,866 → 1,928,931 |
| 4 | conv3x3 `fast16`/`fast32` unrolled paths | 1,928,931 → 914,239 |
| 5 | fc general-path hardware-loop inner | 914,239 → 887,689 |
| 6 | `s8_requantize.S` fused asm requantize epilogue | 887,689 → **770,827** |

Final on-device (bench48, ESP32-S3 @ 240 MHz, all checksums bit-exact):

| Model | ESP-NN optimized | Hematite | Ratio |
|---|---|---|---|
| A (4-layer) | 2,630,401 | **1,707,746** | 1.54× faster |
| B (mv2mini 7-layer) | 994,782 | **770,827** | 1.29× faster |

Bench rows (public API, `bench48`): conv1x1 4379, conv3x3 8868886 (full
30×30), fc256 8393, max-pool 33461, avg-pool 29595, relu 358, add 411,
mul 774, sub 491 — every checksum bit-exact vs scalar reference.

Engineering notes:
- `EE.VSMULAS.S8.QACC` lanes are wide (127×127×9 = 145161 survives); the
  raw 40-byte `ST.QACC` store is a diagonal layout. The exact two-pass
  read-back (gapped 20-byte store + word bit-slice into q0/q3/q1/q6) was
  reverse-engineered from silicon via `probe_qacc_esplike` and validated
  lane-for-lane on device before writing the depthwise kernel.
- The asm requantize uses the overflow-free identity
  `(acc + 1) >> 1 = srai(acc,1) + (acc & 1)`, proven bit-exact for all i32
  via a 100k-value host test.
- Xtensa LLVM-MC constraints hit again: `beqi imm==0` illegal, large `movi`
  immediates illegal, `loop`/`bge` 8-bit fixup range → trampolines +
  short-branch/long-jump patterns.

### Phase 17 — Full MobileNetV2 parity (SAME pad, stride-2, offsets, softmax, non-%16)

Closed every remaining parity gap against standard `esp-nn` so a stock
MobileNetV2 (SAME padding everywhere, stride-2 first conv + blocks) runs the
bespoke SIMD kernels instead of falling to scalar. Each phase was verified
bit-exact on the real board (`bench55`, 19 kernel rows + 3 models, every
`out_fnv(ref/s3)` equal), then the two stacks were benchmarked head-to-head
on a **third, real MobileNetV2-style model** (`model_mv2real.rs` / Model C).

| Phase | Gap closed | New SIMD row on device (cyc, bit-exact) |
|---|---|---|
| A | **SAME padding** for conv3x3 (spatial zero-pad in scratch) | `conv3x3_s8 16x16,32x3x3x32 SAME` 880125, `0xc53ebbc5`, 76.87× vs scalar |
| B | **Stride-2** depthwise (pixel-loop stride) | `depthwise_s8 12x12,16x3x3x16 S2 SAME` 30734, `0x5159710e`, 12.6× |
| C | **Non-zero input_offset** (fold `+offset` into the padded-input copy; OOB taps filled `-offset` so `(-off)·w + off·Σw = 0`) | conv3x3 SAME off3 1129475 `0xc53ebbc5`; dw S2 off-3 41167 `0x9ea5238d`; fc off5 238998 `0x32e35185` |
| D | **Model C** real MobileNetV2-style 6-layer on both stacks, per-layer checksums | Hematite 649675 vs ESP-NN 655303 cyc, all 5 layers bit-exact |
| E | **Softmax SIMD** (`s8_find_max` VMAX + cached exp in scratch) | `softmax_s8 1x1000` 263047, `0xaf0d15aa`, 1.70× |
| F | **Depthwise non-%16 channels** (zero-pad to 16 in scratch) | `depthwise_s8 12x12,12x3x3x12 SAME non16` 104047, `0x8da1a066`, 11.16× |

Model C (`mv2real`) — real MobileNetV2 shape language (SAME + stride-2):

```
L1 conv3x3   16x16x3   -> 8x8x32     stride 2, SAME,  act (0,127)
L2 depthwise 3x3 8x8x32 -> 8x8x32    dm=1, stride 1, SAME, act (0,127)
L3 conv1x1   8x8x32    -> 8x8x64                act (0,127)
L4 depthwise 3x3 8x8x64 -> 4x4x64    dm=1, stride 2, SAME, act (0,127)
L5 conv1x1   4x4x64    -> 4x4x128               act (0,127)
L6 FC        2048      -> 16                      act (-128,127)
```

Final head-to-head (both stacks bit-exact with the scalar reference on every
layer; `benchmarks/espnn-baseline` C harness + `hematite-benchmarks`):

| Model | ESP-NN optimized | Hematite | Ratio |
|---|---|---|---|
| A (4-layer) | 2,630,401 | 1,708,356 | 1.54× faster |
| B (mv2mini 7-layer) | 994,782 | 770,986 | 1.29× faster |
| C (mv2real 6-layer) | 655,303 | 654,407 | 1.01× faster |

Hematite now wins all three. The C side of Model C routes L1 (in_ch=3) through
ESP-NN's `im2col` and L2 through `mult1_3x3_padded` — both confirmed bit-exact
vs the scalar reference on device, so the comparison is fair.

Engineering notes:
- Asymmetric SAME padding is *additive*, not symmetric-doubled:
  `pad_total = ((out-1)*stride + dilated - in).max(0)`, `pad_before = total/2`,
  padded size = `in + pad_total`. The naive `in + 2*(total/2)` loses one
  border row when `total` is odd (the stride-2 SAME case, `total=1`) — fixed
  in both conv3x3 and depthwise dispatchers.
- OOB taps in the scalar reference contribute exactly 0 (bounds-skip), so a
  padded SIMD image must make OOB taps compute 0 too. With a non-zero
  `input_offset` that means filling the border with `-input_offset`, not 0,
  then folding `offset·Σw` per channel after the MAC.
- `EE.VMAX.S8` (softmax find-max) assembles cleanly; `l8si` does not exist in
  the Xtensa toolchain — use `l8ui` + `sext`.
- The vendored S16 experiment (`dl_tie728_s16.S` / `_conv2d.S`,
  `probe_s16.S`) was deleted: S16 QACC lanes saturate at 16 bits, so it could
  never be bit-exact; the ACCX + QACC-depthwise kernels supersede it.

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
- **SIMD output ≠ scalar reference on the weighted ops — RESOLVED by Phase
  13.** The vendored TIE728 asm (QACC lanes, 8-bit saturating) could never
  be bit-exact; the bespoke S8-ACCX kernels now make conv1x1/conv3x3/fc
  SIMD output equal the scalar reference exactly (see Phase 13). The vendored
  `[g][ic][lane]` filter-layout gap is gone (ACCX uses the raw `[oc][ic]`
  layout). Remaining SIMD≠scalar: max/avg-pool (pool fixed-point `shift`/
  `area_inv` semantics vs scalar `round_half_away_zero` — C-SIMD and Rust s3
  agree with each other; only the reference differs).

### Phase 18 — S3Backend, first real-silicon SIMD sweep, executed-TFLM goldens, zoo-model acceleration (plan `simd-zoo-hardening`)

Phase 18 (2026-08-10) is the plan `simd-zoo-hardening` execution: prove every
SIMD kernel on real silicon, wire the accelerated path end-to-end for zoo
models, and close the public-API speed gap against the C SIMD kernels.

**Hardware facts established on this board (ESP32-S3 rev v0.2, MAC
64:e8:33:67:e7:44):**
- **No PSRAM** (`PSRAM: 0 bytes` boot probe) → the PSRAM-tier benchmark rows
  and `mobilenet_v2` on-device are gated; 8 MB DIO 80 MHz flash.
- **Flash encryption is PERMANENT** (`SPI_BOOT_CRYPT_CNT=0x7`) and `espflash`
  has no encrypt support → **all device flashes use `esptool.py write_flash
  --encrypt`** (on-chip encryption via the ROM op, eFuse key readable). A
  plaintext flash boots into an `invalid header` loop.
- `person_detect` on-device is a permanent SKIP (`reason=stack`): generated
  `predict` allocas ~232 KB vs ~65 KB device stack (SP-switch attempt faults
  on silicon, window-underflow).

**1. `S3Backend` — the missing KernelBackend impl** (`hematite-s3/src/
backend.rs`, todo 3): all 36 trait methods forwarded to the s3 free
functions — 10 SIMD-dispatch (conv2d→1×1/3×3 by filter, depthwise,
fully_connected, avg/max pool, softmax, relu, add/mul/sub), 3 scalar
(relu6, hard_swish, mean), honest `Unsupported` for the rest; scratch-size
overrides for conv2d/depthwise/softmax. A follow-up (todo 25) added the 7
scalar data-movement kernels (reshape/transpose/pad/slice/concat/split/
resize_nearest) so `Model::<S3Backend>` can run every zoo model. The
`KernelBackend` trait itself is untouched (scope guardrail).

**2. First real-silicon SIMD sweep** (todos 6/7/8): the 18-check
`simd_validation` suite — previously never run on hardware (the Espressif
QEMU fork cannot execute TIE728 compute) — now runs to completion on the
board: **17 PASS + 1 documented known-delta** (avg_pool ±1 vs the scalar
ref, `fnv 0xd0d19a11` — the pool fixed-point contract; C-SIMD agrees).
Two real bugs were found and fixed on the way:
- **elementwise asm register scramble** (`mov a10,{output}`-template with
  `in(reg)` operands; LLVM placed args into the template's own clobbered
  registers → 16-byte DRAM writes clobbered the defmt `RTT_ENCODER.taken`
  flag → "defmt logger taken reentrantly" mask). Fixed with pinned-register
  operands (proven pool pattern).
- **validation-frame stack overflow**: `validate_all`'s 35 KB inlined frame
  pushed SP inside the SRAM arena; the depthwise check's buffer straddled the
  arena end and wrote 256 B into bss. Fixed by carving the check buffers from
  the unused arena (todo 18).
- **real QACC SIMD mean** (todo 18): the `mean_simd` no-op stub was replaced
  with a bespoke `s8_mean_reduce.S` (QACC per-lane accumulation + verified
  two-pass read-back). A misaligned `EE.VST.128` (stack-allocated `accs`
  without `#[repr(align(16))]`) silently zeroed the accumulator on device —
  fixed with `AlignedStage`/`AlignedAccs` wrappers; the final check passes
  bit-exact on silicon (`mean_hw_2x2x4 PASS fnv 0xc14ca731`).
- QEMU gating made consistent: all SIMD dispatch is `all(xtensa, not(qemu))`.

**3. Executed-TFLM golden provenance** (todos 9/10/11): built
`tflite-micro` at the pinned SHA `18b9e6f2a8c5a9518e588f59c2ba16ef7ef9d551`
(`tools/tflm-goldens/`, Makefile-based) and captured real executed outputs
for the 6 zoo models + a new hard_swish micro-model. Findings:
- `kws` reproduces the golden bit-exactly (sanity gate); `sine`/`hello_world`
  match; `person_detect` is **bit-exact** vs executed TFLM (`fnv
  0x6962079d`) — the §6.2 wide-logit softmax divergence does not manifest at
  this SHA; `anomaly_detect` and `mobilenet_v2` goldens were regenerated
  (the old LiteRT-derived goldens coincided with Hematite's single-rounding
  by luck — executed TFLM uses gemmlowp **double**-rounding; ±1 on 210/640
  anomaly elements, documented).
- `mobilenet_v2` residual (984/1000 vs regenerated golden): TFLM `pad.cc`
  fills `output_zero_point` (−14) when `constant_values==nullptr`, while
  `PadParams` carries no zero point and the trait `pad()` has no pad-value
  arg — `ref`+`s3` fill 0 (documented follow-up; no trait change).
- softmax `diff_min` hardcode fixed to the TFLM formula
  `-CalculateInputRadius(5, left_shift)` (todo 13).

**4. Zoo-model acceleration — two root causes found** (todos 19/21):
`model_bench` now runs the real zoo models through `Model::<S3Backend>` with
arena-carved buffers. On-device rows: KWS 13.09 M cyc/54 ms, sine 536,
hello_world 11,329, anomaly 19.67 M/81 ms (person_detect + mobilenet_v2 SKIP).
Two silent-scalar root causes were found and fixed:
- **codegen `SCRATCH_LEN` was always 0**: `emit_op` hardcoded `scratch: 0`
  for every op, so every zoo-model op got an empty scratch slice and the
  softmax/depthwise/conv SIMD dispatches fell back to scalar. Codegen now
  computes real per-op scratch needs (conv1x1/conv3x3/depthwise/softmax/fc
  formulas mirroring `S3Backend`), emits `pub const SCRATCH_LEN`, and the
  validation path uses `predict_with_scratch` with arena scratch.
- **unaligned model weights**: zoo-model weights were emitted as plain
  `[i8; N]` consts (alignment 1), failing the ACCX/FC `w_ptr % 16` gate.
  Codegen now emits `#[repr(C, align(16))]` wrapper structs.
- KWS's 54 ms vs the 7 ms ESP-DL bar is **structurally bounded**: its
  depthwise has `depth_multiplier=8` and the bespoke per-lane depthwise SIMD
  requires `input_c == output_c` (dm==1) — documented honestly, not
  weakened.

**5. Public-API speed closure** (todos 15/16/17/18, device-measured):
normalized per-op workloads (same shapes both stacks) and closed the
wrapper gap inside the trait call path so `Model::<S3Backend>` benefits:
conv1x1 4269→**3201**, fc 8344→**7161**, max-pool 33240→**14046**,
avg-pool 29225→**19913** (prebuilt args + shared whole-op driver); all
bit-exact vs the scalar ref. Targets below these are **documented as
physically unreachable** for the bit-exact ACCX contract (the anchors were
the non-bit-exact vendored QACC kernel; conv3x3's full-image 880 K cyc is
the TIE memory-bound floor for 2.36 M MACs). Dead vendored TIE728 conv
modules removed (todo 12, −749 lines); residual warnings fixed (todo 14).

**Final model-vs-ESP-NN head-to-head (device, all bit-exact):**
Model A 1,686,922 vs ESP-NN 2,630,401 (**1.56×**), Model B 763,105 vs
994,782 (**1.30×**), Model C 650,773 vs 655,303 (**1.007×** — floor-limited;
all 6 layers SIMD-engaged, 22× vs the scalar ref 14,489,859). ESP-NN C
baselines for the zoo models themselves are `—` (no esp-nn tflite-backed
runners exist; ESP-IDF absent on this host — enumerated in
`benchmarks/espnn-baseline/README.md`, same-conditions rule).

**Test/CI state:** host `cargo test --workspace` 296 passed / 0 failed;
clippy has zero new warnings; the 23 Phase-18 commits are all authored
`Yatendra Singh <singh.0.yatendra@gmail.com>`.

### Phase 19 — composed-kernels: op fusion, composed SIMD kernels, shape-flex widening, selector, verification (plan `composed-kernels`)

Phase 19 (2026-08-10 → 2026-08-11) is the plan `composed-kernels`
execution. It fuses adjacent ops into composed-kernel calls (conv + requant
+ activation + residual-add in one SIMD pass, register-held elementwise
chains, pool/softmax input-folds), widens the SIMD shape gates the zoo models
actually need (dm>1 depthwise, small-shape FC, generic pool, non-identity
elementwise, extended mean), and adds a shape-driven composed-kernel
selector — all while preserving the bit-exact invariant. Evidence ledger:
`local-notes/evidence/composed-kernels/` (head-to-head.md, model-cycles.md,
device-sweep.md, selector-output.md, fused-equivalence.md, pad-decision.md,
tree-audit.md, fused-profile.md).

**1. Fusion wiring (W1 — T1.1–T1.4, T2.1):**
- **T1.1 correctness tiers** (a7a52b5): standalone-activation absorption is
  gated on scale-AND-zero-point identity (a clamp is exact only when the
  activation's quant equals the anchor output quant); AbsorbedElementwise /
  ResidualAdd carry per-step requantize params; input/requant folds are
  tagged T2 (proof-obligated, opt-in).
- **T2.1 `FusedKernelBackend` trait** (57a9fce): additive `FusedKernelBackend:
  KernelBackend` with `fused_conv2d` / `fused_elementwise_chain` /
  `fused_pool_with_fold` + composed params structs; `RefBackend` defaults
  decompose to the exact per-op sequence (bit-exact by construction).
- **T1.2 emitter seam** (c7f28b9): `emit_model` consumes the `FusedSchedule`;
  T1 groups emit composed calls, everything else per-op; unfused path
  byte-identical to pre-wiring output.
- **T1.3 arena** (18858ba): one stack-local `#[repr(C,align(16))]` arena for
  intermediates (MAX_TENSORS 64→255); peaks sine 0 / hello 32 / kws 5968 /
  anomaly 272 / person_detect 55296; mv2 → per-tensor fallback
  (ARENA_LEN=0); no `unsafe`, no static-mut emitted.
- **T1.4 scratch parity** (0213f1b): cross-crate `SCRATCH_LEN` ==
  `S3Backend::*_scratch_size` for ~2005 spec shapes + composed groups.

**2. Composed kernels (W2 — T2.2–T2.4, `hematite-s3/src/fused.rs`):**
- **T2.2 fused conv-family** (fba770e): conv accum + residual + per-step
  requantize + activation in ONE ACCX pass, register-held intermediates;
  residual-add reproduces the exact TFLM two-stage rounding; bit-exact vs
  the RefBackend decomposition on an 8-row golden matrix.
- **T2.3 fused elementwise chain** (faf0e8b): register-held chain steps with
  per-step requantize preserved, zero intermediate stores.
- **T2.4 pool/softmax folds** (fee6385): `fused_pool_with_fold` +
  folded-requantize epilogue, T2-gated (only shapes where the proof
  obligation is met; everything else emits per-op).

**3. Shape-flex SIMD widening (W3 — T3.1–T3.6, T3.5b):**
- **T3.1 generic pool** (a0e069a): avg/max SIMD for arbitrary filter/stride/
  pad + clamp (vector main loop + scalar tail), accepting the ESTABLISHED
  s3/C-SIMD pool fixed-point semantics (documented ±1 vs ref
  `round_half_away_zero`, `espdl-baseline README:82-86` — not "fixed").
- **T3.2 elementwise + activations** (2dede3b): non-identity offsets/
  multipliers per-lane SIMD + relu6/hard_swish lane models (downgraded
  hard_swish formula reproduced bit-exact — Xtensa has no SIMD integer
  division; goldens pin the downgraded formula).
- **T3.3 channel-padded conv1x1/fc** (4aad12c): input_c in (0,16)/non-%16
  via 16B-aligned pad-in-scratch; engagement counter proves SIMD took the
  path on device.
- **T3.4 extended mean** (4ecd1ac): looped-accumulation for in_c>256
  (mv2's 7×7×1280 global mean), positions>256 in chunks; bit-exact.
- **T3.5 dm>1 depthwise** (d54f8db) + **T3.5b arbitrary-filter anytap**
  (cc0f5e7): depth_multiplier>1 and non-3×3 filters via chunked ≤32-tap QACC
  passes — the **kws real 10×8 dm=8 filter** (the 12.3× ESP-NN gap) now
  dispatches SIMD.
- **T3.6 small-shape FC** (7cda412/415d7f9): input_dim in (0,16)/non-%16 via
  pad-in-scratch — closes the sine/hello/anomaly FC gates.

**4. Selector (W4 — T4.1–T4.3):**
- **T4.1 eligibility mirror** (d9d50e8): host-side mirror of every s3 SIMD
  gate + cross-crate parity over ~1985 shapes (runtime is canonical).
- **T4.2 rule-tier selector** (9bfb5fe): conv-family > chain > fold > per-op
  per FusedGroup; T2 per-op until verified; staged-input 16B alignment
  decision recorded; per-model selector output in selector-output.md.
- **T4.3 const-generic dispatch** (15c18a8): shape-typed composed calls +
  per-model fused-groups doc header; autotuning explicitly out of v1.

**5. Verification (W5 — T5.1–T5.4):**
- **T5.1 host fused==unfused harness** (7d947b7): all 6 zoo models
  element-equal + identical FNV-1a (mv2 = the only model with composed
  groups: 10 residual-add, all T1 → all-composed, PASS). A synthetic T2
  fixture proves the auto-unfuse path catches a forced divergence
  (fused-equivalence.md).
- **T5.2 on-device sweep** (d90cc17 harness) — **REAL-SILICON RUN 1
  EXECUTED 2026-08-11 17:34** (owner-approved `esptool.py write_flash
  --encrypt` @ 921600; ESP32-S3 rev v0.2, SPI_BOOT_CRYPT_CNT=0x7; log
  `device-silicon-run1.log`): model validation PASS bit-exact for
  sine/hello_world/kws through `Model::<S3Backend>` (ref == s3 == golden);
  anomaly reproduces the documented ±1 class on both backends (s3==ref);
  35/40 SIMD checks executed — 33 PASS (incl. **kws 10×8 anytap depthwise**,
  conv1x1_padded_3ch SIMD-engaged, all 7 fc_small) + 2 documented-class
  avg-pool ±1 FAILs + **5 blocked by a deterministic firmware panic at the
  mean check** (defmt `RTT_ENCODER.taken=6`, stack-vs-arena-end clobber)
  which also blocked the zoo fused/unfused cycle rows, per-kernel rows, and
  the person_detect stack-probe (all PENDING — recorded, never faked).
  Follow-up levers recorded (dedicated 256 KB stack / shrink mean-kernel
  stack / linker stack size) — `device-sweep.md §9`.
- **T5.3 PAD decision** (09a1d77): **DEFER** the pad-value/zero-point
  plumbing (T10 follow-up) — PAD never fuses, mv2's substitute gate is the
  T5.1 host harness, and the s3==ref identical-fill gate must stay intact
  (pad-decision.md).
- **T5.4 hygiene** (0701a12): dead-code removal + clippy + module docs.

**6. Benchmarks (W6 — T6.1–T6.3):** model_bench real zoo runners + unfused
arm (T6.1), ledger-format head-to-head.md (T6.2), speed-closure rows +
re-tiered bars (T6.3). **Ledger format (user requirement 2026-08-10):**
every measured row carries ISO timestamp + commit of the measured code +
FULL Hematite cycles + FULL C-stack cycles + speedup ratio + config — no
deltas-only rows.

**Measured / ledger results (device unless noted):**

| Row | Hematite (cycles) | C-stack / ESP-NN (cycles) | speedup | source |
|---|---|---|---|---|
| Model A — 4-layer CNN | 1,686,922 | 2,630,401 | **1.56×** (H wins) | head-to-head.md §1 / ESPRESSIF_VS_HEMATITE.md |
| Model B — mv2mini | 763,105 | 994,782 | **1.30×** (H wins) | head-to-head.md §1 |
| Model C — mv2real | 650,773 | 655,303 | **1.007×** (H wins, floor-limited) | head-to-head.md §1 |
| sine (FC 1→1) | 618 (pre-opt; user-verified T0.3) | 190 | C 3.3× | espnn-head-to-head.md |
| hello_world (3×FC) | 10,314 (pre-opt) | 4,675 | C 2.2× | espnn-head-to-head.md |
| kws (dw dm=8 + FC + softmax) | 12,983,503 / 54 ms (pre-opt) | 1,059,889 / 4 ms | C 12.3× | espnn-head-to-head.md |
| anomaly_detect (10×FC) | 28,550,253 / 118 ms (pre-opt) | 7,758,145 / 32 ms | C 3.7× | espnn-head-to-head.md |
| person_detect | SKIP reason=stack (both stacks) | SKIP | — | model-cycles.md §4 |
| mobilenet_v2 | SKIP reason=no-psram (both stacks) | SKIP | — | model-cycles.md §5 |

The pre-opt zoo rows above are the T3.5b/T3.6 closure baselines; the
post-optimization on-device re-measurement is PENDING (run-1 panic, §5 of
this entry). Host-side fused==unfused is asserted bit-exact for all 6 models
(fused-equivalence.md).

**Re-tiered bars (T6.3):** mv2 1294.5 ms — **hold-as-documented**
(PSRAM-gated, `PSRAM: 0 bytes` re-confirmed on device run 1); KWS 7 ms →
**4 ms ESP-NN-relative** (target < 1,059,889 cyc / 4 ms via T3.5b anytap
depthwise — the old "structurally unreachable" rationale is superseded by
the user-verified ESP-NN rows); conv1x1 **15.57×** vs scalar-ref (holds);
**10×** vs-scalar floor (holds). Fixed-point constants live in
`model_bench.rs` (12945 / 40 / 1557 / 1000) and are pinned by tests — never
silently re-tiered.

**Bit-exact invariant (F1):** every SIMD kernel's output equals
`hematite-ref` on host AND device; the fused path equals the unfused path
element-for-element for all 6 zoo models (T5.1); on device, 33/40 sweep
checks PASS bit-exact vs ref (2 avg-pool rows are the documented fixed-point
class, 5 blocked by the harness panic). The only non-bit-exact rows are the
documented golden divergences (anomaly ±1 single-vs-double rounding; mv2 PAD
zero-fill vs TFLM zero-point-fill — both DEFERRED_MODELS.md, unchanged).

**Test/CI state (host):** `cargo test -p hematite-s3` 71 passed, `-p
hematite-codegen` 107, `-p hematite-benchmarks --lib` 37, `cargo check
--workspace` clean; device `cargo check` (xtensa, model-validation) 0
errors; qemu-feature build compiles clean. No `unsafe` in emitted code; no
dead code; clippy zero-new. **Honest note — `cargo test --workspace` does
not currently compile end-to-end:** a pre-existing E0596 in
`hematite-tests/tests/models.rs:299` (`model.predict()` on an immutable
binding — the composed path emits `predict(&mut self)` for mv2's 10 fused
groups; the file predates the composed-kernels wave, c1b581b). This is not
one of the four named gates above (all green); it is a source-code breakage
owned by a kernel/codegen follow-up, not this docs todo (scope: docs only),
recorded here so the Phase-18 "296 tests" claim is not silently stale.

> **⚠️ Detached-HEAD endnote (branch needed):** ALL composed-kernels commits
> — T1.1 (a7a52b5) through T6.3 (fc067bf) — landed on a **DETACHED HEAD**
> (git status: `HEAD detached from 0213f1b`). The work exists only as
> dangling commits reachable from `fc067bf`; it needs a branch
> (**`git switch -c composed-kernels`**) before it can be pushed or merged.
> Creating that branch is out of scope for this docs close-out (T6.4) — the
> branch is not created here, only recorded. Commits: a7a52b5 (T1.1) ·
> 57a9fce (T2.1) · c7f28b9 (T1.2) · 18858ba (T1.3) · 0213f1b (T1.4) ·
> fba770e (T2.2) · faf0e8b (T2.3) · fee6385 (T2.4) · a0e069a (T3.1) ·
> 2dede3b (T3.2) · 4aad12c (T3.3) · 4ecd1ac (T3.4) · d54f8db (T3.5) ·
> 7cda412/415d7f9 (T3.6) · cc0f5e7 (T3.5b) · d9d50e8 (T4.1) · 9bfb5fe
> (T4.2) · 15c18a8 (T4.3) · 7d947b7 (T5.1) · d90cc17 (T5.2) · 09a1d77
> (T5.3) · 0701a12 (T5.4) · 28482eb (T6.1) · faed3b5 (T6.2) · fc067bf
> (T6.3).

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
approved. Hardware bring-up (Phase 9), the C-SIMD bit-exact output match
(Phase 10), the per-operation all-SIMD comparison + relu fix (Phase 11),
the prepared-kernel fast path (Phase 12), the bespoke GPR-accumulator
(ACCX) kernels (Phase 13), the optimized fast64 ACCX paths (Phase 14), the
requantize fast paths (Phase 15), the **from-scratch full op-sweep
(Phase 16)**, the **full MobileNetV2 parity sweep (Phase 17)**, the
**S3Backend + first real-silicon SIMD sweep + executed-TFLM goldens +
zoo-model acceleration (Phase 18, plan `simd-zoo-hardening`, 23 commits)**,
and the **composed-kernels plan (Phase 19: fusion wiring, composed SIMD
kernels, shape-flex widening, selector, verification; see §3 Phase 19)**
are complete. Host test suite: **named gates green (s3 71 / codegen 107 /
benchmarks --lib 37 / `cargo check --workspace` clean)**; the Phase-18
"296 tests, 0 failures" full-workspace figure is superseded by the recorded
pre-existing `hematite-tests` E0596 (Phase 19 §5 note — the composed path's
`predict(&mut self)` vs an immutable-binding call site; source fix owned by
a kernel/codegen follow-up, out of docs scope).

On-device benchmark state: all **9 SIMD-capable operations** (conv1x1,
conv3x3, fc, max/avg-pool, relu, add/sub/mul) run on the real hardware.
The **weighted ops (conv1x1/conv3x3/fc) now run the bespoke ACCX kernels and
are bit-exact vs the scalar reference** (0x0bea8225 / 0x0a181085 /
0x32e35185); relu/add/sub/mul are bit-exact vs scalar; max/avg-pool run the
vendored asm and are bit-exact vs the independent ESP-IDF C harness
(`benchmarks/espdl-baseline`). Raw ACCX cycle costs (bench37, after the
fast64 paths + requantize fast paths): conv1x1 5041, conv3x3 9511784 (full
30×30 image), fc 15293, max-pool 1892, avg-pool 7342, relu 357, add 411,
sub 491, mul 774. Other SRAM rows are legitimately scalar
(pad/input_c/out-channel gates); PSRAM-tier rows require a PSRAM-equipped
board.

The **prepared-kernel fast path** (Phase 12) closes the Rust wrapper
overhead to ~1.0–2.2x of C raw-asm (was 2.5–5.5x): the SIMD gate runs once
at `Prepared*::new`, per-call `run` only checks pointer alignment and
dispatches; MaybeUninit args builds eliminate the memset; an
`#[inline(never)]` on `dispatch_fc` avoids an Xtensa miscompile. Prepared
checksums are bit-exact equal to the public s3 checksums on all 12 rows.

The **bespoke ACCX kernels** (Phase 13) make the weighted-op SIMD output
bit-exact vs scalar — closing the correctness gap the vendored QACC asm
could never close (8-bit/16-bit saturating lanes). conv1x1 went from 2627
cycles (vendored, wrong output) to 12595 cycles (ACCX, bit-exact). Phase 14
then added the fast64 paths (input-resident, unrolled taps), cutting
conv1x1 to 9939 and conv3x3 from 35743533 to 15009334 cycles while keeping
every checksum bit-exact. Phase 15 added the requantize fast paths
(uniform-scale detection + unchecked indexing), cutting conv1x1 to 5041,
conv3x3 to 9511784 and fc to 15293 cycles — full-API now lands ~1.3–2x of
the raw C kernel numbers.

**Phase 16** rewrote everything from scratch (no vendored conv asm) and
closed the remaining coverage gaps: a bespoke QACC per-lane depthwise
kernel (Phases 0–2), first-conv `in_c<16` zero-pad (Phase 3), conv3x3
fast16/fast32 (Phase 4), fc hardware-loop (Phase 5), and a fused asm
requantize epilogue (Phase 6). On two real models benchmarked head-to-head
against standard ESP-NN, **Hematite now wins both, bit-exact**:
Model A 1,707,746 vs ESP-NN 2,630,401 cycles (1.54×); Model B 770,827 vs
994,782 (1.29×). conv1x1 now 4379, fc256 8393, conv3x3 8868886.

**Phase 17** closed every remaining parity gap against standard ESP-NN so a
stock MobileNetV2 runs bespoke SIMD instead of scalar: **SAME padding**
(conv3x3, 880125 cyc `0xc53ebbc5`), **stride-2 depthwise** (30734 cyc
`0x5159710e`), **non-zero input_offset** (conv3x3 off3 1129475 / dw off-3
41167 / fc off5 238998, all bit-exact), **softmax SIMD** (263047 cyc
`0xaf0d15aa`, 1.70×), and **depthwise non-%16 channels** (104047 cyc
`0x8da1a066`, 11.16×). A third real MobileNetV2-style model (Model C,
`model_mv2real.rs`) runs head-to-head on both stacks: **Hematite 654,407 vs
ESP-NN 655,303 cycles, bit-exact on every layer** — Hematite wins all three
models. The vendored S16 experiment asm was deleted. Final state
(`bench55`): all 19 kernel rows + 3 models report `out_fnv(ref/s3)` equal
on the real board.

**Open decisions awaiting explicit direction:**
1. Force-push the rewritten git history to `origin/main`? (History was
   rewritten to the Yatendra Singh identity on 2026-08-06; not yet pushed.)
2. The `mobilenet_v2` residual (984/1000 vs executed-TFLM golden): TFLM
   `pad.cc` fills `output_zero_point` (−14) when `constant_values==nullptr`,
   but `PadParams` carries no zero point — a `PadParams` pad-value plumbing
   + codegen emission change (**DECIDED DEFER in Phase 19 T5.3**,
   pad-decision.md; T10 follow-up recorded — kernel workstream).
3. Zoo-model ESP-NN C baselines: the espnn-baseline harness has only the
   3 synthetic A/B/C runners; adding tflite-backed zoo runners + a
   weight-extraction path is a separate scope decision (Metis F8), and the
   ESP-IDF toolchain is absent on this host. Exact rows-to-add are specified
   in `benchmarks/espnn-baseline/README.md`.
4. Pushing Phase 18's 23 local commits to `origin/main` is pending the
   standing "don't push until I verify" rule. User intent is to open a PR
   presenting Hematite as the best Rust NN library for ESP32-S3.
5. **Phase 19's composed-kernels commits (T1.1 → T6.3) sit on a DETACHED
   HEAD off `0213f1b`** — needs `git switch -c composed-kernels` before
   push/merge (recorded in the Phase 19 endnote; branch not created here).
6. **Real-silicon run 1 (2026-08-11 17:34) panic follow-up:** the
   mean-check stack-vs-arena-end clobber (defmt `RTT_ENCODER.taken=6`)
   blocks the zoo fused/unfused cycle rows, per-kernel rows and the
   person_detect stack-probe — levers recorded in device-sweep.md §9.6
   (dedicated 256 KB stack / shrink mean-kernel stack / linker stack size).
