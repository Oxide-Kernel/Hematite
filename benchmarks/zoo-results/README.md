# zoo-results — on-device zoo model validation ledger (plan todo 5, Metis F10)

Row format (transcribed verbatim from the device serial log):

```
PASS <model> <backend> <fnv1a>
FAIL <model> <backend> <fnv1a> (first mismatch idx: got/want)
SKIP <model> reason=<r> rerun_condition=<c>
```

Device: ESP32-S3 rev v0.2, NO PSRAM (`PSRAM: 0 bytes`), 8MB DIO 80MHz flash,
permanent flash encryption. `backend` = `ref` (hematite-ref scalar) or `s3`
(Model::<S3Backend> — on the device the S3 forwarding takes the real
TIE728/ACCX SIMD paths). Evidence: task-5-device-s3-models.log; the qemu
run (task-5-qemu-model-validation.log) supplements it where the device
cannot run a model (person_detect — see below).

## Rows (device, 2026-08-10)

```
PASS sine                      ref  0x050c5d1f
PASS sine                      s3   0x050c5d1f
PASS hello_world_int8          ref  0x010c56d3
PASS hello_world_int8          s3   0x010c56d3
PASS kws_micro_speech_int8     ref  0xbb84c615
PASS kws_micro_speech_int8     s3   0xbb84c615
FAIL anomaly_detect_int8       ref  0xf2a76cd6  (idx 1: got=41 want=42)
FAIL anomaly_detect_int8       s3   0xf2a76cd6  (idx 1: got=41 want=42; s3==ref)
SKIP person_detect_int8        ref  reason=stack rerun_condition=codegen-intermediates-off-stack
SKIP person_detect_int8        s3   reason=stack rerun_condition=codegen-intermediates-off-stack
SKIP mobilenet_v2_1.0_224_int8 ref  reason=no-psram rerun_condition=board-with-PSRAM
SKIP mobilenet_v2_1.0_224_int8 s3   reason=no-psram rerun_condition=board-with-PSRAM
```

## Notes per row

- **sine / hello_world / kws — PASS both backends.** S3 output is
  bit-identical to ref AND to the executed-TFLite golden (the S3 SIMD path
  agrees with the scalar oracle end-to-end on real silicon).
- **anomaly_detect — FAIL both backends, same fnv (0xf2a76cd6) and the same
  ±1 at idx 1.** This is the DOCUMENTED single-vs-double rounding
  divergence (hematite single-rounding vs the T10-regenerated executed-TFLM
  golden, DEFERRED_MODELS.md §6; 210/640 elements differ by exactly ±1) —
  identical on device, QEMU and host. The relative gate holds: s3 == ref
  (ref_match=true). Assertion intentionally NOT weakened.
- **person_detect — SKIP reason=stack on device; PASS under QEMU.**
  The generated predict allocas 232 KB of intermediates on the stack
  (`sub a1, a1, 0x38ac0` in the ELF); the device stack region is only
  ~65 KB (416 KB DRAM minus the 256 KB bench arena and model scratch
  statics). A dedicated arena-stack SP switch (256 KB, recorded) faults on
  real silicon: first windowed return after the switch → window-underflow
  exception (excvaddr=0, epc1=retw in core::fmt::write); QEMU's emulation
  is lenient, so the same code PASSES there (fnv=0x6962079d — bit-exact vs
  the executed-TFLM golden, both backends). Fix = codegen emitting
  intermediates into the scratch buffer (out of scope, see DEFERRED_MODELS
  / learnings). Not weakened: the SKIP is explicit with reason + rerun
  condition.
- **mobilenet_v2 — SKIP reason=no-psram.** 224×224 model needs ~4 MB PSRAM;
  board has none (probe: task-1-psram-probe.log). Rerun on a PSRAM board.

## Supplementary device finding (todo 8 territory)

After model validation the run proceeds into the SIMD-correctness section
(todo 8's first-ever device execution) and panics with
`defmt logger taken reentrantly` (defmt-rtt acquire) — a watchdog-interrupt
vs defmt-write race; recorded for todo 8, not part of this todo's rows.

---

# composed-kernels era — real-silicon run 1 (2026-08-11 17:34) + head-to-head ledger

## Run-1 device rows (fused `Model::<S3Backend>` path)

Source: real-silicon run-1 device capture (ESP32-S3 rev v0.2 @ 240 MHz,
8 MB flash, permanent flash encryption, `PSRAM: 0 bytes`). Measured code:
commit `fc067bf` (T6.3);
firmware built with `--features model-validation`; flashed via the
owner-approved `esptool.py write_flash --encrypt` @ 921600.

```
=== MODEL VALIDATION (executed-TFLite goldens) ===
model sine: PASS (fnv=0x050c5d1f)
model hello_world_int8: PASS (fnv=0x010c56d3)
model kws_micro_speech_int8: PASS (fnv=0xbb84c615)
model anomaly_detect_int8: FAIL at idx 1: got=41 want=42 (fnv=0xf2a76cd6)   ← documented ±1 class
model person_detect_int8: SKIP reason=stack rerun_condition=codegen-intermediates-off-stack
model mobilenet_v2_1.0_224_int8: SKIP (needs PSRAM; not present under QEMU)
=== MODEL VALIDATION S3 (Model::<S3Backend> vs ref + golden) ===
model sine [s3]: PASS (fnv=0x050c5d1f; matches ref, matches golden)
model hello_world_int8 [s3]: PASS (fnv=0x010c56d3; matches ref, matches golden)
model kws_micro_speech_int8 [s3]: PASS (fnv=0xbb84c615; matches ref, matches golden)
model anomaly_detect_int8 [s3]: FAIL at idx 1: got=41 want=42 (fnv=0xf2a76cd6; ref_match=true, golden_match=false)
model person_detect_int8 [s3]: SKIP reason=stack rerun_condition=codegen-intermediates-off-stack
model mobilenet_v2_1.0_224_int8 [s3]: SKIP reason=no-psram rerun_condition=board-with-PSRAM
```

Same verdicts as the 2026-08-10 rows above (bit-exactness is stable across
runs): sine / hello_world / kws PASS both backends; anomaly reproduces the
documented ±1 class (s3==ref); person_detect / mobilenet_v2 SKIP for the
recorded hardware reasons. In addition, the SIMD-correctness sweep ran 35/40
checks on device (33 PASS incl. kws 10×8 anytap depthwise, conv1x1_padded_3ch,
all 7 fc_small; 2 documented-class avg-pool ±1 FAILs; 5 blocked by a
deterministic mean-check panic — see `device-sweep.md §9` for the full run
record and the known follow-up).

## Head-to-head ledger (mandatory format: ISO timestamp + commit + full cycles both sides)

Full analysis: the head-to-head ledger is recorded per-row below.

### A/B/C benchmark graphs (device, 2026-08-10; fused path)

| Graph | arm | ISO timestamp | commit (measured code) | Hematite cycles (min/med) | C-stack cycles (min/med) | speedup (Hematite/C — H wins when >1) | config |
|---|---|---|---|---|---|---|---|
| A — 4-layer CNN | fused | 2026-08-10 | Hematite `33d498a`; C espnn-baseline T0.3 era | 1,686,922 / 1,686,922 | 2,630,401 / 2,630,423 | **1.56×** | SRAM; conv3x3 32×32×16 + maxpool + conv1x1 + FC; S3 SIMD; out_fnv `0x75eb32f5` |
| B — mv2mini 7-layer | fused | 2026-08-10 | Hematite `33d498a`; C T0.3 era | 763,105 / 763,105 | 994,782 / 994,782 | **1.30×** | SRAM; conv3x3 16×16×3 + pool + dw + 1x1 + dw + 1x1 + FC; out_fnv `0x7f23eb05` |
| C — mv2real 6-layer | fused | 2026-08-10 | Hematite `33d498a`; C T0.3 era | 650,773 / 650,773 | 655,194 / 655,303 | **1.007×** (floor-limited) | SRAM; SAME + stride-2 conv/dw blocks + FC; out_fnv `0x75eb32f5` |

### Zoo models (device, user-verified T0.3, 2026-08-10 — pre-optimization Hematite rows; cycle deltas PENDING-BLOCKED-BY-PANIC per `device-sweep.md §9`)

| Model | arm | ISO timestamp | commit (measured code) | Hematite cycles (min/med, pre-opt) | C-stack cycles (min/med) | speedup (C/H — C wins when >1) | config |
|---|---|---|---|---|---|---|---|
| sine | fused == unfused (W0: zero composed groups) | 2026-08-10 | Hematite pre-opt `model_bench`; C `3d74726` | 618 | 190 / 190 | C 3.3× | FC 1→1, SRAM; out_fnv `0x040c5b8c` all three stacks |
| hello_world | fused == unfused (W0) | 2026-08-10 | Hematite pre-opt; C `3d74726` | 10,314 | 4,675 / 4,675 | C 2.2× | 3×FC, SRAM; out_fnv `0xfaf3a2e1` |
| kws | fused == unfused (W0) | 2026-08-10 | Hematite pre-opt; C `3d74726` | 12,983,503 / 54 ms | 1,059,889 / 1,060,258 / 4 ms | C 12.3× (largest) | dw(dm=8)+FC+softmax, SRAM; out_fnv `0x2131fda5`; **dm=8 10×8 anytap depthwise bit-exact PASS on device run 1** (fnv 0x3ad38cac) |
| anomaly_detect | fused == unfused (W0) | 2026-08-10 | Hematite pre-opt; C `3d74726` | 28,550,253 / 118 ms | 7,758,145 / 7,758,250 / 32 ms | C 3.7× | 10×FC, SRAM; out_fnv `0xe8f86342` |
| person_detect | — | 2026-08-10 | — | **SKIP reason=stack** | **SKIP** | — | arena 55,296 B vs ~65 KB stack; probe PENDING (run-1 panic precedes it) |
| mobilenet_v2 | — | 2026-08-10 | — | **SKIP reason=no-psram** | **SKIP** | — | `PSRAM: 0 bytes`; bar 1294.5 ms hold-as-documented |

Ledger rule (user requirement 2026-08-10): every row carries FULL Hematite
cycles AND FULL C-stack cycles (never deltas only), the commit of the
measured code, and the config — as above. The post-optimization on-device
zoo cycle rows (T3.5b/T3.6 closures) are PENDING: real-silicon run 1
executed but the mean-check panic blocked the model-bench section
(`device-sweep.md §9`, follow-up levers §9.6).
