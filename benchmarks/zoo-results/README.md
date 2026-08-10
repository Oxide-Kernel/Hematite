# zoo-results — on-device zoo model validation ledger (plan todo 5, Metis F10)

Row format (transcribed verbatim from the device serial log,
`local-notes/evidence/simd-zoo-hardening/task-5-device-s3-models.log`):

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
