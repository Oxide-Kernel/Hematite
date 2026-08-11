# DEFERRED_MODELS — models excluded from T5.0 golden capture

This file documents every model that is NOT covered by T5.0 per-model golden
outputs, with the exact reason. These are documented exclusions, not silent
drops (plan T5.2, B5 resolution).

## Executive summary

| Model | Status | Reason |
|---|---|---|
| All 9 esp-dl model dirs (15 `.espdl` files) | **Deferred (artifacts landed)** | Format barrier: esp-dl v3.x ships only proprietary `.espdl` (EDL2/FlatBuffers), zero `.tflite` in repo/history/releases — cannot be executed by TFLite/TFLM interpreter |
| All 6 edge-ml models | **Deferred (unavailable)** | `github.com/edge-ml/edge-ml-model-zoo` does not exist; the real `archimedes-market/edge-ml-model-zoo` repo is metadata-only (single commit, no model binaries, no releases) |
| `speaker_verification` | **Deferred (plan-specified)** | P4/S31-only, no S3 build (plan T5.2, recorded here for completeness) |
| `pp_ocr_v6` (det/rec_s8/rec_s16) | **Deferred (plan-specified)** | P4-only (plan T5.2, recorded here for completeness) |
| `motion_detect`, `color_detect` | **Out of scope by definition** | Algorithmic (no `.espdl`/`.tflite` artifact) — noted not tested (plan T5.2) |
| `anomaly_ae` (substitution) | **⚠️ compile+execute, documented ±1 class** | 210/640 elements differ by exactly ±1 — gemmlowp double-rounding vs hematite single-rounding (§6). Reproduced identically on real silicon (run 1, 2026-08-11: `FAIL at idx 1: got=41 want=42`, s3==ref, fnv 0xf2a76cd6) |
| `mobilenet_v2` (substitution) | **⚠️ compile+execute, PAD-fill class** | 984/1000 elements differ (890 PAD-fill, 94 rounding) — `PadParams` carries no pad-value/zero-point. **Phase 19 T5.3 decision: DEFER the plumbing** (T10 follow-up, pad-decision.md) (§7) |

## 1. ESP-DL models — format barrier (`.espdl`, not `.tflite`)

**Models affected (all 15 artifacts, downloaded into `models/zoo/`):**

| Model | Artifacts landed | Format |
|---|---|---|
| cat_detect | 2 | `.espdl` |
| dog_detect | 2 | `.espdl` |
| hand_detect | 1 | `.espdl` |
| human_face_detect | 4 (espdet 224/416, mnp, msr) | `.espdl` |
| human_face_recognition | 2 (mbf, mfn) | `.espdl` |
| hand_gesture_recognition | 1 | `.espdl` |
| pedestrian_detect | 1 | `.espdl` |
| person_reid | 1 | `.espdl` |
| imagenet_cls | 1 | `.espdl` |

**Evidence (verified 2026-08-05 against espressif/esp-dl @ `12c0616d`):**
- `git ls-tree -r` across ALL branches and tags (v0.1.0 → v3.3.9): zero `.tflite` files.
- `git rev-list --all` full-history scan: zero `.tflite` files ever committed.
- All GitHub Releases (v3.2.0, v3.0.0, v2.0): no assets.
- `.espdl` files start with the `EDL2` magic — a custom FlatBuffers IR
  (`esp-dl/fbs_loader/espdl.fbs`, `IR_VERSION_2023_12_22`), not a TFLite
  flatbuffer. There is no embedded TFLite sub-model.
- esp-dl docs confirm: models are quantized from ONNX/PyTorch via the
  proprietary `espdl_quantize_onnx` / `espdl_quantize_torch` interfaces.

**Consequence:** per-model golden outputs via the TFLite/TFLM Interpreter
(the only non-hand-computed reference path permitted by T5.0) are impossible
for these artifacts. Converting `.espdl` → `.tflite` requires the original
ONNX/PyTorch source + ESP-PPQ quantization, which esp-dl does not distribute
for the zoo models.

**Deferred action (recommended):** T5.2 on-device model verification can run
these `.espdl` binaries through ESP-DL's own loader on ESP32-S3 hardware and
compare against ESP-DL's C++ reference — a device-side cross-check path
outside the TFLite golden mechanism.

## 2. edge-ml models — unavailable anywhere public

**Models affected:**

| Requested name | Actual repo name (models.json) |
|---|---|
| keyword_spotting_v1 | `kws_micro_v2` |
| person_detect_v2 | `person_detect_v3` |
| hand_gesture_v3 | `gesture_5class_v1` |
| anomaly_detect_v2 | `anomaly_ae_v1` |
| vad_micronet_v1 | `vad_micro_v2` |
| defect_binary_v1 | `defect_binary_v1` |

**Evidence (verified 2026-08-05):**
- `github.com/edge-ml/edge-ml-model-zoo` → 404. The `edge-ml` GitHub org
  hosts only the unrelated Microsoft EdgeML framework repos.
- The real repo `archimedes-market/edge-ml-model-zoo` is a **single commit**
  (`f5933d7` "Initial commit"), one branch (`main`), no tags, no releases.
- Full git history contains **zero** `.tflite`/`.espdl`/`.onnx`/`.h5` files —
  only `README.md`, `LICENSE`, `models.json`, and 2 example sources.
- `models.json` describes the 6 models (input shapes, class lists, int8 sizes)
  but contains no URLs. README claims models "ship in three formats" but the
  binaries are not distributed in the repo, on HuggingFace, or via the linked
  Archimedes Market asset page (landing page only; its MCP API has no
  file-download tools).
- No forks contain the model binaries.

**Consequence:** no download URL exists to try; the models cannot be
obtained for golden capture or T5.2 compilation.

## 3. Plan-specified exclusions (recorded for completeness)

- **speaker_verification**: P4/S31-only, no S3 build (plan T5.2).
- **pp_ocr_v6** (det / rec_s8 / rec_s16): P4-only (plan T5.2).

## 4. Algorithmic models (out of NN-engine scope by definition)

- **motion_detect**, **color_detect**: algorithmic, no `.espdl`/`.tflite`
  artifact — noted not tested here (plan T5.2).

## 5. T5.2 substitutions — REAL public int8 .tflite models

The plan's 18-model list was unobtainable as `.tflite` (sections 1–2), so T5.2
obtained real public int8 models covering the same op families. **Per-op
families table** (plan model → substitution → ops exercised → status):

| Plan family (T5.2) | Substitution | Source | Ops | Status |
|---|---|---|---|---|
| `person_detect_v2` | `vww_96_int8.tflite` (VWW person detector) | mlcommons/tiny | conv, depthwise, avgpool, reshape, fc, softmax | ✅ **bit-exact** vs executed-TFLM golden (todo 11; fnv1a 0x6962079d) |
| `keyword_spotting_v1` | `micro_speech_quantized.tflite` | tflite-micro @ pin | reshape, depthwise, fc, softmax | ✅ bit-exact |
| `imagenet_cls` / `mobilenetv2_cls` | `mobilenet_v2_quantized_1x3x224x224.tflite` | tflite-micro @ pin (xtensa) | transpose, pad, conv, depthwise, add, mean, reshape, fc, softmax | ⚠️ compiled, not bit-exact (PAD fill; 984/1000 deltas — §7) |
| `anomaly_detect_v2` | `ad01_int8.tflite` (MLPerf AD01 AE) | mlcommons/tiny | fc ×10 | ⚠️ compiled, not bit-exact (210/640 ±1 double-rounding — §6) |
| (sine regression) | `hello_world_int8.tflite` | tflite-micro @ pin | fc ×3 | ✅ bit-exact |

All 5 have per-directory SHA256 provenance READMEs under `models/zoo/`.
Goldens captured from executed interpreters: ai-edge-litert 2.1.6
(`tools/generate_goldens/zoo/run_model.py`) for person_detect / kws /
hello_world; regenerated (todo 10) from EXECUTED tflite-micro at the pinned
SHA (`tools/tflm-goldens` harness) for mobilenet_v2 / anomaly_detect.

## 6. Residual bit-exactness vs the EXECUTED-TFLM goldens (todo 10 regeneration + todo 11 re-verification)

All five substitution models were re-verified (todo 11, host) against the
goldens regenerated (todo 10) from EXECUTED tflite-micro at the pinned SHA
`18b9e6f2a8c5a9518e588f59c2ba16ef7ef9d551` (`tools/tflm-goldens` host harness,
reference kernels). Post-regeneration per-model status:

| Model | Verdict | Evidence |
|---|---|---|
| `person_detect_vww` | ✅ **bit-exact** (upgraded from compile+execute) | output `[120, -120]` element-for-element == executed-TFLM golden; fnv1a `0x6962079d` == executed-TFLM harness checksum |
| `anomaly_ae` | ⚠️ compile+execute (converted by todo 10) | 210/640 elements differ by exactly ±1 |
| `mobilenet_v2` | ⚠️ compile+execute | 984/1000 elements differ (890 by \|d\|≥3, PAD-fill driven; 94 by ±1/±2, rounding) — §7 |
| `kws_micro_v2`, `hello_world`, `sine` | ✅ bit-exact (unchanged) | — |

The original T5.2 root-cause record (kept for history):

1. **Requantization rounding**: the hematite kernels implement TFLM
   single-rounding `MultiplyByQuantizedMultiplier` (the 64-bit
   `(x*mult + round) >> shift` form, `TFLITE_SINGLE_ROUNDING` path). The
   executed-TFLM build at the pinned SHA uses the **double-rounding** form
   (`SaturatingRoundingDoublingHighMul` + `RoundingDivideByPOT`, gemmlowp
   path; `TFLITE_SINGLE_ROUNDING` is undefined in the micro build). The two
   agree except at exact rounding boundaries, where they differ by ±1. This
   is now the ONLY residual for `anomaly_detect` (210/640 elements, exactly
   ±1) and the small (94/1000) rounding class of `mobilenet_v2`.
2. **Softmax**: the T5.2 investigation found the TFLM reference int8 softmax
   saturates wide-dynamic-range logits to −128 while LiteRT produces
   [120,−120]. **Resolved by regeneration**: the EXECUTED TFLM at the pinned
   SHA produces [120,−120] on person_detect's logits — matching both the
   hematite kernels AND the LiteRT golden (hash-identical golden, fnv
   0x6962079d) — so this divergence does NOT manifest and person_detect is
   bit-exact (verified todo 11).

**Why the two ⚠️ models are not fixed here**: the rounding difference lives in
`hematite-ref` kernels (`MultiplyByQuantizedMultiplier`), owned by the kernel
workstream and out of scope for this task (MUST-NOT: no kernel changes). The
PAD-fill difference is a param-struct limitation (§7). The emitter/parser
produced bit-exact parameter streams (verified op-by-op).

## 7. Emitter/parser gaps closed by T5.2 (for the record)

The zoo models exposed three real emitter/parser gaps, all now implemented and
tested (`cargo test -p hematite-codegen` → 55 tests):

- **Legacy opcode encoding**: xtensa models store the code in
  `deprecated_builtin_code` (field 0) and omit it entirely for ADD (schema
  default) — `parse_opcodes` now resolves missing fields to ADD.
- **PAD (34) + TRANSPOSE (39)**: both emit consts from their const int32
  input buffers (pad amounts `[rank,2]`, perm `[rank]`).
- **Multi-axis MEAN**: `ParsedOptions::Mean.axis` is now `Vec<i32>` with
  `axis_count` (MobileNetV2 global-average-pool reduces over axes [1,2]).
  Also: softmax accepts a `beta = 1.0` options table; RESHAPE falls back to
  the output shape when the options carry 0/−1 dims.

**Note**: the mobilenet_v2 model's 18 PAD ops additionally expose a PAD
kernel-semantics difference, now the DOMINANT residual vs the regenerated
executed-TFLM golden (measured todo 11, element-wise on the final output):
**984/1000** elements differ — 890 by |d| ≥ 3 (max 60), 94 by ±1/±2. Root
cause: TFLM `pad.cc` @ pinned SHA fills `output_zero_point` (−14) ONLY when
`constant_values == nullptr` (true for all 18 mv2 PADs); Hematite's
`PadParams` carries no zero point and the trait `pad(src, params, dst)` has
no pad-value arg, so ref + s3 fill raw 0. The zero-fill propagates through
the conv chain and dominates the output; the ±1/±2 class is the §6 rounding
divergence. The zero-fill was never a unilateral kernel choice — fixing it
requires **param plumbing**: a pad-value / zero-point field on `PadParams`
(hematite-core) + codegen emission, which is a documented follow-up (recorded
in `hematite-s3/src/data_movement.rs` module doc + todo 25 evidence; NOT
attempted in todo 11 — kernel/param code is out of scope). Both backends
share the identical raw-0 fill, so the relative s3 == ref gate holds exactly.
(An earlier 861/1000 estimate from todo 25 is superseded by this direct
measurement.)

## 8. Phase 19 (composed-kernels T5.3) — PAD plumbing decision + real-silicon confirmation

### 8.1 PAD zero-point plumbing — DECISION: DEFER (T5.3, pad-decision.md)

The pad-value/zero-point plumbing (`PadParams` field + codegen emission +
both backends) is **deferred** and recorded as an explicit follow-up
(**T10**, kernel workstream) — it is NOT implemented in this plan. Rationale
(recorded in full in `local-notes/evidence/composed-kernels/pad-decision.md`):

1. **PAD never fuses** (no fusion pattern covers it) — the composed-kernels
   workstream neither emits nor transforms any PAD call, so the fused path
   is orthogonal to the fill semantics.
2. **A substitute gate already covers mv2's claim:** the T5.1 host harness
   asserts fused==unfused element-equal + identical FNV-1a for mv2 (both
   arms share the same raw-0 fill), which verifies the composed-param
   derivation — the only thing this plan changes.
3. **Deferral preserves the s3==ref identical-fill gate** — plumbing a zero
   point through one backend only would break it; the full change must land
   atomically with golden regeneration (T10).

**Consequences (honest):** mv2 stays ⚠️ compiled-not-bit-exact vs the
executed-TFLM golden (984/1000 deltas; 890 PAD-fill class, |d|≥3, max 60;
94 rounding class) until T10 lands. "Bit-exact vs TFLM" must not be claimed
for PAD-heavy models; only the relative ref↔s3, fused↔unfused claims hold.

### 8.2 Real-silicon confirmation (run 1, 2026-08-11 17:34)

The two ⚠️ models' divergences were re-confirmed on real silicon through the
fused `Model::<S3Backend>` path (log `local-notes/evidence/composed-kernels/
device-silicon-run1.log`):

| Model | device run-1 result | consistent with |
|---|---|---|
| `anomaly_detect_int8` | `FAIL at idx 1: got=41 want=42` (fnv 0xf2a76cd6), **s3==ref (`ref_match=true`, golden_match=false)** | §6 documented ±1 double-rounding class — identical on device, QEMU and host; not a new divergence |
| `mobilenet_v2_1.0_224_int8` | `SKIP reason=no-psram` (board probe `PSRAM: 0 bytes` re-confirmed) | §7 — model cannot run without PSRAM; host-side fused==unfused substitute gate holds (T5.1) |

Both rows are recorded, never masked, and the relative gates (s3==ref /
fused==unfused) are asserted and green.
