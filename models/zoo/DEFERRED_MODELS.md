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
| `person_detect_v2` | `vww_96_int8.tflite` (VWW person detector) | mlcommons/tiny | conv, depthwise, avgpool, reshape, fc, softmax | compiled; not bit-exact |
| `keyword_spotting_v1` | `micro_speech_quantized.tflite` | tflite-micro @ pin | reshape, depthwise, fc, softmax | ✅ bit-exact |
| `imagenet_cls` / `mobilenetv2_cls` | `mobilenet_v2_quantized_1x3x224x224.tflite` | tflite-micro @ pin (xtensa) | transpose, pad, conv, depthwise, add, mean, reshape, fc, softmax | compiled; not bit-exact |
| `anomaly_detect_v2` | `ad01_int8.tflite` (MLPerf AD01 AE) | mlcommons/tiny | fc ×10 | ✅ bit-exact |
| (sine regression) | `hello_world_int8.tflite` | tflite-micro @ pin | fc ×3 | ✅ bit-exact |

All 5 have per-directory SHA256 provenance READMEs under `models/zoo/`.
Goldens captured via the executed ai-edge-litert 2.1.6 interpreter
(`tools/generate_goldens/zoo/run_model.py`).

## 6. Why two substitutions compile but are NOT bit-exact (T5.2 finding)

`person_detect_vww` and `mobilenet_v2` compile through `#[model]` and execute,
but their outputs diverge from the executed-TFLite golden at kernel level.
Root-caused (per-op chained comparison of every intermediate tensor against
the interpreter at `BUILTIN_REF` — matches bit-exactly through 14 consecutive
conv/depthwise ops on person_detect, then diverges):

1. **Requantization rounding**: the hematite kernels implement TFLM
   single-rounding `MultiplyByQuantizedMultiplier` (the 64-bit
   `(x*mult + round) >> shift` form, `TFLITE_SINGLE_ROUNDING` path). The host
   ai-edge-litert reference kernels use the **double-rounding** form
   (`SaturatingRoundingDoublingHighMul` + `RoundingDivideByPOT`,
   gemmlowp path). The two agree except at exact rounding boundaries, where
   they differ by ±1 (observed: depthwise op15 on person_detect −95 vs −96;
   FC 115 vs 114 on identical input).
2. **Softmax**: the TFLM reference int8 softmax saturates wide-dynamic-range
   logits to −128 (verified: TFLM semantics on person_detect's [115,−122]
   logits give [127,−128] — exactly what hematite produces). The LiteRT
   (ai-edge-litert) int8 softmax uses a different scaling and produces
   [120,−120]. Algorithmic kernel difference, not a params/emitter gap.

**Why not fixed here**: both differences live in `hematite-ref` kernels
(`MultiplyByQuantizedMultiplier` rounding + softmax), which are owned by the
kernel workstream and explicitly out of scope for T5.2 (see `local-notes/plans/hematite-nn.md`
MUST-NOT). The emitter/parser produced bit-exact parameter streams (verified
op-by-op).

**Fix path (kernel workstream)**: adopt the `TFLITE_SINGLE_ROUNDING`-consistent
gemmlowp double-rounding `MultiplyByQuantizedMultiplier` to match host TFLite,
OR build tflite-micro at the pinned SHA and capture model goldens from a real
TFLM binary (the T5.0 remediation path in `tools/generate_goldens/README.md`).
With either fix, all 6 models should assert bit-exact.

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
kernel-semantics difference (LiteRT pads with the input zero point, the
`pad_op` kernel fills raw 0) — kernel-owned, same fix path as section 6.
