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
