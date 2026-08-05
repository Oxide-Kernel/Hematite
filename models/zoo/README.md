# models/zoo — ESP-DL + edge-ml model zoo (T5.0/T5.2)

Golden-corpus model zoo for hematite-nn Phase 5 (T5.2 model-level inference).

## Landed artifacts

### ESP-DL (espressif/esp-dl) — 15 `.espdl` files in 9 directories

| Model | Sub-models | Files |
|---|---|---|
| `cat_detect` | 2 | espdet_pico_224_224_cat, espdet_pico_416_416_cat |
| `dog_detect` | 2 | espdet_pico_224_224_dog, espdet_pico_416_416_dog |
| `hand_detect` | 1 | espdet_pico_224_224_hand |
| `human_face_detect` | 4 | espdet_pico_224_224_face, espdet_pico_416_416_face, human_face_detect_mnp_s8_v1, human_face_detect_msr_s8_v1 |
| `human_face_recognition` | 2 | human_face_feat_mbf_s8_v1, human_face_feat_mfn_s8_v1 |
| `hand_gesture_recognition` | 1 | mobilenetv2_0_5_128_128_gesture |
| `pedestrian_detect` | 1 | pedestrian_detect_pico_s8_v1 |
| `person_reid` | 1 | person_reid_feat_osn_s8_v1 |
| `imagenet_cls` | 1 | imagenet_cls_mobilenetv2_s8_v1 |

Source: `github.com/espressif/esp-dl`, commit `12c0616de145b704e1149c474b9a1e852e631d67`
(`master`), path `models/<name>/models/s3/`. Each subdirectory has its own
`README.md` with per-file SHA256 and raw source URLs.

### T5.2 substitution models — REAL public int8 `.tflite` (5)

The plan's named 18-model list is not obtainable as `.tflite` (esp-dl = `.espdl`
only; edge-ml-model-zoo has no binaries). These public int8 models cover the
same op families; see `DEFERRED_MODELS.md` for the per-family substitution table.

| Zoo family (plan name) | Model dir | `.tflite` | Ops exercised | Bit-exact |
|---|---|---|---|---|
| `person_detect_v2` | `person_detect_vww/` | person_detect_int8.tflite (VWW, 96²) | conv, depthwise, avgpool, reshape, fc, softmax | ⚠️ compiled, not bit-exact |
| `keyword_spotting_v1` | `keyword_spotting/` | kws_micro_speech_int8.tflite | reshape, depthwise, fc, softmax | ✅ bit-exact |
| `imagenet_cls` / `mobilenetv2_cls` | `mobilenetv2_cls/` | mobilenet_v2_1.0_224_int8.tflite | transpose, pad, conv, depthwise, add, mean, reshape, fc, softmax | ⚠️ compiled, not bit-exact |
| `anomaly_detect_v2` | `anomaly_detect/` | anomaly_detect_int8.tflite (MLPerf AD01 AE) | fc ×10 | ✅ bit-exact |
| (sine regression) | `sine_regression/` | hello_world_int8.tflite | fc ×3 | ✅ bit-exact |

Each dir has its own `README.md` with per-file SHA256, source URL, and the
substitution rationale. **The model goldens in
`hematite-tests/goldens/models/*.rs` are captured from a real executed
ai-edge-litert 2.1.6 interpreter** (`tools/generate_goldens/zoo/run_model.py`),
not from hand computation.

## ⚠️ Format finding (esp-dl zoo — unchanged)

**The esp-dl model zoo ships NO `.tflite` files.** Every model is in the
proprietary `.espdl` format (EDL2 magic; a custom FlatBuffers schema,
`esp-dl/fbs_loader/espdl.fbs`, IR version 2023-12-22). This was verified
exhaustively against the esp-dl repository: zero `.tflite` files in the tree,
across all branches, all tags (v0.x–v3.x), or any release asset. T5.2 therefore
substituted the 5 public int8 `.tflite` models above (full accounting in
`DEFERRED_MODELS.md`).

## ⚠️ Bit-exactness status of model-level tests (T5.2)

`cargo test -p hematite-tests -- models` asserts **6 model-level tests**:
4 models **bit-exact** vs their executed-TFLite golden (sine smoke,
hello_world, kws_micro_speech, anomaly_detect) + 2 models that **compile and
execute** through `#[model]` but are **NOT asserted bit-exact**
(person_detect_vww, mobilenet_v2). The 2 non-bit-exact models diverge at
rounding boundaries where the hematite kernels (TFLM single-rounding
`MultiplyByQuantizedMultiplier`) differ ±1 from the host ai-edge-litert
reference kernels (double-rounding), and at their softmax where the LiteRT int8
softmax algorithm differs from the TFLM reference on wide-dynamic-range logits.
These are kernel-semantics differences in `hematite-ref` (owned by the kernel
workstream), not emitter/parser gaps — the emitter compiles all 6 models and
matches the interpreter bit-exactly through 14 consecutive conv/depthwise ops
on person_detect. Fix path documented in `DEFERRED_MODELS.md`.

## Regenerating this directory

```sh
# From workspace root — downloads all 15 esp-dl artifacts + recomputes SHA256.
bash tools/generate_goldens/zoo/fetch_espdl.sh

# Regenerate the executed-TFLite model goldens (all runnable .tflite under models/):
cargo run -p generate-goldens
```
