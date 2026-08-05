# models/zoo — ESP-DL + edge-ml model zoo (T5.0)

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

### edge-ml-model-zoo — 0 files (see below)

## ⚠️ Critical format finding (read before using this zoo)

**The esp-dl model zoo ships NO `.tflite` files.** Every model is in the
proprietary `.espdl` format (EDL2 magic; a custom FlatBuffers schema,
`esp-dl/fbs_loader/espdl.fbs`, IR version 2023-12-22). This was verified
exhaustively against the esp-dl repository: zero `.tflite` files in the tree,
across all branches, all tags (v0.x–v3.x), or any release asset.

Consequences:

1. **Per-model golden capture via TFLite/TFLM Interpreter is impossible for
   these artifacts.** The `.espdl` format cannot be loaded by any TFLite
   runtime. `tools/generate_goldens` zoo golden path therefore has no
   esp-dl models to execute.
2. The plan's assumption (".espdl archives containing multiple tflite
   sub-models") does not match the current esp-dl v3.x format. The `.espdl`
   file IS the model; there is no embedded TFLite.
3. T5.2 `#[model("models/x.tflite")]` compilation will require either a
   `.espdl` → TFLite conversion path (out of scope: ESP-PPQ is a proprietary
   Python quantizer requiring the original ONNX/PyTorch source, which esp-dl
   does not distribute for these models) or an esp-dl runtime loader on the
   device.

The artifacts are preserved here anyway: they are the canonical, versioned
ESP-DL S3 model binaries (source of truth for on-device deployment), and T5.2
can use them via ESP-DL's own loader if a device-side golden path is
established. See `DEFERRED_MODELS.md` for the full accounting.

## Regenerating this directory

```sh
# From workspace root — downloads all 15 esp-dl artifacts + recomputes SHA256.
bash tools/generate_goldens/zoo/fetch_espdl.sh
```
