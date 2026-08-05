# hand_gesture_recognition — model artifact

**Source**: ESP-DL model zoo (espressif/esp-dl)
**Repo commit**: `12c0616de145b704e1149c474b9a1e852e631d67` (branch `master`)
**Description**: Hand gesture classification, MobileNetV2 0.5 @128×128.

> **FORMAT WARNING**: ESP-DL v3.x ships these models in the proprietary
> `.espdl` (EDL2/FlatBuffers) format — NOT `.tflite`. Verified: the
> esp-dl repo contains zero `.tflite` files in its tree, history, or
> releases (checked across all branches/tags at commit `12c0616de145b704e1149c474b9a1e852e631d67`).
> These artifacts therefore CANNOT be executed by a TFLite/TFLM interpreter
> for golden capture. See `DEFERRED_MODELS.md` at the zoo root.

## Artifacts (SHA256)

- `mobilenetv2_0_5_128_128_gesture.espdl` — 787296 B — `67c3721f57d45bac758ec9c6206fdaff3c7fbf0a6ac103d8ed5ffb4daa0d4a93`

## Source URLs

- https://raw.githubusercontent.com/espressif/esp-dl/12c0616de145b704e1149c474b9a1e852e631d67/models/hand_gesture_recognition/models/s3/mobilenetv2_0_5_128_128_gesture.espdl
