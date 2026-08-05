# pedestrian_detect — model artifact

**Source**: ESP-DL model zoo (espressif/esp-dl)
**Repo commit**: `12c0616de145b704e1149c474b9a1e852e631d67` (branch `master`)
**Description**: Pedestrian detection, PicoDet s8 v1.

> **FORMAT WARNING**: ESP-DL v3.x ships these models in the proprietary
> `.espdl` (EDL2/FlatBuffers) format — NOT `.tflite`. Verified: the
> esp-dl repo contains zero `.tflite` files in its tree, history, or
> releases (checked across all branches/tags at commit `12c0616de145b704e1149c474b9a1e852e631d67`).
> These artifacts therefore CANNOT be executed by a TFLite/TFLM interpreter
> for golden capture. See `DEFERRED_MODELS.md` at the zoo root.

## Artifacts (SHA256)

- `pedestrian_detect_pico_s8_v1.espdl` — 435328 B — `3eb5827e12322728fe99ea9c86d71ea38aa5a5b70064abf7d75bc90140b74015`

## Source URLs

- https://raw.githubusercontent.com/espressif/esp-dl/12c0616de145b704e1149c474b9a1e852e631d67/models/pedestrian_detect/models/s3/pedestrian_detect_pico_s8_v1.espdl
