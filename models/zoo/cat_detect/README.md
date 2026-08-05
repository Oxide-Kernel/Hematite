# cat_detect — model artifact

**Source**: ESP-DL model zoo (espressif/esp-dl)
**Repo commit**: `12c0616de145b704e1149c474b9a1e852e631d67` (branch `master`)
**Description**: ESPDet-Pico cat detection (COCO2017 cat class). Two input resolutions: 224×224 and 416×416.

> **FORMAT WARNING**: ESP-DL v3.x ships these models in the proprietary
> `.espdl` (EDL2/FlatBuffers) format — NOT `.tflite`. Verified: the
> esp-dl repo contains zero `.tflite` files in its tree, history, or
> releases (checked across all branches/tags at commit `12c0616de145b704e1149c474b9a1e852e631d67`).
> These artifacts therefore CANNOT be executed by a TFLite/TFLM interpreter
> for golden capture. See `DEFERRED_MODELS.md` at the zoo root.

## Artifacts (SHA256)

- `espdet_pico_224_224_cat.espdl` — 498576 B — `604501220b26b8a59f44a835b7d37c31444444a8f5661c5ad9469c61fbcd71e1`
- `espdet_pico_416_416_cat.espdl` — 498560 B — `fe40569dfeef4b18c5a6b27066961e699791d1e7819cfed977aae49f0e82019f`

## Source URLs

- https://raw.githubusercontent.com/espressif/esp-dl/12c0616de145b704e1149c474b9a1e852e631d67/models/cat_detect/models/s3/espdet_pico_224_224_cat.espdl
- https://raw.githubusercontent.com/espressif/esp-dl/12c0616de145b704e1149c474b9a1e852e631d67/models/cat_detect/models/s3/espdet_pico_416_416_cat.espdl
