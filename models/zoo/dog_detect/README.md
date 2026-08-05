# dog_detect — model artifact

**Source**: ESP-DL model zoo (espressif/esp-dl)
**Repo commit**: `12c0616de145b704e1149c474b9a1e852e631d67` (branch `master`)
**Description**: ESPDet-Pico dog detection (COCO2017 dog class). Two input resolutions: 224×224 and 416×416.

> **FORMAT WARNING**: ESP-DL v3.x ships these models in the proprietary
> `.espdl` (EDL2/FlatBuffers) format — NOT `.tflite`. Verified: the
> esp-dl repo contains zero `.tflite` files in its tree, history, or
> releases (checked across all branches/tags at commit `12c0616de145b704e1149c474b9a1e852e631d67`).
> These artifacts therefore CANNOT be executed by a TFLite/TFLM interpreter
> for golden capture. See `DEFERRED_MODELS.md` at the zoo root.

## Artifacts (SHA256)

- `espdet_pico_224_224_dog.espdl` — 500064 B — `c51b02eb0263dc88172362a774f8d85b835bfc7d3709bb623937a23a7f978933`
- `espdet_pico_416_416_dog.espdl` — 500000 B — `0bda95a4ae55d07682191ae389bc72f0f5a4e5561b118587712514d65a52722c`

## Source URLs

- https://raw.githubusercontent.com/espressif/esp-dl/12c0616de145b704e1149c474b9a1e852e631d67/models/dog_detect/models/s3/espdet_pico_224_224_dog.espdl
- https://raw.githubusercontent.com/espressif/esp-dl/12c0616de145b704e1149c474b9a1e852e631d67/models/dog_detect/models/s3/espdet_pico_416_416_dog.espdl
