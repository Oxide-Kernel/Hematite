# human_face_detect — model artifact

**Source**: ESP-DL model zoo (espressif/esp-dl)
**Repo commit**: `12c0616de145b704e1149c474b9a1e852e631d67` (branch `master`)
**Description**: Face detection pipeline: ESPDet-Pico one-stage face detector (224/416) + MNP landmark network + MSR feature extractor.

> **FORMAT WARNING**: ESP-DL v3.x ships these models in the proprietary
> `.espdl` (EDL2/FlatBuffers) format — NOT `.tflite`. Verified: the
> esp-dl repo contains zero `.tflite` files in its tree, history, or
> releases (checked across all branches/tags at commit `12c0616de145b704e1149c474b9a1e852e631d67`).
> These artifacts therefore CANNOT be executed by a TFLite/TFLM interpreter
> for golden capture. See `DEFERRED_MODELS.md` at the zoo root.

## Artifacts (SHA256)

- `espdet_pico_224_224_face.espdl` — 480384 B — `c9a991e00aeca4009eb2771e3fccf7a7ff8a47781a90dc66701692fcdf1f1b5e`
- `espdet_pico_416_416_face.espdl` — 499312 B — `1e52445d1cb6a8d8378315e61aa3e9fdf2e3977754016a71be3cce52eed05507`
- `human_face_detect_mnp_s8_v1.espdl` — 129968 B — `e981fe2107281f25e8c54f5f091c1037c8343a9e23f4c51fcc22bd37728c0157`
- `human_face_detect_msr_s8_v1.espdl` — 61168 B — `ab705d4b831eeae9ff21fabfe4471affae1006a4a2c273de022bac26db4df973`

## Source URLs

- https://raw.githubusercontent.com/espressif/esp-dl/12c0616de145b704e1149c474b9a1e852e631d67/models/human_face_detect/models/s3/espdet_pico_224_224_face.espdl
- https://raw.githubusercontent.com/espressif/esp-dl/12c0616de145b704e1149c474b9a1e852e631d67/models/human_face_detect/models/s3/espdet_pico_416_416_face.espdl
- https://raw.githubusercontent.com/espressif/esp-dl/12c0616de145b704e1149c474b9a1e852e631d67/models/human_face_detect/models/s3/human_face_detect_mnp_s8_v1.espdl
- https://raw.githubusercontent.com/espressif/esp-dl/12c0616de145b704e1149c474b9a1e852e631d67/models/human_face_detect/models/s3/human_face_detect_msr_s8_v1.espdl
