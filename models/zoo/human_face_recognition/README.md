# human_face_recognition — model artifact

**Source**: ESP-DL model zoo (espressif/esp-dl)
**Repo commit**: `12c0616de145b704e1149c474b9a1e852e631d67` (branch `master`)
**Description**: Face feature extraction: MBF (MobileFace) and MFN (MobileFaceNet) embeddings.

> **FORMAT WARNING**: ESP-DL v3.x ships these models in the proprietary
> `.espdl` (EDL2/FlatBuffers) format — NOT `.tflite`. Verified: the
> esp-dl repo contains zero `.tflite` files in its tree, history, or
> releases (checked across all branches/tags at commit `12c0616de145b704e1149c474b9a1e852e631d67`).
> These artifacts therefore CANNOT be executed by a TFLite/TFLM interpreter
> for golden capture. See `DEFERRED_MODELS.md` at the zoo root.

## Artifacts (SHA256)

- `human_face_feat_mbf_s8_v1.espdl` — 3522784 B — `764610814b63924c61ca6029f47c5f775ceb6acaf501bd39591e178d89415c33`
- `human_face_feat_mfn_s8_v1.espdl` — 1295168 B — `6dbab04ee99124e26ee931c8980595829d9133b043695fbf6ce6ca593c772e95`

## Source URLs

- https://raw.githubusercontent.com/espressif/esp-dl/12c0616de145b704e1149c474b9a1e852e631d67/models/human_face_recognition/models/s3/human_face_feat_mbf_s8_v1.espdl
- https://raw.githubusercontent.com/espressif/esp-dl/12c0616de145b704e1149c474b9a1e852e631d67/models/human_face_recognition/models/s3/human_face_feat_mfn_s8_v1.espdl
