# anomaly_detect — model artifact

**Source**: MLCommons Tiny benchmark (mlcommons/tiny, `master` @ `4addd0fa08d216e20637637874e084895f289da4`)
**Path**: `benchmark/training/anomaly_detection/trained_models/ad01_int8.tflite`
**Description**: Anomaly-detection autoencoder — int8 [1,640] input, [1,640] output,
  10 fully-connected layers (encoder + decoder). Matches the plan's
  `anomaly_detect_v2` / `anomaly_ae_v1` family (the MLPerf Tiny AD01 benchmark model).

> **SUBSTITUTION**: the edge-ml-model-zoo `anomaly_ae_v1` binary is not publicly
> available (see `DEFERRED_MODELS.md`). `ad01_int8.tflite` is the MLPerf Tiny
> anomaly-detection benchmark model, which exercises the same FC-autoencoder family.

## Artifacts (SHA256)

- `anomaly_detect_int8.tflite` — 276976 B — `87cf24194ef93d1d9b11a591d805526b98008e351655d29883c825c9c106ba24`

## Source URL

- https://raw.githubusercontent.com/mlcommons/tiny/master/benchmark/training/anomaly_detection/trained_models/ad01_int8.tflite
