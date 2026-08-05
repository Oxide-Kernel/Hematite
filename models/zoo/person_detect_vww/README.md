# person_detect_vww — model artifact

**Source**: MLCommons Tiny benchmark (mlcommons/tiny, `master` @ `4addd0fa08d216e20637637874e084895f289da4`)
**Path**: `benchmark/training/visual_wake_words/trained_models/vww_96_int8.tflite`
**Description**: Visual Wake Words person detector — MobileNetV1-style 96×96×3 int8
  classifier with 2 outputs (person / no-person). Matches the plan's `person_detect_v2`
  family (conv/depthwise/pool/fc/softmax).

> **SUBSTITUTION**: The canonical tflite-micro `person_detect.tflite` (and the
> TF-hosted `person_detect_int8.tflite`) use the legacy `quantized_dimension=3`
> weight convention that ai-edge-litert 2.1.6 refuses to load (invalid
> quantization parameters) — it cannot be executed for golden capture. This
> MLCommons VWW model is the same MobileNetV1 VWW architecture with modern
> `quantized_dimension=0` encoding.

## Artifacts (SHA256)

- `person_detect_int8.tflite` — 333288 B — `597a384c8c2c8a1276f04702f25013b7838f2f814f1ca7c174d295b73e3d6b7b`

## Source URL

- https://raw.githubusercontent.com/mlcommons/tiny/master/benchmark/training/visual_wake_words/trained_models/vww_96_int8.tflite
