# keyword_spotting — model artifact

**Source**: tflite-micro at the pinned golden SHA
**Repo commit**: `18b9e6f2a8c5a9518e588f59c2ba16ef7ef9d551`
**Path**: `tensorflow/lite/micro/examples/micro_speech/models/micro_speech_quantized.tflite`
**Description**: micro_speech keyword-spotting model — int8 [1,1960] mel-spectrogram
  input, 4 classes (silence/unknown/yes/no). Matches the plan's
  `keyword_spotting_v1` family (reshape/depthwise_conv/fc/softmax).

> **SUBSTITUTION**: the edge-ml-model-zoo `kws_micro_v2` binary is not publicly
> available (repo is metadata-only — see `DEFERRED_MODELS.md`). This is the
> canonical tflite-micro KWS int8 model at the exact golden pin.

## Artifacts (SHA256)

- `kws_micro_speech_int8.tflite` — 18800 B — `09e5e2a9dfb2d8ed78802bf18ce297bff54281a66ca18e0c23d69ca14f822a83`

## Source URL

- https://raw.githubusercontent.com/tensorflow/tflite-micro/18b9e6f2a8c5a9518e588f59c2ba16ef7ef9d551/tensorflow/lite/micro/examples/micro_speech/models/micro_speech_quantized.tflite
