# mobilenetv2_cls — model artifact

**Source**: tflite-micro at the pinned golden SHA (xtensa pytorch_to_tflite example)
**Repo commit**: `18b9e6f2a8c5a9518e588f59c2ba16ef7ef9d551`
**Path**: `third_party/xtensa/examples/pytorch_to_tflite/mobilenet_v2_quantized_1x3x224x224.tflite`
**Description**: MobileNetV2 image classifier, 224×224, full-integer int8, 1000 classes.
  NCHW input [1,3,224,224] (transposed to NHWC inside the graph). Matches the plan's
  `imagenet_cls` / `mobilenetv2_cls` family. Widest op set in the zoo:
  transpose/pad/conv/depthwise/add/mean/reshape/fc/softmax.

> **SUBSTITUTION**: the esp-dl `imagenet_cls_mobilenetv2_s8_v1` artifact is
> `.espdl`-only (see `DEFERRED_MODELS.md`). This model is the same
> MobileNetV2-224 architecture converted for Xtensa; it uses legacy opcode
> encoding (code in `deprecated_builtin_code`, field 3 absent) that the
> hematite-codegen parser now resolves (default-to-ADD for omitted fields).

## Artifacts (SHA256)

- `mobilenet_v2_1.0_224_int8.tflite` — 3980776 B — `2778683128d56fe4a3da6b8cdbbb00ec12663553f30a9b4fe927747ba637bb2f`

## Source URL

- https://raw.githubusercontent.com/tensorflow/tflite-micro/18b9e6f2a8c5a9518e588f59c2ba16ef7ef9d551/third_party/xtensa/examples/pytorch_to_tflite/mobilenet_v2_quantized_1x3x224x224.tflite
