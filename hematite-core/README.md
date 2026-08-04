# hematite-core — Field-to-TFLM Mapping

Each operator parameter struct mirrors the corresponding TFLM C struct(s).
Fields are listed in declaration order.  C names use the original casing;
Rust names use snake_case.

Sources verified against tflite-micro SHA
`18b9e6f2a8c5a9518e588f59c2ba16ef7ef9d551`:
* `tensorflow/compiler/mlir/lite/core/c/builtin_op_data.h` — `TfLite*Params`
* `tensorflow/lite/kernels/internal/types.h` — kernel-internal params
  (`ConvParams`, `DepthwiseParams`, `FullyConnectedParams`, `PoolParams`,
  `SoftmaxParams`, `ArithmeticParams`, `PreluParams`, `LeakyReluParams`,
  `HardSwishParams`, `LogisticParams`, `TanhParams`, `ReluParams`,
  `SliceParams`, `ConcatenationParams`, `SplitParams`, `PadParams`,
  `ReshapeParams`, `TransposeParams`, `LstmCellParams`)

## Shared types

| Hematite Rust type              | TFLM C field / type                                   |
|---------------------------------|-------------------------------------------------------|
| `Padding::Valid`                | `PaddingType::kValid` / `kTfLitePaddingValid`         |
| `Padding::Same`                 | `PaddingType::kSame` / `kTfLitePaddingSame`           |
| `FusedActivation::None`         | `FusedActivationFunctionType::kNone` / `kTfLiteActNone` |
| `FusedActivation::Relu`         | `FusedActivationFunctionType::kRelu` / `kTfLiteActRelu` |
| `FusedActivation::Relu6`        | `kTfLiteActRelu6` (missing from types.h enum; builtin_op_data.h) |
| `FusedActivation::Relu1`        | `FusedActivationFunctionType::kRelu1` / `kTfLiteActReluN1To1` |
| `QuantParam::quantize_multiplier`   | Q0.31 reciprocal of `DequantizationParams::scale` (quantize direction) |
| `QuantParam::quantize_shift`       | Right-shift for `quantize_multiplier` (quantize op: `output = mbm(input, quantize_multiplier, quantize_shift) + zero_point`) |
| `QuantParam::dequantize_multiplier`| Q0.31 value of `DequantizationParams::scale` (dequantize direction) |
| `QuantParam::dequantize_shift`     | Right-shift for `dequantize_multiplier` (dequantize op: `output = mbm(input - zero_point, dequantize_multiplier, dequantize_shift)`) |
| `QuantParam::zero_point`           | `DequantizationParams::zero_point`                                         |
| `PerChannelQuantParam::output_multiplier_per_channel` | `DepthwiseParams::output_multiplier_per_channel` |
| `PerChannelQuantParam::output_shift_per_channel`     | `DepthwiseParams::output_shift_per_channel`     |

## Tier 0 — Core compute

### Conv2DParams

| Rust field                       | C source field                                    |
|----------------------------------|---------------------------------------------------|
| `input_shape: [i32; 4]`          | Hematite addition — NHWC: `[batch=1, H, W, Cin]`. TFLM passes shapes via `RuntimeShape`, not the params struct. |
| `filter_shape: [i32; 4]`         | Hematite addition — OHWI: `[Cout, FH, FW, Cin]`. |
| `output_shape: [i32; 4]`         | Hematite addition — NHWC: `[batch=1, OH, OW, Cout]`. |
| `padding`                        | `TfLiteConvParams::padding` / `ConvParams::padding_type` |
| `stride_width`                | `TfLiteConvParams::stride_width` / `ConvParams::stride_width` |
| `stride_height`               | `TfLiteConvParams::stride_height` / `ConvParams::stride_height` |
| `dilation_width_factor`       | `TfLiteConvParams::dilation_width_factor` / `ConvParams::dilation_width_factor` |
| `dilation_height_factor`      | `TfLiteConvParams::dilation_height_factor` / `ConvParams::dilation_height_factor` |
| `input_offset`                | `ConvParams::input_offset`                        |
| `weights_offset`              | `ConvParams::weights_offset`                      |
| `output_offset`               | `ConvParams::output_offset`                       |
| `output_multiplier_per_channel` | `DepthwiseParams::output_multiplier_per_channel` (pattern; ConvParams uses per-tensor `output_multiplier`) |
| `output_shift_per_channel`    | `DepthwiseParams::output_shift_per_channel` (pattern; per-channel per plan A4) |
| `quantized_activation_min`    | `ConvParams::quantized_activation_min`            |
| `quantized_activation_max`    | `ConvParams::quantized_activation_max`            |

> **Deviation from per-tensor ConvParams:** `ConvParams::output_multiplier`
> and `output_shift` are per-tensor (the `output_depth == 1` common case),
> but the plan mandates per-channel slices for bit-exact equivalence with
> TFLM per-channel golden fixtures.  We use the
> `output_multiplier_per_channel` pattern from `DepthwiseParams`.

### DepthwiseConv2DParams

| Rust field                       | C source field                                    |
|----------------------------------|---------------------------------------------------|
| `input_shape: [i32; 4]`          | Hematite addition — NHWC |
| `filter_shape: [i32; 4]`         | Hematite addition — depthwise filter: `[1, FH, FW, Cin * depth_multiplier]` |
| `output_shape: [i32; 4]`         | Hematite addition — NHWC |
| `padding`                        | `TfLiteDepthwiseConvParams::padding` / `DepthwiseParams::padding_type` |
| `stride_width`                | `TfLiteDepthwiseConvParams::stride_width` / `DepthwiseParams::stride_width` |
| `stride_height`               | `TfLiteDepthwiseConvParams::stride_height` / `DepthwiseParams::stride_height` |
| `dilation_width_factor`       | `TfLiteDepthwiseConvParams::dilation_width_factor` / `DepthwiseParams::dilation_width_factor` |
| `dilation_height_factor`      | `TfLiteDepthwiseConvParams::dilation_height_factor` / `DepthwiseParams::dilation_height_factor` |
| `depth_multiplier`            | `TfLiteDepthwiseConvParams::depth_multiplier` / `DepthwiseParams::depth_multiplier` |
| `input_offset`                | `DepthwiseParams::input_offset`                   |
| `weights_offset`              | `DepthwiseParams::weights_offset`                 |
| `output_offset`               | `DepthwiseParams::output_offset`                  |
| `output_multiplier_per_channel` | `DepthwiseParams::output_multiplier_per_channel`  |
| `output_shift_per_channel`    | `DepthwiseParams::output_shift_per_channel`       |
| `quantized_activation_min`    | `DepthwiseParams::quantized_activation_min`       |
| `quantized_activation_max`    | `DepthwiseParams::quantized_activation_max`       |

### FullyConnectedParams

| Rust field                    | C source field                                    |
|-------------------------------|---------------------------------------------------|
| `input_dim`                   | Accumulation depth (number of input elements)     |
| `output_dim`                  | Number of output units (= per-channel multiplier count) |
| `input_offset`                | `FullyConnectedParams::input_offset`              |
| `weights_offset`              | `FullyConnectedParams::weights_offset`            |
| `output_offset`               | `FullyConnectedParams::output_offset`             |
| `output_multiplier_per_channel` | Per-channel (plan A4); `FullyConnectedParams::output_multiplier` is per-tensor |
| `output_shift_per_channel`    | Per-channel; `FullyConnectedParams::output_shift` is per-tensor |
| `quantized_activation_min`    | `FullyConnectedParams::quantized_activation_min`  |
| `quantized_activation_max`    | `FullyConnectedParams::quantized_activation_max`  |

### MatMulParams

| Rust field         | C source field                                    |
|--------------------|---------------------------------------------------|
| `m`                | Output rows (= rows of A if `!adj_x`, else cols of A) |
| `n`                | Output cols (= cols of B if `!adj_y`, else rows of B) |
| `k`                | Inner dim (= cols of A if `!adj_x` else rows of A; = rows of B if `!adj_y` else cols of B) |
| `adj_x`            | `TfLiteBatchMatMulParams::adj_x`                  |
| `adj_y`                       | `TfLiteBatchMatMulParams::adj_y`                  |
| `input_offset`                | (from `ArithmeticParams::input1_offset` pattern)  |
| `weights_offset`              | (from `FullyConnectedParams::weights_offset` pattern) |
| `output_offset`               | (from `ArithmeticParams::output_offset` pattern)  |
| `output_multiplier`           | Per-tensor (MatMul uses per-tensor quant)         |
| `output_shift`                | Per-tensor                                        |
| `quantized_activation_min`    | (from activation params pattern)                  |
| `quantized_activation_max`    | (from activation params pattern)                  |

> **Note:** `TfLiteBatchMatMulParams` at this SHA only carries `adj_x` /
> `adj_y` / `asymmetric_quantize_inputs`. The quant fields are derived
> from the kernel-internal `FullyConnectedParams` pattern.

## Tier 1 — Pooling

### PoolParams

| Rust field                    | C source field                                    |
|-------------------------------|---------------------------------------------------|
| `input_shape: [i32; 4]`       | Hematite addition — NHWC: `[batch=1, H, W, C]`    |
| `output_shape: [i32; 4]`      | Hematite addition — NHWC                          |
| `filter_width`                | `TfLitePoolParams::filter_width` / `PoolParams::filter_width` |
| `filter_height`               | `TfLitePoolParams::filter_height` / `PoolParams::filter_height` |
| `stride_width`                | `TfLitePoolParams::stride_width` / `PoolParams::stride_width` |
| `stride_height`               | `TfLitePoolParams::stride_height` / `PoolParams::stride_height` |
| `padding`                     | `TfLitePoolParams::padding` / `PoolParams::padding_type` |
| `activation`                  | `TfLitePoolParams::activation` / `PoolParams::activation` |
| `quantized_activation_min`    | `PoolParams::quantized_activation_min`            |
| `quantized_activation_max`    | `PoolParams::quantized_activation_max`            |

## Tier 1 — Softmax

### SoftmaxParams

| Rust field           | C source field or derivation                      |
|----------------------|---------------------------------------------------|
| `num_rows`           | Number of independent softmax rows (batch * spatial dims). Hematite addition — derived from input shape at codegen. |
| `row_size`           | Elements per softmax row (channel dimension).     |
| `input_multiplier`   | `SoftmaxParams::input_multiplier`                 |
| `input_left_shift`            | `SoftmaxParams::input_left_shift`                 |
| `diff_min`                    | `SoftmaxParams::diff_min`                         |
| `input_offset`                | `SoftmaxParams::zero_point` (zero_point)          |
| `output_offset`               | i8::MIN (the fixed int8 softmax zero-point)       |
| `quantized_activation_min`    | Standard clamp lower bound                        |
| `quantized_activation_max`    | Standard clamp upper bound                        |

> **Notes:** `beta` (`f64`) is omitted — int8 softmax always uses beta = 1.0
> and the field would violate the plan's "no f32 compute" Must-NOT-Have.
> `SoftmaxParams` in types.h also carries `reverse_scaling_divisor`,
> `reverse_scaling_right_shift`, `scale`, and three LUT pointers (`table`,
> `exp_lut`, `one_over_one_plus_x_lut`).  The reverse-scaling fields are
> LogSoftmax-only (deferred).  LUT pointers are omitted from this no_std
> struct — the kernel allocates scratch for LUTs when needed.

## Tier 1 — Standalone activations

### ActivationParams

| Rust field                    | C source field                                    |
|-------------------------------|---------------------------------------------------|
| `input_offset`                | `ReluParams::input_offset`                        |
| `output_offset`               | `ReluParams::output_offset`                       |
| `output_multiplier`           | `ReluParams::output_multiplier`                   |
| `output_shift`                | `ReluParams::output_shift`                        |
| `quantized_activation_min`    | `ActivationParams::quantized_activation_min`      |
| `quantized_activation_max`    | `ActivationParams::quantized_activation_max`      |
| `input_multiplier`            | `LogisticParams::input_multiplier` / `TanhParams::input_multiplier` |
| `input_left_shift`            | `LogisticParams::input_left_shift` / `TanhParams::input_left_shift` |
| `input_range_radius`          | `LogisticParams::input_range_radius` / `TanhParams::input_range_radius` |
| `output_multiplier_alpha`     | `LeakyReluParams::output_multiplier_alpha`        |
| `output_shift_alpha`          | `LeakyReluParams::output_shift_alpha`             |
| `output_multiplier_identity`  | `LeakyReluParams::output_multiplier_identity`     |
| `output_shift_identity`       | `LeakyReluParams::output_shift_identity`          |
| `alpha_offset`                | `PreluParams::alpha_offset`                       |
| `alpha_data`                  | PReLU per-channel alpha (int8, Q7)                |
| `output_multiplier_1`         | `PreluParams::output_multiplier_1`                |
| `output_shift_1`              | `PreluParams::output_shift_1`                     |
| `output_multiplier_2`         | `PreluParams::output_multiplier_2`                |
| `output_shift_2`              | `PreluParams::output_shift_2`                     |
| `reluish_multiplier_fixedpoint_int16` | `HardSwishParams::reluish_multiplier_fixedpoint_int16` |
| `reluish_multiplier_exponent` | `HardSwishParams::reluish_multiplier_exponent`    |
| `output_multiplier_fixedpoint_int16` | `HardSwishParams::output_multiplier_fixedpoint_int16` |
| `output_multiplier_exponent`  | `HardSwishParams::output_multiplier_exponent`     |

## Tier 1 — Elementwise

### ElementwiseParams

| Rust field               | C source field                                  |
|--------------------------|-------------------------------------------------|
| `num_elements`           | Flat element count (broadcast NOT supported — all slices have identical length). Hematite addition. |
| `input1_offset`          | `ArithmeticParams::input1_offset`              |
| `input2_offset`          | `ArithmeticParams::input2_offset`              |
| `output_offset`          | `ArithmeticParams::output_offset`              |
| `output_multiplier`      | `ArithmeticParams::output_multiplier`          |
| `output_shift`           | `ArithmeticParams::output_shift`               |
| `left_shift`             | `ArithmeticParams::left_shift`                 |
| `input1_multiplier`      | `ArithmeticParams::input1_multiplier`          |
| `input1_shift`           | `ArithmeticParams::input1_shift`               |
| `input2_multiplier`      | `ArithmeticParams::input2_multiplier`          |
| `input2_shift`           | `ArithmeticParams::input2_shift`               |
| `quantized_activation_min` | `ArithmeticParams::quantized_activation_min`   |
| `quantized_activation_max` | `ArithmeticParams::quantized_activation_max`   |

## Tier 1 — Quantize / Dequantize

### QuantParam

| Rust field               | Direction                                          |
|--------------------------|----------------------------------------------------|
| `quantize_multiplier`    | Q0.31 for `1/scale` — quantize op uses this (see formula below) |
| `quantize_shift`         | Right-shift for `quantize_multiplier`              |
| `dequantize_multiplier`  | Q0.31 for `scale` — dequantize op uses this       |
| `dequantize_shift`       | Right-shift for `dequantize_multiplier`            |
| `zero_point`             | `DequantizationParams::zero_point`                 |

> Uses `(multiplier, shift)` pairs (not bare Q0.31) so that the quantize
> direction — which encodes the reciprocal scale, typically > 1.0 — is
> representable.
>
> * Quantize:   `output[i] = multiply_by_quantized_multiplier(input[i],
>                quantize_multiplier, quantize_shift) + zero_point`
> * Dequantize: `output[i] = multiply_by_quantized_multiplier(
>                input[i] - zero_point, dequantize_multiplier,
>                dequantize_shift)`

## Tier 2 — Data movement

### ReshapeParams

| Rust field     | C source field                     |
|----------------|------------------------------------|
| `shape`        | `ReshapeParams::shape[4]`          |
| `shape_count`  | `ReshapeParams::shape_count`       |

### TransposeParams

| Rust field      | C source field                                    |
|-----------------|---------------------------------------------------|
| `input_shape: [i32; 4]` | Hematite addition — NHWC; needed for stride computation |
| `perm`          | `TransposeParams::perm[kTransposeMaxDimensions]` (8) |
| `perm_count`   | `TransposeParams::perm_count`                |

### ConcatParams

| Rust field        | C source field                                    |
|-------------------|---------------------------------------------------|
| `axis`            | `TfLiteConcatenationParams::axis` / `ConcatenationParams::axis` |
| `activation`      | `TfLiteConcatenationParams::activation`           |
| `input_shape_a: [i32; 4]` | Hematite addition — NHWC; first input shape    |
| `input_shape_b: [i32; 4]` | Hematite addition — NHWC; second input shape   |
| `output_shape: [i32; 4]`  | Hematite addition — NHWC; output shape         |

### SplitParams

| Rust field        | C source field                                    |
|-------------------|---------------------------------------------------|
| `num_splits`      | `TfLiteSplitParams::num_splits` / `SplitParams::num_split` |
| `axis`            | From `SplitParams::axis` (baked in at codegen)    |
| `input_shape: [i32; 4]`  | Hematite addition — NHWC                     |
| `output_shape_a: [i32; 4]` | Hematite addition — NHWC; first output slice |
| `output_shape_b: [i32; 4]` | Hematite addition — NHWC; second output slice |

> **Note:** `TfLiteSplitParams` only carries `num_splits` (axis comes from
> a runtime input). `TfLiteSplitVParams` is identical. We include `axis`
> directly for convenience — the proc-macro extracts it from the model and
> bakes it into the param struct.

### PadParams

| Rust field           | C source field                            |
|----------------------|-------------------------------------------|
| `input_shape: [i32; 4]`  | Hematite addition — NHWC               |
| `output_shape: [i32; 4]` | Hematite addition — NHWC               |
| `left_padding`       | `PadParams::left_padding[5]` (truncated to `[i32; 4]` per static-shape constraint) |
| `left_padding_count` | `PadParams::left_padding_count`           |
| `right_padding`      | `PadParams::right_padding[5]` (truncated to `[i32; 4]`)             |
| `right_padding_count`| `PadParams::right_padding_count`          |

> **Note:** `TfLitePadParams` / `TfLitePadV2Params` are empty
> (`EmptyStructPlaceholder`).  Pad sizes come from the second runtime
> input tensor in TFLM.  We store them directly in the params struct
> because all shapes are static at codegen time.

### SliceParams

| Rust field      | C source field                |
|-----------------|-------------------------------|
| `input_shape: [i32; 4]` | Hematite addition — NHWC |
| `begin`         | `SliceParams::begin[4]`       |
| `size`          | `SliceParams::size[4]`        |

### ResizeNearestParams

| Rust field           | C source field                                      |
|----------------------|-----------------------------------------------------|
| `input_shape: [i32; 4]`  | Hematite addition — NHWC                        |
| `output_shape: [i32; 4]` | Hematite addition — NHWC                        |
| `align_corners`      | `TfLiteResizeNearestNeighborParams::align_corners` / `ResizeNearestNeighborParams::align_corners` |
| `half_pixel_centers`| `TfLiteResizeNearestNeighborParams::half_pixel_centers` / `ResizeNearestNeighborParams::half_pixel_centers` |

## Tier 3 — Recurrent

### LstmParams

| Rust field           | C source field                                              |
|----------------------|-------------------------------------------------------------|
| `activation`         | `TfLiteUnidirectionalSequenceLSTMParams::activation`        |
| `time_major`         | `TfLiteUnidirectionalSequenceLSTMParams::time_major`        |
| `num_units`          | Hematite addition — hidden-state / cell-size dimension      |
| `input_dim`          | Hematite addition — input feature dimension per timestep    |
| `weights_zero_point` | `LstmCellParams::weights_zero_point`                        |
| `accum_multiplier`   | `LstmCellParams::accum_multiplier`                          |
| `accum_shift`        | `LstmCellParams::accum_shift`                               |
| `state_integer_bits` | `LstmCellParams::state_integer_bits`                        |

> **Removed from upstream:** `cell_clip: f32` and `proj_clip: f32` — the
> plan's Must-NOT-Have list forbids f32 in device code. In the integer
> inference path, cell/projection clipping is an integer clamp already
> expressible via the activation clamping bounds on the gate activations.

### SvdfParams

| Rust field     | C source field                                     |
|----------------|----------------------------------------------------|
| `rank`         | `TfLiteSVDFParams::rank`                           |
| `activation`   | `TfLiteSVDFParams::activation`                     |
| `num_filters`  | Hematite addition — inner dimension of the feature-weight matrix |

### GruParams

| Rust field              | C source field                     |
|-------------------------|------------------------------------|
| `num_units`             | (no TFLM GRU kernel — new)         |
| `input_size`            | (no TFLM GRU kernel — new)         |
| `gate_activation`       | (sigmoid — embedded-nn convention) |
| `candidate_activation`  | (tanh — embedded-nn convention)    |

> **TFLM has no GRU kernel** at this SHA.  `GruParams` is designed for
> compatibility with `embedded-nn`'s GRU signature (plan T2.4, C4).

## Tier 4 — Reductions

### ReduceParams

| Rust field      | C source field                                   |
|-----------------|--------------------------------------------------|
| `keep_dims`     | `TfLiteReducerParams::keep_dims`                 |
| `axis`          | `MeanParams::axis[4]`                            |
| `axis_count`    | `MeanParams::axis_count`                         |
| `input_shape: [i32; 4]` | Hematite addition — NHWC; required for stride/flat-index computation |
| `output_type`   | `TfLiteArgMaxParams::output_type` / `TfLiteArgMinParams::output_type` |

## Golden-fixture const mapping

Every const emitted by `tools/generate_goldens` maps to a field in the
corresponding params struct:

| Fixture const               | Params struct + field                             |
|-----------------------------|---------------------------------------------------|
| `INPUT_SHAPE`              | `input_shape: [i32; 4]` on Conv2DParams / DepthwiseConv2DParams / PoolParams / PadParams / SliceParams / ResizeNearestParams / TransposeParams / ConcatParams / SplitParams / ReduceParams |
| `FILTER_SHAPE`             | Conv2DParams::filter_shape / DepthwiseConv2DParams::filter_shape |
| `OUTPUT_SHAPE`             | `output_shape: [i32; 4]` on most compute structs    |
| `STRIDE_WIDTH` / `STRIDE_HEIGHT` | Conv2DParams / DepthwiseConv2DParams / PoolParams::stride_width / stride_height |
| `PAD_WIDTH` / `PAD_HEIGHT` | `padding` enum + codegen computes pad values        |
| `DILATION_W` / `DILATION_H` | Conv2DParams / DepthwiseConv2DParams::dilation_width_factor / dilation_height_factor |
| `INPUT_OFFSET`             | `input_offset` on most structs                      |
| `OUTPUT_OFFSET`            | `output_offset` on most structs                     |
| `OUTPUT_ACTIVATION_MIN` / `OUTPUT_ACTIVATION_MAX` | `quantized_activation_min` / `max` on most structs |
| `OUTPUT_MULTIPLIER` (per-channel) | Conv2DParams / DepthwiseConv2DParams / FullyConnectedParams ::output_multiplier_per_channel |
| `OUTPUT_SHIFT` (per-channel) | Conv2DParams / DepthwiseConv2DParams / FullyConnectedParams ::output_shift_per_channel |
| `OUTPUT_MULTIPLIER` (per-tensor)  | ElementwiseParams::output_multiplier; MatMulParams::output_multiplier |
| `OUTPUT_SHIFT` (per-tensor)       | ElementwiseParams::output_shift; MatMulParams::output_shift |
| `FILTER_WIDTH` / `FILTER_HEIGHT` | PoolParams::filter_width / filter_height           |
| `INPUT_SCALE`              | SoftmaxParams::input_multiplier / input_left_shift (derived via `quantize_multiplier`) |
| `INPUT_MULTIPLIER` / `INPUT_SHIFT` | SoftmaxParams::input_multiplier / input_left_shift |
| `LEFT_SHIFT`               | SoftmaxParams::input_left_shift / ElementwiseParams::left_shift |
| `DIFF_MIN`                 | SoftmaxParams::diff_min                             |
| `INPUT2_OFFSET`            | ElementwiseParams::input2_offset                    |
| `INPUT1_MULTIPLIER` / `INPUT1_SHIFT` | ElementwiseParams::input1_multiplier / input1_shift |
| `INPUT2_MULTIPLIER` / `INPUT2_SHIFT` | ElementwiseParams::input2_multiplier / input2_shift |
| `OUTPUT_MULTIPLIER_IDENTITY` / `OUTPUT_SHIFT_IDENTITY` | ActivationParams::output_multiplier_identity / output_shift_identity |
| `OUTPUT_MULTIPLIER_ALPHA` / `OUTPUT_SHIFT_ALPHA` | ActivationParams::output_multiplier_alpha / output_shift_alpha |
| `OUTPUT_MULTIPLIER_1` / `OUTPUT_SHIFT_1` | ActivationParams::output_multiplier_1 / output_shift_1 |
| `OUTPUT_MULTIPLIER_2` / `OUTPUT_SHIFT_2` | ActivationParams::output_multiplier_2 / output_shift_2 |
| `ALPHA_OFFSET`             | ActivationParams::alpha_offset                      |
| `ALPHA_DATA`               | ActivationParams::alpha_data                        |
| `SCALE_Q31` (quantize fixture)  | QuantParam::quantize_multiplier + quantize_shift (multiplier+shift pair, not raw Q0.31) |
| `SCALE_Q31` (dequantize fixture) | QuantParam::dequantize_multiplier + dequantize_shift |
| `ZERO_POINT`               | QuantParam::zero_point                              |
| `AXIS`                     | ConcatParams::axis / SplitParams::axis              |
| `DEPTH_MULTIPLIER`         | DepthwiseConv2DParams::depth_multiplier             |
| `ACCUM_DEPTH`              | FullyConnectedParams::input_dim                     |
| `OUTPUT_DEPTH`             | FullyConnectedParams::output_dim                    |
| `PAD_TOP/BOTTOM/LEFT/RIGHT` | PadParams::left_padding / right_padding             |
| `PAD_VALUE`                | Not a params field (pad-value tensor is a separate input) |
| `BEGIN_0..3` / `SIZE_0..3` | SliceParams::begin / size                           |
| `PERM_0..3`                | TransposeParams::perm                               |
| `ALIGN_CORNERS` / `HALF_PIXEL_CENTERS` | ResizeNearestParams::align_corners / half_pixel_centers |
| `INPUT_ZERO_POINT`         | ActivationParams::input_offset (relu fixture convention) |
| `OUTPUT_ZERO_POINT`        | ActivationParams::output_offset (relu fixture convention) |

> Fixture consts without a params home: `PAD_VALUE` (runtime tensor, not
> a params field — pad value comes from the TFLite model as a separate
> input tensor), and `WEIGHTS_DATA` / `BIAS_DATA` / `INPUT_DATA` /
> `EXPECTED_OUTPUT` (tensor data arrays, passed as separate slices to
> trait methods, not params struct fields).
