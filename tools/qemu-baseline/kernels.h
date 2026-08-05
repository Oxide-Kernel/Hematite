/* SPDX-License-Identifier: Apache-2.0
 *
 * Plain-C int8 reference kernels mirroring the hematite scalar reference
 * kernels (`hematite-ref`) / TFLM int8 scalar math.  These are the C-side
 * analog of the `hematite-ref` column-1 baseline of the benchmark suite:
 * i32 accumulate, CMSIS/TFLM single-rounding requantize
 * (`multiply_by_quantized_multiplier`), output zero-point offset, clamp to
 * the fused activation range, saturating cast to int8.
 *
 * The params structs mirror the field set of
 * `hematite-core::op_params::{Conv2DParams, DepthwiseConv2DParams,
 * FullyConnectedParams}` that the scalar kernels read.  `weights_offset` is
 * intentionally absent: the reference kernels do not apply it (documented
 * gotcha in hematite-ref/conv.rs).
 */
#ifndef KERNELS_H
#define KERNELS_H

#include <stdint.h>

typedef struct {
    int32_t input_shape[4];  /* NHWC [1, H, W, Cin]  */
    int32_t filter_shape[4]; /* OHWI [Cout, FH, FW, Cin] */
    int32_t output_shape[4]; /* NHWC [1, OH, OW, Cout] */
    int32_t stride_height, stride_width;
    int32_t dilation_height_factor, dilation_width_factor;
    int32_t input_offset;
    int32_t output_offset;
    const int32_t *output_multiplier_per_channel;
    const int32_t *output_shift_per_channel;
    int32_t quantized_activation_min, quantized_activation_max;
} ConvParams;

typedef struct {
    int32_t input_shape[4];  /* NHWC [1, H, W, Cin] */
    int32_t filter_shape[4]; /* [1, FH, FW, Cin * depth_multiplier] */
    int32_t output_shape[4]; /* NHWC [1, OH, OW, Cout] */
    int32_t depth_multiplier;
    int32_t stride_height, stride_width;
    int32_t dilation_height_factor, dilation_width_factor;
    int32_t input_offset;
    int32_t output_offset;
    const int32_t *output_multiplier_per_channel;
    const int32_t *output_shift_per_channel;
    int32_t quantized_activation_min, quantized_activation_max;
} DepthwiseParams;

typedef struct {
    int32_t input_dim;
    int32_t output_dim;
    int32_t input_offset;
    int32_t output_offset;
    const int32_t *output_multiplier_per_channel;
    const int32_t *output_shift_per_channel;
    int32_t quantized_activation_min, quantized_activation_max;
} FcParams;

/* 2D convolution (general; used for both the 8x8 3x3 conv and the 1x1 conv
 * specs — the hematite ref has a single conv2d for both). */
void conv2d(const int8_t *input, const int8_t *weights, const int32_t *bias,
            const ConvParams *p, int8_t *output);

/* Depthwise 2D convolution, channel-contiguous weights. */
void depthwise_conv2d(const int8_t *input, const int8_t *weights,
                      const int32_t *bias, const DepthwiseParams *p,
                      int8_t *output);

/* Fully-connected (flat dot product per output unit). */
void fully_connected(const int8_t *input, const int8_t *weights,
                     const int32_t *bias, const FcParams *p, int8_t *output);

#endif /* KERNELS_H */
