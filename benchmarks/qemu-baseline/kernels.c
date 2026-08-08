/* SPDX-License-Identifier: Apache-2.0
 *
 * Plain-C int8 reference kernels for the QEMU C baseline.  These mirror
 * `hematite-ref/src/{conv,depthwise_conv,fully_connected}.rs` loop-for-loop
 * and use the same TFLM scalar arithmetic as
 * `hematite-int8::multiply_by_quantized_multiplier` (CMSIS single-rounding):
 *
 *   total_shift = 31 - shift;
 *   round       = 1 << (total_shift - 1);
 *   result      = ((int64)value * (int64)multiplier + round) >> total_shift;
 *   saturate to i32.
 *
 * Combined with `saturating_cast` (clamp to [-128, 127]) this is the
 * "i32 accumulate, single-rounding requantize" path the task requires.
 */
#include "kernels.h"

#include <limits.h>

/* ── TFLM scalar math primitives (mirror hematite-int8) ─────────────────── */

static inline int32_t multiply_by_quantized_multiplier(int32_t value,
                                                       int32_t multiplier,
                                                       int32_t shift)
{
    int64_t total_shift = 31 - (int64_t)shift;
    int64_t round = (int64_t)1 << (total_shift - 1);
    int64_t result = (int64_t)value * (int64_t)multiplier + round;
    result >>= total_shift;
    if (result > (int64_t)INT32_MAX) {
        return INT32_MAX;
    }
    if (result < (int64_t)INT32_MIN) {
        return INT32_MIN;
    }
    return (int32_t)result;
}

static inline int8_t saturating_cast(int32_t v)
{
    if (v > 127) {
        return 127;
    }
    if (v < -128) {
        return -128;
    }
    return (int8_t)v;
}

static inline int32_t clamp_activation(int32_t v, const ConvParams *p)
{
    int32_t c = v;
    if (c > p->quantized_activation_max) {
        c = p->quantized_activation_max;
    } else if (c < p->quantized_activation_min) {
        c = p->quantized_activation_min;
    }
    return c;
}

/* ── conv2d (TFLM ConvEval loop order) ───────────────────────────────────── */

void conv2d(const int8_t *input, const int8_t *weights, const int32_t *bias,
            const ConvParams *p, int8_t *output)
{
    const int32_t input_h = p->input_shape[1];
    const int32_t input_w = p->input_shape[2];
    const int32_t input_c = p->input_shape[3];

    const int32_t filter_h = p->filter_shape[1];
    const int32_t filter_w = p->filter_shape[2];
    const int32_t filter_ic = p->filter_shape[3];

    const int32_t out_h = p->output_shape[1];
    const int32_t out_w = p->output_shape[2];
    const int32_t out_channels = p->output_shape[3];

    /* Pad derived from shapes (no pad field in the params structs):
     *   dilated_extent = (filter_dim - 1) * dilation + 1
     *   pad = ((out_dim - 1) * stride + dilated_extent - in_dim) / 2     */
    const int32_t dilated_filter_h = (filter_h - 1) * p->dilation_height_factor + 1;
    const int32_t dilated_filter_w = (filter_w - 1) * p->dilation_width_factor + 1;
    const int32_t pad_h = ((out_h - 1) * p->stride_height + dilated_filter_h - input_h) / 2;
    const int32_t pad_w = ((out_w - 1) * p->stride_width + dilated_filter_w - input_w) / 2;

    const int32_t input_row_stride = input_w * input_c;
    const int32_t filter_oc_stride = filter_h * filter_w * filter_ic;
    const int32_t filter_row_stride = filter_w * filter_ic;
    const int32_t filter_col_stride = filter_ic;
    const int32_t output_row_stride = out_w * out_channels;

    for (int32_t oh = 0; oh < out_h; oh++) {
        const int32_t input_base_h = oh * p->stride_height - pad_h;
        for (int32_t ow = 0; ow < out_w; ow++) {
            const int32_t input_base_w = ow * p->stride_width - pad_w;
            for (int32_t oc = 0; oc < out_channels; oc++) {
                int32_t acc = bias[oc];
                const int32_t filter_oc_base = oc * filter_oc_stride;

                for (int32_t fh = 0; fh < filter_h; fh++) {
                    const int32_t in_h = input_base_h + fh * p->dilation_height_factor;
                    const int32_t row_in_bounds = (in_h >= 0 && in_h < input_h);

                    for (int32_t fw = 0; fw < filter_w; fw++) {
                        const int32_t in_w = input_base_w + fw * p->dilation_width_factor;
                        if (row_in_bounds && in_w >= 0 && in_w < input_w) {
                            const int8_t *ip =
                                input + (in_h * input_row_stride + in_w * input_c);
                            const int8_t *wp = weights +
                                (filter_oc_base + fh * filter_row_stride +
                                 fw * filter_col_stride);
                            for (int32_t ic = 0; ic < filter_ic; ic++) {
                                const int32_t i_val = (int32_t)ip[ic];
                                const int32_t w_val = (int32_t)wp[ic];
                                acc += (i_val + p->input_offset) * w_val;
                            }
                        }
                        /* else: zero-padding — skip (contributes 0) */
                    }
                }

                const int32_t scaled = multiply_by_quantized_multiplier(
                    acc, p->output_multiplier_per_channel[oc],
                    p->output_shift_per_channel[oc]);
                const int32_t with_offset = scaled + p->output_offset;
                const int32_t clamped = clamp_activation(with_offset, p);
                output[oh * output_row_stride + ow * out_channels + oc] =
                    saturating_cast(clamped);
            }
        }
    }
}

/* ── depthwise_conv2d (TFLM DepthwiseConvPerChannel loop order) ─────────── */

void depthwise_conv2d(const int8_t *input, const int8_t *weights,
                      const int32_t *bias, const DepthwiseParams *p,
                      int8_t *output)
{
    const int32_t input_h = p->input_shape[1];
    const int32_t input_w = p->input_shape[2];
    const int32_t input_c = p->input_shape[3];

    const int32_t filter_h = p->filter_shape[1];
    const int32_t filter_w = p->filter_shape[2];

    const int32_t out_h = p->output_shape[1];
    const int32_t out_w = p->output_shape[2];
    const int32_t out_c = p->output_shape[3];

    const int32_t dm = p->depth_multiplier;

    const int32_t dilated_filter_h = (filter_h - 1) * p->dilation_height_factor + 1;
    const int32_t dilated_filter_w = (filter_w - 1) * p->dilation_width_factor + 1;
    const int32_t pad_h = ((out_h - 1) * p->stride_height + dilated_filter_h - input_h) / 2;
    const int32_t pad_w = ((out_w - 1) * p->stride_width + dilated_filter_w - input_w) / 2;

    const int32_t input_row_stride = input_w * input_c;
    const int32_t filter_row_stride = filter_w * out_c;
    const int32_t filter_col_stride = out_c;
    const int32_t output_row_stride = out_w * out_c;

    /* Loop order: batch -> out_h -> out_w -> in_ch -> dm -> fh -> fw.
     * Output channel: oc = dm + in_ch * depth_multiplier. */
    for (int32_t oh = 0; oh < out_h; oh++) {
        const int32_t input_base_h = oh * p->stride_height - pad_h;
        for (int32_t ow = 0; ow < out_w; ow++) {
            const int32_t input_base_w = ow * p->stride_width - pad_w;
            for (int32_t ic = 0; ic < input_c; ic++) {
                for (int32_t d = 0; d < dm; d++) {
                    const int32_t oc = d + ic * dm;
                    int32_t acc = bias[oc];

                    for (int32_t fh = 0; fh < filter_h; fh++) {
                        const int32_t in_h = input_base_h + fh * p->dilation_height_factor;
                        const int32_t row_in_bounds = (in_h >= 0 && in_h < input_h);

                        for (int32_t fw = 0; fw < filter_w; fw++) {
                            const int32_t in_w = input_base_w + fw * p->dilation_width_factor;
                            if (row_in_bounds && in_w >= 0 && in_w < input_w) {
                                const int32_t input_idx =
                                    in_h * input_row_stride + in_w * input_c + ic;
                                const int32_t filter_idx =
                                    fh * filter_row_stride + fw * filter_col_stride + oc;
                                const int32_t i_val = (int32_t)input[input_idx];
                                const int32_t w_val = (int32_t)weights[filter_idx];
                                acc += (i_val + p->input_offset) * w_val;
                            }
                            /* else: zero-padding — skip */
                        }
                    }

                    const int32_t scaled = multiply_by_quantized_multiplier(
                        acc, p->output_multiplier_per_channel[oc],
                        p->output_shift_per_channel[oc]);
                    const int32_t with_offset = scaled + p->output_offset;
                    int32_t clamped = with_offset;
                    if (clamped > p->quantized_activation_max) {
                        clamped = p->quantized_activation_max;
                    } else if (clamped < p->quantized_activation_min) {
                        clamped = p->quantized_activation_min;
                    }
                    output[oh * output_row_stride + ow * out_c + oc] =
                        saturating_cast(clamped);
                }
            }
        }
    }
}

/* ── fully_connected (TFLM FullyConnected loop order) ────────────────────── */

void fully_connected(const int8_t *input, const int8_t *weights,
                     const int32_t *bias, const FcParams *p, int8_t *output)
{
    const int32_t input_dim = p->input_dim;
    const int32_t output_dim = p->output_dim;

    for (int32_t oc = 0; oc < output_dim; oc++) {
        int32_t acc = bias[oc];
        const int8_t *wp = weights + oc * input_dim;
        for (int32_t d = 0; d < input_dim; d++) {
            const int32_t i_val = (int32_t)input[d];
            const int32_t w_val = (int32_t)wp[d];
            acc += (i_val + p->input_offset) * w_val;
        }

        const int32_t scaled = multiply_by_quantized_multiplier(
            acc, p->output_multiplier_per_channel[oc],
            p->output_shift_per_channel[oc]);
        const int32_t with_offset = scaled + p->output_offset;
        int32_t clamped = with_offset;
        if (clamped > p->quantized_activation_max) {
            clamped = p->quantized_activation_max;
        } else if (clamped < p->quantized_activation_min) {
            clamped = p->quantized_activation_min;
        }
        output[oc] = saturating_cast(clamped);
    }
}
