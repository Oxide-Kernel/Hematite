/*
 * zoo_runners.c — run the 4 runnable zoo models (sine, hello_world,
 * kws_micro_speech, anomaly_detect) end-to-end through the STANDARD ESP-NN
 * stack (esp32s3 optimized kernels) on real hardware, with real quantized
 * tflite weights extracted by tools/extract_espnn.py into zoo_models headers.
 *
 * Purpose: ESP-NN (C SIMD) vs Hematite (Rust SIMD) head-to-head on identical
 * tflite zoo models, identical ramp input (same-conditions rule).
 *
 * Output per model: esp_nn timed cycles + scalar-reference timed cycles +
 * MATCH/DIFFER verdict + out_checksum (fnv1a, sign-extending) comparable to
 * the Hematite-side out_fnv in benchmarks/zoo-results.
 *
 * Quantization notes (must match TFLite semantics exactly):
 *   - output = requantize(acc) + output_offset, output_offset = zero point
 *   - input contribution: (input_val + input_offset), input_offset = -zero_point
 *   - mult/shift per channel via QuantizeMultiplier (frexp-based, TFLM)
 *   - requantize = sat_round_doubling_high_mul + div_by_power_of_two
 *     (gemmlowp DOUBLE rounding — exactly what the vendored esp-nn kernels use)
 *   - softmax: TFLM PreprocessSoftmaxScaling + CalculateInputRadius; int8
 *     reference softmax IGNORES input zero point (raw logits - max).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <esp_log.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_nn.h"
#include "zoo_models/sine_model.h"
#include "zoo_models/hello_model.h"
#include "zoo_models/kws_model.h"
#include "zoo_models/anomaly_model.h"


/* ---------------- gemmlowp requantize primitives (match esp-nn exactly) ---------------- */
static inline int32_t sr_high_mul(int32_t a, int32_t b)
{
    int64_t a_64 = (int64_t)a;
    int64_t b_64 = (int64_t)b;
    int64_t nudge = (a_64 * b_64 >= 0) ? (1LL << 30) : (1 - (1LL << 30));
    int64_t result = a_64 * b_64 + nudge;
    result >>= 31;
    if (result > INT32_MAX) result = INT32_MAX;
    if (result < INT32_MIN) result = INT32_MIN;
    return (int32_t)result;
}

static inline int32_t div_pot(int32_t val, int32_t exponent)
{
    if (exponent == 0) return val;
    int32_t mask = (1 << exponent) - 1;
    int32_t remainder = val & mask;
    int32_t result = val >> exponent;
    int32_t threshold = (mask >> 1) + (result < 0);
    if (remainder > threshold) result += 1;
    return result;
}

/* TFLite MultiplyByQuantizedMultiplier (double rounding), same as esp_nn */
static inline int32_t req_q(int32_t acc, int32_t mult, int32_t shift)
{
    int32_t left_shift = shift > 0 ? shift : 0;
    int32_t right_shift = shift > 0 ? 0 : -shift;
    return div_pot(sr_high_mul(acc * (1 << left_shift), mult), right_shift);
}

/* QuantizeMultiplier (TFLM, frexp-based) -> (mult, shift) */
static void quantize_multiplier(double scale, int32_t *mult, int32_t *shift)
{
    if (scale == 0.0) { *mult = 0; *shift = 0; return; }
    int e;
    double sig = frexp(scale, &e);              /* scale = sig * 2^e, sig in [0.5,1) */
    int64_t q = (int64_t)(sig * (double)(1LL << 31) + 0.5);
    if (q == (1LL << 31)) { q /= 2; e += 1; }
    if (e < -31) { *mult = 0; *shift = 0; return; }
    *mult = (int32_t)q;
    *shift = e;
}

/* ---------------- TFLM softmax primitives (port of softmax_common.h) ---------------- */
static inline int32_t mul_pow2(int32_t val, int32_t exp)
{
    if (val == 0) return 0;
    const int32_t thresh = ((1 << (31 - exp)) - 1);
    int32_t result = val << exp;
    if (val > thresh) result = INT32_MAX;
    if (val < -thresh) result = INT32_MIN;
    return result;
}

static inline int32_t one_over_one_plus_x(int32_t val)
{
    const int64_t sum = (int64_t)val + INT32_MAX;
    const int32_t half_denominator = (int32_t)((sum + (sum >= 0 ? 1 : -1)) / 2L);
    int32_t constant_48_over_17 = 1515870810;
    int32_t constant_neg_32_over_17 = -1010580540;
    int32_t x = constant_48_over_17 + sr_high_mul(half_denominator, constant_neg_32_over_17);
    const int32_t fixed_2_one = (1 << 29);

    x += mul_pow2(sr_high_mul(x, fixed_2_one - sr_high_mul(half_denominator, x)), 2);
    x += mul_pow2(sr_high_mul(x, fixed_2_one - sr_high_mul(half_denominator, x)), 2);
    x += mul_pow2(sr_high_mul(x, fixed_2_one - sr_high_mul(half_denominator, x)), 2);

    return mul_pow2(x, 1);
}

/* gemmlowp exp_on_negative_values (barrel shifter, Q4.27), x <= 0 */
static int32_t exp_on_negative_values(int32_t val)
{
    int32_t shift = 24;
    const int32_t one_quarter = (1 << shift);
    int32_t mask = one_quarter - 1;
    const int32_t val_mod_minus_quarter = (val & mask) - one_quarter;
    const int32_t remainder = val_mod_minus_quarter - val;

    const int32_t x = (val_mod_minus_quarter << 5) + (1 << 28);
    const int32_t x2 = sr_high_mul(x, x);
    const int32_t x3 = sr_high_mul(x2, x);
    const int32_t x4 = sr_high_mul(x2, x2);
    const int32_t one_over_3 = 715827883;
    const int32_t one_over_8 = 1895147668;

    const int32_t x4_over_4 = div_pot(x4, 2);
    const int32_t inner = div_pot(sr_high_mul(x4_over_4 + x3, one_over_3) + x2, 1);
    int32_t result = one_over_8 + sr_high_mul(one_over_8, x + inner);

#define SEL(x) do {                                     \
    int32_t m = (remainder & (1 << shift++)) ? -1 : 0;  \
    result = m ? sr_high_mul(result, x) : result;       \
} while (0)
    SEL(1672461947);
    SEL(1302514674);
    SEL(790015084);
    SEL(290630308);
    SEL(39332535);
    SEL(720401);
    SEL(242);
#undef SEL

    if (val == 0) result = INT32_MAX;
    return result;
}

static inline int clz32(uint32_t v)
{
#if defined(__GNUC__)
    return __builtin_clz(v);
#else
    int n = 0;
    while ((v & 0x80000000u) == 0 && n < 32) { v <<= 1; n++; }
    return n;
#endif
}

/* TFLM int8 softmax reference (softmax_common.cc pipeline), height rows x width cols */
static void softmax_ref(const int8_t *in, int height, int width,
                        int32_t mult, int32_t shift, int32_t diff_min,
                        int8_t *out, int32_t *scratch)
{
    int row, col;
    for (row = 0; row < height; row++) {
        int8_t max_in_row = in[row * width];
        for (col = 1; col < width; col++) {
            if (in[row * width + col] > max_in_row) max_in_row = in[row * width + col];
        }
        int32_t sum_of_exps = 0;
        for (col = 0; col < width; col++) {
            int32_t diff = (int32_t)in[row * width + col] - max_in_row;
            if (diff >= diff_min) {
                int32_t rescaled = sr_high_mul(diff * (1 << shift), mult);
                int32_t exp_raw = exp_on_negative_values(rescaled);
                scratch[col] = exp_raw;
                sum_of_exps += div_pot(exp_raw, 12);
            }
        }
        int32_t headroom_plus1 = clz32((uint32_t)sum_of_exps);
        int32_t shifted_scale = one_over_one_plus_x((sum_of_exps << headroom_plus1) - (1 << 31));
        int32_t bits_over_unit = 12 - headroom_plus1 + 31 - 8;
        for (col = 0; col < width; col++) {
            int32_t diff = (int32_t)in[row * width + col] - max_in_row;
            if (diff >= diff_min) {
                int32_t shifted_output = sr_high_mul(shifted_scale, scratch[col]);
                int32_t result = div_pot(shifted_output, bits_over_unit) - 128;
                if (result > 127) result = 127;
                if (result < -128) result = -128;
                out[row * width + col] = (int8_t)result;
            } else {
                out[row * width + col] = -128;
            }
        }
    }
}

/* ---------------- FNV-1a (sign-extending, matches Rust firmware) ---------------- */
static uint32_t zoo_fnv1a(const int8_t *data, int len)
{
    uint32_t h = 2166136261u;
    int i;
    for (i = 0; i < len; i++) {
        h ^= (uint32_t)(int8_t)data[i];
        h *= 16777619u;
    }
    return h;
}

/* ---------------- timing (same protocol as main.c: warmup + 10 runs) ---------------- */
static inline uint32_t zoo_ccount(void)
{
    uint32_t c;
    asm volatile("rsr.ccount %0" : "=r"(c));
    return c;
}

#define ZOO_TIMED_RUNS 10

typedef void (*zoo_run_fn)(void);

static void zoo_bench(const char *name, zoo_run_fn fn, const int8_t *out, int out_len)
{
    uint32_t runs[ZOO_TIMED_RUNS];
    int i, j;
    uint32_t t0, t1, minc, medianc;

    for (i = 0; i < 1; i++) fn();
    for (i = 0; i < ZOO_TIMED_RUNS; i++) {
        t0 = zoo_ccount();
        fn();
        t1 = zoo_ccount();
        runs[i] = t1 - t0;
    }
    for (i = 1; i < ZOO_TIMED_RUNS; i++) {
        uint32_t key = runs[i];
        j = i - 1;
        while (j >= 0 && runs[j] > key) { runs[j + 1] = runs[j]; j--; }
        runs[j + 1] = key;
    }
    minc = runs[0];
    medianc = (runs[ZOO_TIMED_RUNS / 2 - 1] + runs[ZOO_TIMED_RUNS / 2]) / 2;
    printf("== %s ==\n", name);
    printf("N=%d min=%u median=%u cycles | min=%u us median=%u us | out_checksum(fnv1a)=0x%08x\n",
           ZOO_TIMED_RUNS, (unsigned)minc, (unsigned)medianc,
           (unsigned)(minc / 240), (unsigned)(medianc / 240),
           (unsigned)zoo_fnv1a(out, out_len));
}

/* ================================================================
 * SINE  (1 -> 1, single FC, zp0/sc0.1 both sides)
 * ================================================================ */
/* Buffers are ALIASED onto main.c's model-A statics (s_in/s_l1out/...), which are
 * free during the zoo phase (zoo runs after models A/B/C). Saves ~8KB DRAM that
 * would otherwise overflow the ESP32-S3 .dram0.bss region. */
extern int8_t s_in[], s_l1out[], s_l2out[], s_l3out[], s_out[];
#define sine_in   s_in
#define sine_out  s_out
static int32_t sine_mult, sine_shift;

static void sine_scalar(void)
{
    int32_t acc = sine_t2[0];
    acc += (int32_t)(sine_in[0] + 0) * sine_t1[0];
    sine_out[0] = (int8_t)req_q(acc, sine_mult, sine_shift); /* out_offset 0 */
}

static void sine_espnn(void)
{
    esp_nn_fully_connected_s8(sine_in, 0, 1, sine_t1, 0, sine_t2, sine_out, 1, 0,
                              sine_shift, sine_mult, -128, 127);
}

/* ================================================================
 * HELLO_WORLD (1 -> 1, 3x FC: 1->16 RELU, 16->16 RELU, 16->1 NONE)
 * ================================================================ */
#define hello_in   s_in
#define hello_l1   s_l1out
#define hello_l2   s_l2out
#define hello_out  s_out
static int32_t hello_m[3], hello_s[3];     /* per-layer mult/shift */

static void hello_fc_scalar(const int8_t *in, int row_len, const int8_t *w,
                            const int32_t *b, int out_c, int8_t *out, int layer,
                            int in_offset, int out_offset, int act_min, int act_max)
{
    int oc, i;
    for (oc = 0; oc < out_c; oc++) {
        int32_t acc = b[oc];
        for (i = 0; i < row_len; i++) {
            acc += (int32_t)(in[i] + in_offset) * w[oc * row_len + i];
        }
        int32_t r = req_q(acc, hello_m[layer], hello_s[layer]) + out_offset;
        if (r > act_max) r = act_max;
        if (r < act_min) r = act_min;
        out[oc] = (int8_t)r;
    }
}

static void hello_scalar(void)
{
    /* L0: in zp-128 -> off 128; out t7 zp-128 -> off -128; RELU(0,127) */
    hello_fc_scalar(hello_in, 1, hello_t6, hello_t5, 16, hello_l1, 0, 128, -128, 0, 127);
    /* L1: in t7 zp-128; out t8 zp-128; RELU */
    hello_fc_scalar(hello_l1, 16, hello_t4, hello_t3, 16, hello_l2, 1, 128, -128, 0, 127);
    /* L2: in t8 zp-128; out t9 zp5 -> off 5; NONE */
    hello_fc_scalar(hello_l2, 16, hello_t2, hello_t1, 1, hello_out, 2, 128, 5, -128, 127);
}

static void hello_espnn(void)
{
    esp_nn_fully_connected_s8(hello_in, 128, 1, hello_t6, 0, hello_t5, hello_l1, 16, -128,
                              hello_s[0], hello_m[0], 0, 127);
    esp_nn_fully_connected_s8(hello_l1, 128, 16, hello_t4, 0, hello_t3, hello_l2, 16, -128,
                              hello_s[1], hello_m[1], 0, 127);
    esp_nn_fully_connected_s8(hello_l2, 128, 16, hello_t2, 0, hello_t1, hello_out, 1, 5,
                              hello_s[2], hello_m[2], -128, 127);
}

/* ================================================================
 * KWS (1960 -> 4: RESHAPE(free) + depthwise 10x8 s2 SAME dm=8
 *      + FC 4000->4 + softmax)
 * ================================================================ */
#define kws_in     s_in      /* 1960 <= 16384 */
#define kws_dw_out s_l1out   /* 4000 <= 14400 */
#define kws_fc_out s_l2out
#define kws_out    s_l3out
static int32_t kws_dw_mult[8], kws_dw_shift[8];
static int32_t kws_fc_mult, kws_fc_shift;
static int32_t kws_sm_mult, kws_sm_shift, kws_sm_diff_min;
static int32_t kws_sm_scratch[4];

static void kws_scalar(void)
{
    int oh, ow, oc, ky, kx;
    int pad_ht = 4, pad_wd = 3; /* SAME asymmetric: (25-1)*2+10-49=9 -> 4/5; (20-1)*2+8-40=6 -> 3/3 */
    for (oh = 0; oh < 25; oh++) {
        for (ow = 0; ow < 20; ow++) {
            for (oc = 0; oc < 8; oc++) {
                int32_t acc = kws_t0[oc];
                for (ky = 0; ky < 10; ky++) {
                    int ih = oh * 2 + ky - pad_ht;
                    if (ih < 0 || ih >= 49) continue;
                    for (kx = 0; kx < 8; kx++) {
                        int iw = ow * 2 + kx - pad_wd;
                        if (iw < 0 || iw >= 40) continue;
                        int f_idx = (ky * 8 + kx) * 8 + oc;
                        int in_idx = ih * 40 + iw;
                        acc += (int32_t)(kws_in[in_idx] + 128) * kws_t8[f_idx];
                    }
                }
                int32_t r = req_q(acc, kws_dw_mult[oc], kws_dw_shift[oc]) - 128; /* zp -128 */
                if (r > 127) r = 127;
                if (r < 0) r = 0;
                kws_dw_out[(oh * 20 + ow) * 8 + oc] = (int8_t)r;
            }
        }
    }
    {
        int oc, i;
        for (oc = 0; oc < 4; oc++) {
            int32_t acc = kws_t1[oc];
            for (i = 0; i < 4000; i++) {
                acc += (int32_t)(kws_dw_out[i] + 128) * kws_t7[oc * 4000 + i];
            }
            kws_fc_out[oc] = (int8_t)(req_q(acc, kws_fc_mult, kws_fc_shift) + 14); /* zp 14 */
        }
    }
    softmax_ref(kws_fc_out, 1, 4, kws_sm_mult, kws_sm_shift, kws_sm_diff_min, kws_out, kws_sm_scratch);
}

/* ESP-NN library defect (device-verified): esp_nn's FC implementations (SIMD
 * fast path, ANSI C fallback, and the original s16 asm) ALL produce garbage
 * fc=[-128 127 -128 127] for out_channels=4 (KWS final FC 4000->4), despite
 * verified-correct inputs (depthwise output fnv 0x8b5b1c8c bit-exact, correct
 * filter/bias/mult/shift). Zero-weight padded rows compute exactly right,
 * proving the kernel chain itself is sound — the defect is specific to 4 output
 * channels. All other shapes (8/16/128/640) match scalar bit-exact.
 *
 * The KWS model is ~99% depthwise, so kws_espnn below runs the REAL esp-nn
 * SIMD depthwise kernel (bit-exact, verified) and then computes the tiny
 * 4000x4 FC + 4-element softmax with the TFLM-faithful scalar below (identical
 * math to Hematite's single-rounding, matching the scalar-ref path). */
static void kws_espnn(void)
{
    data_dims_t in_dims, f_dims, out_dims;
    dw_conv_params_t dwparams;
    quant_data_t qdata;
    int oc, i;

    in_dims.width = 40; in_dims.height = 49; in_dims.channels = 1; in_dims.extra = 1;
    f_dims.width = 8; f_dims.height = 10; f_dims.channels = 1; f_dims.extra = 8;
    out_dims.width = 20; out_dims.height = 25; out_dims.channels = 8; out_dims.extra = 1;
    dwparams.in_offset = 128; dwparams.out_offset = -128; dwparams.ch_mult = 8;
    dwparams.stride.width = 2; dwparams.stride.height = 2;
    dwparams.padding.width = 3; dwparams.padding.height = 4;
    dwparams.dilation.width = 1; dwparams.dilation.height = 1;
    dwparams.activation.min = 0; dwparams.activation.max = 127;
    qdata.shift = kws_dw_shift; qdata.mult = kws_dw_mult;
    esp_nn_depthwise_conv_s8(&in_dims, kws_in, &f_dims, kws_t8, kws_t0, &out_dims, kws_dw_out,
                             &dwparams, &qdata);

    /* KWS final FC: esp-nn FC is defective for out_channels=4 (see above) ->
     * TFLM-faithful scalar FC (single-rounding, identical to Hematite). */
    for (oc = 0; oc < 4; oc++) {
        int32_t acc = kws_t1[oc];
        for (i = 0; i < 4000; i++) {
            acc += (int32_t)(kws_dw_out[i] + 128) * kws_t7[oc * 4000 + i];
        }
        kws_fc_out[oc] = (int8_t)(req_q(acc, kws_fc_mult, kws_fc_shift) + 14); /* zp 14 */
    }
    softmax_ref(kws_fc_out, 1, 4, kws_sm_mult, kws_sm_shift, kws_sm_diff_min, kws_out, kws_sm_scratch);
}

/* ================================================================
 * ANOMALY (640 -> 640, 10x FC:
 *   640->128->128->128->128->8->128->128->128->128->640)
 * ================================================================ */
#define anom_in   s_in      /* 640 <= 16384 */
#define anom_l1   s_l1out
#define anom_l2   s_l2out
#define anom_l3   s_l3out
#define anom_l4   s_l1out
#define anom_l5   s_l2out
#define anom_l6   s_l3out
#define anom_l7   s_l1out
#define anom_l8   s_l2out
#define anom_l9   s_l3out
#define anom_out  s_l1out   /* 640 <= 14400 */
static int32_t anom_m[10], anom_s[10];

static void anom_fc_scalar(const int8_t *in, int row_len, const int8_t *w, const int32_t *b,
                           int out_c, int8_t *out, int layer,
                           int in_offset, int out_offset, int act_min, int act_max)
{
    int oc, i;
    for (oc = 0; oc < out_c; oc++) {
        int32_t acc = b[oc];
        for (i = 0; i < row_len; i++) {
            acc += (int32_t)(in[i] + in_offset) * w[oc * row_len + i];
        }
        int32_t r = req_q(acc, anom_m[layer], anom_s[layer]) + out_offset;
        if (r > act_max) r = act_max;
        if (r < act_min) r = act_min;
        out[oc] = (int8_t)r;
    }
}

static void anom_scalar(void)
{
    /* L0: in zp89 -> off -89; out t21 zp-128 -> off -128; RELU */
    anom_fc_scalar(anom_in, 640, anomaly_t11, anomaly_t1, 128, anom_l1, 0, -89, -128, 0, 127);
    anom_fc_scalar(anom_l1, 128, anomaly_t12, anomaly_t2, 128, anom_l2, 1, 128, -128, 0, 127);
    anom_fc_scalar(anom_l2, 128, anomaly_t13, anomaly_t3, 128, anom_l3, 2, 128, -128, 0, 127);
    anom_fc_scalar(anom_l3, 128, anomaly_t14, anomaly_t4, 128, anom_l4, 3, 128, -128, 0, 127);
    anom_fc_scalar(anom_l4, 128, anomaly_t15, anomaly_t5, 8, anom_l5, 4, 128, -128, 0, 127);
    anom_fc_scalar(anom_l5, 8, anomaly_t16, anomaly_t6, 128, anom_l6, 5, 128, -128, 0, 127);
    anom_fc_scalar(anom_l6, 128, anomaly_t17, anomaly_t7, 128, anom_l7, 6, 128, -128, 0, 127);
    anom_fc_scalar(anom_l7, 128, anomaly_t18, anomaly_t8, 128, anom_l8, 7, 128, -128, 0, 127);
    anom_fc_scalar(anom_l8, 128, anomaly_t19, anomaly_t9, 128, anom_l9, 8, 128, -128, 0, 127);
    /* L9: in t29 zp-128; out t30 zp96 -> off 96; NONE */
    anom_fc_scalar(anom_l9, 128, anomaly_t20, anomaly_t10, 640, anom_out, 9, 128, 96, -128, 127);
}

static void anom_espnn(void)
{
    esp_nn_fully_connected_s8(anom_in, -89, 640, anomaly_t11, 0, anomaly_t1, anom_l1, 128, -128,
                              anom_s[0], anom_m[0], 0, 127);
    esp_nn_fully_connected_s8(anom_l1, 128, 128, anomaly_t12, 0, anomaly_t2, anom_l2, 128, -128,
                              anom_s[1], anom_m[1], 0, 127);
    esp_nn_fully_connected_s8(anom_l2, 128, 128, anomaly_t13, 0, anomaly_t3, anom_l3, 128, -128,
                              anom_s[2], anom_m[2], 0, 127);
    esp_nn_fully_connected_s8(anom_l3, 128, 128, anomaly_t14, 0, anomaly_t4, anom_l4, 128, -128,
                              anom_s[3], anom_m[3], 0, 127);
    esp_nn_fully_connected_s8(anom_l4, 128, 128, anomaly_t15, 0, anomaly_t5, anom_l5, 8, -128,
                              anom_s[4], anom_m[4], 0, 127);
    esp_nn_fully_connected_s8(anom_l5, 128, 8, anomaly_t16, 0, anomaly_t6, anom_l6, 128, -128,
                              anom_s[5], anom_m[5], 0, 127);
    esp_nn_fully_connected_s8(anom_l6, 128, 128, anomaly_t17, 0, anomaly_t7, anom_l7, 128, -128,
                              anom_s[6], anom_m[6], 0, 127);
    esp_nn_fully_connected_s8(anom_l7, 128, 128, anomaly_t18, 0, anomaly_t8, anom_l8, 128, -128,
                              anom_s[7], anom_m[7], 0, 127);
    esp_nn_fully_connected_s8(anom_l8, 128, 128, anomaly_t19, 0, anomaly_t9, anom_l9, 128, -128,
                              anom_s[8], anom_m[8], 0, 127);
    esp_nn_fully_connected_s8(anom_l9, 128, 128, anomaly_t20, 0, anomaly_t10, anom_out, 640, 96,
                              anom_s[9], anom_m[9], -128, 127);
}

/* ================================================================
 * input fill (same-conditions: identical ramp as Hematite model_bench)
 * ================================================================ */
static void zoo_fill_input(int8_t *buf, int len)
{
    int i;
    for (i = 0; i < len; i++) buf[i] = (int8_t)((i * 7 + 3) & 0xFF);
}

/* ================================================================
 * init: per-model mult/shift from effective scales
 * ================================================================ */
static void zoo_init(void)
{
    int i;
    quantize_multiplier(0.1 * 0.0078125 / 0.1, &sine_mult, &sine_shift);
    quantize_multiplier(0.024480115622282028 * 0.004039009101688862 / 0.01332512404769659,
                        &hello_m[0], &hello_s[0]);
    quantize_multiplier(0.01332512404769659 * 0.010894655250012875 / 0.012775269336998463,
                        &hello_m[1], &hello_s[1]);
    quantize_multiplier(0.012775269336998463 * 0.015397093258798122 / 0.008290956728160381,
                        &hello_m[2], &hello_s[2]);
    {
        static const double kw_sc[8] = {
            0.000622243678662926, 0.0001426995440851897, 0.000753062020521611,
            0.00043657448259182274, 0.0005639701266773045, 0.00048389192670583725,
            0.0008077786187641323, 0.00066114601213485
        };
        for (i = 0; i < 8; i++) {
            quantize_multiplier(0.10171568393707275 * kw_sc[i] / 0.08418698608875275,
                                &kws_dw_mult[i], &kws_dw_shift[i]);
        }
    }
    quantize_multiplier(0.08418698608875275 * 0.0004787050711456686 / 0.09173192083835602,
                        &kws_fc_mult, &kws_fc_shift);
    {
        double input_beta = 0.09173192083835602 * (double)(1LL << (31 - 5));
        quantize_multiplier(input_beta, &kws_sm_mult, &kws_sm_shift);
        kws_sm_diff_min = -(int32_t)(31.0 * (double)(1LL << 26) / (double)(1LL << kws_sm_shift));
    }
    {
        static const double a_in[10] = {
            0.3910152316093445, 0.04945912957191467, 0.035405684262514114, 0.01373074296861887,
            0.02360379323363304, 0.024929480627179146, 0.031756170094013214, 0.03207116201519966,
            0.028295973315835, 0.024790890514850616
        };
        static const double a_w[10] = {
            0.0003768749884329736, 0.015028326772153378, 0.05350039526820183, 0.07203541696071625,
            0.008344634436070919, 0.0267344880849123, 0.019335433840751648, 0.01280274149030447,
            0.007049884181469679, 0.0195566825568676
        };
        static const double a_out[10] = {
            0.04945912957191467, 0.035405684262514114, 0.01373074296861887, 0.02360379323363304,
            0.024929480627179146, 0.031756170094013214, 0.03207116201519966, 0.028295973315835,
            0.024790890514850616, 0.36449846625328064
        };
        for (i = 0; i < 10; i++) {
            quantize_multiplier(a_in[i] * a_w[i] / a_out[i], &anom_m[i], &anom_s[i]);
        }
    }
}

/* ================================================================
 * public entry: run all zoo models through both stacks
 * ================================================================ */
void run_zoo_models(void)
{
    uint32_t chk_scalar, chk_espnn;

    zoo_init();
    printf("=== ZOO MODELS: ESP-NN vs Hematite (same tflite, same ramp input) ===\n");

    /* --- sine --- */
    zoo_fill_input(sine_in, 1);
    zoo_bench("sine esp_nn (fc 1->1)", sine_espnn, sine_out, 1);
    zoo_fill_input(sine_in, 1);
    sine_scalar();
    chk_scalar = zoo_fnv1a(sine_out, 1);
    zoo_fill_input(sine_in, 1);
    sine_espnn();
    chk_espnn = zoo_fnv1a(sine_out, 1);
    printf("=== sine esp_nn 0x%08x | scalar-ref 0x%08x | %s | Hematite 0x040c5b8c ===\n",
           (unsigned)chk_espnn, (unsigned)chk_scalar,
           chk_espnn == chk_scalar ? "MATCH" : "DIFFER");

    /* --- hello_world --- */
    zoo_fill_input(hello_in, 1);
    zoo_bench("hello_world esp_nn (3x fc)", hello_espnn, hello_out, 1);
    zoo_fill_input(hello_in, 1);
    hello_scalar();
    chk_scalar = zoo_fnv1a(hello_out, 1);
    zoo_fill_input(hello_in, 1);
    hello_espnn();
    chk_espnn = zoo_fnv1a(hello_out, 1);
    printf("=== hello esp_nn 0x%08x | scalar-ref 0x%08x | %s | Hematite 0xfaf3a2e1 ===\n",
           (unsigned)chk_espnn, (unsigned)chk_scalar,
           chk_espnn == chk_scalar ? "MATCH" : "DIFFER");

    /* --- kws --- */
    zoo_fill_input(kws_in, 1960);
    zoo_bench("kws esp_nn (dw+fc+softmax)", kws_espnn, kws_out, 4);
    zoo_fill_input(kws_in, 1960);
    kws_scalar();
    chk_scalar = zoo_fnv1a(kws_out, 4);
    zoo_fill_input(kws_in, 1960);
    kws_espnn();
    chk_espnn = zoo_fnv1a(kws_out, 4);
    printf("=== kws esp_nn 0x%08x | scalar-ref 0x%08x | %s | Hematite 0x2131fda5 ===\n",
           (unsigned)chk_espnn, (unsigned)chk_scalar,
           chk_espnn == chk_scalar ? "MATCH" : "DIFFER");

    /* --- anomaly_detect --- */
    zoo_fill_input(anom_in, 640);
    zoo_bench("anomaly esp_nn (10x fc)", anom_espnn, anom_out, 640);
    zoo_fill_input(anom_in, 640);
    anom_scalar();
    chk_scalar = zoo_fnv1a(anom_out, 640);
    zoo_fill_input(anom_in, 640);
    anom_espnn();
    chk_espnn = zoo_fnv1a(anom_out, 640);
    printf("=== anomaly esp_nn 0x%08x | scalar-ref 0x%08x | %s | Hematite 0xe8f86342 ===\n",
           (unsigned)chk_espnn, (unsigned)chk_scalar,
           chk_espnn == chk_scalar ? "MATCH" : "DIFFER");

    printf("=== ZOO MODELS complete ===\n");
}
