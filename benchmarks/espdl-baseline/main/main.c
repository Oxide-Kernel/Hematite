/* espdl-baseline: benchmark the vendored ESP-DL dl_tie728 TIE728 SIMD conv1x1
 * kernel on real ESP32-S3 hardware via ESP-IDF, so output + cycles can be
 * matched against the Rust hematite s3 crate (bench9) and the scalar C
 * qemu-baseline (benchmarks/qemu-baseline).
 *
 * Runs the SAME conv1x1 64x1x1x64 spec (ember-esp-nn row) with the SAME
 * deterministic fill_pattern as the Rust firmware and qemu-baseline:
 *   input[i]  = (i*7+3)&0xFF
 *   weights[i]= (i*13+11)&0xFF
 *   bias[i]   = i*17-8
 *   output[i] = 0
 * and prints an FNV-1a out_checksum over the 64 output bytes so output can be
 * compared bit-exact across implementations.
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "driver/uart.h"
#include "sdkconfig.h"

/* ---- Tie728ConvArgs layout (must match Rust hematite-s3 Tie728ConvArgs) ----
 * Verified against the asm load_args macro (dl_tie728_s8_conv2d.S:85-89):
 *   +48 filter, +64 mac_shift, +68 bias, +76 activation_alpha,
 *   +84 activation_shift, +96 output_channel_div_8, +100 c_div_x_1,
 *   +104 filter_channel_factor
 */
typedef struct {
    uint8_t  _pad0[48];
    int8_t  *filter;
    uint8_t  _pad1[12];
    int32_t  mac_shift;
    int32_t *bias;
    uint8_t  _pad2[4];
    int32_t  activation_alpha;
    uint8_t  _pad3[4];
    int32_t  activation_shift;
    uint8_t  _pad4[8];
    int32_t  output_channel_div_8;
    int32_t  c_div_x_1;
    int16_t *filter_channel_factor;
} Tie728ConvArgs;

extern void dl_tie728_s8_conv2d_11cn(int8_t *output, const int8_t *input, Tie728ConvArgs *args);

#define IN_C   64
#define OUT_C  64
#define WEIGHTS_LEN (IN_C * OUT_C)

static int8_t  s_input[IN_C] __attribute__((aligned(16)));
static int8_t  s_weights[WEIGHTS_LEN] __attribute__((aligned(16)));
static int32_t s_bias[OUT_C] __attribute__((aligned(16)));
static int8_t  s_output[OUT_C] __attribute__((aligned(16)));
static int32_t s_mult[OUT_C] __attribute__((aligned(16)));   /* = 1<<30, like Rust MULT_64 */
static int32_t s_shift[OUT_C] __attribute__((aligned(16)));  /* = 0, like Rust SHIFT_64 */
static Tie728ConvArgs s_args __attribute__((aligned(16)));

static inline uint32_t read_ccount(void) {
    uint32_t c;
    asm volatile("rsr.ccount %0" : "=r"(c));
    return c;
}

static void fill_pattern(void) {
    for (int i = 0; i < IN_C; i++)        s_input[i]  = (int8_t)((i * 7 + 3) & 0xFF);
    for (int i = 0; i < WEIGHTS_LEN; i++) s_weights[i]= (int8_t)((i * 13 + 11) & 0xFF);
    for (int i = 0; i < OUT_C; i++)       s_bias[i]   = (int32_t)(i * 17 - 8);
    for (int i = 0; i < OUT_C; i++)       s_output[i] = 0;
}

static void init_quant_consts(void) {
    for (int i = 0; i < OUT_C; i++) {
        s_mult[i]  = (int32_t)(1 << 30);
        s_shift[i] = 0;
    }
}

static void init_args(void) {
    memset(&s_args, 0, sizeof(s_args));
    s_args.filter = s_weights;
    s_args.mac_shift = 0;               /* shift[0] = 0 -> per-layer path */
    s_args.bias = s_bias;
    s_args.activation_alpha = 0;
    s_args.activation_shift = -1;       /* no fused activation */
    s_args.output_channel_div_8 = OUT_C / 16;   /* 4 */
    s_args.c_div_x_1 = IN_C / 16 - 1;           /* 3 */
    s_args.filter_channel_factor = NULL;
}

static uint32_t fnv1a(const int8_t *data, size_t len) {
    /* Mirrors the Rust firmware's fnv1a: h ^= b as u32 where b is i8, so
     * negative bytes SIGN-EXTEND (0x80 -> 0xffffff80). This is the
     * checksum convention the Rust s3/ref outputs are matched against. */
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < len; i++) {
        h ^= (uint32_t)(int8_t)data[i];
        h *= 16777619u;
    }
    return h;
}

/* scalar reference mirroring hematite-ref/src/conv.rs conv2d (1x1 case) */
static void scalar_conv1x1(int8_t *out, const int8_t *in, const int8_t *w,
                           const int32_t *b) {
    for (int oc = 0; oc < OUT_C; oc++) {
        int32_t acc = b[oc];
        for (int ic = 0; ic < IN_C; ic++) {
            acc += (int32_t)in[ic] * (int32_t)w[oc * IN_C + ic];
        }
        /* multiply_by_quantized_multiplier(acc, mult=1<<30, shift=0):
         *   total_shift = 31 - 0 = 31, round = 1<<30,
         *   result = (acc * 1<<30 + round) >> 31 = (acc + 1) >> 1
         * then + output_offset(0), clamp [-128,127], saturating_cast. */
        int32_t r = (acc + 1) >> 1;
        if (r < -128) r = -128;
        if (r > 127) r = 127;
        out[oc] = (int8_t)r;
    }
}

/* Rust-public-API mirror: the s3 conv2d_1x1 wrapper the firmware times is
 * validation + SIMD-gate checks + dispatch_1x1 (build Tie728ConvArgs, call the
 * asm entry). Time that whole path so the 380 (raw asm) vs 2628 (Rust s3)
 * cycle gap can be attributed: wrapper overhead vs the asm itself. */
static int conv2d_1x1_full(int8_t *out, const int8_t *in, const int8_t *w,
                           const int32_t *b, int32_t mac_shift, int32_t ocd8,
                           int32_t cdiv) {
    /* gate: mult_uniform(==1<<30), shift_uniform(==0), full_range,
     * offsets==0, input_c%16==0, out_channels%16==0, ptrs 16-aligned.
     * (mirrors hematite-s3 conv2d_1x1 SIMD eligibility) */
    for (int c = 0; c < IN_C; c++) {
        if (s_mult[c] != (int32_t)(1 << 30)) return -1;
        if (s_shift[c] != 0) return -1;
    }
    if (((uint32_t)in & 15) || ((uint32_t)w & 15) || ((uint32_t)b & 15) ||
        ((uint32_t)out & 15))
        return -1;
    Tie728ConvArgs a;
    a.filter = (int8_t *)w;
    a.mac_shift = mac_shift;
    a.bias = (int32_t *)b;
    a.activation_alpha = 0;
    a.activation_shift = -1;
    a.output_channel_div_8 = ocd8;
    a.c_div_x_1 = cdiv;
    a.filter_channel_factor = NULL;
    dl_tie728_s8_conv2d_11cn(out, in, &a);
    return 0;
}

void app_main(void) {
    printf("\n=== Hematite ESP-DL baseline (benchmarks/espdl-baseline) ===\n");
    printf("LABEL: real hardware (ESP32-S3 @ 240MHz), ESP-IDF v5.5.1, vendored dl_tie728 asm\n");

    fill_pattern();
    init_quant_consts();
    init_args();

    const int WARMUP = 1, TIMED = 10;

    /* (a) raw asm entry: what the C harness directly times */
    for (int r = 0; r < WARMUP; r++) {
        dl_tie728_s8_conv2d_11cn(s_output, s_input, &s_args);
    }
    uint32_t runs_raw[TIMED];
    for (int r = 0; r < TIMED; r++) {
        fill_pattern();
        uint32_t t0 = read_ccount();
        dl_tie728_s8_conv2d_11cn(s_output, s_input, &s_args);
        uint32_t t1 = read_ccount();
        runs_raw[r] = t1 - t0;
    }

    /* (b) full Rust-public-API mirror (validation + gate + dispatch + asm) */
    for (int r = 0; r < WARMUP; r++) {
        conv2d_1x1_full(s_output, s_input, s_weights, s_bias, 0, 4, 3);
    }
    uint32_t runs_full[TIMED];
    for (int r = 0; r < TIMED; r++) {
        fill_pattern();
        uint32_t t0 = read_ccount();
        conv2d_1x1_full(s_output, s_input, s_weights, s_bias, 0, 4, 3);
        uint32_t t1 = read_ccount();
        runs_full[r] = t1 - t0;
    }

    /* insertion sort both */
    for (int i = 1; i < TIMED; i++) {
        uint32_t v = runs_raw[i];
        int j = i - 1;
        while (j >= 0 && runs_raw[j] > v) { runs_raw[j + 1] = runs_raw[j]; j--; }
        runs_raw[j + 1] = v;
    }
    for (int i = 1; i < TIMED; i++) {
        uint32_t v = runs_full[i];
        int j = i - 1;
        while (j >= 0 && runs_full[j] > v) { runs_full[j + 1] = runs_full[j]; j--; }
        runs_full[j + 1] = v;
    }
    uint32_t min_raw = runs_raw[0];
    uint32_t med_raw = (runs_raw[TIMED / 2 - 1] + runs_raw[TIMED / 2]) / 2;
    uint32_t min_full = runs_full[0];
    uint32_t med_full = (runs_full[TIMED / 2 - 1] + runs_full[TIMED / 2]) / 2;

    /* scalar reference on identical data for comparison */
    static int8_t s_ref[OUT_C] __attribute__((aligned(16)));
    scalar_conv1x1(s_ref, s_input, s_weights, s_bias);
    uint32_t chk_ref = fnv1a((const int8_t *)s_ref, OUT_C);
    uint32_t chk = fnv1a((const int8_t *)s_output, OUT_C);

    printf("== conv1x1_s8 64x1x1x64 TIE728-SIMD (dl_tie728_s8_conv2d_11cn) ==\n");
    printf("raw-asm:  N=%d min=%u median=%u cycles | min=%.2fus median=%.2fus | out_checksum(fnv1a)=0x%08x\n",
           TIMED, (unsigned)min_raw, (unsigned)med_raw,
           (double)min_raw / 240.0, (double)med_raw / 240.0, (unsigned)chk);
    printf("full-API: N=%d min=%u median=%u cycles | min=%.2fus median=%.2fus\n",
           TIMED, (unsigned)min_full, (unsigned)med_full,
           (double)min_full / 240.0, (double)med_full / 240.0);
    printf("  (Rust bench9 s3: median=2628 cycles; scalar-ref: 0x0bea8225)\n");
    printf("  scalar-ref fnv1a=0x%08x  simd-fnv1a=0x%08x\n",
           (unsigned)chk_ref, (unsigned)chk);
    printf("  simd out[0..15]=");
    for (int i = 0; i < 16; i++) printf("%02x ", (uint8_t)s_output[i]);
    printf(" ref[0..15]=");
    for (int i = 0; i < 16; i++) printf("%02x ", (uint8_t)s_ref[i]);
    printf("\n");
    printf("=== benchmark complete ===\n");
}
