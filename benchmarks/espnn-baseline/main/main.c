/*
 * espnn-baseline — run quantized int8 CNN models end-to-end through
 * the STANDARD ESP-NN stack (espressif/esp-nn, esp32s3 optimized kernels)
 * on real hardware (ESP32-S3 @ 240MHz) and print cycles + output checksum.
 *
 * Two models (identical to the Hematite-side model runners):
 *
 *   MODEL A (4-layer, first comparison):
 *     L1 conv3x3   32x32x16  -> 30x30x16  stride1 VALID  act(0,127)
 *     L2 max_pool  2x2 s2     -> 15x15x16
 *     L3 conv1x1   15x15x16   -> 15x15x32  stride1         act(0,127)
 *     L4 fc        7200       -> 16
 *
 *   MODEL B (mv2mini, MobileNetV2-style — representative mix of
 *   first conv 3->32, depthwise bottlenecks, pointwise expand, fc):
 *     L1 conv3x3   16x16x3    -> 14x14x32  stride1 VALID  act(0,127)
 *     L2 max_pool  2x2 s2     -> 7x7x32
 *     L3 depthwise 3x3 7x7x32 -> 5x5x32    stride1 VALID  act(0,127) dm=1
 *     L4 conv1x1   5x5x32     -> 5x5x64    stride1         act(0,127)
 *     L5 depthwise 3x3 5x5x64 -> 3x3x64    stride1 VALID  act(0,127) dm=1
 *     L6 conv1x1   3x3x64     -> 3x3x128   stride1         act(0,127)
 *     L7 fc        1152       -> 16
 *
 * Deterministic fill (matches hematite-benchmarks spec.rs fill_pattern):
 *   input[i]=(i*7+3)&0xFF, weights[i]=(i*13+11)&0xFF, bias[i]=i*17-8
 *
 * Scalar reference kernels mirror hematite-int8 requantize
 * (multiply_by_quantized_multiplier) so we can verify on-device whether
 * the esp_nn optimized kernels are bit-exact.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <esp_log.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_nn.h"

static const char *TAG = "espnn-baseline";

/* ---------------- model A shapes ---------------- */
#define L1_IN_H 32
#define L1_IN_W 32
#define L1_IN_C 16
#define L1_OUT_H 30
#define L1_OUT_W 30
#define L1_OUT_C 16

#define L2_OUT_H 15
#define L2_OUT_W 15
#define L2_OUT_C 16

#define L3_IN_C 16
#define L3_OUT_C 32
#define L3_OUT_H 15
#define L3_OUT_W 15

#define L4_IN_DIM (L3_OUT_H * L3_OUT_W * L3_OUT_C) /* 7200 */
#define L4_OUT_C 16

/* ---------------- model B (mv2mini) shapes ---------------- */
#define M1_IN_H 16
#define M1_IN_W 16
#define M1_IN_C 3
#define M1_OUT_H 14
#define M1_OUT_W 14
#define M1_OUT_C 32

#define M2_OUT_H 7
#define M2_OUT_W 7
#define M2_OUT_C 32

#define M3_OUT_H 5
#define M3_OUT_W 5
#define M3_OUT_C 32

#define M4_OUT_H 5
#define M4_OUT_W 5
#define M4_OUT_C 64

#define M5_OUT_H 3
#define M5_OUT_W 3
#define M5_OUT_C 64

#define M6_OUT_H 3
#define M6_OUT_W 3
#define M6_OUT_C 128

#define M7_IN_DIM (M6_OUT_H * M6_OUT_W * M6_OUT_C) /* 1152 */
#define M7_OUT_C 16

/* ---------------- model C (mv2real, MobileNetV2-style) shapes ---------------- */
#define C1_IN_H 16
#define C1_IN_W 16
#define C1_IN_C 3
#define C1_OUT_H 8
#define C1_OUT_W 8
#define C1_OUT_C 32

#define C2_OUT_H 8
#define C2_OUT_W 8
#define C2_OUT_C 32

#define C3_OUT_H 8
#define C3_OUT_W 8
#define C3_OUT_C 64

#define C4_OUT_H 4
#define C4_OUT_W 4
#define C4_OUT_C 64

#define C5_OUT_H 4
#define C5_OUT_W 4
#define C5_OUT_C 128

#define C6_IN_DIM (C5_OUT_H * C5_OUT_W * C5_OUT_C) /* 2048 */
#define C6_OUT_C 16

#define MULT_Q30 (1 << 30)

/* ---------------- model A buffers (16-aligned) ---------------- */
static int8_t __attribute__((aligned(16))) s_in[L1_IN_H * L1_IN_W * L1_IN_C];       /* 16384 */
static int8_t __attribute__((aligned(16))) s_l1out[L1_OUT_H * L1_OUT_W * L1_OUT_C]; /* 14400 */
static int8_t __attribute__((aligned(16))) s_l2out[L2_OUT_H * L2_OUT_W * L2_OUT_C]; /* 3600 */
static int8_t __attribute__((aligned(16))) s_l3out[L3_OUT_H * L3_OUT_W * L3_OUT_C]; /* 7200 */
static int8_t __attribute__((aligned(16))) s_out[L4_OUT_C];                          /* 16 */

static int8_t __attribute__((aligned(16))) s_l1w[L1_OUT_C * 3 * 3 * L1_IN_C]; /* 2304 */
static int32_t __attribute__((aligned(16))) s_l1b[L1_OUT_C];
static int8_t __attribute__((aligned(16))) s_l3w[L3_OUT_C * L3_IN_C]; /* 512 */
static int32_t __attribute__((aligned(16))) s_l3b[L3_OUT_C];
static int8_t __attribute__((aligned(16))) s_l4w[L4_IN_DIM * L4_OUT_C]; /* 115200 */
static int32_t __attribute__((aligned(16))) s_l4b[L4_OUT_C];

static int32_t __attribute__((aligned(16))) s_shift[128];
static int32_t __attribute__((aligned(16))) s_mult[128];

/* ---------------- model B buffers (16-aligned) ---------------- */
static int8_t __attribute__((aligned(16))) m_in[M1_IN_H * M1_IN_W * M1_IN_C];       /* 768 */
static int8_t __attribute__((aligned(16))) m_l1out[M1_OUT_H * M1_OUT_W * M1_OUT_C]; /* 6272 */
static int8_t __attribute__((aligned(16))) m_l2out[M2_OUT_H * M2_OUT_W * M2_OUT_C]; /* 1568 */
static int8_t __attribute__((aligned(16))) m_l3out[M3_OUT_H * M3_OUT_W * M3_OUT_C]; /* 800 */
static int8_t __attribute__((aligned(16))) m_l4out[M4_OUT_H * M4_OUT_W * M4_OUT_C]; /* 1600 */
static int8_t __attribute__((aligned(16))) m_l5out[M5_OUT_H * M5_OUT_W * M5_OUT_C]; /* 576 */
static int8_t __attribute__((aligned(16))) m_l6out[M6_OUT_H * M6_OUT_W * M6_OUT_C]; /* 1152 */
static int8_t __attribute__((aligned(16))) m_out[M7_OUT_C];                          /* 16 */

static int8_t __attribute__((aligned(16))) m_l1w[M1_OUT_C * 3 * 3 * M1_IN_C]; /* 432 */
static int32_t __attribute__((aligned(16))) m_l1b[M1_OUT_C];
static int8_t __attribute__((aligned(16))) m_l3w[3 * 3 * M3_OUT_C]; /* 288 (depthwise) */
static int32_t __attribute__((aligned(16))) m_l3b[M3_OUT_C];
static int8_t __attribute__((aligned(16))) m_l4w[M4_OUT_C * M4_OUT_C]; /* 2048 */
static int32_t __attribute__((aligned(16))) m_l4b[M4_OUT_C];
static int8_t __attribute__((aligned(16))) m_l5w[3 * 3 * M5_OUT_C]; /* 576 (depthwise) */
static int32_t __attribute__((aligned(16))) m_l5b[M5_OUT_C];
static int8_t __attribute__((aligned(16))) m_l6w[M6_OUT_C * M6_OUT_C]; /* 8192 */
static int32_t __attribute__((aligned(16))) m_l6b[M6_OUT_C];
static int8_t __attribute__((aligned(16))) m_l7w[M7_IN_DIM * M7_OUT_C]; /* 18432 */
static int32_t __attribute__((aligned(16))) m_l7b[M7_OUT_C];

/* ---------------- model C (mv2real) buffers (16-aligned arena) ----------------
 * Models A/B/C run strictly sequentially in app_main, so model C reuses a
 * single 16-aligned arena instead of its own DRAM footprint. */
#define C_ARENA_BYTES (768 + 2048 + 2048 + 4096 + 1024 + 2048 + 16 \
                     + 864 + 32 * 4 + 288 + 32 * 4 + 2048 + 64 * 4 \
                     + 576 + 64 * 4 + 8192 + 128 * 4 + 32768 + 16 * 4)
static int8_t __attribute__((aligned(16))) c_arena[C_ARENA_BYTES];
#define c_in      ((int8_t *)(c_arena + 0))                            /* 768 */
#define c_l1out   ((int8_t *)(c_arena + 768))                          /* 2048 */
#define c_l2out   ((int8_t *)(c_arena + 768 + 2048))                   /* 2048 */
#define c_l3out   ((int8_t *)(c_arena + 768 + 2048 + 2048))            /* 4096 */
#define c_l4out   ((int8_t *)(c_arena + 768 + 2048 + 2048 + 4096))     /* 1024 */
#define c_l5out   ((int8_t *)(c_arena + 768 + 2048 + 2048 + 4096 + 1024)) /* 2048 */
#define c_out     ((int8_t *)(c_arena + 768 + 2048 + 2048 + 4096 + 1024 + 2048)) /* 16 */
#define c_l1w     ((int8_t *)(c_arena + 768 + 2048 + 2048 + 4096 + 1024 + 2048 + 16)) /* 864 */
#define c_l1b     ((int32_t *)(c_arena + 768 + 2048 + 2048 + 4096 + 1024 + 2048 + 16 + 864)) /* 32 */
#define c_l2w     ((int8_t *)(c_arena + 768 + 2048 + 2048 + 4096 + 1024 + 2048 + 16 + 864 + 32 * 4)) /* 288 */
#define c_l2b     ((int32_t *)(c_arena + 768 + 2048 + 2048 + 4096 + 1024 + 2048 + 16 + 864 + 32 * 4 + 288)) /* 32 */
#define c_l3w     ((int8_t *)(c_arena + 768 + 2048 + 2048 + 4096 + 1024 + 2048 + 16 + 864 + 32 * 4 + 288 + 32 * 4)) /* 2048 */
#define c_l3b     ((int32_t *)(c_arena + 768 + 2048 + 2048 + 4096 + 1024 + 2048 + 16 + 864 + 32 * 4 + 288 + 32 * 4 + 2048)) /* 64 */
#define c_l4w     ((int8_t *)(c_arena + 768 + 2048 + 2048 + 4096 + 1024 + 2048 + 16 + 864 + 32 * 4 + 288 + 32 * 4 + 2048 + 64 * 4)) /* 576 */
#define c_l4b     ((int32_t *)(c_arena + 768 + 2048 + 2048 + 4096 + 1024 + 2048 + 16 + 864 + 32 * 4 + 288 + 32 * 4 + 2048 + 64 * 4 + 576)) /* 64 */
#define c_l5w     ((int8_t *)(c_arena + 768 + 2048 + 2048 + 4096 + 1024 + 2048 + 16 + 864 + 32 * 4 + 288 + 32 * 4 + 2048 + 64 * 4 + 576 + 64 * 4)) /* 8192 */
#define c_l5b     ((int32_t *)(c_arena + 768 + 2048 + 2048 + 4096 + 1024 + 2048 + 16 + 864 + 32 * 4 + 288 + 32 * 4 + 2048 + 64 * 4 + 576 + 64 * 4 + 8192)) /* 128 */
#define c_l6w     ((int8_t *)(c_arena + 768 + 2048 + 2048 + 4096 + 1024 + 2048 + 16 + 864 + 32 * 4 + 288 + 32 * 4 + 2048 + 64 * 4 + 576 + 64 * 4 + 8192 + 128 * 4)) /* 32768 */
#define c_l6b     ((int32_t *)(c_arena + 768 + 2048 + 2048 + 4096 + 1024 + 2048 + 16 + 864 + 32 * 4 + 288 + 32 * 4 + 2048 + 64 * 4 + 576 + 64 * 4 + 8192 + 128 * 4 + 32768)) /* 16 */

/* scratch for esp_nn conv/depthwise */
static int8_t __attribute__((aligned(16))) s_scratch[12 * 1024];

/* ---------------- fill ---------------- */
static void fill_pattern(void)
{
    int i;
    for (i = 0; i < L1_IN_H * L1_IN_W * L1_IN_C; i++) s_in[i] = (int8_t)((i * 7 + 3) & 0xFF);
    for (i = 0; i < L1_OUT_C * 3 * 3 * L1_IN_C; i++) s_l1w[i] = (int8_t)((i * 13 + 11) & 0xFF);
    for (i = 0; i < L1_OUT_C; i++) s_l1b[i] = i * 17 - 8;
    for (i = 0; i < L3_OUT_C * L3_IN_C; i++) s_l3w[i] = (int8_t)((i * 13 + 11) & 0xFF);
    for (i = 0; i < L3_OUT_C; i++) s_l3b[i] = i * 17 - 8;
    for (i = 0; i < L4_IN_DIM * L4_OUT_C; i++) s_l4w[i] = (int8_t)((i * 13 + 11) & 0xFF);
    for (i = 0; i < L4_OUT_C; i++) s_l4b[i] = i * 17 - 8;
    for (i = 0; i < 128; i++) { s_shift[i] = 0; s_mult[i] = MULT_Q30; }
    memset(s_l1out, 0, sizeof(s_l1out));
    memset(s_l2out, 0, sizeof(s_l2out));
    memset(s_l3out, 0, sizeof(s_l3out));
    memset(s_out, 0, sizeof(s_out));
}

static void fill_pattern_mv2(void)
{
    int i;
    for (i = 0; i < M1_IN_H * M1_IN_W * M1_IN_C; i++) m_in[i] = (int8_t)((i * 7 + 3) & 0xFF);
    for (i = 0; i < M1_OUT_C * 3 * 3 * M1_IN_C; i++) m_l1w[i] = (int8_t)((i * 13 + 11) & 0xFF);
    for (i = 0; i < M1_OUT_C; i++) m_l1b[i] = i * 17 - 8;
    for (i = 0; i < 3 * 3 * M3_OUT_C; i++) m_l3w[i] = (int8_t)((i * 13 + 11) & 0xFF);
    for (i = 0; i < M3_OUT_C; i++) m_l3b[i] = i * 17 - 8;
    for (i = 0; i < M4_OUT_C * M4_OUT_C; i++) m_l4w[i] = (int8_t)((i * 13 + 11) & 0xFF);
    for (i = 0; i < M4_OUT_C; i++) m_l4b[i] = i * 17 - 8;
    for (i = 0; i < 3 * 3 * M5_OUT_C; i++) m_l5w[i] = (int8_t)((i * 13 + 11) & 0xFF);
    for (i = 0; i < M5_OUT_C; i++) m_l5b[i] = i * 17 - 8;
    for (i = 0; i < M6_OUT_C * M6_OUT_C; i++) m_l6w[i] = (int8_t)((i * 13 + 11) & 0xFF);
    for (i = 0; i < M6_OUT_C; i++) m_l6b[i] = i * 17 - 8;
    for (i = 0; i < M7_IN_DIM * M7_OUT_C; i++) m_l7w[i] = (int8_t)((i * 13 + 11) & 0xFF);
    for (i = 0; i < M7_OUT_C; i++) m_l7b[i] = i * 17 - 8;
    for (i = 0; i < 128; i++) { s_shift[i] = 0; s_mult[i] = MULT_Q30; }
    memset(m_l1out, 0, sizeof(m_l1out));
    memset(m_l2out, 0, sizeof(m_l2out));
    memset(m_l3out, 0, sizeof(m_l3out));
    memset(m_l4out, 0, sizeof(m_l4out));
    memset(m_l5out, 0, sizeof(m_l5out));
    memset(m_l6out, 0, sizeof(m_l6out));
    memset(m_out, 0, sizeof(m_out));
}

static void fill_pattern_mv2real(void)
{
    int i;
    for (i = 0; i < C1_IN_H * C1_IN_W * C1_IN_C; i++) c_in[i] = (int8_t)((i * 7 + 3) & 0xFF);
    for (i = 0; i < C1_OUT_C * 3 * 3 * C1_IN_C; i++) c_l1w[i] = (int8_t)((i * 13 + 11) & 0xFF);
    for (i = 0; i < C1_OUT_C; i++) c_l1b[i] = i * 17 - 8;
    for (i = 0; i < 3 * 3 * C2_OUT_C; i++) c_l2w[i] = (int8_t)((i * 13 + 11) & 0xFF);
    for (i = 0; i < C2_OUT_C; i++) c_l2b[i] = i * 17 - 8;
    for (i = 0; i < C3_OUT_C * C2_OUT_C; i++) c_l3w[i] = (int8_t)((i * 13 + 11) & 0xFF);
    for (i = 0; i < C3_OUT_C; i++) c_l3b[i] = i * 17 - 8;
    for (i = 0; i < 3 * 3 * C4_OUT_C; i++) c_l4w[i] = (int8_t)((i * 13 + 11) & 0xFF);
    for (i = 0; i < C4_OUT_C; i++) c_l4b[i] = i * 17 - 8;
    for (i = 0; i < C5_OUT_C * C4_OUT_C; i++) c_l5w[i] = (int8_t)((i * 13 + 11) & 0xFF);
    for (i = 0; i < C5_OUT_C; i++) c_l5b[i] = i * 17 - 8;
    for (i = 0; i < C6_IN_DIM * C6_OUT_C; i++) c_l6w[i] = (int8_t)((i * 13 + 11) & 0xFF);
    for (i = 0; i < C6_OUT_C; i++) c_l6b[i] = i * 17 - 8;
    for (i = 0; i < 128; i++) { s_shift[i] = 0; s_mult[i] = MULT_Q30; }
    memset(c_l1out, 0, 2048);
    memset(c_l2out, 0, 2048);
    memset(c_l3out, 0, 4096);
    memset(c_l4out, 0, 1024);
    memset(c_l5out, 0, 2048);
    memset(c_out, 0, 16);
}

/* ---------------- FNV-1a (sign-extending, matches Rust firmware) ---------------- */
static uint32_t fnv1a(const int8_t *data, int len)
{
    uint32_t h = 2166136261u;
    int i;
    for (i = 0; i < len; i++) {
        h ^= (uint32_t)(int8_t)data[i];
        h *= 16777619u;
    }
    return h;
}

static void dump_bytes(const char *tag, const int8_t *p, int n)
{
    int i;
    printf("  %s:", tag);
    for (i = 0; i < n; i++) {
        printf(" %02x", (uint8_t)p[i]);
    }
    printf("\n");
}

/* ---------------- timing ---------------- */
static inline uint32_t read_ccount(void)
{
    uint32_t c;
    asm volatile("rsr.ccount %0" : "=r"(c));
    return c;
}

#define TIMED_RUNS 10
#define MAX_RUNS 64

typedef void (*run_fn)(void);

static void run_bench(const char *name, run_fn fn, const int8_t *out, int out_len, uint32_t *out_checksum)
{
    uint32_t runs[MAX_RUNS];
    int n = 0, i, j;
    uint32_t t0, t1;
    uint32_t minc, medianc;

    for (i = 0; i < 1; i++) { fn(); } /* warmup */
    for (n = 0; n < TIMED_RUNS; n++) {
        t0 = read_ccount();
        fn();
        t1 = read_ccount();
        runs[n] = t1 - t0;
    }
    /* insertion sort */
    for (i = 1; i < n; i++) {
        uint32_t key = runs[i];
        j = i - 1;
        while (j >= 0 && runs[j] > key) { runs[j + 1] = runs[j]; j--; }
        runs[j + 1] = key;
    }
    minc = runs[0];
    medianc = (runs[TIMED_RUNS / 2 - 1] + runs[TIMED_RUNS / 2]) / 2;

    *out_checksum = fnv1a(out, out_len);
    printf("== %s ==\n", name);
    printf("N=%d min=%u median=%u cycles | min=%u us median=%u us | out_checksum(fnv1a)=0x%08x\n",
           TIMED_RUNS, (unsigned)minc, (unsigned)medianc,
           (unsigned)(minc / 240), (unsigned)(medianc / 240),
           (unsigned)*out_checksum);
}

/* ---------------- scalar reference layers (TFLite/hematite semantics) ---------------- */
static int32_t req(int32_t acc, int32_t mult, int32_t shift)
{
    int32_t total_shift = 31 - shift;
    int64_t round = 1LL << (total_shift - 1);
    int64_t r = ((int64_t)acc * mult + round) >> total_shift;
    if (r > 2147483647LL) r = 2147483647LL;
    if (r < -2147483648LL) r = -2147483648LL;
    return (int32_t)r;
}
static int8_t sat8(int32_t v)
{
    if (v > 127) return 127;
    if (v < -128) return -128;
    return (int8_t)v;
}
static int imax(int a, int b) { return a > b ? a : b; }

/* ================= MODEL A scalar refs ================= */
static void ref_conv3x3(void)
{
    /* 32x32x16 -> 30x30x16, stride1, VALID, act(0,127) */
    int oh, ow, oc, kh, kw, ic;
    for (oh = 0; oh < L1_OUT_H; oh++) {
        for (ow = 0; ow < L1_OUT_W; ow++) {
            for (oc = 0; oc < L1_OUT_C; oc++) {
                int32_t acc = s_l1b[oc];
                for (kh = 0; kh < 3; kh++) {
                    for (kw = 0; kw < 3; kw++) {
                        int ih = oh + kh, iw = ow + kw;
                        for (ic = 0; ic < L1_IN_C; ic++) {
                            int w_idx = ((oc * 3 + kh) * 3 + kw) * L1_IN_C + ic;
                            acc += (int32_t)s_in[(ih * L1_IN_W + iw) * L1_IN_C + ic] * s_l1w[w_idx];
                        }
                    }
                }
                int32_t sc = req(acc, MULT_Q30, 0);
                int32_t cl = sc > 127 ? 127 : (sc < 0 ? 0 : sc);
                s_l1out[(oh * L1_OUT_W + ow) * L1_OUT_C + oc] = sat8(cl);
            }
        }
    }
}

static void ref_maxpool(void)
{
    int oh, ow, oc;
    for (oh = 0; oh < L2_OUT_H; oh++) {
        for (ow = 0; ow < L2_OUT_W; ow++) {
            for (oc = 0; oc < L2_OUT_C; oc++) {
                int32_t m = -128;
                int kh, kw;
                for (kh = 0; kh < 2; kh++) {
                    for (kw = 0; kw < 2; kw++) {
                        int8_t v = s_l1out[((oh * 2 + kh) * L1_OUT_W + (ow * 2 + kw)) * L2_OUT_C + oc];
                        if (v > m) m = v;
                    }
                }
                s_l2out[(oh * L2_OUT_W + ow) * L2_OUT_C + oc] = (int8_t)m;
            }
        }
    }
}

static void ref_conv1x1(void)
{
    int h, w, oc, ic;
    for (h = 0; h < L3_OUT_H; h++) {
        for (w = 0; w < L3_OUT_W; w++) {
            for (oc = 0; oc < L3_OUT_C; oc++) {
                int32_t acc = s_l3b[oc];
                for (ic = 0; ic < L3_IN_C; ic++) {
                    acc += (int32_t)s_l2out[(h * L3_OUT_W + w) * L3_IN_C + ic] * s_l3w[oc * L3_IN_C + ic];
                }
                int32_t sc = req(acc, MULT_Q30, 0);
                int32_t cl = sc > 127 ? 127 : (sc < 0 ? 0 : sc);
                s_l3out[(h * L3_OUT_W + w) * L3_OUT_C + oc] = sat8(cl);
            }
        }
    }
}

static void ref_fc(void)
{
    int oc, i;
    for (oc = 0; oc < L4_OUT_C; oc++) {
        int32_t acc = s_l4b[oc];
        for (i = 0; i < L4_IN_DIM; i++) {
            acc += (int32_t)s_l3out[i] * s_l4w[oc * L4_IN_DIM + i];
        }
        int32_t sc = req(acc, MULT_Q30, 0);
        int32_t cl = sc > 127 ? 127 : (sc < -128 ? -128 : sc);
        s_out[oc] = sat8(cl);
    }
}

static void run_scalar_model(void)
{
    ref_conv3x3();
    ref_maxpool();
    ref_conv1x1();
    ref_fc();
}

/* ================= MODEL B (mv2mini) scalar refs ================= */
static void ref_m1_conv3x3(void)
{
    /* 16x16x3 -> 14x14x32, stride1, VALID, act(0,127) */
    int oh, ow, oc, kh, kw, ic;
    for (oh = 0; oh < M1_OUT_H; oh++) {
        for (ow = 0; ow < M1_OUT_W; ow++) {
            for (oc = 0; oc < M1_OUT_C; oc++) {
                int32_t acc = m_l1b[oc];
                for (kh = 0; kh < 3; kh++) {
                    for (kw = 0; kw < 3; kw++) {
                        int ih = oh + kh, iw = ow + kw;
                        for (ic = 0; ic < M1_IN_C; ic++) {
                            int w_idx = ((oc * 3 + kh) * 3 + kw) * M1_IN_C + ic;
                            acc += (int32_t)m_in[(ih * M1_IN_W + iw) * M1_IN_C + ic] * m_l1w[w_idx];
                        }
                    }
                }
                int32_t sc = req(acc, MULT_Q30, 0);
                int32_t cl = sc > 127 ? 127 : (sc < 0 ? 0 : sc);
                m_l1out[(oh * M1_OUT_W + ow) * M1_OUT_C + oc] = sat8(cl);
            }
        }
    }
}

static void ref_m2_maxpool(void)
{
    int oh, ow, oc, kh, kw;
    for (oh = 0; oh < M2_OUT_H; oh++) {
        for (ow = 0; ow < M2_OUT_W; ow++) {
            for (oc = 0; oc < M2_OUT_C; oc++) {
                int32_t m = -128;
                for (kh = 0; kh < 2; kh++) {
                    for (kw = 0; kw < 2; kw++) {
                        int8_t v = m_l1out[((oh * 2 + kh) * M1_OUT_W + (ow * 2 + kw)) * M2_OUT_C + oc];
                        if (v > m) m = v;
                    }
                }
                m_l2out[(oh * M2_OUT_W + ow) * M2_OUT_C + oc] = (int8_t)m;
            }
        }
    }
}

static void ref_m3_depthwise(void)
{
    /* 7x7x32 -> 5x5x32, 3x3 stride1 VALID, dm=1, act(0,127)
     * filter layout HWCN dm=1: filter[(ky*3+kx)*channels + oc] */
    int oh, ow, oc, ky, kx;
    for (oh = 0; oh < M3_OUT_H; oh++) {
        for (ow = 0; ow < M3_OUT_W; ow++) {
            for (oc = 0; oc < M3_OUT_C; oc++) {
                int32_t acc = m_l3b[oc];
                for (ky = 0; ky < 3; ky++) {
                    for (kx = 0; kx < 3; kx++) {
                        int ih = oh + ky, iw = ow + kx;
                        acc += (int32_t)m_l2out[(ih * M2_OUT_W + iw) * M2_OUT_C + oc] *
                               m_l3w[(ky * 3 + kx) * M3_OUT_C + oc];
                    }
                }
                int32_t sc = req(acc, MULT_Q30, 0);
                int32_t cl = sc > 127 ? 127 : (sc < 0 ? 0 : sc);
                m_l3out[(oh * M3_OUT_W + ow) * M3_OUT_C + oc] = sat8(cl);
            }
        }
    }
}

static void ref_m4_conv1x1(void)
{
    int h, w, oc, ic;
    for (h = 0; h < M4_OUT_H; h++) {
        for (w = 0; w < M4_OUT_W; w++) {
            for (oc = 0; oc < M4_OUT_C; oc++) {
                int32_t acc = m_l4b[oc];
                for (ic = 0; ic < M3_OUT_C; ic++) {
                    acc += (int32_t)m_l3out[(h * M4_OUT_W + w) * M3_OUT_C + ic] * m_l4w[oc * M3_OUT_C + ic];
                }
                int32_t sc = req(acc, MULT_Q30, 0);
                int32_t cl = sc > 127 ? 127 : (sc < 0 ? 0 : sc);
                m_l4out[(h * M4_OUT_W + w) * M4_OUT_C + oc] = sat8(cl);
            }
        }
    }
}

static void ref_m5_depthwise(void)
{
    int oh, ow, oc, ky, kx;
    for (oh = 0; oh < M5_OUT_H; oh++) {
        for (ow = 0; ow < M5_OUT_W; ow++) {
            for (oc = 0; oc < M5_OUT_C; oc++) {
                int32_t acc = m_l5b[oc];
                for (ky = 0; ky < 3; ky++) {
                    for (kx = 0; kx < 3; kx++) {
                        int ih = oh + ky, iw = ow + kx;
                        acc += (int32_t)m_l4out[(ih * M4_OUT_W + iw) * M4_OUT_C + oc] *
                               m_l5w[(ky * 3 + kx) * M5_OUT_C + oc];
                    }
                }
                int32_t sc = req(acc, MULT_Q30, 0);
                int32_t cl = sc > 127 ? 127 : (sc < 0 ? 0 : sc);
                m_l5out[(oh * M5_OUT_W + ow) * M5_OUT_C + oc] = sat8(cl);
            }
        }
    }
}

static void ref_m6_conv1x1(void)
{
    int h, w, oc, ic;
    for (h = 0; h < M6_OUT_H; h++) {
        for (w = 0; w < M6_OUT_W; w++) {
            for (oc = 0; oc < M6_OUT_C; oc++) {
                int32_t acc = m_l6b[oc];
                for (ic = 0; ic < M5_OUT_C; ic++) {
                    acc += (int32_t)m_l5out[(h * M6_OUT_W + w) * M5_OUT_C + ic] * m_l6w[oc * M5_OUT_C + ic];
                }
                int32_t sc = req(acc, MULT_Q30, 0);
                int32_t cl = sc > 127 ? 127 : (sc < 0 ? 0 : sc);
                m_l6out[(h * M6_OUT_W + w) * M6_OUT_C + oc] = sat8(cl);
            }
        }
    }
}

static void ref_m7_fc(void)
{
    int oc, i;
    for (oc = 0; oc < M7_OUT_C; oc++) {
        int32_t acc = m_l7b[oc];
        for (i = 0; i < M7_IN_DIM; i++) {
            acc += (int32_t)m_l6out[i] * m_l7w[oc * M7_IN_DIM + i];
        }
        int32_t sc = req(acc, MULT_Q30, 0);
        int32_t cl = sc > 127 ? 127 : (sc < -128 ? -128 : sc);
        m_out[oc] = sat8(cl);
    }
}

static void run_scalar_model_mv2(void)
{
    ref_m1_conv3x3();
    ref_m2_maxpool();
    ref_m3_depthwise();
    ref_m4_conv1x1();
    ref_m5_depthwise();
    ref_m6_conv1x1();
    ref_m7_fc();
}

/* ================= MODEL C (mv2real) scalar refs ================= */
/* SAME padding: pad_total = imax(0, (out-1)*stride + dilated - in);
 * pad_before = pad_total/2; taps outside the input are skipped (contribute 0). */
static void ref_c1_conv3x3(void)
{
    /* 16x16x3 -> 8x8x32, stride2, SAME, act(0,127) */
    int oh, ow, oc, kh, kw, ic;
    int pad_ht = imax(0, ((C1_OUT_H - 1) * 2 + 3 - C1_IN_H) / 2); /* =0 */
    int pad_wd = imax(0, ((C1_OUT_W - 1) * 2 + 3 - C1_IN_W) / 2); /* =0 */
    for (oh = 0; oh < C1_OUT_H; oh++) {
        for (ow = 0; ow < C1_OUT_W; ow++) {
            for (oc = 0; oc < C1_OUT_C; oc++) {
                int32_t acc = c_l1b[oc];
                for (kh = 0; kh < 3; kh++) {
                    int ih = oh * 2 + kh - pad_ht;
                    if (ih < 0 || ih >= C1_IN_H) continue;
                    for (kw = 0; kw < 3; kw++) {
                        int iw = ow * 2 + kw - pad_wd;
                        if (iw < 0 || iw >= C1_IN_W) continue;
                        for (ic = 0; ic < C1_IN_C; ic++) {
                            int w_idx = ((oc * 3 + kh) * 3 + kw) * C1_IN_C + ic;
                            acc += (int32_t)c_in[(ih * C1_IN_W + iw) * C1_IN_C + ic] * c_l1w[w_idx];
                        }
                    }
                }
                int32_t sc = req(acc, MULT_Q30, 0);
                int32_t cl = sc > 127 ? 127 : (sc < 0 ? 0 : sc);
                c_l1out[(oh * C1_OUT_W + ow) * C1_OUT_C + oc] = sat8(cl);
            }
        }
    }
}

static void ref_c2_depthwise(void)
{
    /* 8x8x32 -> 8x8x32, 3x3 stride1 SAME, dm=1, act(0,127) */
    int oh, ow, oc, ky, kx;
    int pad_ht = imax(0, ((C2_OUT_H - 1) * 1 + 3 - C1_OUT_H) / 2); /* =1 */
    int pad_wd = imax(0, ((C2_OUT_W - 1) * 1 + 3 - C1_OUT_W) / 2); /* =1 */
    for (oh = 0; oh < C2_OUT_H; oh++) {
        for (ow = 0; ow < C2_OUT_W; ow++) {
            for (oc = 0; oc < C2_OUT_C; oc++) {
                int32_t acc = c_l2b[oc];
                for (ky = 0; ky < 3; ky++) {
                    int ih = oh * 1 + ky - pad_ht;
                    if (ih < 0 || ih >= C1_OUT_H) continue;
                    for (kx = 0; kx < 3; kx++) {
                        int iw = ow * 1 + kx - pad_wd;
                        if (iw < 0 || iw >= C1_OUT_W) continue;
                        acc += (int32_t)c_l1out[(ih * C1_OUT_W + iw) * C2_OUT_C + oc] *
                               c_l2w[(ky * 3 + kx) * C2_OUT_C + oc];
                    }
                }
                int32_t sc = req(acc, MULT_Q30, 0);
                int32_t cl = sc > 127 ? 127 : (sc < 0 ? 0 : sc);
                c_l2out[(oh * C2_OUT_W + ow) * C2_OUT_C + oc] = sat8(cl);
            }
        }
    }
}

static void ref_c3_conv1x1(void)
{
    /* 8x8x32 -> 8x8x64 */
    int h, w, oc, ic;
    for (h = 0; h < C3_OUT_H; h++) {
        for (w = 0; w < C3_OUT_W; w++) {
            for (oc = 0; oc < C3_OUT_C; oc++) {
                int32_t acc = c_l3b[oc];
                for (ic = 0; ic < C2_OUT_C; ic++) {
                    acc += (int32_t)c_l2out[(h * C3_OUT_W + w) * C2_OUT_C + ic] * c_l3w[oc * C2_OUT_C + ic];
                }
                int32_t sc = req(acc, MULT_Q30, 0);
                int32_t cl = sc > 127 ? 127 : (sc < 0 ? 0 : sc);
                c_l3out[(h * C3_OUT_W + w) * C3_OUT_C + oc] = sat8(cl);
            }
        }
    }
}

static void ref_c4_depthwise(void)
{
    /* 8x8x64 -> 4x4x64, 3x3 stride2 SAME, dm=1, act(0,127) */
    int oh, ow, oc, ky, kx;
    int pad_ht = imax(0, ((C4_OUT_H - 1) * 2 + 3 - C3_OUT_H) / 2); /* =0 */
    int pad_wd = imax(0, ((C4_OUT_W - 1) * 2 + 3 - C3_OUT_W) / 2); /* =0 */
    for (oh = 0; oh < C4_OUT_H; oh++) {
        for (ow = 0; ow < C4_OUT_W; ow++) {
            for (oc = 0; oc < C4_OUT_C; oc++) {
                int32_t acc = c_l4b[oc];
                for (ky = 0; ky < 3; ky++) {
                    int ih = oh * 2 + ky - pad_ht;
                    if (ih < 0 || ih >= C3_OUT_H) continue;
                    for (kx = 0; kx < 3; kx++) {
                        int iw = ow * 2 + kx - pad_wd;
                        if (iw < 0 || iw >= C3_OUT_W) continue;
                        acc += (int32_t)c_l3out[(ih * C3_OUT_W + iw) * C4_OUT_C + oc] *
                               c_l4w[(ky * 3 + kx) * C4_OUT_C + oc];
                    }
                }
                int32_t sc = req(acc, MULT_Q30, 0);
                int32_t cl = sc > 127 ? 127 : (sc < 0 ? 0 : sc);
                c_l4out[(oh * C4_OUT_W + ow) * C4_OUT_C + oc] = sat8(cl);
            }
        }
    }
}

static void ref_c5_conv1x1(void)
{
    /* 4x4x64 -> 4x4x128 */
    int h, w, oc, ic;
    for (h = 0; h < C5_OUT_H; h++) {
        for (w = 0; w < C5_OUT_W; w++) {
            for (oc = 0; oc < C5_OUT_C; oc++) {
                int32_t acc = c_l5b[oc];
                for (ic = 0; ic < C4_OUT_C; ic++) {
                    acc += (int32_t)c_l4out[(h * C5_OUT_W + w) * C4_OUT_C + ic] * c_l5w[oc * C4_OUT_C + ic];
                }
                int32_t sc = req(acc, MULT_Q30, 0);
                int32_t cl = sc > 127 ? 127 : (sc < 0 ? 0 : sc);
                c_l5out[(h * C5_OUT_W + w) * C5_OUT_C + oc] = sat8(cl);
            }
        }
    }
}

static void ref_c6_fc(void)
{
    /* fc 2048 -> 16, act(-128,127) */
    int oc, i;
    for (oc = 0; oc < C6_OUT_C; oc++) {
        int32_t acc = c_l6b[oc];
        for (i = 0; i < C6_IN_DIM; i++) {
            acc += (int32_t)c_l5out[i] * c_l6w[oc * C6_IN_DIM + i];
        }
        int32_t sc = req(acc, MULT_Q30, 0);
        int32_t cl = sc > 127 ? 127 : (sc < -128 ? -128 : sc);
        c_out[oc] = sat8(cl);
    }
}

static void run_scalar_model_mv2real(void)
{
    ref_c1_conv3x3();
    ref_c2_depthwise();
    ref_c3_conv1x1();
    ref_c4_depthwise();
    ref_c5_conv1x1();
    ref_c6_fc();
}

/* ================= MODEL A esp_nn runner ================= */
static void run_espnn_model(void)
{
    data_dims_t in_dims, f_dims, out_dims;
    conv_params_t cparams;
    quant_data_t qdata;

    /* L1 conv3x3 */
    in_dims.width = L1_IN_W; in_dims.height = L1_IN_H; in_dims.channels = L1_IN_C; in_dims.extra = 1;
    f_dims.width = 3; f_dims.height = 3; f_dims.channels = L1_IN_C; f_dims.extra = L1_OUT_C;
    out_dims.width = L1_OUT_W; out_dims.height = L1_OUT_H; out_dims.channels = L1_OUT_C; out_dims.extra = 1;
    cparams.in_offset = 0; cparams.out_offset = 0;
    cparams.stride.width = 1; cparams.stride.height = 1;
    cparams.padding.width = 0; cparams.padding.height = 0;
    cparams.dilation.width = 1; cparams.dilation.height = 1;
    cparams.activation.min = 0; cparams.activation.max = 127;
    qdata.shift = s_shift; qdata.mult = s_mult;
    esp_nn_conv_s8(&in_dims, s_in, &f_dims, s_l1w, s_l1b, &out_dims, s_l1out, &cparams, &qdata);
    esp_nn_max_pool_s8(s_l1out, L1_OUT_W, L1_OUT_H, s_l2out, L2_OUT_W, L2_OUT_H,
                       2, 2, 2, 2, 0, 0, -128, 127, L2_OUT_C);

    /* L3 conv1x1 */
    in_dims.width = L3_OUT_W; in_dims.height = L3_OUT_H; in_dims.channels = L3_IN_C; in_dims.extra = 1;
    f_dims.width = 1; f_dims.height = 1; f_dims.channels = L3_IN_C; f_dims.extra = L3_OUT_C;
    out_dims.width = L3_OUT_W; out_dims.height = L3_OUT_H; out_dims.channels = L3_OUT_C; out_dims.extra = 1;
    cparams.in_offset = 0; cparams.out_offset = 0;
    cparams.stride.width = 1; cparams.stride.height = 1;
    cparams.padding.width = 0; cparams.padding.height = 0;
    cparams.dilation.width = 1; cparams.dilation.height = 1;
    cparams.activation.min = 0; cparams.activation.max = 127;
    qdata.shift = s_shift; qdata.mult = s_mult;
    esp_nn_conv_s8(&in_dims, s_l2out, &f_dims, s_l3w, s_l3b, &out_dims, s_l3out, &cparams, &qdata);

    /* L4 fc */
    esp_nn_fully_connected_per_ch_s8(s_l3out, 0, L4_IN_DIM, s_l4w, 0, s_l4b, s_out,
                                     L4_OUT_C, 0, s_shift, s_mult, -128, 127);
}

/* ================= MODEL B (mv2mini) esp_nn runner ================= */
static void run_espnn_model_mv2(void)
{
    data_dims_t in_dims, f_dims, out_dims;
    conv_params_t cparams;
    dw_conv_params_t dwparams;
    quant_data_t qdata;

    /* L1 conv3x3 16x16x3 -> 14x14x32 */
    in_dims.width = M1_IN_W; in_dims.height = M1_IN_H; in_dims.channels = M1_IN_C; in_dims.extra = 1;
    f_dims.width = 3; f_dims.height = 3; f_dims.channels = M1_IN_C; f_dims.extra = M1_OUT_C;
    out_dims.width = M1_OUT_W; out_dims.height = M1_OUT_H; out_dims.channels = M1_OUT_C; out_dims.extra = 1;
    cparams.in_offset = 0; cparams.out_offset = 0;
    cparams.stride.width = 1; cparams.stride.height = 1;
    cparams.padding.width = 0; cparams.padding.height = 0;
    cparams.dilation.width = 1; cparams.dilation.height = 1;
    cparams.activation.min = 0; cparams.activation.max = 127;
    qdata.shift = s_shift; qdata.mult = s_mult;
    esp_nn_conv_s8(&in_dims, m_in, &f_dims, m_l1w, m_l1b, &out_dims, m_l1out, &cparams, &qdata);

    /* L2 maxpool 14x14x32 -> 7x7x32 */
    esp_nn_max_pool_s8(m_l1out, M1_OUT_W, M1_OUT_H, m_l2out, M2_OUT_W, M2_OUT_H,
                       2, 2, 2, 2, 0, 0, -128, 127, M2_OUT_C);

    /* L3 depthwise 3x3 7x7x32 -> 5x5x32 dm=1 */
    in_dims.width = M2_OUT_W; in_dims.height = M2_OUT_H; in_dims.channels = M2_OUT_C; in_dims.extra = 1;
    f_dims.width = 3; f_dims.height = 3; f_dims.channels = M2_OUT_C; f_dims.extra = M2_OUT_C;
    out_dims.width = M3_OUT_W; out_dims.height = M3_OUT_H; out_dims.channels = M3_OUT_C; out_dims.extra = 1;
    dwparams.in_offset = 0; dwparams.out_offset = 0; dwparams.ch_mult = 1;
    dwparams.stride.width = 1; dwparams.stride.height = 1;
    dwparams.padding.width = 0; dwparams.padding.height = 0;
    dwparams.dilation.width = 1; dwparams.dilation.height = 1;
    dwparams.activation.min = 0; dwparams.activation.max = 127;
    qdata.shift = s_shift; qdata.mult = s_mult;
    esp_nn_depthwise_conv_s8(&in_dims, m_l2out, &f_dims, m_l3w, m_l3b, &out_dims, m_l3out, &dwparams, &qdata);

    /* L4 conv1x1 5x5x32 -> 5x5x64 */
    in_dims.width = M4_OUT_W; in_dims.height = M4_OUT_H; in_dims.channels = M3_OUT_C; in_dims.extra = 1;
    f_dims.width = 1; f_dims.height = 1; f_dims.channels = M3_OUT_C; f_dims.extra = M4_OUT_C;
    out_dims.width = M4_OUT_W; out_dims.height = M4_OUT_H; out_dims.channels = M4_OUT_C; out_dims.extra = 1;
    cparams.in_offset = 0; cparams.out_offset = 0;
    cparams.stride.width = 1; cparams.stride.height = 1;
    cparams.padding.width = 0; cparams.padding.height = 0;
    cparams.dilation.width = 1; cparams.dilation.height = 1;
    cparams.activation.min = 0; cparams.activation.max = 127;
    qdata.shift = s_shift; qdata.mult = s_mult;
    esp_nn_conv_s8(&in_dims, m_l3out, &f_dims, m_l4w, m_l4b, &out_dims, m_l4out, &cparams, &qdata);

    /* L5 depthwise 3x3 5x5x64 -> 3x3x64 dm=1 */
    in_dims.width = M4_OUT_W; in_dims.height = M4_OUT_H; in_dims.channels = M4_OUT_C; in_dims.extra = 1;
    f_dims.width = 3; f_dims.height = 3; f_dims.channels = M4_OUT_C; f_dims.extra = M4_OUT_C;
    out_dims.width = M5_OUT_W; out_dims.height = M5_OUT_H; out_dims.channels = M5_OUT_C; out_dims.extra = 1;
    dwparams.in_offset = 0; dwparams.out_offset = 0; dwparams.ch_mult = 1;
    dwparams.stride.width = 1; dwparams.stride.height = 1;
    dwparams.padding.width = 0; dwparams.padding.height = 0;
    dwparams.dilation.width = 1; dwparams.dilation.height = 1;
    dwparams.activation.min = 0; dwparams.activation.max = 127;
    qdata.shift = s_shift; qdata.mult = s_mult;
    esp_nn_depthwise_conv_s8(&in_dims, m_l4out, &f_dims, m_l5w, m_l5b, &out_dims, m_l5out, &dwparams, &qdata);

    /* L6 conv1x1 3x3x64 -> 3x3x128 */
    in_dims.width = M6_OUT_W; in_dims.height = M6_OUT_H; in_dims.channels = M5_OUT_C; in_dims.extra = 1;
    f_dims.width = 1; f_dims.height = 1; f_dims.channels = M5_OUT_C; f_dims.extra = M6_OUT_C;
    out_dims.width = M6_OUT_W; out_dims.height = M6_OUT_H; out_dims.channels = M6_OUT_C; out_dims.extra = 1;
    cparams.in_offset = 0; cparams.out_offset = 0;
    cparams.stride.width = 1; cparams.stride.height = 1;
    cparams.padding.width = 0; cparams.padding.height = 0;
    cparams.dilation.width = 1; cparams.dilation.height = 1;
    cparams.activation.min = 0; cparams.activation.max = 127;
    qdata.shift = s_shift; qdata.mult = s_mult;
    esp_nn_conv_s8(&in_dims, m_l5out, &f_dims, m_l6w, m_l6b, &out_dims, m_l6out, &cparams, &qdata);

    /* L7 fc 1152 -> 16 */
    esp_nn_fully_connected_per_ch_s8(m_l6out, 0, M7_IN_DIM, m_l7w, 0, m_l7b, m_out,
                                     M7_OUT_C, 0, s_shift, s_mult, -128, 127);
}

/* ================= MODEL C (mv2real) esp_nn runner ================= */
static void run_espnn_model_mv2real(void)
{
    data_dims_t in_dims, f_dims, out_dims;
    conv_params_t cparams;
    dw_conv_params_t dwparams;
    quant_data_t qdata;

    /* L1 conv3x3 stride2 SAME 16x16x3 -> 8x8x32 */
    in_dims.width = C1_IN_W; in_dims.height = C1_IN_H; in_dims.channels = C1_IN_C; in_dims.extra = 1;
    f_dims.width = 3; f_dims.height = 3; f_dims.channels = C1_IN_C; f_dims.extra = C1_OUT_C;
    out_dims.width = C1_OUT_W; out_dims.height = C1_OUT_H; out_dims.channels = C1_OUT_C; out_dims.extra = 1;
    cparams.in_offset = 0; cparams.out_offset = 0;
    cparams.stride.width = 2; cparams.stride.height = 2;
    cparams.padding.width = 0; cparams.padding.height = 0;
    cparams.dilation.width = 1; cparams.dilation.height = 1;
    cparams.activation.min = 0; cparams.activation.max = 127;
    qdata.shift = s_shift; qdata.mult = s_mult;
    esp_nn_conv_s8(&in_dims, c_in, &f_dims, c_l1w, c_l1b, &out_dims, c_l1out, &cparams, &qdata);

    /* L2 depthwise 3x3 stride1 SAME 8x8x32 -> 8x8x32 dm=1 */
    in_dims.width = C1_OUT_W; in_dims.height = C1_OUT_H; in_dims.channels = C1_OUT_C; in_dims.extra = 1;
    f_dims.width = 3; f_dims.height = 3; f_dims.channels = C1_OUT_C; f_dims.extra = C2_OUT_C;
    out_dims.width = C2_OUT_W; out_dims.height = C2_OUT_H; out_dims.channels = C2_OUT_C; out_dims.extra = 1;
    dwparams.in_offset = 0; dwparams.out_offset = 0; dwparams.ch_mult = 1;
    dwparams.stride.width = 1; dwparams.stride.height = 1;
    dwparams.padding.width = 1; dwparams.padding.height = 1;
    dwparams.dilation.width = 1; dwparams.dilation.height = 1;
    dwparams.activation.min = 0; dwparams.activation.max = 127;
    qdata.shift = s_shift; qdata.mult = s_mult;
    esp_nn_depthwise_conv_s8(&in_dims, c_l1out, &f_dims, c_l2w, c_l2b, &out_dims, c_l2out, &dwparams, &qdata);

    /* L3 conv1x1 8x8x32 -> 8x8x64 */
    in_dims.width = C2_OUT_W; in_dims.height = C2_OUT_H; in_dims.channels = C2_OUT_C; in_dims.extra = 1;
    f_dims.width = 1; f_dims.height = 1; f_dims.channels = C2_OUT_C; f_dims.extra = C3_OUT_C;
    out_dims.width = C3_OUT_W; out_dims.height = C3_OUT_H; out_dims.channels = C3_OUT_C; out_dims.extra = 1;
    cparams.in_offset = 0; cparams.out_offset = 0;
    cparams.stride.width = 1; cparams.stride.height = 1;
    cparams.padding.width = 0; cparams.padding.height = 0;
    cparams.dilation.width = 1; cparams.dilation.height = 1;
    cparams.activation.min = 0; cparams.activation.max = 127;
    qdata.shift = s_shift; qdata.mult = s_mult;
    esp_nn_conv_s8(&in_dims, c_l2out, &f_dims, c_l3w, c_l3b, &out_dims, c_l3out, &cparams, &qdata);

    /* L4 depthwise 3x3 stride2 SAME 8x8x64 -> 4x4x64 dm=1 */
    in_dims.width = C3_OUT_W; in_dims.height = C3_OUT_H; in_dims.channels = C3_OUT_C; in_dims.extra = 1;
    f_dims.width = 3; f_dims.height = 3; f_dims.channels = C3_OUT_C; f_dims.extra = C4_OUT_C;
    out_dims.width = C4_OUT_W; out_dims.height = C4_OUT_H; out_dims.channels = C4_OUT_C; out_dims.extra = 1;
    dwparams.in_offset = 0; dwparams.out_offset = 0; dwparams.ch_mult = 1;
    dwparams.stride.width = 2; dwparams.stride.height = 2;
    dwparams.padding.width = 0; dwparams.padding.height = 0;
    dwparams.dilation.width = 1; dwparams.dilation.height = 1;
    dwparams.activation.min = 0; dwparams.activation.max = 127;
    qdata.shift = s_shift; qdata.mult = s_mult;
    esp_nn_depthwise_conv_s8(&in_dims, c_l3out, &f_dims, c_l4w, c_l4b, &out_dims, c_l4out, &dwparams, &qdata);

    /* L5 conv1x1 4x4x64 -> 4x4x128 */
    in_dims.width = C4_OUT_W; in_dims.height = C4_OUT_H; in_dims.channels = C4_OUT_C; in_dims.extra = 1;
    f_dims.width = 1; f_dims.height = 1; f_dims.channels = C4_OUT_C; f_dims.extra = C5_OUT_C;
    out_dims.width = C5_OUT_W; out_dims.height = C5_OUT_H; out_dims.channels = C5_OUT_C; out_dims.extra = 1;
    cparams.in_offset = 0; cparams.out_offset = 0;
    cparams.stride.width = 1; cparams.stride.height = 1;
    cparams.padding.width = 0; cparams.padding.height = 0;
    cparams.dilation.width = 1; cparams.dilation.height = 1;
    cparams.activation.min = 0; cparams.activation.max = 127;
    qdata.shift = s_shift; qdata.mult = s_mult;
    esp_nn_conv_s8(&in_dims, c_l4out, &f_dims, c_l5w, c_l5b, &out_dims, c_l5out, &cparams, &qdata);

    /* L6 fc 2048 -> 16 */
    esp_nn_fully_connected_per_ch_s8(c_l5out, 0, C6_IN_DIM, c_l6w, 0, c_l6b, c_out,
                                     C6_OUT_C, 0, s_shift, s_mult, -128, 127);
}

/* ================= bench wrappers ================= */
static void bench_espnn(void) { run_espnn_model(); }
static void bench_scalar(void) { run_scalar_model(); }
static void bench_espnn_mv2(void) { run_espnn_model_mv2(); }
static void bench_scalar_mv2(void) { run_scalar_model_mv2(); }
static void bench_espnn_mv2real(void) { run_espnn_model_mv2real(); }
static void bench_scalar_mv2real(void) { run_scalar_model_mv2real(); }

static void dump_layer_checksums(const char *tag)
{
    printf("  %s: L1=0x%08x L2=0x%08x L3=0x%08x out=0x%08x\n",
           tag,
           (unsigned)fnv1a(s_l1out, L1_OUT_H * L1_OUT_W * L1_OUT_C),
           (unsigned)fnv1a(s_l2out, L2_OUT_H * L2_OUT_W * L2_OUT_C),
           (unsigned)fnv1a(s_l3out, L3_OUT_H * L3_OUT_W * L3_OUT_C),
           (unsigned)fnv1a(s_out, L4_OUT_C));
}

static void dump_layer_checksums_mv2(const char *tag)
{
    printf("  %s: L1=0x%08x L2=0x%08x L3=0x%08x L4=0x%08x L5=0x%08x L6=0x%08x out=0x%08x\n",
           tag,
           (unsigned)fnv1a(m_l1out, M1_OUT_H * M1_OUT_W * M1_OUT_C),
           (unsigned)fnv1a(m_l2out, M2_OUT_H * M2_OUT_W * M2_OUT_C),
           (unsigned)fnv1a(m_l3out, M3_OUT_H * M3_OUT_W * M3_OUT_C),
           (unsigned)fnv1a(m_l4out, M4_OUT_H * M4_OUT_W * M4_OUT_C),
           (unsigned)fnv1a(m_l5out, M5_OUT_H * M5_OUT_W * M5_OUT_C),
           (unsigned)fnv1a(m_l6out, M6_OUT_H * M6_OUT_W * M6_OUT_C),
           (unsigned)fnv1a(m_out, M7_OUT_C));
}

static void dump_layer_checksums_mv2real(const char *tag)
{
    printf("  %s: L1=0x%08x L2=0x%08x L3=0x%08x L4=0x%08x L5=0x%08x out=0x%08x\n",
           tag,
           (unsigned)fnv1a(c_l1out, C1_OUT_H * C1_OUT_W * C1_OUT_C),
           (unsigned)fnv1a(c_l2out, C2_OUT_H * C2_OUT_W * C2_OUT_C),
           (unsigned)fnv1a(c_l3out, C3_OUT_H * C3_OUT_W * C3_OUT_C),
           (unsigned)fnv1a(c_l4out, C4_OUT_H * C4_OUT_W * C4_OUT_C),
           (unsigned)fnv1a(c_l5out, C5_OUT_H * C5_OUT_W * C5_OUT_C),
           (unsigned)fnv1a(c_out, C6_OUT_C));
}

void app_main(void)
{
    int scratch_size;
    uint32_t espnn_chk = 0, scalar_chk = 0, espnn_mv2_chk = 0, scalar_mv2_chk = 0;

    printf("=== Hematite ESP-NN baseline (benchmarks/espnn-baseline) ===\n");
    printf("LABEL: real hardware (ESP32-S3 @ 240MHz), ESP-IDF v5.5.1, standard ESP-NN (espressif/esp-nn v1.2.5)\n");

    data_dims_t sdims, sf_dims, so_dims;
    conv_params_t scparams;
    sdims.width = L1_IN_W; sdims.height = L1_IN_H; sdims.channels = L1_IN_C; sdims.extra = 1;
    sf_dims.width = 3; sf_dims.height = 3; sf_dims.channels = L1_IN_C; sf_dims.extra = L1_OUT_C;
    so_dims.width = L1_OUT_W; so_dims.height = L1_OUT_H; so_dims.channels = L1_OUT_C; so_dims.extra = 1;
    scparams.padding.width = 0; scparams.padding.height = 0;
    scparams.stride.width = 1; scparams.stride.height = 1;

    scratch_size = esp_nn_get_conv_scratch_size(&sdims, &sf_dims, &so_dims, &scparams);
    ESP_LOGI(TAG, "esp_nn_get_conv_scratch_size=%d", scratch_size);
    if (scratch_size > (int)sizeof(s_scratch)) {
        printf("ERROR: scratch too small (%d > %d)\n", scratch_size, (int)sizeof(s_scratch));
        return;
    }
    esp_nn_set_conv_scratch_buf(s_scratch);
    esp_nn_set_depthwise_conv_scratch_buf(s_scratch);

    /* ============ MODEL A (4-layer) ============ */
    printf("MODEL A: conv3x3 32x32x16->30x30x16 | maxpool 2x2 | conv1x1 16->32 | fc 7200->16\n");
    fill_pattern();
    run_bench("model A esp_nn (conv3x3+maxpool+conv1x1+fc)", bench_espnn, s_out, L4_OUT_C, &espnn_chk);
    fill_pattern();
    run_scalar_model();
    dump_layer_checksums("model A scalar-ref layers");
    fill_pattern();
    run_espnn_model();
    dump_layer_checksums("model A esp_nn layers");
    run_bench("model A scalar-ref (TFLite semantics)", bench_scalar, s_out, L4_OUT_C, &scalar_chk);
    printf("=== model A esp_nn 0x%08x | scalar-ref 0x%08x | %s ===\n",
           (unsigned)espnn_chk, (unsigned)scalar_chk,
           espnn_chk == scalar_chk ? "MATCH" : "DIFFER");

    /* ============ MODEL B (mv2mini) ============ */
    printf("MODEL B (mv2mini): conv3x3 16x16x3->14x14x32 | maxpool | dw 32->32 | conv1x1 32->64 | dw 64->64 | conv1x1 64->128 | fc 1152->16\n");
    fill_pattern_mv2();
    run_bench("model B esp_nn (mv2mini: conv+dw+conv1x1+fc)", bench_espnn_mv2, m_out, M7_OUT_C, &espnn_mv2_chk);
    fill_pattern_mv2();
    run_scalar_model_mv2();
    dump_layer_checksums_mv2("model B scalar-ref layers");
    dump_bytes("model B scalar-ref L5[0..64]", m_l5out, 64);
    fill_pattern_mv2();
    run_espnn_model_mv2();
    dump_layer_checksums_mv2("model B esp_nn layers");
    dump_bytes("model B esp_nn L5[0..64]", m_l5out, 64);
    run_bench("model B scalar-ref (TFLite semantics)", bench_scalar_mv2, m_out, M7_OUT_C, &scalar_mv2_chk);
    printf("=== model B esp_nn 0x%08x | scalar-ref 0x%08x | %s ===\n",
           (unsigned)espnn_mv2_chk, (unsigned)scalar_mv2_chk,
           espnn_mv2_chk == scalar_mv2_chk ? "MATCH" : "DIFFER");

    /* ============ MODEL C (mv2real, MobileNetV2-style SAME/stride-2) ============ */
    {
        uint32_t espnn_c_chk = 0, scalar_c_chk = 0;
        printf("MODEL C (mv2real): conv3x3 s2 SAME 16x16x3->8x8x32 | dw 3x3 s1 SAME 32->32 | conv1x1 32->64 | dw 3x3 s2 SAME 64->64 | conv1x1 64->128 | fc 2048->16\n");
        fill_pattern_mv2real();
        run_bench("model C esp_nn (mv2real: conv+dw+conv1x1+dw+conv1x1+fc)", bench_espnn_mv2real, c_out, C6_OUT_C, &espnn_c_chk);
        fill_pattern_mv2real();
        run_scalar_model_mv2real();
        dump_layer_checksums_mv2real("model C scalar-ref layers");
        fill_pattern_mv2real();
        run_espnn_model_mv2real();
        dump_layer_checksums_mv2real("model C esp_nn layers");
        run_bench("model C scalar-ref (TFLite semantics)", bench_scalar_mv2real, c_out, C6_OUT_C, &scalar_c_chk);
        printf("=== model C esp_nn 0x%08x | scalar-ref 0x%08x | %s ===\n",
               (unsigned)espnn_c_chk, (unsigned)scalar_c_chk,
               espnn_c_chk == scalar_c_chk ? "MATCH" : "DIFFER");
    }

    printf("=== benchmark complete - ESP-NN halt ===\n");

    vTaskDelay(pdMS_TO_TICKS(1000));
}
