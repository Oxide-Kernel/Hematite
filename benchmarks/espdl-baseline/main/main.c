/* espdl-baseline: benchmark the vendored ESP-DL dl_tie728 TIE728 SIMD kernels
 * on real ESP32-S3 hardware via ESP-IDF, so output + cycles can be matched
 * against the Rust hematite s3 crate (bench10) for EVERY SIMD-capable op:
 * conv1x1, conv3x3, fc, relu, max/avg pool, add/mul/sub.
 *
 * Runs the SAME deterministic fill_pattern as the Rust firmware and
 * qemu-baseline:
 *   input[i]  = (i*7+3)&0xFF
 *   weights[i]= (i*13+11)&0xFF
 *   bias[i]   = i*17-8
 *   output[i] = 0
 * and prints an FNV-1a out_checksum over each output so output can be
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

/* ---- Args structs (layouts must match Rust hematite-s3) ---------------- */

/* Tie728ConvArgs (conv1x1 + fc via dl_tie728_s8_conv2d_11cn):
 *   +48 filter, +64 mac_shift, +68 bias, +76 activation_alpha,
 *   +84 activation_shift, +96 output_channel_div_8, +100 c_div_x_1,
 *   +104 filter_channel_factor */
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

/* Tie728Conv33Args (conv3x3 via dl_tie728_s8_conv2d_33cn):
 *   +48 filter, +64 mac_shift, +68 bias, +76 activation_alpha,
 *   +80 activation_alpha_ptr, +84 activation_shift, +96 ocd8, +100 cdiv,
 *   +104 filter_channel_factor, +108 dilation_x_offset, +112 dilation_y_offset */
typedef struct {
    uint8_t  _pad0[48];
    int8_t  *filter;
    uint8_t  _pad1[12];
    int32_t  mac_shift;
    int32_t *bias;
    uint8_t  _pad2[4];
    int32_t  activation_alpha;
    int32_t *activation_alpha_ptr;
    int32_t  activation_shift;
    uint8_t  _pad3[8];
    int32_t  output_channel_div_8;
    int32_t  c_div_x_1;
    int16_t *filter_channel_factor;
    int32_t  dilation_x_offset;
    int32_t  dilation_y_offset;
} Tie728Conv33Args;

/* Tie728ReluArgs (dl_tie728_s8_relu_11c): +76 activation_alpha,
 * +84 activation_shift, +88 c_rs1_1 = (c-16)/32, +92 c_rs2_1 = ((c-16)%32)/16 */
typedef struct {
    uint8_t  _pad0[76];
    int32_t  activation_alpha;
    uint8_t  _pad1[4];
    int32_t  activation_shift;
    int32_t  c_rs1_1;
    int32_t  c_rs2_1;
} Tie728ReluArgs;

/* Tie728MaxPoolArgs (dl_tie728_s8_max_pool2d_22c1):
 *   +4 input_channel, +16 input_y_offset, +20 input_x_offset,
 *   +48 filter_height, +52 filter_width, +60 c_remainder, +104 c_div_x_1 */
typedef struct {
    uint8_t  _pad0[4];
    int32_t  input_channel;
    uint8_t  _pad1[8];
    int32_t  input_y_offset;
    int32_t  input_x_offset;
    uint8_t  _pad2[24];
    int32_t  filter_height;
    int32_t  filter_width;
    uint8_t  _pad3[4];
    int32_t  c_remainder;
    uint8_t  _pad4[40];
    int32_t  c_div_x_1;
} Tie728MaxPoolArgs;

/* Tie728AvgPoolArgs (dl_tie728_s8_avg_pool2d_22c1): like max +56 shift,
 * +64 avg_pool_area_inv (i8[16]), +104 c_div_x_1 */
typedef struct {
    uint8_t  _pad0[4];
    int32_t  input_channel;
    uint8_t  _pad1[8];
    int32_t  input_y_offset;
    int32_t  input_x_offset;
    uint8_t  _pad2[24];
    int32_t  filter_height;
    int32_t  filter_width;
    int32_t  shift;
    uint8_t  _pad3[4];
    int8_t   avg_pool_area_inv[16];
    uint8_t  _pad4[24];
    int32_t  c_div_x_1;
} Tie728AvgPoolArgs;

/* AddSubAlignedArgs (dl_tie728_s8_add/sub_w1_16_w2_16): +44 length */
typedef struct {
    uint8_t  _pad0[44];
    uint32_t length;
} AddSubAlignedArgs;

/* MulAlignedArgs (dl_tie728_s8_mul_w1_16_w2_16): +64 c_div_x_1, +80 mul_shift */
typedef struct {
    uint8_t  _pad0[64];
    int32_t  c_div_x_1;
    uint8_t  _pad1[12];
    int32_t  mul_shift;
} MulAlignedArgs;

/* ---- asm entry points ---- */
extern void dl_tie728_s8_conv2d_11cn(int8_t *output, const int8_t *input, Tie728ConvArgs *args);
extern void dl_tie728_s8_conv2d_33cn(int8_t *output, const int8_t *input, Tie728Conv33Args *args);
extern void dl_tie728_s8_relu_11c(int8_t *output, const int8_t *input, Tie728ReluArgs *args);
extern void dl_tie728_s8_max_pool2d_22c1(int8_t *output, const int8_t *input, Tie728MaxPoolArgs *args);
extern void dl_tie728_s8_avg_pool2d_22c1(int8_t *output, const int8_t *input, Tie728AvgPoolArgs *args);
extern void dl_tie728_s8_add_w1_16_w2_16(int8_t *output, const int8_t *in1, const int8_t *in2, AddSubAlignedArgs *args);
extern void dl_tie728_s8_sub_w1_16_w2_16(int8_t *output, const int8_t *in1, const int8_t *in2, AddSubAlignedArgs *args);
extern void dl_tie728_s8_mul_w1_16_w2_16(int8_t *output, const int8_t *in1, const int8_t *in2, MulAlignedArgs *args);

extern void probe_accx(const int16_t *filter, const int16_t *input,
                       int32_t *accx_out);
extern void probe_accx_load(const int32_t *pattern, int32_t *accx_out);
extern void probe_accx_mac_sweep(const int16_t *filter, const int16_t *input,
                                 int32_t *out64);
extern void probe_s8accx(const int8_t *filter, const int8_t *input,
                         int32_t *out3);
extern void probe_qacc_layout(const int8_t *filter, const int8_t *input,
                              int32_t *out40, int32_t macs, int32_t reverse);
extern void probe_qacc_s16(const int8_t *filter, const int8_t *input,
                            int32_t *out_low, int32_t *out_high, int32_t macs);
extern void probe_qacc_perchan(const int8_t *filter, const int8_t *input,
                               int32_t *out16, int32_t *out_low,
                               int32_t *out_high, int32_t macs);
extern void probe_qacc_esplike(const int8_t *filter, const int8_t *input,
                               int32_t *out32, int32_t macs);
extern void s8_accx_conv1x1(const int8_t *input, const int8_t *filter,
                            int32_t *acc_out, int32_t in_c, int32_t out_c);
extern void s8_accx_conv1x1_orig(const int8_t *input, const int8_t *filter,
                                 int32_t *acc_out, int32_t in_c, int32_t out_c);
extern void s8_accx_conv3x3(const int8_t *input, const int8_t *filter,
                            int32_t *acc_out, int32_t in_c, int32_t out_c,
                            int32_t row_delta);
extern void s8_accx_depthwise(const int8_t *input, const int8_t *filter,
                              int32_t *acc_out, int32_t in_c, int32_t out_c,
                              int32_t row_delta);

/* ---- anytap depthwise (s8_accx_depthwise_anytap.S) ---- */
typedef struct {
    const int8_t *input;
    const int8_t *filter;
    int32_t      *acc_out;
    uint32_t      in_c;
    uint32_t      out_c;
    uint32_t      row_delta;
    uint32_t      taps;
    uint32_t      filter_w;
    uint32_t      col_start;
} AnyTapCtx;
extern void s8_accx_depthwise_anytap(AnyTapCtx *ctx);

/* ---- bc1 broadcast depthwise (s8_accx_depthwise_anytap_bc1.S) ---- */
typedef struct {
    const int8_t *input;
    const int8_t *filter;
    int32_t      *acc_out;
    uint32_t      in_c;
    uint32_t      out_c;
    uint32_t      row_delta;
    uint32_t      taps;
    uint32_t      filter_w;
    uint32_t      col_start;
} Bc1Ctx;
extern void s8_accx_depthwise_anytap_bc1(Bc1Ctx *ctx);

/* ---- VLDBC.8 broadcast semantics probe ---- */
extern void probe_vldbc(const int8_t *src, int8_t *out, int32_t offset);
static int8_t __attribute__((aligned(16))) s_vldbc_in[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
static int8_t __attribute__((aligned(16))) s_vldbc_out[16];
static int8_t __attribute__((aligned(16))) s_vldbc_neg[16] = {0x80, 0xFF, 0x7F, 0x00, 0x80, 0xFF, 0x7F, 0x00, 0x80, 0xFF, 0x7F, 0x00, 0x80, 0xFF, 0x7F, 0x00};
static void probe_vldbc_run(void) {
    for (int off = 0; off < 4; off++) {
        probe_vldbc(s_vldbc_in, s_vldbc_out, off);
        printf("[probe_vldbc] pos off=%d (addr%%16=%d) out=", off, (int)((uintptr_t)(s_vldbc_in + off) & 15));
        for (int i = 0; i < 16; i++) printf("%d ", s_vldbc_out[i]);
        printf("\n");
    }
    for (int off = 0; off < 4; off++) {
        probe_vldbc(s_vldbc_neg, s_vldbc_out, off);
        printf("[probe_vldbc] NEG off=%d (byte=0x%02x) out=", off, (unsigned)(uint8_t)s_vldbc_neg[off]);
        for (int i = 0; i < 16; i++) printf("%d ", s_vldbc_out[i]);
        printf("\n");
    }
}


/* ---- conv1x1 64x1x1x64 ---- */
#define IN_C   64
#define OUT_C  64
#define WEIGHTS_LEN (IN_C * OUT_C)

/* ---- conv3x3 32x32 64x3x3x64 -> 30x30x64 ---- */
#define C3_IN_H   32
#define C3_IN_W   32
#define C3_IN_C   64
#define C3_OUT_H  30
#define C3_OUT_W  30
#define C3_OUT_C  64
#define C3_IN_LEN    (C3_IN_H * C3_IN_W * C3_IN_C)
#define C3_W_LEN     (C3_OUT_C * 3 * 3 * C3_IN_C)
#define C3_OUT_LEN   (C3_OUT_H * C3_OUT_W * C3_OUT_C)

/* ---- fc 256 -> 64 ---- */
#define FC_IN_DIM  256
#define FC_OUT_DIM 64
#define FC_W_LEN   (FC_IN_DIM * FC_OUT_DIM)

/* ---- TEMP-DEBUG: anomaly-shaped fc rows (pure kernel isolation).
 * Groups = in_c/16 is the kernel-loop driver; out_c is just the oc loop
 * count, so out=16 keeps DRAM small while exercising in_c=640/128. */
#define A640_IN_DIM 640
#define A640_OUT_DIM 16
#define A640_W_LEN  (A640_IN_DIM * A640_OUT_DIM)
#define A128_IN_DIM 128
#define A128_OUT_DIM 16
#define A128_W_LEN  (A128_IN_DIM * A128_OUT_DIM)

/* ---- pools 32x32x16 -> 16x16x16 ---- */
#define P_IN_H  32
#define P_IN_W  32
#define P_C     16
#define P_IN_LEN   (P_IN_H * P_IN_W * P_C)
#define P_OUT_LEN  (16 * 16 * P_C)

/* ---- elementwise / relu: 256 elements ---- */
#define N256  256

static int8_t  s_input[IN_C] __attribute__((aligned(16)));
static int8_t  s_weights[WEIGHTS_LEN] __attribute__((aligned(16)));
static int8_t  s_tw[WEIGHTS_LEN] __attribute__((aligned(16)));  /* transformed [g][ic][lane] */
static int32_t s_bias[OUT_C] __attribute__((aligned(16)));
static int8_t  s_output[OUT_C] __attribute__((aligned(16)));

static int8_t  s_c3_in[C3_IN_LEN] __attribute__((aligned(16)));
static int8_t  s_c3_w[C3_W_LEN] __attribute__((aligned(16)));
static int32_t s_c3_b[C3_OUT_C] __attribute__((aligned(16)));
static int8_t  s_c3_out[C3_OUT_LEN] __attribute__((aligned(16)));

static int8_t  s_fc_in[FC_IN_DIM] __attribute__((aligned(16)));
static int8_t  s_fc_w[FC_W_LEN] __attribute__((aligned(16)));
static int32_t s_fc_b[FC_OUT_DIM] __attribute__((aligned(16)));
static int8_t  s_fc_out[FC_OUT_DIM] __attribute__((aligned(16)));

/* TEMP-DEBUG: anomaly-shaped fc buffers (pure kernel isolation). */
static int8_t  s_a640_in[A640_IN_DIM] __attribute__((aligned(16)));
static int8_t  s_a640_w[A640_W_LEN] __attribute__((aligned(16)));
static int32_t s_a640_accx[A640_OUT_DIM] __attribute__((aligned(16)));
static int8_t  s_a128_in[A128_IN_DIM] __attribute__((aligned(16)));
static int8_t  s_a128_w[A128_W_LEN] __attribute__((aligned(16)));
static int32_t s_a128_accx[A128_OUT_DIM] __attribute__((aligned(16)));
/* TEMP-DEBUG (flash-latency hypothesis): same weight shape but `static const`
 * so the linker places it in DROM (flash, 0x3c...) instead of DRAM (SRAM). */
static const int8_t s_a640_w_const[A640_W_LEN] __attribute__((aligned(16))) = {
    [0 ... (A640_W_LEN - 1)] = 0x0b
};
static const int8_t s_a128_w_const[A128_W_LEN] __attribute__((aligned(16))) = {
    [0 ... (A128_W_LEN - 1)] = 0x0b
};
/* TEMP-DEBUG: FULL 640x128 DROM weight stream (80KB, exceeds DCache) — the
 * exact anomaly op0/op9 weight footprint. Const => DROM, no DRAM cost. */
#define A640B_IN_DIM  640
#define A640B_OUT_DIM 128
#define A640B_W_LEN   (A640B_IN_DIM * A640B_OUT_DIM)
static const int8_t s_a640b_w_const[A640B_W_LEN] __attribute__((aligned(16))) = {
    [0 ... (A640B_W_LEN - 1)] = 0x0b
};
static int8_t  s_a640b_in[A640B_IN_DIM] __attribute__((aligned(16)));
static int32_t s_a640b_accx[A640B_OUT_DIM] __attribute__((aligned(16)));

static int8_t  s_p_in[P_IN_LEN] __attribute__((aligned(16)));
static int8_t  s_p_out[P_OUT_LEN] __attribute__((aligned(16)));

static int8_t  s_r_in[N256] __attribute__((aligned(16)));
static int8_t  s_r_out[N256] __attribute__((aligned(16)));

static int8_t  s_e1[N256] __attribute__((aligned(16)));
static int8_t  s_e2[N256] __attribute__((aligned(16)));
static int8_t  s_e_out[N256] __attribute__((aligned(16)));

static int32_t s_mult[OUT_C] __attribute__((aligned(16)));   /* = 1<<30 */
static int32_t s_shift[OUT_C] __attribute__((aligned(16)));  /* = 0 */

/* Depthwise 3x3 stride1 pad0 dm1 test (mirrors mv2mini L3/L5 shape family):
 * input 7x7x32 -> output 5x5x32, filter 3x3x32 HWCN dm=1. */
#define DW_IN_H 7
#define DW_IN_W 7
#define DW_C 32
#define DW_OUT_H 5
#define DW_OUT_W 5
#define DW_IN_LEN (DW_IN_H * DW_IN_W * DW_C)
#define DW_W_LEN (9 * DW_C)
#define DW_OUT_LEN (DW_OUT_H * DW_OUT_W * DW_C)
static int8_t  s_dw_in[DW_IN_LEN] __attribute__((aligned(16)));
static int8_t  s_dw_w[DW_W_LEN] __attribute__((aligned(16)));
static int32_t s_dw_b[DW_C] __attribute__((aligned(16)));
static int32_t s_dw_accx[DW_C] __attribute__((aligned(16)));
static int8_t  s_dw_out[DW_OUT_LEN] __attribute__((aligned(16)));
static int8_t  s_dw_ref[DW_OUT_LEN] __attribute__((aligned(16)));

static inline uint32_t read_ccount(void) {
    uint32_t c;
    asm volatile("rsr.ccount %0" : "=r"(c));
    return c;
}

static void fill_pattern_conv1x1(void) {
    for (int i = 0; i < IN_C; i++)        s_input[i]  = (int8_t)((i * 7 + 3) & 0xFF);
    for (int i = 0; i < WEIGHTS_LEN; i++) s_weights[i]= (int8_t)((i * 13 + 11) & 0xFF);
    for (int i = 0; i < OUT_C; i++)       s_bias[i]   = (int32_t)(i * 17 - 8);
    for (int i = 0; i < OUT_C; i++)       s_output[i] = 0;
}

/* TIE728 11cn filter layout: [g][ic][lane], dst[g*(in_c*16)+ic*16+lane]
 * = src[(g*16+lane)*in_c + ic]  (src is [oc][ic]). */
static void transform_weights_11cn(void) {
    for (int g = 0; g < OUT_C / 16; g++) {
        for (int ic = 0; ic < IN_C; ic++) {
            for (int lane = 0; lane < 16; lane++) {
                s_tw[g * (IN_C * 16) + ic * 16 + lane] =
                    s_weights[(g * 16 + lane) * IN_C + ic];
            }
        }
    }
}

static void fill_pattern_conv1x1_tw(void) {
    fill_pattern_conv1x1();
    transform_weights_11cn();
}

static void fill_pattern_conv3x3(void) {
    for (int i = 0; i < C3_IN_LEN; i++)  s_c3_in[i]  = (int8_t)((i * 7 + 3) & 0xFF);
    for (int i = 0; i < C3_W_LEN; i++)   s_c3_w[i]   = (int8_t)((i * 13 + 11) & 0xFF);
    for (int i = 0; i < C3_OUT_C; i++)   s_c3_b[i]   = (int32_t)(i * 17 - 8);
    for (int i = 0; i < C3_OUT_LEN; i++) s_c3_out[i] = 0;
}

static void fill_pattern_fc(void) {
    for (int i = 0; i < FC_IN_DIM; i++)  s_fc_in[i]  = (int8_t)((i * 7 + 3) & 0xFF);
    for (int i = 0; i < FC_W_LEN; i++)   s_fc_w[i]   = (int8_t)((i * 13 + 11) & 0xFF);
    for (int i = 0; i < FC_OUT_DIM; i++) s_fc_b[i]   = (int32_t)(i * 17 - 8);

    /* TEMP-DEBUG: fill the anomaly-shaped fc buffers too. */
    for (int i = 0; i < A640_IN_DIM; i++)  s_a640_in[i]  = (int8_t)((i * 7 + 3) & 0xFF);
    for (int i = 0; i < A640_W_LEN; i++)   s_a640_w[i]   = (int8_t)((i * 13 + 11) & 0xFF);
    for (int i = 0; i < A128_IN_DIM; i++)  s_a128_in[i]  = (int8_t)((i * 7 + 3) & 0xFF);
    for (int i = 0; i < A128_W_LEN; i++)   s_a128_w[i]   = (int8_t)((i * 13 + 11) & 0xFF);
    for (int i = 0; i < A640B_IN_DIM; i++) s_a640b_in[i] = (int8_t)((i * 7 + 3) & 0xFF);
    for (int i = 0; i < FC_OUT_DIM; i++) s_fc_out[i] = 0;
}

static void fill_pattern_pool(void) {
    for (int i = 0; i < P_IN_LEN; i++)  s_p_in[i]  = (int8_t)((i * 7 + 3) & 0xFF);
    for (int i = 0; i < P_OUT_LEN; i++) s_p_out[i] = 0;
}

static void fill_pattern_relu(void) {
    for (int i = 0; i < N256; i++) s_r_in[i]  = (int8_t)((i * 7 + 3) & 0xFF);
    for (int i = 0; i < N256; i++) s_r_out[i] = 0;
}

static void fill_pattern_elem(void) {
    for (int i = 0; i < N256; i++) {
        s_e1[i]    = (int8_t)((i * 7 + 3) & 0xFF);    /* input slice */
        s_e2[i]    = (int8_t)((i * 13 + 11) & 0xFF);  /* weights slice */
        s_e_out[i] = 0;
    }
}

static void init_quant_consts(void) {
    for (int i = 0; i < OUT_C; i++) {
        s_mult[i]  = (int32_t)(1 << 30);
        s_shift[i] = 0;
    }
}

static uint32_t fnv1a(const int8_t *data, size_t len) {
    /* Mirrors the Rust firmware's fnv1a: h ^= b as u32 where b is i8, so
     * negative bytes SIGN-EXTEND (0x80 -> 0xffffff80). */
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < len; i++) {
        h ^= (uint32_t)(int8_t)data[i];
        h *= 16777619u;
    }
    return h;
}

static void dump_bytes(const char *tag, const int8_t *p, size_t len) {
    printf("  %s =", tag);
    for (size_t i = 0; i < len; i++) printf(" %02x", (uint8_t)p[i]);
    printf("\n");
}

/* ---- ACCX probe (probe_accx.S) ----
 * EE.VMULAS.S16.ACCX F, I: element-wise MAC. ACCX is 256-bit. SRS.ACCX extracts
 * with pos {0,1} (selects 128-bit half) + GPR shift. Sweep to map lane layout.
 * Ramp filter=[1..8], input=[1]*8: element-wise lanes after ONE MAC = 1..8.
 * 127s: lanes = 127*127 = 16129 each. */
static void probe_accx_run(void) {
    static int16_t in[8] __attribute__((aligned(16)));
    static int16_t filt[8] __attribute__((aligned(16)));
    static int32_t accx_out[8] __attribute__((aligned(16)));
    static int32_t sweep[64] __attribute__((aligned(16)));
    static const int16_t ramp[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    static int32_t pattern[8] = {0x00000000, 0x00000001, 0x0000FFFF, 0xFFFF0000,
                                 0x7FFFFFFF, 0x80000000, 0x12345678, 0x89ABCDEF};

    printf("-- ACCX probe (VMULAS.S16.ACCX element-wise, ONE MAC) --\n");

    printf("  LOAD probe: LD.ACCX.IP 8xint32 pattern -> SRS.ACCX pos0/1 shift0..3 =");
    probe_accx_load(pattern, accx_out);
    for (int i = 0; i < 8; i++) printf(" %08x", (unsigned)accx_out[i]);
    printf("\n");

    for (int i = 0; i < 8; i++) { in[i] = 1; filt[i] = ramp[i]; }
    probe_accx_mac_sweep(filt, in, sweep);
    printf("  SWEEP ramp filt=1..8 in=1 (ONE MAC; lanes should be 1..8):\n");
    printf("    pos0 shifts 0..31:");
    for (int i = 0; i < 32; i++) printf(" %d", (int)sweep[i]);
    printf("\n    pos1 shifts 0..31:");
    for (int i = 32; i < 64; i++) printf(" %d", (int)sweep[i]);
    printf("\n");

    for (int i = 0; i < 8; i++) { in[i] = 127; filt[i] = 127; }
    probe_accx_mac_sweep(filt, in, sweep);
    printf("  SWEEP 127s (ONE MAC; lanes should be 16129):\n");
    printf("    pos0 shifts 0..31:");
    for (int i = 0; i < 32; i++) printf(" %d", (int)sweep[i]);
    printf("\n    pos1 shifts 0..31:");
    for (int i = 32; i < 64; i++) printf(" %d", (int)sweep[i]);
    printf("\n");
}

/* ---- S8 ACCX probe (probe_s8accx.S) ----
 * EE.VMULAS.S8.ACCX F, I: 16-wide element MAC. Two identical MACs accumulated.
 * If S8 products are 16-bit and lanes 32-bit: ramp in=[1]*16 filt=1..16
 *   -> sum per MAC = 136; two MACs = 272.
 * If products 8-bit saturating: 127*127 -> 127 per element -> 16*127*2 = 4064.
 * SRS.ACCX shift0 = exact accumulator. */
static void probe_s8accx_run(void) {
    static int8_t in[16] __attribute__((aligned(16)));
    static int8_t filt[16] __attribute__((aligned(16)));
    static int32_t out3[3] __attribute__((aligned(16)));

    printf("-- S8 ACCX probe (VMULAS.S8.ACCX 16-wide, TWO MACs) --\n");

    for (int i = 0; i < 16; i++) { in[i] = 1; filt[i] = (int8_t)(i + 1); }
    probe_s8accx(filt, in, out3);
    printf("  ramp filt=1..16 in=1: TWO MACs = %d %d %d\n",
           (int)out3[0], (int)out3[1], (int)out3[2]);
    printf("  (shift0: expect 272 = 2*136 if 16-bit products / 32-bit lanes)\n");

    for (int i = 0; i < 16; i++) { in[i] = 127; filt[i] = 127; }
    probe_s8accx(filt, in, out3);
    printf("  127s: TWO MACs = %d %d %d\n",
           (int)out3[0], (int)out3[1], (int)out3[2]);
    printf("  (shift0: 2*16*16129 = 516128 if 16-bit products; 4064 if 8-bit saturating)\n");
}

/* ---- QACC layout probe (probe_qacc.S) ----
 * VMULAS.S8.QACC = per-lane element-wise accumulate (depthwise primitive).
 * The 40-byte store (QACC_L.L.128/H.32 + QACC_H.L.128/H.32) is the read-back.
 * Probe 1: filter=1, input=1..16, 1 MAC -> lane i = i+1 (distinct small),
 *          dump all 40 bytes to see WHERE each lane lives.
 * Probe 2: same, 9 MACs -> lane i = 9*(i+1) (accumulation check).
 * Probe 3: all-127, 1 MAC -> 16129 if lanes >=16-bit wide, 127 if 8-bit sat.
 * Probe 4: single-lane sweep -> for each lane k set input[k]=1 only, 1 MAC,
 *          find which bytes hold the '1' (exact lane->byte mapping). */
static void probe_qacc_layout_run(void) {
    static int8_t in[16] __attribute__((aligned(16)));
    static int8_t filt[16] __attribute__((aligned(16)));
    static int32_t out40[10] __attribute__((aligned(16)));
    static int32_t pcout[16] __attribute__((aligned(16))); /* 32B zip + low + high */

    printf("-- QACC layout probe (VMULAS.S8.QACC per-lane, 40-byte store) --\n");

    for (int i = 0; i < 16; i++) { in[i] = (int8_t)(i + 1); filt[i] = 1; }
    probe_qacc_layout(filt, in, out40, 1, 0);
    printf("  filter=1 in=1..16, 1 MAC (lane i should hold i+1=1..16):\n  ");
    for (int b = 0; b < 40; b++) printf("%02x ", (unsigned)((unsigned char *)out40)[b]);
    printf("\n");

    probe_qacc_layout(filt, in, out40, 9, 0);
    printf("  filter=1 in=1..16, 9 MACs (lane i = 9*(i+1)=9..144):\n  ");
    for (int b = 0; b < 40; b++) printf("%02x ", (unsigned)((unsigned char *)out40)[b]);
    printf("\n");

    for (int i = 0; i < 16; i++) { in[i] = 127; filt[i] = 127; }
    probe_qacc_layout(filt, in, out40, 1, 0);
    printf("  127*127 1 MAC (expect 16129=0x3f01 in wide lanes, 0x7f if 8-bit sat):\n  ");
    for (int b = 0; b < 40; b++) printf("%02x ", (unsigned)((unsigned char *)out40)[b]);
    printf("\n");

    probe_qacc_layout(filt, in, out40, 9, 0);
    printf("  127*127 9 MACs (expect 9*16129=145161=0x23709 if wide):\n  ");
    for (int b = 0; b < 40; b++) printf("%02x ", (unsigned)((unsigned char *)out40)[b]);
    printf("\n");

    printf("  single-lane sweep (lane k: input[k]=1 only, 1 MAC; bytes holding 01):\n");
    for (int k = 0; k < 16; k++) {
        for (int i = 0; i < 16; i++) { in[i] = (i == k) ? 1 : 0; filt[i] = 1; }
        probe_qacc_layout(filt, in, out40, 1, 0);
        printf("  lane %2d: ", k);
        for (int b = 0; b < 40; b++) printf("%02x", (unsigned)((unsigned char *)out40)[b]);
        printf("\n");
    }
    printf("  REVERSED single-lane sweep (q0=input, q1=filter):\n");
    for (int k = 0; k < 16; k++) {
        for (int i = 0; i < 16; i++) { in[i] = (i == k) ? 1 : 0; filt[i] = 1; }
        probe_qacc_layout(filt, in, out40, 1, 1);
        printf("  lane %2d: ", k);
        for (int b = 0; b < 40; b++) printf("%02x", (unsigned)((unsigned char *)out40)[b]);
        printf("\n");
    }

    printf("  bit-sweep (all lanes=1, macs=1<<j so lane value = 2^j; traces bit j):\n");
    for (int i = 0; i < 16; i++) { in[i] = 1; filt[i] = 1; }
    for (int j = 0; j < 16; j++) {
        probe_qacc_layout(filt, in, out40, 1 << j, 0);
        printf("  bit%2d: ", j);
        for (int b = 0; b < 40; b++) printf("%02x", (unsigned)((unsigned char *)out40)[b]);
        printf("\n");
    }

    printf("  lane-sweep macs=3 (in[k]=1 only, 3 MACs -> lane=3; odd-lane high bits):\n");
    for (int k = 0; k < 16; k++) {
        for (int i = 0; i < 16; i++) { in[i] = (i == k) ? 1 : 0; filt[i] = 1; }
        probe_qacc_layout(filt, in, out40, 3, 0);
        printf("  lane %2d: ", k);
        for (int b = 0; b < 40; b++) printf("%02x", (unsigned)((unsigned char *)out40)[b]);
        printf("\n");
    }

    printf("  S16 extraction sweep (SRCMB.S16 shift4=low out_low, shift20=even out_high):\n");
    for (int k = 0; k < 16; k++) {
        for (int i = 0; i < 16; i++) { in[i] = (i == k) ? 1 : 0; filt[i] = 1; }
        probe_qacc_s16(filt, in, out40, &out40[4], 1);
        printf("  lane %2d low :", k);
        for (int h = 0; h < 8; h++) printf(" %04x", (unsigned short)(((unsigned short *)out40)[h]));
        printf("  high:");
        for (int h = 0; h < 8; h++) printf(" %04x", (unsigned short)(((unsigned short *)out40)[h + 8]));
        printf("\n");
    }

    printf("  S16 value-sweep (lane 1: in[1]=V, filter=1, 1 MAC -> acc=V):\n");
    for (int i = 0; i < 16; i++) { in[i] = 0; filt[i] = 1; }
    for (int v = 1; v < 32; v = (v << 1)) {
        in[1] = (int8_t)v;
        probe_qacc_s16(filt, in, out40, &out40[4], 1);
        printf("  V=%3d low[0]=%04x low[1]=%04x\n", v,
               (unsigned short)(((unsigned short *)out40)[0]),
               (unsigned short)(((unsigned short *)out40)[1]));
    }
    in[1] = 127;
    probe_qacc_s16(filt, in, out40, &out40[4], 1);
    printf("  V=127 low[0]=%04x\n", (unsigned short)(((unsigned short *)out40)[0]));
    in[1] = -1;
    probe_qacc_s16(filt, in, out40, &out40[4], 1);
    printf("  V=-1  low[0]=%04x\n", (unsigned short)(((unsigned short *)out40)[0]));

    printf("  PER-CHANNEL full extraction (gapped store + l16si + SRCMB4/20 + VZIP.16):\n");
    {
        int32_t *out16 = pcout;       /* 32 bytes: zip16[0..16) + zip16[16..32) */
        int32_t *out_low = &pcout[8]; /* 16 bytes */
        int32_t *out_high = &pcout[12]; /* 16 bytes */
        for (int k = 0; k < 16; k++) {
            for (int i = 0; i < 16; i++) { in[i] = (i == k) ? 5 : 0; filt[i] = 2; }
            /* lane k acc = 5*2 = 10 (1 MAC); dump post-VZIP halves */
            probe_qacc_perchan(filt, in, out16, out_low, out_high, 1);
            printf("  lane %2d V10 zip16:", k);
            for (int h = 0; h < 8; h++) printf(" %04x", (unsigned short)(((unsigned short *)out16)[h]));
            printf("  | low4:");
            for (int h = 0; h < 8; h++) printf(" %04x", (unsigned short)(((unsigned short *)out_low)[h]));
            printf("  | high20:");
            for (int h = 0; h < 8; h++) printf(" %04x", (unsigned short)(((unsigned short *)out_high)[h]));
            printf("\n");
        }
        /* all-16-lanes distinct values, 1 MAC: lane k = (k+1)*2 */
        for (int i = 0; i < 16; i++) { in[i] = (int8_t)(i + 1); filt[i] = 2; }
        probe_qacc_perchan(filt, in, out16, out_low, out_high, 1);
        printf("  all16 V=(k+1)*2 zip16:");
        for (int h = 0; h < 8; h++) printf(" %04x", (unsigned short)(((unsigned short *)out16)[h]));
        printf("  |");
        for (int h = 0; h < 8; h++) printf(" %04x", (unsigned short)(((unsigned short *)out16)[h + 8]));
        printf("\n");

        /* RAW 40-byte contiguous store for all-16 distinct values (the key
         * calibration picture: shows the full diagonal at once) */
        probe_qacc_layout(filt, in, out40, 1, 0);
        printf("  all16 RAW40:");
        for (int b = 0; b < 40; b++) printf("%02x", (unsigned)((unsigned char *)out40)[b]);
        printf("\n");
        /* same with 2 MACs (acc = 2*(k+1)*2) to see carry into upper bits */
        probe_qacc_layout(filt, in, out40, 2, 0);
        printf("  all16 RAW40 macs2:");
        for (int b = 0; b < 40; b++) printf("%02x", (unsigned)((unsigned char *)out40)[b]);
        printf("\n");
    }

    printf("  ESP-LIKE 16x i32 readback (esp-nn verbatim two-pass bit-slice):\n");
    {
        static int32_t esq[16] __attribute__((aligned(16)));
        /* all16 V=(k+1)*2, 1 MAC -> expect lane k = (k+1)*2 */
        for (int i = 0; i < 16; i++) { in[i] = (int8_t)(i + 1); filt[i] = 2; }
        probe_qacc_esplike(filt, in, esq, 1);
        printf("  esplike 1MAC:");
        for (int h = 0; h < 16; h++) printf(" %d", (int)esq[h]);
        printf("\n");
        /* all16 V=(k+1)*2, 2 MACs -> expect lane k = 2*(k+1)*2 = (k+1)*4 */
        probe_qacc_esplike(filt, in, esq, 2);
        printf("  esplike 2MAC:");
        for (int h = 0; h < 16; h++) printf(" %d", (int)esq[h]);
        printf("\n");
    }
}

/* ---- scalar refs (mirror hematite-ref exactly) ---- */

/* multiply_by_quantized_multiplier(value, mult, shift) in i64 */
static int32_t req(const int32_t value, const int32_t mult, const int32_t shift) {
    int total_shift = 31 - shift;
    int64_t round = (int64_t)1 << (total_shift - 1);
    int64_t result = ((int64_t)value * (int64_t)mult + round) >> total_shift;
    if (result < INT32_MIN) return INT32_MIN;
    if (result > INT32_MAX) return INT32_MAX;
    return (int32_t)result;
}

static int8_t sat_cast(int32_t v) {
    if (v < -128) return -128;
    if (v > 127) return 127;
    return (int8_t)v;
}

static int32_t round_half_away_zero(int32_t n, int32_t d) {
    return (n > 0) ? (n + d / 2) / d : (n - d / 2) / d;
}

static void scalar_conv1x1(int8_t *out, const int8_t *in, const int8_t *w,
                           const int32_t *b) {
    for (int oc = 0; oc < OUT_C; oc++) {
        int32_t acc = b[oc];
        for (int ic = 0; ic < IN_C; ic++) {
            acc += (int32_t)in[ic] * (int32_t)w[oc * IN_C + ic];
        }
        out[oc] = sat_cast(req(acc, 1 << 30, 0));
    }
}

static void scalar_conv3x3(int8_t *out, const int8_t *in, const int8_t *w,
                           const int32_t *b) {
    for (int oc = 0; oc < C3_OUT_C; oc++) {
        for (int oh = 0; oh < C3_OUT_H; oh++) {
            for (int ow = 0; ow < C3_OUT_W; ow++) {
                int32_t acc = b[oc];
                for (int kh = 0; kh < 3; kh++) {
                    for (int kw = 0; kw < 3; kw++) {
                        for (int ic = 0; ic < C3_IN_C; ic++) {
                            int in_idx = ((oh + kh) * C3_IN_W + (ow + kw)) * C3_IN_C + ic;
                            int w_idx = ((oc * 3 + kh) * 3 + kw) * C3_IN_C + ic;
                            acc += (int32_t)in[in_idx] * (int32_t)w[w_idx];
                        }
                    }
                }
                out[(oh * C3_OUT_W + ow) * C3_OUT_C + oc] = sat_cast(req(acc, 1 << 30, 0));
            }
        }
    }
}

static void scalar_fc(int8_t *out, const int8_t *in, const int8_t *w,
                      const int32_t *b) {
    for (int oc = 0; oc < FC_OUT_DIM; oc++) {
        int32_t acc = b[oc];
        for (int ic = 0; ic < FC_IN_DIM; ic++) {
            acc += (int32_t)in[ic] * (int32_t)w[oc * FC_IN_DIM + ic];
        }
        out[oc] = sat_cast(req(acc, 1 << 30, 0));
    }
}

static void scalar_max_pool2d(int8_t *out, const int8_t *in) {
    for (int oh = 0; oh < 16; oh++) {
        for (int ow = 0; ow < 16; ow++) {
            for (int c = 0; c < P_C; c++) {
                int8_t m = INT8_MIN;
                for (int kh = 0; kh < 2; kh++) {
                    for (int kw = 0; kw < 2; kw++) {
                        int in_idx = ((oh * 2 + kh) * P_IN_W + (ow * 2 + kw)) * P_C + c;
                        if (in[in_idx] > m) m = in[in_idx];
                    }
                }
                out[(oh * 16 + ow) * P_C + c] = sat_cast((int32_t)m);
            }
        }
    }
}

static void scalar_avg_pool2d(int8_t *out, const int8_t *in) {
    for (int oh = 0; oh < 16; oh++) {
        for (int ow = 0; ow < 16; ow++) {
            for (int c = 0; c < P_C; c++) {
                int32_t acc = 0;
                for (int kh = 0; kh < 2; kh++) {
                    for (int kw = 0; kw < 2; kw++) {
                        int in_idx = ((oh * 2 + kh) * P_IN_W + (ow * 2 + kw)) * P_C + c;
                        acc += in[in_idx];
                    }
                }
                out[(oh * 16 + ow) * P_C + c] = sat_cast(round_half_away_zero(acc, 4));
            }
        }
    }
}

static void scalar_relu(int8_t *out, const int8_t *in) {
    for (int i = 0; i < N256; i++) {
        int32_t val = (int32_t)in[i];
        int32_t act = val > 0 ? val : 0;
        out[i] = sat_cast(req(act, 1 << 30, 1));
    }
}

static void scalar_add(int8_t *out, const int8_t *a, const int8_t *b) {
    for (int i = 0; i < N256; i++) {
        out[i] = sat_cast(req((int32_t)a[i] + (int32_t)b[i], 1 << 30, 1));
    }
}

static void scalar_sub(int8_t *out, const int8_t *a, const int8_t *b) {
    for (int i = 0; i < N256; i++) {
        out[i] = sat_cast(req((int32_t)a[i] - (int32_t)b[i], 1 << 30, 1));
    }
}

static void scalar_mul(int8_t *out, const int8_t *a, const int8_t *b) {
    for (int i = 0; i < N256; i++) {
        out[i] = sat_cast(req((int32_t)a[i] * (int32_t)b[i], 1 << 30, 0));
    }
}

/* ---- SIMD kernel wrappers (build args structs, call asm) ---- */

static void conv1x1_simd_entry(void) {
    Tie728ConvArgs a;
    memset(&a, 0, sizeof(a));
    a.filter = s_tw;            /* transformed [g][ic][lane] weights */
    a.mac_shift = 1;            /* = 1 - output_shift(0), matches Rust bench12 */
    a.bias = s_bias;
    a.activation_alpha = 0;
    a.activation_shift = -1;
    a.output_channel_div_8 = OUT_C / 16;
    a.c_div_x_1 = IN_C / 16 - 1;
    a.filter_channel_factor = NULL;
    dl_tie728_s8_conv2d_11cn(s_output, s_input, &a);
}

static void conv3x3_simd_entry(void) {
    Tie728Conv33Args a;
    memset(&a, 0, sizeof(a));
    a.filter = s_c3_w;
    a.mac_shift = 0;
    a.bias = s_c3_b;
    a.activation_alpha = 0;
    a.activation_alpha_ptr = NULL;
    a.activation_shift = -1;
    a.output_channel_div_8 = C3_OUT_C / 16;
    a.c_div_x_1 = C3_IN_C / 16 - 1;
    a.filter_channel_factor = NULL;
    a.dilation_x_offset = 1;
    a.dilation_y_offset = 1;
    dl_tie728_s8_conv2d_33cn(s_c3_out, s_c3_in, &a);
}

static void fc_simd_entry(void) {
    Tie728ConvArgs a;
    memset(&a, 0, sizeof(a));
    a.filter = s_fc_w;
    a.mac_shift = 0;
    a.bias = s_fc_b;
    a.activation_alpha = 0;
    a.activation_shift = -1;
    a.output_channel_div_8 = FC_OUT_DIM / 16;
    a.c_div_x_1 = FC_IN_DIM / 16 - 1;
    a.filter_channel_factor = NULL;
    dl_tie728_s8_conv2d_11cn(s_fc_out, s_fc_in, &a);
}

static void maxpool_simd_entry(void) {
    Tie728MaxPoolArgs a;
    memset(&a, 0, sizeof(a));
    a.input_channel = P_C;
    a.input_y_offset = P_IN_W * P_C;
    a.input_x_offset = P_C;
    a.filter_height = 2;
    a.filter_width = 2;
    a.c_remainder = 0;
    a.c_div_x_1 = P_OUT_LEN / 16 - 1;
    dl_tie728_s8_max_pool2d_22c1(s_p_out, s_p_in, &a);
}

static void avgpool_simd_entry(void) {
    Tie728AvgPoolArgs a;
    memset(&a, 0, sizeof(a));
    a.input_channel = P_C;
    a.input_y_offset = P_IN_W * P_C;
    a.input_x_offset = P_C;
    a.filter_height = 2;
    a.filter_width = 2;
    a.shift = 8;
    for (int i = 0; i < 16; i++) a.avg_pool_area_inv[i] = 64;
    a.c_div_x_1 = P_OUT_LEN / 16 - 1;
    dl_tie728_s8_avg_pool2d_22c1(s_p_out, s_p_in, &a);
}

static void relu_simd_entry(void) {
    Tie728ReluArgs a;
    memset(&a, 0, sizeof(a));
    a.activation_alpha = 0;
    a.activation_shift = 0;
    a.c_rs1_1 = (N256 - 16) / 32;
    a.c_rs2_1 = ((N256 - 16) % 32) / 16;
    dl_tie728_s8_relu_11c(s_r_out, s_r_in, &a);
}

static void add_simd_entry(void) {
    AddSubAlignedArgs a;
    memset(&a, 0, sizeof(a));
    a.length = N256;
    dl_tie728_s8_add_w1_16_w2_16(s_e_out, s_e1, s_e2, &a);
}

static void sub_simd_entry(void) {
    AddSubAlignedArgs a;
    memset(&a, 0, sizeof(a));
    a.length = N256;
    dl_tie728_s8_sub_w1_16_w2_16(s_e_out, s_e1, s_e2, &a);
}

static void mul_simd_entry(void) {
    MulAlignedArgs a;
    memset(&a, 0, sizeof(a));
    a.c_div_x_1 = N256 / 16 - 1;
    a.mul_shift = 1;   /* 1 - output_shift(0) */
    dl_tie728_s8_mul_w1_16_w2_16(s_e_out, s_e1, s_e2, &a);
}

/* ---- QACC extraction probe (debugging only) ----
 * With input=0 and weights=0, QACC = bias after 128b_vector_bias, so the
 * output bytes reveal how int32 bias values land in QACC lanes and how
 * SRCMB.S8 + vector_round_result transform them. */
static void probe_qacc(void) {
    static const int32_t biasv[16] = {
        0x00000000, 0x00000001, 0x0000FFFF, 0xFFFF0000,
        0x7FFFFFFF, 0x80000000, 0x12345678, 0x89ABCDEF,
        0x0000007F, 0x00000080, 0x0000FF00, 0x00FF0000,
        0xFFFFFFFF, 0x00010001, 0x00000100, 0x01000000,
    };
    static int8_t probe_out[16] __attribute__((aligned(16)));
    Tie728ConvArgs a;
    for (int i = 0; i < IN_C; i++) s_input[i] = 0;
    for (int i = 0; i < WEIGHTS_LEN; i++) s_weights[i] = 0;
    for (int i = 0; i < 16; i++) s_bias[i] = biasv[i];
    for (int shift = 0; shift < 4; shift++) {
        memset(&a, 0, sizeof(a));
        a.filter = s_weights;
        a.mac_shift = shift;
        a.bias = s_bias;
        a.activation_alpha = 0;
        a.activation_shift = -1;
        a.output_channel_div_8 = 1;   /* one group of 16 */
        a.c_div_x_1 = 0;              /* one input chunk */
        a.filter_channel_factor = NULL;
        dl_tie728_s8_conv2d_11cn(probe_out, s_input, &a);
        dump_bytes("qacc", probe_out, 16);
        printf("  ^ mac_shift=%d\n", shift);
    }
    /* input=1, weights=1, bias=0 -> lane value = number of MACs (=64) if lanes
     * accumulate without saturation. */
    for (int i = 0; i < IN_C; i++) s_input[i] = 1;
    for (int i = 0; i < WEIGHTS_LEN; i++) s_weights[i] = 1;
    for (int i = 0; i < 16; i++) s_bias[i] = 0;
    for (int shift = 0; shift < 3; shift++) {
        memset(&a, 0, sizeof(a));
        a.filter = s_weights;
        a.mac_shift = shift;
        a.bias = s_bias;
        a.activation_alpha = 0;
        a.activation_shift = -1;
        a.output_channel_div_8 = 1;
        a.c_div_x_1 = IN_C / 16 - 1;  /* 3 chunks of 16 */
        a.filter_channel_factor = NULL;
        dl_tie728_s8_conv2d_11cn(probe_out, s_input, &a);
        dump_bytes("qacc1s", probe_out, 16);
        printf("  ^ mac_shift=%d input=1 weights=1 (64 MACs)\n", shift);
    }

    /* SATURATION-POINT PROBE: input=weights=127, bias=0, mac_shift=0, vary the
     * number of 16-byte input chunks (c_div_x_1 = 0..4 -> 1..5 chunks = 16..80
     * MACs per lane). 127*127 = 16129 per chunk.
     *   16-bit lanes: exact up to 2 chunks (32258 < 32767), saturates/wraps at 3+.
     *   32-bit lanes: exact for all (up to 129032). */
    for (int i = 0; i < IN_C; i++) s_input[i] = 127;
    for (int i = 0; i < WEIGHTS_LEN; i++) s_weights[i] = 127;
    for (int i = 0; i < 16; i++) s_bias[i] = 0;
    for (int cdiv = 0; cdiv <= 4; cdiv++) {
        memset(&a, 0, sizeof(a));
        a.filter = s_weights;
        a.mac_shift = 0;
        a.bias = s_bias;
        a.activation_alpha = 0;
        a.activation_shift = -1;
        a.output_channel_div_8 = 1;
        a.c_div_x_1 = cdiv;
        a.filter_channel_factor = NULL;
        dl_tie728_s8_conv2d_11cn(probe_out, s_input, &a);
        dump_bytes("qacc127", probe_out, 16);
        printf("  ^ c_div_x_1=%d (%d MACs/lane, expect %d if 16-bit ok, %d if 32-bit)\n",
               cdiv, (cdiv + 1) * 16, (cdiv + 1) * 16 * 127 * 127, (cdiv + 1) * 16 * 127 * 127);
    }
}

/* ---- generic bench harness ---- */
typedef void (*bench_fn)(void);

static int32_t s_accx[OUT_C] __attribute__((aligned(16)));
static int8_t s_accx_out[OUT_C] __attribute__((aligned(16)));
static int32_t s_fc_accx[FC_OUT_DIM] __attribute__((aligned(16)));
static int8_t s_fc_accx_out[FC_OUT_DIM] __attribute__((aligned(16)));

static void kern_conv1x1_new(void) {
    s8_accx_conv1x1(s_input, s_weights, s_accx, IN_C, OUT_C);
    for (int oc = 0; oc < OUT_C; oc++)
        s_accx_out[oc] = sat_cast(req(s_accx[oc] + s_bias[oc], 1 << 30, 0));
}
static void kern_conv1x1_new_pure(void) {
    s8_accx_conv1x1(s_input, s_weights, s_accx, IN_C, OUT_C);
}
static void kern_conv1x1_orig(void) {
    s8_accx_conv1x1_orig(s_input, s_weights, s_accx, IN_C, OUT_C);
    for (int oc = 0; oc < OUT_C; oc++)
        s_accx_out[oc] = sat_cast(req(s_accx[oc] + s_bias[oc], 1 << 30, 0));
}
static void kern_conv1x1_orig_pure(void) {
    s8_accx_conv1x1_orig(s_input, s_weights, s_accx, IN_C, OUT_C);
}
static void kern_fc_new(void) {
    s8_accx_conv1x1(s_fc_in, s_fc_w, s_fc_accx, FC_IN_DIM, FC_OUT_DIM);
    for (int oc = 0; oc < FC_OUT_DIM; oc++)
        s_fc_accx_out[oc] = sat_cast(req(s_fc_accx[oc] + s_fc_b[oc], 1 << 30, 0));
}
static void kern_fc_new_pure(void) {
    s8_accx_conv1x1(s_fc_in, s_fc_w, s_fc_accx, FC_IN_DIM, FC_OUT_DIM);
}
static void kern_fc_orig(void) {
    s8_accx_conv1x1_orig(s_fc_in, s_fc_w, s_fc_accx, FC_IN_DIM, FC_OUT_DIM);
    for (int oc = 0; oc < FC_OUT_DIM; oc++)
        s_fc_accx_out[oc] = sat_cast(req(s_fc_accx[oc] + s_fc_b[oc], 1 << 30, 0));
}
static void kern_fc_orig_pure(void) {
    s8_accx_conv1x1_orig(s_fc_in, s_fc_w, s_fc_accx, FC_IN_DIM, FC_OUT_DIM);
}

/* TEMP-DEBUG: anomaly-shaped pure kernel rows. */
static void kern_a640_pure(void) {
    s8_accx_conv1x1(s_a640_in, s_a640_w, s_a640_accx, A640_IN_DIM, A640_OUT_DIM);
}

static void kern_a128_pure(void) {
    s8_accx_conv1x1(s_a128_in, s_a128_w, s_a128_accx, A128_IN_DIM, A128_OUT_DIM);
}

/* Flash-latency hypothesis: identical call but weights come from DROM consts. */
static void kern_a640_pure_drom(void) {
    s8_accx_conv1x1(s_a640_in, s_a640_w_const, s_a640_accx, A640_IN_DIM, A640_OUT_DIM);
}

static void kern_a128_pure_drom(void) {
    s8_accx_conv1x1(s_a128_in, s_a128_w_const, s_a128_accx, A128_IN_DIM, A128_OUT_DIM);
}

/* Full 640x128 DROM stream — the exact anomaly op0/op9 footprint. */
static void kern_a640b_pure_drom(void) {
    s8_accx_conv1x1(s_a640b_in, s_a640b_w_const, s_a640b_accx, A640B_IN_DIM, A640B_OUT_DIM);
}
static int32_t s_c3_accx[C3_OUT_C] __attribute__((aligned(16)));
static int8_t s_ref[C3_OUT_LEN] __attribute__((aligned(16)));
static int8_t s_ref64[FC_OUT_DIM] __attribute__((aligned(16)));

static void conv3x3_accx_full(const int8_t *filter) {
    const int row_delta = (C3_IN_W - 3) * C3_IN_C;
    for (int oh = 0; oh < C3_OUT_H; oh++) {
        for (int ow = 0; ow < C3_OUT_W; ow++) {
            const int px = (oh * C3_IN_W + ow) * C3_IN_C;
            const int po = (oh * C3_OUT_W + ow) * C3_OUT_C;
            s8_accx_conv3x3(s_c3_in + px, filter, s_c3_accx, C3_IN_C, C3_OUT_C, row_delta);
            for (int oc = 0; oc < C3_OUT_C; oc++)
                s_ref[po + oc] = sat_cast(req(s_c3_accx[oc] + s_c3_b[oc], 1 << 30, 0));
        }
    }
}
static void kern_c3_new(void) { conv3x3_accx_full(s_c3_w); }
static void kern_c3_new_pure(void) {
    const int row_delta = (C3_IN_W - 3) * C3_IN_C;
    for (int oh = 0; oh < C3_OUT_H; oh++)
        for (int ow = 0; ow < C3_OUT_W; ow++)
            s8_accx_conv3x3(s_c3_in + (oh * C3_IN_W + ow) * C3_IN_C, s_c3_w,
                            s_c3_accx, C3_IN_C, C3_OUT_C, row_delta);
}

/* ---- bespoke depthwise (s8_accx_depthwise.S): QACC per-lane ---- */

static void fill_depthwise(void) {
    for (int i = 0; i < DW_IN_LEN; i++) s_dw_in[i] = (int8_t)((i * 7 + 3) & 0xFF);
    for (int i = 0; i < DW_W_LEN; i++) s_dw_w[i] = (int8_t)((i * 13 + 11) & 0xFF);
    for (int i = 0; i < DW_C; i++) s_dw_b[i] = i * 17 - 8;
}

/* scalar depthwise 3x3 stride1 pad0 dm1, HWCN layout filter[(tap)*out_c + ch] */
static void scalar_depthwise(void) {
    const int row_stride = DW_IN_W * DW_C;
    for (int oh = 0; oh < DW_OUT_H; oh++) {
        for (int ow = 0; ow < DW_OUT_W; ow++) {
            for (int oc = 0; oc < DW_C; oc++) {
                int32_t acc = s_dw_b[oc];
                for (int fy = 0; fy < 3; fy++) {
                    for (int fx = 0; fx < 3; fx++) {
                        const int tap = fy * 3 + fx;
                        const int in_idx = (oh + fy) * row_stride + (ow + fx) * DW_C + oc;
                        const int w_idx = tap * DW_C + oc;
                        acc += (int32_t)s_dw_in[in_idx] * (int32_t)s_dw_w[w_idx];
                    }
                }
                s_dw_ref[(oh * DW_OUT_W + ow) * DW_C + oc] =
                    sat_cast(req(acc, 1 << 30, 0));
            }
        }
    }
}

static void kern_depthwise(void) {
    const int row_delta = (DW_IN_W - 3) * DW_C;
    for (int oh = 0; oh < DW_OUT_H; oh++) {
        for (int ow = 0; ow < DW_OUT_W; ow++) {
            const int px = (oh * DW_IN_W + ow) * DW_C;
            const int po = (oh * DW_OUT_W + ow) * DW_C;
            s8_accx_depthwise(s_dw_in + px, s_dw_w, s_dw_accx, DW_C, DW_C, row_delta);
            for (int oc = 0; oc < DW_C; oc++)
                s_dw_out[po + oc] = sat_cast(req(s_dw_accx[oc] + s_dw_b[oc], 1 << 30, 0));
        }
    }
}
static void kern_depthwise_pure(void) {
    const int row_delta = (DW_IN_W - 3) * DW_C;
    for (int oh = 0; oh < DW_OUT_H; oh++)
        for (int ow = 0; ow < DW_OUT_W; ow++)
            s8_accx_depthwise(s_dw_in + (oh * DW_IN_W + ow) * DW_C, s_dw_w,
                              s_dw_accx, DW_C, DW_C, row_delta);
}

/* ---- kws 10x8 dm8 SAME stride2 anytap probe (isolates kernel cost) ----
 * Mirrors the Rust dispatch's padded representation for the kws layer:
 *   input   [1,49,40,1]  -> padded [58,46,16]   (pad_total_h=9, pad_total_w=6)
 *   filter  [1,10,8,8]   -> padded 80 taps x 16 (padded_c = pad16(out_c=8))
 *   output  [1,25,20,8]  -> padded out_c = 16
 *   stride 2/2, dm=8, row_delta = (padded_w - filter_w)*padded_c = 608
 */
#define KWS_PAD_H 58
#define KWS_PAD_W 46
#define KWS_PC    16
#define KWS_FH    10
#define KWS_FW    8
#define KWS_OH    25
#define KWS_OW    20
#define KWS_PAD_H 58
#define KWS_PAD_W 46
#define KWS_PC    16
#define KWS_FH    10
#define KWS_FW    8
#define KWS_OH    25
#define KWS_OW    20
/* Reuse the conv3x3 buffers (rows above already ran; s_c3_in 65536B >=
 * 58*46*16=42688B padded input, s_c3_w 36864B >= 80*16 filter). */
#define s_kws_in  s_c3_in
#define s_kws_w   s_c3_w
static int32_t s_kws_part[KWS_PC] __attribute__((aligned(16)));
static int32_t s_kws_accx[KWS_PC] __attribute__((aligned(16)));

static void fill_kws(void) {
    for (int i = 0; i < KWS_PAD_H * KWS_PAD_W * KWS_PC; i++) s_kws_in[i] = (int8_t)((i * 7 + 3) & 0xFF);
    for (int i = 0; i < KWS_FH * KWS_FW * KWS_PC; i++) s_kws_w[i] = (int8_t)((i * 13 + 11) & 0xFF);
    for (int i = 0; i < KWS_PC; i++) s_kws_part[i] = 0;
}

/* Pure kernel path: 500 px x 3 chunks of 32 taps, partials folded, NO
 * staging/wsum/requantize — the exact anytap call+fold cost. */
static void kern_anytap_kws_pure(void) {
    const uint32_t row_delta = (KWS_PAD_W - KWS_FW) * KWS_PC;
    const uint32_t filt_w = KWS_FW;
    for (int oh = 0; oh < KWS_OH; oh++) {
        for (int ow = 0; ow < KWS_OW; ow++) {
            const int px = (oh * 2 * KWS_PAD_W + ow * 2) * KWS_PC;
            int tap_start = 0;
            while (tap_start < KWS_FH * KWS_FW) {
                int taps = KWS_FH * KWS_FW - tap_start;
                if (taps > 32) taps = 32;
                const int row = tap_start / KWS_FW;
                const int col = tap_start % KWS_FW;
                const int8_t *in_ptr = s_kws_in + px +
                    (row * (KWS_FW * KWS_PC + row_delta) + col * KWS_PC);
                AnyTapCtx ctx;
                ctx.input = in_ptr;
                ctx.filter = s_kws_w + tap_start * KWS_PC;
                ctx.acc_out = s_kws_part;
                ctx.in_c = KWS_PC;
                ctx.out_c = KWS_PC;
                ctx.row_delta = row_delta;
                ctx.taps = (uint32_t)taps;
                ctx.filter_w = filt_w;
                ctx.col_start = (uint32_t)col;
                s8_accx_depthwise_anytap(&ctx);
                for (int i = 0; i < KWS_PC; i++)
                    s_kws_accx[i] = s_kws_accx[i] + s_kws_part[i];
                tap_start += taps;
            }
        }
    }
}

/* bc1 broadcast-kernel mirror of the Rust dispatch: single-channel padded
 * input (58x46), filter [80][16] padded_c, per-pixel x 3 chunks of 32 taps,
 * partials folded — NO wsum/requantize, exactly like kern_anytap_kws_pure. */
#define s_kws_in1  (s_c3_in)   /* 65536B >= 58*46=2668B single-channel */
/* Stage the single-channel padded input: channel-0 byte of each (h,w) from
 * the padded_c-wide fill, matching the Rust dispatch's single-channel copy. */
static void fill_kws_bc1(void) {
    for (int i = 0; i < KWS_FH * KWS_FW * KWS_PC; i++)
        s_kws_w[i] = (i % KWS_PC < 8) ? (int8_t)((i * 13 + 11) & 0xFF) : 0;  /* pad lanes 8-15 zero */
    for (int h = 0; h < KWS_PAD_H; h++)
        for (int w = 0; w < KWS_PAD_W; w++)
            s_kws_in1[h * KWS_PAD_W + w] = (int8_t)(((h * KWS_PAD_W + w) * 16 * 7 + 3) & 0xFF);
    for (int i = 0; i < KWS_PC; i++) { s_kws_part[i] = 0; s_kws_accx[i] = 0; }
}
static void kern_bc1_kws_pure(void) {
    const uint32_t row_delta = (KWS_PAD_W - KWS_FW) * 1;   /* single-channel */
    const uint32_t filt_w = KWS_FW;
    for (int oh = 0; oh < KWS_OH; oh++) {
        for (int ow = 0; ow < KWS_OW; ow++) {
            const int px = (oh * 2 * KWS_PAD_W + ow * 2) * 1; /* k_in_c==1 */
            int tap_start = 0;
            while (tap_start < KWS_FH * KWS_FW) {
                int taps = KWS_FH * KWS_FW - tap_start;
                if (taps > 32) taps = 32;
                const int row = tap_start / KWS_FW;
                const int col = tap_start % KWS_FW;
                const int8_t *in_ptr = s_kws_in1 + px +
                    (row * (KWS_FW * 1 + row_delta) + col * 1);
                Bc1Ctx ctx;
                ctx.input = in_ptr;
                ctx.filter = s_kws_w + tap_start * KWS_PC;   /* [tap][16] */
                ctx.acc_out = s_kws_part;
                ctx.in_c = 1;
                ctx.out_c = KWS_PC;
                ctx.row_delta = row_delta;
                ctx.taps = (uint32_t)taps;
                ctx.filter_w = filt_w;
                ctx.col_start = (uint32_t)col;
                s8_accx_depthwise_anytap_bc1(&ctx);
                for (int i = 0; i < KWS_PC; i++)
                    s_kws_accx[i] = s_kws_accx[i] + s_kws_part[i];
                tap_start += taps;
            }
        }
    }
}
/* Correct scalar over the SAME single-channel padded 58x46 input: per output
 * pixel, 80 taps at (oh*2 + row - pad, ow*2 + col - pad) with pad=1, reading
 * the padded single-channel buffer directly (0 for OOB = the -fill/off fold
 * analogue). */
static void kern_scalar_kws_bc1(void) {
    const int32_t out_h = 25, out_w = 20, out_c = 8, fh = 10, fw = 8;
    for (int32_t oh = 0; oh < out_h; oh++) {
        for (int32_t ow = 0; ow < out_w; ow++) {
            for (int32_t oc = 0; oc < KWS_PC; oc++) {
                int32_t acc = 0;
                for (int32_t k = 0; k < fh; k++) {
                    for (int32_t l = 0; l < fw; l++) {
                        const int32_t ih = oh * 2 + k;   /* kernel reads the
                                                            directly-filled
                                                            padded buffer at
                                                            (oh*2+row, ow*2+col) */
                        const int32_t iw = ow * 2 + l;
                        const int32_t inb = (ih >= 0 && ih < KWS_PAD_H && iw >= 0 && iw < KWS_PAD_W)
                            ? s_kws_in1[ih * KWS_PAD_W + iw] : 0;
                        acc += inb * s_kws_w[(k * fw + l) * KWS_PC + oc];
                    }
                }
                s_kws_accx[oc] = s_kws_accx[oc] + acc;  /* fold like the kernel */
            }
        }
    }
}

/* Scalar depthwise mirror of hematite-ref for the kws shape: [1,49,40,1] ->
 * [1,25,20,8], SAME stride2, 10x8 filter, input_offset 128, act 0..127.
 * Uses the raw (unpadded) s_c3_in buffer as the 49x40x1 input. */
static void kern_scalar_kws(void) {
    const int32_t in_h = 49, in_w = 40, in_c = 1;
    const int32_t out_h = 25, out_w = 20, out_c = 8;
    const int32_t dm = 8, fh = 10, fw = 8, stride = 2, pad = 1;
    const int32_t in_off = 128, out_off = -128, act_min = 0, act_max = 127;
    const int32_t input_row_stride = in_w * in_c;
    const int32_t filter_row_stride = fw * out_c;
    for (int32_t oh = 0; oh < out_h; oh++) {
        const int32_t ib_h = oh * stride - pad;
        for (int32_t ow = 0; ow < out_w; ow++) {
            const int32_t ib_w = ow * stride - pad;
            for (int32_t ic = 0; ic < in_c; ic++) {
                for (int32_t d = 0; d < dm; d++) {
                    const int32_t oc = d + ic * dm;
                    int32_t acc = 0; /* bias 0 for the probe */
                    for (int32_t k = 0; k < fh; k++) {
                        const int32_t in_row = ib_h + k;
                        const int32_t row_ok = in_row >= 0 && in_row < in_h;
                        for (int32_t l = 0; l < fw; l++) {
                            const int32_t in_col = ib_w + l;
                            if (row_ok && in_col >= 0 && in_col < in_w) {
                                const int32_t in_idx =
                                    (in_row * input_row_stride + in_col * in_c + ic);
                                const int32_t f_idx =
                                    (k * filter_row_stride + l * out_c + oc);
                                const int32_t iv = (int32_t)s_c3_in[in_idx] + in_off;
                                const int32_t wv = (int32_t)s_c3_w[f_idx];
                                acc += iv * wv;
                            }
                        }
                    }
                    int32_t scaled = req(acc, 1 << 30, 0);
                    int32_t vo = scaled + out_off;
                    if (vo > act_max) vo = act_max;
                    if (vo < act_min) vo = act_min;
                    s_kws_accx[0] += vo; /* fold into checksum-ish sink */
                }
            }
        }
    }
}

static void run_bench(const char *label, bench_fn fill, bench_fn kern,
                      const int8_t *outbuf, size_t outlen) {
    const int WARMUP = 1, TIMED = 10;
    fill();
    for (int r = 0; r < WARMUP; r++) kern();
    uint32_t runs[TIMED];
    for (int r = 0; r < TIMED; r++) {
        fill();
        uint32_t t0 = read_ccount();
        kern();
        uint32_t t1 = read_ccount();
        runs[r] = t1 - t0;
    }
    for (int i = 1; i < TIMED; i++) {
        uint32_t v = runs[i];
        int j = i - 1;
        while (j >= 0 && runs[j] > v) { runs[j + 1] = runs[j]; j--; }
        runs[j + 1] = v;
    }
    uint32_t min = runs[0];
    uint32_t med = (runs[TIMED / 2 - 1] + runs[TIMED / 2]) / 2;
    uint32_t chk = fnv1a(outbuf, outlen);
    printf("== %s ==\n", label);
    printf("  N=%d min=%u median=%u cycles | min=%.2fus median=%.2fus | out_checksum(fnv1a)=0x%08x\n",
           TIMED, (unsigned)min, (unsigned)med,
           (double)min / 240.0, (double)med / 240.0, (unsigned)chk);
}

void app_main(void) {
    printf("\n=== Hematite ESP-DL baseline (benchmarks/espdl-baseline) ===\n");
    printf("LABEL: real hardware (ESP32-S3 @ 240MHz), ESP-IDF v5.5.1, vendored dl_tie728 asm\n");

    init_quant_consts();

    probe_qacc();
    probe_accx_run();
    probe_s8accx_run();
    probe_qacc_layout_run();
    probe_vldbc_run();

    /* conv1x1 64x1x1x64 */
    run_bench("conv1x1_s8 64x1x1x64 TIE728-SIMD (dl_tie728_s8_conv2d_11cn)",
              fill_pattern_conv1x1_tw, conv1x1_simd_entry, s_output, OUT_C);
    scalar_conv1x1(s_ref64, s_input, s_weights, s_bias);
    printf("  scalar-ref fnv1a=0x%08x  (Rust bench10: ref=0x0bea8225 s3=0x5eee898e)\n",
           (unsigned)fnv1a((const int8_t *)s_ref64, OUT_C));
    fill_pattern_conv1x1_tw();
    conv1x1_simd_entry();
    dump_bytes("s3 out", s_output, OUT_C);
    dump_bytes("ref out", s_ref64, OUT_C);
    printf("  s3 fnv1a=0x%08x (bench12 device s3=0x61f7a941, ref=0x0bea8225)\n",
           (unsigned)fnv1a((const int8_t *)s_output, OUT_C));

    /* --- BESPOKE ACCX conv1x1 (s8_accx_conv1x1.S): bit-exact GPR-accumulator --- */
    printf("== conv1x1_s8 64x1x1x64 BESPOKE-ACCX new (s8_accx_conv1x1) ==\n");
    run_bench("conv1x1_s8 64x1x1x64 BESPOKE-ACCX new", fill_pattern_conv1x1_tw,
              kern_conv1x1_new, s_accx_out, OUT_C);
    printf("  ref=0x0bea8225\n");
    printf("== conv1x1_s8 64x1x1x64 BESPOKE-ACCX orig (branchy) ==\n");
    run_bench("conv1x1_s8 64x1x1x64 BESPOKE-ACCX orig", fill_pattern_conv1x1_tw,
              kern_conv1x1_orig, s_accx_out, OUT_C);
    printf("  ref=0x0bea8225\n");
    printf("== conv1x1_s8 64x1x1x64 BESPOKE-ACCX new PURE (no requantize) ==\n");
    run_bench("conv1x1_s8 64x1x1x64 BESPOKE-ACCX new PURE", fill_pattern_conv1x1_tw,
              kern_conv1x1_new_pure, (const int8_t *)s_accx, OUT_C * 4);
    printf("== conv1x1_s8 64x1x1x64 BESPOKE-ACCX orig PURE ==\n");
    run_bench("conv1x1_s8 64x1x1x64 BESPOKE-ACCX orig PURE", fill_pattern_conv1x1_tw,
              kern_conv1x1_orig_pure, (const int8_t *)s_accx, OUT_C * 4);

    /* conv3x3 32x32 64x3x3x64 */
    run_bench("conv3x3_s8 32x32,64x3x3x64 VALID TIE728-SIMD (dl_tie728_s8_conv2d_33cn)",
              fill_pattern_conv3x3, conv3x3_simd_entry, s_c3_out, C3_OUT_LEN);
    scalar_conv3x3(s_ref, s_c3_in, s_c3_w, s_c3_b);
    printf("  scalar-ref fnv1a=0x%08x  (Rust bench10: ref=0x0a181085 s3=0xd1a9b601)\n",
           (unsigned)fnv1a((const int8_t *)s_ref, C3_OUT_LEN));
    printf("== conv3x3_s8 32x32,64x3x3x64 BESPOKE-ACCX new ==\n");
    run_bench("conv3x3_s8 32x32,64x3x3x64 BESPOKE-ACCX new", fill_pattern_conv3x3,
              kern_c3_new, s_ref, C3_OUT_LEN);
    printf("  ref=0x0a181085\n");
    printf("== conv3x3_s8 32x32,64x3x3x64 BESPOKE-ACCX new PURE ==\n");
    run_bench("conv3x3_s8 32x32,64x3x3x64 BESPOKE-ACCX new PURE", fill_pattern_conv3x3,
              kern_c3_new_pure, (const int8_t *)s_c3_accx, C3_OUT_C * 4);

    /* --- BESPOKE depthwise (s8_accx_depthwise.S): QACC per-lane --- */
    fill_depthwise();
    scalar_depthwise();
    printf("== depthwise_s8 7x7,32x3x3x32 BESPOKE-QACC (s8_accx_depthwise) ==\n");
    run_bench("depthwise_s8 7x7,32x3x3x32 BESPOKE-QACC", fill_depthwise,
              kern_depthwise, s_dw_out, DW_OUT_LEN);
    printf("  ref fnv1a=0x%08x\n",
           (unsigned)fnv1a((const int8_t *)s_dw_ref, DW_OUT_LEN));
    printf("  kernel out fnv1a=0x%08x\n",
           (unsigned)fnv1a((const int8_t *)s_dw_out, DW_OUT_LEN));
    run_bench("depthwise_s8 7x7,32x3x3x32 BESPOKE-QACC PURE", fill_depthwise,
              kern_depthwise_pure, (const int8_t *)s_dw_accx, DW_C * 4);

    /* --- kws 10x8 dm8 SAME stride2 anytap kernel-only probe --- */
    run_bench("depthwise_s8 kws 49x40,1x10x8x8 dm8 S2 ANYTAP PURE (kernel only)",
              fill_kws, kern_anytap_kws_pure, (const int8_t *)s_kws_accx,
              KWS_PC * 4);
    run_bench("depthwise_s8 kws 49x40,1x10x8x8 dm8 S2 SCALAR (raw bounds-skip)",
              fill_kws, kern_scalar_kws, (const int8_t *)s_kws_accx,
              KWS_PC * 4);
    run_bench("depthwise_s8 kws 49x40,1x10x8x8 dm8 S2 BC1 PURE (broadcast kernel)",
              fill_kws_bc1, kern_bc1_kws_pure, (const int8_t *)s_kws_accx,
              KWS_PC * 4);
    run_bench("depthwise_s8 kws 49x40,1x10x8x8 dm8 S2 BC1 SCALAR (same data)",
              fill_kws_bc1, kern_scalar_kws_bc1, (const int8_t *)s_kws_accx,
              KWS_PC * 4);

    /* fc 256 -> 64 */
    run_bench("fc_s8 256row,64out TIE728-SIMD (dl_tie728_s8_conv2d_11cn)",
              fill_pattern_fc, fc_simd_entry, s_fc_out, FC_OUT_DIM);
    scalar_fc(s_ref64, s_fc_in, s_fc_w, s_fc_b);
    printf("  scalar-ref fnv1a=0x%08x  (Rust bench10: ref=0x32e35185 s3=0x16542aba)\n",
           (unsigned)fnv1a((const int8_t *)s_ref64, FC_OUT_DIM));

    /* --- BESPOKE ACCX fc256 (s8_accx_conv1x1.S): bit-exact GPR-accumulator --- */
    printf("== fc_s8 256row,64out BESPOKE-ACCX new ==\n");
    run_bench("fc_s8 256row,64out BESPOKE-ACCX new", fill_pattern_fc,
              kern_fc_new, s_fc_accx_out, FC_OUT_DIM);
    printf("  ref=0x32e35185\n");
    printf("== fc_s8 256row,64out BESPOKE-ACCX orig ==\n");
    run_bench("fc_s8 256row,64out BESPOKE-ACCX orig", fill_pattern_fc,
              kern_fc_orig, s_fc_accx_out, FC_OUT_DIM);
    printf("  ref=0x32e35185\n");
    printf("== fc_s8 256row,64out BESPOKE-ACCX new PURE ==\n");
    run_bench("fc_s8 256row,64out BESPOKE-ACCX new PURE", fill_pattern_fc,
              kern_fc_new_pure, (const int8_t *)s_fc_accx, FC_OUT_DIM * 4);
    printf("== fc_s8 256row,64out BESPOKE-ACCX orig PURE ==\n");
    run_bench("fc_s8 256row,64out BESPOKE-ACCX orig PURE", fill_pattern_fc,
              kern_fc_orig_pure, (const int8_t *)s_fc_accx, FC_OUT_DIM * 4);

    run_bench("fc_s8 640row,16out BESPOKE-ACCX PURE (in_c=640) HWLOOP", fill_pattern_fc,
              kern_a640_pure, (const int8_t *)s_a640_accx, A640_OUT_DIM * 4);
    run_bench("fc_s8 128row,16out BESPOKE-ACCX PURE (in_c=128) HWLOOP", fill_pattern_fc,
              kern_a128_pure, (const int8_t *)s_a128_accx, A128_OUT_DIM * 4);
    printf("== DROM vs SRAM weights (flash-latency hypothesis) ==\n");
    run_bench("fc_s8 640row,16out PURE DROM-w (in_c=640) HWLOOP", fill_pattern_fc,
              kern_a640_pure_drom, (const int8_t *)s_a640_accx, A640_OUT_DIM * 4);
    run_bench("fc_s8 128row,16out PURE DROM-w (in_c=128) HWLOOP", fill_pattern_fc,
              kern_a128_pure_drom, (const int8_t *)s_a128_accx, A128_OUT_DIM * 4);
    printf("== full 640x128 DROM stream (80KB, exact anomaly op0 footprint) ==\n");
    run_bench("fc_s8 640row,128out PURE DROM-w (in_c=640) HWLOOP", fill_pattern_fc,
              kern_a640b_pure_drom, (const int8_t *)s_a640b_accx, A640B_OUT_DIM * 4);

    /* max pool 2x2x16 */
    run_bench("max_pool_s8 2x2x16 TIE728-SIMD (dl_tie728_s8_max_pool2d_22c1)",
              fill_pattern_pool, maxpool_simd_entry, s_p_out, P_OUT_LEN);
    scalar_max_pool2d(s_ref, s_p_in);
    printf("  scalar-ref fnv1a=0x%08x  (Rust bench10: ref=0x651bfdc5 s3=0x50d8f9c5)\n",
           (unsigned)fnv1a((const int8_t *)s_ref, P_OUT_LEN));

    /* avg pool 2x2x16 */
    run_bench("avg_pool_s8 2x2x16 TIE728-SIMD (dl_tie728_s8_avg_pool2d_22c1)",
              fill_pattern_pool, avgpool_simd_entry, s_p_out, P_OUT_LEN);
    scalar_avg_pool2d(s_ref, s_p_in);
    printf("  scalar-ref fnv1a=0x%08x  (Rust bench10: ref=0xb8a6ddc5 s3=0xdedd2dc5)\n",
           (unsigned)fnv1a((const int8_t *)s_ref, P_OUT_LEN));

    /* relu 256 */
    run_bench("relu_s8 256 TIE728-SIMD (dl_tie728_s8_relu_11c)",
              fill_pattern_relu, relu_simd_entry, s_r_out, N256);
    scalar_relu(s_ref64, s_r_in);
    printf("  scalar-ref fnv1a=0x%08x  (Rust bench10: ref=0x6c620b3d s3=0x6c620b3d)\n",
           (unsigned)fnv1a((const int8_t *)s_ref64, N256));

    /* add 256 */
    run_bench("add_s8 256 TIE728-SIMD (dl_tie728_s8_add_w1_16_w2_16)",
              fill_pattern_elem, add_simd_entry, s_e_out, N256);
    scalar_add(s_ref64, s_e1, s_e2);
    printf("  scalar-ref fnv1a=0x%08x  (Rust bench10: ref=0x14834bbb s3=0x14834bbb)\n",
           (unsigned)fnv1a((const int8_t *)s_ref64, N256));

    /* sub 256 */
    run_bench("sub_s8 256 TIE728-SIMD (dl_tie728_s8_sub_w1_16_w2_16)",
              fill_pattern_elem, sub_simd_entry, s_e_out, N256);
    scalar_sub(s_ref64, s_e1, s_e2);
    printf("  scalar-ref fnv1a=0x%08x  (Rust bench10: ref=0x62d74671 s3=0x62d74671)\n",
           (unsigned)fnv1a((const int8_t *)s_ref64, N256));

    /* mul 256 */
    run_bench("mul_s8 256 TIE728-SIMD (dl_tie728_s8_mul_w1_16_w2_16)",
              fill_pattern_elem, mul_simd_entry, s_e_out, N256);
    scalar_mul(s_ref64, s_e1, s_e2);
    printf("  scalar-ref fnv1a=0x%08x  (Rust bench10: ref=0xd3c0a7f1 s3=0xd3c0a7f1)\n",
           (unsigned)fnv1a((const int8_t *)s_ref64, N256));

    printf("=== benchmark complete ===\n");
}
