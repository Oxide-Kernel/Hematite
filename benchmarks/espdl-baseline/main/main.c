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

/* S16 QACC probe (probe_s16.S): establishes whether VSMULAS.S16.QACC
 * accumulates into 32-bit lanes and reveals the full 40-byte QACC store
 * layout + the SRCMB.S16 shift-4/20 halves. */
extern void probe_s16_qacc(const int16_t *input, const int16_t *filter,
                           int32_t *out_qacc, int16_t *out_srcmb4,
                           int16_t *out_srcmb20, int32_t c_div_x_1);
extern void probe_s16_perchan(const int16_t *input, const int16_t *filter,
                              int16_t *out16, int32_t c_div_x_1);
extern void probe_accx(const int16_t *filter, const int16_t *input,
                       int32_t *accx_out);
extern void probe_accx_load(const int32_t *pattern, int32_t *accx_out);
extern void probe_accx_mac_sweep(const int16_t *filter, const int16_t *input,
                                 int32_t *out64);
extern void probe_s8accx(const int8_t *filter, const int8_t *input,
                         int32_t *out3);
extern void s8_accx_conv1x1(const int8_t *input, const int8_t *filter,
                            int32_t *acc_out, int32_t in_c, int32_t out_c);

/* Vendored S16 conv2d kernel (dl_tie728_s16_conv2d.S) — takes int16 buffers.
 * args: filter@48, mac_shift@64, bias@68, output_channel_div_8@96,
 *       c_div_x_1@100. */
typedef struct {
    uint8_t  _pad0[48];
    int16_t *filter;
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
} Tie728S16ConvArgs;

extern void dl_tie728_s16_conv2d_11cn(int16_t *output, const int16_t *input,
                                      Tie728S16ConvArgs *args);

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

static int8_t  s_p_in[P_IN_LEN] __attribute__((aligned(16)));
static int8_t  s_p_out[P_OUT_LEN] __attribute__((aligned(16)));

static int8_t  s_r_in[N256] __attribute__((aligned(16)));
static int8_t  s_r_out[N256] __attribute__((aligned(16)));

static int8_t  s_e1[N256] __attribute__((aligned(16)));
static int8_t  s_e2[N256] __attribute__((aligned(16)));
static int8_t  s_e_out[N256] __attribute__((aligned(16)));

static int32_t s_mult[OUT_C] __attribute__((aligned(16)));   /* = 1<<30 */
static int32_t s_shift[OUT_C] __attribute__((aligned(16)));  /* = 0 */

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

/* ---- S16 QACC probe (probe_s16.S) ----
 * input=[1]*8, filter=[1..8 repeated in a 16-element buffer], c_div_x_1=0:
 *   QACC[lane] = sum over 8 input pos of input[pos] * filter[lane]
 *              = 8 * filter[lane]  ->  lanes [8,16,24,32,40,48,56,64]
 * then a saturation probe with input=weights=127 over 1..4 chunks. */
static void probe_s16(void) {
    static int16_t in[8] __attribute__((aligned(16)));
    static int16_t filt[16] __attribute__((aligned(16)));
    static int32_t qacc_store[10] __attribute__((aligned(16)));
    static int16_t s4[8] __attribute__((aligned(16)));
    static int16_t s20[8] __attribute__((aligned(16)));
    static const int16_t ramp[8] = {1, 2, 3, 4, 5, 6, 7, 8};

    printf("-- S16 QACC probe (VSMULAS.S16.QACC) --\n");
    for (int i = 0; i < 8; i++) in[i] = 1;
    for (int i = 0; i < 8; i++) { filt[i] = ramp[i]; filt[i + 8] = ramp[i]; }
    probe_s16_qacc(in, filt, qacc_store, s4, s20, 0);
    printf("  QACC store (40 bytes):");
    for (int i = 0; i < 10; i++) printf(" %08x", (unsigned)qacc_store[i]);
    printf("\n");
    printf("  srcmb@4  =");
    for (int i = 0; i < 8; i++) printf(" %04x", (unsigned)(uint16_t)s4[i]);
    printf("\n  srcmb@20 =");
    for (int i = 0; i < 8; i++) printf(" %04x", (unsigned)(uint16_t)s20[i]);
    printf("\n  expect lanes 8,16,...,64 (input=1 filter=1..8) if 32-bit exact\n");

    for (int i = 0; i < 8; i++) { in[i] = 127; filt[i] = 127; filt[i + 8] = 127; }
    for (int cdiv = 0; cdiv <= 3; cdiv++) {
        probe_s16_qacc(in, filt, qacc_store, s4, s20, cdiv);
        printf("  127s c_div_x_1=%d (%d MACs/lane): QACC store =",
               cdiv, (cdiv + 1) * 8);
        for (int i = 0; i < 10; i++) printf(" %08x", (unsigned)qacc_store[i]);
        printf("\n");
    }

    /* DIRECT vendored S16 kernel call: mac_shift=0 = plain SRCMB.S16@shift0.
     * With input=[1]*8, filter=[1..8], 8 MACs -> lanes should be 8,16,...,64
     * if VSMULAS.S16.QACC accumulates correctly. With 127s, 8 MACs -> 129032
     * (low16 0xf808); 16+ MACs saturate if lanes are 16-bit. */
    printf("-- vendored dl_tie728_s16_conv2d_11cn direct --\n");
    static int16_t s16_out[8] __attribute__((aligned(16)));
    static int16_t s16_perchan[16] __attribute__((aligned(16)));
    Tie728S16ConvArgs sa;
    memset(&sa, 0, sizeof(sa));
    sa.mac_shift = 0;
    sa.bias = NULL;
    sa.activation_alpha = 0;
    sa.activation_shift = -1;
    sa.output_channel_div_8 = 1;
    sa.filter_channel_factor = NULL;
    for (int i = 0; i < 8; i++) in[i] = 1;
    for (int i = 0; i < 8; i++) { filt[i] = ramp[i]; filt[i + 8] = ramp[i]; }
    sa.filter = filt;
    sa.c_div_x_1 = 0;
    dl_tie728_s16_conv2d_11cn(s16_out, in, &sa);
    printf("  ramp in=1 filt=1..8 8MACs mac_shift=0 out =");
    for (int i = 0; i < 8; i++) printf(" %d", (int)s16_out[i]);
    printf("\n  (expect 8 16 24 32 40 48 56 64)\n");

    /* clean all-ones sweep: in=[1]*8 filt=[1]*16, c_div_x_1=0..3 -> 8,16,24,32 */
    for (int i = 0; i < 8; i++) in[i] = 1;
    for (int i = 0; i < 16; i++) filt[i] = 1;
    for (int cdiv = 0; cdiv <= 3; cdiv++) {
        sa.c_div_x_1 = cdiv;
        dl_tie728_s16_conv2d_11cn(s16_out, in, &sa);
        printf("  ones c_div_x_1=%d (%d MACs) out =", cdiv, (cdiv + 1) * 8);
        for (int i = 0; i < 8; i++) printf(" %d", (int)s16_out[i]);
        printf("\n");
    }
    for (int i = 0; i < 8; i++) { in[i] = 127; filt[i] = 127; filt[i + 8] = 127; }
    for (int cdiv = 0; cdiv <= 3; cdiv++) {
        sa.c_div_x_1 = cdiv;
        dl_tie728_s16_conv2d_11cn(s16_out, in, &sa);
        printf("  127s c_div_x_1=%d (%d MACs) out =", cdiv, (cdiv + 1) * 8);
        for (int i = 0; i < 8; i++) printf(" %d", (int)s16_out[i]);
        printf("\n");
    }

    /* FULL per-channel extraction: 16 halfwords out of q0+q1 after VZIP.16.
     * ramp (in=1 filt=1..8, 8 MACs): true accs 8..64. If S16 lanes are 32-bit
     * these should appear as the 16 halfwords (lo/hi pairs). */
    for (int i = 0; i < 8; i++) in[i] = 1;
    for (int i = 0; i < 8; i++) { filt[i] = ramp[i]; filt[i + 8] = ramp[i]; }
    probe_s16_perchan(in, filt, s16_perchan, 0);
    printf("  perchan ramp 8MACs 16 halfwords =");
    for (int i = 0; i < 16; i++) printf(" %d", (int)s16_perchan[i]);
    printf("\n  (true accs 8 16 24 32 40 48 56 64; hi halves 0)\n");
    for (int i = 0; i < 8; i++) { in[i] = 127; filt[i] = 127; filt[i + 8] = 127; }
    for (int cdiv = 0; cdiv <= 3; cdiv++) {
        probe_s16_perchan(in, filt, s16_perchan, cdiv);
        printf("  perchan 127s c_div_x_1=%d (%d MACs) 16 halfwords =", cdiv, (cdiv + 1) * 8);
        for (int i = 0; i < 16; i++) printf(" %d", (int)s16_perchan[i]);
        printf("\n");
    }
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
    probe_s16();
    probe_accx_run();
    probe_s8accx_run();
    static int8_t s_ref[C3_OUT_LEN] __attribute__((aligned(16)));
    static int8_t s_ref64[FC_OUT_DIM] __attribute__((aligned(16)));

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
    {
        static int32_t s_accx[OUT_C] __attribute__((aligned(16)));
        static int8_t s_accx_out[OUT_C] __attribute__((aligned(16)));
        fill_pattern_conv1x1_tw(); /* fills s_input + s_weights (raw) */
        uint32_t t0 = read_ccount();
        s8_accx_conv1x1(s_input, s_weights, s_accx, IN_C, OUT_C);
        uint32_t t1 = read_ccount();
        for (int oc = 0; oc < OUT_C; oc++)
            s_accx_out[oc] = sat_cast(req(s_accx[oc] + s_bias[oc], 1 << 30, 0));
        printf("== conv1x1_s8 64x1x1x64 BESPOKE-ACCX (s8_accx_conv1x1) ==\n");
        printf("  cycles=%u | out_checksum(fnv1a)=0x%08x (ref=0x0bea8225)\n",
               (unsigned)(t1 - t0),
               (unsigned)fnv1a((const int8_t *)s_accx_out, OUT_C));
        printf("  accx[0..7]=%d %d %d %d %d %d %d %d\n",
               (int)s_accx[0], (int)s_accx[1], (int)s_accx[2], (int)s_accx[3],
               (int)s_accx[4], (int)s_accx[5], (int)s_accx[6], (int)s_accx[7]);
        printf("  out[0..7]=%d %d %d %d %d %d %d %d (ref=%d %d %d %d %d %d %d %d)\n",
               (int)s_accx_out[0], (int)s_accx_out[1], (int)s_accx_out[2],
               (int)s_accx_out[3], (int)s_accx_out[4], (int)s_accx_out[5],
               (int)s_accx_out[6], (int)s_accx_out[7],
               (int)s_ref64[0], (int)s_ref64[1], (int)s_ref64[2],
               (int)s_ref64[3], (int)s_ref64[4], (int)s_ref64[5],
               (int)s_ref64[6], (int)s_ref64[7]);
    }

    /* conv3x3 32x32 64x3x3x64 */
    run_bench("conv3x3_s8 32x32,64x3x3x64 VALID TIE728-SIMD (dl_tie728_s8_conv2d_33cn)",
              fill_pattern_conv3x3, conv3x3_simd_entry, s_c3_out, C3_OUT_LEN);
    scalar_conv3x3(s_ref, s_c3_in, s_c3_w, s_c3_b);
    printf("  scalar-ref fnv1a=0x%08x  (Rust bench10: ref=0x0a181085 s3=0xd1a9b601)\n",
           (unsigned)fnv1a((const int8_t *)s_ref, C3_OUT_LEN));

    /* fc 256 -> 64 */
    run_bench("fc_s8 256row,64out TIE728-SIMD (dl_tie728_s8_conv2d_11cn)",
              fill_pattern_fc, fc_simd_entry, s_fc_out, FC_OUT_DIM);
    scalar_fc(s_ref64, s_fc_in, s_fc_w, s_fc_b);
    printf("  scalar-ref fnv1a=0x%08x  (Rust bench10: ref=0x32e35185 s3=0x16542aba)\n",
           (unsigned)fnv1a((const int8_t *)s_ref64, FC_OUT_DIM));

    /* --- BESPOKE ACCX fc256 (s8_accx_conv1x1.S): bit-exact GPR-accumulator --- */
    {
        static int32_t s_fc_accx[FC_OUT_DIM] __attribute__((aligned(16)));
        static int8_t s_fc_accx_out[FC_OUT_DIM] __attribute__((aligned(16)));
        fill_pattern_fc(); /* fills s_fc_in + s_fc_w (raw) */
        uint32_t t0 = read_ccount();
        s8_accx_conv1x1(s_fc_in, s_fc_w, s_fc_accx, FC_IN_DIM, FC_OUT_DIM);
        uint32_t t1 = read_ccount();
        for (int oc = 0; oc < FC_OUT_DIM; oc++)
            s_fc_accx_out[oc] = sat_cast(req(s_fc_accx[oc] + s_fc_b[oc], 1 << 30, 0));
        printf("== fc_s8 256row,64out BESPOKE-ACCX (s8_accx_conv1x1) ==\n");
        printf("  cycles=%u | out_checksum(fnv1a)=0x%08x (ref=0x32e35185)\n",
               (unsigned)(t1 - t0),
               (unsigned)fnv1a((const int8_t *)s_fc_accx_out, FC_OUT_DIM));
        printf("  accx[0..7]=%d %d %d %d %d %d %d %d\n",
               (int)s_fc_accx[0], (int)s_fc_accx[1], (int)s_fc_accx[2],
               (int)s_fc_accx[3], (int)s_fc_accx[4], (int)s_fc_accx[5],
               (int)s_fc_accx[6], (int)s_fc_accx[7]);
        printf("  out[0..7]=%d %d %d %d %d %d %d %d (ref=%d %d %d %d %d %d %d %d)\n",
               (int)s_fc_accx_out[0], (int)s_fc_accx_out[1], (int)s_fc_accx_out[2],
               (int)s_fc_accx_out[3], (int)s_fc_accx_out[4], (int)s_fc_accx_out[5],
               (int)s_fc_accx_out[6], (int)s_fc_accx_out[7],
               (int)s_ref64[0], (int)s_ref64[1], (int)s_ref64[2],
               (int)s_ref64[3], (int)s_ref64[4], (int)s_ref64[5],
               (int)s_ref64[6], (int)s_ref64[7]);
    }

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
