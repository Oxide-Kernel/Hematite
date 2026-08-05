/* SPDX-License-Identifier: Apache-2.0
 *
 * QEMU C baseline harness for the hematite benchmark suite.
 *
 * Establishes a C baseline for the four ember-esp-nn kernel rows from
 * `hematite-benchmarks/src/spec.rs`:
 *
 *   conv_s8           8x8, 64x3x3x3   (input [1,8,8,3]  -> out [1,8,8,64])
 *   depthwise_conv_s8 18x18, 1x3x3x16 (input [1,18,18,16] -> out [1,18,18,16])
 *   fc_s8             271 -> 3
 *   conv1x1_s8        64x1x1x64       (input [1,1,1,64] -> out [1,1,1,64])
 *
 * Timing methodology mirrors `hematite-benchmarks/src/timing.rs`
 * (`run_repeated`): one untimed warm-up run, then N >= 10 timed runs
 * (clamped to 10..64), min + median over the timed runs (first run never a
 * data point), CCOUNT 32-bit wrapping deltas, integer-only
 * `cycles * 1000 / 240_000_000` ms conversion.  Buffer fill happens ONCE
 * before timing; the timed closure is only the kernel call (same as the
 * firmware's `bench_kernel`).
 *
 * EVERY timing printed here is a QEMU-EMULATION number — it is an emulated
 * cycle count, NOT a hardware measurement.
 */
#include "kernels.h"
#include "uart.h"

#include <stdint.h>

#define CPU_HZ 240000000ULL /* locked CPU clock the bench profile assumes */
#define WARMUP_RUNS 1u      /* mirror BenchmarkConfig::default() */
#define TIMED_RUNS 10u      /* mirror BenchmarkConfig::default() (>= 10) */
#define MAX_RUNS 64u        /* mirror RunLog::MAX_RUNS */

/* ── CCOUNT (Xtensa cycle counter, 32-bit) ───────────────────────────────── */

static inline uint32_t read_ccount(void)
{
    uint32_t c;
    __asm__ __volatile__("rsr.ccount %0" : "=r"(c) : : "memory");
    return c;
}

/* 32-bit wrapping delta (mirror run_repeated's
 * `(c1.wrapping_sub(c0)) & 0xFFFF_FFFF`). */
static inline uint32_t wrap_delta(uint32_t c1, uint32_t c0)
{
    return (uint32_t)(c1 - c0);
}

/* Integer cycle->time conversion, mirroring timing.rs:
 *   cycles_to_us = cycles * 1_000_000 / 240_000_000 == cycles / 240
 *   cycles_to_ms = cycles * 1000      / 240_000_000 == cycles / 240_000
 * The divided forms are exact integer equivalents (no u64 division, which
 * the freestanding multilib does not provide). */
static uint32_t cycles_to_us(uint32_t cycles)
{
    return cycles / 240u;
}

static uint32_t cycles_to_ms(uint32_t cycles)
{
    return cycles / 240000u;
}

/* ── deterministic fill (mirror spec.rs fill_pattern) ────────────────────── */

static void fill_pattern(int8_t *input, int32_t in_len, int8_t *weights,
                         int32_t w_len, int32_t *bias, int32_t b_len,
                         int8_t *output, int32_t out_len)
{
    for (int32_t i = 0; i < in_len; i++) {
        input[i] = (int8_t)((uint8_t)((i * 7 + 3) & 0xFF));
    }
    for (int32_t i = 0; i < w_len; i++) {
        weights[i] = (int8_t)((uint8_t)((i * 13 + 11) & 0xFF));
    }
    for (int32_t i = 0; i < b_len; i++) {
        bias[i] = i * 17 - 8;
    }
    for (int32_t i = 0; i < out_len; i++) {
        output[i] = 0;
    }
}

/* FNV-1a checksum of the output tensor: proves the kernel actually ran on
 * the filled buffers (deterministic evidence, not a timing number). */
static uint32_t out_checksum(const int8_t *out, int32_t n)
{
    uint32_t h = 2166136261u;
    for (int32_t i = 0; i < n; i++) {
        h ^= (uint8_t)out[i];
        h *= 16777619u;
    }
    return h;
}

/* ── run_repeated analog ─────────────────────────────────────────────────── */

static uint32_t g_checksum; /* set by the bench closure */

static void run_bench(const char *name, void (*fn)(void))
{
    uint32_t cycles[MAX_RUNS];

    /* Warm-up (untimed): one pass warms i-cache/d-cache (C3). */
    for (uint32_t i = 0; i < WARMUP_RUNS; i++) {
        fn();
    }

    /* Methodology floor: N >= 10 timed runs (clamp 10..MAX_RUNS). */
    uint32_t n = TIMED_RUNS;
    if (n < 10) {
        n = 10;
    }
    if (n > MAX_RUNS) {
        n = MAX_RUNS;
    }

    for (uint32_t i = 0; i < n; i++) {
        uint32_t c0 = read_ccount();
        fn();
        uint32_t c1 = read_ccount();
        cycles[i] = (uint32_t)(c1 - c0); /* wrapping, widened later */
    }

    /* Sort ascending (insertion sort; n <= 64). */
    for (uint32_t i = 1; i < n; i++) {
        uint32_t key = cycles[i];
        int32_t j = (int32_t)i - 1;
        while (j >= 0 && cycles[j] > key) {
            cycles[j + 1] = cycles[j];
            j--;
        }
        cycles[j + 1] = key;
    }

    uint32_t min_c = cycles[0];
    uint32_t med_c;
    if (n % 2 == 1) {
        med_c = cycles[n / 2];
    } else {
        /* upper-middle for even lengths, integer math (mirror timing.rs) */
        med_c = (uint32_t)(((uint64_t)cycles[n / 2 - 1] + (uint64_t)cycles[n / 2]) >> 1);
    }

    uart_puts("\r\n== ");
    uart_puts(name);
    uart_puts(" ==\r\n");
    uart_puts("  QEMU-EMULATION | N=");
    uart_dec32(n);
    uart_puts(" timed + ");
    uart_dec32(WARMUP_RUNS);
    uart_puts(" untimed warm-up\r\n");
    uart_puts("  min    cycles=");
    uart_hex32(min_c);
    uart_puts(" us@240MHz=");
    uart_dec32(cycles_to_us(min_c));
    uart_puts(" ms@240MHz=");
    uart_dec32(cycles_to_ms(min_c));
    uart_puts("\r\n");
    uart_puts("  median cycles=");
    uart_hex32(med_c);
    uart_puts(" us@240MHz=");
    uart_dec32(cycles_to_us(med_c));
    uart_puts(" ms@240MHz=");
    uart_dec32(cycles_to_ms(med_c));
    uart_puts("\r\n");
    uart_puts("  out_checksum(fnv1a)=");
    uart_hex32(g_checksum);
    uart_puts("\r\n");
}

/* ── per-kernel buffers + params (ember-esp-nn rows from spec.rs) ─────────── */

/* conv 8x8: in [1,8,8,3]=192, w [64,3,3,3]=1728, bias 64, out [1,8,8,64]=4096 */
static int8_t c8_in[192];
static int8_t c8_w[1728];
static int32_t c8_b[64];
static int8_t c8_out[4096];
static ConvParams c8_p;

static void bench_conv8(void)
{
    conv2d(c8_in, c8_w, c8_b, &c8_p, c8_out);
    g_checksum = out_checksum(c8_out, 4096);
}

/* depthwise 18x18: in [1,18,18,16]=5184, w [1,3,3,16]=144, bias 16,
 * out [1,18,18,16]=5184 */
static int8_t dw_in[5184];
static int8_t dw_w[144];
static int32_t dw_b[16];
static int8_t dw_out[5184];
static DepthwiseParams dw_p;

static void bench_depthwise(void)
{
    depthwise_conv2d(dw_in, dw_w, dw_b, &dw_p, dw_out);
    g_checksum = out_checksum(dw_out, 5184);
}

/* fc 271->3: in 271, w 271*3=813, bias 3, out 3 */
static int8_t fc_in[271];
static int8_t fc_w[813];
static int32_t fc_b[3];
static int8_t fc_out[3];
static FcParams fc_p;

static void bench_fc(void)
{
    fully_connected(fc_in, fc_w, fc_b, &fc_p, fc_out);
    g_checksum = out_checksum(fc_out, 3);
}

/* conv1x1 64: in [1,1,1,64]=64, w [64,1,1,64]=4096, bias 64, out [1,1,1,64]=64 */
static int8_t c1_in[64];
static int8_t c1_w[4096];
static int32_t c1_b[64];
static int8_t c1_out[64];
static ConvParams c1_p;

static void bench_conv1x1(void)
{
    conv2d(c1_in, c1_w, c1_b, &c1_p, c1_out);
    g_checksum = out_checksum(c1_out, 64);
}

/* ── per-channel requantize constants ─────────────────────────────────────── */

/* Non-empty .data: forces the DRAM load segment to carry file content at
 * its VMA (0x3FC88000). With .data empty, espflash emits the DRAM segment
 * with vaddr 0 and the bootloader hangs on it. Also doubles as a boot
 * self-test: main() prints this value — if the bootloader failed to copy
 * .data into DRAM it reads 0 instead of 0x0BADC0DE. */
static volatile int32_t boot_marker = 0x0BADC0DE;

static int32_t mult3[3];
static int32_t shift3[3];
static int32_t mult16[16];
static int32_t shift16[16];
static int32_t mult64[64];
static int32_t shift64[64];

static void init_quant_consts(int32_t *mult, int32_t *shift, int32_t n)
{
    for (int32_t i = 0; i < n; i++) {
        mult[i] = 1 << 30; /* Q0.31 0.5-pair used by the spec rows */
        shift[i] = 0;
    }
}

/* ── entry point ─────────────────────────────────────────────────────────── */

void main(void)
{
    uart_puts("\r\n=== Hematite C QEMU baseline (tools/qemu-baseline) ===\r\n");
    uart_puts("LABEL: QEMU emulation smoke - these are NOT hardware measurements\r\n");
    uart_puts("Clock: Xtensa CCOUNT under QEMU (-icount 3); ms@240MHz is the\r\n");
    uart_puts("       arithmetic conversion cycles*1000/240000000 (integer)\r\n");
    uart_puts("boot_marker (.data copy check)=");
    uart_hex32((uint32_t)boot_marker);
    uart_puts("\r\n\r\n");

    init_quant_consts(mult3, shift3, 3);
    init_quant_consts(mult16, shift16, 16);
    init_quant_consts(mult64, shift64, 64);

    /* conv 8x8 params (EMBER_CONV_8X8_PARAMS from spec.rs) */
    {
        static const int32_t in_shape[4] = {1, 8, 8, 3};
        static const int32_t f_shape[4] = {64, 3, 3, 3};
        static const int32_t o_shape[4] = {1, 8, 8, 64};
        for (int i = 0; i < 4; i++) {
            c8_p.input_shape[i] = in_shape[i];
            c8_p.filter_shape[i] = f_shape[i];
            c8_p.output_shape[i] = o_shape[i];
        }
        c8_p.stride_height = 1;
        c8_p.stride_width = 1;
        c8_p.dilation_height_factor = 1;
        c8_p.dilation_width_factor = 1;
        c8_p.input_offset = 0;
        c8_p.output_offset = 0;
        c8_p.output_multiplier_per_channel = mult64;
        c8_p.output_shift_per_channel = shift64;
        c8_p.quantized_activation_min = -128;
        c8_p.quantized_activation_max = 127;
    }

    /* depthwise 18x18 params (EMBER_DEPTHWISE_18X18_PARAMS) */
    {
        static const int32_t in_shape[4] = {1, 18, 18, 16};
        static const int32_t f_shape[4] = {1, 3, 3, 16};
        static const int32_t o_shape[4] = {1, 18, 18, 16};
        for (int i = 0; i < 4; i++) {
            dw_p.input_shape[i] = in_shape[i];
            dw_p.filter_shape[i] = f_shape[i];
            dw_p.output_shape[i] = o_shape[i];
        }
        dw_p.depth_multiplier = 1;
        dw_p.stride_height = 1;
        dw_p.stride_width = 1;
        dw_p.dilation_height_factor = 1;
        dw_p.dilation_width_factor = 1;
        dw_p.input_offset = 0;
        dw_p.output_offset = 0;
        dw_p.output_multiplier_per_channel = mult16;
        dw_p.output_shift_per_channel = shift16;
        dw_p.quantized_activation_min = -128;
        dw_p.quantized_activation_max = 127;
    }

    /* fc params (EMBER_FC_271_PARAMS) */
    {
        fc_p.input_dim = 271;
        fc_p.output_dim = 3;
        fc_p.input_offset = 0;
        fc_p.output_offset = 0;
        fc_p.output_multiplier_per_channel = mult3;
        fc_p.output_shift_per_channel = shift3;
        fc_p.quantized_activation_min = -128;
        fc_p.quantized_activation_max = 127;
    }

    /* conv1x1 params (EMBER_CONV_1X1_64_PARAMS) */
    {
        static const int32_t in_shape[4] = {1, 1, 1, 64};
        static const int32_t f_shape[4] = {64, 1, 1, 64};
        static const int32_t o_shape[4] = {1, 1, 1, 64};
        for (int i = 0; i < 4; i++) {
            c1_p.input_shape[i] = in_shape[i];
            c1_p.filter_shape[i] = f_shape[i];
            c1_p.output_shape[i] = o_shape[i];
        }
        c1_p.stride_height = 1;
        c1_p.stride_width = 1;
        c1_p.dilation_height_factor = 1;
        c1_p.dilation_width_factor = 1;
        c1_p.input_offset = 0;
        c1_p.output_offset = 0;
        c1_p.output_multiplier_per_channel = mult64;
        c1_p.output_shift_per_channel = shift64;
        c1_p.quantized_activation_min = -128;
        c1_p.quantized_activation_max = 127;
    }

    /* fill once, then time only the kernel call (mirror bench_kernel) */
    fill_pattern(c8_in, 192, c8_w, 1728, c8_b, 64, c8_out, 4096);
    run_bench("conv_s8 8x8, 64x3x3x3 (ember-esp-nn)", bench_conv8);

    fill_pattern(dw_in, 5184, dw_w, 144, dw_b, 16, dw_out, 5184);
    run_bench("depthwise_conv_s8 18x18, 1x3x3x16 (ember-esp-nn)", bench_depthwise);

    fill_pattern(fc_in, 271, fc_w, 813, fc_b, 3, fc_out, 3);
    run_bench("fc_s8 271->3 (ember-esp-nn)", bench_fc);

    fill_pattern(c1_in, 64, c1_w, 4096, c1_b, 64, c1_out, 64);
    run_bench("conv1x1_s8 64x1x1x64 (ember-esp-nn 15.57x bar)", bench_conv1x1);

    uart_puts("\r\n=== benchmark complete - QEMU halt ===\r\n");
}
