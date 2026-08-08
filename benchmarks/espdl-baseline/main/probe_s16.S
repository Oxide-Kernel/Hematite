// probe_s16.S — S16 QACC probe (debugging only, NOT vendored ESP-DL code).
//
// Establishes, on real hardware, whether EE.VSMULAS.S16.QACC accumulates into
// 32-bit QACC lanes (unlike EE.VSMULAS.S8.QACC which we PROVED saturates at
// 8 bits/lane) and recovers the exact byte layout of a full 40-byte QACC
// store, so the bespoke S16 conv kernel can extract true 32-bit accumulators.
//
// Because each S16 group processes 8 output channels (8 lanes), and the S16
// kernel loads filter_v0 AND filter_v1 (16 int16 elements) per group, the
// probe works over 8 input elements with a 16-element filter buffer.

#include "dl_tie728_s16_conv2d.S"

    .text
    .global probe_s16_qacc
    .type   probe_s16_qacc, @function
    .align 4
// a2: const int16_t *input   (8 elements, 16-aligned)
// a3: const int16_t *filter  (16 elements, 16-aligned)
// a4: int32_t *out_qacc      (10 int32 = 40 bytes, 16-aligned)
// a5: int16_t *out_srcmb4    (8 int16)
// a6: int16_t *out_srcmb20   (8 int16)
// a7: int32_t  c_div_x_1     (input_channel/8 - 1)
probe_s16_qacc:
    entry sp, 16

    EE.ZERO.QACC
    tie728_s16_conv2d_11c8 q0, q1, q2, a2, a3, a7

    // full 40-byte QACC store (same sequence as the s8 per-channel macro)
    mov   a8, a4
    EE.ST.QACC_L.L.128.IP  a8, 16
    EE.ST.QACC_L.H.32.IP   a8, 16
    EE.ST.QACC_H.L.128.IP  a8, 16
    EE.ST.QACC_H.H.32.IP   a8, 4

    // SRCMB.S16 at byte shifts 4 and 20 (the two 16-bit halves)
    movi  a8, 4
    EE.SRCMB.S16.QACC q3, a8, 0
    EE.VST.128.IP     q3, a5, 16

    movi  a8, 20
    EE.SRCMB.S16.QACC q3, a8, 0
    EE.VST.128.IP     q3, a6, 16

    retw
    .size probe_s16_qacc, . - probe_s16_qacc

// Full per-channel-style 32-bit lane extraction probe. Mirrors the exact
// instruction sequence from tie728_s8_conv2d_per_channel_result in the
// vendored dl_tie728_s8.S (SRCMB@4, 40-byte QACC store, SRCMB@20, l16si
// rearrange, VZIP.16), so we can see whether true 32-bit lanes are
// recoverable from a VSMULAS.S16.QACC accumulation.
//
// a2: const int16_t *input   (8 elements, 16-aligned)
// a3: const int16_t *filter  (16 elements, 16-aligned)
// a4: int16_t *out16         (16 int16 = the 16 reconstructed halfwords)
// a5: int32_t  c_div_x_1
    .text
    .global probe_s16_perchan
    .type   probe_s16_perchan, @function
    .align 4
probe_s16_perchan:
    entry sp, 128

    EE.ZERO.QACC
    tie728_s16_conv2d_11c8 q0, q1, q2, a2, a3, a5

    // ---- exact per-channel extraction sequence ----
    movi a8, 4
    EE.SRCMB.S16.QACC q0, a8, 0              // "lower 16 bit"

    mov  a9, a1
    EE.ST.QACC_L.L.128.IP  a9, 16
    EE.ST.QACC_L.H.32.IP   a9, 16
    EE.ST.QACC_H.L.128.IP  a9, 16
    EE.ST.QACC_H.H.32.IP   a9, 4

    movi a8, 20
    EE.SRCMB.S16.QACC q1, a8, 0              // "even 16 bit"

    addi a8, a1, 5
    l16si a8, a8, 0
    s16i  a8, a1, 2
    l16si a8, a1, 10
    s16i  a8, a1, 4
    addi a8, a1, 15
    l16si a8, a8, 0
    s16i  a8, a1, 6
    l16si a8, a1, 32
    s16i  a8, a1, 8
    addi a8, a1, 37
    l16si a8, a8, 0
    s16i  a8, a1, 10
    l16si a8, a1, 42
    s16i  a8, a1, 12
    addi a8, a1, 47
    l16si a8, a8, 0
    s16i  a8, a1, 14

    EE.VLD.128.IP q0, a1, 0                  // q0 = rearranged "odd 16 bit"
    EE.VZIP.16 q0, q1                        // interleave odd/even

    EE.VST.128.IP q0, a4, 16                 // first 8 halfwords
    EE.VST.128.IP q1, a4, 16                 // next 8 halfwords

    retw
    .size probe_s16_perchan, . - probe_s16_perchan
