# s8_accx_depthwise_anytap_bc1.S — single-channel broadcast depthwise kernel.
#
# SPDX-License-Identifier: Apache-2.0
#
# Fast path for the common in_c == 1 depthwise layer (e.g. the kws
# [1,49,40,1] -> [1,25,20,8] dm=8 10x8 SAME stride2 model): when the input has
# ONE channel, the padded-input staging used by `s8_accx_depthwise_anytap.S`
# would replicate that single channel into padded_c = 16 lanes for EVERY tap
# (padded_h * padded_w * 16 bytes — 42.7 KB for the kws shape). Instead we
# stage a SINGLE-CHANNEL padded input (padded_h * padded_w bytes, ~2.7 KB) and
# broadcast each input byte to all 16 lanes with EE.VLDBC.8 — the exact
# broadcast primitive the vendored dl_tie728_s8_avg_pool2d.S uses to load
# avg_pool_area_inv. This eliminates ~94% of the staging traffic AND makes the
# kernel fit in small scratch.
#
# Everything else is identical to `s8_accx_depthwise_anytap.S`: same chunked
# ≤32-tap QACC passes (20-bit signed lanes), same two-pass 40-byte QACC
# read-back into 16 i32 lanes, same per-group store. One output pixel per call.
#
# KEY DIFFERENCE vs the anytap kernel: for in_c == 1 every output lane reads
# the SAME input byte per tap, so the input pointer does NOT advance by g*16
# between output groups — it stays at the tap base. The filter (staged as
# [tap][padded_c]) DOES advance by g*16 per group.
#
# ABI (windowed, 1 arg — a struct pointer; same multi-arg-miscompile
# discipline as AnyTapCtx):
#   s8_accx_depthwise_anytap_bc1(&Bc1Ctx)
#     a10 = &Bc1Ctx  (callee a2)
#
# Bc1Ctx layout (repr(C), 32-bit Xtensa; must match accx.rs):
#    0  input      ptr   single-channel padded input; starting tap address for
#                        this pass (Rust pre-adds the row/col tap offset)
#    4  filter     ptr   padded filter + tap_start*padded_c  ([tap][padded_c])
#    8  acc_out    ptr   16-aligned i32 partial-accumulator buffer
#                        (padded_c lanes; kernel WRITES, never adds)
#    12 in_c       u32   MUST be 1 (input byte advance per tap)
#    16 out_c      u32   padded_c; % 16 == 0, >= 16; filter tap stride
#    20 row_delta  u32   (padded_w - filter_w) * 1  (single-channel)
#    24 taps       u32   taps in THIS pass (1..=32)
#    28 filter_w   u32   taps per row (the row-advance period)
#    32 col_start  u32   starting column within the row (0..filter_w-1)
#
#   Preconditions: filter 16-byte aligned, acc_out 16-byte aligned, input
#   byte-addressable (no alignment requirement), all buffers sized.

    .align 4
    .text
    .global s8_accx_depthwise_anytap_bc1
    .type   s8_accx_depthwise_anytap_bc1, @function

s8_accx_depthwise_anytap_bc1:
    entry a1, 128

    movi a9, 0
    s32i a9, a1, 20             # group index (bytes: g*16), init 0

.Lbc1_group:
    l32i a3, a2, 0              # input base (NO +g*16 — single channel)
    l32i a4, a2, 4              # filter base
    l32i a9, a1, 20
    add a4, a4, a9              # filter + g*16
    l32i a5, a2, 8              # acc_out base
    slli a10, a9, 2
    add a5, a5, a10             # acc_out + g*64
    l32i a7, a2, 16             # out_c (filter VLD delta + group-exit bound)
    l32i a8, a2, 20             # row_delta
    l32i a9, a2, 24             # taps in this pass
    l32i a10, a2, 28            # filter_w (row-advance period)
    l32i a11, a2, 32            # col_start

    EE.ZERO.QACC

.Lbc1_tap:
    beqz a9, .Lbc1_extract
    EE.VLDBC.8 q0, a3           # broadcast 1 input byte to 16 lanes
    addi a3, a3, 1              # advance 1 byte (in_c == 1)
    EE.VLD.128.XP q1, a4, a7    # filter vector (16 ch), +out_c
    EE.VMULAS.S8.QACC q0, q1    # per-lane MAC
    addi a9, a9, -1
    addi a11, a11, 1
    bne a11, a10, .Lbc1_tap
    movi a11, 0
    add a3, a3, a8              # input += row_delta (next filter row)
    j .Lbc1_tap

.Lbc1_extract:
    # ---- Pass1: QACC_L -> q0 (lanes 0-3) and q3 (lanes 4-7) ----
    mov a12, a1
    ee.st.qacc_l.l.128.ip a12, 16
    ee.st.qacc_l.h.32.ip  a12, -16

    l32i.n a9, a1, 8            # w2
    l32i.n a11, a1, 4           # w1
    l32i.n a15, a1, 0           # w0
    slli a14, a11, 24
    sext  a8, a15, 19
    slli a10, a9, 16
    slli a13, a11, 4
    extui a9, a9, 16, 16
    srai a13, a13, 12
    extui a15, a15, 20, 12
    srai a14, a14, 12
    srai a10, a10, 12
    extui a11, a11, 28, 4
    or a10, a10, a11
    or a14, a14, a15
    ee.movi.32.q q0, a8, 0
    ee.movi.32.q q0, a14, 1
    ee.movi.32.q q0, a13, 2
    ee.movi.32.q q0, a10, 3

    l32i.n a11, a1, 16          # w4
    l32i.n a14, a1, 12          # w3
    slli a13, a11, 20
    ee.st.qacc_h.l.128.ip a12, 16
    ee.st.qacc_h.h.32.ip  a12, -16
    srai a11, a11, 12
    srai a13, a13, 12
    slli a8, a14, 28
    slli a15, a14, 8
    srai a15, a15, 12
    srai a8, a8, 12
    l32i.n a12, a1, 8           # w2 (re-read AFTER QACC_H store)
    or a8, a8, a9               # a9 = (w2>>16)&0xFFFF from Pass1
    extui a14, a14, 24, 8
    l32i.n a9, a1, 0            # w0 (re-read AFTER QACC_H store)
    or a13, a13, a14
    ee.movi.32.q q3, a8, 0
    ee.movi.32.q q3, a15, 1
    ee.movi.32.q q3, a13, 2
    ee.movi.32.q q3, a11, 3

    # ---- Pass2: QACC_H now in a1+0..19 -> q1 (lanes 8-11), q6 (12-15) ----
    l32i.n a14, a1, 4           # w1 (QACC_H)
    sext  a10, a9, 19           # a9 = w0 (QACC_H)
    extui a9, a9, 20, 12
    slli a13, a12, 16           # a12 = w2 (QACC_H)
    slli a8, a14, 24
    extui a12, a12, 16, 16
    srai a13, a13, 12
    srai a8, a8, 12
    slli a15, a14, 4
    srai a15, a15, 12
    or a8, a8, a9
    extui a14, a14, 28, 4
    l32i.n a9, a1, 12           # w3 (QACC_H)
    or a13, a13, a14
    ee.movi.32.q q1, a10, 0
    ee.movi.32.q q1, a8, 1
    ee.movi.32.q q1, a15, 2
    ee.movi.32.q q1, a13, 3

    l32i.n a15, a1, 16          # w4 (QACC_H)
    slli a11, a9, 28
    slli a10, a9, 8
    srai a10, a10, 12
    srai a11, a11, 12
    extui a9, a9, 24, 8
    or a11, a11, a12            # a12 = (w2>>16)&0xFFFF (QACC_H)
    slli a8, a15, 20
    srai a15, a15, 12
    srai a8, a8, 12
    or a8, a8, a9
    ee.movi.32.q q6, a11, 0
    ee.movi.32.q q6, a10, 1
    ee.movi.32.q q6, a8, 2
    ee.movi.32.q q6, a15, 3

    # store 16x i32 partial to acc_out + g*64
    ee.vst.128.ip q0, a5, 16
    ee.vst.128.ip q3, a5, 16
    ee.vst.128.ip q1, a5, 16
    ee.vst.128.ip q6, a5, 16

    # advance group (filter-only; input stays at the tap base)
    l32i a9, a1, 20
    addi a9, a9, 16
    s32i a9, a1, 20
    bge a9, a7, .Lbc1_done
    j .Lbc1_group

.Lbc1_done:
    retw
    .size   s8_accx_depthwise_anytap_bc1, . - s8_accx_depthwise_anytap_bc1
