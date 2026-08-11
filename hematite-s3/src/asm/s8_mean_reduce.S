# s8_mean_reduce.S — bespoke QACC per-lane MEAN-reduction kernel (bit-exact).
#
# SPDX-License-Identifier: Apache-2.0
#
# MEAN over the spatial axes (H,W) of an NHWC int8 tensor, keeping channels:
#   acc[oc] = sum(pos=0..positions-1) input[pos*in_c + oc]
# followed by the bit-exact Rust epilogue (round-half-away division by
# positions, per-tensor requantize). This is the shape MobileNetV2's global
# average pool (a MEAN) uses.
#
# Per-lane accumulation uses EE.VMULAS.S8.QACC q0, q1 — the element-wise
# lane accumulate (q1 = 16x 0x01 ones), the SAME primitive the depthwise
# kernel uses. The wide QACC lanes (>= 16-bit) bound `positions`: the
# accumulated lane sum is at most 127*positions, so positions must be
# < 258 (the caller gates on positions <= 256). QACC read-back is the
# verified two-pass diagonal reconstruction (Phase 0, see s8_accx_depthwise.S).
#
# Restriction: in_c % 16 == 0 (the caller zero-pads the channel dimension up
# to 16 in scratch when in_c % 16 != 0), in_c >= 16, positions >= 1,
# input/ones/acc_out 16-byte aligned (EE.VLD.128 / EE.VST.128 require it).
#
# ABI (windowed, 5 args — caller a10..a14 = callee a2..a6):
#   s8_mean_reduce(input, ones, acc_out, positions, in_c)
#     a2=input (flattened [positions, in_c], 16-aligned)
#     a3=ones (16 bytes 0x01, 16-aligned)
#     a4=acc_out (in_c int32, 16-aligned; advances 4*16 bytes per group)
#     a5=positions (number of spatial positions summed)
#     a6=in_c (multiple of 16, >= 16)
#   Clobbers a8..a15 + q0/q1/q3/q6; preserves a2/a3/a4.

    .align 4
    .text
    .global s8_mean_reduce
    .type   s8_mean_reduce, @function

s8_mean_reduce:
    entry a1, 128

    srai a8, a6, 4              # num_groups = in_c / 16
    beqz a8, .Lmr_done

    movi a9, 0                  # group byte offset
    s32i a8, a1, 20             # save num_groups (extract clobbers a8/a9)
    s32i a9, a1, 24             # save group offset
    s32i a6, a1, 28             # save in_c (extract clobbers a6)
    s32i a5, a1, 32             # save positions (extract clobbers a5)

.Lmr_group:
    mov a10, a2
    add a10, a10, a9            # input base for group = input + g*16
    mov a12, a5                 # positions remaining
    EE.ZERO.QACC

.Lmr_pos:
    beqz a12, .Lmr_extract
    mov a11, a3
    EE.VLD.128.IP q1, a11, 16   # q1 = ones (16x 0x01)
    EE.VLD.128.XP q0, a10, a6   # input 16 lanes, advance in_c (XP = reg)
    EE.VMULAS.S8.QACC q0, q1    # lane-accumulate input (filter = ones)
    addi a12, a12, -1
    j .Lmr_pos

    # ---- Pass1: QACC_L -> q0 (lanes 0-3) and q3 (lanes 4-7) ----
.Lmr_extract:
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

    # store 16x i32 to acc_out + g*16 (a4 advances 64 bytes per group)
    ee.vst.128.ip q0, a4, 16
    ee.vst.128.ip q3, a4, 16
    ee.vst.128.ip q1, a4, 16
    ee.vst.128.ip q6, a4, 16

    # advance group
    l32i a9, a1, 24
    addi a9, a9, 16
    s32i a9, a1, 24
    l32i a8, a1, 20
    l32i a5, a1, 32             # restore positions
    l32i a6, a1, 28             # restore in_c
    bge a9, a6, .Lmr_done       # exit when g*16 >= in_c (short fwd branch)
    j .Lmr_group                # loop back (extract is ~180 bytes, so the
                                # backward branch cannot be an 8-bit blt)

.Lmr_done:
    retw
    .size   s8_mean_reduce, . - s8_mean_reduce
