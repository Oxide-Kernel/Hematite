# s8_accx_depthwise.S — bespoke QACC per-lane depthwise kernel (bit-exact).
#
# SPDX-License-Identifier: Apache-2.0
#
# Depthwise CANNOT use EE.VMULAS.S8.ACCX (that is a dot-product REDUCTION into
# a single 32-bit accumulator). Depthwise needs PER-LANE accumulation: output
# channel oc only sees input channel oc. The primitive is
#   EE.VMULAS.S8.QACC qfilter, qinput
# which accumulates element-wise into 16 8-bit lanes of QACC (wide >= 16-bit
# lanes, empirically verified: 127*127*9 = 145161 = 0x23709 survives).
#
# QACC read-back (Phase 0, verified on device with probe_qacc_esplike):
# the 40-byte ST.QACC store is a DIAGONAL layout that cannot be read cleanly.
# The exact esp-nn two-pass reconstruction IS the correct read-back:
#   1. gapped 20B store: st.qacc_l.l.128.ip 16; st.qacc_l.h.32.ip -16
#      -> a1+0..19 = QACC_L words w0..w4 (w_i = l32i a1+4*i)
#   2. q0 = [sext(w0,19), ((w1<<24)>>12)|((w0>>20)&0xFFF), (w1<<4)>>12,
#            ((w2<<16)>>12)|((w1>>28)&0xF)]
#   3. q3 = [((w3<<28)>>12)|((w2>>16)&0xFFFF), (w3<<8)>>12,
#            ((w4<<20)>>12)|((w3>>24)&0xFF), w4>>12]
#   4. st.qacc_h.l.128.ip 16; st.qacc_h.h.32.ip -16 OVERWRITES a1+0..19
#   5. q1 = [sext(w0,19), ((w1<<24)>>12)|((w0>>20)&0xFFF), (w1<<4)>>12,
#            ((w2<<16)>>12)|((w1>>28)&0xF)]   (words now QACC_H)
#   6. q6 = [((w3<<28)>>12)|((w2>>16)&0xFFFF), (w3<<8)>>12,
#            ((w4<<20)>>12)|((w3>>24)&0xFF), w4>>12]
# Verified on device: lanes = 2,4,6,...,32 for V=(k+1)*2 1 MAC.
#
# Restriction: 3x3, stride 1, pad 0, depth_multiplier 1, channels % 16 == 0.
# One output pixel per call; the Rust caller loops over output pixels
# (oh, ow), passing the top-left window address of each pixel.
#   ABI (windowed, 6 args — caller a10..a15 = callee a2..a7):
#     s8_accx_depthwise(input, filter, acc_out, in_c, out_c, row_delta)
#       a2=input (pixel top-left tap, NHWC), a3=filter (HWCN dm=1,
#         [1,3,3,out_c] = filter[(tap)*out_c + ch], tap=ky*3+kx),
#       a4=acc_out (out_c int32), a5=in_c, a6=out_c, a7=row_delta
#   Precondition: in_c % 16 == 0, out_c % 16 == 0, out_c >= 16,
#       input/filter 16-byte aligned, acc_out 4-byte aligned.

    .align 4
    .text
    .global s8_accx_depthwise
    .type   s8_accx_depthwise, @function

s8_accx_depthwise:
    entry a1, 128

    srai a8, a6, 4              # num_groups = out_c / 16
    beqz a8, .Ldw_done

    movi a9, 0                  # group index
    s32i a8, a1, 20             # save num_groups (readback clobbers a8/a9)
    s32i a9, a1, 24             # save group index
    s32i a6, a1, 28             # save out_c (readback clobbers a6)
    s32i a5, a1, 32             # save in_c (readback clobbers a5)

.Ldw_group:
    mov a10, a2
    add a10, a10, a9            # input base for group = pixel + g*16
    mov a11, a3
    add a11, a11, a9            # filter base for group = filter + g*16
    movi a12, 9                 # taps remaining
    movi a13, 0                 # col within row (0..2)

    EE.ZERO.QACC

.Ldw_tap:
    beqz a12, .Ldw_extract
    EE.VLD.128.XP q0, a10, a5   # input vector (16 ch), +in_c
    EE.VLD.128.XP q1, a11, a6   # filter vector (16 ch), +out_c
    EE.VMULAS.S8.QACC q0, q1    # per-lane MAC
    addi a12, a12, -1
    addi a13, a13, 1
    beqi a13, 3, .Ldw_rowadv
    j .Ldw_tap
.Ldw_rowadv:
    movi a13, 0
    add a10, a10, a7            # row_delta = (in_w - 3) * in_c
    j .Ldw_tap

.Ldw_extract:
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

    # store 16x i32 to acc_out + g*16
    ee.vst.128.ip q0, a4, 16
    ee.vst.128.ip q3, a4, 16
    ee.vst.128.ip q1, a4, 16
    ee.vst.128.ip q6, a4, 16

    # advance group
    l32i a9, a1, 24
    addi a9, a9, 16
    s32i a9, a1, 24
    l32i a8, a1, 20
    l32i a5, a1, 32             # restore in_c
    l32i a6, a1, 28             # restore out_c
    bge a9, a6, .Ldw_done       # exit when g*16 >= out_c (short fwd branch)
    j .Ldw_group                # loop back (long jump; blt 8-bit range fails
                                # because the extraction block is ~180 bytes)

.Ldw_done:
    retw
    .size   s8_accx_depthwise, . - s8_accx_depthwise
