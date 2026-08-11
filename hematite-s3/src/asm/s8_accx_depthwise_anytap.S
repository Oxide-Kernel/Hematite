# s8_accx_depthwise_anytap.S — tap-parameterized per-lane QACC depthwise kernel.
#
# SPDX-License-Identifier: Apache-2.0
#
# The 3x3-only `s8_accx_depthwise.S` kernel is the silicon-proven ancestor; this
# kernel generalizes it to ARBITRARY filter sizes (T3.5b). Depthwise uses the
# per-lane primitive
#   EE.VMULAS.S8.QACC qfilter, qinput
# which accumulates element-wise into 16 8-bit lanes of QACC.
#
# QACC LANE-WIDTH CONSTRAINT (why the kernel is chunked):
# the from-silicon 40-byte store read-back carries 20 bits per lane
# (sext reg,imm19 — see probe_qacc.S / s8_accx_depthwise.S). Signed 20-bit
# lanes hold [-524288, +524287]. A single int8 product is up to
# 127*127 = 16129 (or -128*127 = -16256), so up to 32 taps per QACC pass is
# SAFE for arbitrary int8 data (32*16129 = 516128 < 524287;
# 32*-16256 = -520192 > -524288). A 33rd tap can overflow. The kernel
# therefore accumulates in CHUNKS of at most 32 taps: each call runs ONE
# 32-tap-or-less QACC pass and writes the exact 16-lane partial sums; the
# Rust caller adds the partials into its running i32 accumulators. Chunked
# sums are mathematically identical to a single 32-bit sum (i32 wrapping
# adds), so output stays bit-exact vs the scalar reference for ANY filter
# size (the kws 10x8 = 80-tap filter needs 3 passes).
#
# One output pixel per call; the Rust caller strides over the output image
# and over the 32-tap chunks.
#
# ABI (windowed, 1 arg — a struct pointer; the Xtensa LLVM multi-arg
# (9-slot) call miscompile class is avoided exactly like ReqCtx):
#   s8_accx_depthwise_anytap(&AnyTapCtx)
#     a10 = &AnyTapCtx  (callee a2)
#
# AnyTapCtx layout (repr(C), 32-bit Xtensa; must match accx.rs):
#    0  input      ptr   starting tap address for this pass (Rust pre-adds
#                        the row/col tap offset)
#    4  filter     ptr   filter + tap_start*out_c  (tap stride = out_c)
#    8  acc_out    ptr   16-aligned i32 partial-accumulator buffer
#                        (out_c lanes; kernel WRITES, never adds)
#    12 in_c       u32   input channel stride (padded_c)
#    16 out_c      u32   channel count + filter tap stride (padded_c);
#                        % 16 == 0, >= 16 (the dispatch pads)
#    20 row_delta  u32   (padded_w - filter_w) * padded_c
#    24 taps       u32   taps in THIS pass (1..=32)
#    28 filter_w   u32   taps per row (the row-advance period)
#    32 col_start  u32   starting column within the row (0..filter_w-1)
#
#   Preconditions: input/filter 16-byte aligned, acc_out 16-byte aligned,
#   all buffers sized.

    .align 4
    .text
    .global s8_accx_depthwise_anytap
    .type   s8_accx_depthwise_anytap, @function

s8_accx_depthwise_anytap:
    entry a1, 128

    movi a9, 0
    s32i a9, a1, 20             # group index (bytes: g*16), init 0

.Ldwa_group:
    l32i a3, a2, 0              # input base
    l32i a9, a1, 20
    add a3, a3, a9              # input + g*16
    l32i a4, a2, 4              # filter base
    add a4, a4, a9              # filter + g*16
    l32i a5, a2, 8              # acc_out base
    slli a10, a9, 2
    add a5, a5, a10             # acc_out + g*64 (g*16 lanes * 4 bytes)
    l32i a6, a2, 12             # in_c  (input VLD delta)
    l32i a7, a2, 16             # out_c (filter VLD delta + group-exit bound)
    l32i a8, a2, 20             # row_delta
    l32i a9, a2, 24             # taps in this pass
    l32i a10, a2, 28            # filter_w (row-advance period)
    l32i a11, a2, 32            # col_start

    EE.ZERO.QACC

.Ldwa_tap:
    beqz a9, .Ldwa_extract
    EE.VLD.128.XP q0, a3, a6    # input vector (16 ch), +in_c
    EE.VLD.128.XP q1, a4, a7    # filter vector (16 ch), +out_c
    EE.VMULAS.S8.QACC q0, q1    # per-lane MAC
    addi a9, a9, -1
    addi a11, a11, 1
    bne a11, a10, .Ldwa_tap
    movi a11, 0
    add a3, a3, a8              # input += row_delta (next filter row)
    j .Ldwa_tap

.Ldwa_extract:
    # ---- Pass1: QACC_L -> q0 (lanes 0-3) and q3 (lanes 4-7) ----
    # (verbatim from s8_accx_depthwise.S / the esp-nn two-pass read-back)
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

    # advance group
    l32i a9, a1, 20
    addi a9, a9, 16
    s32i a9, a1, 20
    bge a9, a7, .Ldwa_done      # exit when g*16 >= out_c (a7 survives;
                                # it is only a VLD delta / compare bound)
    j .Ldwa_group

.Ldwa_done:
    retw
    .size   s8_accx_depthwise_anytap, . - s8_accx_depthwise_anytap
