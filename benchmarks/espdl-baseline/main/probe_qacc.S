# probe_qacc.S — derive the 40-byte QACC store layout -> 16x i32 lane mapping.
#
# VMULAS.S8.QACC is the per-lane (element-wise) accumulate used for depthwise:
#   QACC[lane] += input[lane] * filter[lane]     (lane = 0..15, 16 lanes)
#
# The only documented read-back route is the 40-byte store:
#   EE.ST.QACC_L.L.128.IP  out+0,  16   -> QACC_L low-128 (16 bytes)
#   EE.ST.QACC_L.H.32.IP   out+16, -16  -> QACC_L high-32  (4 bytes)  [total 20]
#   EE.ST.QACC_H.L.128.IP  out+20, 16   -> QACC_H low-128 (16 bytes)
#   EE.ST.QACC_H.H.32.IP   out+36, -16  -> QACC_H high-32  (4 bytes)  [total 40]
#
# ABI (windowed):
#   probe_qacc(filter, input, out40, macs, reverse)
#        a2=filter (16x int8), a3=input (16x int8),
#        a4=out40 (40 bytes), a5=macs (repeat count), a6=reverse (0/1)
#
# Use filter[i]=1, input[i]=i+1 -> lane i holds i+1 (distinct small values)
# so the 40-byte dump reveals exactly where each lane lives.

    .align 4
    .text
    .global probe_qacc_layout
    .type   probe_qacc_layout, @function
probe_qacc_layout:
    entry a1, 96

    EE.ZERO.QACC
    # --- reverse mode: q0=input(a3), q1=filter(a2); else q0=filter(a2), q1=input(a3)
    beqz.n a6, .Lno_swap
    EE.VLD.128.IP q0, a3, 16      # q0 = input
    EE.VLD.128.IP q1, a2, 16      # q1 = filter
    j .Lmac_done
.Lno_swap:
    EE.VLD.128.IP q0, a2, 16      # q0 = filter
    EE.VLD.128.IP q1, a3, 16      # q1 = input
.Lmac_done:

    movi a7, 0
.Lmac_loop:
    EE.VMULAS.S8.QACC q0, q1      # per-lane element-wise MAC
    addi.n a7, a7, 1
    blt a7, a5, .Lmac_loop

    # Contiguous 40-byte store (no overwrite):
    #   QACC_L.L.128 -> out[0..16), QACC_L.H.32 -> out[16..20),
    #   QACC_H.L.128 -> out[20..36), QACC_H.H.32 -> out[36..40)
    mov.n a6, a4
    ee.st.qacc_l.l.128.ip a6, 16
    ee.st.qacc_l.h.32.ip  a6, 4
    ee.st.qacc_h.l.128.ip a6, 16
    ee.st.qacc_h.h.32.ip  a6, 4

    retw
    .size   probe_qacc_layout, . - probe_qacc_layout

# ---------------------------------------------------------------------------
# probe_qacc_s16(filter, input, out_low, out_high, macs)
#   a2=filter(16x int8), a3=input(16x int8), a4=out_low(16 bytes),
#   a5=out_high(16 bytes), a6=macs
# Does macs x VMULAS.S8.QACC then extracts via the vendored per-channel
# method: SRCMB.S16.QACC at shift 4 (-> out_low, 8x int16) and at shift 20
# (-> out_high, 8x int16). Determines which (shift, halfword) holds lane k.
# ---------------------------------------------------------------------------
    .align 4
    .text
    .global probe_qacc_s16
    .type   probe_qacc_s16, @function
probe_qacc_s16:
    entry a1, 96

    EE.ZERO.QACC
    EE.VLD.128.IP q0, a2, 16      # q0 = filter
    EE.VLD.128.IP q1, a3, 16      # q1 = input

    movi a7, 0
.Ls16_mac_loop:
    EE.VMULAS.S8.QACC q0, q1      # per-lane element-wise MAC
    addi.n a7, a7, 1
    blt a7, a6, .Ls16_mac_loop

    movi a8, 4
    EE.SRCMB.S16.QACC q2, a8, 0   # lower 16 bit (8x int16 -> q2)
    movi a8, 20
    EE.SRCMB.S16.QACC q3, a8, 0   # even 16 bit (8x int16 -> q3)

    EE.VST.128.IP q2, a4, 16      # out_low[0..16)
    EE.VST.128.IP q3, a5, 16      # out_high[0..16)

    retw
    .size   probe_qacc_s16, . - probe_qacc_s16

# ---------------------------------------------------------------------------
# probe_qacc_perchan(filter, input, out16, out_low, out_high, macs)
#   a2=filter(16x int8), a3=input(16x int8), a4=out16 (32 bytes = 16x int16
#   final VZIP result), a5=out_low (16 bytes, SRCMB shift4), a6=out_high
#   (16 bytes, SRCMB shift20), a7=macs (6th reg arg).
# Replicates the EXACT vendored tie728_s8_conv2d_per_channel_result sequence:
#   1. SRCMB.S16.QACC out, shift=4
#   2. gapped 40B store (L.L.128 IP16, L.H.32 IP16, H.L.128 IP16, H.H.32 IP4)
#   3. SRCMB.S16.QACC tmp, shift=20
#   4. l16si odd-lane rearrange from store
#   5. VLD output_v from a1, VZIP.16 output_v, tmp_q1
# out16 = final 16 halfwords (post-VZIP), out_low/out_high = the two SRCMB
# reads BEFORE VZIP.
# ---------------------------------------------------------------------------
    .align 4
    .text
    .global probe_qacc_perchan
    .type   probe_qacc_perchan, @function
probe_qacc_perchan:
    entry a1, 128

    EE.ZERO.QACC
    EE.VLD.128.IP q0, a2, 16      # q0 = filter
    EE.VLD.128.IP q1, a3, 16      # q1 = input

    movi a8, 0
.Lpc_mac_loop:
    EE.VMULAS.S8.QACC q0, q1
    addi.n a8, a8, 1
    blt a8, a7, .Lpc_mac_loop

    # 1. SRCMB shift4 -> q2 (output_v), saved to out_low (a5) later
    movi a8, 4
    EE.SRCMB.S16.QACC q2, a8, 0

    # 2. gapped store to a1 (stack scratch)
    mov.n a9, a1
    ee.st.qacc_l.l.128.ip a9, 16
    ee.st.qacc_l.h.32.ip  a9, 16
    ee.st.qacc_h.l.128.ip a9, 16
    ee.st.qacc_h.h.32.ip  a9, 4

    # 3. SRCMB shift20 -> q3 (tmp_q1)
    movi a8, 20
    EE.SRCMB.S16.QACC q3, a8, 0

    # 4. l16si odd-lane rearrange (mirror the vendored macro exactly):
    addi a9, a1, 5
    l16si a10, a9, 0              # a1+5
    s16i  a10, a1, 2
    l16si a10, a1, 10             # a1+10
    s16i  a10, a1, 4
    addi a9, a1, 15
    l16si a10, a9, 0              # a1+15
    s16i  a10, a1, 6
    l16si a10, a1, 32             # a1+32
    s16i  a10, a1, 8
    addi a9, a1, 37
    l16si a10, a9, 0              # a1+37
    s16i  a10, a1, 10
    l16si a10, a1, 42             # a1+42
    s16i  a10, a1, 12
    addi a9, a1, 47
    l16si a10, a9, 0              # a1+47
    s16i  a10, a1, 14

    # save SRCMB reads before VZIP
    EE.VST.128.IP q2, a5, 16      # out_low (shift4)
    EE.VST.128.IP q3, a6, 16      # out_high (shift20)

    # 5. VLD output_v from a1, VZIP.16
    EE.VLD.128.IP q2, a1, 0       # output_v = odd 16 bit (rearranged)
    EE.VZIP.16 q2, q3             # interleave odd(output_v), even(tmp_q1)
    EE.VST.128.IP q2, a4, 16      # out16[0..16) post-VZIP low
    EE.VST.128.IP q3, a4, 16      # out16[16..32) post-VZIP high

    retw
    .size   probe_qacc_perchan, . - probe_qacc_perchan

# ---------------------------------------------------------------------------
# probe_qacc_esplike(filter, input, out32, macs)
#   a2=filter(16x int8), a3=input(16x int8), a4=out32 (64 bytes = 16x i32),
#   a5=macs (5th reg arg).
# Implements the EXACT esp-nn depthwise readback (lines 300-396 of
# esp_nn_depthwise_conv_s8_mult1_3x3_padded_esp32s3.S) verbatim:
#   gapped 20B store (st.qacc_l.l.128.ip 16; st.qacc_l.h.32.ip -16 -> a1+0..19)
#   w0..w4 = a1+0/+4/+8/+12/+16
#   q0 = [sext(w0,19), ((w1<<24)>>12)|((w0>>20)&0xFFF), (w1<<4)>>12,
#         ((w2<<16)>>12)|((w1>>28)&0xF)]
#   q3 = [((w3<<28)>>12)|((w2>>16)&0xFFFF), (w3<<8)>>12,
#         ((w4<<20)>>12)|((w3>>24)&0xFF), w4>>12]
#   then QACC_H store OVERWRITES a1+0..19, re-read w0..w4
#   q1 = [sext(w0,19), ((w1<<24)>>12)|((w0>>20)&0xFFF), (w1<<4)>>12,
#         ((w2<<16)>>12)|((w1>>28)&0xF)]
#   q6 = [((w3<<28)>>12)|((w2>>16)&0xFFFF), (w3<<8)>>12,
#         ((w4<<20)>>12)|((w3>>24)&0xFF), w4>>12]
# out32 = q0(0..3) q3(4..7) q1(8..11) q6(12..15), each lane one i32.
# sext reg,imm19 = sign-extend bit 19 (20-bit) of the register.
# ---------------------------------------------------------------------------
    .align 4
    .text
    .global probe_qacc_esplike
    .type   probe_qacc_esplike, @function
probe_qacc_esplike:
    entry a1, 128

    EE.ZERO.QACC
    EE.VLD.128.IP q0, a2, 16      # q0 = filter
    EE.VLD.128.IP q1, a3, 16      # q1 = input

    movi a8, 0
.Le_mac_loop:
    EE.VMULAS.S8.QACC q0, q1
    addi.n a8, a8, 1
    blt a8, a5, .Le_mac_loop

    # gapped 20B store: QACC_L -> a1+0..19
    mov a12, a1
    ee.st.qacc_l.l.128.ip a12, 16
    ee.st.qacc_l.h.32.ip  a12, -16

    # ---- Pass1: QACC_L -> q0 (lanes 0-3) and q3 (lanes 4-7) ----
    l32i.n a9, a1, 8              # w2
    l32i.n a11, a1, 4             # w1
    l32i.n a15, a1, 0             # w0
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

    l32i.n a11, a1, 16            # w4
    l32i.n a14, a1, 12            # w3
    slli a13, a11, 20
    ee.st.qacc_h.l.128.ip a12, 16
    ee.st.qacc_h.h.32.ip  a12, -16
    srai a11, a11, 12
    srai a13, a13, 12
    slli a8, a14, 28
    slli a15, a14, 8
    srai a15, a15, 12
    srai a8, a8, 12
    l32i.n a12, a1, 8             # w2 (re-read AFTER QACC_H store)
    or a8, a8, a9                 # a9 still = (w2>>16)&0xFFFF from Pass1
    extui a14, a14, 24, 8
    l32i.n a9, a1, 0              # w0 (re-read AFTER QACC_H store)
    or a13, a13, a14
    ee.movi.32.q q3, a8, 0
    ee.movi.32.q q3, a15, 1
    ee.movi.32.q q3, a13, 2
    ee.movi.32.q q3, a11, 3

    # ---- Pass2: QACC_H now in a1+0..19 -> q1 (lanes 8-11) and q6 (12-15) ----
    l32i.n a14, a1, 4             # w1 (QACC_H)
    sext  a10, a9, 19             # a9 = w0 (QACC_H)
    extui a9, a9, 20, 12
    slli a13, a12, 16             # a12 = w2 (QACC_H)
    slli a8, a14, 24
    extui a12, a12, 16, 16
    srai a13, a13, 12
    srai a8, a8, 12
    slli a15, a14, 4
    srai a15, a15, 12
    or a8, a8, a9
    extui a14, a14, 28, 4
    l32i.n a9, a1, 12             # w3 (QACC_H)
    or a13, a13, a14
    ee.movi.32.q q1, a10, 0
    ee.movi.32.q q1, a8, 1
    ee.movi.32.q q1, a15, 2
    ee.movi.32.q q1, a13, 3

    l32i.n a15, a1, 16            # w4 (QACC_H)
    slli a11, a9, 28
    slli a10, a9, 8
    srai a10, a10, 12
    srai a11, a11, 12
    extui a9, a9, 24, 8
    or a11, a11, a12              # a12 = (w2>>16)&0xFFFF (QACC_H)
    slli a8, a15, 20
    srai a15, a15, 12
    srai a8, a8, 12
    or a8, a8, a9
    ee.movi.32.q q6, a11, 0
    ee.movi.32.q q6, a10, 1
    ee.movi.32.q q6, a8, 2
    ee.movi.32.q q6, a15, 3

    # store 16x i32
    ee.vst.128.ip q0, a4, 16
    ee.vst.128.ip q3, a4, 16
    ee.vst.128.ip q1, a4, 16
    ee.vst.128.ip q6, a4, 16

    retw
    .size   probe_qacc_esplike, . - probe_qacc_esplike
