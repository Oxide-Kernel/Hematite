# s8_accx_conv3x3.S — bespoke GPR-accumulator 3x3 conv kernel (bit-exact).
#
# Strategy (validated on device by probe_s8accx): EE.VMULAS.S8.ACCX is a
# 16-wide element-wise dot-product reduction into a 32-bit accumulator with
# full 16-bit products (127x127 = 16129 preserved). EE.SRS.ACCX gpr, 0, 0
# extracts the exact int32 sum. Accumulates 9 taps per output channel:
#   acc[oc] = sum(tap) sum(ic) filter[(oc*9+tap)*in_c+ic] * input[tap_off+ic]
#
# RAW OHWI weight layout works directly (element-wise reduction), so NO weight
# transform is needed. Outputs raw int32 accs to acc_out; the Rust caller
# requantizes per channel.
#
# Restriction: dilation must be 1 (horizontal tap delta = 0); the only runtime
# advance between 3x3 rows is `row_delta` = (in_w - 3) * in_c bytes. A full
# input row stride is row_delta + 3*in_c.
#
# Optimization (this revision):
#   * Fast path for in_c == 64: 9 taps fully unrolled. This TIE exposes only
#     8 Q registers (q0..q7), so the 3x3 window (36 vectors) cannot be kept
#     resident. Per tap we software-pipeline 4 filter vectors (q0..q3) against
#     4 input vectors (q4..q7), interleaving loads and MACs to hide VLD
#     latency. The 3 columns of a row are loaded contiguously (VLD.IP), and
#     `row_delta` bridges rows. Single-level hardware `loop` over output
#     channels (nested hardware loops are unreliable on this TIE/LLVM combo).
#   * General path for any other in_c: the proven branch-based loop structure.
#
# ABI (windowed, 6 args — caller a10..a15 = callee a2..a7):
#   s8_accx_conv3x3(input, filter, acc_out, in_c, out_c, row_delta)
#       a2=input (window top-left tap), a3=filter (out_c*9*in_c int8),
#       a4=acc_out (out_c int32), a5=in_c, a6=out_c, a7=row_delta
#   Precondition: in_c % 16 == 0, out_c >= 1, dilation == 1,
#       input/filter 16-byte aligned, acc_out 4-byte aligned.

    .align 4
    .text
    .global s8_accx_conv3x3
    .type   s8_accx_conv3x3, @function

    # One tap block (in_c == 64 fast path): q0..q3 = filter, q4..q7 = input.
    # Loads are hoisted so each VMULAS uses a vector loaded 2 instr earlier.
    .macro TAP_64 fin, qin
    EE.VLD.128.IP q0, a9, 16
    EE.VLD.128.IP q1, a9, 16
    EE.VLD.128.IP q2, a9, 16
    EE.VLD.128.IP q3, a9, 16
    EE.VLD.128.IP q4, \fin, 16
    EE.VLD.128.IP q5, \fin, 16
    EE.VLD.128.IP q6, \fin, 16
    EE.VLD.128.IP q7, \fin, 16
    EE.VMULAS.S8.ACCX q0, q4
    EE.VMULAS.S8.ACCX q1, q5
    EE.VMULAS.S8.ACCX q2, q6
    EE.VMULAS.S8.ACCX q3, q7
    .endm

s8_accx_conv3x3:
    entry a1, 64

    beqi a5, 64, .Lc3fast64       # in_c == 64: unrolled fast path

    # ---- General path (any in_c % 16 == 0) ----
    movi a10, 0                 # oc index
.Loc3:
    bge a10, a6, .Ldone3
    EE.ZERO.ACCX
    mov a15, a2                 # iptr = window base (top-left tap)
    movi a11, 9                 # taps remaining
    movi a13, 0                 # col = 0
.Ltap3:
    beqz a11, .Lextract3
    movi a12, 0                 # ic offset within tap
.Lic3:
    bge a12, a5, .LicDone3
    EE.VLD.128.IP q0, a3, 16    # filter[oc*9*in_c + tap*in_c + ic..]
    EE.VLD.128.IP q1, a15, 16   # input[tap_off + ic..]
    EE.VMULAS.S8.ACCX q0, q1
    addi a12, a12, 16
    j .Lic3
.LicDone3:
    addi a11, a11, -1
    beqz a11, .Lextract3
    addi a13, a13, 1
    beqi a13, 3, .LrowAdv3
    j .Ltap3                    # col delta 0 (dilation 1)
.LrowAdv3:
    movi a13, 0
    add a15, a15, a7            # row_delta = (in_w - 3) * in_c
    j .Ltap3
.Lextract3:
    movi a12, 0
    EE.SRS.ACCX a14, a12, 0
    s32i a14, a4, 0
    addi a4, a4, 4
    addi a10, a10, 1
    j .Loc3
.Ldone3:
    retw

    # ---- Fast path: in_c == 64. 9 taps unrolled, single loop over oc. ----
    # Input walking: a10 = window base; rows bridged by row_delta (a7).
    #   row0: taps 0,1,2 read a10[0..63], [64..127], [128..191]
    #   after tap 2: a10 += row_delta + 192  -> row1 col0
    #   row1: taps 3,4,5 ; after tap 5: a10 += row_delta + 192 -> row2 col0
    #   row2: taps 6,7,8
    # Filter walking: a9 = filter row for oc; 9 taps of 64 bytes each.
.Lc3fast64:
    movi a8, 0                  # oc index (branch loop; LLVM MC loop/branch
                                # fixup ranges too small for the 9-tap body)
.Lc3f64loop:
    blt a8, a6, .Lc3f64body     # short forward branch into the body
    j .Lc3f64done               # long jump out (a8 >= out_c)
.Lc3f64body:
    EE.ZERO.ACCX
    mov a13, a3                 # a13 = running filter row ptr
    mov a9, a13
    mov a10, a2                 # window base (row0 col0)
    TAP_64 a10, 0               # tap 0: row0 col0 (a10 advances to col1)
    TAP_64 a10, 0               # tap 1: row0 col1 (a10 advances to col2)
    TAP_64 a10, 0               # tap 2: row0 col2
    add a10, a10, a7            # + row_delta -> row1 col0 (a10 already +192)
    TAP_64 a10, 0               # tap 3: row1 col0
    TAP_64 a10, 0               # tap 4: row1 col1
    TAP_64 a10, 0               # tap 5: row1 col2
    add a10, a10, a7            # + row_delta -> row2 col0
    TAP_64 a10, 0               # tap 6: row2 col0
    TAP_64 a10, 0               # tap 7: row2 col1
    TAP_64 a10, 0               # tap 8: row2 col2
    movi a12, 0
    EE.SRS.ACCX a15, a12, 0
    s32i a15, a4, 0
    addi a4, a4, 4
    addi a3, a3, 576            # next filter row: 9*in_c = 576
    addi a8, a8, 1
    j .Lc3f64loop
.Lc3f64done:
    retw
