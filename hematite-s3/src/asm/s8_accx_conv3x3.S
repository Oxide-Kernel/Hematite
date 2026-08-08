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
# advance between 3x3 rows is `row_delta` = (in_w - 3) * in_c bytes.
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
s8_accx_conv3x3:
    entry a1, 64

    movi a10, 0         # oc index
    movi a13, 0         # col counter (0,1,2)

.Loc3:
    bge a10, a6, .Ldone3
    EE.ZERO.ACCX

    mov a15, a2         # iptr = window base (top-left tap)
    movi a11, 9         # taps remaining
    movi a13, 0         # col = 0

.Ltap3:
    beqz a11, .Lextract3

    movi a12, 0         # ic offset within tap
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
    j .Ltap3             # col delta 0 (dilation 1): iptr already at next col
.LrowAdv3:
    movi a13, 0
    add a15, a15, a7     # row_delta = (in_w - 3) * in_c
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
