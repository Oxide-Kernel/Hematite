# s8_accx_conv1x1.S — bespoke GPR-accumulator conv1x1/fc kernel (bit-exact).
#
# Strategy (validated on device by probe_s8accx):
#   EE.VMULAS.S8.ACCX F, I performs a 16-wide element-wise dot-product
#   reduction: accx += sum(i=0..15) F[i]*I[i], with 16-bit products
#   accumulated into 32-bit lanes. EE.SRS.ACCX gpr, 0, 0 extracts the exact
#   int32 accumulator. This is bit-exact for int8 convs of any depth.
#
#   For output channel oc:  out_acc[oc] = sum(ic) filter[oc*in_c+ic] *
#   input[ic]. Because the reduction is element-wise (F[lane]*I[lane]), the
#   RAW [oc][ic] weight layout works directly — NO transform needed.
#
# ABI (windowed):
#   s8_accx_conv1x1(input, filter, acc_out, in_c, out_c)
#       a2=input (in_c int8), a3=filter (out_c*in_c int8),
#       a4=acc_out (out_c int32), a5=in_c, a6=out_c
#   Precondition: in_c % 16 == 0, out_c >= 1. All pointers 16-byte aligned.

    .align 4
    .text
    .global s8_accx_conv1x1
    .type   s8_accx_conv1x1, @function
s8_accx_conv1x1:
    entry a1, 64

    movi a7, 0          # oc index
    movi a8, 16         # group step
    # a9 = running filter ptr (starts at filter)
    mov a9, a3

.Loc:
    bge a7, a6, .Ldone
    EE.ZERO.ACCX

    movi a10, 0         # ic offset within row
    # a11 = running filter row ptr (= a9)
    mov a11, a9
    # a12 = running input ptr (= a2)
    mov a12, a2

.Lic:
    bge a10, a5, .Lextract
    EE.VLD.128.IP q0, a11, 16
    EE.VLD.128.IP q1, a12, 16
    EE.VMULAS.S8.ACCX q0, q1
    addi a10, a10, 16
    j .Lic

.Lextract:
    movi a13, 0
    EE.SRS.ACCX a14, a13, 0
    s32i a14, a4, 0
    addi a4, a4, 4

    addi a7, a7, 1
    add a9, a9, a5      # next filter row
    j .Loc

.Ldone:
    retw
