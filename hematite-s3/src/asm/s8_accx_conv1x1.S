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
# Optimization (this revision):
#   * Input-resident fast path for in_c == 64: the 4 input vectors live in
#     q0..q3 for the whole kernel, so each output channel only walks the
#     filter (4 VLD, then 4 VMULAS with loads fully hoisted to hide VLD
#     latency). Single-level hardware `loop` over output channels.
#   * General path for any other in_c: the proven branch-based loop structure
#     (nested Xtensa hardware loops are unreliable on this TIE/LLVM combo).
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

    srai a8, a5, 4              # a8 = groups = in_c / 16
    beqi a8, 4, .Lfast64        # in_c == 64: input-resident fast path

    # ---- General path (any in_c % 16 == 0) ----
    # Hardware `loop` inner (single loop; outer oc loop stays branch-based —
    # nested hardware loops are unreliable on this TIE/LLVM combo). groups>=1
    # always (in_c>=16), so `loop` runs the body exactly a8 times.
    movi a7, 0                  # oc index
    mov a9, a3                  # a9 = running filter row ptr
.Loc:
    bge a7, a6, .Ldone
    EE.ZERO.ACCX
    mov a10, a2                 # a10 = running input ptr
    mov a11, a9                 # a11 = running filter ptr
    mov a12, a8                 # a12 = groups counter (hardware loop)
    loop a12, .Lic_end
        EE.VLD.128.IP q0, a11, 16
        EE.VLD.128.IP q1, a10, 16
        EE.VMULAS.S8.ACCX q0, q1
    .Lic_end:
    movi a13, 0
    EE.SRS.ACCX a14, a13, 0
    s32i a14, a4, 0
    addi a4, a4, 4
    addi a7, a7, 1
    add a9, a9, a5              # next filter row
    j .Loc

.Ldone:
    retw

    # ---- Fast path: in_c == 64. Input resident in q0..q3. ----
.Lfast64:
    mov a10, a2
    EE.VLD.128.IP q0, a10, 16
    EE.VLD.128.IP q1, a10, 16
    EE.VLD.128.IP q2, a10, 16
    EE.VLD.128.IP q3, a10, 16

    mov a7, a3                  # a7 = running filter row ptr
    loop a6, .Lf64done
        EE.ZERO.ACCX
        mov a9, a7
        EE.VLD.128.IP q4, a9, 16    # filter g0
        EE.VLD.128.IP q5, a9, 16    # filter g1
        EE.VLD.128.IP q6, a9, 16    # filter g2
        EE.VLD.128.IP q7, a9, 16    # filter g3
        EE.VMULAS.S8.ACCX q4, q0
        EE.VMULAS.S8.ACCX q5, q1
        EE.VMULAS.S8.ACCX q6, q2
        EE.VMULAS.S8.ACCX q7, q3
        movi a12, 0
        EE.SRS.ACCX a13, a12, 0
        s32i a13, a4, 0
        addi a4, a4, 4
        add a7, a7, a5           # next filter row (a5 == 64)
    .Lf64done:
    retw
