# s8_accx_conv1x1_orig.S — original branchy kernel (A/B baseline).
    .align 4
    .text
    .global s8_accx_conv1x1_orig
    .type   s8_accx_conv1x1_orig, @function
s8_accx_conv1x1_orig:
    entry a1, 64
    movi a7, 0
    movi a8, 16
    mov a9, a3
.Loc:
    bge a7, a6, .Ldone
    EE.ZERO.ACCX
    movi a10, 0
    mov a11, a9
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
    add a9, a9, a5
    j .Loc
.Ldone:
    retw
