# probe_accx.S — verify EE.VMULAS.S16.ACCX / EE.SRS.ACCX / EE.VMULAS.S8.ACCX.
#
# ABI (windowed):
#   probe_s8accx(filter, input, out3)
#        a2=filter (16x int8), a3=input (16x int8), a4=out3 (3x int32)
#   ZERO.ACCX; ONE VMULAS.S8.ACCX (16-wide dot reduction);
#   SRS.ACCX at shift 0 / 1 / triangular-1(shift gpr=1) -> 3 values.
#   Also probe with a second identical MAC to check accumulation width.

    .align 4
    .text
    .global probe_s8accx
    .type   probe_s8accx, @function
probe_s8accx:
    entry a1, 96

    EE.ZERO.ACCX
    EE.VLD.128.IP q0, a2, 16
    EE.VLD.128.IP q1, a3, 16
    EE.VMULAS.S8.ACCX q0, q1
    EE.VMULAS.S8.ACCX q0, q1

    movi a6, 0
    EE.SRS.ACCX a5, a6, 0
    s32i a5, a4, 0
    movi a6, 1
    EE.SRS.ACCX a5, a6, 0
    s32i a5, a4, 4
    movi a6, 2
    EE.SRS.ACCX a5, a6, 0
    s32i a5, a4, 8

    retw
