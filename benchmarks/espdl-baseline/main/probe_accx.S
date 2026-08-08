# probe_accx.S — verify EE.VMULAS.S16.ACCX / EE.SRS.ACCX / EE.LD.ACCX.IP semantics.
#
# ABI (windowed):
#   probe_accx_mac_sweep(filter, input, out64)
#                                 a2=filter (8x int16), a3=input (8x int16),
#                                 a4=out64 (64x int32)
#   ZERO.ACCX; ONE VMULAS.S16.ACCX; then SRS.ACCX at pos0 and pos1 for
#   shifts 0..31 (32+32 = 64 values) to map the ACCX lane layout empirically.
#
#   probe_accx_load(ptr, out)     a2=ptr (8x int32 pattern), a3=out (8x int32)
#   ZERO.ACCX; LD.ACCX.IP; SRS.ACCX pos0/pos1 shift 0..3 (8 values)

    .align 4
    .text
    .global probe_accx_mac_sweep
    .type   probe_accx_mac_sweep, @function
probe_accx_mac_sweep:
    entry a1, 96

    EE.ZERO.ACCX

    # load filter (8x int16 = 16 bytes) into q0, input (8x int16 = 16 bytes) into q1
    EE.VLD.128.IP q0, a2, 16
    EE.VLD.128.IP q1, a3, 16

    # ONE element-wise MAC: ACCX[i] += filter[i]*input[i]
    EE.VMULAS.S16.ACCX q0, q1

    # sweep pos 0, shifts 0..31
    movi a6, 0
    movi a7, 32
    movi a8, 0
.Lpos0:
    EE.SRS.ACCX a5, a6, 0
    s32i a5, a4, 0
    addi a4, a4, 4
    addi a6, a6, 1
    addi a8, a8, 1
    blt a8, a7, .Lpos0

    # sweep pos 1, shifts 0..31
    movi a6, 0
    movi a7, 32
    movi a8, 0
.Lpos1:
    EE.SRS.ACCX a5, a6, 1
    s32i a5, a4, 0
    addi a4, a4, 4
    addi a6, a6, 1
    addi a8, a8, 1
    blt a8, a7, .Lpos1

    retw


    .align 4
    .text
    .global probe_accx_load
    .type   probe_accx_load, @function
probe_accx_load:
    entry a1, 64

    EE.ZERO.ACCX
    EE.LD.ACCX.IP a2, 32

    # read back pos0/pos1 at shifts 0..3 = 8 values
    movi a6, 0
    EE.SRS.ACCX a5, a6, 0
    s32i a5, a3, 0
    movi a6, 1
    EE.SRS.ACCX a5, a6, 0
    s32i a5, a3, 4
    movi a6, 2
    EE.SRS.ACCX a5, a6, 0
    s32i a5, a3, 8
    movi a6, 3
    EE.SRS.ACCX a5, a6, 0
    s32i a5, a3, 12
    movi a6, 0
    EE.SRS.ACCX a5, a6, 1
    s32i a5, a3, 16
    movi a6, 1
    EE.SRS.ACCX a5, a6, 1
    s32i a5, a3, 20
    movi a6, 2
    EE.SRS.ACCX a5, a6, 1
    s32i a5, a3, 24
    movi a6, 3
    EE.SRS.ACCX a5, a6, 1
    s32i a5, a3, 28

    retw
