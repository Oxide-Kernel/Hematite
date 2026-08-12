    .align 4
    .text
    .global probe_vldbc
    .type   probe_vldbc, @function
# probe_vldbc(const int8_t *src, int8_t *out, int32_t offset)  a2=src a3=out a4=offset
# Load one byte via VLDBC.8 from src+offset (may be UNALIGNED), store the
# 16-lane vector. Decisive test of VLDBC.8 address semantics: does it read the
# byte at the exact (unaligned) address, or align down to 16 bytes?
probe_vldbc:
    entry a1, 64
    add a2, a2, a4             # src += offset (test unaligned addressing)
    EE.VLDBC.8 q0, a2
    ee.vst.128.ip q0, a3, 16
    retw
    .size   probe_vldbc, . - probe_vldbc
