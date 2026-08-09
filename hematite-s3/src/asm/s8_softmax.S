// s8_softmax.S — bespoke softmax SIMD helpers.
//
// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
// Windowed ABI (call8): arguments arrive in a2.., caller places them in
// a10.. (window rotates by 8). EE.VMAX.S8 q0,q1 takes the element-wise max
// of two 16×int8 vectors — the SIMD find-max primitive for softmax.
//
// s8_find_max(input, len) -> max
//   a2 = *const i8 input (16-byte aligned)
//   a3 = u32 len (>= 1)
//   returns the signed max over input[0..len) in a2.
//   For len >= 16: VMAX over floor(len/16) vector chunks, horizontal reduce
//   the final vector, then a scalar tail for len%16. For len < 16: scalar.
//   Clobbers a4..a10 + q0/q1; preserves a11..a15.

    .text
    .align  4
    .global s8_find_max
    .type   s8_find_max, @function
s8_find_max:
    entry   a1, 32
    mov.n   a5, a2                 // base input ptr (tail needs it)
    mov.n   a6, a3                 // len (tail bound)
    movi    a7, -128               // running max
    movi.n  a8, 16
    bltu    a3, a8, .Lfmax_scalar
    // SIMD: load first vector
    movi.n  a4, 0
    ee.vld.128.ip q0, a2, 16
    addi    a4, a4, 16
.Lfmax_loop:
    bge     a4, a6, .Lfmax_done
    ee.vld.128.ip q1, a2, 16
    ee.vmax.s8 q0, q0, q1
    addi    a4, a4, 16
    j       .Lfmax_loop
.Lfmax_done:
    // horizontal reduce q0 (16 int8) -> max in a7
    addi    a8, a1, 16
    ee.vst.128.ip q0, a8, 16
    movi.n  a11, 16
    movi.n  a9, 0
.Lfmax_hloop:
    add     a8, a1, a9
    addi    a8, a8, 16
    l8ui    a10, a8, 0
    sext    a10, a10, 8
    blt     a7, a10, .Lfmax_hmx
    j       .Lfmax_hnext
.Lfmax_hmx:
    mov.n   a7, a10
.Lfmax_hnext:
    addi.n  a9, a9, 1
    blt     a9, a11, .Lfmax_hloop
    // scalar tail: input[a4 .. a6)
.Lfmax_tail:
    bge     a4, a6, .Lfmax_ret
    add     a8, a5, a4
    l8ui    a10, a8, 0
    sext    a10, a10, 8
    blt     a7, a10, .Lfmax_tmx
    j       .Lfmax_tnext
.Lfmax_tmx:
    mov.n   a7, a10
.Lfmax_tnext:
    addi.n  a4, a4, 1
    j       .Lfmax_tail
.Lfmax_scalar:
    movi.n  a4, 0
    j       .Lfmax_tail
.Lfmax_ret:
    mov.n   a2, a7
    retw.n
    .size   s8_find_max, . - s8_find_max
