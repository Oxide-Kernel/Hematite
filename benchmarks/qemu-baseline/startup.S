/* SPDX-License-Identifier: Apache-2.0
 *
 * Freestanding Xtensa call0-ABI entry point for ESP32-S3 under QEMU.
 *
 * The 2nd-stage bootloader (embedded by espflash `save-image --merge`)
 * jumps here via the image-header entry point (ELF `e_entry`).  Because the
 * whole binary is RAM-resident (see linker.ld), the bootloader has already
 * copied .text/.rodata/.data into place; we only need to:
 *
 *   1. mask all interrupts (PS.INTLEVEL = 15),
 *   2. set the stack pointer,
 *   3. zero .bss,
 *   4. call main.
 *
 * The call0 ABI is used throughout (`-mabi=call0`), so no register-window
 * overflow/underflow handlers are required and no windowed instructions are
 * emitted.
 *
 * Absolute addresses are loaded with `movi`; the assembler relaxes those to
 * l32r against its `.xt.lit` pool, which the Xtensa linker relaxation pass
 * (--relax) then folds back into immediate sequences at link time.
 */

    .section .text._start
    .align  4
    .global _start
    .type   _start, @function
_start:
    rsil    a2, 15                  /* PS.INTLEVEL = 15: disable interrupts */
    movi    a1, _stack_top          /* set stack pointer (stack grows down) */
    /* Zero .bss */
    movi    a2, _bss_start
    movi    a3, _bss_end
.Lbss_loop:
    bgeu    a2, a3, .Lbss_done
    movi    a4, 0
    s32i    a4, a2, 0
    addi    a2, a2, 4
    j       .Lbss_loop
.Lbss_done:
    call0   main
    /* main should never return; if it does, halt. */
.Lhalt:
    j       .Lhalt
    .size   _start, .-_start
