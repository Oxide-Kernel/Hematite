/* SPDX-License-Identifier: Apache-2.0
 *
 * Freestanding ESP32-S3 linker script for the QEMU C baseline.
 *
 * Memory regions mirror esp-hal 1.1.1 `ld/esp32s3/memory.x`:
 *   .text   -> irom_seg 0x42000020 (executed from flash via cache)
 *   .rodata.appdesc -> drom_seg 0x3C000020 (the ESP-IDF app descriptor)
 *   .rodata/.data/.bss/.stack -> dram_seg (internal SRAM data window)
 *
 * The ESP-IDF v5.5 2nd-stage bootloader REQUIRES an app to carry one DROM
 * flash segment and one IROM flash segment (its set_cache_and_start_app()
 * programs the flash MMU from them; an all-RAM app faults there).  The
 * DROM segment therefore holds ONLY the app descriptor, which the
 * bootloader reads via direct flash reads (bootloader_flash_read).
 *
 * Everything the app actually READS (string literals, const tables) lives
 * in DRAM.  Under QEMU the DROM flash-cache window (0x3C000000) maps one
 * 64 KB page off for this app's layout, returning the IROM code bytes
 * instead of the .rodata — so no runtime reads go through the drom cache.
 *
 * Entry: `_start` (call0 ABI — no register-window overflow handlers needed).
 */
ENTRY(_start)

STACK_SIZE = 0x4000; /* 16 KiB stack, grows down from _stack_top */

MEMORY
{
  /* External flash, instruction (executes via the flash cache). */
  irom_seg (RX) : ORIGIN = 0x42000020, len = 0x100000

  /* External flash, data — holds the app descriptor only. */
  drom_seg (R)  : ORIGIN = 0x3C000020, len = 0x1000

  /* Internal SRAM data window. */
  dram_seg (RW) : ORIGIN = 0x3FC88000, len = 0x60000
}

SECTIONS
{
  .text : ALIGN(4)
  {
    _stext = .;
    /* Entry code first so the image entry point is the very start of the
     * IROM segment (_start's l32r literal pool must follow it). */
    *(.text._start)
    *(.literal .text .literal.* .text.*)
    /* Xtensa relaxation literal tables, kept at the end of the segment. */
    *(.xt.lit)
    *(.xt.prop)
    _etext = .;
  } > irom_seg

  .rodata.appdesc : ALIGN(4)
  {
    /* App descriptor first: the IDF bootloader reads it from the start of
     * segment 0. KEEP() so --gc-sections cannot drop it (nothing references
     * the symbol from code). */
    KEEP(*(.rodata.appdesc))
  } > drom_seg

  .rodata : ALIGN(4)
  {
    _rodata_start = .;
    *(.rodata .rodata.*)
    _rodata_end = .;
  } > dram_seg

  .data : ALIGN(4)
  {
    _data_start = .;
    *(.data .data.*)
    _data_end = .;
  } > dram_seg

  .bss (NOLOAD) : ALIGN(4)
  {
    _bss_start = .;
    *(.bss .bss.* COMMON)
    _bss_end = .;
  } > dram_seg

  /* Stack grows down from _stack_top. Must stay the last dram_seg section. */
  .stack (NOLOAD) : ALIGN(16)
  {
    . += STACK_SIZE;
    _stack_top = .;
  } > dram_seg

  /DISCARD/ :
  {
    *(.comment)
    *(.note .note.*)
    *(.eh_frame .eh_frame.*)
  }
}
