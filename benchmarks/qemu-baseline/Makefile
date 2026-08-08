# SPDX-License-Identifier: Apache-2.0
#
# Build + run the QEMU C baseline for ESP32-S3.
#
# Toolchain (pre-installed, no ESP-IDF needed):
#   - xtensa-esp32s3-elf-gcc (espup-installed esp-15.2.0_20250920)
#   - espflash 4.5.0 (embeds the ESP-IDF 2nd-stage bootloader)
#   - Espressif QEMU fork 9.2.2 at $(HOME)/.esp-qemu
#
# Pipeline: gcc (freestanding, call0 ABI) -> ELF -> espflash save-image
# --merge -> merged flash image -> qemu-system-xtensa -machine esp32s3.

SHELL := /bin/bash

QEMU     := $(HOME)/.esp-qemu/qemu/bin/qemu-system-xtensa
ESPFLASH := $(HOME)/.cargo/bin/espflash
XT_GCC   := $(shell find $(HOME)/.rustup/toolchains/esp -name 'xtensa-esp32s3-elf-gcc' 2>/dev/null | head -1)

CFLAGS  := -O2 -mabi=call0 -mtext-section-literals -ffreestanding -nostdlib \
           -fno-builtin -Wall -Wextra -ffunction-sections -fdata-sections
# startup.S must use -mtext-section-literals like the C files: it routes the
# assembler-relaxed movi->l32r constants into .xt.lit (which the Xtensa
# linker handles and we place at the end of .text), instead of .literal._start
# (which the linker rejects with "dangerous relocation: literal placed
# after use").
ASFLAGS := -O2 -mabi=call0 -mtext-section-literals -ffreestanding -nostdlib
# -lgcc provides the compiler runtime (__udivdi3/__umoddi3 for u64 math)
# that -nostdlib would otherwise leave out.
# No --relax: the Xtensa relaxation pass rejects the entry code's l32r
# pools with "dangerous relocation: literal placed after use" unless they
# are in .xt.lit; we instead place .xt.lit/.xt.prop explicitly at the end
# of .text in the linker script.
LDFLAGS := -nostdlib -T linker.ld -Wl,--gc-sections -Wl,--no-warn-rwx-segments -lgcc

OBJS := startup.o appdesc.o uart.o kernels.o main.o

all: baseline.bin

baseline.elf: $(OBJS) linker.ld
	$(XT_GCC) $(LDFLAGS) -o $@ $(OBJS)

%.o: %.c kernels.h uart.h
	$(XT_GCC) $(CFLAGS) -c $< -o $@

%.o: %.S
	$(XT_GCC) $(ASFLAGS) -c $< -o $@

baseline.bin: baseline.elf
	$(ESPFLASH) save-image --chip esp32s3 --merge $< $@ -S

# Run under QEMU with serial to a file (-monitor none so the monitor does not
# share stdio; -serial file: is the UART0 console). QEMU never exits on its
# own: run-log kills it after RUN_SECONDS.
RUN_SECONDS ?= 30
SERIAL_OUT ?= serial.log
run: baseline.bin
	$(QEMU) -nographic -machine esp32s3 -drive file=$<,if=mtd,format=raw -monitor none -serial file:$(SERIAL_OUT) -icount 3 & \
	QPID=$$!; sleep $(RUN_SECONDS); kill -INT $$QPID 2>/dev/null; wait $$QPID 2>/dev/null; true

run-log: baseline.bin
	$(QEMU) -nographic -machine esp32s3 -drive file=$<,if=mtd,format=raw -monitor none -serial file:run1.log -icount 3 & \
	QPID=$$!; sleep $(RUN_SECONDS); kill -INT $$QPID 2>/dev/null; wait $$QPID 2>/dev/null; true

clean:
	rm -f $(OBJS) baseline.elf baseline.bin run1.log serial.log

.PHONY: all run run-log clean

print-env:
	@echo "XT_GCC:   $(XT_GCC)"
	@$(XT_GCC) --version | head -1
	@echo "ESPFLASH: $(ESPFLASH)"
	@$(ESPFLASH) --version
	@echo "QEMU:     $(QEMU)"
	@$(QEMU) --version | head -1
