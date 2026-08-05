/* SPDX-License-Identifier: Apache-2.0
 *
 * Minimal UART0 driver (freestanding, no ESP-IDF) for the ESP32-S3
 * QEMU C baseline.
 *
 * Registers (ESP32-S3 TRM / IDF v5.5 esp32s3 soc headers; UART0 base
 * 0x6000_0000 — NOT 0x6001_3000, which is the ESP32-C3 address):
 *   UART_FIFO   @ +0x00  — write TX byte / read RX byte
 *   UART_STATUS @ +0x1C  — TXFIFO_CNT in bits [25:16] (bytes in TX FIFO)
 *
 * The S3 TX FIFO is 128 entries. We poll until fewer than 64 entries are
 * queued before writing one byte (poll-based, no interrupts). Baud rate is
 * left at the ROM default (115200) — the benchmark never touches UART
 * config, and QEMU's `-serial file:` consumes the TX stream.
 */
#ifndef UART_H
#define UART_H

#include <stdint.h>

void uart_putc(char c);
void uart_puts(const char *s);
void uart_hex32(uint32_t v);       /* prints 0x + 8 hex digits */
void uart_dec32(uint32_t v);       /* prints unsigned decimal */

#endif /* UART_H */
