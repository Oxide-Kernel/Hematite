/* SPDX-License-Identifier: Apache-2.0
 *
 * Minimal UART0 driver (freestanding, no ESP-IDF) for the ESP32-S3
 * QEMU C baseline. See uart.h for the register map notes.
 *
 * NOTE: the ESP32-S3 UART0 base is 0x6000_0000 (NOT 0x6001_3000 — that is
 * the ESP32-C3 address).  Verified against the IDF v5.5 esp32s3 soc headers
 * (REG_UART_BASE(i) = DR_REG_UART_BASE + i*0x10000 + (i>1 ? 0xe000 : 0))
 * and QEMU's esp32s3_reg.h (DR_REG_UART_BASE = 0x60000000).
 */
#include "uart.h"

#define UART0_BASE   0x60000000UL
#define UART0_FIFO   (*(volatile uint8_t *)(UART0_BASE + 0x00))
#define UART0_STATUS (*(volatile uint32_t *)(UART0_BASE + 0x1C))

/* TX FIFO count lives in UART_STATUS bits [25:16] (IDF v5.5 esp32s3
 * uart_reg.h: UART_TXFIFO_CNT_S = 16, mask 0x3FF).  Wait until fewer than
 * 64 bytes are queued before writing. */
static inline void uart_wait_tx_room(void)
{
    while (((UART0_STATUS >> 16) & 0x3FFu) >= 64u) {
        /* spin */
    }
}

void uart_putc(char c)
{
    uart_wait_tx_room();
    UART0_FIFO = (uint8_t)c;
}

void uart_puts(const char *s)
{
    while (*s != '\0') {
        uart_putc(*s++);
    }
}

void uart_hex32(uint32_t v)
{
    static const char hex[] = "0123456789abcdef";
    uart_puts("0x");
    for (int shift = 28; shift >= 0; shift -= 4) {
        uart_putc(hex[(v >> shift) & 0xFu]);
    }
}

void uart_dec32(uint32_t v)
{
    char buf[10];
    int i = 0;
    if (v == 0) {
        uart_putc('0');
        return;
    }
    while (v > 0) {
        buf[i++] = (char)('0' + (v % 10));
        v /= 10;
    }
    while (i > 0) {
        uart_putc(buf[--i]);
    }
}
