/* SPDX-License-Identifier: Apache-2.0
 *
 * ESP-IDF application descriptor (esp_app_desc_t), placed at the very start
 * of the first image segment (.rodata.appdesc, first in .rodata — see
 * linker.ld).
 *
 * The IDF v5.5 2nd-stage bootloader (the one espflash 4.5.0 embeds) casts
 * the first segment's data to esp_app_desc_t and reads
 * min_efuse_blk_rev_full / max_efuse_blk_rev_full from offsets 0xAC/0xB0
 * (bootloader_common_loader.c -> bootloader_common_check_efuse_blk_validity
 * called from esp_image_format.c process_segment, segment == 0).  Without a
 * real descriptor the bootloader reads arbitrary .rodata bytes and rejects
 * the app with "Image requires efuse blk rev >= vN.M".  Both revision fields
 * are 0 here, so the IS_FIELD_SET() gate is false and the check is skipped.
 *
 * The `magic` also lets espflash's save-image accept the descriptor symbol
 * without --ignore-app-descriptor.
 */
#include <stdint.h>

#define ESP_APP_DESC_MAGIC_WORD 0xABCD5432u

typedef struct {
    uint32_t magic;               /* ESP_APP_DESC_MAGIC_WORD */
    uint32_t secure_version;
    uint32_t version_len;         /* length of version[], incl. NUL */
    char version[32];
    char project_name[32];
    char time[16];
    char date[16];
    char idf_ver[32];
    uint8_t app_elf_sha256[32];   /* filled by espflash save-image */
    uint32_t min_efuse_blk_rev_full; /* 0 -> efuse blk rev check skipped */
    uint32_t max_efuse_blk_rev_full; /* 0 -> skipped */
    uint32_t reserved[4];
    uint8_t flags[2];
    uint8_t reserved2[2];
} esp_app_desc_t;

__attribute__((section(".rodata.appdesc"), used, aligned(4)))
const esp_app_desc_t esp_app_desc = {
    .magic = ESP_APP_DESC_MAGIC_WORD,
    .secure_version = 0,
    .version_len = 6u, /* sizeof("1.0.0") including NUL */
    .version = "1.0.0",
    .project_name = "hematite-qemu-baseline",
    .time = "00:00:00",
    .date = "20260805",
    .idf_ver = "freestanding (no ESP-IDF)",
    .app_elf_sha256 = {0},
    .min_efuse_blk_rev_full = 0,
    .max_efuse_blk_rev_full = 0,
    .reserved = {0, 0, 0, 0},
    .flags = {0, 0},
    .reserved2 = {0, 0},
};
