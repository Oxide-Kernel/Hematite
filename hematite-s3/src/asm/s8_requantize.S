# s8_requantize.S — bespoke uniform requantize epilogue (bit-exact).
#
# SPDX-License-Identifier: Apache-2.0
#
    .text
    .align 4
    .global s8_requantize_uniform
    .type   s8_requantize_uniform, @function

# ---------------------------------------------------------------------------
# s8_requantize_uniform(ctx)  — windowed ABI, 1 arg (a2 = &ReqCtx repr(C)).
#
# In-place per-channel requantize for the UNIFORM fast path only:
#   mult == 1<<30 && shift == 1  -> identity:  out[oc] = clamp(raw+bias+off)
#   mult == 1<<30 && shift == 0  -> half-round: out[oc] = clamp((raw+bias+1)>>1 + off)
# (Any other uniform pair is NOT handled here; Rust keeps the general path.)
#
# ReqCtx layout (must match accx.rs, Xtensa 32-bit repr(C)):
#   0  accs.ptr    4  accs.len
#   8  bias.ptr    12 bias.len
#   16 mult.ptr    20 mult.len
#   24 shift.ptr   28 shift.len
#   32 output_offset   36 act_min   40 act_max   44 out_base
#   48 output.ptr   52 output.len
#   56 uniform_mult   60 uniform_shift
#
# The (acc+1)>>1 overflow-free form: srai(acc,1) + (acc&1) is bit-identical to
# ((acc as i64 + 1) >> 1) as i32 for EVERY i32 acc (verified; the naive
# add-then-srai wraps at i32::MAX).
# ---------------------------------------------------------------------------
s8_requantize_uniform:
    entry a1, 32

    l32i a3, a2, 0        # accs.ptr
    l32i a4, a2, 4        # n
    beqz a4, .Lq_done

    l32i a5, a2, 8        # bias.ptr
    l32i a6, a2, 48       # output.ptr
    l32i a7, a2, 44       # out_base
    l32i a8, a2, 32       # output_offset
    l32i a9, a2, 36       # act_min
    l32i a10, a2, 40      # act_max
    l32i a11, a2, 56      # uniform_mult
    l32i a12, a2, 60      # uniform_shift

    add a6, a6, a7        # out = output.ptr + out_base

    movi a13, 0x40
    slli a13, a13, 24     # 1 << 30
    bne a11, a13, .Lq_done    # defensive: Rust only calls for mult==1<<30
    beqi a12, 1, .Lq_identity
    beqz a12, .Lq_half
    j .Lq_done            # other shifts handled by Rust

# -- identity: out[oc] = clamp(acc + out_offset, min, max) ---------------
.Lq_identity:
    beqz a4, .Lq_done
    l32i a14, a3, 0       # raw
    l32i a15, a5, 0       # bias
    add a14, a14, a15     # acc (wraps, same as Rust)
    add a14, a14, a8      # + out_offset
    blt a14, a9, .Lq_id_low
    bge a14, a10, .Lq_id_high
    s8i a14, a6, 0
    j .Lq_id_next
.Lq_id_low:
    s8i a9, a6, 0
    j .Lq_id_next
.Lq_id_high:
    s8i a10, a6, 0
.Lq_id_next:
    addi a3, a3, 4
    addi a5, a5, 4
    addi a6, a6, 1
    addi a4, a4, -1
    bnez a4, .Lq_identity
    j .Lq_done

# -- half-round: out[oc] = clamp(((acc+1)>>1) + out_offset, min, max) ----
.Lq_half:
    beqz a4, .Lq_done
    l32i a14, a3, 0       # raw
    l32i a15, a5, 0       # bias
    add a14, a14, a15     # acc (wraps, same as Rust)
    srai a15, a14, 1      # acc >> 1 (floor)
    extui a14, a14, 0, 1  # acc & 1
    add a14, a15, a14     # (acc+1)>>1  overflow-free
    add a14, a14, a8      # + out_offset
    blt a14, a9, .Lq_half_low
    bge a14, a10, .Lq_half_high
    s8i a14, a6, 0
    j .Lq_half_next
.Lq_half_low:
    s8i a9, a6, 0
    j .Lq_half_next
.Lq_half_high:
    s8i a10, a6, 0
.Lq_half_next:
    addi a3, a3, 4
    addi a5, a5, 4
    addi a6, a6, 1
    addi a4, a4, -1
    bnez a4, .Lq_half
    j .Lq_done

.Lq_done:
    retw
    .size   s8_requantize_uniform, . - s8_requantize_uniform
