#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hematite Contributors.
"""Generate the minimal int8 HardSwish-only TFLite model for the
tools/tflm-goldens harness (todo T10: re-tier the hard_swish golden fixture
from the DOWNGRADED integer-rational approximation to executed TFLM output).

This is a DEV-time script — the output model is committed, never fetched at
build time (same policy as tools/generate_sine_model.py).

Model: int8 input [1,1,1,8] (scale 1.0, zero_point 0) -> HARD_SWISH ->
int8 output [1,1,1,8] (scale 1.0, zero_point 0).

Schema constants are the TF-2.14-era values of the vendored schema
(tools/tflite-schema/schema.fbs), which MATCH tflite-micro at the pinned SHA
18b9e6f2a8c5a9518e588f59c2ba16ef7ef9d551 (verified: schema_generated.h has
BuiltinOperator_HARD_SWISH = 117 there):
  BuiltinOperator.HARD_SWISH = 117
  BuiltinOptions union index of HardSwishOptions = 90 (NONE = 0)

The HardSwishOptions table is empty (schema: `table HardSwishOptions {}`).
TFLM's HardSwishInit/Prepare/Eval never read builtin_data, but the union
discriminator is still written for schema fidelity.

Deterministic: run twice -> byte-identical output.

Usage:
    python3 generate_hard_swish_model.py [out.tflite]
"""

from flatbuffers import Builder

INT8 = 9
HARD_SWISH = 117  # BuiltinOperator
HARD_SWISH_OPTIONS = 90  # BuiltinOptions union index
TFL3_IDENT = b"TFL3"


def generate():
    """Build the model with flatbuffers.Builder."""
    b = Builder(2048)

    def make_int32_vector(values):
        n = len(values)
        b.StartVector(4, n, 4)
        for v in reversed(values):
            b.PrependInt32(v)
        return b.EndVector(n)

    def make_int64_vector(values):
        n = len(values)
        b.StartVector(8, n, 8)
        for v in reversed(values):
            b.PrependInt64(v)
        return b.EndVector(n)

    def make_float_vector(values):
        n = len(values)
        b.StartVector(4, n, 4)
        for v in reversed(values):
            b.PrependFloat32(v)
        return b.EndVector(n)

    def make_table_vector(tables):
        n = len(tables)
        b.StartVector(4, n, 4)
        for t in reversed(tables):
            b.PrependUOffsetTRelative(t)
        return b.EndVector(n)

    # === QUANTIZATION (input: scale 1.0, zp 0) ===
    in_zp_vec = make_int64_vector([0])
    in_scale_vec = make_float_vector([1.0])
    b.StartObject(6)
    b.PrependUOffsetTRelativeSlot(5, 0, 0)  # quantized_dimension
    b.PrependUOffsetTRelativeSlot(4, 0, 0)  # details
    b.PrependUOffsetTRelativeSlot(3, in_zp_vec, 0)
    b.PrependUOffsetTRelativeSlot(2, in_scale_vec, 0)
    input_quant = b.EndObject()

    # === QUANTIZATION (output: scale 1.0, zp 0) ===
    out_zp_vec = make_int64_vector([0])
    out_scale_vec = make_float_vector([1.0])
    b.StartObject(6)
    b.PrependUOffsetTRelativeSlot(5, 0, 0)
    b.PrependUOffsetTRelativeSlot(4, 0, 0)
    b.PrependUOffsetTRelativeSlot(3, out_zp_vec, 0)
    b.PrependUOffsetTRelativeSlot(2, out_scale_vec, 0)
    output_quant = b.EndObject()

    # === TENSORS ===
    # Tensor 0: input [1,1,1,8] int8, buffer 1 (empty; harness copies input)
    t0_shape = make_int32_vector([1, 1, 1, 8])
    t0_name = b.CreateString("input")
    b.StartObject(7)
    b.PrependBoolSlot(5, False, False)  # is_variable
    b.PrependUOffsetTRelativeSlot(4, input_quant, 0)
    b.PrependUOffsetTRelativeSlot(3, t0_name, 0)
    b.PrependUint32Slot(2, 1, 0)  # buffer=1
    b.PrependInt8Slot(1, INT8, 0)
    b.PrependUOffsetTRelativeSlot(0, t0_shape, 0)
    tensor0 = b.EndObject()

    # Tensor 1: output [1,1,1,8] int8, buffer 0 (empty sentinel)
    t1_shape = make_int32_vector([1, 1, 1, 8])
    t1_name = b.CreateString("output")
    b.StartObject(7)
    b.PrependBoolSlot(5, False, False)
    b.PrependUOffsetTRelativeSlot(4, output_quant, 0)
    b.PrependUOffsetTRelativeSlot(3, t1_name, 0)
    b.PrependUint32Slot(2, 0, 0)  # buffer=0
    b.PrependInt8Slot(1, INT8, 0)
    b.PrependUOffsetTRelativeSlot(0, t1_shape, 0)
    tensor1 = b.EndObject()

    sg_tensors_vec = make_table_vector([tensor0, tensor1])

    # === HARD SWISH OPTIONS (empty table, union type 90) ===
    b.StartObject(0)
    hs_opts = b.EndObject()

    # === OPERATOR ===
    op_inputs_vec = make_int32_vector([0])
    op_outputs_vec = make_int32_vector([1])
    b.StartObject(6)
    b.PrependUOffsetTRelativeSlot(4, hs_opts, 0)  # builtin_options
    b.PrependUint8Slot(3, HARD_SWISH_OPTIONS, 0)  # builtin_options_type
    b.PrependUOffsetTRelativeSlot(2, op_outputs_vec, 0)
    b.PrependUOffsetTRelativeSlot(1, op_inputs_vec, 0)
    b.PrependUint32Slot(0, 0, 0)  # opcode_index
    op0 = b.EndObject()
    sg_ops_vec = make_table_vector([op0])

    # === SUBGRAPH ===
    sg_inputs_vec = make_int32_vector([0])
    sg_outputs_vec = make_int32_vector([1])
    sg_name = b.CreateString("main")
    b.StartObject(5)
    b.PrependUOffsetTRelativeSlot(4, sg_name, 0)
    b.PrependUOffsetTRelativeSlot(3, sg_ops_vec, 0)
    b.PrependUOffsetTRelativeSlot(2, sg_outputs_vec, 0)
    b.PrependUOffsetTRelativeSlot(1, sg_inputs_vec, 0)
    b.PrependUOffsetTRelativeSlot(0, sg_tensors_vec, 0)
    subgraph0 = b.EndObject()
    sg_vec = make_table_vector([subgraph0])

    # === OPERATOR CODE ===
    b.StartObject(4)
    b.PrependInt32Slot(3, HARD_SWISH, 0)  # builtin_code = 117
    b.PrependInt32Slot(2, 1, 1)  # version = 1
    opcode0 = b.EndObject()
    opcodes_vec = make_table_vector([opcode0])

    # === BUFFERS ===
    b.StartObject(3)
    buf0 = b.EndObject()  # Buffer 0: empty sentinel
    b.StartObject(3)
    buf1 = b.EndObject()  # Buffer 1: empty (input placeholder)
    bufs_vec = make_table_vector([buf0, buf1])

    # === MODEL ===
    b.StartObject(8)
    b.PrependUOffsetTRelativeSlot(4, bufs_vec, 0)
    b.PrependUOffsetTRelativeSlot(2, sg_vec, 0)
    b.PrependUOffsetTRelativeSlot(1, opcodes_vec, 0)
    b.PrependUint32Slot(0, 3, 0)  # version = 3
    model = b.EndObject()

    # === FINISH ===
    b.Finish(model, file_identifier=TFL3_IDENT)
    return b.Output()


if __name__ == "__main__":
    import sys

    buf = generate()
    out_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "models/hard_swish_int8.tflite"
    )
    with open(out_path, "wb") as f:
        f.write(buf)
    print(f"Written {len(buf)} bytes to {out_path}")
    print(f"First 8 bytes: {buf[:8].hex()}")
