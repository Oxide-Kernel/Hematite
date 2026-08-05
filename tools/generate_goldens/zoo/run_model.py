#!/usr/bin/env python3
"""Per-model golden harness: run a .tflite model through a real executed TFLite
interpreter and emit the exact int8 output for the Rust generator.

This is the "documented cross-check" reference path for model-level goldens:
the values come from an executed TFLite runtime (ai-edge-litert, the successor
to tflite-runtime), NOT from hand computation.

Usage:
    run_model.py <model.tflite> <output.txt>

Input: filled with the tensor zero_point (deterministic neutral input).

Output format (plain text, deliberately dependency-free for the Rust side):
    line 1: ai-edge-litert <version>
    line 2: input shape, space-separated ints
    line 3: output shape, space-separated ints
    line 4: input zero_point
    line 5: output dtype (int8/int32/...)
    line 6: comma-separated output values
"""

import sys

import numpy as np

from ai_edge_litert.interpreter import Interpreter
from importlib.metadata import version as pkg_version


def main() -> None:
    if len(sys.argv) < 3:
        print("usage: run_model.py <model.tflite> <output.txt>", file=sys.stderr)
        sys.exit(2)

    model_path = sys.argv[1]
    out_path = sys.argv[2]

    interp = Interpreter(model_path=model_path)
    interp.allocate_tensors()

    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]

    shape = tuple(int(x) for x in in_d["shape"])
    qp = in_d["quantization_parameters"]
    zp = int(qp["zero_points"][0]) if qp.get("zero_points") is not None else 0

    data = np.full(shape, zp, dtype=in_d["dtype"])
    interp.set_tensor(in_d["index"], data)
    interp.invoke()

    res = interp.get_tensor(out_d["index"])
    out_shape = [int(x) for x in res.shape]
    out_dtype = np.dtype(out_d["dtype"]).name
    out_list = res.reshape(-1).tolist()

    in_shape = list(shape)
    lines = [
        pkg_version("ai-edge-litert"),
        " ".join(str(x) for x in in_shape),
        " ".join(str(x) for x in out_shape),
        str(zp),
        out_dtype,
        ",".join(str(x) for x in out_list),
    ]
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"captured {model_path} -> {out_path} (out {out_shape}, {out_dtype}, {len(out_list)} elems)")


if __name__ == "__main__":
    main()
