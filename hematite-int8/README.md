# hematite-int8

TFLite-Micro-exact int8 quantization math for **Hematite** — a pure-Rust,
`no_std` int8 neural-network inference library.

- `multiply_by_quantized_multiplier` — pure **32-bit** TFLM single-rounding
  requantize (16-bit limb decomposition, no i64 software emulation on
  Xtensa); bit-exact vs. the i64 reference (exhaustive host test).
- `rounding_divide_by_pot` — TFLM `RoundingDivideByPOT`.
- `saturating_cast` — TFLM i32→i8 saturation.
- `requantize` — full per-channel requantize.
- `quantize_multiplier` — float scale → `(multiplier, shift)`.

`no_std`, no floating-point code in the default build. Full documentation:
<https://hematite.readthedocs.io/>.

License: Apache-2.0