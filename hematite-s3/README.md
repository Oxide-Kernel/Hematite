# hematite-s3

The **ESP32-S3 optimized backend** for **Hematite** — a pure-Rust,
`no_std` int8 neural-network inference library.

`S3Backend` implements the `KernelBackend` contract with **bespoke
Xtensa TIE728 SIMD kernels** written from scratch (Rust + inline
assembly, `asm/s8_accx_*.S`):

- conv1x1 (incl. channel-padded), conv3x3 (incl. SAME/stride-2),
  depthwise (dm=1 and dm>1, anytap, single-channel broadcast bc1),
  fully-connected (tiny-fc fast path), softmax, generic pool,
  elementwise, activations, reductions, data movement.
- **32-bit GPR accumulators** (`EE.VMULAS.S8.ACCX`) — bit-exact by
  construction vs. the scalar reference, unlike the 8-bit-saturating
  QACC-lane vendor kernels.
- SIMD dispatch is `cfg`-gated: on the host the same code runs the
  scalar fallback, bit-identical to the device.
- Host-compilable, so every kernel is testable without hardware.

Full documentation: <https://hematite.readthedocs.io/>.

License: Apache-2.0