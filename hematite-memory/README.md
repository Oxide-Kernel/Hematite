# hematite-memory

Compile-time **USMP-style liveness arena planner** for **Hematite** — a
pure-Rust, `no_std` int8 neural-network inference engine for the
ESP32-S3.

`liveness_plan` computes tensor offsets and the peak intermediate size
(`ArenaPlan`) over an op schedule; `ScratchLayout` sizes per-op kernel
scratch. It runs at **macro time** (inside `#[model]`), so generated
models are zero-allocation: intermediates in one `[i8; ARENA_LEN]` stack
local, per-op scratch in caller-provided `[u8; SCRATCH_LEN]`.

Zero external dependencies. Full documentation:
<https://hematite.readthedocs.io/>.

License: Apache-2.0