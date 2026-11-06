# Changes in this fork

This fork starts from Bertholomus AI's DeepSeek-V4.1 fork of TensorFold at commit 3ff3b2ac (18 Oct 2026). Every change
below is ours, made between 18 October and 6 November 2026 and released under Apache-2.0 as upstream. Upstream's LICENSE,
LICENSES/ and THIRD_PARTY_NOTICES.md are unchanged; NOTICE gains one section at its end.

## Added

| Path | What it is |
|---|---|
| `zig/src/gpu/hip/` | The AMD HIP backend: devices, memory, streams, events, graphs, staging and the kit loader |
| `zig/src/gpu/` (kernel, interface, manifest, staging, msgpack, kit_tool) | A backend-neutral GPU layer the CUDA and HIP paths share |
| `zig/src/native/hip.zig` | The native HIP entry point |
| `zig/kernels/hip/dsv41/` | Strix Halo (gfx1151) kernels: EXL3 weight matmul, sparse attention, top-k, routed experts, prompt reading, RoPE, norms |
| `zig/src/families/dsv41/` (16 new files) | The DeepSeek-V4.1 family's HIP paths: two-box collectives over TCP, the cable stripe, kept states for long context, drafting switch |
| `zig/src/core/lanes/` (8 new files) | Grammar lanes and lane scheduling used by the server |
| `zig/tests/dsv41_hip/` | GPU smoke, ABI, glue, code-object and collectives tests |
| `zig/src/server/` (reply_dsv41, tests) | DeepSeek-V4.1 reply handling on this backend, and tests for top-k, structure and telemetry |
| `tools/dsv41-hip/` | Build, input preparation, serving scripts, output checks (gates, numerics, qualification), the benchmark, and the design notes (ADAPTER, COLLECTIVES-DESIGN, KERNELS, BUILD_AND_RUN, RESULTS) |
| `tools/dsv41-hip/` input tools (prepare_inputs.py, inventory_checkpoint.py, input_common.py, snapshot_files.py, source_lock.json, assets/) | The CPU input preparation `prepare_inputs.sh` and `verify_inputs.sh` run, vendored unchanged from the component kit at def87879; the chat template is Tech2wild's (MIT, see NOTICE) |
| `tools/dsv41-zig/inputs/`, `tools/dsv41-zig/kit/` | Token map and per-node input preparation, and kit packing |
| `docs/RESULTS.md`, `docs/TUNING.md`, `docs/img/` | Measured results with charts, and the two-box tuning notes |
| `tools/dsv41-hip/ctx_sweep.py` | Writing speed against context length, drafts on and off (RESULTS, Speed at long context) |
| `docs/COMMUNITY-RESULTS.md`, `.github/ISSUE_TEMPLATE/` | Results from other people's hardware, and the issue form that collects them |
| `README.md`, `CHANGES.md` | This fork's README and this file. Upstream's README moved to `docs/UPSTREAM-README.md`. |

## Modified upstream files

| Path | Change |
|---|---|
| `build.zig`, `zig/build/cuda.zig` | Build steps for the HIP backend and its kernels |
| `zig/src/core/engine_api.zig`, `lane_host.zig`, `lanes/*.zig` | The lane host and engine API work across both GPU backends; requests wait for room in the pool |
| `zig/src/families/dsv41/*.zig` (42 files) | The family runs on HIP as well as CUDA: weights, prompt and decode paths, sampling, drafting, checkpoints and the two-box link |
| `zig/src/native/cuda.zig` | The CUDA entry point uses the shared GPU layer |
| `zig/src/server/*.zig` (9 files) | Reply parity with the official DeepSeek encoder and parser on this backend, and server settings for two-box serving |

The exact diff against upstream is `git diff 3ff3b2ac..main`.
