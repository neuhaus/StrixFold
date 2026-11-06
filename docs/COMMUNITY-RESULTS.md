# Results from your hardware

StrixFold's published numbers come from one pair of boxes. This page collects runs on other people's: other Strix Halo
brands, other cables, network cards instead of USB4, other ROCm and kernel versions. Numbers from hardware we don't own
are the best check on ours.

## Add yours

Open an issue with the **Results from your hardware** form (New issue on GitHub), or post the same details in the
`#benchmarks` channel on the [Discord](https://discord.gg/D4FgAJnKd9). We add each run here with your name or handle,
unless you ask us not to.

What to run, on the build you serve:

1. `tools/dsv41-hip/serve-rank0.sh bench`: the code, prose and prompt-reading cases (`RUN/bench.json`).
2. `python3 tools/dsv41-hip/ctx_sweep.py http://127.0.0.1:18080 dsv41 zig/src 0,2000,8000,32000 2 sweep.json RUN/rank0.log`:
   writing speed against context length.
3. Optional: the full prompt set with `tools/dsv41-hip/qual/qual_bench.sh` (BUILD_AND_RUN.md section 6), the method
   behind the published table.

Say which settings you served with. `DRAFTS` matters most: drafting on (the default) roughly doubles writing speed and
gives exactly the same text.

## Runs

| Who | Boxes | Link | GPU stack | Settings | Code, prose (tok/s) | Prompt reading | Notes |
|---|---|---|---|---|---:|---:|---|
| StrixFold (6 Nov 2026) | 2 × AMD Ryzen AI Halo, 128 GB | 2 × USB4, MTU 1500 | ROCm 10.0.0-4, kernel 7.2.7 | `serve.conf` defaults, `DRAFTS=1` | 74.5, 42.4 | 1,381 at 1,818 tokens | balanced power; the bench cases above |
