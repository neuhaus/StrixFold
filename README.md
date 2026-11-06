# StrixFold

[![Discord](https://img.shields.io/badge/Discord-StrixFold-5865F2?logo=discord&logoColor=white)](https://discord.gg/D4FgAJnKd9)
[![X](https://img.shields.io/badge/X-@Shoestring__Dev-000000?logo=x&logoColor=white)](https://x.com/Shoestring_Dev)

TensorFold for AMD Strix Halo: DeepSeek-V4.1-Flash on two Strix Halo boxes, split across both over plain USB4
networking.

This is a fork of [TensorFold](https://github.com/ashhart/TensorFold) by way of
[Bertholomus AI's DeepSeek fork](https://github.com/bertholomus/TensorFold). It adds an AMD HIP backend, kernels
written for Strix Halo (gfx1151), and tensor parallel across two boxes over their USB4 ports. On two Ryzen AI Max+ 395
machines it runs DeepSeek-V4.1-Flash at about 72% of the speed of two NVIDIA DGX Sparks, with the same output quality.

![One chat: 52.8 tokens per second, against 13.3 for the previous engine and about 73 for two DGX Sparks](docs/img/one-chat.svg)

![Prompt reading: 1,543 tokens per second, against about 270 for the previous engine and 2,180 for two DGX Sparks](docs/img/prompt-reading.svg)

![Several chats at once: from 52.8 tokens per second for one chat to 108 for eight](docs/img/several-chats.svg)

## Results

| | One chat | Prompt reading | Eight chats at once |
|---|---:|---:|---:|
| DwarfStar, the previous two-box engine (18 Oct 2026) | 13.3 tok/s | ~270 tok/s | |
| This port, first run (19 Oct) | 20.6 tok/s | 829 tok/s | |
| This port now | **52.8 tok/s** | **1,543 tok/s** | **108 tok/s** |
| 2 × DGX Spark, same weights (reference) | ≈73 tok/s | 2,180 tok/s | |

- One chat is 72.3% of two DGX Sparks; prompt reading is 70.8%.
- Writing speeds are with drafting on, the shipped default (`DRAFTS=1` in `serve.conf`). With it off, one chat writes
  about 32 tok/s. Drafting gives exactly the same text either way.
- Writing speed holds at long context: from an empty prompt to 48K tokens of context it drops about 4%
  ([Speed at long context](docs/RESULTS.md#speed-at-long-context)).
- Quality matches the Spark build: MMLU-Pro 84.8%, HumanEval+ 92%.
- Context windows from 16K up to 1M tokens.
- How each number was measured, with the raw tables: [docs/RESULTS.md](docs/RESULTS.md).
- Numbers from other people's boxes: [docs/COMMUNITY-RESULTS.md](docs/COMMUNITY-RESULTS.md). Add yours.

## Tested on

| | |
|---|---|
| Boxes | 2 × AMD Ryzen AI Max+ 395 (Radeon 8060S, gfx1151), 128 GB unified memory each |
| Link | two USB4 cables (thunderbolt_net), MTU 1500 on both |
| GPU stack | ROCm 10.0.0-4 (HIP 7.15, clang 23), Linux kernel 7.2.7 |
| Toolchain | Zig 0.17.0, Python 3.13 |
| Model | DeepSeek-V4.1-Flash-1017 as an EXL3 2.9 bpw pack (197 GB), plus its Engram shards (203 GB). Not included: download them yourself. |

No NVIDIA driver, CUDA or NCCL is used. Text only: image input is not supported on this backend.

## What this fork adds

- An AMD HIP backend for the DeepSeek-V4.1 family. The branch this fork started from had no HIP code; TensorFold's main
  line has its own HIP backend, and the DeepSeek and two-box USB4 changes are being moved onto it as small pull requests
  ([#2065](https://github.com/ashhart/TensorFold/issues/2065), [#2066](https://github.com/ashhart/TensorFold/issues/2066)).
- Strix Halo kernels for the EXL3 weights, sparse attention, top-k and the routed experts.
- Tensor parallel across two boxes over plain USB4 networking, with prompt reading running while data moves between
  them, and the traffic split over both cables.
- Drafting that gives exactly the same text with drafts on or off, and switches itself off when six or more chats are
  running.
- Long context up to 1M tokens, and a fix for a crash between the two boxes on long prompts.
- Output checks (logits hashes and decode receipts) and the benchmark tools, so you can verify your own build.

The full list, file by file, is in [CHANGES.md](CHANGES.md).

## Build and run

The commands this port runs, in order (build, prepare the inputs, serve, check, benchmark), are in
[tools/dsv41-hip/BUILD_AND_RUN.md](tools/dsv41-hip/BUILD_AND_RUN.md).

## Tuning the two boxes

What made the link between the boxes fast and steady (MTU, which traffic goes on which cable, TCP retransmit floors,
power settings) is in [docs/TUNING.md](docs/TUNING.md). Most of it applies to any two-box Strix Halo setup.

## Known limits

- Tested on this exact setup only. Other Strix Halo machines, ROCm versions and cables may behave differently.
- Text only.
- Provided as-is. Issues are welcome; support is best effort.

## Community

Setup help, benchmarks, model recipes and two-box tuning are on the [StrixFold Discord](https://discord.gg/D4FgAJnKd9).
Bug reports are welcome there or as GitHub issues. Measured StrixFold on your own boxes? Send the numbers with the
**Results from your hardware** issue form and they go on [docs/COMMUNITY-RESULTS.md](docs/COMMUNITY-RESULTS.md).
News and releases are also posted on X as [@Shoestring_Dev](https://x.com/Shoestring_Dev).

## Credits and licence

Apache-2.0, as upstream. TensorFold is by Ash Hart and the TensorFold contributors. The DeepSeek-V4.1 family this
builds on is by Bertholomus AI (Albert Lee). The EXL3 weight format follows ExLlamaV3 (MIT), credited in the kernel
files. Upstream's LICENSE and notices are unchanged; this fork's changes are listed in [CHANGES.md](CHANGES.md) and
the NOTICE file. Upstream's own README is kept at [docs/UPSTREAM-README.md](docs/UPSTREAM-README.md).
