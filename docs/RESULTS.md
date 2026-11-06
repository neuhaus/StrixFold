# Results

Measured on the two boxes described in the README, 18–22 October 2026. Nothing here is projected.

## How we measure

- **Prompt set:** the same eight categories and prompts as the published two-DGX-Spark run: coding, json, narrative,
  prose, math, reasoning, summary and format.
- **Settings:** temperature 0, thinking off, every prompt cold, five repetitions, and the quiet windows on both boxes.
- **One chat (decode):** completion tokens minus one, divided by the time from the first content event to the end. We
  report the mean of the eight category medians.
- **Prompt reading:** one output token, the whole request timed.
- **Several chats:** the same prompts sent as 1 to 8 conversations at once. The total tokens per second is the mean of
  the eight category medians.
- **Raw data:** every run writes a manifest with the commit, binaries, kit objects, configuration, toolchain and the
  prompt token ids.

## One chat

![One chat](img/one-chat.svg)

| Engine | Date | Tokens per second | Against 2 × DGX Spark |
|---|---|---:|---:|
| DwarfStar (kyuz0's ROCm port of ds4), Q2 GGUF | 18 Oct | 13.3 | |
| This port, first working run | 19 Oct | 20.6 | |
| This port, two-cable build | 20 Oct | **52.84** | **72.3%** |

DwarfStar reaches 14.7 tok/s with the CPU set to its performance setting. The boxes serve at the default balanced
power setting.

## Prompt reading

![Prompt reading](img/prompt-reading.svg)

| Engine | Prompt | Tokens per second | Against 2 × DGX Spark |
|---|---:|---:|---:|
| DwarfStar | 8,159 tokens | ~268–285 | |
| This port, first run | 8,165 tokens | 829 | |
| This port now | 2,952 tokens | 1,413.8 | 73.6% (Spark 1,921.6) |
| This port now | 11,594 tokens | **1,543.4** | **70.8%** (Spark 2,180.2) |

## Several chats at once

![Several chats at once](img/several-chats.svg)

| Chats at once | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Total tokens per second | 52.84 | 72.26 | 84.15 | 88.51 | 91.04 | 95.07 | 101.01 | 107.96 |

Drafting is on. From six chats at once, the drafting switch stops drafting, which gives the 7- and 8-chat numbers
their lift (+5.5% and +10.4%). Spark's published run has single batches for 1–4 chats and no data for 5–8, so we only
compare one chat and prompt reading against it.

## Quality

The same weights give the same quality as the Spark build: MMLU-Pro 84.8% and HumanEval+ 92%. The output checks
(`tools/dsv41-hip/gates.json`, `qualify.sh`) hold the logits hashes and decode receipts each build must match.
Drafts on gives exactly the same text as drafts off.

## Speed at long context

Measured 5 November 2026 on the deployed build: one chat, the same coding task every time with real code in front of
it as context (this repository's own Zig sources, a different slice for each request so nothing is cached), 400 tokens
written, two runs at each length, drafting on and off. Writing speed is in tokens per second; prompt reading is the
prompt's tokens over the time to the first token.

| Context | Drafting off | Drafting on | Prompt reading |
|---|---:|---:|---:|
| the task alone (~50 tokens) | 32.7 | 79.1 | |
| 2K | 32.3 | 72.9 | 1,171 |
| 4K | 32.4 | 77.0 | 1,281 |
| 8K | 32.4 | 72.9 | 1,344 |
| 16K | 31.8 | 74.7 | 1,369 |
| 32K | 31.5 | 75.7 | 1,377 |
| 48K | 31.5 | 73.7 | 1,379 |

- With drafting off, writing loses about 4% from the task alone to 48K tokens of context.
- With drafting on, a round takes 59-66 ms at every length and keeps 4.4-4.9 tokens. What moves the drafted number is
  how many drafted tokens are kept, and that depends on the text being written, not on the length of the context.
- This is one coding task, which drafts well, so its drafted speed is above the eight-category 52.8.

To measure your own pair: `python3 tools/dsv41-hip/ctx_sweep.py http://127.0.0.1:18080 dsv41 zig/src
0,2000,4000,8000,16000,32000,48000 2 sweep.json RUN/rank0.log` (the log adds rounds and accepted drafts).

## Long context

Context windows from 16K up to 1M tokens. At 1M, a needle test found all three needles (run on a fine-tune with the same layout). The first read of a 1M
prompt takes 948 s; asking again with the prompt cached takes 2.3 s.

One 1M conversation on the official weights (6 November 2026, drafting on): a 1,030,061-token prompt of code read in
962 s (1,071 tok/s), and the fact hidden 44% of the way in came back. The next turn resumed from the kept state and its
first token came 3.9 s later. The two short replies (54 and 78 tokens) wrote at 40.1 and 48.9 tok/s; at 1M a
drafted round takes about 100 ms, against about 63 ms at short context.

All chats share one context window, and its length is capped by the RoPE tables' rows, so a 1M window holds one 1M
conversation, or several shorter ones whose lengths add up to it.

The older, more detailed engineering results (19 Oct, per-category tables) are in `tools/dsv41-hip/RESULTS.md`.
