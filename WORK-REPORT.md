# StrixFold HIP verbs work report

Last updated November 7, 2026 (Europe/Berlin).

## Host-buffer SEND versus WRITE measurements, November 7

The first SEND/WRITE comparison used the Donnerkeule GPU producer/verifier harness with **host-registered payload buffers** (`--host-mr`), not direct GPU DMA-BUF payloads. Both ranks used identical benchmark binaries, StrixFold's ROCm 10.1 runtime/container launcher, the production verbs provider, and the existing Donnerkeule options. Both cables remained connected. TCP closed after QP/MR bootstrap and every verification line reported `tcp_data_calls=0`. No StrixFold transport implementation was changed.

The older installed October 16 harness did not support persistent SEND receives. The current harness source was therefore compiled with ROCm 10.1 on fuzzy and deployed identically to misty. Binary SHA-256 is `13372cc2323edec087407b8a7aaa0bb09c51e58c884ee7450764f6b99fadf399`. The system HIP 6.4 headers lacked the DMA-BUF enum and were unsuitable for this source. Build script, original test sources, source/binary/provider hashes, and driver counters are preserved with the measurements.

The three modes were WRITE_WITH_IMM with reusable notification receives and a verbs readiness barrier for safe window reuse, SEND with receives posted per iteration, and SEND with payload receives kept pre-posted. The last mode removes the per-message readiness barrier. Message sizes were 64 B, 4 KiB, 16 KiB, 64 KiB, 1 MiB, 4 MiB, and 20 MiB at depth 1. The three largest sizes were also measured at depth 3. Each case ran three rounds, with 128 iterations per round through 64 KiB and 16 iterations above that, and a 5000-us polling spin budget. Mode order alternated with size/depth. QP setup and memory registration were outside the timed loop. The harness includes GPU production, host staging, receive-side copy, and GPU byte/guard verification. Its end-to-end times include all those phases. They are not pure wire latency or a prediction of model token throughput.

Each round reports its average microseconds per payload message. The summary takes the slower rank's average in each paired round, then the median of the three rounds. All 30 paired cases passed, giving 180 rank-round verification passes and 12,672 verified payload messages (30,397,317,120 payload bytes across both directions), each with a 64-byte guard. Driver error/retry counters checked at the end showed no data-WR retransmission, RNR retransmission, timeout, or copy/path-send errors.

| Payload, depth 1 | WRITE_WITH_IMM cycle, us | SEND cycle, us | Pre-posted SEND cycle, us | Pre-posted SEND cycle change versus WRITE |
| --- | --- | --- | --- | --- |
| 64 B | 173.8 | 2301.9 | 1126.0 | 6.48x as long |
| 4 KiB | 200.4 | 182.6 | 139.6 | 30.4% shorter |
| 16 KiB | 210.9 | 212.3 | 179.6 | 14.9% shorter |
| 64 KiB | 269.2 | 230.4 | 171.6 | 36.3% shorter |
| 1 MiB | 1168.8 | 1021.5 | 949.6 | 18.8% shorter |
| 4 MiB | 1843.0 | 3393.9 | 1866.4 | 1.3% longer |
| 20 MiB | 8731.3 | 10308.7 | 8405.0 | 3.7% shorter |

The saved PHASE records separate GPU production/staging/verification from readiness, WR posting, and CQ waiting. Summing the latter three for the slower rank in each round gives the following protocol-phase medians. These include synchronization and peer progress, so they are still broader than physical link latency.

| Payload, depth 1 | WRITE protocol phases, us | SEND protocol phases, us | Pre-posted SEND protocol phases, us |
| --- | --- | --- | --- |
| 64 B | 88.4 | 2084.4 | 900.0 |
| 4 KiB | 111.5 | 100.4 | 69.0 |
| 16 KiB | 116.8 | 121.2 | 97.6 |
| 64 KiB | 138.6 | 116.5 | 84.6 |
| 1 MiB | 858.0 | 756.6 | 720.8 |
| 4 MiB | 1369.1 | 2605.7 | 1345.2 |
| 20 MiB | 5934.9 | 7860.4 | 5845.3 |

| Payload, depth 3 | WRITE effective GiB/s | SEND effective GiB/s | Pre-posted SEND effective GiB/s |
| --- | --- | --- | --- |
| 1 MiB | 1.45 | 1.37 | 1.62 |
| 4 MiB | 2.16 | 1.54 | 2.09 |
| 20 MiB | 2.70 | 2.52 | 2.51 |

Throughput here is payload bytes divided by the end-to-end cycle time per message, per direction, with simultaneous traffic in both directions. It includes staging and checking overhead. The 20 MiB depth-3 persistent SEND case had one slow round at 28.68 ms/message versus 7.66 and 7.79 ms in the other rounds. That outlier is retained in the raw data and ranges. The 64-byte SEND regression also remains unexplained. Results support further evaluation of pre-posted SEND for medium-sized gathers, while providing no consistent large-message improvement or basis for replacing WRITE everywhere.

Artifacts, complete logs, reproducible controller, and summaries are under `benchmarks/2026-11-07-send-write/`. The existing WRITE-based StrixFold service was restored in run `20261107-restored-after-send-write`. Its Paris probe passed and the public `/health` endpoint on `192.168.211.2:8000` responded successfully from the Mac.

## TCP versus Donnerkeule verbs, November 7

### Stock Thunderbolt TCP follow-up

The follow-up replaces the Ethernet TCP baseline with stock `thunderbolt_net` on both physical USB4 cables, MTU 9000, `TP_STRIPE` set to the second cable's address, and `COLL_PROXY_STRIPE_KIB=16`. The main proxy runs on CPU 6 and the second on CPU 7. The same benchmark client, prompt hashes, C=1, 256-token output limit, drafting, graphs, and context pool are used. Three repetitions were measured per prompt length. Donnerkeule services were initially unbound to free DMA rings. At the user's request, `thunderbolt_ibverbs` was then completely unloaded on both hosts, and the full TCP measurement was repeated. Only that second, clean run is included below.

| Prompt tokens | Transport | Median first output, s | Effective prompt throughput, tokens/s | Median decode, tokens/s | Median request duration, s |
| --- | --- | --- | --- | --- | --- |
| 2000 | TCP over two Thunderbolt cables | 1.6330 | 1224.8 | 63.50 | 5.649 |
| 2000 | Donnerkeule verbs | 1.5586 | 1283.2 | 55.59 | 6.146 |
| 63000 | TCP over two Thunderbolt cables | 42.3485 | 1487.7 | 64.50 | 46.315 |
| 63000 | Donnerkeule verbs | 39.0799 | 1612.1 | 57.82 | 43.491 |

With this TCP baseline, verbs reaches first output about 4.6% sooner at 2K and 7.7% sooner at 63K. TCP generates about 14.2% and 11.6% more tokens per second, respectively. Text-output hashes match the original measurements for every request. Small cumulative stock-netdev RX error counters were observed during testing. Application requests completed successfully. Network-counter evidence is saved beside the results.

### Generation diagnosis and proxy design

An additional 2K control alternates requests with drafting enabled and disabled, three repetitions each. The client records server-side generation rate, per-round costs, and speculative counters. With drafting enabled, both transports perform 67 rounds and accept 188 draft tokens. Median server-side rates are 63.41 tokens/s for two-cable TCP and 55.58 for verbs. With drafting disabled, both perform 255 rounds. Median rates are 31.31 and 27.18 tokens/s. The difference persists with the same round counts and without speculative work. Median one-token round duration is about 31.93 ms for TCP and 36.79 ms for verbs. This locates the difference in executing the rounds rather than in different draft acceptance.

An additional single-physical-cable TCP control uses one proxy thread with the same MTU, prompt, output limit, and alternating draft settings. All three runs per setting again use 67 rounds and 188 accepted drafts when drafting, or 255 rounds without drafting.

| 2K generation control | Median server tokens/s, drafts on | Median server tokens/s, drafts off |
| --- | --- | --- |
| TCP, one physical cable and one proxy | 58.71 | 30.68 |
| TCP, two physical cables and two proxies | 63.41 | 31.31 |
| Verbs, one application graph proxy, driver uses both physical cables | 55.58 | 27.18 |

The second TCP cable/proxy increases drafted generation by about 8.0%, and non-drafted generation by 2.1%. Even the one-cable TCP control is faster than the current verbs integration. Consequently, a second application worker alone does not explain the full difference.

The verbs application path copies each payload into a registered transmit window, sends a receive-credit notification, waits for the peer credit, posts WRITE_WITH_IMM, processes completions, and copies received bytes into the proxy slab. The active descriptor backend in Donnerkeule also uses an ordered TX workqueue. These are concrete differences from TCP's two proxy workers. Their individual contributions have not yet been separately timed. Adding a second application worker does not by itself make that ordered kernel workqueue concurrent. No transport implementation changes were made during this diagnosis.

The main proxy already shares `DataWire` between TCP and verbs. Its second worker is TCP-specific. Sharing that worker with a second verbs QP could reduce duplicated transport-specific scheduling and error handling. Current C=1 graph execution generally waits for each gather before progressing, so assigning whole successive gathers to alternating workers alone does not establish simultaneous use of both workers. A second worker's performance benefit needs measurement.

Important correction to an intermediate diagnosis. The QPs' home devices are in domain 0, but this does **not** confine payloads to the first cable. Kernel source collects eligible paths across peers with the same remote-host identity, and live counters increased on both domains during the verbs control. Both cables are already carrying verbs payloads. The earlier one-cable inference from `usb4_rdma0` was incorrect. Logical `lanes=2` and physical cable count are distinct. Counter evidence is saved as `verbs-peer-counters.txt`.

After these measurements, stock `thunderbolt_net` was unloaded to release the DMA rings and Donnerkeule was reloaded using its original installed options. The public verbs service was restored in run `20261107-restored-after-thunderbolt-tcp`, with pool 65487, drafting enabled, and HTTP on `0.0.0.0:8000`. Its Paris probe and a network health check from the Mac passed. Switching between these full-ring configurations requires releasing the competing driver resources.

Measured on fuzzy and misty with the same `a4e4814` runtime binaries, verified weights, kernel kit, and input bundle. Upstream default changes were integrated locally as `639686d` and the deployed configuration enables drafting. Each run used TP2, one stream, one concurrent request, `DRAFTS=1`, HIP graphs, pool 65487 (63470 usable tokens), and no retained prompt cache. Only the transport changed between the two benchmark configurations. TCP used `COLL_KIND=proxy` over `eno1` at 2500 Mb/s full duplex. Verbs used `COLL_KIND=verbs` over `usb4_rdma0`, one rail, and 4 MiB host windows.

The API was isolated on loopback port 18081 during measurement. After a 128-token warmup, each transport processed exactly 2000 and 63000 prompt tokens, three times each, producing 256 output tokens per request. Prompt tokens were checked with `/v1/tokenize` and response usage. All requests reported zero cached prompt tokens. Synthetic filler precedes the same LRU-cache coding task. These are warm-runtime measurements with a cold prompt cache. Operating-system page caches were not cleared. Verbs was measured first, followed by TCP. C=4 was deferred at the user's request.

| Prompt tokens | Transport | Median first output, s | Range, s | Median decode, tokens/s | Median request duration, s |
| --- | --- | --- | --- | --- | --- |
| 2000 | TCP Ethernet | 6.8829 | 6.8187–6.8925 | 41.62 | 13.008 |
| 2000 | Donnerkeule verbs | 1.5586 | 1.4380–1.5591 | 55.59 | 6.146 |
| 63000 | TCP Ethernet | 200.7627 | 199.9994–201.2367 | 42.43 | 206.772 |
| 63000 | Donnerkeule verbs | 39.0799 | 39.0650–39.0848 | 57.82 | 43.491 |

Verbs improved median time to first output by 4.42x at 2K and 5.14x at 63K. Median decode rate increased by 33.6% and 36.3%, respectively. Whole-request duration improved by 2.12x and 4.75x. First-output time includes prompt processing and the first emitted text. Decode uses the existing `bench_stream.py` convention `(completion_tokens - 1) / (last_text_time - first_text_time)` with speculative streaming bursts.

Prompt-processing throughput is estimated as `prompt_tokens / ttft_s` for each request, then summarized across the three repetitions. Time to first output includes request handling and generation of the first emitted text, so this measures effective prompt throughput with that overhead included. The measurements did not record a separate server-side prefill duration.

| Prompt tokens | Transport | Median effective prompt throughput, tokens/s | Range, tokens/s |
| --- | --- | --- | --- |
| 2000 | TCP Ethernet | 290.6 | 290.2–293.3 |
| 2000 | Donnerkeule verbs | 1283.2 | 1282.8–1390.8 |
| 63000 | TCP Ethernet | 313.8 | 313.1–315.0 |
| 63000 | Donnerkeule verbs | 1612.1 | 1611.9–1612.7 |

The median effective prompt throughput improves by 4.42x at 2K and 5.14x at 63K with verbs.

Every repetition and both transports returned identical text for each prompt, with matching server token-hash prefixes (`863e5f2ae169` for 2K and `8d5601666bda` for 63K). Draft widths and accepted-draft counts varied with measured transport timing. This did not change the outputs. No transport errors occurred during these runs.

Raw measurements, replies, benchmark client, configurations, and rank 0 logs are under `benchmarks/2026-11-07-tcp-verbs/`. The regular public verbs service was restored on `0.0.0.0:8000` in run `20261107-restored-verbs`. Its Paris probe passed, and `/health` was successfully reached from the Mac at `http://192.168.211.2:8000/health`.

## Source and design

- Local branch `codex/strixfold-hip-verbs`. Commit `6424768` adds verbs for HIP collectives, and `a4e4814` moves lane commands to verbs. Base `2f832d6`.
- `COLL_KIND=verbs` uses one Donnerkeule USB4 rail. The graph proxy and the prefill copy pipeline each have their own RC queue pair. A zero-byte `SEND_WITH_IMM` grants a receive credit, and `WRITE_WITH_IMM` carries each payload between registered host buffers.
- In verbs mode, a third RC queue pair uses SEND for prompt token IDs and commands that start work on the follower. Ethernet TCP exchanges startup information and stays open to detect an idle peer exit. USB4 carries the command frames, tensor payloads, and per-write credits. TCP remains the default transport.
- The DeepSeek HIP engine can reuse KV prefixes held in GPU memory during serving. It has no disk persistence for KV state. The disk rank caches hold packed model weights. The shared CLI exposes `--learn` for the GLM Metal host, which implements disk-backed prompt states separately.
- Large writes are configurable with `COLL_CHUNK_MIB`. Both ranks agree on the setting at startup. The default is 4 MiB.

## Verification

| Check | Result |
| --- | --- |
| Zig 0.17.0, ROCm 10.1 native build | 117/117 steps succeeded. 149 host tests passed and one opt-in hardware test was skipped in the host suite. |
| Standalone RC wire on fuzzy and misty | Bidirectional bytes, fragmented messages, planted mismatch, and peer exit passed before the kernel upgrade. |
| Kernel after reboot | Both hosts booted `7.2.15-100.fc43.x86_64`. Each `usb4_rdma0` port advertised a 248 MiB maximum message and 4096-byte active MTU. |
| HIP collective harness, 4 MiB window | Each host checked 2,642 collective calls and five replays of a 94-gather HIP graph, with zero bad words. The workload included 40 MiB and 200 MiB prefill gathers. |
| HIP collective harness, 200 MiB window | Same checks passed with zero bad words and a 209,715,248-byte registered window (200 MiB plus the header). |
| Planted prefill mismatch | Both ranks returned `Mismatch` after about 1 ms. |
| Abrupt peer exit during prefill | After misty’s test process exited with SIGKILL, fuzzy returned `Timeout` in 2,001 ms. The first attempt ran the test as container PID 1, which prevented its self-kill. The corrected runner used a child process. |
| Verbs lane frames on both hosts | The updated harness sent six ordered frames up to 4,259,844 bytes and checked every byte on rank 1. Both hosts then passed 2,642 collectives and five graph replays with zero bad words. |
| Native build with verbs lane commands | 99/99 build steps succeeded. 202 host tests passed and one hardware-only test was skipped. The container lacked `strings`, so the NVIDIA dependency string scan was run separately with the host tool and passed for both serving binaries. |
| Serving binary deployment | Both hosts verified the same binary and kit manifest SHA-256 checksums. The server binary printed its usage through the ROCm 10.1 container launcher on both hosts. |

The synthetic 200 MiB gather took about 87 ms with 4 MiB chunks and 89–90 ms as a single write. There were only two timed gathers per run, while model files were moving over the LAN. This is a correctness result and a timing sample, not an end-to-end performance comparison.

The collective harness used an isolated copy of the native kit with its required list narrowed to the staging and collection kernels. It loaded and checked the real GPU code objects. Full model serving, token comparisons against TCP, and cold prefill benchmarks are pending.

## Model staging and launch preparation

- On each host, `/opt/models/strixfold` points to `/home/sn/models/strixfold`. The completed EXL3 checkpoint is being copied from fuzzy to misty.
- Fuzzy’s original Xet downloads left partial Engram files with process-specific names. HTTP range probes matched the start, middle, and end of the two largest partial files. Misty reused the 25,081,309,114-byte prefix of shard 47 and verified the completed shard against its pinned SHA-256. Misty resumed shard 48 at byte 23,741,007,342 at 15:43 Berlin time.
- The Hugging Face HTTP download to misty has no bandwidth cap. Curl’s `--speed-limit 16384 --speed-time 60` is a stall cutoff that aborts below 16 KiB/s for 60 seconds so a retry can resume. The 64 MiB/s cap applies to copies from fuzzy to misty over the LAN.
- Because of fuzzy’s NVMe issue, the user chose to keep model data on the internal SSD with throttled writes. After misty verifies both snapshots, the queued return copy will reuse fuzzy’s prefixes and cap writes at 16 MiB/s. Rank 0’s cache and shared inputs will use the same cap.
- Misty has the pinned Python input dependencies and the rank-cache tools. The input job waits for the verified snapshots, then prepares both caches on misty and stages rank 0’s cache on fuzzy.
- The pinned AMD Triton build produced 277 variants with zero compilation failures. The assembled serving kit passed verification on both hosts and contains 302 kernel entries and 169 code objects. Manifest SHA-256 is `5eb6ec6fcdc8820534758b31915f8c8769924c663e45e20131485a0eb8bb0874`.
- Misty logged the final Hugging Face Engram shard as verified at 15:56 and the complete EXL3 and Engram snapshots as verified at 16:12:44. The throttled copy back to fuzzy was active at host time 16:30. These are host log timestamps. A later check found that misty was about one hour ahead of Berlin time.
- The return copy initially held the completed-download lock, which blocked cache preparation on misty. The preparation job was restarted against the verified manifest and began building the caches in parallel. The return script now releases that lock before copying.
- The first return transfer stopped advancing at 63,422,752,698 bytes of shard 47 around the time the host clocks moved backward. Kernel logs showed no new NVMe errors. The cause is unconfirmed. At the user's request, the weight copy resumed from that prefix at an 80 MiB/s cap. Fuzzy sustained 80.0 MiB/s over an eight-second sample. The separate rank-cache job is currently configured for 16 MiB/s.
- The return copy now relies on rsync to verify resumed transfers and promotes each shard after rsync succeeds. The separate per-shard and full-snapshot SHA-256 scans on fuzzy were removed from the copy job. Misty's source snapshots had already passed the pinned SHA-256 checks. The serving job still requires the repository's rank-input verification report.
- The return copy finished at 16:19 Berlin time, after the host clocks had moved back. Both finalized Engram shards are present on fuzzy at their expected sizes. By 16:24, misty had built both rank caches and compared 106,190,962,504 bytes per rank against the checkpoint with zero mismatches. The preparation process was still finalizing the bundle manifest before the rank 0 cache copy.
- Upstream commit `70cfb8c` was integrated as local commit `639686d` on branch `codex/strixfold-hip-verbs-drafts`. It makes `DRAFTS=1` the documented default and adds the context sweep tool. The deployed verbs configuration now has `DRAFTS=1`, `CONTEXT=65487`, and `SERVE_HOST=0.0.0.0` with `SERVE_PORT=8000` on both hosts. The 64K startup was stopped before serving because the previous launch attempt was still using the old configuration.
- The desktop and the UTC reference agreed at 14:32 UTC. Both servers were configured for Europe/Berlin and were about one hour ahead of the reference, with fuzzy about 21 seconds ahead of misty. Both ran chronyd with an unreachable sole source (`192.168.211.1`) and reported unsynchronized clocks.
- Both hosts now use `server 192.168.210.1 iburst` in `/etc/chrony.conf`. Chronyd restarted successfully on each host. Each sent eight NTP requests to UDP port 123 and received zero replies. The clocks were still unsynchronized at the verification check.
- Misty has a queued job to fully verify its rank inputs after preparation, then verify fuzzy's rank inputs after the throttled copy. If both pass, it starts the two ranks with the standard first-light settings (8K context, one stream, greedy, graphs on), binds HTTP to `127.0.0.1:18080` on fuzzy, and runs the known-answer Paris prompt. The launcher uses Ethernet for setup and Donnerkeule for runtime traffic. The initial probe leaves the serving pair running on success.

## Where to check

- November 7, 2026. Overlapping interrupted launch attempts left GPU lock holders on both hosts. Those identified processes were stopped. Run `20261107T082130Z` started successfully with `COLL_KIND=verbs`, `DRAFTS=1`, pool 65487, and HTTP bound to `0.0.0.0:8000`. The engine advertises 63470 usable context tokens. The Paris probe passed in 0.343 s. Both `/health` and `/v1/models` were reached from the Mac at `http://192.168.211.2:8000`. This deployment uses the existing `a4e4814` binaries. Upstream commit `70cfb8c`, integrated locally as `639686d`, changes documentation and defaults without changing Zig runtime sources. The service currently accepts requests without an API key, as configured.

- Misty’s model and cache logs are under `~/.local/state/strixfold-weights/`. See `transfer.log`, `engram-resume.log`, `copy-back.log`, `inputs-tools/prepare-rank-inputs.log`, and `inputs-tools/build-triton.log`.
- Misty’s queue logs to `~/.local/state/strixfold-weights/deploy-a4e4814/verify-and-smoke.log`. The first attempt exited after reading a previous preparation failure from the shared log. The corrected watcher tracks the active preparation process and is waiting for the completed bundle.
- Each host’s disposable collective-test logs are in its `/dev/shm/strixfold-coll.*` directory. These are lost on reboot. Fuzzy’s `~/.local/state/strixfold-weights/coll-labs` lists both locations.
- The queued smoke test required both verification reports, checked matching deployment identities, and acquired both GPU locks before launch. The TCP-versus-verbs comparison was subsequently completed on November 7 as recorded above.
