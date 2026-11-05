# Build and run: DeepSeek-V4.1-Flash-1017 (EXL3 2.9 bpw) on two Strix Halo boxes

These are the commands this port runs, in order: build, prepare the inputs, deploy, serve, check and benchmark. The two
boxes are called `node0` (rank 0, the server) and `node1` (rank 1, also the build box); give them those ssh names, or
change `RANK0_HOST` / `RANK1_HOST` in `serve.conf`. Everything lives under `~/tfp` on both boxes (sources, deployments,
runs) and `~/models/tf-port` (checkpoints and inputs); the scripts and `serve.conf` use those paths, so keep them or
edit `serve.conf` in one place. The addresses below are the two boxes' USB4 links: copy the commands, not the
addresses.

## 1. What you need

| Item | The boxes these numbers come from |
|---|---|
| Hosts | 2 x Ryzen AI Max+ 395, Radeon 8060S **gfx1151**, 128 GB unified memory each (HIP reports 124.00 GiB, 121.66 free) |
| GPU stack | ROCm 10.0.0-4 (HIP 7.15, clang 23) under `/opt/rocm/core-10.0`; Linux kernel 7.2.7; GPU stack, kernel and firmware held at those versions |
| Link | two USB4 cables (thunderbolt_net): `thunderbolt0` 10.40.0.1/2 carries the engine's control and data links, `thunderbolt1` 10.40.1.1/2 the stripe and bulk copies; MTU **1500** on both. Setup and tuning: [docs/TUNING.md](../../docs/TUNING.md) |
| Toolchain | Zig 0.17.0 (`~/opt/zig/zig`); Python 3.13; a venv for the input tools (`~/venvs/tfp-inputs`) pinned by `requirements-inputs.lock.txt` (numpy 2.3.5, tokenizers 0.22.1, torch 2.9.0+cpu); for the Triton kit, the pinned AMD Triton build in [tri/README.md](tri/README.md) |
| Memory | per rank: 98.1 GiB of weights (GTT 98.8 GiB measured) plus the KV arena (2.5-2.8 GiB at 8K context and one stream); keep at least 8 GiB MemAvailable while serving |
| Disk, each box | the checkpoint `~/models/tf-port/libertai-2.9bpw` (LibertAI EXL3 2.9 bpw @ c178a012, 197 GB), the Engram shards 47/48 of DeepSeek-V4.1-Flash-1017 @ 1bae0e10 in `~/models/tf-port/ds1017-1bae0e10` (203 GB), and the input bundle `~/models/tf-port/inputs-tp2-c178a012` (that rank's 106 GB rank cache plus 0.1 GB of tables) |

No NVIDIA driver, CUDA, NCCL or NVIDIA container is used or needed anywhere below. Image input is not supported (text
only).

## 2. Build (node1, from a clean committed tree)

The AMD Triton kit (the auxiliary kernels compiled ahead of time) is built once per Triton pin, into
`~/tfp/tri/kit/gfx1151`: steps 1-3 of [tri/README.md](tri/README.md). Then:

```
cd ~/tfp/src/TensorFold            # your clone of this repository
tools/dsv41-hip/zig_build.sh install                                   # CPU tools: tf-dsv41-rank-cache, tf-dsv41-check (no CUDA)
tools/dsv41-hip/zig_build.sh test test-dsv41-hip-host native-hip dsv41-hip-gpu-test -Dhip-triton-kit=$HOME/tfp/tri/kit/gfx1151
tools/dsv41-hip/run_gpu_tests.sh                                       # GPU smoke / ABI / glue tests, under the GPU lock
```
`zig_build.sh` runs `zig build` confined to cores 8-15/24-31 (800% CPU, 16 GiB) at low priority. The native build
installs `zig-out/native-hip/{bin/tensorfold-native, bin/tf-dsv41-lanes, bin/tf-dsv41-layer, bin/tf-hip-kit,
share/tensorfold/hip/gfx1151/dsv41}` (the AMD kit: HIP code objects plus the AOT Triton kit, every object by SHA-256
in its `manifest.json`). A green build ends with `Build Summary: N/N steps succeeded`. GPU work on either box takes the
lock file `/var/tmp/halo-gpu.lock` (`flock`), so two jobs never share a GPU.

## 3. Inputs (CPU only, once per checkpoint revision)

```
tools/dsv41-hip/prepare_inputs.sh ~/models/tf-port/libertai-2.9bpw ~/models/tf-port/ds1017-1bae0e10 ~/models/tf-port/inputs-tp2-c178a012
tools/dsv41-hip/verify_inputs.sh  ~/models/tf-port/inputs-tp2-c178a012 1 ~/models/tf-port/libertai-2.9bpw ~/models/tf-port/ds1017-1bae0e10 \
    243b984d1069b190e762b3f9eaf12c90b2a23dd1e8f33880019f60d61264e8e9            # rank 1 on node1; rank 0 on node0 from its deployment
```
`prepare_inputs.sh` authenticates every checkpoint and Engram file against the pinned Hugging Face metadata, writes the
token map, the Engram constants (`kit/engram.json`), the RoPE tables (`kit/rope-*.f32`, 65,536 rows by default: they
set the longest context the engine can serve), both rank caches and the chat template (c282975e), byte-checks each
cache in full (`tf-dsv41-check --bytes 1`) and hashes everything into `inputs_manifest.json` (SHA-256 243b984d... for
this revision; `serve.conf` pins it). About 1 TB is read and 212 GB written. Copy the bundle to node0 (rank 0 needs
its own rank cache and the tables). `verify_inputs.sh` re-checks a box's copy before that rank loads it (about 15
minutes).

## 4. Deploy (both boxes run one commit's bytes)

```
C=$(git rev-parse HEAD)
tools/dsv41-hip/deploy_rank0.sh --native zig-out/native-hip --bin-commit $C \
    --ds-kit ~/models/tf-port/inputs-tp2-c178a012/kit                                   # node0:~/tfp/deploy/<commit12>/
tools/dsv41-hip/deploy_rank0.sh --host local --native zig-out/native-hip --bin-commit $C \
    --ds-kit ~/models/tf-port/inputs-tp2-c178a012/kit                                   # node1:~/tfp/deploy/<commit12>/
```
Each deployment holds `tools/dsv41-hip` from `git archive`, the native tree, the RoPE tables and Engram constants
(`ds-kit/`), `DEPLOY.json` (commit, tree, the SHA-256 of every file) and `SHA256SUMS`, checked on arrival and made
read-only; `~/tfp/deploy/current` points at it. The script refuses a tree with uncommitted changes, and only a commit on
`main` (`--integ`) becomes `current`.

## 5. Serve

One configuration file drives both ranks (`tools/dsv41-hip/serve.conf`, commented line by line): the deployment, the
input bundle and its pinned hash, context, streams, verify rows, drafts, graphs, transport settings, timeouts and the
probe. Put your copy at the same path under each box's `$HOME`, then from the deployment's tools on node1:

```
cd ~/tfp/deploy/current/tools/dsv41-hip
TF_SERVE_CONF='$HOME/tfp/run/conf/my.conf' ./first_light.sh up       # locks (node0, then node1), rank 1, rank 0, /health, probe
TF_SERVE_CONF='$HOME/tfp/run/conf/my.conf' ./first_light.sh status
TF_SERVE_CONF='$HOME/tfp/run/conf/my.conf' ./first_light.sh down     # rank 0 SIGTERM (rank 1 leaves with it), locks released
TF_SERVE_CONF='$HOME/tfp/run/conf/my.conf' ./first_light.sh restart
```
`up` refuses to start unless both ranks resolve to the same deploy commit, kit manifest and configuration, and the
bundle's manifest hash equals `BUNDLE_SHA256`; it is ready only when the probe's reply contains `PROBE_EXPECT`. The
ranks can also be driven one by one: `serve-rank1.sh start|wait-linked|stop|status` on node1, `serve-rank0.sh
start|ready|chat|stop|status` on node0 (same `RUN_ID` and `TF_SERVE_CONF` on both).

The server listens on node0 at `SERVE_HOST:SERVE_PORT` (127.0.0.1:18080 in `serve.conf`; OpenAI-compatible):
```
ssh node0 'curl -s http://127.0.0.1:18080/v1/chat/completions -H "Content-Type: application/json" -d "{
  \"model\": \"dsv41\", \"messages\": [{\"role\": \"user\", \"content\": \"What is the capital of France?\"}],
  \"max_tokens\": 64, \"temperature\": 0, \"stream\": false}"'
```
`/health`, `/v1/models`, `/v1/tokenize` and `/metrics` answer as well. Logs and identities of a run are in
`RUN_ROOT/RUN_ID/` on each box (`rank0.log`, `rank1.log`, `resolved-rank*.json` with the argv, the whole environment
and every hash, `probe.json`).

### Settings that matter

`serve.conf` ships the conservative first-run settings: one stream (`STREAMS=1`, `PARALLEL=1`), an 8,192-token context,
drafts off (`DRAFTS=0`), HIP graphs on (`GRAPHS=1`, over `COLL_KIND=proxy`).

- **Several chats at once:** `STREAMS` and `PARALLEL` up to 8 (the measured 1-8 chat results in
  [docs/RESULTS.md](../../docs/RESULTS.md)). Every stream shares the pool's `CONTEXT` positions.
- **Longer context:** `CONTEXT` up to the RoPE tables' rows minus 49. With the default 65,536 rows that is 63,488
  tokens; longer windows need inputs prepared with more rows (the 1M-token runs in RESULTS did that).
- **Drafts:** `DRAFTS=1` turns on DSpark drafting; it gives exactly the same text as drafts off.

### Opt-in USB4 verbs transport (experimental)

For two Linux HIP ranks with a working Donnerkeule provider, copy `serve.conf` and set:

```sh
MASTER_IP=192.0.2.10              # replace with rank 0's private Ethernet address
TP_STRIPE=
COLL_PROXY_STRIPE_KIB=0
COLL_KIND=verbs
COLL_VERBS_DEVICE_RANK0=usb4_rdma0
COLL_VERBS_DEVICE_RANK1=usb4_rdma0
COLL_VERBS_GID_INDEX_RANK0=1
COLL_VERBS_GID_INDEX_RANK1=1
```

Select the local HCA and GID for the same physical rail at each end. TCP exchanges the startup settings and queue-pair
descriptors, then stays open to detect a peer exit while the follower is idle. A dedicated RC SEND queue pair carries
prompt IDs and lane commands. Gather payloads use registered host buffers on separate queue pairs for graph-sized slots
and larger staged transfers. The latter still uses the existing GPU copy pipeline; a single work request on
that path is at most 4 MiB of payload plus its header by default. For larger writes, set `COLL_CHUNK_MIB` on both
ranks; the payload plus its header must fit the limit reported by both ports, and each rank registers two windows
of that size. Benchmark different chunk sizes against the default to check how much GPU copy overlap they retain.
No kernel module changes or DMA-BUF support are required at the default size.
The two peers must select the same collective kind, and a verbs failure poisons the session. Restart both ranks with
their previous TCP settings to switch back.

Before serving, use the installed `tf-dsv41-coll-check` on both ranks with `TF_COLL_KIND=verbs`,
`TF_COLL_VERBS_DEVICE`, `TF_COLL_VERBS_GID_INDEX`, and `--kit DIR --graph 5 --prefill-chunks 1`.
Check that both ranks report zero bad words, including the 40 MiB and 200 MiB gathers and graph replays. Test peer
loss and restart, then use the same checkpoint, precision, request settings and prompt IDs for TCP/verbs output
comparisons and timings. Match cache settings and run cold prefill with fresh state. The wire test alone is not a
served model or performance qualification.

## 6. Check and measure

```
cd tools/dsv41-hip
qual/qual_serve.sh   '$HOME/tfp/run/conf/my.conf' RUN_ID [--long 1]  # served functional suite, no-NVIDIA check, memory ledger, restart
qual/served_faults.sh '$HOME/tfp/run/conf/my.conf' BASE_ID           # rank kill/stop, peer mismatches, stale inputs
qual/qual_bench.sh bench '$HOME/tfp/run/conf/my.conf' RUN_ID --reps 5 --prefill 2000,8000,16000
SOAK_S=900 qual/qual_bench.sh soak '$HOME/tfp/run/conf/my.conf' RUN_ID --long-every 5
python3 qual/sampling_check.py OUT_DIR                               # the sampler against its pinned contract (CPU)
```
The served suite (`qual/serve_qual.py`) exercises the endpoint with streaming, system and multi-turn messages,
`chat_template_kwargs: {"enable_thinking": true|false}`, `stop`, `max_tokens`, capacity limits, malformed and image
input, tools (`tool_choice` "auto"), seeded sampling and `return_token_ids`. Each two-box run takes node0's GPU lock,
then node1's (or neither), holds them for that run only and releases them on every exit path. Results land in
`~/tfp/qual/out/`.

Timed runs can open quiet windows on both boxes: set `QUIET_BIN` to a program of yours that takes `on TAG` / `off TAG`
and holds other work on that box meanwhile (the default `~/bin/halo-quiet` is skipped where it isn't installed). The
published numbers were measured inside such windows.

`qualify.sh` and `gates.json` are this port's own qualification record: every gate with the evidence it was closed on.
That evidence lives on the boxes it was measured on, so on yours the suites above are the ones to run.

## 7. Debugging

| Symptom | Where to look / what it means |
|---|---|
| `collectives handshake refused: <field>=a vs b` | the two ranks run different settings, kits or plans: one configuration file, one deployment |
| `no Triton variant of _X in the kit for this launch` (HTTP 500 `MissingTritonVariant`) | the kit lacks that specialization; `GPU_STRICT=0` only helps where a less specialized variant exists |
| a request ends without a finish reason, rank logs show `PeerLost` / `Timeout` | the other rank died, stalled or lost its link (2 s progress deadline, `COLL_PROGRESS_MS`); restart both ranks |
| `inputs_manifest.json is X, the config pins Y` | the bundle is not the one the configuration names |
| slow decode | other CPU load on either box (the TP service thread and USB4 interrupts share cores 0-7/16-23); measure inside quiet windows |

## 8. Known issues

- A request whose prompt plus `max_tokens` exactly fills the context is refused with a 500; one token over is a 400,
  as it should be.
- After the other rank fails, rank 0 stays up answering 500 while `/health` still says "ok": restart both ranks.
- Images are refused: the text capability never opens the vision tower.
