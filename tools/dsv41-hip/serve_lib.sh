# serve_lib.sh: sourced by serve-rank0.sh, serve-rank1.sh and first_light.sh. Resolves serve.conf (or $TF_SERVE_CONF)
# for one rank on this box into SERVE_CMD (the argv) and SERVE_ENV (the whole environment the rank starts with:
# nothing else leaks in), checks the inputs, and writes the resolved form with identities (resolved.json).
# shellcheck shell=bash

SERVE_TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SERVE_CONF=${TF_SERVE_CONF:-$SERVE_TOOLS/serve.conf}

die() { echo "$(basename "$0"): $*" >&2; exit 1; }

# serve_load RANK: the config for RANK on this box (paths expanded with this box's $HOME and RANK).
serve_load() {
  RANK=$1
  [ -f "$SERVE_CONF" ] || die "no config $SERVE_CONF"
  # shellcheck source=serve.conf
  source "$SERVE_CONF"
  DEPLOY_DIR_RAW=$DEPLOY_DIR  # unexpanded: another box's own $HOME (first_light.sh names rank 0's tools with it)
  local v
  # in this order: a path may name one expanded before it ($BUNDLE)
  for v in DEPLOY_DIR BUNDLE MODEL_DIR RANK_CACHE DS_KIT ENGRAM_DIR TOKEN_MAP RUN_ROOT; do eval "$v=\"${!v}\""; done
  BIN=$DEPLOY_DIR/native-hip/bin
  KIT=$DEPLOY_DIR/$KIT_SUBDIR
  [ "$PARALLEL" -ge 1 ] && [ "$PARALLEL" -le "$STREAMS" ] || die "PARALLEL $PARALLEL must be 1 .. STREAMS ($STREAMS)"
  local f size rows=
  for f in rope-plain-cos rope-plain-sin rope-compressed-cos rope-compressed-sin; do
    size=$(stat -L -c %s "$DS_KIT/$f.f32" 2>/dev/null) || die "no RoPE table $DS_KIT/$f.f32 (DS_KIT)"
    [ -z "$rows" ] || [ $((size / 128)) -lt "$rows" ] && rows=$((size / 128))
  done
  # the engine reads the pool's positions plus a round's 48 rows and one (model.zig rope_rows), 128 B a row
  [ "$CONTEXT" -le $((rows - 49)) ] || die "CONTEXT $CONTEXT is past the RoPE tables' $rows rows (at most $((rows - 49)))"
  case "$COLL_KIND" in
    host|proxy) ;;
    verbs)
      [ "$WORLD" = 2 ] || die "COLL_KIND=verbs requires WORLD=2"
      [ -z "$TP_STRIPE" ] && [ "${COLL_PROXY_STRIPE_KIB:-0}" = 0 ] || die "COLL_KIND=verbs uses one rail: clear TP_STRIPE and set COLL_PROXY_STRIPE_KIB=0"
      if [ "$RANK" = 0 ]; then
        COLL_VERBS_DEVICE=${COLL_VERBS_DEVICE_RANK0:-}
        COLL_VERBS_GID_INDEX=${COLL_VERBS_GID_INDEX_RANK0:-1}
      else
        COLL_VERBS_DEVICE=${COLL_VERBS_DEVICE_RANK1:-}
        COLL_VERBS_GID_INDEX=${COLL_VERBS_GID_INDEX_RANK1:-1}
      fi
      [ -n "$COLL_VERBS_DEVICE" ] || die "set COLL_VERBS_DEVICE_RANK$RANK"
      ;;
    *) die "COLL_KIND must be host, proxy or verbs" ;;
  esac
  [ "$GRAPHS" = 0 ] || [ "$COLL_KIND" != host ] || die "GRAPHS=1 requires COLL_KIND=proxy or verbs"
  [ "$GRAPHS" = 0 ] || [ $((ROUND_ROWS * 64640 * 4)) -le $((COLL_SLOT_KIB * 1024)) ] ||
    die "GRAPHS=1: the head gather at ROUND_ROWS=$ROUND_ROWS ($((ROUND_ROWS * 64640 * 4)) B) exceeds COLL_SLOT_KIB=$COLL_SLOT_KIB"
  # (a configuration written before these settings existed runs with them off)
  REPLAY_FLOOR=${REPLAY_FLOOR:-0} HC_SIDE=${HC_SIDE:-0} ENGRAM_AIO=${ENGRAM_AIO:-0} TOP_K_MAX=${TOP_K_MAX:-0} SEED_RANDOM=${SEED_RANDOM:-0}
  L2_PREFETCH=${L2_PREFETCH:-0}
  DRAFT_BYPASS_AT=${DRAFT_BYPASS_AT:-0}
  [[ $DRAFT_BYPASS_AT =~ ^(0|[2-8])$ ]] || die "DRAFT_BYPASS_AT must be 0 or 2..8"
  # the switch acts only on drafted TP2 shared rounds that can hold that many streams; elsewhere it is off
  if [ "$DRAFTS" != 1 ] || [ "$WORLD" != 2 ] || [ "$DRAFT_BYPASS_AT" -gt "$STREAMS" ]; then DRAFT_BYPASS_AT=0; fi
  for v in REPLAY_FLOOR HC_SIDE ENGRAM_AIO SEED_RANDOM L2_PREFETCH; do case ${!v} in 0|1) ;; *) die "$v must be 0 or 1, not ${!v}";; esac; done
  [[ $TOP_K_MAX =~ ^[0-9]+$ ]] || die "TOP_K_MAX must be a whole number (0: off), not $TOP_K_MAX"
}

# serve_check: every input this rank reads is here (paths only; verify_inputs.sh checks the bytes).
serve_check() {
  local f
  for f in "$MODEL_DIR/config.json" "$MODEL_DIR/tokenizer.json" "$MODEL_DIR/chat_template.jinja" "$RANK_CACHE" "$DS_KIT/engram.json" "$ENGRAM_DIR" "$TOKEN_MAP" "$KIT/manifest.json" \
           "$BIN/tensorfold-native" "$BIN/tf-dsv41-lanes" "$BUNDLE/inputs_manifest.json" "$DEPLOY_DIR/DEPLOY.json"; do
    [ -e "$f" ] || die "rank $RANK: missing $f"
  done
  local sum
  sum=$(sha256sum "$BUNDLE/inputs_manifest.json" | cut -d' ' -f1)
  [ "$sum" = "$BUNDLE_SHA256" ] || die "rank $RANK: inputs_manifest.json is $sum, the config pins $BUNDLE_SHA256"
  # U16: RANK_CACHE holds exactly the cache file the manifest names for this rank (the engine takes any
  # rank<R>of<W>-*.bin, so a stale or substituted cache under another key is refused here), and with VERIFY_REPORT=1
  # this box's verify_inputs.sh report (BUNDLE.verify-rank<R>-<host>.json) passed for this manifest
  python3 - "$BUNDLE/inputs_manifest.json" "$RANK_CACHE" "$RANK" "$WORLD" "${VERIFY_REPORT:-1}" \
    "$BUNDLE.verify-rank$RANK-$(hostname).json" "$BUNDLE_SHA256" <<'PY' || die "rank $RANK: the inputs are not the verified bundle's"
import json, os, sys
manifest, cache, rank, world, need_report, report, sha = sys.argv[1:8]
want = [c["build"]["report"]["file"] for c in json.load(open(manifest))["rank_caches"] if c["rank"] == int(rank)]
have = sorted(n for n in os.listdir(cache) if n.startswith(f"rank{rank}of{world}-") and n.endswith(".bin"))
if len(want) != 1 or have != want:
    sys.exit(f"rank cache {cache} holds {have or 'no cache'}, the manifest names {want}")
if need_report == "1":
    if not os.path.exists(report):
        sys.exit(f"no verify report {report} (tools/dsv41-hip/verify_inputs.sh)")
    r = json.load(open(report))
    if r.get("status") != "pass" or r.get("rank") != int(rank) or r.get("manifest_sha256") != sha:
        sys.exit(f"verify report {report}: status {r.get('status')}, rank {r.get('rank')}, manifest "
                 f"{r.get('manifest_sha256', 'not recorded')}; wanted pass for rank {rank} and manifest {sha}")
PY
}

# serve_idle (U25): this box is idle before a rank starts: no other engine process, and MemAvailable at least the
# rank's peak (PEAK_GIB) plus 8 GiB.
serve_idle() {
  local others avail
  others=$(pgrep -x -d, 'ds4|ds4-server|ds4-bench|tensorfold-nati|tf-dsv41-lanes|tf-dsv41-layer' || true)
  [ -z "$others" ] || die "rank $RANK: $(hostname) is not idle: engine process(es) $others"
  avail=$(awk '/^MemAvailable:/ { print int($2 / 1048576) }' /proc/meminfo)
  [ "$avail" -ge $((PEAK_GIB + 8)) ] || die "rank $RANK: MemAvailable ${avail} GiB on $(hostname), a rank needs ${PEAK_GIB} + 8 GiB"
}

# serve_env: SERVE_ENV, the rank's whole environment (env -i starts it with exactly this).
serve_env() {
  SERVE_ENV=(HOME="$HOME" USER="${USER:-}" PATH=/usr/local/bin:/usr/bin:/bin LANG=C.UTF-8
    TF_GPU_STRICT="$GPU_STRICT" TF_HIP_DEVICE="$HIP_DEVICE" TENSORFOLD_HIP_MEMORY_LIMIT_GB="$MEMORY_LIMIT_GB"
    TF_DS_MODALITIES="$MODALITIES" TF_DS_GRAPHS="$GRAPHS" TF_DS_ARENA_GIB="$ARENA_GIB"
    TF_COLL_STRIPE_MIB="$COLL_STRIPE_MIB" TF_COLL_RTO_MIN_US="$COLL_RTO_MIN_US" TF_COLL_PROGRESS_MS="$COLL_PROGRESS_MS"
    TF_COLL_STAGING="$COLL_STAGING" TF_COLL_CHUNK_MIB="${COLL_CHUNK_MIB:-4}" TF_COLL_KIND="$COLL_KIND"
    TF_DS_AUX="$AUX" TF_DS_REPLAY_FLOOR="$REPLAY_FLOOR")
  [ "$COLL_KIND" != host ] && SERVE_ENV+=(TF_COLL_PROXY_CPU="$COLL_PROXY_CPU" TF_COLL_SLOT_KIB="$COLL_SLOT_KIB"
    TF_COLL_PROXY_STRIPE_KIB="${COLL_PROXY_STRIPE_KIB:-0}" TF_COLL_PROXY_CPU2="${COLL_PROXY_CPU2:-7}")
  [ "$COLL_KIND" = verbs ] && SERVE_ENV+=(TF_COLL_VERBS_DEVICE="$COLL_VERBS_DEVICE" TF_COLL_VERBS_GID_INDEX="$COLL_VERBS_GID_INDEX")
  [ -n "${COLL_GPU_WAIT_S:-}" ] && SERVE_ENV+=(TF_COLL_GPU_WAIT_S="$COLL_GPU_WAIT_S")  # (a conf without them: the engine's)
  [ -n "${COLL_GPU_WARN_S:-}" ] && SERVE_ENV+=(TF_COLL_GPU_WARN_S="$COLL_GPU_WARN_S")
  [ -n "$TP_DATA_PORT" ] && SERVE_ENV+=(TF_TP_DATA_PORT="$TP_DATA_PORT")
  [ -n "$TP_STRIPE" ] && SERVE_ENV+=(TF_TP_STRIPE="$TP_STRIPE")
  [ -n "$ROUND_MS" ] && SERVE_ENV+=(TF_DS_ROUND_MS="$ROUND_MS")
  [ -n "$DRAFT_MS" ] && SERVE_ENV+=(TF_DS_DRAFT_MS="$DRAFT_MS")
  [ -n "${FIX_K:-}" ] && [ "$RANK" = 0 ] && SERVE_ENV+=(TF_DS_FIX_K="$FIX_K")
  [ -n "${FROZEN_TIMING:-}" ] && [ "$RANK" = 0 ] && SERVE_ENV+=(TF_LANES_FROZEN_TIMING="$FROZEN_TIMING")
  [ "$RANK" = 0 ] && SERVE_ENV+=(TF_DS_DRAFT_BYPASS_AT="$DRAFT_BYPASS_AT")
  [ -n "${TRACE_ROUNDS:-}" ] && [ "$RANK" = 0 ] && SERVE_ENV+=(TF_DS_TRACE_ROUNDS="$TRACE_ROUNDS")
  [ -n "${GRAPHS_EAGER:-}" ] && SERVE_ENV+=(TF_DS_GRAPHS_EAGER="$GRAPHS_EAGER")
  # the prefetch's forked stream overlaps the graph's main branch only with HIP's segment scheduling (a debug variable)
  [ "$L2_PREFETCH" = 1 ] && SERVE_ENV+=(DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=1)
  # the split doubles Chunk's cap-independent buffers (+474 MB at 16K, +0.62-0.68 GiB per rank at 63,488): serve.conf keeps it on (PREFILL_SPLIT_LARGE=1); LARGE=0 clears it above 16384
  if [ -n "${PREFILL_SPLIT:-}" ] && [ "$PREFILL_SPLIT" != 0 ] && [ "$CONTEXT" -gt 16384 ] && [ "${PREFILL_SPLIT_LARGE:-0}" != 1 ]; then
    echo "serve_lib: PREFILL_SPLIT off at CONTEXT $CONTEXT (> 16384, arena cost unmeasured; PREFILL_SPLIT_LARGE=1 keeps it)" >&2
    PREFILL_SPLIT=
  fi
  [ -n "${PREFILL_SPLIT:-}" ] && SERVE_ENV+=(TF_DS_PREFILL_SPLIT="$PREFILL_SPLIT")
  [ -n "${PREFILL_SPLIT_ZIP:-}" ] && SERVE_ENV+=(TF_DS_PREFILL_SPLIT_ZIP="$PREFILL_SPLIT_ZIP")
  [ -n "${LOGITS_SHA:-}" ] && SERVE_ENV+=(TF_DS_LOGITS_SHA="$LOGITS_SHA")
  [ -n "${WINDOW_LOGITS:-}" ] && SERVE_ENV+=(TF_DS_WINDOW_LOGITS="$WINDOW_LOGITS")
  # the launch's identity in the peers' handshake (U46): the deployed commit, the bundle's and serve.conf's hashes
  local commit
  commit=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["commit"])' "$DEPLOY_DIR/DEPLOY.json")
  SERVE_ENV+=(TF_DS_DEPLOY="$commit" TF_DS_BUNDLE="$BUNDLE_SHA256" TF_DS_CONF="$(sha256sum "$SERVE_CONF" | cut -d' ' -f1)")
  if [ "$RANK" = 0 ]; then
    SERVE_ENV+=(TF_TP_WORLD="$WORLD" TF_TP_MASTER="$MASTER_IP" TF_TP_PORT="$TP_PORT" TF_DS_RANK_CACHE="$RANK_CACHE"
      TF_DS_KIT="$DS_KIT" TF_DS_ENGRAM="$ENGRAM_DIR" TF_DS_TOKEN_MAP="$TOKEN_MAP" TF_DS_STREAMS="$STREAMS"
      TF_DS_ROUND_ROWS="$ROUND_ROWS" TENSORFOLD_HIP_KERNELS="$KIT" TF_DS_HC_SIDE="$HC_SIDE" TF_DS_ENGRAM_AIO="$ENGRAM_AIO"
      TF_TOP_K_MAX="$TOP_K_MAX" TF_SEED_RANDOM="$SEED_RANDOM" TF_DS_L2_PREFETCH="$L2_PREFETCH")
  fi
}

# serve_lanes_cmd: SERVE_CMD, tf-dsv41-lanes for this rank with the configuration's engine settings (rank 1's serving
# command; with rank 0 and --requests, the lane gate of the same configuration: qual/lane_gate.sh).
serve_lanes_cmd() {
  SERVE_CMD=("$BIN/tf-dsv41-lanes" "$MODEL_DIR" "$RANK_CACHE" "$RANK" "$WORLD" "$MASTER_IP" "$TP_PORT" "$DS_KIT"
    --pool "$CONTEXT" --streams "$STREAMS" --round-rows "$ROUND_ROWS" --drafts "$DRAFTS"
    --engram "$ENGRAM_DIR" --token-map "$TOKEN_MAP" --modalities "$MODALITIES" --hip-kit "$KIT"
    --arena-gib "$ARENA_GIB" --graphs "$GRAPHS" --side "$HC_SIDE" --aio "$ENGRAM_AIO" --prefetch "$L2_PREFETCH")
  [ -n "$ROUND_MS" ] && SERVE_CMD+=(--round-ms "$ROUND_MS")
  [ -n "$DRAFT_MS" ] && SERVE_CMD+=(--draft-ms "$DRAFT_MS")
  return 0
}

# serve_cmd: SERVE_CMD, the rank's argv. Rank 1's --pool, --streams, --round-rows and --graphs are rank 0's --context,
# TF_DS_STREAMS, TF_DS_ROUND_ROWS and TF_DS_GRAPHS (upstream #25), from the same variables (tf-dsv41-lanes reads
# --graphs, --side, --aio and --prefetch, not TF_DS_GRAPHS, TF_DS_HC_SIDE, TF_DS_ENGRAM_AIO and TF_DS_L2_PREFETCH).
serve_cmd() {
  if [ "$RANK" = 0 ]; then
    SERVE_CMD=("$BIN/tensorfold-native" serve "$MODEL_DIR" --host "$SERVE_HOST" --port "$SERVE_PORT" --name "$SERVE_NAME"
      --context "$CONTEXT" --parallel "$PARALLEL" --max-tokens "$MAX_TOKENS" --temperature "$TEMPERATURE"
      --prompt-cache-gib "$PROMPT_CACHE_GIB" --no-update-check)
    [ "$THINKING" = 1 ] && SERVE_CMD+=(--thinking) || SERVE_CMD+=(--no-thinking)
    [ "$DRAFTS" = 1 ] || SERVE_CMD+=(--no-drafts)
  else
    serve_lanes_cmd
    [ -n "${LOGITS_SHA:-}" ] && SERVE_CMD+=(--logits-sha "$LOGITS_SHA")
  fi
  return 0
}

# serve_resolved FILE: the resolved configuration and this box's identities (what ran, from which bytes).
serve_resolved() {
  python3 - "$1" "$RANK" "$(hostname)" "$SERVE_CONF" "$DEPLOY_DIR" "$KIT" "$RANK_CACHE" "${SERVE_CMD[@]}" -- "${SERVE_ENV[@]}" <<'PY'
import hashlib, json, os, sys, time
out, rank, host, conf, deploy, kit, cache = sys.argv[1:8]
rest = sys.argv[8:]
cut = rest.index("--")
cmd, env = rest[:cut], rest[cut + 1:]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
dep = json.load(open(os.path.join(deploy, "DEPLOY.json")))
caches = sorted(n for n in os.listdir(cache) if n.startswith(f"rank{rank}of"))
json.dump({
    "schema": "tf-port-serve/1", "rank": int(rank), "host": host, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "config": conf, "config_sha256": sha(conf), "deploy": {"dir": os.path.realpath(deploy), "commit": dep.get("commit"), "tree": dep.get("tree")},
    "binary_sha256": sha(cmd[0]), "kit_manifest_sha256": sha(os.path.join(kit, "manifest.json")), "rank_cache_files": caches,
    "argv": cmd, "env": dict(e.split("=", 1) for e in env),
}, open(out, "w"), indent=1)
PY
}

# serve_spawn PIDFILE LOG LOCKED: start SERVE_CMD with exactly SERVE_ENV in the background (on ENGINE_CPUS when set),
# under this box's GPU lock unless LOCKED is 1 (the caller holds it); PIDFILE gets the rank's own pid (not the lock
# wrapper's: a signal to it reaches the rank), written once the lock is held and the rank starts.
serve_spawn() {
  local pidfile=$1 log=$2 locked=$3 lock=() pin=() wrap=() wv=WRAP$RANK
  [ "$locked" = 1 ] || lock=(flock -w "$LOCK_WAIT_S" /var/tmp/halo-gpu.lock)
  [ -n "${ENGINE_CPUS:-}" ] && pin=(taskset -c "$ENGINE_CPUS")
  [ -n "${!wv:-}" ] && read -ra wrap <<< "${!wv}"
  rm -f "$pidfile"
  # in the run's directory (the log's): the deployment is read-only, and a wrapper (rocprofv3) writes beside it
  (cd "$(dirname "$log")" && exec nohup "${lock[@]}" sh -c 'echo $$ > "$0"; exec "$@"' "$pidfile" "${pin[@]}" env -i "${SERVE_ENV[@]}" "${wrap[@]}" "${SERVE_CMD[@]}") </dev/null >"$log" 2>&1 &
  local wrapper=$! i
  for i in $(seq 1 "$LOCK_WAIT_S"); do
    [ -s "$pidfile" ] && return 0
    kill -0 "$wrapper" 2>/dev/null || die "the rank did not start (lock wait or exec failed; $log)"
    sleep 1
  done
  die "the GPU lock on $(hostname) was not free within ${LOCK_WAIT_S}s"
}

# serve_latest: the latest run's id on this box (RUN_ROOT/latest points at it), or nothing.
serve_latest() { basename "$(readlink "$RUN_ROOT/latest" 2>/dev/null)" 2>/dev/null || true; }

# serve_pid FILE: the live pid in FILE, or nothing.
serve_pid() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null && cat "$1"; return 0; }

# serve_stop PIDFILE NAME: SIGTERM, then SIGKILL after STOP_TIMEOUT_S.
serve_stop() {
  local pid; pid=$(serve_pid "$1")
  [ -n "$pid" ] || { echo "$2: not running"; return 0; }
  kill -TERM "$pid"
  local i
  for i in $(seq 1 "$STOP_TIMEOUT_S"); do kill -0 "$pid" 2>/dev/null || { echo "$2: stopped ($pid)"; return 0; }; sleep 1; done
  kill -KILL "$pid" 2>/dev/null || true
  echo "$2: killed after ${STOP_TIMEOUT_S}s ($pid)"
}

# serve_lock_hold DIR [WAIT_S]: hold this box's GPU lock for the run (a holder process; its pid in DIR/lock.pid once
# held), waiting at most WAIT_S (default LOCK_WAIT_S); a holder that did not get it in time is taken down, never left.
serve_lock_hold() {
  local dir=$1 wait=${2:-$LOCK_WAIT_S}
  rm -f "$dir/lock.pid"
  nohup flock -w "$wait" /var/tmp/halo-gpu.lock sh -c "echo \$\$ > '$dir/lock.pid'; exec sleep 86400" </dev/null >/dev/null 2>&1 &
  local holder=$! i
  for i in $(seq 1 "$wait"); do
    [ -s "$dir/lock.pid" ] && return 0
    kill -0 "$holder" 2>/dev/null || break
    sleep 1
  done
  [ -s "$dir/lock.pid" ] && return 0
  # flock gave up, or got the lock just now: its child (the sleeping holder) and flock itself go
  pkill -P "$holder" 2>/dev/null; kill "$holder" 2>/dev/null; rm -f "$dir/lock.pid"
  die "the GPU lock on $(hostname) was not free within ${wait}s"
}

serve_lock_release() { local pid; pid=$(serve_pid "$1/lock.pid"); [ -n "$pid" ] && kill "$pid"; rm -f "$1/lock.pid"; return 0; }
