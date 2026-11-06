"""ctx_sweep.py URL NAME SRC TARGETS REPS OUT [LOG]: writing speed against context length (docs/RESULTS.md, Speed at
long context).

Each request is one streamed greedy chat: about TARGET tokens of real code as context (the .zig files under SRC, such as
this repository's zig/src; a different slice for every request, so the prompt cache misses), then the same coding task,
400 tokens written. Every target runs once with drafts on and once with "draft": false, REPS times. TARGETS is a comma
list of token counts (0: the task alone). Per request: prompt and completion tokens (the server's usage), TTFT, decode
tok/s = (completion tokens - 1) / (last content event - first), prefill tok/s = prompt tokens / TTFT; given the server's
log LOG (rank0.log), also its rounds, accepted drafts, tokens per round and ms per round. OUT gets every request; stdout
one JSON line per request."""
import json
import os
import re
import sys
import time
import urllib.request
import uuid

TASK = ("Ignore the code above for this task. Write a Python class implementing an LRU cache with get/put in O(1), "
        "with type hints and a short usage example.")


def corpus(src):
    parts = []
    for root, _, files in sorted(os.walk(src)):
        for f in sorted(files):
            if f.endswith(".zig"):
                parts.append(open(os.path.join(root, f), encoding="utf-8", errors="replace").read())
    return "\n".join(parts)


def run(url, name, prompt, draft):
    body = {"model": name, "messages": [{"role": "user", "content": prompt}], "max_tokens": 400, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}}
    if not draft:
        body["draft"] = False
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.monotonic()
    first = last = None
    rid, usage = None, None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            obj = json.loads(line[5:])
            rid = rid or obj.get("id")
            usage = obj.get("usage") or usage
            for ch in obj.get("choices") or []:
                if (ch.get("delta") or {}).get("content"):
                    now = time.monotonic() - t0
                    first = now if first is None else first
                    last = now
    n = usage["completion_tokens"]
    span = last - first
    return {"id": rid, "prompt_tokens": usage["prompt_tokens"], "completion_tokens": n, "ttft_s": round(first, 3),
            "decode_s": round(span, 3), "decode_tok_per_s": round((n - 1) / span, 2),
            "prefill_tok_per_s": round(usage["prompt_tokens"] / first, 1)}


def log_stats(log, rid):
    """The server's `done <id> ... rounds=R accepted=A/B` line for this request, waiting up to 10 s for it."""
    for _ in range(20):
        for line in open(log, errors="replace").read().splitlines()[::-1]:
            if f"done {rid} " in line:
                m = dict(re.findall(r"(\w+)=(\S+)", line))
                return {"rounds": int(m["rounds"]), "accepted": m.get("accepted")}
        time.sleep(0.5)
    return {}


def main():
    url, name, src, targets, reps, out = sys.argv[1:7]
    log = sys.argv[7] if len(sys.argv) > 7 else None
    text = corpus(src)
    recs, k = [], 0
    for rep in range(int(reps)):
        for tgt in [int(t) for t in targets.split(",")]:
            for draft in (True, False):
                k += 1
                chars = int(tgt * 3.1)  # about 3.1 characters a token for Zig source
                start = (k * 7919) % max(1, len(text) - chars)
                head = f"Session {uuid.uuid4().hex[:12]}.\n\n"
                ctx = f"Here is part of a codebase for context:\n\n```zig\n{text[start:start + chars]}\n```\n\n" if tgt else ""
                try:
                    rec = run(url, name, head + ctx + TASK, draft)
                    if log:
                        rec.update(log_stats(log, rec["id"]))
                    if rec.get("rounds"):
                        rec["tok_per_round"] = round(rec["completion_tokens"] / rec["rounds"], 2)
                        rec["ms_per_round"] = round(1000 * rec["decode_s"] / max(1, rec["rounds"] - 1), 1)
                except Exception as e:  # recorded; the next request still runs
                    rec = {"error": f"{type(e).__name__}: {e}"}
                rec.update({"target": tgt, "draft": draft, "rep": rep})
                recs.append(rec)
                print(json.dumps(rec), flush=True)
                json.dump(recs, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
