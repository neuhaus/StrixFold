import hashlib
import importlib.util
import json
import pathlib
import sys
import time
import urllib.request

tools, output, transport = sys.argv[1:]
spec = importlib.util.spec_from_file_location("bench_stream", pathlib.Path(tools) / "bench_stream.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
url = "http://127.0.0.1:18081"
def count(prompt):
    body = json.dumps({"model": "dsv41", "messages": [{"role": "user", "content": prompt}]}).encode()
    request = urllib.request.Request(url + "/v1/tokenize", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)["count"]


def prompt_at(target):
    suffix = "\nWrite a Python class implementing an LRU cache with get/put in O(1), with type hints and a short usage example."
    prompt = "Reference material follows.\n" + "The quick brown fox jumps over the lazy dog. " * ((target - 100) // 10)
    have = count(prompt + suffix)
    padding = max(0, target - have)
    for _ in range(3):
        actual = count(prompt + " x" * padding + suffix)
        if actual == target:
            break
        padding += target - actual
    prompt += " x" * padding
    if actual != target:
        raise ValueError(f"wanted {target} tokens, got {actual}")
    return prompt + suffix


cases = [(f"prompt_{target}", prompt_at(target), 256) for target in (2000, 63000)]
results = []
bench.run(url, "dsv41", bench.PROMPTS["code"], 128, 900)
for rep in range(3):
    for name, prompt, limit in cases:
        print(json.dumps({"starting": name, "rep": rep, "transport": transport}), flush=True)
        rec = bench.run(url, "dsv41", prompt, limit, 1800)
        rec.update(transport=transport, case=name, rep=rep,
                   reply_sha256=hashlib.sha256(rec["reply"].encode()).hexdigest(),
                   prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
        results.append(rec)
        pathlib.Path(output).write_text(json.dumps(results, indent=2))
        print(json.dumps({k: v for k, v in rec.items() if k not in ("reply", "usage")}), flush=True)
print(json.dumps({"complete": transport, "requests": len(results)}), flush=True)
