import importlib.util
import json
import pathlib
import sys
import types

tools, output, transport = sys.argv[1:]
spec = importlib.util.spec_from_file_location("benchmark", pathlib.Path(tools) / "qual/benchmark.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
bench.A = types.SimpleNamespace(url="http://127.0.0.1:18081", model="dsv41")
base = "Reference material follows.\n" + "The quick brown fox jumps over the lazy dog. " * 190
suffix = "\nWrite a Python class implementing an LRU cache with get/put in O(1), with type hints and a short usage example."
padding = 2000 - len(bench.prompt_ids(base + suffix))
for _ in range(3):
    prompt = base + " x" * padding + suffix
    delta = 2000 - len(bench.prompt_ids(prompt))
    if delta == 0:
        break
    padding += delta
assert len(bench.prompt_ids(prompt)) == 2000
records = []
for rep in range(3):
    for draft in (True, False):
        record = bench.one(prompt, 256, draft)
        if record["status"] != 200 or record["error"]:
            raise RuntimeError(record)
        record.update(transport=transport, draft=draft, rep=rep)
        records.append(record)
        pathlib.Path(output).write_text(json.dumps(records, indent=2))
        print(json.dumps(record), flush=True)
