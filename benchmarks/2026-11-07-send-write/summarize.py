"""Summarize each three-round paired test using the slower host in each round."""
import json
import pathlib
import statistics

root = pathlib.Path(__file__).resolve().parent
records = json.loads((root / "measurements.json").read_text())
results = [r for r in records if r["kind"] == "RESULT"]
summary = []
for case in dict.fromkeys(r["case"] for r in results):
    rows = [r for r in results if r["case"] == case]
    assert len(rows) == 6
    slower = [max((r for r in rows if int(r["round"]) == rep),
                  key=lambda r: float(r["us_per_message"])) for rep in range(3)]
    phases = [next(r for r in records if r["kind"] == "PHASE" and r["case"] == case
                   and r["host"] == s["host"] and r["round"] == s["round"]) for s in slower]
    cycles = [float(r["us_per_message"]) for r in slower]
    protocol = [sum(float(r[k]) for k in ("ready_us", "wr_post_us", "cq_wait_us")) for r in phases]
    median = statistics.median(cycles)
    row = dict(case=case, mode=rows[0]["mode"], bytes=int(rows[0]["bytes"]), depth=int(rows[0]["depth"]),
               cycle_median_us=median, cycle_min_us=min(cycles), cycle_max_us=max(cycles),
               protocol_median_us=statistics.median(protocol),
               effective_gib_s=int(rows[0]["bytes"]) * 1e6 / median / 2**30)
    summary.append(row)
    print(json.dumps(row))
assert len(summary) == 30
(root / "summary.json").write_text(json.dumps(summary, indent=2))
