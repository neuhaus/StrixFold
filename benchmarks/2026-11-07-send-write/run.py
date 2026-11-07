"""Run paired, byte-verified host-buffer SEND/WRITE measurements and restore serving."""
import concurrent.futures
import json
import pathlib
import shlex
import subprocess
import time

OUT = pathlib.Path(__file__).resolve().parent
STATE = "/home/sn/.local/state/strixfold-weights"
LAB = STATE + "/send-write-bench-20261107"
DEPLOY = STATE + "/deploy-a4e4814"
MODES = [("write", "WRITE_IMM", True), ("send", "SEND", False), ("send_persistent", "SEND", True)]


def run_host(host, name, args):
    command = ["flock", "-n", "/var/tmp/halo-gpu.lock", "timeout", "100s", "env",
               "TF_COLL_KIND=verbs", "TF_COLL_VERBS_DEVICE=usb4_rdma0",
               DEPLOY + "/run-native.sh", LAB + "/tbv_gpu_verbs-current", *args]
    if host == "misty":
        command += ["--host", "192.168.211.2"]
    result = subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", host, shlex.join(command)],
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=115)
    (OUT / f"{name}-{host}.log").write_text(result.stdout)
    if result.returncode or "FAIL:" in result.stdout or result.stdout.count("VERIFY PASS") != 3:
        raise RuntimeError(f"{name} on {host} failed. See saved log. Exit {result.returncode}")
    rows = []
    for line in result.stdout.splitlines():
        if line.startswith(("RESULT ", "PHASE ")):
            rows.append({"kind": line.split()[0], "host": host,
                         **dict(field.split("=", 1) for field in line.split()[1:])})
    return rows


def main():
    records = []
    cases = [(size, 1) for size in (64, 4096, 16384, 65536, 1048576, 4194304, 20971520)]
    cases += [(size, 3) for size in (1048576, 4194304, 20971520)]
    try:
        for index, (size, depth) in enumerate(cases):
            modes = MODES if index % 2 == 0 else list(reversed(MODES))
            for label, operation, persistent in modes:
                name = f"{size}-depth{depth}-{label}"
                args = ["--device", "usb4_rdma0", "--gid-index", "1", "--port", "19760",
                        "--op", operation, "--host-mr", "--bytes", str(size), "--depth", str(depth),
                        "--iterations", "128" if size <= 65536 else "16", "--rounds", "3",
                        "--spin-us", "5000"]
                if persistent:
                    args.append("--persistent-rq")
                print(json.dumps({"starting": name}), flush=True)
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    server = pool.submit(run_host, "fuzzy", name, args)
                    time.sleep(0.2)
                    client = pool.submit(run_host, "misty", name, args)
                    rows = server.result() + client.result()
                for row in rows:
                    row.update(mode=label, case=name)
                records.extend(rows)
                (OUT / "measurements.json").write_text(json.dumps(records, indent=2))
                print(json.dumps({"complete": name,
                                  "us_per_message": [r["us_per_message"] for r in rows if r["kind"] == "RESULT"]}),
                      flush=True)
    finally:
        command = (f"RUN_ID=20261107-restored-after-send-write TF_SERVE_CONF={DEPLOY}/verbs.conf "
                   f"{DEPLOY}/tools/dsv41-hip/first_light.sh up")
        print("Restoring public Verbs service", flush=True)
        result = subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", "misty", command],
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
        (OUT / "restore.log").write_text(result.stdout)
        print(result.stdout, flush=True)
        result.check_returncode()


if __name__ == "__main__":
    main()
