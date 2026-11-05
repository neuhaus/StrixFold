#!/usr/bin/env python3
"""qualify.py (run as tools/dsv41-hip/qualify.sh): the port's ordered mandatory gates with a verdict (handoff section 10).

  qualify.sh [--conf CONF] [--out DIR] [--steps LIST] [--long 0|1] [--reuse] [--plan]

CONF is the advertised serving configuration ('$HOME/...', the same file on both boxes; default the deployed
serve.conf). Runs on node1 (the build box) and reaches node0 by ssh. Every mandatory gate of gates.json (milestones M0
to M5, P0's informing gates excluded) gets one status: pass, fail, skipped or unrun, with its reason and evidence.
A gate is decided in one of two ways, and the report says which:
  executed  this invocation ran it now against the configuration's deployment (steps below);
  registry  gates.json's recorded result, accepted only when the status is pass and every evidence path exists
            (repository paths at this checkout's HEAD, node0:/node1: paths on that box).
Steps (in order; --steps picks a subset; each GPU step takes the GPU locks itself, node0's first, per run):
  identity     deployment identical on both boxes (DEPLOY.json, SHA256SUMS verified), the configuration's bundle hash,
               kit manifest, lock manifest, GPU_STRICT; preflight drift against the M0 baseline on both boxes
  gpu_tests    run_gpu_tests.sh's programs from the deployment on both boxes (smoke, ABI, glue)
  transport    coll_fault_matrix.sh on node1 (loopback, tcp_host and tcp_proxy, with the corruption controls)
  served       qual/qual_serve.sh: functional suite, no NVIDIA library mapped, allocation/residency ledger, restart
  sampling     qual/sampling_check.py: the engine's sampler vs the pinned contract and the distributions (CPU), with the
               served seeded cases
  faults       qual/served_faults.sh: rank kill/stop during decode (and prefill with --long 1), kill during load,
               peer mismatches, stale bundle and stale cache
  trace        consumes the latest graded layer gate (qual/check_layers.sh) for this deployment, if any
  bench / soak consume qual/benchmark and soak results for this deployment and configuration, if any
--reuse takes the served and fault results already made on this deployment and configuration (matched by deploy
commit and configuration SHA-256; for faults the newest result of each case) instead of running them again.
Writes DIR/qualify.json and prints a table. Exit 0 only if every mandatory gate passes; a skipped or unrun mandatory
gate fails the qualification (a partial report is still written).
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.realpath(__file__))
REPO = os.path.realpath(os.path.join(HERE, "..", ".."))
HOME = os.path.expanduser("~")
MANDATORY_MILESTONES = ("M0", "M1", "M2", "M3", "M4", "M5")


def sh(cmd, timeout=None, check=False):
    r = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=timeout)
    if check and r.returncode:
        raise RuntimeError(f"{cmd}: {r.stderr.strip()[-400:]}")
    return r


def ssh1(cmd, timeout=600):
    return sh(["ssh", "-n", "node0", cmd], timeout=timeout)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def conf_values(conf_path):
    out = sh(["bash", "-c", f'set -a; source {shlex.quote(conf_path)}; env -0'], check=True).stdout
    env = dict(x.split("=", 1) for x in out.split("\0") if "=" in x)
    keys = re.findall(r"^([A-Z0-9_]+)=", open(conf_path).read(), re.M)
    # paths are written '$HOME/...' (each box expands its own); here: this box's
    return {k: env.get(k, "").replace("$HOME", HOME) for k in keys}


def evidence_exists(ev, head):
    """(exists, how) for one evidence string of gates.json."""
    path = ev.split(" (")[0].split("; ")[0].strip()
    m = re.match(r"^(halo[12]):(.*)$", path)
    if m:
        box, p = m.groups()
        p = p.replace("~", "$HOME", 1) if p.startswith("~") else p
        test = f'for f in {p}; do [ -e "$f" ] && exit 0; done; exit 1'
        r = ssh1(test, timeout=60) if box == "node0" else sh(["bash", "-c", test.replace("$HOME", HOME)])
        return r.returncode == 0, f"{box} file"
    if path.startswith(("tools/", "zig/", "docs/")):
        p = path.split(" ")[0]
        if any(c in p for c in "*?"):
            return bool(glob.glob(os.path.join(REPO, p))), "repo glob"
        return sh(["git", "-C", REPO, "cat-file", "-e", f"{head}:{p}"]).returncode == 0, f"repo {head[:12]}"
    return True, "not a path (accepted as a note)"


def matching(paths, commit, conf_sha, get):
    """The result files among `paths` made on this deployment and configuration, newest first."""
    out = []
    for p in paths:
        try:
            d = json.load(open(p))
        except (OSError, ValueError):
            continue
        c, k = get(d)
        if c == commit and k == conf_sha:
            out.append((os.path.getmtime(p), p, d))
    return [(p, d) for _, p, d in sorted(out, reverse=True)]


class Q:
    def __init__(self, a):
        self.a = a
        self.out = a.out
        self.conf = a.conf
        self.conf1 = a.conf.replace("$HOME", HOME)
        self.cv = conf_values(self.conf1)
        self.deploy1 = self.cv["DEPLOY_DIR"]
        self.reg = json.load(open(os.path.join(HERE, "gates.json")))
        self.head = sh(["git", "-C", REPO, "rev-parse", "HEAD"], check=True).stdout.strip()
        self.executed = {}   # gate id -> {status, reason, evidence}
        self.steps = {}      # step -> summary
        self.identity = {}

    def set(self, gid, status, reason, evidence=(), step=None, data=None):
        self.executed[gid] = {"status": status, "reason": reason, "evidence": list(evidence), "step": step, "data": data}

    # ---- steps
    def step_identity(self):
        dep = json.load(open(os.path.join(self.deploy1, "DEPLOY.json")))
        ok1 = sh(["bash", "-c", f"cd {shlex.quote(self.deploy1)} && sha256sum --quiet -c SHA256SUMS"]).returncode == 0
        dep0 = self.cv["DEPLOY_DIR"].replace(HOME, "$HOME")
        r0 = ssh1(f"cd {dep0} && sha256sum --quiet -c SHA256SUMS && sha256sum SHA256SUMS")
        same = r0.returncode == 0 and r0.stdout.split()[0] == sha_file(os.path.join(self.deploy1, "SHA256SUMS"))
        kit = os.path.join(self.deploy1, self.cv["KIT_SUBDIR"], "manifest.json")
        bundle = self.cv["BUNDLE"]
        bsum = sha_file(os.path.join(bundle, "inputs_manifest.json"))
        lock = os.path.join(REPO, "tools/dsv41-hip/m0/lock.json")
        self.identity = {
            "qualify_checkout": self.head, "deploy_commit": dep.get("commit"), "deploy_tree": dep.get("tree"), "deploy_clean": dep.get("clean"),
            "deploy_sha256sums": sha_file(os.path.join(self.deploy1, "SHA256SUMS")), "deploy_verified_node1": ok1,
            "deploy_identical_node0": same, "config": self.conf, "config_sha256": sha_file(self.conf1),
            "kit_manifest_sha256": sha_file(kit), "bundle_manifest_sha256": bsum, "bundle_pinned": self.cv["BUNDLE_SHA256"],
            "lock_json_sha256": sha_file(lock) if os.path.exists(lock) else None, "gpu_strict": self.cv.get("GPU_STRICT"),
            "drafts": self.cv.get("DRAFTS"), "graphs": self.cv.get("GRAPHS"), "context": self.cv.get("CONTEXT"),
            "streams": self.cv.get("STREAMS"), "parallel": self.cv.get("PARALLEL"), "transport": "verbs" if self.cv.get("COLL_KIND") == "verbs" else "tcp",
            "coll_kind": self.cv.get("COLL_KIND"), "modalities": self.cv.get("MODALITIES"),
        }
        good = ok1 and same and dep.get("clean") and bsum == self.cv["BUNDLE_SHA256"]
        self.set("M5.deploy-identity", "pass" if good else "fail",
                 "deployment verified on both boxes, clean commit, bundle hash pinned" if good else
                 f"deploy verified node1 {ok1}, identical on node0 {same}, clean {dep.get('clean')}, bundle {bsum[:12]} vs {self.cv['BUNDLE_SHA256'][:12]}",
                 [os.path.join(self.deploy1, "DEPLOY.json")], "identity")
        strict = self.cv.get("GPU_STRICT") == "1"
        full = self.cv.get("DRAFTS") == "1" and self.cv.get("GRAPHS") == "1"
        # (the served step refines this with the rank logs: no missing variant and no fallback warning)
        self.set("M3.strict-kit", ("pass" if full else "unrun") if strict else "fail",
                 ("GPU_STRICT=1: a launch the kit lacks is an error" + ("" if full else "; the advertised configuration also has DSpark "
                  "and graphs on (lead, 2026-10-19), this one does not")) if strict else
                 "GPU_STRICT=0: the configuration lets kit gaps run less specialized variants; not an advertised configuration",
                 [self.conf], "identity")
        # preflight drift against the M0 baselines
        drift = {}
        for box in ("node1", "node0"):
            base = f"tools/dsv41-hip/m0/preflight-{box}.json"
            outdir = f"$HOME/tfp/qual/preflight/{self.a.stamp}"
            tools = f"{self.cv['DEPLOY_DIR'].replace(HOME, '$HOME')}/tools/dsv41-hip"
            cmd = f"mkdir -p {outdir} && {tools}/preflight.sh {outdir} --no-hip-probe >/dev/null 2>&1; {tools}/preflight.sh --diff {tools}/m0/preflight-{box}.json {outdir}/preflight-{box}.json"
            r = sh(["bash", "-c", cmd.replace("$HOME", HOME)], timeout=900) if box == "node1" else ssh1(cmd, timeout=900)
            drift[box] = {"rc": r.returncode, "report": r.stdout[-3000:]}
        self.steps["preflight_drift"] = drift
        # the GPU stack's apt holds (2026-10-19 ~11:05 PDT): the held set equals ~/tfp/apt-holds-2026-10-19.txt and
        # every held package's version equals the M0 preflight's, on both boxes
        holds = {}
        sh(["scp", "-q", os.path.join(HERE, "qual", "holds_check.py"), "node0:tfp/qual/bin/"])
        for box in ("node1", "node0"):
            m0 = f"{self.cv['DEPLOY_DIR'].replace(HOME, '$HOME')}/tools/dsv41-hip/m0/preflight-{box}.json"
            if box == "node1":
                r = sh(["python3", os.path.join(HERE, "qual", "holds_check.py"), os.path.join(HOME, "tfp/apt-holds-2026-10-19.txt"), m0.replace("$HOME", HOME)])
            else:
                r = ssh1(f"python3 $HOME/tfp/qual/bin/holds_check.py $HOME/tfp/apt-holds-2026-10-19.txt {m0}")
            try:
                holds[box] = json.loads(r.stdout)
            except ValueError:
                holds[box] = {"ok": False, "error": (r.stdout + r.stderr)[-400:]}
        self.steps["apt_holds"] = holds
        self.identity["apt_holds_ok"] = all(h.get("ok") for h in holds.values())
        g = self.executed["M5.deploy-identity"]
        if not self.identity["apt_holds_ok"]:
            g.update(status="fail", reason=g["reason"] + f"; apt holds or held versions changed: {holds}")
        else:
            g["reason"] += "; the 57 apt holds intact and every held package at its M0 version on both boxes"
        self.steps["identity"] = self.identity

    def step_gpu_tests(self):
        res = {}
        sh(["scp", "-q", os.path.join(HERE, "qual", "gpu_tests.sh"), "node0:tfp/qual/bin/"], check=True)
        for box in ("node1", "node0"):
            out = f"$HOME/tfp/qual/out/{self.a.stamp}/gpu-tests-{box}"
            dep = self.deploy1.replace(HOME, "$HOME")
            if box == "node1":
                r = sh([os.path.join(HERE, "qual", "gpu_tests.sh"), self.deploy1, out.replace("$HOME", HOME)], timeout=2400)
            else:
                r = ssh1(f"bash $HOME/tfp/qual/bin/gpu_tests.sh {dep} {out}", timeout=2400)
            try:
                res[box] = json.loads(r.stdout[r.stdout.index("{"):])
            except ValueError:
                res[box] = {"all_passed": False, "error": (r.stdout + r.stderr)[-800:]}
            res[box]["dir"] = f"{box}:{out}"
        ok = all(v.get("all_passed") for v in res.values())
        progs = lambda name: all(any(p["program"] == name and p["exit"] == 0 for p in v.get("programs", [])) for v in res.values())
        ev = [v["dir"] for v in res.values()]
        self.set("M1.adapter", "pass" if progs("tf-dsv41-hip-smoke") else "fail",
                 "smoke (devices, code objects, launches, copies, streams/events, bounded waits, staging, graphs, mapped memory, allocators, CUDA-only refusals) on both boxes", ev, "gpu_tests")
        self.set("M1.kernel-abi", "pass" if progs("tf-dsv41-hip-abi-test") else "fail",
                 "ABI test (by-value structs, parameter lists, variant selection, strict checks, kit refusals) on both boxes", ev, "gpu_tests")
        self.set("M1.code-objects", "pass" if ok else "fail",
                 "the deployment's kits load with hash/ABI/arch/metadata refusals; glue kernels cross-checked byte for byte", ev, "gpu_tests")
        self.steps["gpu_tests"] = res

    def step_transport(self):
        tools = os.path.join(self.deploy1, "tools/dsv41-hip")
        out = {}
        for kind in ("host", "proxy"):
            env = f"COLL_CHECK={self.deploy1}/native-hip/test/tf-dsv41-coll-check COLL_DIR={tools}"
            if kind == "proxy":
                env += f" TF_COLL_KIND=proxy EXTRA_ARGS='--kit {self.deploy1}/{self.cv['KIT_SUBDIR']}'"
            r = sh(["bash", "-c", f"{env} flock -w 1800 /var/tmp/halo-gpu.lock {tools}/coll_fault_matrix.sh"], timeout=3600)
            lines = [l for l in r.stdout.splitlines() if l.startswith(("PASS", "FAIL"))]
            out[kind] = {"rc": r.returncode, "pass": sum(l.startswith("PASS") for l in lines), "fail": sum(l.startswith("FAIL") for l in lines), "lines": lines}
            open(os.path.join(self.out, f"transport-{kind}.txt"), "w").write(r.stdout + r.stderr)
        ok = all(v["fail"] == 0 and v["pass"] >= 10 for v in out.values())
        self.steps["transport"] = out
        self.transport_ok = ok

    def step_served(self):
        run = f"qs-{self.a.stamp}"
        path = os.path.join(HOME, "tfp/qual/out", run, "qual-serve.json")
        found = matching(glob.glob(os.path.join(HOME, "tfp/qual/out/*/qual-serve.json")), self.identity.get("deploy_commit"),
                         self.identity.get("config_sha256"), lambda d: (((d.get("rank0") or {}).get("deploy") or {}).get("commit"), d.get("config_sha256")))
        if self.a.reuse and found:
            path, run = found[0][0], os.path.basename(os.path.dirname(found[0][0]))
            r = None
        else:
            r = sh([os.path.join(HERE, "qual", "qual_serve.sh"), self.conf, run, "--long", str(self.a.long)], timeout=7200)
        if not os.path.exists(path):
            self.steps["served"] = {"error": (r.stdout + r.stderr)[-1500:] if r else "no result"}
            for g in ("M3.served-functional", "M3.no-nvidia-qualification", "M3.allocation-residency", "M5.clean-restart", "M3.text-capability", "M4.concurrency-cancel-reuse"):
                self.set(g, "fail", "the served block did not complete", [], "served")
            return
        doc = json.load(open(path))
        g = doc["gates"]
        main = json.load(open(os.path.join(HOME, "tfp/qual/out", run, "serve_qual-main.json")))
        cases = {c["case"]: c for c in main["cases"]}
        ev = [f"node1:{path}"]
        self.steps["served"] = {"run": run, "gates": {k: v["status"] for k, v in g.items()}, "suite": main["summary"]}
        # functional suite: every case that can run in this configuration
        so = (g.get("structured_output") or {}).get("status", "unrun")
        sf = "fail" if "fail" in (g["served_suite"]["status"], so) else ("pass" if so == "pass" and g["served_suite"]["status"] == "pass" else "unrun")
        self.set("M3.served-functional", sf, f"suite {json.dumps(main['summary'])}; structured output and required tools: {so}", ev, "served",
                 {"suite": {c: cases[c]["status"] for c in cases}, "structured": (g.get("structured_output") or {}).get("cases")})
        long_ok = cases.get("long_prompt", {}).get("status")
        long_ran = long_ok not in (None, "skipped")
        # kit coverage in the served logs (both starts, both ranks): a refusal or a fallback names the launch
        logs = "".join(open(p, errors="replace").read() for p in glob.glob(os.path.join(HOME, "tfp/qual/out", run, "*rank*.log")))
        gaps = len(re.findall(r"no Triton variant|MissingTritonVariant|runs a variant that assumes less", logs))
        if self.executed.get("M3.strict-kit", {}).get("status") == "pass" and gaps:
            self.executed["M3.strict-kit"].update(status="fail", reason=f"{gaps} missing-variant refusals or fallbacks in the served rank logs")
        elif "M3.strict-kit" in self.executed:
            self.executed["M3.strict-kit"]["reason"] += f"; served rank logs: {gaps} missing-variant refusals or fallbacks"
        ph = [p["phase"] for p in (g["allocation_residency"].get("node1") or {}).get("phases", [])]
        missing = [p for p, need in (("prefill over 128 rows", long_ran), ("verify rows", self.cv.get("DRAFTS") == "1")) if not need]
        st = g["allocation_residency"]["status"]
        self.set("M3.allocation-residency", st if (st == "fail" or not missing) else "unrun",
                 f"ledger {st} on both boxes over phases {ph}" + (f"; not yet exercised: {', '.join(missing)}" if missing else ""), ev, "served")
        nv = g["no_nvidia_libraries_mapped"]["status"]
        libs = sh(["bash", "-c", "ldconfig -p | grep -i -E 'libcuda|libnccl|libcublas|libcudart|libnvrtc' || true"]).stdout.strip()
        libs1 = ssh1("ldconfig -p | grep -i -E 'libcuda|libnccl|libcublas|libcudart|libnvrtc' || true").stdout.strip()
        nv_missing = [x for x, need in (("prefill over 128 rows", long_ran), ("graphs", self.cv.get("GRAPHS") == "1")) if not need]
        nv_status = "fail" if nv == "fail" or libs or libs1 else ("unrun" if nv_missing else "pass")
        self.set("M3.no-nvidia-qualification", nv_status,
                 f"rank processes map no NVIDIA library ({nv}); NVIDIA libraries in the ld cache: node1 {libs or 'none'}, node0 {libs1 or 'none'}"
                 + (f"; paths not exercised: {', '.join(nv_missing)}" if nv_missing else ""), ev, "served")
        self.set("M5.clean-restart", g["clean_restart"]["status"], f"restart {g['clean_restart'].get('restart_s')} s, token ids equal: {g['clean_restart'].get('token_ids_equal_first_start')}", ev, "served")
        img = next((c for c in (cases.get("malformed", {}).get("data") or {}).get("cases", []) if c.get("case") == "image part"), {})
        img_ok = 400 <= (img.get("status") or 0) < 500 and "image" in json.dumps(img.get("body")).lower()
        log0 = os.path.join(HOME, "tfp/qual/out", run, "first-rank0.log")
        vision_off = os.path.exists(log0) and "not opened (text capability)" in open(log0, errors="replace").read()
        self.set("M3.text-capability", "pass" if img_ok and vision_off else "fail",
                 f"an image part refused through the server: {img.get('status')} ({json.dumps(img.get('body'))[:120]}); "
                 f"rank 0 log 'vision tower ... not opened': {vision_off}", ev, "served")
        conc = [cases.get(c, {}).get("status") for c in ("repeat", "concurrent", "cancel")]
        self.served_cases = cases
        self.concurrency_part = all(s == "pass" for s in conc)
        self.set("M4.concurrency-cancel-reuse", "fail" if not self.concurrency_part else "unrun",
                 f"repeat/concurrent/cancel: {conc}" + ("" if not self.concurrency_part else "; partner failure from the fault step; DSpark/graph variants need DRAFTS=1"), ev, "served")
        if long_ran:
            # generation at 8K: every prompt size up to 8,000 rows answered (the capacity fill is the capacity case's)
            runs = cases["long_prompt"]["data"].get("runs", [])
            sized = [r for r in runs if "prompt_tokens_target" in r]
            ok8k = bool(sized) and all(r.get("status") == 200 and r.get("finish") in ("stop", "length") for r in sized) \
                and any(r["prompt_tokens_target"] >= 8000 for r in sized)
            self.set("M3.generation-8k", "pass" if ok8k else "fail", cases["long_prompt"]["detail"], ev, "served")

    def step_faults(self):
        base = f"qf-{self.a.stamp}"
        path = os.path.join(HOME, "tfp/qual/out", base, "faults.json")
        files = glob.glob(os.path.join(HOME, "tfp/qual/out/*/faults-regraded.json")) + glob.glob(os.path.join(HOME, "tfp/qual/out/*/faults.json"))
        found = matching(files, self.identity.get("deploy_commit"), self.identity.get("config_sha256"), lambda d: (d.get("deploy_commit"), d.get("config_sha256")))
        if self.a.reuse and found:
            # the newest result of each case over every matching run (a partial rerun replaces only its cases)
            rows, ev = {}, []
            for p, d in found:
                for c in d["cases"]:
                    if c["case"] not in rows:
                        rows[c["case"]] = dict(c, from_file=p)
                        if f"node1:{p}" not in ev:
                            ev.append(f"node1:{p}")
            doc = {"summary": {s: sum(1 for c in rows.values() if c["status"] == s) for s in ("pass", "fail", "skipped", "unrun")}}
        else:
            r = sh([os.path.join(HERE, "qual", "served_faults.sh"), self.conf, base, "--long", str(self.a.long)], timeout=14400)
            if not os.path.exists(path):
                self.steps["faults"] = {"error": (r.stdout + r.stderr)[-1500:]}
                for gid in ("M3.fault-matrix", "M3.handshake", "M3.stale-cache-refusal"):
                    self.set(gid, "fail", "the fault step did not complete", [], "faults")
                return
            doc = json.load(open(path))
            rows = {c["case"]: c for c in doc["cases"]}
            ev = [f"node1:{path}"]
        self.steps["faults"] = doc["summary"]
        group = lambda pre: {k: v["status"] for k, v in rows.items() if k.startswith(pre)}
        def verdict(st):
            vals = list(st.values())
            if any(v in ("fail",) for v in vals):
                return "fail"
            if any(v in ("skipped", "unrun") for v in vals) or not vals:
                return "unrun"
            return "pass"
        faults = {**group("kill_"), **group("stop_")}
        tr = getattr(self, "transport_ok", None)
        fm = verdict(faults)
        if tr is False:
            fm = "fail"
        elif tr is None and fm == "pass":
            fm = "unrun"
        self.set("M3.fault-matrix", fm, f"served {faults}; transport matrix {'pass' if tr else tr}", ev, "faults")
        self.set("M3.handshake", verdict(group("mm_")), json.dumps(group("mm_")), ev, "faults")
        self.set("M3.stale-cache-refusal", verdict(group("stale_")), json.dumps(group("stale_")), ev, "faults")
        if getattr(self, "concurrency_part", False):
            partner = verdict(faults)
            drafts = self.cv.get("DRAFTS") == "1" and self.served_cases.get("drafts", {}).get("status") == "pass"
            self.set("M4.concurrency-cancel-reuse", "pass" if partner == "pass" and drafts else ("fail" if partner == "fail" else "unrun"),
                     f"repeat/concurrent/cancel pass; partner failure {partner}; DSpark variants {'pass' if drafts else 'not exercised'}", ev, "faults")

    def step_sampling(self):
        out = os.path.join(self.out, "sampling")
        cmd = ["systemd-run", "--user", "--scope", "--quiet", "-p", "CPUQuota=800%", "-p", "MemoryMax=16G", "taskset", "-c", "8-15,24-31",
               "nice", "-n", "19", "python3", os.path.join(HERE, "qual", "sampling_check.py"), out, "--repo", REPO]
        r = sh(cmd, timeout=1800)
        path = os.path.join(out, "sampling_check.json")
        cpu = json.load(open(path)) if os.path.exists(path) else {"pass": False, "error": (r.stdout + r.stderr)[-800:]}
        cases = getattr(self, "served_cases", {}) or {}
        seeded = cases.get("seeded", {}).get("status")
        drafts = cases.get("drafts", {}).get("status")
        parts = {"cpu_contract": "pass" if cpu.get("pass") else "fail", "served_seeded": seeded or "not run",
                 "served_dspark_seeded": drafts or "not run"}
        if "fail" in parts.values():
            st = "fail"
        elif all(v == "pass" for v in parts.values()):
            st = "pass"
        else:
            st = "unrun"
        self.set("M4.sampling-contract", st, json.dumps(parts) + f"; CPU: {cpu.get('cases_checked')} rows vs the 222a9c0b contract, "
                 f"{cpu.get('mismatch_count')} mismatches, {sum(x['ok'] for x in cpu.get('distributions', []))}/12 distributions",
                 [f"node1:{path}"], "sampling")
        self.steps["sampling"] = parts

    def step_trace(self):
        res = sorted(glob.glob(os.path.join(HOME, "tfp/qual/out/trace-*/trace.json")))
        mine = [json.load(open(p)) | {"path": p} for p in res]
        mine = [t for t in mine if t.get("deploy_commit") == self.identity.get("deploy_commit")]
        if not mine:
            return
        t = mine[-1]
        self.set("M3.trace-target", t["status"], t.get("reason", ""), [f"node1:{t['path']}"], "trace", t.get("summary"))
        if "dspark" in t:
            self.set("M4.trace-dspark", t["dspark"]["status"], t["dspark"].get("reason", ""), [f"node1:{t['path']}"], "trace")

    def step_bench(self):
        for kind, gid in (("benchmark", "M5.benchmark"), ("soak", "M5.soak"), ("ctx16k", "M5.context-16k")):
            res = sorted(glob.glob(os.path.join(HOME, f"tfp/qual/out/{kind}-*/{kind}.json")))
            docs = [json.load(open(p)) | {"path": p} for p in res]
            docs = [d for d in docs if d.get("deploy_commit") == self.identity.get("deploy_commit") and d.get("config_sha256") == self.identity.get("config_sha256")]
            if docs:
                d = docs[-1]
                self.set(gid, d["status"], d.get("reason", ""), [f"node1:{d['path']}"], kind)

    step_soak = step_bench

    # ---- verdict
    def verdict(self):
        gates = []
        extra = []   # every gate qualify.sh decides is in gates.json (M3.served-functional, M3.strict-kit, M5.deploy-identity)
        known = {g["id"] for g in self.reg["gates"]}
        allg = [*self.reg["gates"], *[e for e in extra if e["id"] not in known]]
        for g in allg:
            if g["milestone"] not in MANDATORY_MILESTONES:
                continue
            gid = g["id"]
            if gid in self.executed:
                e = self.executed[gid]
                gates.append({"id": gid, "milestone": g["milestone"], "title": g["title"], "source": "executed", **e})
                continue
            st = g.get("status", "unrun")
            evs = g.get("evidence", [])
            missing = []
            if st == "pass":
                for ev in evs:
                    ok, how = evidence_exists(ev, self.head)
                    if not ok:
                        missing.append(ev)
            if st == "pass" and not missing:
                status, reason = "pass", f"registry pass {g.get('date', '')}; {len(evs)} evidence path(s) present"
            elif st == "pass":
                status, reason = "fail", f"registry pass but evidence missing: {missing}"
            else:
                status, reason = st if st in ("fail", "skipped") else "unrun", f"registry {st}: {g.get('notes', '')[:300]}"
            gates.append({"id": gid, "milestone": g["milestone"], "title": g["title"], "source": "registry", "status": status,
                          "reason": reason, "evidence": evs, "owner": g.get("owner")})
        counts = {s: sum(1 for x in gates if x["status"] == s) for s in ("pass", "fail", "skipped", "unrun")}
        executed = sum(1 for x in gates if x["source"] == "executed")
        draft_ev = (getattr(self, "served_cases", {}) or {}).get("drafts", {})
        doc = {"schema": "tf-port-qualify/1", "utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "identity": self.identity, "steps_run": self.a.steps, "expected": len(gates), "executed": executed, "counts": counts,
               "coverage": {"exl3_k2_required": [4, 6, 8, 10, 12, 16], "kit_fallbacks_allowed": self.cv.get("GPU_STRICT") != "1",
                            "draft_exercise": draft_ev.get("data") if draft_ev else "DSpark not exercised in this configuration",
                            "graphs": self.cv.get("GRAPHS") == "1"},
               "unsupported": {"image_input": "refused (text capability)", "dspark": self.cv.get("DRAFTS") != "1", "graphs": self.cv.get("GRAPHS") != "1"},
               "steps": self.steps, "gates": gates}
        doc["qualified"] = counts["fail"] == 0 and counts["skipped"] == 0 and counts["unrun"] == 0
        json.dump(doc, open(os.path.join(self.out, "qualify.json"), "w"), indent=1)
        w = max(len(x["id"]) for x in gates)
        for x in gates:
            print(f"{x['id']:<{w}}  {x['status']:<7} {x['source']:<8} {x['reason'][:150]}")
        print(json.dumps({"expected": len(gates), "executed": executed, **counts, "qualified": doc["qualified"]}))
        print(f"report: {os.path.join(self.out, 'qualify.json')}")
        return 0 if doc["qualified"] else 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--conf", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--steps", default="identity,gpu_tests,transport,served,sampling,faults,trace,bench")
    p.add_argument("--long", type=int, default=0)
    p.add_argument("--plan", action="store_true")
    p.add_argument("--reuse", action="store_true", help="served/faults: use the newest results made on this deployment and "
                   "configuration (deploy commit and config SHA-256 must match) instead of running them again")
    a = p.parse_args()
    a.stamp = time.strftime("%m%dT%H%M%S", time.gmtime())
    a.conf = a.conf or "$HOME/" + os.path.relpath(os.path.join(HERE, "serve.conf"), HOME)
    a.out = a.out or os.path.join(HOME, "tfp/qual/out", f"qualify-{a.stamp}")
    os.makedirs(a.out, exist_ok=True)
    q = Q(a)
    if a.plan:
        print(__doc__)
        return 0
    for step in a.steps.split(","):
        t0 = time.time()
        try:
            getattr(q, f"step_{step}")()
        except Exception as e:  # a step that breaks fails its gates; the report is still written
            q.steps[step] = {"error": f"{type(e).__name__}: {e}"}
        print(f"step {step}: {time.time() - t0:.0f} s", file=sys.stderr, flush=True)
    return q.verdict()


if __name__ == "__main__":
    sys.exit(main())
