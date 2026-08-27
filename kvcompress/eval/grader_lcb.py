# Copyright (c) 2026, Salesforce, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Grade VaSE-engine LiveCodeBench (code-generation) completions at pass@1 via REAL CODE EXECUTION
(NOT math_verify), with the same I/O + indexing + aggregation layer as grade_ours.py. Run with the
env.sh PYTHONPATH (kvcompress/harness). Reads completions_shard<S>.jsonl (problem index =
S + line), works on partial runs. Reports pass@1 / term_rate / n / runs per method.

------------------------------------------------------------------------------------------------
TEST CASES (read this if grading prints "(no test cases ...)"):
  LiveCodeBench test cases are NOT inline in data/livecodebench/test.jsonl. Each record only carries
  {"tests": {"fname": "<global_id>.json", "md5": ...}}. The actual cases live in
      data/livecodebench/livecodebench_tests/<fname>
  materialised by data/livecodebench/download_tests.py. That script uses
  load_dataset("livecodebench/code_generation_lite", version_tag="release_v6", trust_remote_code=True)
  which FAILS on datasets>=3 ("Dataset scripts are no longer supported"). Workaround actually used to
  materialise the files in this repo (network required, one-time): download the 6 raw shards
  test.jsonl..test6.jsonl from the HF dataset repo, concatenate them in order (== release_v6, 1055
  problems), index by each record's `global_id`, decode public_test_cases (json) + private_test_cases
  (json, else base64->zlib->pickle), and write
      {"input_output": json.dumps({"inputs":[...], "outputs":[...], "fn_name": metadata.func_name})}
  to livecodebench_tests/<fname>. If the dir is absent the grader prints "(no test cases ...)" for
  every problem and reports n=0 -- re-materialise it first (see the materialise snippet at bottom).

RECORD / TEST-CASE SCHEMA (what was found):
  data/livecodebench/test.jsonl : 383 records (MEDIUM-difficulty subset of release_v6). Keys:
    global_id, question_id, contest_id, contest_date, prompt, tests{fname,md5}  (load_data adds idx).
  livecodebench_tests/<fname>.json : {"input_output": "<json string>"} whose decoded payload is
    {"inputs": [str,...], "outputs": [str,...], "fn_name": str|None}.
  Two problem types (split: 176 stdin / 207 functional):
    * fn_name == None  -> STDIN/STDOUT: `inputs[i]` is fed on stdin, compare program stdout to
      `outputs[i]` (per-line rstrip, drop trailing blank lines; numeric-token fallback).
    * fn_name == "<name>" -> FUNCTIONAL/call-based (LeetCode style, starter `class Solution:`):
      each line of `inputs[i]` is ONE argument as a json/py literal; instantiate Solution() (or use a
      top-level fn) and call `<name>(*args)`; compare return to json/py-literal of `outputs[i]`.
  A completion passes a problem only if it passes ALL of that problem's test cases (pass@1).

SANDBOXING: every candidate runs in a SUBPROCESS with a hard wall-clock timeout (--timeout s/test)
  and a memory cap (--mem_mb via RLIMIT_AS, POSIX). stdin problems run one fresh process per test;
  functional problems run one driver process that applies a per-test SIGALRM timeout. Any exception /
  timeout / nonzero exit / mismatch => that test fails. A hung or crashing child can never hang or
  crash the grader (outer subprocess timeout is the backstop). Candidates are TRUSTED self-gen output
  -- no syscall sandboxing beyond timeouts + memory cap.

CLI (drop-in twin of grade_ours.py):
  python grade_lcb.py --base ./gate2/Qwen3-4B/livecodebench_K2048 \
      --methods dense,random_pp,range_sink_sample_attn,attn,caote --data_name livecodebench
  python grade_lcb.py --selftest      # offline unit test of the execution core
------------------------------------------------------------------------------------------------
"""
import argparse, glob, json, os, re, sys, subprocess, tempfile, hashlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# ----------------------------------------------------------------------------- functional driver
# Standalone program: argv = candidate.py  tests.json  per_test_timeout_seconds . Prints PASS/FAIL.
DRIVER_SRC = r'''
import sys, json, ast, signal

def _parse_args(inp):
    lines = inp.split("\n")
    while lines and lines[-1] == "":
        lines.pop()
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except Exception:
            try:
                out.append(ast.literal_eval(ln))
            except Exception:
                out.append(ln)
    return out

def _parse_out(o):
    try:
        return json.loads(o)
    except Exception:
        try:
            return ast.literal_eval(o)
        except Exception:
            return o.strip()

def _norm(x):
    if isinstance(x, (list, tuple)):
        return [_norm(v) for v in x]
    return x

def _eq(a, b):
    try:
        if a == b:
            return True
    except Exception:
        pass
    try:
        if _norm(a) == _norm(b):
            return True
    except Exception:
        pass
    try:
        if abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(b))):
            return True
    except Exception:
        pass
    return False

class _TO(Exception):
    pass

def _alarm(s, f):
    raise _TO()

def main():
    cand, tpath, per = sys.argv[1], sys.argv[2], int(sys.argv[3])
    code = open(cand).read()
    t = json.load(open(tpath))
    fn_name, inputs, outputs = t["fn_name"], t["inputs"], t["outputs"]
    g = {"__name__": "__cand__"}
    try:
        exec("from typing import *", g)
        exec("import math, collections, itertools, heapq, bisect, re, functools, string, sys", g)
    except Exception:
        pass
    try:
        exec(code, g)
    except Exception:
        print("FAIL"); return
    def get_fn():
        if "Solution" in g and isinstance(g["Solution"], type):
            inst = g["Solution"]()
            if hasattr(inst, fn_name):
                return getattr(inst, fn_name)
        if fn_name and fn_name in g and callable(g[fn_name]):
            return g[fn_name]
        return None
    try:
        signal.signal(signal.SIGALRM, _alarm)
    except Exception:
        pass
    for inp, exp in zip(inputs, outputs):
        args = _parse_args(inp)
        want = _parse_out(exp)
        try:
            signal.alarm(max(1, per))
        except Exception:
            pass
        try:
            fn = get_fn()
            if fn is None:
                print("FAIL"); return
            got = fn(*args)
        except _TO:
            print("FAIL"); return
        except Exception:
            print("FAIL"); return
        finally:
            try:
                signal.alarm(0)
            except Exception:
                pass
        if not _eq(got, want):
            print("FAIL"); return
    print("PASS")

main()
'''


# ----------------------------------------------------------------------------- code extraction
def extract_code(text):
    """Last fenced code block; prefer the last ```python block, strip a leading bare-lang line.
    Fall back to the largest def/class/import-anchored tail, else "" ."""
    if not text:
        return ""
    fences = re.findall(r"```(.*?)```", text, re.S)
    if fences:
        cand = None
        for f in fences:  # prefer LAST python-tagged block
            head = f.split("\n", 1)[0].strip().lower()
            if head in ("python", "py", "python3"):
                cand = f
        if cand is None:
            cand = fences[-1]
        first, _, rest = cand.partition("\n")
        if first.strip().lower() in ("python", "py", "python3", ""):
            cand = rest
        return cand.strip("\n")
    m = re.search(r"(?m)^[ \t]*(?:def |class |import |from )", text)
    return text[m.start():] if m else ""


# ----------------------------------------------------------------------------- sandbox helpers
def _limits(cpu_sec, mem_mb):
    """Return a POSIX preexec_fn that caps CPU seconds and address space; None if unsupported."""
    try:
        import resource  # noqa
    except Exception:
        return None

    def _set():
        import resource
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (int(cpu_sec) + 1, int(cpu_sec) + 2))
        except Exception:
            pass
        if mem_mb and mem_mb > 0:
            try:
                b = int(mem_mb) * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (b, b))
            except Exception:
                pass
    return _set


def norm_stdout(s):
    lines = s.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines = [ln.rstrip() for ln in lines]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def stdout_match(actual, expected):
    if norm_stdout(actual) == norm_stdout(expected):
        return True
    at, et = actual.split(), expected.split()  # whitespace-token / numeric fallback
    if len(at) == len(et) and at:
        for a, e in zip(at, et):
            if a == e:
                continue
            try:
                if abs(float(a) - float(e)) <= 1e-6 * max(1.0, abs(float(e))):
                    continue
            except Exception:
                pass
            return False
        return True
    return False


def judge_stdin(code, inputs, outputs, timeout, mem_mb, py):
    with tempfile.TemporaryDirectory() as d:
        cand = os.path.join(d, "cand.py")
        with open(cand, "w") as f:
            f.write(code)
        pre = _limits(timeout, mem_mb)
        for inp, exp in zip(inputs, outputs):
            try:
                p = subprocess.run([py, cand], input=(inp or "").encode("utf-8"),
                                   capture_output=True, timeout=timeout, preexec_fn=pre)
            except Exception:
                return 0
            if p.returncode != 0:
                return 0
            if not stdout_match(p.stdout.decode("utf-8", "replace"), exp or ""):
                return 0
        return 1


def judge_functional(code, inputs, outputs, fn_name, timeout, mem_mb, py):
    with tempfile.TemporaryDirectory() as d:
        cand = os.path.join(d, "cand.py")
        drv = os.path.join(d, "driver.py")
        tj = os.path.join(d, "tests.json")
        with open(cand, "w") as f:
            f.write(code)
        with open(drv, "w") as f:
            f.write(DRIVER_SRC)
        with open(tj, "w") as f:
            json.dump({"inputs": inputs, "outputs": outputs, "fn_name": fn_name}, f)
        total = min(timeout * max(1, len(inputs)) + 5, timeout * 80 + 30)  # wall backstop
        pre = _limits(total, mem_mb)
        try:
            p = subprocess.run([py, drv, cand, tj, str(timeout)],
                               capture_output=True, timeout=total, preexec_fn=pre)
        except Exception:
            return 0
        toks = p.stdout.decode("utf-8", "replace").split()
        return 1 if toks and toks[-1] == "PASS" else 0


def judge_completion(comp, problem, timeout, mem_mb, py, max_tests=0):
    """problem = (fn_name, inputs, outputs) or None (ungradeable). Returns 1/0, or None if ungradeable.
    max_tests>0 caps the number of test cases per problem (speeds up giant stress-test problems; a
    completion still must pass ALL graded cases)."""
    if problem is None:
        return None
    fn_name, inputs, outputs = problem
    if not inputs:
        return None
    if max_tests and len(inputs) > max_tests:
        inputs, outputs = inputs[:max_tests], outputs[:max_tests]
    code = extract_code(comp)
    if not code.strip():
        return 0
    if fn_name:
        return judge_functional(code, inputs, outputs, fn_name, timeout, mem_mb, py)
    return judge_stdin(code, inputs, outputs, timeout, mem_mb, py)


# ----------------------------------------------------------------------------- problem loading (lazy)
class ProblemStore:
    """Maps problem index -> test cases, loaded LAZILY. The on-disk test files total ~2.3GB (a few
    are >150MB stress tests), so they are NEVER all parsed at once: get(gi) parses one file on demand
    and keeps it in a size-bounded cache (cache_mb) so repeated completions for the same problem don't
    re-parse, while giant files are dropped after use. Reading test.jsonl directly and
    load_data('livecodebench','test','./data') give the SAME index order (load_data only adds idx and
    sorts by it -- a no-op on disk order); NEITHER inlines test cases, both point at tests.fname."""

    def __init__(self, problems_jsonl, tests_dir, cache_mb=1500):
        self.tests_dir = tests_dir
        self.cap = int(cache_mb) * 1024 * 1024
        self.cache, self.cache_bytes = {}, 0
        self.fnames = []
        if os.path.exists(problems_jsonl):
            for l in open(problems_jsonl):
                if l.strip():
                    self.fnames.append((json.loads(l).get("tests") or {}).get("fname"))

    def n_total(self):
        return len(self.fnames)

    def has_tests(self, gi):
        if gi < 0 or gi >= len(self.fnames):
            return False
        fn = self.fnames[gi]
        return bool(fn) and os.path.exists(os.path.join(self.tests_dir, fn))

    def get(self, gi):
        """Return (fn_name, inputs, outputs) or None. Call from a single thread (loader is not locked)."""
        if gi < 0 or gi >= len(self.fnames):
            return None
        fn = self.fnames[gi]
        if not fn:
            return None
        if fn in self.cache:
            return self.cache[fn]
        path = os.path.join(self.tests_dir, fn)
        if not os.path.exists(path):
            return None
        try:
            io = json.loads(json.load(open(path))["input_output"])
            prob = (io.get("fn_name"), io.get("inputs") or [], io.get("outputs") or [])
        except Exception:
            return None
        try:
            sz = os.path.getsize(path)
        except Exception:
            sz = 0
        if self.cache_bytes + sz <= self.cap:  # cache small/medium files; drop giant ones after use
            self.cache[fn] = prob
            self.cache_bytes += sz
        return prob


# ----------------------------------------------------------------------------- grading (mirrors grade_ours)
def grade_records(method_dir, store, data_name, max_tokens, timeout, mem_mb, workers, py, cache,
                  limit=0, max_tests=0):
    """Return (records, n_runs) where records = list of (gi, flag, term) per completion.
    Same glob + '/<data_name>_bs' filter + index (shard+line) + generate_lens term axis as grade_ours.
    Completions are grouped by problem index so each (lazy) test file is loaded at most once per pass
    and freed before the next index (bounds peak memory)."""
    by_gi, slots, runs = {}, [], set()
    # POSITIONAL-CORRUPTION GUARD, mirroring grade_records() in grader.py. Without it this grader
    # silently double-counts: a shard whose line count runs past the next shard's start index makes
    # two files claim the same problem, and `by_gi.setdefault(gi, []).append(...)` just stacks both
    # completions onto that problem. That is exactly how the 14B LCB TriAttention cell (4 run dirs,
    # 3 duplicated indices each) was graded to 0.5744 on 08-02 without anyone noticing — grader.py
    # would have refused it. Overruns past n_total were likewise dropped in silence below; both now
    # raise, because a cell that needs repair must not be quietly gradeable.
    seen_by_run = {}
    for cf in sorted(glob.glob(os.path.join(method_dir, "**", "completions_shard*.jsonl"), recursive=True)):
        if f"/{data_name}_bs" not in cf:
            continue
        m = re.search(r"completions_shard(\d+)\.jsonl$", os.path.basename(cf))
        shard = int(m.group(1)) if m else 0
        run_dir = os.path.dirname(cf)
        runs.add(run_dir)
        seen = seen_by_run.setdefault(run_dir, set())
        comps = [json.loads(l)["completion"] for l in open(cf) if l.strip()]
        oi = cf.replace("completions_shard", "other_info_shard").replace(".jsonl", ".json")
        glens = json.load(open(oi)).get("generate_lens", []) if os.path.exists(oi) else []
        for i, c in enumerate(comps):
            gi = shard + i
            if gi >= store.n_total():
                raise RuntimeError(
                    f"POSITIONAL CORRUPTION: {cf} line {i} -> problem index {gi} >= n_total "
                    f"{store.n_total()}. Run kvcompress/eval/trim_shard_overlaps.py on this subdir.")
            if gi in seen:
                raise RuntimeError(
                    f"POSITIONAL CORRUPTION: problem index {gi} appears twice within run dir "
                    f"{run_dir} (overlapping/resumed shard lines). Run "
                    f"kvcompress/eval/trim_shard_overlaps.py on this subdir, then re-grade.")
            seen.add(gi)
            if not store.has_tests(gi):
                continue
            if limit and gi >= limit:
                continue
            t = int(glens[i] < max_tokens - 1000) if i < len(glens) else int(bool(c.strip()))
            slot = len(slots)
            slots.append(None)
            by_gi.setdefault(gi, []).append((slot, t, c))

    if slots:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            for gi in sorted(by_gi):
                problem = store.get(gi)  # single-threaded load (one file resident at a time)
                batch = by_gi[gi]
                if problem is None:
                    for slot, t, c in batch:
                        slots[slot] = (gi, 0, t)
                    continue

                def work(item):
                    slot, t, c = item
                    key = (gi, hashlib.md5(c.encode("utf-8", "replace")).hexdigest())
                    f = cache.get(key)
                    if f is None:
                        f = judge_completion(c, problem, timeout, mem_mb, py, max_tests)
                        cache[key] = f
                    return slot, (gi, int(f or 0), t)

                for slot, rec in ex.map(work, batch):
                    slots[slot] = rec
    return [r for r in slots if r is not None], len(runs)


def agg(recs, n_runs):
    n = len(recs)
    g = lambda vals: round(float(np.mean(vals)), 4) if n else None
    return dict(n=n, runs=n_runs,
                pass1=g([r[1] for r in recs]),
                term_rate=g([r[2] for r in recs]))


# ----------------------------------------------------------------------------- self-test
def _selftest():
    py = sys.executable
    print(f"# selftest python={py}")
    ok_all = True

    def check(name, got, exp):
        nonlocal ok_all
        res = "PASS" if got == exp else "FAIL"
        if got != exp:
            ok_all = False
        print(f"  [{res}] {name}: got={got} expected={exp}")

    # --- code extraction ---
    comp = "reasoning...\n```python\nprint('first')\n```\nmore\n```python\nprint('last')\n```\n"
    check("extract_last_python_block", extract_code(comp).strip(), "print('last')")

    # --- STDIN problem: read two ints, print sum ---
    inputs = ["2 3\n"]
    outputs = ["5\n"]
    correct = "a,b=map(int,input().split())\nprint(a+b)\n"
    wrong = "a,b=map(int,input().split())\nprint(a-b)\n"
    loop = "while True:\n    pass\n"
    check("stdin_correct->pass", judge_stdin(correct, inputs, outputs, 6, 2048, py), 1)
    check("stdin_wrong->fail", judge_stdin(wrong, inputs, outputs, 6, 2048, py), 0)
    check("stdin_infloop->timeout-fail", judge_stdin(loop, inputs, outputs, 3, 2048, py), 0)

    # --- FUNCTIONAL problem: add(a,b) (two args, one per line); also list-return + numeric ---
    fin = ["2\n3\n", "10\n-4\n"]
    fout = ["5", "6"]
    fcorrect = "class Solution:\n    def add(self, a, b):\n        return a + b\n"
    fwrong = "class Solution:\n    def add(self, a, b):\n        return a * b\n"
    floop = "class Solution:\n    def add(self, a, b):\n        while True:\n            pass\n"
    check("func_correct->pass", judge_functional(fcorrect, fin, fout, "add", 6, 2048, py), 1)
    check("func_wrong->fail", judge_functional(fwrong, fin, fout, "add", 6, 2048, py), 0)
    check("func_infloop->timeout-fail", judge_functional(floop, fin, fout, "add", 3, 2048, py), 0)

    # --- functional list/tuple + typing-annotation tolerance ---
    lin = ["[3, 1, 2]"]
    lout = ["[1, 2, 3]"]
    lcode = "from typing import List\nclass Solution:\n    def srt(self, a: List[int]) -> List[int]:\n        return sorted(a)\n"
    check("func_list_typing->pass", judge_functional(lcode, lin, lout, "srt", 6, 2048, py), 1)

    # --- end-to-end: build a fake completions tree + tiny problem set, run grade_records ---
    with tempfile.TemporaryDirectory() as d:
        tdir = os.path.join(d, "tests"); os.makedirs(tdir)
        with open(os.path.join(tdir, "0.json"), "w") as f:
            json.dump({"input_output": json.dumps({"inputs": inputs, "outputs": outputs, "fn_name": None})}, f)
        with open(os.path.join(tdir, "1.json"), "w") as f:
            json.dump({"input_output": json.dumps({"inputs": fin, "outputs": fout, "fn_name": "add"})}, f)
        pj = os.path.join(d, "test.jsonl")
        with open(pj, "w") as f:
            f.write(json.dumps({"tests": {"fname": "0.json"}}) + "\n")
            f.write(json.dumps({"tests": {"fname": "1.json"}}) + "\n")
        store = ProblemStore(pj, tdir)
        check("store_count", store.n_total(), 2)
        check("store_lazy_get", store.get(1)[0], "add")
        rundir = os.path.join(d, "base", "dense", "livecodebench_bs8_x", "run_0"); os.makedirs(rundir)
        # shard0 line0 -> problem 0 (stdin), line1 -> problem 1 (functional); both correct
        with open(os.path.join(rundir, "completions_shard0.jsonl"), "w") as f:
            f.write(json.dumps({"completion": "```python\n" + correct + "```"}) + "\n")
            f.write(json.dumps({"completion": "```python\n" + fcorrect + "```"}) + "\n")
        recs, nruns = grade_records(os.path.join(d, "base", "dense"), store, "livecodebench",
                                    16000, 20, 2048, 4, py, {}, 0, 0)  # generous timeout: wiring check
        s = agg(recs, nruns)
        check("e2e_index+pass1", (s["pass1"], s["n"], s["runs"]), (1.0, 2, 1))

    print(f"# selftest overall: {'ALL PASS' if ok_all else 'SOME FAILED'}")
    return 0 if ok_all else 1


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="LiveCodeBench pass@1 grader (real code execution).")
    ap.add_argument("--base", help="e.g. ./gate2/Qwen3-4B/livecodebench_K2048")
    ap.add_argument("--methods", default="dense,random_pp,range_sink_sample_attn,attn,caote")
    ap.add_argument("--data_name", default="livecodebench")
    ap.add_argument("--max_tokens", type=int, default=16000)
    ap.add_argument("--dense_max_tokens", type=int, default=32768)
    ap.add_argument("--tests_dir", default="./data/livecodebench/livecodebench_tests")
    ap.add_argument("--problems_jsonl", default="./data/livecodebench/test.jsonl")
    ap.add_argument("--timeout", type=int, default=6, help="hard wall-clock seconds per test case")
    ap.add_argument("--mem_mb", type=int, default=4096, help="per-candidate RLIMIT_AS cap (0=off)")
    ap.add_argument("--workers", type=int, default=16, help="parallel candidates (threads -> subprocs)")
    ap.add_argument("--limit", type=int, default=0, help="only grade problem indices < limit (debug)")
    ap.add_argument("--max_tests", type=int, default=0,
                    help="cap test cases per problem (0=all; speeds up giant stress-test problems)")
    ap.add_argument("--tests_cache_mb", type=int, default=1500,
                    help="parsed-test-file cache budget (giant files dropped after use)")
    ap.add_argument("--python", default=sys.executable, help="interpreter for candidate subprocesses")
    ap.add_argument("--selftest", action="store_true", help="run offline unit tests and exit")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(_selftest())
    if not args.base:
        ap.error("--base is required (or use --selftest)")

    store = ProblemStore(args.problems_jsonl, args.tests_dir, args.tests_cache_mb)
    n_total = store.n_total()
    n_have = sum(1 for gi in range(n_total) if store.has_tests(gi))
    if n_total == 0:
        print(f"# ERROR: no problems loaded from {args.problems_jsonl}")
        sys.exit(1)
    if n_have == 0:
        print(f"# WARNING: 0/{n_total} test files present under {args.tests_dir} -- "
              f"run data/livecodebench/download_tests.py (see header) to materialise them; "
              f"every problem will be ungradeable until then.")

    print(f"# lcb-grader {args.base} data={args.data_name} "
          f"problems={n_total} with_tests={n_have} "
          f"timeout={args.timeout}s mem={args.mem_mb}MB workers={args.workers} "
          f"max_tests={args.max_tests or 'all'}")
    hdr = f'{"method":24s} {"pass@1":>8} {"term_rate":>9} {"n":>6} {"runs":>5}'
    print(hdr)
    cache = {}
    for mth in args.methods.split(","):
        md = os.path.join(args.base, mth)
        if not os.path.isdir(md):
            print(f"{mth:24s} {'(missing)':>8}")
            continue
        mt = args.dense_max_tokens if mth == "dense" else args.max_tokens
        recs, n_runs = grade_records(md, store, args.data_name, mt, args.timeout, args.mem_mb,
                                     args.workers, args.python, cache, args.limit, args.max_tests)
        s = agg(recs, n_runs)
        print(f'{mth:24s} {str(s["pass1"]):>8} {str(s["term_rate"]):>9} {s["n"]:>6} {s["runs"]:>5}')


if __name__ == "__main__":
    main()
