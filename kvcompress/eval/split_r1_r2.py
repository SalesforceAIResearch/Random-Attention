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
"""For every cell whose run count is exactly 2x the task standard, report the two halves as
SEPARATE standard-R results — r1 = runs 0..std-1, r2 = runs std..2std-1 (Heng, 08-15) — instead
of "standard + pooled-extended". Selection is by run index, never by outcome; the two halves are
therefore two independent standard-R replicates and directly comparable to every other
standard-R cell in the table.

Non-2x extended cells (e.g. R24 vs std 16) keep the old standard-vs-extended presentation from
split_extended_runs.py and are listed here only as SKIPPED.

Output: markdown table on stdout (appended to RESULTS_TABLES.md by the caller).
"""
import glob
import os
import subprocess
import sys

ROOT = os.environ.get("RA_ENGINE", os.path.join(os.environ.get("RA_ROOT", "."), "kvcompress/harness"))
VB = sys.executable
GRADER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grader.py")
GRADER_LCB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grader_lcb.py")
STD = {"math": 2, "gpqa": 4, "aime25": 16, "aime26": 16, "hmmt": 16, "livecodebench": 4}
NPROB = {"math": 500, "gpqa": 198, "aime25": 29, "aime26": 30, "hmmt": 60, "livecodebench": 383}


def task_of(leaf):
    for t in ("aime25", "aime26", "math", "gpqa", "hmmt", "livecodebench"):
        if leaf.startswith(t):
            return t
    return None


def count(d):
    n = 0
    for f in glob.glob(os.path.join(d, "**", "completions_shard*.jsonl"), recursive=True):
        with open(f, "rb") as fh:
            n += sum(1 for _ in fh)
    return n


def grade(base, method, data):
    if data == "livecodebench":
        out = subprocess.run([VB, GRADER_LCB, "--base", base, "--methods", method,
                              "--timeout", "20", "--workers", "8"],
                             capture_output=True, text=True, cwd=ROOT).stdout
        for line in out.splitlines():
            f = line.split()
            if f and f[0] == method and len(f) >= 5:
                return f[1], f[3]      # pass@1, n
    else:
        out = subprocess.run([VB, GRADER, "--base", base, "--data_name", data, "--methods", method],
                             capture_output=True, text=True, cwd=ROOT).stdout
        for line in out.splitlines():
            f = line.split()
            if f and f[0] == method and len(f) >= 7:
                return f[2], f[6]      # flag_acc, n (runs col f[6] unused here)
    return None, None


def half_view(cell, method, sub, lo, hi, tag):
    view = f"{cell}/_{tag}_{method}"
    leaf = os.path.join(view, os.path.basename(sub.rstrip("/")))
    subprocess.run(["rm", "-rf", view])
    os.makedirs(leaf, exist_ok=True)
    for r in range(lo, hi):
        src = os.path.join(os.getcwd(), sub.rstrip("/"), f"run_{r}")
        if os.path.isdir(src):
            os.symlink(src, os.path.join(leaf, f"run_{r - lo}"))
    return view, f"_{tag}_{method}"


def main():
    os.chdir(ROOT)
    print("| cell | method | task | r1 / r2 / … (run-index slices, each standard-R) | max−min |")
    print("|---|---|---|---|---|")
    for mdir in sorted(glob.glob("gate2/*/*_K*/*/")):
        parts = mdir.rstrip("/").split("/")
        if len(parts) != 4 or "_attic" in mdir or parts[3].startswith("_"):
            continue
        cell, method = "/".join(parts[:3]), parts[3]
        for sub in glob.glob(os.path.join(mdir, "*/")):
            data = task_of(os.path.basename(sub.rstrip("/")))
            if not data or data not in STD:
                continue
            n = count(sub)
            if not n:
                continue
            R = n / NPROB[data]
            if abs(R - round(R)) > 0.02:
                continue
            R = int(round(R))
            s = STD[data]
            if R <= s or R % s != 0:
                if R > s:
                    print(f"| {cell} | {method} | {data} | SKIPPED (R{R} not a multiple of std R{s}) |",
                          file=sys.stderr)
                continue
            k = R // s
            accs = []
            for i in range(k):
                v, m = half_view(cell, method, sub, i * s, (i + 1) * s, f"r{i+1}")
                a, _n = grade(cell, m, data)
                subprocess.run(["rm", "-rf", v])
                try:
                    accs.append(float(a))
                except (TypeError, ValueError):
                    print(f"UNPARSEABLE grade for {cell}/{method} slice r{i+1}: {a!r}",
                          file=sys.stderr)
                    accs = None
                    break
            if accs:
                cells_txt = " / ".join(f"{a:.4f}" for a in accs)
                spread = max(accs) - min(accs)
                print(f"| {cell} | {method} | {data} | {cells_txt} (each R{s}) | spread {spread:.4f} |")
                sys.stdout.flush()


if __name__ == "__main__":
    main()
