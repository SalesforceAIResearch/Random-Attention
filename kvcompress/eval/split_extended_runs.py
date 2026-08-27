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
"""For every cell whose R exceeds the task standard, report the standard-R value and the extension
value as SEPARATE numbers (Heng, 08-09) instead of a single pooled figure annotated "(R8=2x)".

Rationale: a pooled extended number is not comparable to a standard-R number elsewhere in the same
table, and it hides whether the extension moved the result. Printing both makes the extension's
effect visible and keeps the standard-R column internally comparable.

Method: build a symlink view containing ONLY runs 0..std-1 (selection by run index, never by
outcome), grade it, and print base vs extended side by side.

Usage: python split_extended_runs.py            # audits every extended cell found on disk
"""
import glob
import os
import re
import subprocess
import sys

ROOT = os.environ.get("RA_ENGINE", os.path.join(os.environ.get("RA_ROOT", "."), "kvcompress/harness"))
VB = sys.executable
GRADER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grader.py")
STD = {"math": 2, "gpqa": 4, "aime25": 16, "aime26": 16, "hmmt": 16}
NPROB = {"math": 500, "gpqa": 198, "aime25": 29, "aime26": 30, "hmmt": 60}


def task_of(leaf):
    for t in ("aime25", "aime26", "math", "gpqa", "hmmt"):
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
    out = subprocess.run([VB, GRADER, "--base", base, "--data_name", data, "--methods", method],
                         capture_output=True, text=True, cwd=ROOT).stdout
    for line in out.splitlines():
        f = line.split()
        if f and f[0] == method and len(f) >= 7:
            return f[2], f[6]      # flag_acc, n
    return None, None


def main():
    os.chdir(ROOT)
    print(f"{'cell':40s} {'method':26s} {'task':7s} {'standard-R':>22s} {'extended':>20s}  Δ")
    for mdir in sorted(glob.glob("gate2/*/*_K*/*/")):
        parts = mdir.rstrip("/").split("/")
        if len(parts) != 4 or "_attic" in mdir:
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
            if R <= STD[data] + 0.01 or abs(R - round(R)) > 0.02:
                continue
            R = int(round(R))
            view = f"{cell}/_stdR_{method}"
            leaf = os.path.join(view, os.path.basename(sub.rstrip("/")))
            subprocess.run(["rm", "-rf", view])
            os.makedirs(leaf, exist_ok=True)
            for r in range(STD[data]):
                src = os.path.join(os.getcwd(), sub.rstrip("/"), f"run_{r}")
                if os.path.isdir(src):
                    os.symlink(src, os.path.join(leaf, f"run_{r}"))
            base_acc, base_n = grade(cell, f"_stdR_{method}", data)
            ext_acc, ext_n = grade(cell, method, data)
            subprocess.run(["rm", "-rf", view])
            if base_acc and ext_acc:
                d = float(ext_acc) - float(base_acc)
                print(f"{cell:40s} {method:26s} {data:7s} "
                      f"{base_acc:>8s} (R{STD[data]}, n={base_n:>5s}) "
                      f"{ext_acc:>8s} (R{R}, n={ext_n:>5s})  {d:+.4f}")
                sys.stdout.flush()


if __name__ == "__main__":
    main()
