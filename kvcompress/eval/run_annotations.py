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
"""Per-cell run-count annotations, derived from disk (never from memory).

Rule (Heng, 08-09): every reported cell carries its R explicitly, and where R exceeds the task's
standard the multiplier is shown — so a 12-run cell is never silently compared against a 4-run one.
Metric is flag_acc throughout (acc_strict is NOT used: it penalises non-termination, and tri
terminates more often than rp in 10/14 cells, so strict shifts -0.75pp against rp on average;
switching metrics would also be metric-shopping).

Usage:
  python run_annotations.py                      # audit every cell under gate2/
  python run_annotations.py --base gate2/Qwen3-4B/gpqa_K512
Emits: cell, method, n, R, standard R, multiplier, annotation string for the paper table.
"""
import argparse
import glob
import os
import re

ROOT = os.environ.get("RA_ENGINE", os.path.join(os.environ.get("RA_ROOT", "."), "kvcompress/harness"))

# problems per task, and the STANDARD number of runs a cell ships with.
TASK = {                      # task: (n_problems, standard_R)
    "math":          (500, 2),
    "gpqa":          (198, 4),
    "aime25":        (29, 16),
    "aime26":        (30, 16),
    "hmmt":          (60, 16),
    "livecodebench": (383, 4),
}


def task_of(subdir):
    for t in ("aime25", "aime26", "livecodebench", "math", "gpqa", "hmmt"):
        if subdir.startswith(t):
            return t
    return None


def count_lines(paths):
    n = 0
    for p in paths:
        try:
            with open(p, "rb") as fh:
                n += sum(1 for _ in fh)
        except OSError:
            pass
    return n


def annotate(n, task):
    """-> (R, standard_R, multiplier, annotation)."""
    nprob, std = TASK[task]
    R = n / nprob if nprob else 0
    mult = R / std if std else 0
    if abs(R - round(R)) > 0.02:
        return R, std, mult, f"R≈{R:.2f} PARTIAL (ragged — not table-ready)"
    R = int(round(R))
    if R == std:
        return R, std, mult, f"R{R} (standard)"
    if R < std:
        return R, std, mult, f"R{R} PARTIAL ({R}/{std} of standard — annotate or exclude)"
    if abs(mult - round(mult)) < 0.02:
        return R, std, mult, f"R{R} (={int(round(mult))}× standard R{std})"
    return R, std, mult, f"R{R} ({mult:.2f}× standard R{std})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None, help="restrict to one cell dir (relative to reasoning_tasks)")
    ap.add_argument("--min_n", type=int, default=1)
    args = ap.parse_args()

    os.chdir(ROOT)
    # dir layout: gate2/<MODEL>/<CELL>/<METHOD>/<SUBDIR>/run_N/completions_shard*.jsonl
    pattern = f"{args.base}/*/*/" if args.base else "gate2/*/*/*/*/"
    seen = {}
    for sub in glob.glob(pattern.rstrip("/") + "/"):
        parts = sub.rstrip("/").split("/")
        if args.base:
            cell, method, leaf = args.base, parts[-2], parts[-1]
        else:
            if len(parts) < 5:
                continue
            cell, method, leaf = "/".join(parts[:3]), parts[3], parts[4]
        t = task_of(leaf)
        if t is None or "_attic" in sub or "buggy" in sub:
            continue
        n = count_lines(glob.glob(os.path.join(sub, "run_*", "completions_shard*.jsonl")))
        if n < args.min_n:
            continue
        seen.setdefault((cell, method, t), 0)
        seen[(cell, method, t)] += n

    print(f"{'cell':44s} {'method':26s} {'task':14s} {'n':>6}  annotation")
    for (cell, method, t), n in sorted(seen.items()):
        R, std, mult, ann = annotate(n, t)
        flag = "  <-- CHECK" if ("PARTIAL" in ann or "ragged" in ann) else ""
        print(f"{cell:44s} {method:26s} {t:14s} {n:6d}  {ann}{flag}")


if __name__ == "__main__":
    main()
