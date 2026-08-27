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
"""Emit the ORGANIZED results section from the fresh regrade TSV.

Replaces the ad-hoc scoreboard that grew row-by-row through the program. Structure:
  1. MAIN TABLE      — model x task at the main budget, all methods
  2. COMPRESSION SWEEP — model x task x budget
  3. R-ANNOTATION     — every cell's run count vs the task standard, extras made explicit
Metric is flag_acc throughout (acc_strict rejected on evidence: it penalises non-termination and
tri terminates more often than rp in 10/14 cells). Cells whose R exceeds the standard are printed
as "value (R8 =2x)" so an extended cell is never silently compared against a standard one; where a
standard-R subset is also available it is shown in brackets as the comparable value.

Usage: python organize_results.py [--tsv regrade32k_results_fresh.tsv] [--md]
"""
import argparse
import collections
import os

# 08-16 UNIFIED CONVENTION (Heng): 4x = K4096 for every model — the 07-27 phi trace-relative
# relabel is retired (its "~2x-longer traces" premise is refuted by measured generate_lens:
# phi AIME median 9.2k tok vs Qwen3-4B 15.2k). phi K8192 = 2x sweep point.
MAIN_BUDGET_OVERRIDE = {}

MAIN_BUDGET = {  # task -> the main-grid budget (~4x)
    "math": 1024, "gpqa": 2048, "aime": 4096, "aime25": 4096, "aime26": 4096,
    "hmmt": 4096, "livecodebench": 3072,
}
STANDARD_R = {"math": 2, "gpqa": 4, "aime": 16, "aime25": 16, "aime26": 16, "hmmt": 16, "livecodebench": 4}
METHOD_LABEL = [
    ("dense", "dense"), ("attn", "SnapKV"), ("attn_rkv_l05", "R-KV"),
    ("vase_faithful", "VaSE"), ("range_sink_sample_attn", "VaSE(legacy)"),
    ("triattn_ph_memofix", "TriAttn"), ("random_pp", "rp (ours)"), ("rp_r512", "rp (ours)"),
]
MODEL_ORDER = ["Qwen3-4B", "Qwen3-14B", "Qwen3-32B", "phi-4-reasoning"]
TASK_ORDER = ["math", "gpqa", "aime", "hmmt", "livecodebench"]   # aime = pooled aime25+26 (n-weighted), per the one-cell rule


def parse(tsv):
    rows = []
    with open(tsv) as fh:
        head = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 8 or f[0] == "base":
                continue
            d = dict(zip(head, f))
            parts = d["base"].split("/")
            if len(parts) < 3:
                continue
            d["model"], cell = parts[1], parts[2]
            d["budget"] = int(cell.split("_K")[-1]) if "_K" in cell else None
            try:
                d["flag"] = float(d["flag_acc"]); d["n"] = int(d["n"]); d["R"] = int(d["runs"])
            except ValueError:
                continue
            # n-floor: fragments (a stalled fill, an abandoned probe) otherwise render as real cells.
            # e.g. a 40-line 14B aime arm scored 0.9500 and would have sat in the main table.
            std = STANDARD_R.get(d["data"], 0)
            if std and d["R"] < max(1, std // 2):
                d["excluded"] = f"R{d['R']} < half of standard R{std} (n={d['n']}) — fragment"
                rows.append(d)
                continue
            d["excluded"] = None
            rows.append(d)
    return rows


# a column may be satisfied by several on-disk dir names (legacy layouts, residual-length variants,
# and the provenance-clean subsets built for corrupted cells). First match wins.
ALIASES = {
    "vase_faithful": ["vase_faithful", "range_sink_sample_attn"],
    "random_pp": ["random_pp", "rp_r512", "random_pp_sameclass0808"],
    "triattn_ph_memofix": ["triattn_ph_memofix", "triattn_ph_memofix_sameclass0808"],
    "attn_rkv_l05": ["attn_rkv_l05", "attn_rkv"],
}


def pick(cells, col):
    for name in ALIASES.get(col, [col]):
        if name in cells:
            return cells[name]
    return None


def parse_lcb(path):
    """LCB is graded by a separate code-execution pipeline into its own TSV."""
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        f = line.split("\t")
        if len(f) < 4 or f[0] == "model":
            continue
        try:
            n = int(f[3].split()[0])
        except ValueError:
            continue
        out.append({"model": f[0], "data": "livecodebench", "budget": 3072,
                    "method": f[1].strip(), "flag": float(f[2]), "n": n,
                    "R": max(1, round(n / 383)), "excluded": None})
    return out


def ann(task, R):
    std = STANDARD_R.get(task, 0)
    if not std or R == std:
        return ""
    if R > std and R % std == 0:
        return f" (R{R}={R // std}×)"
    return f" (R{R} vs std R{std})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default=os.path.join(os.environ.get("RA_ROOT", "."), "regrade32k_results_fresh.tsv"))
    ap.add_argument("--lcb", default=os.path.join(os.environ.get("RA_ROOT", "."), "figures/results_lcb.tsv"))
    args = ap.parse_args()
    rows = parse(args.tsv)
    rows += parse_lcb(args.lcb)
    by = collections.defaultdict(dict)
    for r in rows:
        if r.get("excluded"):
            continue
        by[(r["model"], r["data"], r["budget"])][r["method"]] = r

    # AIME one-cell rule (Heng): pool aime25+aime26 n-weighted into a single 'aime' task.
    # A method missing one year is annotated rather than silently half-pooled.
    for (model, task, budget) in [k for k in by if k[1] == "aime25"]:
        c25 = by[(model, "aime25", budget)]
        c26 = by.get((model, "aime26", budget), {})
        pooled = {}
        for m in set(c25) | set(c26):
            a, b_ = c25.get(m), c26.get(m)
            if a and b_:
                n = a["n"] + b_["n"]
                pooled[m] = dict(a, flag=(a["flag"] * a["n"] + b_["flag"] * b_["n"]) / n,
                                 n=n, R=min(a["R"], b_["R"]))
            else:
                src = a or b_
                pooled[m] = dict(src)
                pooled[m]["only"] = "aime25" if a else "aime26"
        by[(model, "aime", budget)] = pooled
    for k in [k for k in by if k[1] in ("aime25", "aime26")]:
        del by[k]

    labels = dict(METHOD_LABEL)
    order = [m for m, _ in METHOD_LABEL]

    print("## 1. MAIN TABLE — main budget per task, flag_acc (R annotated where non-standard)\n")
    cols = ["dense", "attn", "attn_rkv_l05", "vase_faithful", "triattn_ph_memofix", "random_pp"]
    print("| model | task | " + " | ".join(labels[c] for c in cols) + " |")
    print("|" + "---|" * (len(cols) + 2))
    for model in MODEL_ORDER:
        for task in TASK_ORDER:
            key = (model, task, MAIN_BUDGET_OVERRIDE.get((model, task), MAIN_BUDGET.get(task)))
            if key not in by:
                continue
            cells = by[key]
            out = []
            for c in cols:
                r = pick(cells, c)
                if not r and c == "dense":
                    # dense = no eviction; its K-dir is storage location, not condition —
                    # fall back to any budget's dense arm for this model/task (annotated)
                    for (m2, t2, b2), cc in sorted(by.items()):
                        if m2 == model and t2 == task and "dense" in cc:
                            r = dict(cc["dense"]); r["densefrom"] = b2
                            break
                if r and r.get("densefrom"):
                    out.append(f"{r['flag']:.4f} (dense@K{r['densefrom']} dir{ann(task, r['R'])})")
                elif r and r.get("only"):
                    out.append(f"{r['flag']:.4f} ({r['only']} only, R{r['R']})")
                elif r:
                    out.append(f"{r['flag']:.4f}{ann(task, r['R'])}")
                else:
                    out.append("—")
            print(f"| {model} | {task} | " + " | ".join(out) + " |")

    print("\n## 2. COMPRESSION SWEEP — all budgets\n")
    print("| model | task | budget | " + " | ".join(labels[c] for c in cols[1:]) + " |")
    print("|" + "---|" * (len(cols) + 2))
    for model in MODEL_ORDER:
        for task in TASK_ORDER:
            budgets = sorted({k[2] for k in by if k[0] == model and k[1] == task and k[2]})
            for b in budgets:
                cells = by[(model, task, b)]
                out = []
                for c in cols[1:]:
                    r = pick(cells, c)
                    if r and r.get("only"):
                        out.append(f"{r['flag']:.4f} ({r['only']} only, R{r['R']})")
                    elif r:
                        out.append(f"{r['flag']:.4f}{ann(task, r['R'])}")
                    else:
                        out.append("—")
                if all(o == "—" for o in out):
                    continue
                print(f"| {model} | {task} | K={b} | " + " | ".join(out) + " |")

    dropped = [r for r in rows if r.get("excluded")]
    if dropped:
        print("\n## 2b. EXCLUDED FRAGMENTS (below the n-floor; never table-eligible)\n")
        print("| cell | task | method | n | reason |")
        print("|---|---|---|---|---|")
        for r in sorted(dropped, key=lambda x: (x["model"], x["data"])):
            print(f"| {r['model']}/{r['data']}_K{r['budget']} | {r['data']} | {r['method']} "
                  f"| {r['n']} | {r['excluded']} |")

    print("\n## 3. NON-STANDARD R (extended or partial cells — extras made explicit)\n")
    print("| cell | task | method | n | R | vs standard |")
    print("|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda x: (x["model"], x["data"], x["budget"] or 0)):
        std = STANDARD_R.get(r["data"], 0)
        if std and r["R"] != std:
            rel = f"{r['R'] / std:.2f}× standard R{std}"
            print(f"| {r['model']}/{r['data']}_K{r['budget']} | {r['data']} | {r['method']} "
                  f"| {r['n']} | R{r['R']} | {rel} |")


if __name__ == "__main__":
    main()
