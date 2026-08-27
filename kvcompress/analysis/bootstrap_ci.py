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
"""One-off audit: per-problem accuracy + clustered bootstrap 95% CI for AIME.
Reuses grade_ours.grade_records so grading is identical to the report numbers.
Bootstrap resamples the 30 PROBLEMS (the real unit of variance), not completions."""
import sys, math, random
from collections import defaultdict
from kvcompress.eval import grader as G

random.seed(0)  # deterministic; argless Date/random not needed

BASE = "./gate2/Qwen3-4B/aime_K4096"
CELLS = [
    ("aime25", "range_sink_sample_attn", "vase"),
    ("aime25", "random_pp",              "random_pp"),
    ("aime26", "range_sink_sample_attn", "vase"),
    ("aime26", "random_pp",              "random_pp"),
]
PAPER = {  # (full, vase, snapkv)
    "aime25": (0.6641, 0.5917, 0.4792),
    "aime26": (0.6271, 0.6208, 0.5333),
}

def load(data_name):
    examples = G.load_data(data_name, "test", "./data")
    golds = [G.parse_ground_truth(d, data_name)[1] for d in examples]
    return golds

def boot_ci(per_prob, n_boot=10000):
    """per_prob: dict gi -> list of 0/1 flags. Bootstrap over problem ids."""
    ids = list(per_prob.keys())
    means = []
    for _ in range(n_boot):
        samp = [random.choice(ids) for _ in ids]            # resample problems w/ replacement
        flat = [v for gi in samp for v in per_prob[gi]]
        means.append(sum(flat) / len(flat))
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return lo, hi

print(f"{'cell':28s} {'n_compl':>7} {'n_prob':>6} {'R~':>4} {'flag_acc':>8} {'95% CI (clustered/problem)':>28}")
results = {}
for data_name, method, label in CELLS:
    golds = load(data_name)
    md = f"{BASE}/{method}"
    recs, n_runs = G.grade_records(md, golds, data_name, 32768)  # recs: (gi, flag, term)
    per_prob = defaultdict(list)
    for gi, flag, term in recs:
        per_prob[gi].append(int(flag))
    n_compl = len(recs)
    n_prob = len(per_prob)
    flag_acc = sum(f for _, f, _ in recs) / n_compl if n_compl else float("nan")
    lo, hi = boot_ci(per_prob)
    results[(data_name, label)] = (flag_acc, lo, hi, n_compl, n_prob)
    print(f"{data_name+'/'+label:28s} {n_compl:>7} {n_prob:>6} {n_compl/max(n_prob,1):>4.1f} "
          f"{flag_acc:>8.4f}   [{lo:.4f}, {hi:.4f}]")

print("\n=== anchors & overlap checks ===")
for data_name in ("aime25", "aime26"):
    full, pvase, psnap = PAPER[data_name]
    vacc, vlo, vhi, *_ = results[(data_name, "vase")]
    racc, rlo, rhi, *_ = results[(data_name, "random_pp")]
    print(f"\n[{data_name}]  paper full={full:.3f}  paper vase={pvase:.3f}")
    print(f"  vase(ours)     = {vacc:.4f}  CI[{vlo:.3f},{vhi:.3f}]  "
          f"| paper-vase {pvase:.3f} in CI? {vlo<=pvase<=vhi}  | full {full:.3f} in CI? {vlo<=full<=vhi}  | vase>full? {vacc>full}")
    print(f"  random_pp(ours)= {racc:.4f}  CI[{rlo:.3f},{rhi:.3f}]  "
          f"| vase(ours) {vacc:.3f} in random_pp CI? {rlo<=vacc<=rhi}")
    print(f"  one problem = {1/results[(data_name,'vase')][4]*results[(data_name,'vase')][3]:.0f} compl; "
          f"vase-vs-full gap = {abs(vacc-full)*100:.1f}pp = {abs(vacc-full)*results[(data_name,'vase')][4]:.1f} problems-worth")
