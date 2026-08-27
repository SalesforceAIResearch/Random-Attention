#!/usr/bin/env python3
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
"""Paired significance test for LCB cells (problem-clustered bootstrap + exact sign test).
Reuses grader_lcb.grade_records so grading semantics are identical to the reported tables.
Usage: lcb_paired.py --base <dir> --data_name livecodebench_hard --tests_dir ... --problems_jsonl ...
       --method_a random_pp --methods_b vase_faithful[,...] [--n_boot 2000]
"""
import argparse, collections, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kvcompress.eval import grader_lcb as G


def per_problem(records):
    acc = collections.defaultdict(list)
    for gi, flag, _term in records:
        acc[gi].append(1.0 if flag else 0.0)
    return {gi: sum(v) / len(v) for gi, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data_name", default="livecodebench_hard")
    ap.add_argument("--tests_dir", required=True)
    ap.add_argument("--problems_jsonl", required=True)
    ap.add_argument("--method_a", default="random_pp")
    ap.add_argument("--methods_b", required=True)
    ap.add_argument("--max_tokens", type=int, default=32768)
    ap.add_argument("--timeout", type=int, default=20)
    ap.add_argument("--mem_mb", type=int, default=4096)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    store = G.ProblemStore(a.problems_jsonl, a.tests_dir)
    cache = {}
    py = sys.executable

    def grade(method):
        recs, n_runs = G.grade_records(os.path.join(a.base, method), store, a.data_name,
                                       a.max_tokens, a.timeout, a.mem_mb, a.workers, py, cache)
        return per_problem(recs), n_runs

    pa, ra = grade(a.method_a)
    print(f"# lcb-paired {a.base} data={a.data_name} A={a.method_a} (clustered bootstrap x{a.n_boot} + sign test)")
    print(f"{'B (vs A)':26} {'nprob':>5} {'acc_A':>7} {'acc_B':>7} {'delta':>8} {'95% CI':>19} {'p_boot':>7} {'sign(+/=/-)':>13}")
    for mb in a.methods_b.split(","):
        pb, rb = grade(mb)
        gis = sorted(set(pa) & set(pb))
        da = [pa[g] for g in gis]; db = [pb[g] for g in gis]
        delta = sum(x - y for x, y in zip(da, db)) / len(gis)
        rng = random.Random(a.seed)
        boots = []
        for _ in range(a.n_boot):
            idx = [rng.randrange(len(gis)) for _ in gis]
            boots.append(sum(da[i] - db[i] for i in idx) / len(gis))
        boots.sort()
        lo, hi = boots[int(0.025 * a.n_boot)], boots[int(0.975 * a.n_boot)]
        p = 2 * min(sum(b <= 0 for b in boots), sum(b >= 0 for b in boots)) / a.n_boot
        wins = sum(x > y for x, y in zip(da, db)); loss = sum(x < y for x, y in zip(da, db))
        ties = len(gis) - wins - loss
        print(f"{mb:26} {len(gis):>5} {sum(da)/len(gis):>7.4f} {sum(db)/len(gis):>7.4f} "
              f"{delta:>+8.4f} [{lo:>+7.4f},{hi:>+7.4f}] {min(p,1):>7.4f} {wins:>4}/={ties}/-{loss}")


if __name__ == "__main__":
    main()
