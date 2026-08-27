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
"""Paired, problem-clustered significance tests for method-vs-method accuracy deltas.

Completions are positional, so method A and method B are graded on the SAME problems; seeds cluster
within a problem. The right unit of inference is the PROBLEM: we compare per-problem mean accuracy
(over seeds/runs), then (a) a clustered percentile bootstrap over problems for a 95% CI on the
delta, and (b) an exact two-sided sign test on problems where the two methods differ.

flag_acc is the primary metric; --metric strict uses flag*term instead (grade with the CORRECT
--max_tokens: default 32768 = every REPORT run's cap).

Usage:
  python stats_paired.py --base gate2/Qwen3-14B/hmmt_K4096 --data_name hmmt \
      --method_a random_pp --methods_b caote,range_sink_sample_attn,attn,triattn_ph
"""
import argparse
import math
import numpy as np
from collections import defaultdict
from Utils.parser import parse_ground_truth
from Utils.data_loader import load_data
from kvcompress.eval.grader import grade_records


def per_problem(recs, metric):
    by = defaultdict(list)
    for gi, f, t in recs:
        by[gi].append(f * t if metric == "strict" else f)
    return {gi: float(np.mean(v)) for gi, v in by.items()}, \
           {gi: len(v) for gi, v in by.items()}


def sign_test_p(n_pos, n_neg):
    """Exact two-sided sign test (binomial, p=0.5) on the non-tied problems."""
    n = n_pos + n_neg
    if n == 0:
        return 1.0
    k = min(n_pos, n_neg)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2.0 ** n
    return min(1.0, 2.0 * tail)


def graded_per_problem(base_dir, meth, datas, golds_by, metric, max_tokens):
    """Per-problem means pooled over one or more datasets; problems are namespaced
    by dataset so pooled AIME25+AIME26 clusters stay distinct."""
    import os
    per, cnt = {}, {}
    for d in datas:
        recs, _ = grade_records(os.path.join(base_dir, meth), golds_by[d], d, max_tokens)
        p, n = per_problem(recs, metric)
        for gi, v in p.items():
            per[(d, gi)] = v
            cnt[(d, gi)] = n[gi]
    return per, cnt


def compare(base, data_name, a, b, golds_by, metric, max_tokens, n_boot, seed, base_b=None):
    datas = data_name.split(",")
    pa, na = graded_per_problem(base, a, datas, golds_by, metric, max_tokens)
    pb, nb = graded_per_problem(base_b or base, b, datas, golds_by, metric, max_tokens)
    common = sorted(set(pa) & set(pb))
    if not common:
        return None
    xa = np.array([pa[g] for g in common])
    xb = np.array([pb[g] for g in common])
    d = xa - xb
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # bootstrap two-sided p: fraction of resampled deltas on the other side of 0
    p_boot = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    n_pos, n_neg = int((d > 0).sum()), int((d < 0).sum())
    return dict(n_problems=len(common),
                seeds_a=int(np.median([na[g] for g in common])),
                seeds_b=int(np.median([nb[g] for g in common])),
                acc_a=float(xa.mean()), acc_b=float(xb.mean()),
                delta=float(d.mean()), ci_lo=float(lo), ci_hi=float(hi),
                p_boot=float(p_boot),
                sign=f"+{n_pos}/={len(common)-n_pos-n_neg}/-{n_neg}",
                p_sign=sign_test_p(n_pos, n_neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data_name", default="math")
    ap.add_argument("--method_a", default="random_pp")
    ap.add_argument("--methods_b", required=True, help="comma-separated")
    ap.add_argument("--base_b", default=None,
                    help="base dir for the B methods when it differs from --base (positional gi "
                         "alignment holds across dirs — both index the same task golds)")
    ap.add_argument("--metric", choices=["flag", "strict"], default="flag")
    ap.add_argument("--max_tokens", type=int, default=32768)
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    golds_by = {}
    for d in args.data_name.split(","):
        examples = load_data(d, "test", "./data")
        golds_by[d] = [parse_ground_truth(x, d)[1] for x in examples]

    print(f"# paired-{args.metric} {args.base} data={args.data_name} A={args.method_a} "
          f"(problem-clustered bootstrap x{args.n_boot} + exact sign test)")
    hdr = (f'{"B (vs A)":26s} {"nprob":>5} {"R":>4} {"acc_A":>7} {"acc_B":>7} {"delta":>7} '
           f'{"95% CI":>18} {"p_boot":>7} {"sign(+/=/-)":>14} {"p_sign":>7}')
    print(hdr)
    for b in args.methods_b.split(","):
        r = compare(args.base, args.data_name, args.method_a, b, golds_by,
                    args.metric, args.max_tokens, args.n_boot, args.seed, base_b=args.base_b)
        if r is None:
            print(f"{b:26s} (no overlapping problems / missing)")
            continue
        sig = "*" if (r["ci_lo"] > 0 or r["ci_hi"] < 0) else " "
        print(f'{b:26s} {r["n_problems"]:5d} {r["seeds_a"]:>2}/{r["seeds_b"]:<2} '
              f'{r["acc_a"]:7.4f} {r["acc_b"]:7.4f} {r["delta"]:+7.4f} '
              f'[{r["ci_lo"]:+7.4f},{r["ci_hi"]:+7.4f}]{sig} {r["p_boot"]:7.4f} '
              f'{r["sign"]:>14} {r["p_sign"]:7.4f}')


if __name__ == "__main__":
    main()
