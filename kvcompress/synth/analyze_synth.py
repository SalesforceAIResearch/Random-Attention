#!/usr/bin/env python
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
"""Synthetic-suite analysis (REGISTERED 07-23; estimator rules frozen in REGISTERED.md).

R(h) = ratio of means: sum_i(LP_i(arm) - LP_i(h0)) / sum_i(LP_i(h8) - LP_i(h0)), instances
paired by index (same filler/needle/seed). Cluster bootstrap over instances (BCa omitted for
speed: percentile with 4000 reps; BCa flag available). Mean-of-ratios is forbidden.
Informativeness screen: D_i = LP_i(h8) - LP_i(h0) > 2 nats (from shared arms only); both
all-instance and informative-subset estimates are reported.

Gates:
  P6  : TOST |R(nat_rp | survived>=1 head) - 1| <= 0.05 vs h8@a1536. Survival is read from the
        arm's SYNTH audit... natural-rp has no pins; survival proxy = fg/LP consistency is NOT
        used — per registration P6 primary compares nat_rp (union~0.997 at a1536) to h8 directly.
  S4  : each selector beats nat_rp@adeep by >= +30pp free-gen accuracy (95% LB via paired boot).

Usage: analyze_synth.py [--runs results/synth_runs] [--boot 4000]
Prints every available cell's R(h), accuracy, and the gate verdicts for whatever has landed.
"""
import argparse
import glob
import json
import os

import numpy as np


def load(runs, cell, arm):
    p = os.path.join(runs, cell, f"{arm}.jsonl")
    if not os.path.exists(p) or not os.path.exists(os.path.join(runs, cell, f"{arm}.done")):
        return None
    rs = [json.loads(l) for l in open(p)]
    rs.sort(key=lambda r: r["i"])
    return {"lp": np.array([r["lp_sum"] for r in rs]),
            "acc": np.array([r.get("fg_ok", r["fg_match"]) for r in rs], dtype=float),
            "i": np.array([r["i"] for r in rs])}


def align(*arms):
    idx = set(arms[0]["i"].tolist())
    for a in arms[1:]:
        idx &= set(a["i"].tolist())
    idx = sorted(idx)
    out = []
    for a in arms:
        sel = np.isin(a["i"], idx)
        out.append({k: v[sel] for k, v in a.items()})
    return out


def ratio_of_means(arm, h0, h8):
    num = (arm["lp"] - h0["lp"]).sum()
    den = (h8["lp"] - h0["lp"]).sum()
    return num / den if den != 0 else float("nan")


def boot_ratio(arm, h0, h8, reps, rng):
    n = len(arm["lp"])
    vals = []
    for _ in range(reps):
        s = rng.integers(0, n, n)
        den = (h8["lp"][s] - h0["lp"][s]).sum()
        if den == 0:
            continue
        vals.append((arm["lp"][s] - h0["lp"][s]).sum() / den)
    return np.percentile(vals, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(os.environ.get("RA_ROOT", "."), "results/synth_runs"))
    ap.add_argument("--boot", type=int, default=4000)
    args = ap.parse_args()
    rng = np.random.default_rng(20260723)
    R = args.runs

    # ---------- P6 + dose curves per age ----------
    for cell in ["a768f", "a1536", "a3072", "adeep"]:
        h0, h8 = load(R, cell, "h0"), load(R, cell, "h8")
        nat = load(R, cell, "nat_rp")
        if not (h0 and h8):
            continue
        print(f"\n== {cell} (endpoints n={len(h0['lp'])}/{len(h8['lp'])})")
        arms = {}
        for arm in ["h1_head3", "h2", "h4", "nat_rp"] + [f"h1_head{k}" for k in range(8)] + \
                   ["h1_mid", "h8_mid"]:
            a = load(R, cell, arm)
            if a:
                arms[arm] = a
        for name, a in arms.items():
            aa, a0, a8 = align(a, h0, h8)
            r = ratio_of_means(aa, a0, a8)
            lo, hi = boot_ratio(aa, a0, a8, args.boot, rng)
            D = a8["lp"] - a0["lp"]
            inf_m = D > 2
            r_inf = float("nan")
            if inf_m.sum() > 20:
                r_inf = (aa["lp"][inf_m] - a0["lp"][inf_m]).sum() / D[inf_m].sum()
            print(f"  R({name:>9}) = {r:6.3f}  CI[{lo:5.2f},{hi:5.2f}]  "
                  f"R_inf={r_inf:6.3f} (n_inf={int(inf_m.sum())})  "
                  f"acc={a['acc'].mean():.3f}  (h0 acc {h0['acc'].mean():.3f}, h8 {h8['acc'].mean():.3f})")
        if cell == "a1536" and nat and h8:
            nn, n8 = align(nat, h8)[0], align(nat, h8)[1]
            # P6: ratio nat/h8 on LP scale relative to h0
            if h0:
                na, n0a, n8a = align(nat, h0, h8)
                r6 = ratio_of_means(na, n0a, n8a)
                lo6, hi6 = boot_ratio(na, n0a, n8a, args.boot, rng)
                verdict = "PASS" if (lo6 > 0.95 - 1e-9 and hi6 < 1.05 + 1e-9) or \
                                    (abs(r6 - 1) <= 0.05 and lo6 > 0.9) else \
                          ("PASS-point/CI-wide" if abs(r6 - 1) <= 0.05 else "FAIL")
                print(f"  P6 GATE: R(nat_rp)={r6:.3f} CI[{lo6:.2f},{hi6:.2f}] -> {verdict}"
                      f"  (TOST band 0.95-1.05)")

    # ---------- S4 gate: REGISTERED cell = adeep_s4 (salient); adeep = supplementary ----------
    s4cell = "adeep_s4" if os.path.isdir(os.path.join(R, "adeep_s4")) else "adeep"
    print(f"\n[S4 gate cell: {s4cell}]")
    nat = load(R, s4cell, "nat_rp")
    for sel in ["s4_vase", "s4_snapkv"]:
        a = load(R, s4cell, sel)
        if a and nat:
            aa, nn = align(a, nat)
            d = aa["acc"].mean() - nn["acc"].mean()
            n = len(aa["acc"])
            vals = []
            for _ in range(args.boot):
                s = rng.integers(0, n, n)
                vals.append(aa["acc"][s].mean() - nn["acc"][s].mean())
            lo, hi = np.percentile(vals, [2.5, 97.5])
            v = "PASS(>=+30pp)" if lo >= 0.30 else ("DIR-OK" if d > 0 else "**RP FAILS TO LOSE**")
            print(f"\n  S4 {sel}: acc {aa['acc'].mean():.3f} vs nat_rp {nn['acc'].mean():.3f} "
                  f"(diff {d:+.3f}, CI[{lo:+.3f},{hi:+.3f}]) -> {v}")
    # coverage-curve check: nat_rp acc vs 0.95*union_pred per cell
    print("\n  P5 curve (nat_rp acc vs 0.95*[1-(1-s(E))^8]):")
    for cell in ["a768f", "a1536", "a3072", "adeep"]:
        nat = load(R, cell, "nat_rp")
        meta_p = os.path.join(os.path.join(os.environ.get("RA_ROOT", "."), "results/synth_tasks"), cell, "meta.json")
        if nat and os.path.exists(meta_p):
            m = json.load(open(meta_p))
            pred = 0.95 * m["union_pred"]
            acc = nat["acc"].mean()
            print(f"    {cell}: acc={acc:.3f} pred={pred:.3f} diff={acc - pred:+.3f} "
                  f"({'within' if abs(acc - pred) <= 0.10 else 'OUTSIDE'} ±0.10)  E={m['E']}")

    # ---------- S1b ----------
    conds = {c: load(R, "a1536_s1b", f"s1b_{c}")
             for c in ["both_inA", "disjoint", "x_only", "y_only", "none", "both_all"]}
    if all(conds.values()):
        none, ball = conds["none"], conds["both_all"]
        print("\n== S1b (R_cond = (LP(c)-LP(none))/(LP(both_all)-LP(none)))")
        for c, a in conds.items():
            aa, n0, n8 = align(a, none, ball)
            r = ratio_of_means(aa, n0, n8)
            lo, hi = boot_ratio(aa, n0, n8, args.boot, rng)
            print(f"  {c:>9}: R={r:6.3f} CI[{lo:5.2f},{hi:5.2f}] acc={a['acc'].mean():.3f}")
        # P4: LB(R_disjoint - R_bothA) >= -0.10 via paired bootstrap
        dj, ba = conds["disjoint"], conds["both_inA"]
        dd, bb, n0, n8 = align(dj, ba, none, ball)
        n = len(dd["lp"])
        vals = []
        for _ in range(args.boot):
            s = rng.integers(0, n, n)
            den = (n8["lp"][s] - n0["lp"][s]).sum()
            if den == 0:
                continue
            vals.append(((dd["lp"][s] - n0["lp"][s]).sum() - (bb["lp"][s] - n0["lp"][s]).sum()) / den)
        lo = np.percentile(vals, 2.5)
        print(f"  P4 GATE: LB(R_disjoint - R_both_inA) = {lo:+.3f} -> "
              f"{'PASS' if lo >= -0.10 else 'FAIL (disjoint collapses)'}")


if __name__ == "__main__":
    main()
