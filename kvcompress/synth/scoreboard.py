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
"""GPU-free full scoreboard for the synthetic mechanism suite (§5): every saved arm's
R (ratio-of-means, cluster bootstrap CI) AND fg_ok accuracy (the registered secondary,
REGISTERED.md 07-23 amendment) with CI, from results/synth_runs JSONLs alone.

Also prints heterogeneity checks on the headline arms — R by instance half (disjoint
values + fillers) and by needle-value quartile — and the instance-diversity audit
(distinct values / variable names), for the "does this depend on the planted value or
trace?" reviewer question.

NOTE: do NOT add a "value appears in free tail" metric — the free tail begins after the
forced value tokens, so the model can copy the answer from the recency window (the
documented pre-fg_ok artifact; h0 scores ~0.22 on it). fg_ok (greedy-argmax at the value
steps) is the only valid accuracy readout.

Usage: scoreboard.py [--runs results/synth_runs] [--tasks results/synth_tasks] [--boot 4000]
"""
import argparse
import glob
import json
import os

import numpy as np


def load(runs, cell, arm):
    p = f"{runs}/{cell}/{arm}.jsonl"
    if not (os.path.exists(p) and os.path.exists(f"{runs}/{cell}/{arm}.done")):
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


def R_ci(arm, h0, h8, boot, rng, mask=None):
    a, z, e = align(arm, h0, h8)
    if mask is not None:
        keep = mask[a["i"]]
        a, z, e = [{k: v[keep] for k, v in d.items()} for d in (a, z, e)]
    den = (e["lp"] - z["lp"]).sum()
    r = (a["lp"] - z["lp"]).sum() / den if den != 0 else float("nan")
    n = len(a["lp"])
    vals = []
    for _ in range(boot):
        s = rng.integers(0, n, n)
        d = (e["lp"][s] - z["lp"][s]).sum()
        if d != 0:
            vals.append((a["lp"][s] - z["lp"][s]).sum() / d)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return r, lo, hi, n


def acc_ci(a, boot, rng):
    n = len(a["acc"])
    vals = [a["acc"][rng.integers(0, n, n)].mean() for _ in range(boot)]
    return a["acc"].mean(), np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(os.environ.get("RA_ROOT", "."), "results/synth_runs"))
    ap.add_argument("--tasks", default=os.path.join(os.environ.get("RA_ROOT", "."), "results/synth_tasks"))
    ap.add_argument("--boot", type=int, default=4000)
    args = ap.parse_args()
    rng = np.random.default_rng(20260728)
    R = args.runs

    CELLS = {"a768f": ("h0", "h8"), "a1536": ("h0", "h8"), "a3072": ("h0", "h8"),
             "a1536_fragv": ("fragv_h0", "fragv_h8"),
             "a1536_s1b": ("s1b_none", "s1b_both_all"),
             "a1536_s2r": ("s2r_none", "s2r_both_all"),
             "a1536_s2s": ("s2s_none", "s2s_both_all")}
    WAKE = {"a1536_wakeR": ("a1536", "h0", "h8"), "a1536_wakeD": ("a1536", "h0", "h8")}

    print(f"{'cell':<14} {'arm':<18} {'R':>7} {'R CI':>15} {'acc':>6} {'acc CI':>15} {'n':>4}")
    for cell, (e0, e8) in CELLS.items():
        h0, h8 = load(R, cell, e0), load(R, cell, e8)
        if not (h0 and h8):
            continue
        for arm in sorted(os.path.basename(p)[:-6] for p in glob.glob(f"{R}/{cell}/*.jsonl")):
            if "parity" in arm:
                continue
            a = load(R, cell, arm)
            if a is None:
                continue
            r, lo, hi, n = R_ci(a, h0, h8, args.boot, rng)
            m, alo, ahi = acc_ci(a, args.boot, rng)
            print(f"{cell:<14} {arm:<18} {r:7.3f} [{lo:6.3f},{hi:6.3f}] "
                  f"{m:6.3f} [{alo:5.3f},{ahi:5.3f}] {n:4d}")
    for cell, (ref, e0, e8) in WAKE.items():
        h0, h8 = load(R, ref, e0), load(R, ref, e8)
        for arm in sorted(os.path.basename(p)[:-6] for p in glob.glob(f"{R}/{cell}/*.jsonl")):
            a = load(R, cell, arm)
            if a is None:
                continue
            r, lo, hi, n = R_ci(a, h0, h8, args.boot, rng)
            m, alo, ahi = acc_ci(a, args.boot, rng)
            print(f"{cell:<14} {arm:<18} {r:7.3f} [{lo:6.3f},{hi:6.3f}] "
                  f"{m:6.3f} [{alo:5.3f},{ahi:5.3f}] {n:4d}  (a1536 endpoints)")
    for cell in ("adeep", "adeep_s4"):
        for arm in sorted(os.path.basename(p)[:-6] for p in glob.glob(f"{R}/{cell}/*.jsonl")):
            a = load(R, cell, arm)
            if a is None:
                continue
            m, alo, ahi = acc_ci(a, args.boot, rng)
            print(f"{cell:<14} {arm:<18} {'--':>7} {'':>15} "
                  f"{m:6.3f} [{alo:5.3f},{ahi:5.3f}] {len(a['acc']):4d}  (acc-only)")

    print("\n== Heterogeneity (a1536 headline arms): R by instance half / value quartile ==")
    meta = json.load(open(f"{args.tasks}/a1536/meta.json"))
    vx = np.array([v["vx"] for v in meta["values"]])
    h0, h8 = load(R, "a1536", "h0"), load(R, "a1536", "h8")
    nmax = len(vx)
    q = np.quantile(vx[:nmax], [0.25, 0.5, 0.75])
    splits = {"first250": np.arange(nmax) < 250, "last250": np.arange(nmax) >= 250,
              "vQ1": vx <= q[0], "vQ2": (vx > q[0]) & (vx <= q[1]),
              "vQ3": (vx > q[1]) & (vx <= q[2]), "vQ4": vx > q[2]}
    for arm_name in ["frag_carriers", "h1_head5", "h1_head3", "nat_rp", "fs8"]:
        a = load(R, "a1536", arm_name)
        if a is None:
            continue
        parts = []
        for lbl, msk in splits.items():
            r, lo, hi, n = R_ci(a, h0, h8, args.boot, rng, mask=msk)
            parts.append(f"{lbl} {r:.3f}[{lo:.2f},{hi:.2f}](n={n})")
        print(f"  {arm_name:<14} " + "  ".join(parts))

    print("\n== Instance diversity audit (a1536) ==")
    print(f"  distinct v_x values: {len(set(vx.tolist()))}/{len(vx)}; "
          f"range [{vx.min()},{vx.max()}]")
    print(f"  distinct variable names: {sorted(set(v['var'] for v in meta['values']))}")


if __name__ == "__main__":
    main()
