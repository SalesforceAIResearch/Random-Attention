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
"""S3 round-1 grader — implements the FROZEN estimators of S3_ACCURACY_PLAN.md v2.1
(REGISTERED.md 07-31 block) exactly. GPU-free; reads results/synth_runs JSONLs.

Rep-pooling: per-instance accuracy = mean over replicates; per-instance lp = mean over
replicates (used for rep-pooled R and paired dR). LB/UB = one-sided 95% bootstrap quantile
(5th/95th pct) over instances, 4000 reps, seed 20260731.

Outputs the gate verdicts, precondition checks, S3-P1..P7 adjudications, the per-fact table
for all g_*/g2_* arms, and the canonicality drift table vs the 07-29 s2r originals.
"""
import argparse
import json
import os

import numpy as np

O = os.path.join(os.environ.get("RA_ROOT", "."), "results/synth_runs")
T = os.path.join(os.environ.get("RA_ROOT", "."), "results/synth_tasks")
CAL_A, CAL_B = 11.577, -9.026


def load(cell, arm, need_gok=False):
    p = f"{O}/{cell}/{arm}.jsonl"
    if not (os.path.exists(p) and os.path.exists(f"{O}/{cell}/{arm}.done")):
        return None
    rs = sorted((json.loads(l) for l in open(p)), key=lambda r: r["i"])
    d = {"i": np.array([r["i"] for r in rs]),
         "lp": np.array([r["lp_sum"] for r in rs]),
         "acc": np.array([r["fg_ok"] for r in rs], dtype=float)}
    if need_gok:
        d["gok"] = np.array([r["gok"] for r in rs], dtype=bool)
    return d


def pool(cell, arm, reps=("_r0", "_r1"), need_gok=False, strict=True):
    """Rep-pooled per-instance arrays; also returns per-rep dicts for diagnostics.
    strict: assert every expected replicate is present (round-2 hardening) rather than
    silently degrading to fewer reps."""
    ds = [load(cell, arm + r, need_gok) for r in reps]
    if strict and any(d is None for d in ds):
        missing = [r for r, d in zip(reps, ds) if d is None]
        raise AssertionError(f"{cell}/{arm}: missing replicates {missing}")
    ds = [d for d in ds if d is not None]
    if not ds:
        return None, []
    idx = sorted(set.intersection(*(set(d["i"].tolist()) for d in ds)))
    out = {}
    for k in ["lp", "acc"]:
        out[k] = np.mean([d[k][np.isin(d["i"], idx)] for d in ds], axis=0)
    out["i"] = np.array(idx)
    if need_gok:
        out["gok_reps"] = [d["gok"][np.isin(d["i"], idx)] for d in ds]
    return out, ds


def boot_q(vals_fn, n, q, reps=4000, rng=None):
    bs = [vals_fn(rng.integers(0, n, n)) for _ in range(reps)]
    return float(np.percentile(bs, q))


def R_of(arm, h0, h8):
    idx = sorted(set(arm["i"]) & set(h0["i"]) & set(h8["i"]))
    a = arm["lp"][np.isin(arm["i"], idx)]
    z = h0["lp"][np.isin(h0["i"], idx)]
    e = h8["lp"][np.isin(h8["i"], idx)]
    return float((a - z).sum() / (e - z).sum())


def perfact(d, nv=10):
    accx = d[:, 0:4].all(axis=1).astype(float)
    accy = d[:, nv - 4:nv].all(axis=1).astype(float)
    return accx, accy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=4000)
    args = ap.parse_args()
    rng = np.random.default_rng(20260731)
    V = {}

    # ---- gates ------------------------------------------------------------------
    g0 = json.load(open(os.path.join(os.environ.get("RA_ROOT", "."), "results/s3_queue/GATE_G0.json")))
    V["G-S3-0"] = "PASS" if g0["bitwise_equal"] and g0["parity_ok"] == "64/64" else "FAIL (round VOID)"
    print(f"G-S3-0: {V['G-S3-0']}  {json.dumps(g0)}")

    r0 = load("a1536", "s3_intact2_25_r0")
    m = np.isin(r0["i"], np.arange(400))
    a400 = r0["acc"][m].mean()
    in_band = abs(a400 - 0.627) <= 0.08
    V["G-S3-R"] = "PASS" if in_band else "check r1 (frozen procedure)"
    print(f"G-S3-R: acc(s3_intact2_25_r0, inst<400) = {a400:.3f} vs 0.627 ±0.08 -> {V['G-S3-R']}")

    # ---- rep-pooled arms ----------------------------------------------------------
    A = {}
    for cell, arm in [("a1536", "s3_intact2_25"), ("a1536", "s3_intact2_35"),
                      ("a1536", "s3_intact2_23"), ("a1536", "s3_frag2_rot"),
                      ("a1536", "s3_frag2_fs4"), ("a1536_fragv", "s3_intact2v"),
                      ("a1536_fragv", "s3_v_car2"), ("a1536_fragv", "s3_noval2"),
                      ("a1536_fragv", "s3_scaf2")]:
        A[arm], _ = pool(cell, arm)
        if A[arm] is None:
            print(f"  [missing] {cell}/{arm}")
    print("\n== rep-pooled accuracies ==")
    for k, v in A.items():
        if v is not None:
            print(f"  {k:<16} acc={v['acc'].mean():.3f}  n={len(v['i'])}")

    # ---- preconditions -----------------------------------------------------------
    pre_fragv = A["s3_intact2v"] is not None and A["s3_intact2v"]["acc"].mean() >= 0.40
    print(f"\nPrecondition P3/P4 (acc intact2v >= 0.40): "
          f"{A['s3_intact2v']['acc'].mean():.3f} -> {'GRADEABLE' if pre_fragv else 'OUT OF RANGE'}")

    # ---- P1 / P2 -----------------------------------------------------------------
    a25, a35, a23, fr = A["s3_intact2_25"], A["s3_intact2_35"], A["s3_intact2_23"], A["s3_frag2_rot"]
    idx = sorted(set(a25["i"]) & set(a35["i"]) & set(a23["i"]) & set(fr["i"]))
    sel = lambda d: d["acc"][np.isin(d["i"], idx)]
    mp = (12 * sel(a25) + 10 * sel(a35) + 10 * sel(a23)) / 32
    D = sel(fr) - mp
    n = len(D)
    lb = boot_q(lambda s: D[s].mean(), n, 5, args.boot, rng)
    V["S3-P1"] = "PASS" if lb >= -0.05 else ("MISS (NI not shown)" if D.mean() >= -0.05
                                             else "MISS (spread costs retrieval)")
    print(f"\nS3-P1: acc(frag2_rot)={sel(fr).mean():.3f} mean_pair_w={mp.mean():.3f} "
          f"D={D.mean():+.3f} LB95={lb:+.3f} (>= -0.05) -> {V['S3-P1']}")
    V["S3-P2"] = "PASS" if D.mean() > 0 else "MISS (premium logprob-only)"
    print(f"S3-P2: point D={D.mean():+.3f} -> {V['S3-P2']}")

    # ---- P3 / P4 -----------------------------------------------------------------
    iv, vc, nv2, sc = A["s3_intact2v"], A["s3_v_car2"], A["s3_noval2"], A["s3_scaf2"]
    idx3 = sorted(set(iv["i"]) & set(vc["i"]))
    ivs, vcs = iv["acc"][np.isin(iv["i"], idx3)], vc["acc"][np.isin(vc["i"], idx3)]
    ratio = vcs.mean() / ivs.mean()
    rlb = boot_q(lambda s: vcs[s].mean() / max(ivs[s].mean(), 1e-9), len(idx3), 5, args.boot, rng)
    if pre_fragv:
        ok3 = ratio >= 0.8 and rlb >= 0.6
        if ok3:
            V["S3-P3"] = "PASS"
        else:
            h0v, h8v = load("a1536_fragv", "fragv_h0"), load("a1536_fragv", "fragv_h8")
            dR = R_of(vc, h0v, h8v) - R_of(iv, h0v, h8v)
            V["S3-P3"] = (f"MISS-knee (dR={dR:+.3f} >= -0.05: payload still carries info)"
                          if dR >= -0.05 else f"MISS-real (dR={dR:+.3f}: payload insufficient)")
    else:
        V["S3-P3"] = "OUT OF RANGE"
    print(f"S3-P3: ratio={ratio:.3f} (>=0.8) LB={rlb:.3f} (>=0.6) -> {V['S3-P3']}")
    ub4 = boot_q(lambda s: nv2["acc"][s].mean(), len(nv2["acc"]), 95, args.boot, rng)
    V["S3-P4"] = ("PASS" if ub4 <= 0.08 else "MISS (bounds fv_noval)") if pre_fragv else "OUT OF RANGE"
    print(f"S3-P4: acc(noval2)={nv2['acc'].mean():.3f} UB95={ub4:.3f} (<=0.08) -> {V['S3-P4']}  "
          f"[scaf2 descriptive: {sc['acc'].mean():.3f}]")

    # ---- P5 (per-fact, strong groups) -------------------------------------------
    meta = json.load(open(f"{T}/a1536_s2r/meta.json"))
    NV = meta["n_val_tokens"]
    assert NV == 10, NV
    P5 = {}
    for arm in ["g2_split_x", "g2_split_y", "g2_co25"]:
        pooled, _ = pool("a1536_s2r", arm, need_gok=True)
        gx = np.mean([perfact(g, NV)[0] for g in pooled["gok_reps"]], axis=0)
        gy = np.mean([perfact(g, NV)[1] for g in pooled["gok_reps"]], axis=0)
        P5[arm] = {"i": pooled["i"], "accx": gx, "accy": gy}
        print(f"  {arm:<12} acc_x={gx.mean():.3f} acc_y={gy.mean():.3f} n={len(pooled['i'])}")
    co = P5["g2_co25"]
    for clause, arm, key in [("X", "g2_split_x", "accx"), ("Y", "g2_split_y", "accy")]:
        ref = co[key].mean()
        if ref < 0.15:
            V[f"S3-P5-{clause}"] = f"OUT OF RANGE (co25 {key}={ref:.3f} < 0.15)"
            continue
        sp = P5[arm]
        idx5 = sorted(set(sp["i"]) & set(co["i"]))
        d5 = sp[key][np.isin(sp["i"], idx5)] - co[key][np.isin(co["i"], idx5)]
        lb5 = boot_q(lambda s: d5[s].mean(), len(d5), 5, args.boot, rng)
        V[f"S3-P5-{clause}"] = "PASS" if lb5 >= -0.05 else "MISS (split costs retrieval)"
        print(f"S3-P5-{clause}: D={d5.mean():+.3f} LB95={lb5:+.3f} (>=-0.05) -> {V[f'S3-P5-{clause}']}")

    # ---- P6 descriptive -----------------------------------------------------------
    print(f"\nS3-P6 (descriptive): frag2_rot {fr['acc'].mean():.3f} vs frag2_fs4 "
          f"{A['s3_frag2_fs4']['acc'].mean():.3f} (fs4 digit-placement confound declared)")

    # ---- P7 calibration ------------------------------------------------------------
    print("\n== S3-P7 frozen calibration ==")
    h0a, h8a = load("a1536", "h0"), load("a1536", "h8")
    h0v, h8v = load("a1536_fragv", "fragv_h0"), load("a1536_fragv", "fragv_h8")
    devs = []
    for arm, (e0, e8) in [("s3_intact2_25", (h0a, h8a)), ("s3_intact2_35", (h0a, h8a)),
                          ("s3_intact2_23", (h0a, h8a)), ("s3_frag2_rot", (h0a, h8a)),
                          ("s3_frag2_fs4", (h0a, h8a)), ("s3_intact2v", (h0v, h8v)),
                          ("s3_v_car2", (h0v, h8v)), ("s3_noval2", (h0v, h8v)),
                          ("s3_scaf2", (h0v, h8v))]:
        d = A[arm]
        r = R_of(d, e0, e8)
        pred = 1 / (1 + np.exp(-(CAL_A * r + CAL_B)))
        dev = d["acc"].mean() - pred
        devs.append(abs(dev))
        print(f"  {arm:<16} R={r:.3f} pred={pred:.3f} acc={d['acc'].mean():.3f} dev={dev:+.3f}"
              f"{'  <-- >0.10' if abs(dev) > 0.10 else ''}")
    V["S3-P7"] = ("PASS" if max(devs) <= 0.10 and np.mean(devs) <= 0.05
                  else f"MISS (max {max(devs):.3f}, mean {np.mean(devs):.3f})")
    print(f"S3-P7: max|dev|={max(devs):.3f} (<=0.10), mean={np.mean(devs):.3f} (<=0.05) -> {V['S3-P7']}")

    # ---- weak-group descriptive + canonicality drift ------------------------------
    print("\n== weak g_* per-fact table (descriptive) + R drift vs 07-29 originals ==")
    g_h0, _ = pool("a1536_s2r", "g_none", reps=("",), need_gok=True)
    g_h8, _ = pool("a1536_s2r", "g_both_all", reps=("",), need_gok=True)
    for cond in ["both_all", "none", "grp_split", "grp_coA", "grp_coB", "disjoint",
                 "coloc5", "x_only", "y_only"]:
        gp, _ = pool("a1536_s2r", f"g_{cond}", reps=("",), need_gok=True)
        if gp is None:
            continue
        gx, gy = perfact(gp["gok_reps"][0], NV)
        rnew = R_of(gp, g_h0, g_h8)
        old = load("a1536_s2r", f"s2r_{cond}")
        o_h0, o_h8 = load("a1536_s2r", "s2r_none"), load("a1536_s2r", "s2r_both_all")
        rold = R_of(old, o_h0, o_h8) if old else float("nan")
        print(f"  {cond:<10} acc_x={gx.mean():.3f} acc_y={gy.mean():.3f} "
              f"R_new={rnew:.3f} R_old={rold:.3f} drift={rnew - rold:+.3f}")

    print("\n== VERDICT SUMMARY ==")
    for k, v in V.items():
        print(f"  {k:<10} {v}")
    json.dump(V, open(os.path.join(os.environ.get("RA_ROOT", "."), "results/s3_queue/ROUND1_VERDICTS.json"), "w"))


if __name__ == "__main__":
    main()
