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
"""S3 ROUND-2 grader — frozen estimators of S3_ACCURACY_PLAN.md 'S3 ROUND 2 — v2'
(REGISTERED.md 07-31 round-2 block, commit 8c7ece9). Fresh round-2 arms only in every graded
contrast; preconditions first; per-fact gok slices; pinned calibration sigma(10.735 R - 8.638).
"""
import json
import os

import numpy as np

O = os.path.join(os.environ.get("RA_ROOT", "."), "results/synth_runs")
T = os.path.join(os.environ.get("RA_ROOT", "."), "results/synth_tasks")
A2, B2 = 10.735, -8.638
BOOT = 4000
rng = np.random.default_rng(20260731)


def load(cell, arm, gok=False):
    p = f"{O}/{cell}/{arm}.jsonl"
    assert os.path.exists(p) and os.path.exists(f"{O}/{cell}/{arm}.done"), f"missing {cell}/{arm}"
    rs = sorted((json.loads(l) for l in open(p)), key=lambda r: r["i"])
    d = {"i": np.array([r["i"] for r in rs]),
         "lp": np.array([r["lp_sum"] for r in rs]),
         "acc": np.array([r["fg_ok"] for r in rs], dtype=float)}
    if gok:
        d["gok"] = np.array([r["gok"] for r in rs], dtype=bool)
    return d


def pool(cell, arm, reps=("_r0", "_r1"), gok=False):
    ds = [load(cell, arm + r, gok) for r in reps]
    idx = sorted(set.intersection(*(set(d["i"].tolist()) for d in ds)))
    out = {"i": np.array(idx)}
    for k in ("lp", "acc"):
        out[k] = np.mean([d[k][np.isin(d["i"], idx)] for d in ds], axis=0)
    if gok:
        NV = 10
        ax = np.mean([d["gok"][np.isin(d["i"], idx)][:, 0:4].all(1) for d in ds], axis=0)
        ay = np.mean([d["gok"][np.isin(d["i"], idx)][:, NV - 4:NV].all(1) for d in ds], axis=0)
        out["accx"], out["accy"] = ax, ay
    return out


def lb(d, q=5):
    n = len(d)
    return float(np.percentile([d[rng.integers(0, n, n)].mean() for _ in range(BOOT)], q))


def paired(a, b, key):
    idx = sorted(set(a["i"]) & set(b["i"]))
    return a[key][np.isin(a["i"], idx)] - b[key][np.isin(b["i"], idx)]


def main():
    meta = json.load(open(f"{T}/a1536_s2r/meta.json"))
    assert meta["n_val_tokens"] == 10
    V = {}
    s2 = {a: pool("a1536_s2r", a, gok=True) for a in
          ["r2_cap_solo", "r2_co25_b", "r2_split_x_b", "r2_split_y_b", "r2_bothfrag"]}
    print("== fresh s2r per-fact accuracies ==")
    for a, d in s2.items():
        print(f"  {a:<14} acc_x={d['accx'].mean():.3f} acc_y={d['accy'].mean():.3f} n={len(d['i'])}")

    # preconditions
    pre = {"P1": s2["r2_split_x_b"]["accx"].mean() >= 0.15,
           "P1y": s2["r2_split_y_b"]["accy"].mean() >= 0.15,
           "P2": s2["r2_cap_solo"]["accx"].mean() >= 0.15,
           "P6": s2["r2_bothfrag"]["accx"].mean() >= 0.15}
    print("preconditions:", {k: ("ok" if v else "OUT OF RANGE") for k, v in pre.items()})

    def sup(name, a, b, key, bar):
        d = paired(a, b, key)
        l = lb(d)
        return f"D={d.mean():+.3f} LB={l:+.3f} (>= {bar:+.2f})", l >= bar

    if pre["P1"]:
        s, ok = sup("P1", s2["r2_split_x_b"], s2["r2_co25_b"], "accx", 0.10)
        V["R2-P1"] = f"{'PASS' if ok else 'MISS'} ({s})"
    else:
        V["R2-P1"] = "OUT OF RANGE"
    if pre["P1y"]:
        s, ok = sup("P1y", s2["r2_split_y_b"], s2["r2_co25_b"], "accy", 0.10)
        V["R2-P1y"] = f"{'PASS' if ok else 'MISS'} ({s})"
    else:
        V["R2-P1y"] = "OUT OF RANGE"
    if pre["P2"]:
        d1 = paired(s2["r2_cap_solo"], s2["r2_split_x_b"], "accx")
        d2 = paired(s2["r2_split_x_b"], s2["r2_co25_b"], "accx")
        l1, l2 = lb(d1), lb(d2)
        ok = l1 >= -0.03 and l2 >= -0.03
        V["R2-P2"] = (f"{'PASS' if ok else 'MISS'} (solo-split D={d1.mean():+.3f} LB={l1:+.3f}; "
                      f"split-co D={d2.mean():+.3f} LB={l2:+.3f}; bars >= -0.03)")
    else:
        V["R2-P2"] = "OUT OF RANGE"

    # a1536 arms
    i3 = pool("a1536", "r2_intact3")
    fr_b = pool("a1536", "r2_frag2_rot_b")
    i25_b = pool("a1536", "r2_intact2_25_b")
    h8f = pool("a1536", "r2_h8_fresh", reps=("_r0",))
    h0 = load("a1536", "h0")
    V["R2-P3"] = (f"descriptive knee: 0.161 -> 0.386 -> 0.597 -> intact3 {i3['acc'].mean():.3f} "
                  f"-> h8_fresh {h8f['acc'].mean():.3f}")

    # R2-P4 mixed-m pooled descriptive
    fr4 = pool("a1536", "s3_frag2_rot")
    frB = {"i": fr_b["i"], "acc": None}
    idx = sorted(set(fr4["i"]) & set(fr_b["i"]))
    fr_m4 = (fr4["acc"][np.isin(fr4["i"], idx)] + fr_b["acc"][np.isin(fr_b["i"], idx)]) / 2
    i25_m4 = (pool("a1536", "s3_intact2_25")["acc"][np.isin(fr4["i"], idx)] +
              i25_b["acc"][np.isin(i25_b["i"], idx)]) / 2
    a35 = pool("a1536", "s3_intact2_35")["acc"][np.isin(fr4["i"], idx)]
    a23 = pool("a1536", "s3_intact2_23")["acc"][np.isin(fr4["i"], idx)]
    mpw = (12 * i25_m4 + 10 * a35 + 10 * a23) / 32
    D4 = fr_m4 - mpw
    n = len(D4)
    ci = np.percentile([D4[rng.integers(0, n, n)].mean() for _ in range(BOOT)], [2.5, 97.5])
    V["R2-P4"] = (f"descriptive pooled D={D4.mean():+.3f} CI[{ci[0]:+.3f},{ci[1]:+.3f}] "
                  f"(round-1 P1 verdict stands)")

    # R2-P5 pinned calibration, fresh endpoints
    def Rof(arm):
        idx = sorted(set(arm["i"]) & set(h0["i"]) & set(h8f["i"]))
        a = arm["lp"][np.isin(arm["i"], idx)]
        z = h0["lp"][np.isin(h0["i"], idx)]
        e = h8f["lp"][np.isin(h8f["i"], idx)]
        return float((a - z).sum() / (e - z).sum())
    devs = []
    for nm, d in [("r2_intact3", i3), ("r2_frag2_rot_b", fr_b), ("r2_intact2_25_b", i25_b)]:
        r = Rof(d)
        pred = 1 / (1 + np.exp(-(A2 * r + B2)))
        dev = d["acc"].mean() - pred
        devs.append(abs(dev))
        print(f"  P5 {nm:<16} R={r:.3f} pred={pred:.3f} acc={d['acc'].mean():.3f} dev={dev:+.3f}")
    V["R2-P5"] = ("PASS" if max(devs) <= 0.10 and np.mean(devs) <= 0.05
                  else f"MISS") + f" (max {max(devs):.3f}, mean {np.mean(devs):.3f})"

    if pre["P6"]:
        sx, okx = sup("P6x", s2["r2_bothfrag"], s2["r2_co25_b"], "accx", 0.10)
        sy, oky = sup("P6y", s2["r2_bothfrag"], s2["r2_co25_b"], "accy", 0.00)
        V["R2-P6"] = f"{'PASS' if okx and oky else 'MISS'} (x: {sx}; y: {sy})"
    else:
        V["R2-P6"] = "OUT OF RANGE"

    print("\n== ROUND-2 VERDICTS ==")
    for k, v in V.items():
        print(f"  {k:<7} {v}")
    json.dump(V, open(os.path.join(os.environ.get("RA_ROOT", "."), "results/s3_queue/ROUND2_VERDICTS.json"), "w"))


if __name__ == "__main__":
    main()
