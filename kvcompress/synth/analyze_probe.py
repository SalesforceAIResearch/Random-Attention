#!/usr/bin/env python
"""Model-agnostic scoring of planted-fact probe arms (companion to gen_synth_tables.py, which is
tied to the registered Qwen3-4B cohort). For every <arm>.jsonl in a cell: retrieval accuracy
(greedy free-gen reproduces the value), recall R = sum(LP - LP_h0) / sum(LP_hall - LP_h0) against
the deleted-everywhere / kept-everywhere endpoints, and for two-fact cells (n_val_tokens > n_val_x)
the per-fact retrieval of x and y from the per-token argmax flags.
  python analyze_probe.py --tasks results/synth_tasks_phi4 --runs results/synth_runs_phi4 --cell a1536 --h0 h0 --hall h10
"""
import argparse, glob, json, os
import numpy as np


def load(path):
    rs = sorted((json.loads(l) for l in open(path)), key=lambda r: r["i"])
    return {"i": np.array([r["i"] for r in rs]), "lp": np.array([r["lp_sum"] for r in rs]),
            "acc": np.array([r["fg_ok"] for r in rs], float),
            "gok": np.array([r["gok"] for r in rs], bool) if "gok" in rs[0] else None}


def R_of(a, z, e, boot=2000, rng=None):
    idx = sorted(set(a["i"]) & set(z["i"]) & set(e["i"]))
    A, Z, E = (d["lp"][np.isin(d["i"], idx)] for d in (a, z, e))
    r = float((A - Z).sum() / (E - Z).sum())
    if rng is None:
        return r, None, None
    v = []
    n = len(A)
    for _ in range(boot):
        k = rng.integers(0, n, n)
        den = (E[k] - Z[k]).sum()
        if den:
            v.append((A[k] - Z[k]).sum() / den)
    return r, float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True); ap.add_argument("--runs", required=True)
    ap.add_argument("--cell", required=True); ap.add_argument("--h0", default="h0"); ap.add_argument("--hall", default="h10")
    ap.add_argument("--json", default=None, help="write per-arm metrics here")
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.tasks, a.cell, "meta.json")))
    NV, NVX = meta.get("n_val_tokens", 4), meta.get("n_val_x", 4)
    rd = os.path.join(a.runs, a.cell)
    arms = sorted(os.path.basename(p)[:-6] for p in glob.glob(os.path.join(rd, "*.jsonl"))
                  if os.path.exists(os.path.join(rd, os.path.basename(p)[:-6] + ".done")))
    z = load(os.path.join(rd, a.h0 + ".jsonl")) if a.h0 in arms else None
    e = load(os.path.join(rd, a.hall + ".jsonl")) if a.hall in arms else None
    rng = np.random.default_rng(0)
    out = {}
    print(f"cell {a.cell}  E={meta['E']}  n_kv_heads={meta.get('n_kv_heads')}  NV={NV} NVX={NVX}  endpoints {a.h0}/{a.hall}")
    print(f"{'arm':<22} {'n':>4} {'retr':>6} {'R':>7} {'R 95% CI':>16} {'lp_mean':>8} {'retr_x':>7} {'retr_y':>7}")
    for arm in arms:
        d = load(os.path.join(rd, arm + ".jsonl"))
        r, lo, hi = R_of(d, z, e, rng=rng) if (z is not None and e is not None) else (float("nan"), None, None)
        rec = {"n": int(len(d["i"])), "retr": float(d["acc"].mean()), "R": r, "R_lo": lo, "R_hi": hi, "lp_mean": float(d["lp"].mean())}
        if d["gok"] is not None and NV > NVX:
            rec["retr_x"] = float(d["gok"][:, :NVX].all(1).mean()); rec["retr_y"] = float(d["gok"][:, NV - NVX:].all(1).mean())
        out[arm] = rec
        ci = f"[{lo:.2f},{hi:.2f}]" if lo is not None else ""
        print(f"{arm:<22} {rec['n']:>4} {rec['retr']:>6.3f} {r:>7.3f} {ci:>16} {rec['lp_mean']:>8.2f} "
              f"{rec.get('retr_x', float('nan')):>7.3f} {rec.get('retr_y', float('nan')):>7.3f}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
