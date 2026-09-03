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
"""Headline mechanism figure (fig_mechanism.pdf) for the ICLR paper
"Random Attention: Rethinking KV Cache Eviction for Efficient Reasoning".

Four-panel story:
  (a) F1 spine  — natural per-head-random survival: logprob recovery R and free-recall
      accuracy vs eviction events E, against the zero-free-parameter any-copy union
      prediction 0.95*[1-(1-s(E))^8], s(E)=(1024/1088)^E.  Accuracy tracks the union
      curve at E=3 and falls far below from E=15 on: recovery is graded in copy count.
  (b) payload — exact retrieval at two-carrier cache cost when the whole sentence, only the
      value digits, or everything except the digits survives (= tables/synth_storage.tex).
  (c) W2 mirror — real MATH500 4x scatter-block-size dose: flat until per-head spreading
      becomes impossible (B=256, a head holds only 4 blocks), only then below plain recency.
  (d) R2-P3     — carrier-mass knee (descriptive, single-era fresh 07-31 curve): retrieval
      accuracy vs which carrier columns survive.  The three pairs share column count (2)
      but differ in carrier quality ({2,3} weakest -> {2,5} strongest); the triple {2,3,5}
      jumps to 0.833 and all-8 saturates at 0.986.

Numbers: canonical replication values parsed from results/synth_runs/FULL_SCOREBOARD.txt
(endpoint-paired ratio-of-means; number policy WRITING_PACK_MECHANISM.md SS A/7); s_E and
union_pred cross-checked against results/synth_tasks/<cell>/meta.json.  The intact-16
carrier mean (0.332) is recomputed from the h1_head{2,3,5} solo rows per the number policy.
W2 real-task points are hard-coded (provenance: SYNTH_RESULTS.md W2 entries 167-183,
Qwen3-4B MATH500 4x flag accuracy, n=1000/cell; recency reference 0.843).
Panel (d) points are hard-coded (provenance: SYNTH_RESULTS.md R2-P3 entry, line 287,
07-31 round-2 adjudication, descriptive; pinned-point lineage REGISTERED.md line 178).
All five are same-era FRESH 07-31 arms (pairs = fresh round-1 re-measurements; triple
r2_intact3 + all-8 r2_h8_fresh = round-2), canonical per the fresh-run policy — do NOT
mix in the original 07-24-era pair values (0.150/0.393/0.627).

Usage:  [PAPER=1] python plot_mechanism.py   ->  fig_mechanism.png / .pdf
PAPER=1 -> camera-ready styling (serif/Times fonts, no suptitle), like the sibling scripts.
"""
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCOREBOARD = os.path.join(ROOT, "results", "synth_runs", "FULL_SCOREBOARD.txt")
RUNS = os.path.join(ROOT, "results", "synth_runs")
TASKS = os.path.join(ROOT, "results", "synth_tasks")

PAPER = bool(os.environ.get("PAPER"))
# paper palette (same hues as plot_compression.py / plot_throughput.py)
RED, BLUE, GRAY, DGRAY = "#e34948", "#2a78d6", "#b6b4b0", "#57544f"
if PAPER:
    import glob as _glob
    import matplotlib.font_manager as _fm
    for _f in _glob.glob(os.path.join(HERE, "fonts", "*.ttf")):
        _fm.fontManager.addfont(_f)
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "Liberation Serif",
                                        "STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix"})
plt.rcParams.update({"axes.labelsize": 12.5, "axes.titlesize": 12.5,
                     "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
                     "legend.fontsize": 9.5})

# ---------------------------------------------------------------- data loading
ROW = re.compile(r"^(\S+)\s+(\S+)\s+(-?[\d.]+|--)\s*(?:\[\s*(-?[\d.]+),\s*(-?[\d.]+)\])?"
                 r"\s+(-?[\d.]+)\s*\[\s*(-?[\d.]+),\s*(-?[\d.]+)\]\s+(\d+)")

def load_scoreboard(path):
    """-> {(cell, arm): dict(R, R_lo, R_hi, acc, acc_lo, acc_hi, n)}; R None on acc-only rows."""
    d = {}
    with open(path) as f:
        for line in f:
            m = ROW.match(line)
            if not m:
                continue
            cell, arm, R, rlo, rhi, acc, alo, ahi, n = m.groups()
            d[(cell, arm)] = dict(
                R=None if R == "--" else float(R),
                R_lo=None if rlo is None else float(rlo),
                R_hi=None if rhi is None else float(rhi),
                acc=float(acc), acc_lo=float(alo), acc_hi=float(ahi), n=int(n))
    return d

SB = load_scoreboard(SCOREBOARD)

def meta(cell):
    with open(os.path.join(TASKS, cell, "meta.json")) as f:
        return json.load(f)

# cells on the F1 spine: eviction-event count E and needle age (tokens)
CELLS = [("a768f", 3, 768), ("a1536", 15, 1536), ("a3072", 39, 3072), ("adeep", 57, 4224)]
for cell, E, _ in CELLS:                       # cross-check meta vs the closed form
    mt = meta(cell)
    assert mt["E"] == E, (cell, mt["E"], E)
    s = (1024 / 1088) ** E
    assert abs(mt["s_E"] - s) < 1e-9
    assert abs(mt["union_pred"] - (1 - (1 - s) ** 8)) < 1e-9

def yerr(rows, lo_key, hi_key, val_key):
    v = np.array([r[val_key] for r in rows], float)
    lo = np.array([r[lo_key] for r in rows], float)
    hi = np.array([r[hi_key] for r in rows], float)
    return v, np.vstack([v - lo, hi - v])

# ------------------------------------------------------------------- figure
fig, (axA, axB, axC) = plt.subplots(
    1, 3, figsize=(10.6, 2.8), gridspec_kw={"width_ratios": [1.35, 1.0, 1.0]})
# the by-age keep-log panel is a standalone appendix figure (fig_keeplog)
figK, axD = plt.subplots(1, 1, figsize=(4.8, 3.3))

# ---- (a) pooling: retrieval vs which/how many storing heads keep the needle --
# Values are rows of tables/synth_storage.tex (same generator, same raw arms).
P_cfg = ["h5", "h2", r"$\{2,3\}$", r"$\{3,5\}$", r"$\{2,5\}$", r"$\{2,3,5\}$", "all 8"]
P_acc = [0.028, 0.015, 0.161, 0.386, 0.597, 0.833, 0.986]
xa = np.arange(len(P_acc))
axA.axvspan(-0.5, 1.5, color="#f0efe9", zorder=0)
axA.text(0.5, 0.93, "one head\nalone", ha="center", va="top", fontsize=8.5, color=DGRAY)
axA.bar(xa[:2], P_acc[:2], width=0.62, color=GRAY, zorder=3)
axA.bar(xa[2:], P_acc[2:], width=0.62, color=RED, zorder=3)
for x, a in zip(xa, P_acc):
    axA.annotate(f"{a:.2f}", (x, a), xytext=(0, 3), textcoords="offset points",
                 ha="center", fontsize=8.5, color=DGRAY)
axA.set_xticks(xa)
axA.set_xticklabels(P_cfg, fontsize=9)
axA.set_xlabel("heads holding the needle")
axA.set_ylabel("retrieval accuracy")
axA.set_xlim(-0.5, len(P_acc) - 0.5)
axA.set_ylim(0, 1.10)
axA.set_title("(a) Copies in different heads pool", pad=8)

# ---- (b) two facts in different heads combine (super-additive) --------------
# a1536_s2r recall-both probe: x alone in h3, y alone in h5, both in those two heads.
def _R_s2r(arm):
    import json as _j
    ref0 = [_j.loads(l)["lp_sum"] for l in open(f"{RUNS}/a1536_s2r/s2r_none.jsonl")]
    ref1 = [_j.loads(l)["lp_sum"] for l in open(f"{RUNS}/a1536_s2r/s2r_both_all.jsonl")]
    v = [_j.loads(l)["lp_sum"] for l in open(f"{RUNS}/a1536_s2r/{arm}.jsonl")]
    n = min(len(v), len(ref0), len(ref1))
    v, r0, r1 = np.array(v[:n]), np.array(ref0[:n]), np.array(ref1[:n])
    return float((v - r0).sum() / (r1 - r0).sum())

C_lab = ["$x$ in h3\nalone", "$y$ in h5\nalone", "both,\nin h3 and h5"]
C_val = [_R_s2r("s2r_x_only"), _R_s2r("s2r_y_only"), _R_s2r("s2r_disjoint")]
xb = np.arange(3)
axB.bar(xb, C_val, width=0.62, color=[GRAY, GRAY, RED], zorder=3)
axB.axhline(C_val[0] + C_val[1], ls="--", lw=1.4, color=DGRAY, zorder=2)
axB.text(-0.42, C_val[0] + C_val[1] + 0.008, "sum of the two alone",
         ha="left", va="bottom", fontsize=8.5, color=DGRAY)
for x, a in zip(xb, C_val):
    axB.annotate(f"{a:.2f}", (x, a), xytext=(0, 3), textcoords="offset points",
                 ha="center", fontsize=9, color=DGRAY)
axB.set_xticks(xb)
axB.set_xticklabels(C_lab, fontsize=9)
axB.set_ylabel(r"recall $R$, both values")
axB.set_ylim(0, 0.44)
axB.set_title("(b) Two facts in different heads", pad=8)

# ---- (c) the block-size cost is set by blocks per head, not block length ------
# results/regrade_sweep_0726.tsv, Qwen3-4B MATH500, n=1000/cell; delta vs the B=1
# policy at the same budget, against how many blocks of that size fit one head.
B = [1, 4, 16, 64, 256]
acc4 = [0.874, 0.867, 0.872, 0.867, 0.808]
acc8 = [0.789, 0.793, 0.782, 0.767, 0.625]
n4 = [1024 // b for b in B]
n8 = [512 // b for b in B]
d4 = [(a - acc4[0]) * 100 for a in acc4]
d8 = [(a - acc8[0]) * 100 for a in acc8]
axC.axhline(0, ls="-", lw=0.8, color=DGRAY, zorder=1)
axC.plot(B, d4, "o-", lw=2.0, ms=5.5, color=RED, zorder=3, label=r"$4\times$ ($\budget{=}1024$)".replace("\\budget{=}", "K="))
axC.plot(B, d8, "s--", lw=1.8, ms=5, color=BLUE, zorder=3, label=r"$8\times$ (K=512)")
axC.set_xscale("log", base=4)
axC.set_xticks(B)
axC.set_xticklabels([str(b) for b in B])
axC.minorticks_off()
axC.annotate("4 blocks left", xy=(256, -6.6), xytext=(-64, -4), textcoords="offset points",
             fontsize=8.5, color=DGRAY)
axC.annotate("2 blocks left", xy=(256, -16.4), xytext=(-62, 4), textcoords="offset points",
             fontsize=8.5, color=DGRAY)
axC.set_xlabel("block size (tokens)")
axC.set_ylabel(r"$\Delta$ accuracy")
axC.set_ylim(-18, 2)
axC.legend(loc="lower left", frameon=False, fontsize=8.5, handlelength=1.6)
axC.set_title("(c) Real MATH500: shape is free", pad=8)

# ---- (d) how much of each age band each policy keeps, per head ---------------
UNION_TSV = os.path.join(ROOT, "figures", "union_by_age.tsv")
U = {}
with open(UNION_TSV) as fh:
    for ln in fh:
        if ln.startswith("#") or not ln.strip():
            continue
        parts = ln.rstrip("\n").split("\t")
        if len(parts) < 6:
            continue
        m, lo, hi, sv, un, ind = parts[:6]
        try:
            U.setdefault(m, []).append((int(lo), float(sv), float(un), float(ind)))
        except ValueError:
            continue  # header row
BANDS = [(512, "512"), (1024, "1k"), (2048, "2k"), (4096, "4k")]
keys = [b for b, _ in BANDS]
xu = np.arange(len(keys))
GREEN = "#3f8f5b"
# TriAttention curve = `triattn_ph_memofix` (the faithful port; the earlier
# `triattn_ph` keep-log was recorded with the retracted frozen-policy
# implementation and is not plotted).
for meth, col, mk, lab in [("random_pp", RED, "o", "Random Attention"),
                           ("vase_faithful", BLUE, "s", "VaSE"),
                           ("triattn_ph_memofix", GREEN, "^", "TriAttention")]:
    rows = {lo: (sv, un, ind) for lo, sv, un, ind in U[meth]}
    axD.plot(xu, [rows[k][0] for k in keys], marker=mk, ls="-", lw=2.0, ms=6,
             color=col, zorder=3, label=lab)
axD.axvspan(0.5, 1.5, color="#f0efe9", zorder=0)
axD.set_yscale("log")
axD.text(1.0, 0.62, "band reasoning\nlooks back into", ha="center", va="top",
         fontsize=8.5, color=DGRAY)
axD.annotate("", xy=(1.0, 0.188), xytext=(1.0, 0.0858),
             arrowprops=dict(arrowstyle="<->", color=DGRAY, lw=1.0, shrinkA=1, shrinkB=1))
axD.text(1.12, 0.127, r"$2.2\times$", fontsize=9, color=DGRAY, va="center")
axD.set_xticks(xu)
axD.set_xticklabels([f"{lab}--{BANDS[i+1][1] if i+1 < len(BANDS) else '8k'}"
                     for i, (_, lab) in enumerate(BANDS)], fontsize=9)
axD.set_xlabel("token age (distance back)")
axD.set_ylabel("fraction a head still holds")
axD.set_ylim(0.0018, 0.95)
axD.legend(loc="upper right", frameon=False, fontsize=8.5, handlelength=1.5,
           borderaxespad=0.2)
axD.set_title("Where each policy spends its budget", pad=8)

for ax in (axA, axB, axC, axD):
    ax.grid(True, axis="y", color="#e1e0d9", lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

if not PAPER:
    fig.suptitle("Mechanism: copies accumulate; fragmentation is free (synthetic <-> real)",
                 fontsize=12, fontweight="600", y=1.02)
fig.tight_layout(w_pad=2.6)
out = os.path.join(HERE, "fig_mechanism")
fig.savefig(out + ".pdf", bbox_inches="tight")
fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
figK.tight_layout()
outK = os.path.join(HERE, "fig_keeplog")
figK.savefig(outK + ".pdf", bbox_inches="tight")
figK.savefig(outK + ".png", dpi=200, bbox_inches="tight")
print("wrote", out + ".pdf / .png")
print("panel (a) pooling acc:", P_acc)
print("panel (b) combination R:", [f"{v:.3f}" for v in C_val], "sum", f"{C_val[0]+C_val[1]:.3f}")
print("panel (c) blocks/head vs delta:", list(zip(n4, [round(x,1) for x in d4])), list(zip(n8, [round(x,1) for x in d8])))
print("panel (d) per-head s by band:", {m: [round(r[1],3) for r in v[2:6]] for m, v in U.items()})
