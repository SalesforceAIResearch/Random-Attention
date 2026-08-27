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
"""Draw the KV-compression regime figures from results.tsv (produced by grade_regime_all.sh).
results.tsv columns: model  task  comp  K  method  flag_acc  acc_strict  n
Usage: python plot_compression.py [results.tsv] [out_prefix]  ->  <prefix>.png / .pdf

REPLOT COMMANDS (run from figures/; regenerates both regime figures from the TSVs):
  P=python
  $P plot_compression.py results_regime_4b_phi.tsv fig_regime_4b_phi
  $P plot_compression.py results_regime_phi.tsv    fig_regime_phi
Only cells present in the TSV are plotted; partial coverage renders as shorter lines."""
import os, sys, csv, collections
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

TSV = sys.argv[1] if len(sys.argv) > 1 else "results.tsv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "fig_compression"
PAPER = bool(os.environ.get("PAPER"))       # PAPER=1 -> camera-ready styling (labels, fonts, no suptitle)

# validated categorical hues (dataviz reference palette, light mode)
C = {"random_pp": "#e34948", "vase_faithful": "#2a78d6", "triattn_ph": "#1baf7a",
     "attn": "#eda100", "dense": "#898781"}
LBL = {"random_pp": "Random Attention", "vase_faithful": "VaSE", "triattn_ph": "TriAttention",
       "attn": "SnapKV", "dense": "dense"}
TASK_TITLE = {"math": "MATH", "gpqa": "GPQA", "aime": "AIME", "hmmt": "HMMT"}
if PAPER:
    LBL["dense"] = "Full"
    TASK_TITLE = {"math": "MATH500", "gpqa": "GPQA-D", "aime": "AIME", "hmmt": "HMMT"}
    # Times New Roman when installed; STIXGeneral (Times-compatible) otherwise
    import glob as _glob
    import matplotlib.font_manager as _fm
    for _f in _glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "*.ttf")):
        _fm.fontManager.addfont(_f)
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "Liberation Serif",
                                        "STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix"})
COMP_ORDER = ["2x", "4x", "8x", "16x"]; COMP_X = {"2x": 1, "4x": 2, "8x": 3, "16x": 4}
MODELS = ["Qwen3-4B", "phi-4-reasoning", "Qwen3-32B"]
TASKS = ["math", "gpqa", "aime", "hmmt"]

# load: d[model][task][method] = list of (x, acc)
d = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
dense = collections.defaultdict(dict)
with open(TSV) as f:
    for r in csv.DictReader(f, delimiter="\t"):
        acc = float(r["flag_acc"]); m, t, c, meth = r["model"], r["task"], r["comp"], r["method"]
        if meth == "dense": dense[m][t] = acc; continue
        if c in COMP_X: d[m][t][meth].append((COMP_X[c], acc))

present = [(m, t) for m in MODELS for t in TASKS if d[m][t]]
if not present:
    print("no data in", TSV); sys.exit(0)
ncol = len([t for t in TASKS if any(d[m][t] for m in MODELS)]) or 1
nrow = len([m for m in MODELS if d[m]]) or 1
rows = [m for m in MODELS if any((m, t) in present for t in TASKS)]
cols = [t for t in TASKS if any((m, t) in present for m in rows)]
fig, axes = plt.subplots(len(rows), len(cols), figsize=(3.2*len(cols), 2.7*len(rows)),
                         squeeze=False, sharex=True)
for ri, m in enumerate(rows):
    for ci, t in enumerate(cols):
        ax = axes[ri][ci]
        # panel y-cap: full [0,1] only when the data reaches high; otherwise trim dead space
        vals = [y for pp in d[m][t].values() for _, y in pp] + ([dense[m][t]] if t in dense[m] else [])
        ymax = max(vals)
        top = 1.0 if ymax > 0.82 else min(1.0, (int((ymax + 0.05) * 20) + 1) / 20)
        xmax_panel = max(x for pp in d[m][t].values() for x, _ in pp)
        n_x = 0                         # infeasibility x-marks already placed (dodge horizontally)
        Z = {"random_pp": 6, "triattn_ph": 4, "vase_faithful": 3, "attn": 3}   # rp drawn on top
        ends = []                       # (y_end, x_end, meth) for the two-pass label dodge
        for meth in ["random_pp", "triattn_ph", "vase_faithful",  "attn"]:
            pts = sorted(d[m][t].get(meth, []))
            if not pts: continue
            xs, ys = zip(*pts)
            MK = {"random_pp": ("o", 5.5, 1.5), "triattn_ph": ("D", 4.5, 1.5),
                  "vase_faithful": ("x", 6.5, 2.0), "attn": ("s", 4.5, 1.5)}
            mk, ms, mew = MK.get(meth, ("o", 5, 1.5))
            ax.plot(xs, ys, "-", marker=mk, color=C[meth], lw=2, ms=ms, mew=mew,
                    label=LBL[meth], zorder=Z.get(meth, 3))
            ends.append((ys[-1], xs[-1], meth))
            if xs[-1] < xmax_panel:     # method infeasible at the deepest compression -> x mark
                ax.plot([xmax_panel + 0.09 * n_x], [0.045 * top], marker="x", ms=9, mew=2.6,
                        color=C[meth], zorder=4, clip_on=False)
                n_x += 1
        # second pass: greedy top-down dodge in data coords — labels keep >= min_sep spacing
        min_sep = 0.055 * top
        y_lab_prev = None
        for y_end, x_end, meth in sorted(ends, reverse=True):
            y_lab = y_end if y_lab_prev is None else min(y_end, y_lab_prev - min_sep)
            y_lab_prev = y_lab
            ax.annotate(f"{y_end:.2f}", (x_end, y_lab), color=C[meth], va="center",
                        fontsize=9.5, fontweight="bold", xytext=(5, 0),
                        textcoords="offset points", zorder=7)
        if t in dense[m]:
            ax.axhline(dense[m][t], color=C["dense"], ls="--", lw=1.3, alpha=.8, zorder=1)
        ax.set_ylim(0, top); ax.set_xticks([1, 2, 3, 4]); ax.set_xticklabels(["2×", "4×", "8×", "16×"])
        ax.grid(True, axis="y", color="#e1e0d9", lw=.8); ax.set_axisbelow(True)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        if ri == 0: ax.set_title(TASK_TITLE.get(t, t.upper()), fontsize=14, fontweight="bold")
        if ci == 0: ax.set_ylabel(("accuracy" if PAPER and len(rows) == 1 else m), fontsize=13)
        ax.tick_params(labelsize=11)
axes[0][0].legend(fontsize=10.5, frameon=False, loc="lower left")
# if not PAPER:
#     fig.suptitle("Accuracy vs KV-cache compression (flag_acc)", fontsize=13, fontweight="600", y=1.0)
fig.supxlabel("Compression Factor", fontsize=18)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}.{ext}", dpi=200, bbox_inches="tight")
print(f"wrote {OUT}.png / {OUT}.pdf  ({len(present)} panels: {present})")
