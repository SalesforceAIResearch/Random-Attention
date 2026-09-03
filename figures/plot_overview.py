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
"""Overview figure (fig_overview.pdf) for page 1 of the ICLR paper.

(a) Mean accuracy over the five task columns of Table 1 (main_grid.tex, hand-maintained
    in the paper repo = source of truth) for every eviction method on the three headline
    models; full attention as a dashed line per model.
(b) vLLM serving throughput at 32k-token generations (results_vllm.tsv, out32k rows):
    full attention, TriAttention, Random Attention; margin over TriAttention annotated.

Usage:  [PAPER=1] python plot_overview.py  ->  fig_overview.pdf / .png
"""
import os, re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = bool(os.environ.get("PAPER"))
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
plt.rcParams.update({"axes.labelsize": 11.5, "axes.titlesize": 11.5,
                     "xtick.labelsize": 10, "ytick.labelsize": 9.5})

# ---------------------------------------------------------------- (a) accuracy from Table 1
GRID = os.environ.get("MAIN_GRID", os.path.expanduser(
    "~/Random-Attention-ICLR-27/tables/main_grid.tex"))
ROWKEY = {"\\dense{}": "full", "\\snapkv{}": "snapkv", "\\rkv{}": "rkv", "\\vase{}": "vase",
          "\\triattn{}": "triattn", "\\method{} (ours)": "rp"}
MAIN_BLOCKS = ["Qwen3-4B", "Phi-4-reasoning", "Qwen3-32B"]   # block order in main_grid.tex
MODELS_A = ["Qwen3-4B", "Phi-4-reasoning", "Qwen3-14B", "Qwen3-32B"]  # display order (matches panel b)
APPX_GRID = os.environ.get("APPX_GRID", os.path.expanduser(
    "~/Random-Attention-ICLR-27/tables/appendix_grid.tex"))
acc = {}
def parse_grid(path, blocks):
    blk = -1
    for ln in open(path):
        ln = ln.strip()
        if ln.startswith("%") or "&" not in ln:
            continue
        cells = [c.strip() for c in ln.rstrip("\\").split("&")]
        key = cells[0].replace("\\rowcolor{oursbg}", "").strip()
        if key not in ROWKEY:
            continue
        if ROWKEY[key] == "full":
            blk += 1
        vals = []
        for c in cells[1:]:
            m = re.search(r"(\d\.\d{3})", c)
            if m:
                vals.append(float(m.group(1)))
        assert len(vals) == 5, (ln, vals)
        if os.environ.get("TASKS", "all") == "nocode":
            vals = vals[:4]
        acc[(blocks[blk], ROWKEY[key])] = float(np.mean(vals))
    return blk
assert parse_grid(GRID, MAIN_BLOCKS) == 2, "expected three model blocks in main_grid.tex"
assert parse_grid(APPX_GRID, ["Qwen3-14B"]) == 0, "expected one model block in appendix_grid.tex"

# ---------------------------------------------------------------- (b) vLLM throughput
TP = {}
for ln in open(os.path.join(HERE, "results_vllm.tsv")):
    if ln.startswith("#") or ln.startswith("model\t") or not ln.strip():
        continue
    p = ln.rstrip("\n").split("\t")
    if len(p) >= 5 and p[1] == "out32k" and p[3] == "out_tok_s":
        TP[(p[0], p[2])] = float(p[4])
MODELS_B = ["Qwen3-4B", "phi-4-reasoning", "Qwen3-14B", "Qwen3-32B"]

# ---------------------------------------------------------------- draw
fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.4, 2.4),
                               gridspec_kw={"width_ratios": [2.7, 2.7], "wspace": 0.26})
METH = [("full", "full attention", DGRAY), ("snapkv", "SnapKV", BLUE), ("rkv", "R-KV", BLUE),
        ("vase", "VaSE", BLUE), ("triattn", "TriAttention", BLUE), ("rp", "Random Attention", RED)]
w = 0.13
for gi, model in enumerate(MODELS_A):
    x0 = gi
    for mi, (k, lab, col) in enumerate(METH):
        x = x0 + (mi - 2.5) * w
        v = acc[(model, k)]
        alpha = 0.55 if k in ("snapkv", "rkv", "vase") else 1.0
        axA.bar(x, v, width=w * 0.92, color=col, alpha=alpha, zorder=3,
                label=lab if gi == 0 else None)
    d = 100 * (acc[(model, "rp")] - acc[(model, "triattn")])
    ytop = max(acc[(model, "rp")], acc[(model, "triattn")])
    axA.annotate(f"{d:+.1f}", (x0 + 2.0 * w, ytop), xytext=(0, 3),
                 textcoords="offset points", ha="center", va="bottom",
                 fontsize=8, color="#57544f", zorder=5)
axA.set_xticks(range(len(MODELS_A)))
axA.set_xticklabels(["Qwen3-4B", "Phi-4", "Qwen3-14B", "Qwen3-32B"], fontsize=8.5)
axA.set_ylim(0.3, 0.8)
axA.set_yticks(np.arange(0.3, 0.81, 0.1))
axA.set_ylabel("Avg. Accuracy" if os.environ.get("TASKS", "all") != "nocode"
               else "mean accuracy, math and science")
axA.set_title("(a) Accuracy at ~4\u00d7 compression", loc="center")
axA.grid(axis="y", color="#e6e4df", zorder=0)
axA.set_axisbelow(True)
for s in ("top", "right"):
    axA.spines[s].set_visible(False)

METH_B = [("dense", "full attention", DGRAY), ("triattn", "TriAttention", BLUE),
          ("random_pp", "Random Attention", RED)]
wb = 0.24
for gi, model in enumerate(MODELS_B):
    for mi, (k, lab, col) in enumerate(METH_B):
        x = gi + (mi - 1) * wb
        v = TP[(model, k)]
        axB.bar(x, v, width=wb * 0.9, color=col, zorder=3, label=lab if gi == 0 else None)
    rp, tri, dense = TP[(model, "random_pp")], TP[(model, "triattn")], TP[(model, "dense")]
    axB.annotate(f"{rp / dense:.2f}\u00d7", (gi + wb, rp), xytext=(0, 3),
                 textcoords="offset points", ha="center", fontsize=8.5, color=DGRAY)
    axB.annotate(f"+{100 * (rp / tri - 1):.0f}%", (gi + wb, rp), xytext=(0, 14),
                 textcoords="offset points", ha="center", fontsize=9, color=RED,
                 fontweight="bold")
axB.set_xticks(range(len(MODELS_B)))
axB.set_xticklabels(["Qwen3-4B", "Phi-4", "Qwen3-14B", "Qwen3-32B"], fontsize=8.5)
axB.set_ylabel("tokens / s")
axB.set_title("(b) vLLM serving throughput, 32k", loc="center")
axB.set_ylim(0, 2600)
axB.grid(axis="y", color="#e6e4df", zorder=0)
axB.set_axisbelow(True)
for s in ("top", "right"):
    axB.spines[s].set_visible(False)
hA, lA = axA.get_legend_handles_labels()
hB, lB = axB.get_legend_handles_labels()
# shared legend: full attention (line), the four baselines, Random Attention
order = ["full attention", "SnapKV", "R-KV", "VaSE", "TriAttention", "Random Attention"]
H = dict(zip(lA, hA)); H.update({l: h for l, h in zip(lB, hB) if l not in H})
fig.legend([H[k] for k in order], order, ncol=6, fontsize=9, frameon=False,
           loc="lower center", bbox_to_anchor=(0.5, -0.1), handlelength=1.5,
           columnspacing=1.4)

fig.savefig(os.path.join(HERE, "fig_overview.pdf"), bbox_inches="tight", pad_inches=0.02)
fig.savefig(os.path.join(HERE, "fig_overview.png"), bbox_inches="tight", dpi=170)
for m in MODELS_A:
    print(m, {k: round(acc[(m, k)], 3) for k in ["full", "snapkv", "rkv", "vase", "triattn", "rp"]})
