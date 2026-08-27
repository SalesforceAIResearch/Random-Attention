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
"""Equal-memory serving figure (fig_isomem.pdf) for the ICLR paper.

Two panels (Qwen3-4B, Qwen3-14B): horizontal bars of decode throughput relative to full attention
at 32k generations when each method runs at the largest batch that fits one
143 GB H200 at K=3072 (data: figures/results_isomem.tsv, clean 08-09 passes).
Bar annotations carry the max batch; full-attention reference line at 1x. TriAttention
is drawn gray: its row runs our unfused PyTorch port of their scorer (no fused
kernel exists by their own account), so it is implementation-limited -- the
kernel-fair TriAttention comparison is the vLLM table.

Usage:  [PAPER=1] python plot_isomem.py  ->  fig_isomem.pdf / .png
"""
import os

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
plt.rcParams.update({"axes.labelsize": 12.5, "axes.titlesize": 12.5,
                     "xtick.labelsize": 10.5, "ytick.labelsize": 10.5})

# ---------------------------------------------------------------- data
ROWS = {}
with open(os.path.join(HERE, "results_isomem.tsv")) as f:
    for ln in f:
        if ln.startswith("#") or ln.startswith("model\t") or not ln.strip():
            continue
        model, out_len, method, batch, tok_s, peak = ln.split("\t")
        ROWS[(model, int(out_len), method)] = (int(batch), float(tok_s), float(peak))

OUT = 32768
METHODS = [("triattn", "TriAttention*"), ("rkv", "R-KV"), ("snapkv", "SnapKV"),
           ("vase", "VaSE"), ("random_pp", "Random Attention")]

fig, axes = plt.subplots(1, 2, figsize=(9.4, 2.9), sharey=True)
for ax, model in zip(axes, ["Qwen3-4B", "Qwen3-14B"]):
    dense_b, dense_t, _ = ROWS[(model, OUT, "dense")]
    ys = np.arange(len(METHODS))
    for y, (meth, lab) in zip(ys, METHODS):
        b, t, _ = ROWS[(model, OUT, meth)]
        x = t / dense_t
        col = RED if meth == "random_pp" else (GRAY if meth == "triattn" else BLUE)
        ax.barh(y, x, height=0.62, color=col, zorder=3)
        ax.annotate(f"{x:.1f}$\\times$", (x, y), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=9.5,
                    color=DGRAY)
        ax.annotate(f"batch {b}", (0.15, y), va="center", ha="left",
                    fontsize=8.5, color="white", zorder=4)
    ax.axvline(1.0, ls="--", lw=1.1, color=DGRAY, zorder=2)
    ax.set_yticks(ys)
    ax.set_yticklabels([lab for _, lab in METHODS], fontsize=10.5)
    ax.set_xlim(0, 11.8)
    ax.set_xlabel(r"decode throughput, $\times$ full attention")
    ax.set_title(model, pad=6)
    ax.grid(True, axis="x", color="#e1e0d9", lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

fig.tight_layout(w_pad=1.6)
out = os.path.join(HERE, "fig_isomem")
fig.savefig(out + ".pdf", bbox_inches="tight")
fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
print("wrote", out + ".pdf / .png")
