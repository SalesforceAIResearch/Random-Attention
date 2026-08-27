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
"""Presentation figure for the VaSE-protocol throughput bench (fixed bs=16, H200, Qwen3-4B).
Reads logs/throughput_vase_protocol.tsv + logs/throughput_ksweep.tsv (+ figures/throughput_overrides.tsv)
so it re-renders as repair/K-sweep rows land.  Panels:
  A  decode tok/s @ 32k generation (dense vs evictors)     — the 3.1x story
  B  peak GPU mem @ 32k generation                          — the 7.7x story
  C  tok/s vs KV budget K @ 16k generation (evictors)       — rp leads at nearly every K
Usage: python figures/plot_throughput.py [outstem]
"""
import csv, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("RA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
C = {"random_pp": "#e34948", "vase": "#2a78d6", "triattn": "#1baf7a",
     "caote": "#2e9e73", "snapkv": "#eda100", "dense": "#777777"}
NAME = {"random_pp": "random_pp (ours)", "vase": "VaSE", "snapkv": "SnapKV",
        "caote": "CAOTE", "triattn": "TriAttention", "dense": "Full attention"}
PAPER = bool(os.environ.get("PAPER"))     # PAPER=1 -> camera-ready styling; CAOTE excluded
EXCLUDE = {"caote"} if PAPER else set()
if PAPER:
    NAME["random_pp"] = "Random Attention (ours)"

def read(path, keyfields):
    rows = {}
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            try:
                tok = float(r["tok_s"]); gb = float(r["peak_GB"])
            except (ValueError, KeyError):
                continue                      # skip ERR/OOM rows
            rows[tuple(r[k] for k in keyfields)] = (tok, gb)   # last valid row wins
    return rows

proto = read(f"{ROOT}/logs/throughput_vase_protocol.tsv", ("method", "output_len"))
ksweep = read(f"{ROOT}/logs/throughput_ksweep.tsv", ("method", "K"))
# (ksweep TSV vase/snapkv K2048 relabel fixed in-place 07-09; no runtime relabel needed)
for k, v in read(f"{ROOT}/figures/throughput_overrides.tsv", ("method", "K")).items():
    ksweep[k] = v
# K=1024 @16k lives in the protocol TSV
for m in ("random_pp", "vase", "snapkv", "caote", "triattn"):
    if (m, "16384") in proto:
        ksweep.setdefault((m, "1024"), proto[(m, "16384")])

isomem = {}
if os.path.exists(f"{ROOT}/logs/throughput_isomem.tsv"):
    with open(f"{ROOT}/logs/throughput_isomem.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            try: isomem[r["method"]] = (int(r["max_batch"]), float(r["tok_s"]), float(r["peak_GB"]))
            except (ValueError, KeyError): pass
HAVE_D = "dense" in isomem and len(isomem) >= 4

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
if PAPER:
    import glob as _glob
    import matplotlib.font_manager as _fm
    for _f in _glob.glob(f"{ROOT}/figures/fonts/*.ttf"):
        _fm.fontManager.addfont(_f)
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "Liberation Serif",
                                        "STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix",
                         "font.size": 16, "axes.titlesize": 20, "axes.labelsize": 19,
                         "xtick.labelsize": 16, "ytick.labelsize": 16,
                         "legend.fontsize": 14.5})
if HAVE_D:
    fig, (axA, axB, axC, axD) = plt.subplots(1, 4, figsize=(21.5, 4.6))
else:
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(16.5, 4.6))
fig.subplots_adjust(wspace=0.42)
if not PAPER:
    fig.suptitle("KV-cache eviction efficiency — VaSE official protocol (fixed batch 16, K=1024, H200, Qwen3-4B)",
                 fontsize=14, fontweight="bold", y=1.02)

order = [m for m in ("dense", "triattn", "snapkv", "vase", "caote", "random_pp") if m not in EXCLUDE]
# ---- A: throughput @32k ----
ms = [m for m in order if (m, "32768") in proto]
ys = range(len(ms))
axA.barh(list(ys), [proto[(m, "32768")][0] for m in ms],
         color=[C[m] for m in ms], height=0.62)
for y, m in zip(ys, ms):
    v = proto[(m, "32768")][0]
    axA.text(v + 8, y, f"{v:.0f}", va="center", fontweight="bold", color=C[m])
axA.set_yticks(list(ys)); axA.set_yticklabels([NAME[m] for m in ms])
axA.set_xlabel("decode throughput (tok/s)")
axA.set_title("Throughput (32k gen)", fontweight="bold")
axA.set_xlim(0, 660)
if ("dense", "32768") in proto and ("random_pp", "32768") in proto:
    sp = proto[("random_pp", "32768")][0] / proto[("dense", "32768")][0]
    dv = proto[("dense", "32768")][0]
    axA.text(dv + 75, 0, f" ← eviction = {sp:.1f}× faster", va="center",
             fontsize=16, fontweight="bold", color="#e34948")

# ---- B: peak mem @32k ----
ms = [m for m in order if (m, "32768") in proto]
axB.barh(list(range(len(ms))), [proto[(m, "32768")][1] for m in ms],
         color=[C[m] for m in ms], height=0.62)
for y, m in enumerate(ms):
    v = proto[(m, "32768")][1]
    axB.text(v + 1.2, y, f"{v:.1f} GB", va="center", fontweight="bold", color=C[m])
axB.set_yticks(list(range(len(ms)))); axB.set_yticklabels([])
axB.set_xlabel("peak GPU memory (GB)")
axB.set_title("Peak memory (32k gen)", fontweight="bold")
axB.set_xlim(0, 95)
if ("dense", "32768") in proto:
    r = proto[("dense", "32768")][1] / proto[("random_pp", "32768")][1]
    axB.text(26, len(ms) - 2.4, f"{r:.1f}× less memory", va="center",
             fontsize=16, fontweight="bold", color="#e34948")

# ---- C: tok/s vs K (16k gen) ----
KS = ["256", "512", "1024", "2048", "4096", "8192"]
xs = [k for k in KS if any((m, k) in ksweep for m in C)]
xpos = {k: i for i, k in enumerate(xs)}
for m in (m for m in ("triattn", "snapkv", "caote", "vase", "random_pp") if m not in EXCLUDE):
    pts = [(xpos[k], ksweep[(m, k)][0]) for k in xs if (m, k) in ksweep]
    if not pts:
        continue
    lw = 3 if m == "random_pp" else 2
    axC.plot(*zip(*pts), marker="o", ms=6, lw=lw, color=C[m], label=NAME[m],
             zorder=5 if m == "random_pp" else 3)
axC.set_xticks(range(len(xs)))
axC.set_xticklabels([k if int(k) < 1000 else f"{int(k)//1024}k" for k in xs])
axC.set_xlabel("KV budget K (tokens)")
d16 = next((v for (m, o), (v, _) in [] if False), None)
import csv as _csv
with open(f"{ROOT}/logs/throughput_vase_protocol.tsv") as _f:
    for _r in _csv.DictReader(_f, delimiter="\t"):
        if _r["method"] == "dense" and _r["output_len"] == "16384":
            try: d16 = float(_r["tok_s"])
            except ValueError: pass
if d16:
    axC.axhline(d16, color="#777777", ls="--", lw=1.6)
    axC.text(0.02, d16 + 8, f"Full attention: {d16:.0f}", fontsize=13, color="#555555")
    axC.set_ylim(bottom=d16 - 78)
    axC.set_yticks(range(350, 601, 50))
axC.set_ylabel("decode throughput (tok/s)")
axC.set_title("Throughput vs KV budget", fontweight="bold")
h, l = axC.get_legend_handles_labels()
axC.legend(h[::-1], l[::-1], frameon=True, facecolor="white", edgecolor="none", framealpha=1.0, fontsize=13.5, loc="lower center", ncol=2, columnspacing=1.1, handlelength=1.6)
axC.grid(axis="y", alpha=0.25)

if HAVE_D:
    ms = [m for m in order if m in isomem]
    axD.barh(range(len(ms)), [isomem[m][1] for m in ms], color=[C[m] for m in ms], height=0.62)
    for y, m in enumerate(ms):
        if m == "dense":
            continue                      # callout below carries dense's numbers
        b, tp, _ = isomem[m]
        axD.text(tp + 90, y, f"{tp:,.0f}  (bs {b})", va="center", fontsize=14,
                 fontweight="bold", color=C[m])
    axD.set_yticks(range(len(ms))); axD.set_yticklabels([])
    axD.set_xlabel("serving throughput (tok/s)")
    axD.set_title("Iso-memory serving (143 GB)", fontweight="bold")
    axD.set_xlim(0, max(isomem[m][1] for m in ms) * 1.42)
    if "dense" in isomem and "random_pp" in isomem:
        rr = isomem["random_pp"][1] / isomem["dense"][1]
        axD.text(isomem["dense"][1] + 90, ms.index("dense"),
                 f"{isomem['dense'][1]:,.0f} (bs {isomem['dense'][0]})  ← {rr:.0f}× at equal memory",
                 va="center", fontsize=15, fontweight="bold", color="#e34948")

if not PAPER:   # in the paper this provenance text lives in the caption/appendix
    fig.text(0.5, -0.045,
             "Protocol = VaSE benchmark/run_thpt.sh: identical fixed batch (16) for every method incl. full attention; input 128; "
             "faithful n_large=K/4; peak mem = torch.cuda.max_memory_allocated. All panels measured on one H200 (143 GB). "
             "Iso-memory panel: each method at its largest batch that fits (dense probed 30->OOM, 28 fits; evictors bisected). "
             "Aside: full attention at 32k (80.8 GB) does not fit VaSE's own 80 GB A100 (their script: '# dense OOM').",
             ha="center", fontsize=8.5, color="#555555")

out = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/figures/fig_throughput"
fig.savefig(out + ".png", dpi=170, bbox_inches="tight")
fig.savefig(out + ".pdf", bbox_inches="tight")
print(f"wrote {out}.png / .pdf  (proto rows: {len(proto)}, ksweep pts: {len(ksweep)})")
