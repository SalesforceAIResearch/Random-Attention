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
"""Feasibility overlay (phi-4-reasoning, MATH): measured rp(K) accuracy vs the prompt-fit
prediction. The prompt-fit criterion: random_pp protects min(prompt_len, K_sel-1) prompt
tokens with K_sel = K - residual(64); when a problem's prompt does NOT fit, the question
itself is scattered and the trace degenerates (REPORT §prompt-fit / MATH-16x correction).
Predicted feasible fraction at K = P(prompt_len + margin <= K_sel), margin=0 (bare) drawn
with the measured rp, tri-F (tracks rp: no prompt protection rule survives infeasibility),
and VaSE (survives via implicit attention on the question) curves. GPU-free replot."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.environ.get("RA_ENGINE", os.path.join(os.environ.get("RA_ROOT", "."), "kvcompress/harness"))
OUT = os.path.dirname(os.path.abspath(__file__))
RESID = 64

MEASURED = {  # K -> flag_acc, from regrade32k_results_fresh.tsv (same-day sweep rows, R2 n=1000)
    "random_pp":          {256: 0.059, 320: 0.118, 384: 0.630, 448: 0.753, 512: 0.810, 1024: 0.910, 2048: 0.935},
    "triattn_ph_memofix": {256: 0.060, 448: 0.710, 512: None, 1024: 0.891, 2048: 0.921},
    "vase_faithful":      {256: 0.605, 448: 0.728, 512: 0.780, 2048: 0.893},
}


def prompt_lens():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.join(os.environ.get("RA_MODELS_DIR", "models"), "phi-4-reasoning"))
    data = [json.loads(l) for l in open(os.path.join(BASE, "data/math/test.jsonl"))]
    lens = []
    for d in data:
        q = d["problem"].strip() if "problem" in d else d["question"].strip()
        msg = [{"role": "user", "content": q + "\nPlease reason step by step, and put your final answer within \\boxed{}."}]
        enc = tok.apply_chat_template(msg, tokenize=True, add_generation_prompt=True, return_dict=True)
        lens.append(len(enc["input_ids"]))
    return sorted(lens)


def main():
    lens = prompt_lens()
    n = len(lens)
    ks = list(range(192, 2304, 16))
    pred = [sum(1 for L in lens if L <= k - RESID - 1) / n for k in ks]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(ks, pred, color="0.35", lw=2, ls="--",
            label="predicted feasible fraction  P(prompt fits in $K_{sel}$)")
    styles = {"random_pp": ("tab:blue", "o", "random_pp (measured)"),
              "triattn_ph_memofix": ("tab:red", "s", "TriAttention-F (measured)"),
              "vase_faithful": ("tab:green", "^", "VaSE (measured)")}
    for m, (c, mk, lab) in styles.items():
        pts = sorted((k, v) for k, v in MEASURED[m].items() if v is not None)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=c, marker=mk, lw=1.5, label=lab)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("token budget K (log scale)")
    ax.set_ylabel("accuracy / fraction")
    ax.set_title("phi-4-reasoning MATH: the collapse is prompt-fit infeasibility, not selection")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig_feasibility_phi_math.{ext}"), dpi=200)
    with open(os.path.join(OUT, "fig_feasibility_phi_math_data.tsv"), "w") as f:
        f.write("K\tpred_feasible\n")
        for k, p in zip(ks, pred):
            f.write(f"{k}\t{p:.4f}\n")
    q = [lens[int(n * x)] for x in (0.05, 0.5, 0.95)]
    print(f"prompt tokens: n={n} p5={q[0]} median={q[1]} p95={q[2]} max={lens[-1]}")
    print("saved fig_feasibility_phi_math.{png,pdf} + data tsv")


if __name__ == "__main__":
    main()
