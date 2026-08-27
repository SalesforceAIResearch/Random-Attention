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
"""Derived per-dataset efficiency (option a): no new GPU runs.
Per (task@4x, method):  GPU-time/problem = gen_tokens / aggregate tok/s   (bs amortizes out),
computed with BOTH median and mean gen length, plus  correct answers per GPU-hour
= 3600 / mean_time * flag_acc.
tok/s from the K-sweep at each task's 4x budget (MATH K=1024, GPQA K=2048, AIME/HMMT K=4096);
gen lengths from other_info generate_lens; accuracy from regrade32k_results.tsv (flag_acc).
Dense tok/s: 16k-generation row (repair) if present, else the 32k row (footnote: overestimates
dense cost on short tasks). Usage: python figures/effective_throughput.py
"""
import csv, glob, json, os, statistics as st

ROOT = os.environ.get("RA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RT = os.environ.get("RA_ENGINE", f"{ROOT}/kvcompress/harness")
TASK_K = {"math": "1024", "gpqa": "2048", "aime": "4096", "hmmt": "4096"}
METHODS = ["dense", "random_pp", "vase", "snapkv", "caote", "triattn"]
# gen-length source dirs per (task, method) — first glob that yields other_info wins
DIRS = {
 ("math","dense"):    [f"{RT}/gate2/Qwen3-4B/dense/math_*"],
 ("math","random_pp"):[f"{RT}/gate2/Qwen3-4B/random_pp/math_*"],
 ("math","vase"):     [f"{RT}/gate2/Qwen3-4B/vase_faithful/math_*", f"{RT}/gate2/Qwen3-4B/vase_faithful/*/math_*",
                       f"{RT}/gate2/Qwen3-4B/range_sink_sample_attn/math_*", f"{RT}/gate2/Qwen3-4B/math_K1024/vase_faithful"],
 ("math","snapkv"):   [f"{RT}/gate2/Qwen3-4B/attn/math_*"],
 ("math","caote"):    [f"{RT}/gate2/Qwen3-4B/caote/math_*"],
 ("math","triattn"):  [f"{RT}/gate2/Qwen3-4B/math_K1024/triattn_ph"],
}
for t, base in (("gpqa","gpqa_K2048"), ("aime","aime_K4096"), ("hmmt","hmmt_K4096")):
    DIRS[(t,"dense")]     = [f"{RT}/gate2/Qwen3-4B/{base}/dense", f"{RT}/gate2/Qwen3-4B/dense/{t}_*"]
    DIRS[(t,"random_pp")] = [f"{RT}/gate2/Qwen3-4B/{base}/random_pp"]
    DIRS[(t,"vase")]      = [f"{RT}/gate2/Qwen3-4B/{base}/vase_faithful"]
    DIRS[(t,"snapkv")]    = [f"{RT}/gate2/Qwen3-4B/{base}/attn"]
    DIRS[(t,"caote")]     = [f"{RT}/gate2/Qwen3-4B/{base}/caote"]
    DIRS[(t,"triattn")]   = [f"{RT}/gate2/Qwen3-4B/{base}/triattn_ph"]
ACC_METHOD = {"vase": "vase_faithful", "snapkv": "attn", "triattn": "triattn_ph"}

def toks():  # (method,K)->tok_s ; K1024 from protocol 16k rows (ksweep TSV relabel fixed in-place 07-09)
    out = {}
    with open(f"{ROOT}/logs/throughput_ksweep.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            try: out[(r["method"], r["K"])] = float(r["tok_s"])
            except ValueError: pass
    dense16 = None
    with open(f"{ROOT}/logs/throughput_vase_protocol.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            try: v = float(r["tok_s"])
            except ValueError: continue
            if r["output_len"] == "16384":
                if r["method"] == "dense": dense16 = v
                else: out.setdefault((r["method"], "1024"), v)
            if r["method"] == "dense" and r["output_len"] == "32768": out.setdefault(("dense","32k"), v)
    for k in TASK_K.values():
        out[("dense", k)] = dense16 if dense16 else out.get(("dense", "32k"))
    return out, dense16 is not None

def lens(task, method):
    for pat in DIRS.get((task, method), []):
        vals = []
        for d in glob.glob(pat):
            for f in glob.glob(f"{d}/run_*/other_info*.json") + glob.glob(f"{d}/*/run_*/other_info*.json"):
                if f.endswith("other_info.json") and glob.glob(os.path.dirname(f) + "/other_info_shard*.json"):
                    continue                       # avoid double count when both merged+shards exist
                try: vals += [x for x in json.load(open(f)).get("generate_lens", []) if isinstance(x,(int,float))]
                except Exception: pass
        if vals:
            return st.median(vals), st.mean(vals), len(vals)
    return None, None, 0

def accs():
    out = {}
    with open(f"{ROOT}/regrade32k_results.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if "Qwen3-4B" not in r["base"]: continue
            base = r["base"].rstrip("/")
            task = ("math" if base.endswith("Qwen3-4B") and r["data"]=="math"
                    else next((t for t,b in (("gpqa","gpqa_K2048"),("aime","aime_K4096"),("hmmt","hmmt_K4096")) if base.endswith(b)), None))
            if not task: continue
            try: a, n = float(r["flag_acc"]), int(r["n"])
            except ValueError: continue
            out.setdefault((task, r["method"]), []).append((a, n))
    return {k: sum(a*n for a,n in v)/sum(n for a,n in v) for k, v in out.items()}

T, have_dense16 = toks(); A = accs()
# fallback flag_acc from REPORT §2 (cells the regrade TSV lacks); AIME = mean(aime25, aime26)
for k, v in {("math","vase_faithful"): 0.809, ("math","triattn_ph"): 0.856,
             ("gpqa","dense"): 0.562, ("gpqa","vase_faithful"): 0.461, ("gpqa","triattn_ph"): 0.523,
             ("hmmt","vase_faithful"): 0.417}.items():
    A.setdefault(k, v)
print(f"{'task':6} {'method':10} {'tok/s':>7} {'med-len':>8} {'mean-len':>9} {'acc':>6} | {'min/prob(med)':>13} {'min/prob(avg)':>13} {'correct/GPU-h':>13}")
md = ["| task | method | tok/s | median gen | mean gen | acc | min/prob (med) | min/prob (avg) | correct/GPU-h |",
      "|---|---|---|---|---|---|---|---|---|"]
for task in TASK_K:
    for m in METHODS:
        ts = T.get((m, TASK_K[task])); med, mean, n = lens(task, m)
        acc = A.get((task, ACC_METHOD.get(m, m)))
        if not ts or not med:
            print(f"{task:6} {m:10} {'—':>7}   (missing: {'tok/s' if not ts else ''}{'gen-lens' if not med else ''})"); continue
        tmed, tavg = med/ts/60, mean/ts/60
        cph = (3600/(mean/ts))*acc if acc else None
        star = "*" if m == "dense" and not have_dense16 else ""
        print(f"{task:6} {m:10} {ts:7.1f} {med:8.0f} {mean:9.0f} {acc if acc else 0:6.3f} | {tmed:13.2f} {tavg:13.2f} {cph if cph else 0:13.1f}{star}")
        md.append(f"| {task} | {m} | {ts:.0f}{star} | {med:.0f} | {mean:.0f} | {acc:.3f} | {tmed:.2f} | {tavg:.2f} | **{cph:.1f}** |" if cph else
                  f"| {task} | {m} | {ts:.0f}{star} | {med:.0f} | {mean:.0f} | — | {tmed:.2f} | {tavg:.2f} | — |")
    print()
    md.append("| | | | | | | | | |")
if not have_dense16:
    note = "*dense tok/s = 32k-generation row (16k repair pending) -> dense per-task cost OVERestimated on short tasks."
    print(note); md.append(f"\n{note}")
md.append("""
**Methodology (provenance — do not mistake for batched-eval tok/s).** tok/s = VaSE's official
throughput protocol (`benchmark/throughput.py`): dedicated GPU per method, fixed bs=16,
input_len=128, output_len=16384 (dense row: 16k), warmup + 2 timed runs, K at each task's 4x
budget, nl=K/4 (sources: `logs/throughput_vase_protocol.tsv`, `logs/throughput_ksweep.tsv`).
Gen lengths = eval-run `generate_lens` (valid token counts); flag_acc = `regrade32k_results.tsv`.
Caveats: (1) triattn tok/s reflects our unfused Python cross-layer scoring, NOT a fused kernel —
do not report as TriAttention's true speed; (2) bench prompts are 128 tokens vs longer eval
prompts, and dense uses the 16k-output row even on HMMT (mean gen ~18-20k) -> dense tok/s slightly
OVERestimated there, i.e. dense correct/GPU-h is flattered = conservative for the eviction rows.""")
open(f"{ROOT}/figures/effective_throughput.md", "w").write("\n".join(md) + "\n")
print(f"\nwrote {ROOT}/figures/effective_throughput.md")
