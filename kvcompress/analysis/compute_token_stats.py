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
"""Per-method generation-length (token) statistics from other_info*.json (generate_lens).
Aggregates the matched-budget base cells; reports median / mean / p90 / cap% / tok-s per method.
Usage: python compute_token_stats.py [model] [task]   (defaults: Qwen3-4B math). 'all' for every model/task."""
import json, glob, os, sys, statistics as st

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen3-4B"
TASK  = sys.argv[2] if len(sys.argv) > 2 else "math"
CAP = 32768 - 1500   # >= this = hit the 32k generation cap (never terminated)
# eviction-mode dir name -> report label
NAME = {"dense": "dense", "random_pp": "random_pp", "range_sink_sample_attn": "vase",
        "vase_faithful": "vase", "attn": "snapkv", "caote": "caote", "triattn_ph": "triattn"}
ORDER = ["dense", "random_pp", "vase", "snapkv", "caote", "triattn"]

def cells(model, task):
    """yield (label, generate_lens list, batch_times list) for base cells of this model/task."""
    acc = {}
    for oi in glob.glob(f"gate2/{model}/**/other_info*.json", recursive=True):
        parts = oi.split("/")
        # task_subdir is 3 up from other_info (…/<method>/<task_subdir>/run_N/other_info)
        sub = parts[-3]
        mdir = parts[-4]                      # method dir (old layout) OR K-dir (new)
        t = sub.split("_bs")[0]
        if t != task: continue
        meth = NAME.get(mdir) or NAME.get(mdir.split("/")[-1])
        if meth is None:  # new layout: method is one level deeper; mdir may be the K-dir
            continue
        # only the matched base budget (4x): infer from task
        base = {"math": "1024", "gpqa": "2048", "aime": "4096", "hmmt": "4096"}.get(task)
        if f"budget={base}" not in sub and "dense" not in sub: continue
        try: o = json.load(open(oi))
        except Exception: continue
        gl = o.get("generate_lens", []); bt = o.get("batch_times", [])
        a = acc.setdefault(meth, [[], []]); a[0] += gl; a[1] += bt
    return acc

def report(model, task):
    acc = cells(model, task)
    if not acc: print(f"[{model} {task}] no other_info cells found"); return
    print(f"\n### {model} — {task}  (matched 4× budget; gen-length = output tokens)")
    print(f"{'method':10s} {'n':>5} {'median':>7} {'mean':>7} {'p90':>6} {'cap%':>6} {'tok/s':>7}")
    for meth in ORDER:
        if meth not in acc: continue
        gl, bt = acc[meth]
        if not gl: continue
        gl_s = sorted(gl); med = st.median(gl); mean = sum(gl)/len(gl)
        p90 = gl_s[int(.9*len(gl_s))-1]; cap = 100*sum(1 for x in gl if x >= CAP)/len(gl)
        toks = (sum(gl)/sum(bt)) if bt and sum(bt) > 0 else float("nan")
        print(f"{meth:10s} {len(gl):5d} {med:7.0f} {mean:7.0f} {p90:6.0f} {cap:6.1f} {toks:7.1f}")

if MODEL == "all":
    for m in ["Qwen3-4B", "Qwen3-14B", "Qwen3-32B", "phi-4-reasoning", "DeepSeek-R1-Distill-Llama-8B"]:
        for t in ["math", "gpqa", "aime", "hmmt"]: report(m, t)
else:
    report(MODEL, TASK)
