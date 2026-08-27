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
"""S1/S1b/S4 forced-decode runner (REGISTERED 07-23; spec SYNTH_36H_SCHEDULE.md).

Teacher-forces the scripted sequence through the SAME manual decode loop the accuracy harness
uses (generation_utils.batch_exist_generate's structure: EvictCache built directly +
model(step, past_key_values=cache, logits_to_keep=1)). Every scripted token enters via the
true decode path — identical eviction semantics to all accuracy runs, no equivalence argument.
Per forced step we record the model's fp32 logprob of the forced token BEFORE feeding it
(value-token steps = primary metric). After the script: 32 free greedy tokens = secondary.

Usage:
  run_synth.py --cell results/synth_tasks/a1536 --arm h1_head3 \
      --eviction_mode synth_pin --spec '[{"span":"x","heads":[3],"layers":"all"}]' \
      --n 250 --bs 32 --gpu 0
  --spec entries use "span": "x"|"y"; lo/hi filled from the cell's meta.json.
  --eviction_mode range_sink_sample_attn|attn (S4 selector arms; no --spec) or synth_pin
  ('natural-rp' == synth_pin with --spec '[]': unprotected per-head uniform random).
  --parity: inert budget (65536) + plain-forward reference LP; asserts |diff| < tol.

Output: <out>/<cell>/<arm>.jsonl: {"i","lp_toks","lp_sum","freegen","fg_match"} per instance
+ <arm>.done marker. SYNTH_AUDIT_DIR gets per-event span-retention counts (h verification).
"""
import argparse
import gc
import json
import os
import sys

import torch

sys.path.insert(0, os.environ.get("RA_ENGINE", os.path.join(os.environ.get("RA_ROOT", "."), "kvcompress/harness"))); sys.path.insert(0, os.environ.get("RA_ROOT", "."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--eviction_mode", required=True)
    ap.add_argument("--spec", default=None)
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--n0", type=int, default=0)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--model_dir", default=os.path.join(os.environ.get("RA_MODELS_DIR", "models"), "Qwen3-4B"))
    ap.add_argument("--out", default=os.path.join(os.environ.get("RA_ROOT", "."), "results/synth_runs"))
    ap.add_argument("--token_budget", type=int, default=1088)
    ap.add_argument("--n_large", type=int, default=256)     # K/4 for the faithful vase arm
    ap.add_argument("--free_tokens", type=int, default=32)
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--seed", type=int, default=None,
                    help="registered per-arm eviction-RNG seed (S3 amendment; recorded in .done)")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    meta = json.load(open(os.path.join(args.cell, "meta.json")))
    inst = torch.load(os.path.join(args.cell, "instances.pt"))[args.n0:args.n]
    stub = torch.tensor(meta["stub_ids"], dtype=torch.long)
    tb = 8192 if args.parity else args.token_budget   # parity: inert (> every cell total) but small
    assert tb % meta["res"] == 0
    assert args.eviction_mode in ("synth_pin", "range_sink_sample_attn", "attn", "triattn_ph", "attn_rkv"), \
        "registered arms only (natural-rp = synth_pin with spec [])"
    if args.eviction_mode == "triattn_ph":
        os.environ["TRIATTN_STATS"] = os.path.join(os.environ.get("RA_ENGINE", os.path.join(os.environ.get("RA_ROOT", "."), "kvcompress/harness")), "stats_pl/qwen3-4b.pt")

    if args.eviction_mode == "synth_pin":
        spec = json.loads(args.spec or "[]")
        for s in spec:
            base_lo, base_hi = meta["span_" + s.pop("span")]
            if "tok" in s:                       # single-token entry (fragmentation conditions)
                t = int(s.pop("tok"))
                s["lo"], s["hi"] = base_lo + t, base_lo + t + 1
            else:
                s["lo"], s["hi"] = base_lo, base_hi
        os.environ["SYNTH_SPEC"] = json.dumps(spec)
        os.environ.setdefault("SYNTH_AUDIT", "1")
        adir = os.path.join(args.out, os.path.basename(args.cell.rstrip("/")), args.arm + "_audit")
        os.makedirs(adir, exist_ok=True)
        os.environ["SYNTH_AUDIT_DIR"] = adir

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from kvcompress.engine.modify_forward import wrap_evict_attn_forward
    from kvcompress.engine.cache_utils import EvictCache
    wrap_evict_attn_forward(args.model_dir)
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="flash_attention_2")
    model.eval()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    cache_config = {"token_budget": tb, "residual_length": meta["res"],
                    "rkv_lambda": 0.5, "eviction_mode": args.eviction_mode,
                    "smooth": args.eviction_mode in ("range_sink_sample_attn", "attn"),
                    "n_large": args.n_large, "temperature": 1.0, "verbose": False}

    od = os.path.join(args.out, os.path.basename(args.cell.rstrip("/")))
    os.makedirs(od, exist_ok=True)
    fo = open(os.path.join(od, f"{args.arm}.jsonl"), "w")
    Lf = meta["L_forced"]
    NV = meta.get("n_val_tokens", 4)
    eos = tok.eos_token_id

    for b0 in range(0, len(inst), args.bs):
        batch = inst[b0:b0 + args.bs].cuda()                 # [B, Lf] forced ids
        B = batch.shape[0]
        cache = EvictCache(config=model.config, **cache_config)
        cur = stub.view(1, -1).expand(B, -1).cuda()          # prefill stub
        lp = torch.zeros(B, Lf, dtype=torch.float32, device="cuda")
        free = torch.full((B, args.free_tokens), eos, dtype=torch.long, device="cuda")
        gok = torch.zeros(B, NV, dtype=torch.bool, device="cuda")   # argmax-correct per value token
        with torch.inference_mode():
            for t in range(Lf + args.free_tokens):
                out = model(cur, past_key_values=cache, use_cache=True, logits_to_keep=1)
                logits = out.logits[:, -1, :].float()
                cache = out.past_key_values
                if t < Lf:                                    # record LP of forced token, then force it
                    ls = torch.log_softmax(logits, dim=-1)
                    lp[:, t] = ls.gather(-1, batch[:, t:t + 1]).squeeze(-1)
                    if t >= Lf - NV:                             # greedy-correct indicator at value steps:
                        gok[:, t - (Lf - NV)] = (logits.argmax(-1) == batch[:, t])
                    cur = batch[:, t:t + 1]
                else:                                         # free greedy tail
                    nxt = logits.argmax(-1, keepdim=True)
                    free[:, t - Lf] = nxt.squeeze(-1)
                    cur = nxt
        for j in range(B):
            i = args.n0 + b0 + j
            v = meta["values"][i]
            fj = free[j].tolist()
            if eos in fj:
                fj = fj[:fj.index(eos)]
            fg = tok.decode(fj, skip_special_tokens=True)
            tgt = str(v.get("target", v["vx"]))
            rec = {"i": i,
                   "lp_toks": [float(lp[j, p]) for p in range(Lf - NV, Lf)],
                   "lp_sum": float(lp[j, Lf - NV:Lf].sum()),
                   # fg_ok == greedy free-gen exact match: all-NV argmax-correct under forced
                   # correct prefixes (greedy would feed the same tokens => identical distrib.)
                   "fg_ok": bool(gok[j].all()),
                   # per-value-token argmax correctness (S3 amendment 07-31): enables per-fact
                   # accuracy on multi-value probes (s2r: toks 0-3 = x, 6-9 = y)
                   "gok": [bool(g) for g in gok[j].tolist()],
                   "freegen": fg[:64], "fg_match": fg.strip()[:len(tgt)] == tgt}
            if args.parity:
                # reference = ONE-SHOT prefill under the same wrapped kernel with an inert
                # cache (tb=8192 > total, zero evictions); plain model(full) is impossible —
                # the patched attention requires an EvictCache.
                full = torch.cat([stub, inst[i - args.n0]]).view(1, -1).cuda()
                ref_cache = EvictCache(config=model.config, **cache_config)
                with torch.inference_mode():
                    lg = model(full, past_key_values=ref_cache, use_cache=True,
                               logits_to_keep=NV + 1).logits.float()
                del ref_cache
                ls = torch.log_softmax(lg[0, -NV - 1:-1], dim=-1)
                ref = float(sum(ls[k, full[0, -NV + k]] for k in range(NV)))
                rec["lp_ref"] = ref
                rec["parity_ok"] = bool(abs(ref - rec["lp_sum"]) < 0.25)  # bf16 kernel tolerance
            fo.write(json.dumps(rec) + "\n")
        fo.flush()
        # hard per-batch teardown: the old EvictCache (5.2GB) is otherwise retained across
        # batches (measured: +1-3GB/batch -> OOM at ~40GB); reserved returned to the driver
        del out, cache, logits, lp, free, gok
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[{args.arm}] {b0 + B}/{len(inst)}  mem={torch.cuda.memory_allocated()/2**30:.1f}GB", flush=True)
    fo.close()
    import socket
    import subprocess
    try:
        sha = subprocess.check_output(
            ["git", "-C", os.environ.get("RA_ROOT", "."), "rev-parse", "--short", "HEAD"],
            text=True).strip()
    except Exception:
        sha = "unknown"
    json.dump({"arm": args.arm, "mode": args.eviction_mode, "spec": args.spec,
               "tb": tb, "n": len(inst), "bs": args.bs, "seed": args.seed,
               "host": socket.gethostname(), "gpu": torch.cuda.get_device_name(0),
               "git": sha}, open(os.path.join(od, f"{args.arm}.done"), "w"))


if __name__ == "__main__":
    main()
