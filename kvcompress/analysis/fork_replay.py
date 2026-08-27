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
"""fork_replay.py — GPU half of the fork autopsy (QUALITATIVE_WIN_PATTERNS §replay).

For one fork (cell, gi, run): teacher-force the ORIGINAL rp trace up to the token before the
dismissal (f_tok), with the rp eviction engine live (so the cache reaching that point has the
same distribution as during generation), then FREE-generate to the answer. Two conditions:
  control:   plain random_pp (span survives only by chance — for these forks P≈0)
  force:     FORCE_KEEP_RANGE=lo:hi set by the launcher — the engine pins the span's absolute
             positions in every head at matched budget (the L pinned tokens displace L random
             keeps; L=1-2 here).
If force-keeping flips the outcome distribution toward the gold answer, span loss CAUSED the
wrong turn; if not, the wrong turn is a generic reliability effect (the F8 null gate stands).

One process = one (cond, rep) replay on one GPU. The launcher fans (cond, rep) over GPUs.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fork_autopsy import CELLS, DATA, read_completion, boxed_letter  # noqa: E402

BASE = os.environ.get("RA_ENGINE", os.path.join(os.environ.get("RA_ROOT", "."), "kvcompress/harness"))
K_BUDGET, RESIDUAL = 2048, 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=list(CELLS))
    ap.add_argument("--gi", type=int, required=True)
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--f_tok", type=int, required=True)   # dismissal token index (prompt+gen coords)
    ap.add_argument("--cond", required=True, choices=["control", "force"])
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--seed0", type=int, default=20260815)
    ap.add_argument("--max_total", type=int, default=32768)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # cond consistency: launcher must set FORCE_KEEP_RANGE iff cond == force
    fkr = os.environ.get("FORCE_KEEP_RANGE")
    assert (args.cond == "force") == bool(fkr), f"cond={args.cond} but FORCE_KEEP_RANGE={fkr!r}"

    os.chdir(BASE)
    sys.path.insert(0, BASE)   # the `modified` eviction package resolves from the eval tree
    import torch
    from kvcompress.engine.modify_forward import wrap_evict_attn_forward
    cc = CELLS[args.cell]
    wrap_evict_attn_forward(cc["model"])
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from kvcompress.engine.cache_utils import EvictCache

    tok = AutoTokenizer.from_pretrained(cc["model"])
    dev = "cuda:0"
    model = AutoModelForCausalLM.from_pretrained(
        cc["model"], torch_dtype=torch.bfloat16, device_map=dev,
        attn_implementation="flash_attention_2").eval()

    data = [json.loads(l) for l in open(DATA)]
    q = data[args.gi]["question"].strip()
    enc = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=True,
                                  add_generation_prompt=True, return_dict=True)
    p_ids = enc["input_ids"]
    pl = len(p_ids)
    ptxt = tok.decode(p_ids, skip_special_tokens=True)
    full = read_completion(cc["rp_dir"], args.run, args.gi)
    assert full is not None, "trace missing"
    gen = full[len(ptxt):] if full.startswith(ptxt) else full[full.find(ptxt[-60:]) + 60:]
    g_ids = tok(gen, add_special_tokens=False)["input_ids"]
    n_force = args.f_tok - pl
    assert 0 < n_force <= len(g_ids), f"n_force={n_force} vs gen len {len(g_ids)}"
    prefix = g_ids[:n_force]

    torch.manual_seed(args.seed0 + 1000 * (args.cond == "force") + args.rep)
    cache = EvictCache(config=model.config, token_budget=K_BUDGET, residual_length=RESIDUAL,
                       eviction_mode="random_pp", smooth=True, n_large=K_BUDGET // 4,
                       rkv_lambda=None, temperature=0.6, verbose=False)

    ids = torch.tensor([p_ids], device=dev)
    am = torch.ones_like(ids)
    with torch.no_grad():
        out = model(ids, attention_mask=am, past_key_values=cache, use_cache=True,
                    logits_to_keep=1)
    cache = out.past_key_values

    def one(tid):
        nonlocal cache, am
        cur = torch.tensor([[tid]], device=dev)
        am = torch.cat([am, torch.ones((1, 1), device=dev, dtype=am.dtype)], 1)
        with torch.no_grad():
            o = model(cur, attention_mask=am, past_key_values=cache, use_cache=True,
                      logits_to_keep=1)
        cache = o.past_key_values
        return o.logits[:, -1, :]

    logits = None
    for t in prefix:                                   # teacher-forced dismissal prefix
        logits = one(t)

    new_toks, text_ids = 0, []
    eos = tok.eos_token_id
    max_new = args.max_total - pl - n_force
    while new_toks < max_new:
        probs = torch.softmax(logits.float() / 0.6, dim=-1)
        sp, si = torch.sort(probs, descending=True)
        cum = torch.cumsum(sp, dim=-1)
        sp[cum - sp > 0.95] = 0.0                      # top-p 0.95
        nxt = si.gather(-1, torch.multinomial(sp, 1)).item()
        if nxt == eos:
            break
        text_ids.append(nxt)
        new_toks += 1
        logits = one(nxt)

    cont = tok.decode(text_ids, skip_special_tokens=True)
    letter, _ = boxed_letter(cont)
    row = dict(cell=args.cell, gi=args.gi, run=args.run, cond=args.cond, rep=args.rep,
               f_tok=args.f_tok, fkr=fkr or "-", n_new=new_toks, letter=letter,
               correct=(letter == "A"), tail=cont[-400:])
    with open(args.out, "a") as f:
        f.write(json.dumps(row) + "\n")
    txtdir = os.path.splitext(args.out)[0] + "_texts"
    os.makedirs(txtdir, exist_ok=True)
    with open(os.path.join(txtdir, f"{args.cell}_gi{args.gi}_{args.cond}_r{args.rep}.txt"), "w") as f:
        f.write(cont)
    print(f"REPLAY {args.cell} gi{args.gi} {args.cond} rep{args.rep}: "
          f"letter={letter} correct={letter == 'A'} n_new={new_toks}", flush=True)


if __name__ == "__main__":
    main()
