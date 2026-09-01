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
"""S1/S1b/S4 task builder — emits token-id instance files per cell (single tokenization
source of truth; REGISTERED.md 07-23 freeze; spec SYNTH_36H_SCHEDULE.md).

Design (frozen): NO protection rule anywhere. A short prefill stub (chat template + <think>
opener) is the only prefilled content; needle(s), filler, and query all enter through the
DECODE path (teacher-forced), so eviction treats them exactly as generated reasoning.
The query is terminal — always inside the W=64 recency buffer at measurement.

Per instance (all lengths cell-constant so absolute positions are cell-constant):
    [stub (prefill)] || filler_pre | NEEDLE | filler_mid | (NEEDLE_Y for S1b) | filler_post
                     | QUERY | VALUE(2 tok, measurement) ... +32 free-gen tail at runtime

Ages are denominated in EVENT COUNT E (registered): E = number of eviction events between the
needle exiting the recency buffer and measurement; synthetic survival s(E) = (1024/1088)^E.

Outputs per cell under --out/<cell>/:
    instances.pt   LongTensor [n, L_forced] forced token ids (stub NOT included)
    meta.json      stub_ids, span_x/span_y absolute [lo,hi), value token ids & positions,
                   L_forced, E_x, ages, per-instance value strings

Self-checks (abort on failure): detok(span)==needle text; value encodes to exactly 2 tokens;
all instances same length; needle absent from filler text (string screen on variable AND value).
"""
import argparse
import json
import os
import random
import re
import sys

import torch
from transformers import AutoTokenizer

TB, RES = 1088, 64          # token_budget (K=1024 + W=64), residual_length

VARS = ["zq", "kx", "vb", "jw", "qm", "xr", "wn", "pz"]


def harvest_filler(paths, tok, need_tokens, seed):
    """Pool of reasoning token streams sliced from dense MATH completions (post-<think>)."""
    rng = random.Random(seed)
    texts = []
    for p in paths:
        for line in open(p):
            try:
                c = json.loads(line)["completion"]
            except Exception:
                continue
            i = c.find("<think>")
            if i < 0:
                continue
            body = c[i + 8: i + 8 + 20000]
            if len(body) > 2000:
                texts.append(body)
        if sum(len(t) for t in texts) > 80 * need_tokens:  # ~4 chars/token headroom
            break
    rng.shuffle(texts)
    return texts


def build_filler_ids(texts, tok, n_tok, forbid, rng):
    """Concatenate screened reasoning text, tokenize, and cut to exactly n_tok tokens."""
    # chars-per-token differs by tokenizer (Qwen3 ~3.5 on math text, Phi-4 ~4.5+), so grow the buffer
    # until the TOKENIZED length covers n_tok instead of assuming a fixed chars budget.
    buf, have, ids, tries = [], 0, [], 0
    while len(ids) < n_tok:
        while have < n_tok * 5 or not buf:
            t = texts[rng.randrange(len(texts))]
            if any(f in t for f in forbid):
                tries += 1
                if tries > 10000:
                    raise RuntimeError("filler pool exhausted (screen rejects everything)")
                continue
            buf.append(t)
            have += len(t)
        ids = tok("".join(buf), add_special_tokens=False).input_ids
        have = 0 if len(ids) < n_tok else have   # short: draw more text and re-tokenize
    return ids[:n_tok]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=os.path.join(os.environ.get("RA_MODELS_DIR", "models"), "Qwen3-4B"))
    ap.add_argument("--filler_glob", default=os.path.join(os.environ.get("RA_ROOT", "."), "results/Qwen3-4B/dense/math_bs2_dense/run_0/completions_shard*.jsonl"))
    ap.add_argument("--out", default=os.path.join(os.environ.get("RA_ROOT", "."), "results/synth_tasks"))
    ap.add_argument("--n", type=int, default=500)   # max n; nested prefixes 250/400/500
    ap.add_argument("--seed", type=int, default=20260723)
    ap.add_argument("--minimal_stub", action="store_true",
                    help="build the prefill stub as a bare user/assistant turn (no system prompt) -- needed for "
                         "models whose default template injects a long system prompt (Phi-4-reasoning: 241 tokens); "
                         "the stub must stay under one residual chunk (64 tokens)")
    ap.add_argument("--force_out", action="store_true",
                    help="override the clobber guard; only for a deliberate rebuild of the SAME model's tasks")
    args = ap.parse_args()

    # CLOBBER GUARD (added 08-03). --out defaults to the QWEN task dir, and grading is POSITIONAL
    # against these instance files: overwriting them silently invalidates every registered §5 Qwen
    # number (a1536, a1536_s2r, a1536_wakeR, a1536_wakeD) with no error and no way to tell after the
    # fact. A cross-model build (e.g. phi-4-reasoning) MUST write elsewhere. Refuse to write a
    # non-Qwen build into an existing task dir unless --force_out is given.
    import os as _os, glob as _glob
    # The instance files live in PER-CELL SUBDIRS (<out>/a1536/instances.pt, <out>/a1536_wakeD/...),
    # NOT at <out>/instances.pt -- an earlier version of this guard checked the latter and could
    # therefore never fire. Glob one level down.
    _existing = _glob.glob(_os.path.join(args.out, "*", "instances.pt"))
    _is_qwen = "Qwen3-4B" in args.model_dir
    if (not _is_qwen) and _existing and not getattr(args, "force_out", False):
        raise SystemExit(
            f"REFUSING TO CLOBBER {args.out}: it holds {len(_existing)} instances.pt built for a different\n"
            f"model ({[_os.path.basename(_os.path.dirname(x)) for x in _existing][:6]}...) and\n"
            f"every registered §5 number was graded against it (grading is positional).\n"
            f"Pass --out <new dir>, e.g. --out results/synth_tasks_phi4")

    import glob
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    paths = sorted(glob.glob(args.filler_glob))
    assert paths, f"no filler files at {args.filler_glob}"

    # ---- stub (prefill): chat template + <think> opener --------------------------------
    msgs = [{"role": "user", "content":
             "Work through a long chain of small reasoning steps. Track every defined variable."}]
    if args.minimal_stub:
        # Phi-4-style tags, taken from the tokenizer's own template output (user turn + assistant opener)
        full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        i = full.rfind("<|im_start|>user")
        assert i >= 0, "minimal_stub assumes <|im_start|>user ... <|im_start|>assistant<|im_sep|> tags"
        stub_text = full[i:]
    else:
        stub_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    stub_ids = tok(stub_text, add_special_tokens=False).input_ids
    from transformers import AutoConfig
    n_kv = int(AutoConfig.from_pretrained(args.model_dir).num_key_value_heads)
    assert len(stub_ids) < 64, f"stub too long ({len(stub_ids)}) — must stay below one residual chunk"

    # ---- cells: (name, filler_pre, age_tokens[needle-end -> measurement]) ---------------
    # Ages -> registered event counts: total_len ~ stub+pre+span+age; E = floor((L-TB)/RES)+1 (>=0)
    AGES = {"a768f": (448, 768), "a1536": (448, 1536), "a3072": (448, 3072),
            "adeep": (448, 4224), "a1536_s1b": (448, 1536), "adeep_s4": (448, 4224),
            "a1536_wakeR": (448, 1536), "a1536_wakeD": (448, 1536),
            "a1536_fragv": (448, 1536),
            # 07-29 follow-up cells (registered post-suite, frozen before grading):
            # s2r = recall-both probe (no arithmetic; accuracy-bearing two-fact cell)
            # s2s = easy-sum probe (3-digit addends -> 4-digit sum; arithmetic feasible)
            "a1536_s2r": (448, 1536), "a1536_s2s": (448, 1536)}
    only = os.environ.get("BUILD_ONLY")
    if only:
        AGES = {k: v for k, v in AGES.items() if k in only.split(",")}
    rng = random.Random(args.seed)
    texts = harvest_filler(paths, tok, 6 * 10**6, args.seed)

    os.makedirs(args.out, exist_ok=True)
    summary = {}
    for cell, (pre_n, age) in AGES.items():
        crng = random.Random((args.seed, cell).__hash__() & 0xFFFFFFFF)
        inst, values, spans = [], [], None
        two_tok_vals = []
        # pre-screen candidate 4-digit values with EXACT 2-token encodings (leading nonzero)
        # Qwen3 digit-tokenizes numbers: 4-digit value == exactly 4 tokens, uniformly
        # (verified 07-23: {4: 1286/1286}). REGISTERED amendment: n_val_tokens 2 -> 4.
        # S1b sum uniformity: both values in [5000, 9999] so vx+vy is always 5 digits.
        # s2s uses 3-digit addends in [600, 999]: value = 3 tokens uniformly and the
        # sum is ALWAYS 4 digits (1200..1998) -> uniform n_val_tokens. 3-digit strings
        # keep the filler collision screen feasible (2-digit strings would starve it).
        cand = list(range(600, 1000)) if cell == "a1536_s2s" else list(range(5000, 10000))
        crng.shuffle(cand)
        if cell == "a1536_s2s":
            cand = cand * 4                      # 400 distinct values; pairs stay distinct per instance
        two_tok_vals = cand
        for i in range(args.n):
            var = "zq" if cell == "a1536_fragv" else VARS[i % len(VARS)]
            v_x = two_tok_vals[2 * i]
            v_y = two_tok_vals[2 * i + 1]
            if cell == "adeep_s4":
                needle = f"\nIMPORTANT: the passcode is {v_x}. Remember it.\n"
            else:
                needle = f"\nLet {var} = {v_x}.\n"
            if cell == "a1536_wakeR":
                needle_y = f"\nTo restate: {var} = {v_x} exactly.\n"     # verbatim-use wake
            elif cell == "a1536_wakeD":
                needle_y = f"\nSo {var} doubled gives {2 * v_x}.\n"      # derived-use wake
            else:
                needle_y = f"\nAlso let {var}2 = {v_y}.\n"
            if cell in ("a1536_s1b", "a1536_s2s"):
                query = f"\nNow I need both stored constants. {var} + {var}2 = "
            elif cell == "a1536_s2r":
                query = f"\nWait — I need both stored constants now. ({var}, {var}2) = ("
            elif cell == "adeep_s4":
                query = f"\nWhat was the passcode again? The passcode is "
            else:
                query = f"\nWait — I need the stored constants now. {var} = "
            n_ids = tok(needle, add_special_tokens=False).input_ids
            ny_ids = tok(needle_y, add_special_tokens=False).input_ids
            q_ids = tok(query, add_special_tokens=False).input_ids
            if cell == "a1536_s2r":                 # recall-both: measured region covers BOTH values
                target = f"{v_x}, {v_y}"
            elif cell in ("a1536_s1b", "a1536_s2s"):
                target = v_x + v_y
            else:
                target = v_x
            val_ids = tok(str(target), add_special_tokens=False).input_ids
            # Value token counts are TOKENIZER-dependent (Qwen3: one token per digit -> 4-digit = 4 tokens,
            # s2r "x, y" = 9-10; Phi-4: 4-digit = 2 tokens, s2r = 6). They must be constant within a cell
            # (grading is positional over the last NV forced tokens): fix them from instance 0, assert after.
            nvx_i = len(tok(str(v_x), add_special_tokens=False).input_ids)
            if i == 0:
                nv_cell, nv_x = len(val_ids), nvx_i
            assert len(val_ids) == nv_cell and nvx_i == nv_x, (cell, target, len(val_ids), nvx_i)
            assert tok.decode(n_ids).find(str(v_x)) > 0
            forbid = [f" {var} ", str(v_x), str(v_y), f"{var}2", "passcode"]
            # Constant-size boxes: needle token counts vary per instance (vars/digits tokenize
            # differently), so each needle sits in a fixed NB-token box padded with in-box filler
            # (span pins the whole box: needle + <=4 neutral tokens; budget cost <=24/1024 slots).
            # Query-length variation is absorbed by the post filler so totals stay cell-constant.
            NB = 16
            assert len(n_ids) <= NB and len(ny_ids) <= NB
            pad = build_filler_ids(texts, tok, 2 * NB, forbid, crng)
            nbox = n_ids + pad[:NB - len(n_ids)]
            nybox = ny_ids + pad[NB:NB + (NB - len(ny_ids))]
            pre = build_filler_ids(texts, tok, pre_n, forbid, crng)
            # y-needle sits 256 tokens after x (S1b uses it; other cells carry it inertly
            # so ALL cells share one template family -> spans cell-constant, arms differ only in SYNTH_SPEC)
            mid = build_filler_ids(texts, tok, 256, forbid, crng)
            QBUDGET = 40
            assert len(q_ids) <= QBUDGET
            post_n = age - NB - 256 - NB - QBUDGET
            post = build_filler_ids(texts, tok, post_n + (QBUDGET - len(q_ids)), forbid, crng)
            seq = pre + nbox + mid + nybox + post + q_ids + val_ids
            base = len(stub_ids)
            lo_x = base + len(pre); hi_x = lo_x + NB
            lo_y = base + len(pre) + NB + 256; hi_y = lo_y + NB
            if spans is None:
                spans = (lo_x, hi_x, lo_y, hi_y, len(seq))
                if cell == "a1536_fragv":
                    pre_txt = tok(f"\nLet {var} = ", add_special_tokens=False).input_ids
                    val_off = len(pre_txt)          # value digits at box offsets [val_off, val_off+4)
            
            assert spans == (lo_x, hi_x, lo_y, hi_y, len(seq)), "cell lengths must be constant"
            assert str(v_x) in tok.decode(seq[lo_x - base:hi_x - base])
            inst.append(seq)
            values.append({"var": var, "vx": v_x, "vy": v_y, "target": target,
                           "val_ids": val_ids})
        L = spans[4]
        total = len(stub_ids) + L
        E = max(0, (total - TB) // RES + 1)
        d = os.path.join(args.out, cell)
        os.makedirs(d, exist_ok=True)
        torch.save(torch.tensor(inst, dtype=torch.long), os.path.join(d, "instances.pt"))
        meta = {"stub_ids": stub_ids, "L_forced": L, "total": total, "E": E,
                "span_x": [spans[0], spans[1]], "span_y": [spans[2], spans[3]],
                "n_val_tokens": nv_cell, "n_val_x": nv_x, "n_kv_heads": n_kv,
                "stub_minimal": bool(args.minimal_stub), "model_dir": args.model_dir,
                "value_pos": [len(stub_ids) + L - nv_cell, len(stub_ids) + L],
                "s_E": (1024 / 1088) ** E, "union_pred": 1 - (1 - (1024 / 1088) ** E) ** n_kv,
                "tb": TB, "res": RES, "values": values,
                **({"val_off": val_off} if cell == "a1536_fragv" else {})}
        json.dump(meta, open(os.path.join(d, "meta.json"), "w"))
        summary[cell] = {"total": total, "E": E, "s_E": round(meta["s_E"], 4),
                         "union_all_heads": round(meta["union_pred"], 4), "n_kv_heads": n_kv,
                         "n_val_tokens": nv_cell, "n_val_x": nv_x, "stub_tokens": len(stub_ids)}
        print(cell, summary[cell])
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
