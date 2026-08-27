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
# -*- coding: utf-8 -*-
"""
fork_autopsy.py -- mechanical stage of the qualitative WRONG-TURN fork study (GPQA K=2048 cells).

For every fork problem (rp-side wrong turns) and every flip-adjacent control, compute the
probability that the *consideration source span* was still present in the random_pp KV cache
(in >= 1 of the model's KV heads, within one layer) at the moment the dismissal token was
generated.

===========================================================================================
SURVIVAL MODEL -- derived from the engine, kvcompress/engine/cache_utils.py (verbatim
line references to kvcompress/engine/cache_utils.py):

  * Trigger (post_attention_update, l.199-205): eviction runs on DECODE steps only, when
        cur_len > token_budget  and  cur_len % residual_length == 0
    where cur_len is the *physical cache length* (reset to token_budget after each round,
    l.1096). token_budget K = 2048, residual_length = 64 for these cells.

  * _run_eviction (l.255-259): K_sel = token_budget - residual = 1984 slots are kept out of
    the candidate set = slots [0, cur_len - residual); the newest `residual`=64 slots (the
    residual/recency window) are kept unconditionally and moved behind the compacted keep-set
    (l.1090-1091).

  * random_pp branch (l.372-376):
        pl_eff = min(prompt_len, K_sel - 1)
        rs     = torch.rand(batch, n_kv_heads, n_cand)   # fresh iid U[0,1] PER HEAD, per
                                                         # layer, per round
        rs[..., :pl_eff] = +inf                          # prompt slots always kept
        keep   = topk(rs, K_sel)
    -> each KV head keeps all pl_eff prompt slots plus a uniform random (K_sel - pl_eff)-subset
    of the (n_cand - pl_eff) non-prompt candidates, WITHOUT replacement, INDEPENDENTLY per
    head (verified: the rand tensor has an n_heads dim and there is no cross-head sharing on
    this branch; contrast `random_unified` l.534 which shares one draw).

  * Universal chronological compaction (l.990): the keep-set is sorted before the gather, so
    slot order == time order after every round; hence the prompt occupies slots [0, pl_eff)
    at every round and the slot-based prompt protection is stable, and the candidate set at
    any round is exactly the oldest cur_len-64 tokens still alive.

  * Steady state (prompt_len < K): after a round the cache holds exactly K tokens; the next
    trigger fires 64 decode tokens later at cur_len = K + 64 = 2112. So every round has
        n_cand = 2112 - 64 = 2048 = K   candidate slots,
    of which pl_eff are protected, and n_cand - K_sel = 64 candidates are evicted from the
    (K - pl_eff) unprotected ones.  Per-round survival of one specific unprotected candidate
    token, per head:
        q = 1 - 64 / (K - pl_eff) = (K_sel - pl_eff) / (K - pl_eff)

  * Round schedule in cumulative-token coordinates (cumulative == cache length until the
    first round): the first round fires at
        T_1 = 64 * (floor(max(K, prompt_len) / 64) + 1)      (= 2112 when prompt_len < K)
    and round m fires at T_m = T_1 + 64*(m-1).  At round m the protected residual window is
    absolute positions [T_m - 64, T_m); a live token at absolute position t is a candidate
    iff t < T_m - 64.  Round m happens before token f is generated iff T_m <= f.

  * Rounds experienced by token t up to the generation of token f:
        N(x)    = 0 if x < T_1 else (x - T_1) // 64 + 1      (# rounds with T_m <= x)
        R(t, f) = max(0, N(f) - N(t + 64))
    Per-token survival at f (exact for a single token -- uniform independent draws each
    round):  p(t) = 1 if t < prompt_len else q ** R(t, f).

  * Span of L tokens [b, b+L):  per head,
        P_head = 1 - prod_{t=b..b+L-1} (1 - p(t))
    APPROXIMATION (disclosed): token survivals within a head are treated as independent; in
    truth each round evicts exactly 64 of the (K - pl_eff) candidates WITHOUT replacement, so
    survivals are (weakly, ~64/(K-pl_eff)) negatively correlated. For L << K - pl_eff the
    error is negligible.

  * Across heads: draws are independent per KV head (see above), so
        P_any = 1 - (1 - P_head) ** H,   H = num_key_value_heads (phi-4-reasoning: 10,
                                             Qwen3-4B: 8 -- read from config.json).
    NOTE: draws are independent across LAYERS too; the probability that the span survives in
    >= 1 head of >= 1 of the ~40/36 layers is 1 - (1-P_head)**(H*n_layers), i.e. much larger.
    We report the per-layer quantity the task specifies.

PROMPT RECONSTRUCTION (how pl is obtained): reproduces eval_hf.py exactly for gpqa --
question = example["question"].strip() (Utils/parser.py parse_question), qwen-instruct
question_format is the identity, use_few_shot off, surround_with_messages on -> messages =
[{"role": "user", "content": question}] and tokenizer.apply_chat_template(messages,
tokenize=True, add_generation_prompt=True) (eval_hf.py l.52-60, l.104-126). pl = number of
prompt tokens = the engine's prompt_len (prefill n_new). The saved "completion" field is the
FULL decoded sequence (prompt + generation, skip_special_tokens=True), so the decoded prompt
must be a literal prefix of it -- asserted per problem, and generation text is what follows.
Char -> token positions are obtained by re-tokenizing the generation prefix (add_special_
tokens=False); this can drift by a token or two vs the sampled stream (merge boundaries,
stripped special tokens) -- negligible against the 64-token round cadence.
===========================================================================================

Run on a compute node (tokenizer loads):
  cd $RA_ENGINE && python $RA_ROOT/kvcompress/analysis/fork_autopsy.py
"""

import json
import math
import os
import re
import sys

BASE = os.environ.get("RA_ENGINE", os.path.join(os.environ.get("RA_ROOT", "."), "kvcompress/harness"))
DATA = os.path.join(BASE, "data/gpqa/test.jsonl")
MODELS = os.environ.get("RA_MODELS_DIR", "models")

K_BUDGET = 2048
RESIDUAL = 64
K_SEL = K_BUDGET - RESIDUAL  # 1984

CELLS = {
    "phi_gpqa": dict(
        model=os.path.join(MODELS, "phi-4-reasoning"),
        rp_dir=os.path.join(BASE, "gate2/phi-4-reasoning/gpqa_K2048/random_pp/"
                                  "gpqa_bs4_budget=2048_random_pp_smooth"),
        tri_dir=os.path.join(BASE, "gate2/phi-4-reasoning/gpqa_K2048/triattn_ph_memofix/"
                                   "gpqa_bs4_budget=2048_triattn_ph"),
    ),
    "4b_gpqa": dict(
        model=os.path.join(MODELS, "Qwen3-4B"),
        rp_dir=os.path.join(BASE, "gate2/Qwen3-4B/gpqa_K2048/random_pp/"
                                  "gpqa_bs4_budget=2048_random_pp"),
        tri_dir=os.path.join(BASE, "gate2/Qwen3-4B/gpqa_K2048/triattn_ph_memofix/"
                                   "gpqa_bs4_budget=2048_triattn_ph"),
    ),
}

# ---------------------------------------------------------------------------------------
# Problem spec (from the fork inventory). Anchors are the longest contiguous LITERAL
# fragments of the report quotes (report '...' are the reading agent's elisions).
# b modes:  priority = first anchor found (in order), first occurrence
#           earliest = min position over all anchors found
#           numeric_fact = control rule: earliest occurrence in the generation of a numeric
#                          fact lifted from the question text
# f modes:  dismiss = anchors in priority order, LAST occurrence (dismissal = final commit);
#           last_boxed = char pos of the last \boxed{
# Any f-anchor miss falls back to last_boxed (flagged). run=None -> auto-pick (flips: first
# wrong completed rp run of 0-3; controls: first correct rp run of 0-3).
# ---------------------------------------------------------------------------------------
PROBLEMS = [
    # ---- phi_gpqa flips (rp-side wrong turns) ----
    dict(cell="phi_gpqa", gi=122, cls="flip", run=0,
         b_mode="priority",
         b_anchors=["the problem might require relativity effects", "might require relativity"],
         f_mode="dismiss", f_anchors=["not close to the speed of light"]),
    dict(cell="phi_gpqa", gi=86, cls="flip", run=2,
         b_mode="earliest",  # earliest carbon-count engagement is the true first fork
         b_anchors=["9 carbons", "nine carbons", "carbon count", "3.7 ppm"],
         f_mode="dismiss",
         f_anchors=["correctly includes the aldehyde proton", "aldehyde proton"]),
    dict(cell="phi_gpqa", gi=174, cls="flip", run=3,
         b_mode="earliest",  # may be ABSENT (commit-without-raising) -> dropped
         b_anchors=["quadrupole", "multipole"],
         # dismissal = the final commit ("... So answer D"), not the first dipole mention
         f_mode="dismiss",
         f_anchors=["So answer D", "due to dipole radiation", "dipole radiation"]),
    dict(cell="phi_gpqa", gi=192, cls="flip", run=1,
         b_mode="earliest", b_anchors=["dN/dp"],  # slip-type fork
         f_mode="dismiss", f_anchors=["dN/dp = -5 / p^6", "-5 / p^6", "5r^4", "5 r^4"]),
    dict(cell="phi_gpqa", gi=140, cls="flip", run=2,
         b_mode="earliest", b_anchors=["benzyne", "ortho"],  # FLAG-VAGUE in inventory
         f_mode="dismiss", f_anchors=["two regioisomeric"]),
    dict(cell="phi_gpqa", gi=85, cls="flip", run=None,   # FLAG-TOO-VAGUE: no quotes/run in
         b_mode="numeric_fact", b_anchors=[],            # report -> control-style fallback
         f_mode="last_boxed", f_anchors=[]),
    dict(cell="phi_gpqa", gi=69, cls="flip", run=1,      # mixed-mode: only run_1 wrong-turn
         b_mode="earliest", b_anchors=["W(CO)6", "W(CO)_6", "W(CO)_{6}"],
         f_mode="dismiss", f_anchors=["D4h", "D_{4h}", "D_4h"]),
    # ---- 4b_gpqa flips ----
    # gi176: the literal "radial velocities are not relevant here" occurs ONCE and EARLY
    # (only 5 of the 52 'Doppler' mentions precede it) -- the inventory's dismissal quote
    # elides ~38k chars; its intended dismissal is the FINAL commit "(1.5)^2 x 1 = 2.25 ->
    # boxed{B}" at trace end (audited against the extracted dump 2026-08-08). Anchor there;
    # the early utterance is visible via b/f_alt anyway.
    dict(cell="4b_gpqa", gi=176, cls="flip", run=0,
         b_mode="priority", b_anchors=["700 km/s"],
         f_mode="dismiss",
         f_anchors=["(1.5)^2 \\times 1 = 2.25", "\\times 1 = 2.25", "2.25"]),
    dict(cell="4b_gpqa", gi=61, cls="flip", run=0,
         b_mode="earliest",  # invented chemistry; consideration may be commit-without-raising
         b_anchors=["ozonolysis", "Lindlar", "alkyne"],
         f_mode="dismiss", f_anchors=["cleaving the alkyne"]),
    dict(cell="4b_gpqa", gi=160, cls="flip", run=None,   # FLAG-TOO-VAGUE
         b_mode="earliest", b_anchors=["electron scattering", "scattering", "electron"],
         f_mode="last_boxed", f_anchors=[]),
    dict(cell="4b_gpqa", gi=189, cls="flip", run=None,   # FLAG-TOO-VAGUE
         b_mode="earliest", b_anchors=["nucleophil"],
         f_mode="last_boxed", f_anchors=[]),
]

CONTROLS = [
    ("phi_gpqa", 68), ("phi_gpqa", 70), ("phi_gpqa", 84), ("phi_gpqa", 87),
    ("phi_gpqa", 121), ("phi_gpqa", 123), ("phi_gpqa", 139), ("phi_gpqa", 141),
    ("4b_gpqa", 60), ("4b_gpqa", 62), ("4b_gpqa", 135), ("4b_gpqa", 137),
    ("4b_gpqa", 159), ("4b_gpqa", 161), ("4b_gpqa", 175), ("4b_gpqa", 177),
]
for _c, _g in CONTROLS:
    PROBLEMS.append(dict(cell=_c, gi=_g, cls="control", run=None,
                         b_mode="numeric_fact", b_anchors=[],
                         f_mode="last_boxed", f_anchors=[]))


# ---------------------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------------------
_shard_cache = {}

def read_completion(run_dir, run, gi):
    """gi = shard_start + line-index-within-shard (shards 0,24,...,168)."""
    d = os.path.join(run_dir, f"run_{run}")
    key = d
    if key not in _shard_cache:
        starts = sorted(int(m.group(1)) for f in os.listdir(d)
                        for m in [re.match(r"completions_shard(\d+)\.jsonl$", f)] if m)
        _shard_cache[key] = starts
    starts = _shard_cache[key]
    if not starts:
        return None
    start = max(s for s in starts if s <= gi)
    path = os.path.join(d, f"completions_shard{start}.jsonl")
    idx = gi - start
    with open(path) as f:
        for i, line in enumerate(f):
            if i == idx:
                return json.loads(line)["completion"]
    return None


def boxed_letter(gen):
    """Last \\boxed{...}; returns (letter or None, char_pos_of_last_boxed or None)."""
    ms = list(re.finditer(r"\\boxed\{", gen))
    if not ms:
        return None, None
    m = ms[-1]
    depth, out = 1, []
    for ch in gen[m.end():m.end() + 120]:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
    t = re.sub(r"\\text|\\mathrm|\\mathbf|\\rm|[{}()\[\]\s$]", "", "".join(out))
    letter = None
    if t[:1] in "ABCD" and (len(t) == 1 or not t[1:2].isalnum()):
        letter = t[0]
    return letter, m.start()


def numeric_fact_variants(question):
    """Numeric facts from the question text + spelling variants, longest first."""
    facts = []
    for m in re.finditer(r"\d+(?:[.,]\d+)*(?:\s*[e^]\s*\{?[+-]?\d+\}?)?", question):
        s = m.group(0).rstrip(".,")
        if len(s) < 2 or s == "10":
            continue
        vs = {s, s.replace(",", "")}
        if "^" in s:
            base, exp = s.split("^", 1)
            exp = exp.strip().strip("{}")
            vs |= {f"{base.strip()}^{exp}", f"{base.strip()}^{{{exp}}}"}
        head = re.match(r"\d+(?:\.\d+)?", s.replace(",", ""))
        if head and len(head.group(0)) >= 3:
            vs.add(head.group(0))
        for v in vs:
            if len(v) >= 2 and v not in facts:
                facts.append(v)
    facts.sort(key=len, reverse=True)
    return facts


# ---------------------------------------------------------------------------------------
# Survival model (see module docstring for the derivation)
# ---------------------------------------------------------------------------------------
def survival(pl, b, f, L, H):
    assert pl < K_SEL - 1, f"prompt_len {pl} >= K_sel-1: prompt not fully protected"
    q = 1.0 - RESIDUAL / (K_BUDGET - pl)          # per-round per-token keep prob (non-prompt)
    T1 = 64 * (max(K_BUDGET, pl) // 64 + 1)       # cumulative pos of the first eviction

    def N(x):
        return 0 if x < T1 else (x - T1) // 64 + 1

    R_first = max(0, N(f) - N(b + 64))
    log_all_dead = 0.0
    alive_certain = False
    for t in range(b, b + L):
        p = 1.0 if t < pl else q ** max(0, N(f) - N(t + 64))
        if p >= 1.0:
            alive_certain = True
            break
        log_all_dead += math.log1p(-p)
    P_head = 1.0 if alive_certain else 1.0 - math.exp(log_all_dead)
    P_any = 1.0 - (1.0 - P_head) ** H
    return q, T1, R_first, P_head, P_any


# ---------------------------------------------------------------------------------------
def main():
    from transformers import AutoTokenizer

    data = [json.loads(l) for l in open(DATA)]
    assert len(data) == 198
    assert all(d["answer"] == "A" for d in data), "gold layout is not all-A"
    print("[gold] verified: all 198 gold answers are 'A' (shared data file -> holds for "
          "both phi and 4b cells)", file=sys.stderr)

    toks, cfgs = {}, {}
    for cell, cc in CELLS.items():
        toks[cell] = AutoTokenizer.from_pretrained(cc["model"])
        cfgs[cell] = json.load(open(os.path.join(cc["model"], "config.json")))
        print(f"[cfg] {cell}: H_kv={cfgs[cell]['num_key_value_heads']} "
              f"layers={cfgs[cell]['num_hidden_layers']}", file=sys.stderr)

    prompt_cache = {}

    def get_prompt(cell, gi):
        key = (cell, gi)
        if key not in prompt_cache:
            tok = toks[cell]
            q = data[gi]["question"].strip()          # parse_question + identity format
            enc = tok.apply_chat_template([{"role": "user", "content": q}],
                                          tokenize=True, add_generation_prompt=True,
                                          return_dict=True)
            ids = enc["input_ids"]
            prompt_cache[key] = (len(ids), tok.decode(ids, skip_special_tokens=True))
        return prompt_cache[key]

    def get_gen(cell, side, run, gi):
        """Returns (gen_text, flags). Splits the saved full decode at the decoded prompt."""
        full = read_completion(CELLS[cell][side + "_dir"], run, gi)
        if full is None:
            return None, ["missing"]
        pl, ptxt = get_prompt(cell, gi)
        if full.startswith(ptxt):
            return full[len(ptxt):], []
        tail = ptxt[-60:]
        i = full.find(tail)
        if i >= 0:
            return full[i + len(tail):], ["prompt_align_fuzzy"]
        return full, ["prompt_align_FAILED"]

    def grade(cell, side, gi):
        """Boxed letters for runs 0-3."""
        out = []
        for r in range(4):
            g, fl = get_gen(cell, side, r, gi)
            out.append(None if g is None else boxed_letter(g)[0])
        return out

    rows, dropped = [], []
    for spec in PROBLEMS:
        cell, gi, cls = spec["cell"], spec["gi"], spec["cls"]
        tok = toks[cell]
        H = cfgs[cell]["num_key_value_heads"]
        flags = []

        rp_letters = grade(cell, "rp", gi)
        run = spec["run"]
        if run is None:
            if cls == "control":
                ok = [r for r, l in enumerate(rp_letters) if l == "A"]
                run = ok[0] if ok else 0
                if not ok:
                    flags.append("no_correct_rp_run")
            else:
                bad = [r for r, l in enumerate(rp_letters)
                       if l is not None and l != "A"]
                run = bad[0] if bad else 0
                flags.append(f"run_autopicked_{run}")

        if cls == "control":
            tri_letters = grade(cell, "tri", gi)
            rp_maj = sum(l == "A" for l in rp_letters) >= 3
            tri_maj = sum(l == "A" for l in tri_letters) >= 3
            flags.append(f"rp_runs={''.join(l or '-' for l in rp_letters)}")
            flags.append(f"tri_runs={''.join(l or '-' for l in tri_letters)}")
            if not (rp_maj and tri_maj):
                flags.append("SKIP_not_maj_correct_both")

        gen, gflags = get_gen(cell, "rp", run, gi)
        flags += gflags
        if gen is None:
            dropped.append((cell, gi, "completion missing"))
            continue
        pl, _ = get_prompt(cell, gi)

        # ---- locate b (consideration source span) ----
        b_char = b_len = None
        b_how = ""
        if spec["b_mode"] in ("priority", "earliest"):
            hits = []
            for a in spec["b_anchors"]:
                p = gen.find(a)
                if p >= 0:
                    hits.append((p, a))
                    if spec["b_mode"] == "priority":
                        break
            if hits:
                b_char, a = min(hits)
                b_len, b_how = len(a), f"anchor:{a[:28]}"
        else:  # numeric_fact
            best = None
            for v in numeric_fact_variants(data[gi]["question"]):
                p = gen.find(v)
                if p >= 0 and (best is None or p < best[0]):
                    best = (p, v)
            if best:
                b_char, b_len, b_how = best[0], len(best[1]), f"numfact:{best[1][:20]}"
            else:
                b_char, b_len, b_how = 0, None, "fallback_genstart"
                flags.append("no_numeric_fact_match")
        if b_char is None:
            dropped.append((cell, gi, f"b unlocatable (anchors {spec['b_anchors']}; "
                                      "commit-without-raising?)"))
            continue

        # ---- locate f (dismissal / final answer) ----
        letter, boxed_pos = boxed_letter(gen)
        f_char = None
        f_how = ""
        if spec["f_mode"] == "dismiss":
            for a in spec["f_anchors"]:
                p = gen.rfind(a)
                if p >= 0:
                    f_char, f_how = p, f"anchor:{a[:28]}"
                    break
        if f_char is None:
            if boxed_pos is not None:
                f_char, f_how = boxed_pos, "last_boxed"
            else:
                f_char, f_how = len(gen), "gen_end"
            if spec["f_mode"] == "dismiss":
                flags.append("f_anchor_missed->last_boxed")

        # ---- char -> token positions (absolute, prompt included) ----
        b_tok = pl + len(tok(gen[:b_char], add_special_tokens=False)["input_ids"])
        if b_len is None:                      # fallback_genstart: 20-token proxy span
            L = 20
        else:
            e_tok = pl + len(tok(gen[:b_char + b_len], add_special_tokens=False)["input_ids"])
            L = max(1, e_tok - b_tok)
        f_tok = pl + len(tok(gen[:f_char], add_special_tokens=False)["input_ids"])
        if f_tok <= b_tok:
            flags.append("f<=b")
        # alternate horizon: the FINAL commitment (last \boxed{, else generation end) --
        # reported for every row so the "dismissal utterance" vs "final answer" ambiguity
        # is carried in the data instead of hidden by the anchor choice.
        fa_char = boxed_pos if boxed_pos is not None else len(gen)
        f_alt = pl + len(tok(gen[:fa_char], add_special_tokens=False)["input_ids"])
        gen_toks = pl + len(tok(gen, add_special_tokens=False)["input_ids"])

        q, T1, R, P_head, P_any = survival(pl, b_tok, f_tok, L, H)
        _, _, R_alt, _, P_any_alt = survival(pl, b_tok, f_alt, L, H)
        rows.append(dict(cell=cell, gi=gi, cls=cls, run=run, letter=letter or "-",
                         pl=pl, b=b_tok, f=f_tok, L=L, R=R, q=q,
                         P_head=P_head, P_any=P_any,
                         f_alt=f_alt, R_alt=R_alt, P_any_alt=P_any_alt,
                         gen_toks=gen_toks, H=H,
                         b_how=b_how, f_how=f_how, flags=";".join(flags) or "-"))

    # ---- output table ----
    cols = ["cell", "gi", "cls", "run", "letter", "pl", "b", "f", "L", "R", "q",
            "P_head", "P_any", "f_alt", "R_alt", "P_any_alt", "gen_toks", "H",
            "b_how", "f_how", "flags"]
    print("\t".join(cols))
    for r in rows:
        print("\t".join(
            f"{r[c]:.6f}" if c in ("P_head", "P_any", "P_any_alt") else
            f"{r[c]:.5f}" if c == "q" else str(r[c]) for c in cols))

    # ---- sanity checks ----
    print("\n[sanity 1] short-horizon control(s): every row with R == 0 must have "
          "P_any == 1", file=sys.stderr)
    z = [r for r in rows if r["R"] == 0]
    for r in z[:6]:
        print(f"    {r['cell']} gi{r['gi']} ({r['cls']}): b={r['b']} f={r['f']} R=0 "
              f"P_any={r['P_any']:.6f}", file=sys.stderr)
    if not z:
        print("    (no R==0 row occurred; synthetic: pl=300, b=400, f=2100 < T1=2112)",
              file=sys.stderr)
        qq, tt, RR, ph, pa = survival(300, 400, 2100, 20, 8)
        print(f"    -> R={RR} P_any={pa:.6f}", file=sys.stderr)
    print("[sanity 2] span born 20k tokens before f at K=2048 must be ~dead: "
          "pl=300, b=400, L=20, f=20400, H=8", file=sys.stderr)
    qq, tt, RR, ph, pa = survival(300, 400, 20400, 20, 8)
    print(f"    -> q={qq:.5f} T1={tt} R={RR} P_head={ph:.3e} P_any={pa:.3e}",
          file=sys.stderr)

    if dropped:
        print("\n[dropped]", file=sys.stderr)
        for c, g, why in dropped:
            print(f"    {c} gi{g}: {why}", file=sys.stderr)


if __name__ == "__main__":
    main()
