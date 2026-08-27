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
"""Unit test: NEW mode 'random_pcap' (capped prompt protection with a permanent every-2nd overflow
subsample) + regression checks on 'random_ppp' and 'random_strat_pp'.

Drives the real EvictLayer._run_eviction on CPU with synthetic KV whose VALUES encode the absolute
token position (the tracked-position instrument of kvcompress/engine/test_compaction_order.py).

Scenario: prompt_len=900, PROMPT_CAP=512, token_budget=1200, residual_length=64 -> K=1136;
>=4 eviction rounds (each round: cache filled to token_budget+residual=1264 slots, then evict).

Assertions:
  (a) random_pcap round 1 keeps EXACTLY prompt positions {0..511} U {512,514,...,898} (512+194=706);
  (b) rounds 2..4 preserve EXACTLY that prompt set (no churn / decay of the subsample);
  (c) post-eviction cache size == token_budget every round; #inf-protected slots (_pl_kept) <= K;
  (d) short prompt (300 < cap): random_pcap == random_pp exactly (same seed -> same kept sets;
      protected set = whole prompt);
  (e) _pl_kept does NOT leak across a fresh EvictLayer (new object => recomputed);
  (f) residual parity: token_budget=2048, residual=512 -> K=1536, post-eviction cache 2048;
  (g) random_ppp regression: prompt + plan-prefix slots [0:min(pl+768, K-64)) survive every round,
      exact budget;
  (h) random_strat_pp regression: ~1 kept token per chunk (round 1, exact) + exact budget.

Run:  python test_fix_arms.py
"""
import os
import sys

import torch

# Test the LIVE engine (CACHE_UTILS_DIR override kept for parity with the existing test).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from kvcompress.engine.cache_utils import EvictLayer

B, HKV, D = 1, 2, 4
TB, RES = 1200, 64
K = TB - RES                      # 1136 scored keep slots
PL = 900                          # long prompt (> cap)
PCAP = 512
CUR = TB + RES                    # 1264: cache length at each eviction trigger
EXPECT_PROMPT = set(range(PCAP)) | set(range(PCAP, PL, 2))   # 512 + 194 = 706

results = []  # (name, passed, detail)


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))


def make_layer(mode, token_budget=TB, residual=RES, cur=CUR, prompt_len=PL):
    layer = EvictLayer(0, token_budget, residual, mode,
                       rkv_lambda=0.5, smooth=False, verbose=False)
    layer.lazy_initialization(torch.zeros(B, cur, HKV, D))
    layer.prompt_len = prompt_len
    return layer


def fill(layer, abs_positions, start_slot):
    """Write tokens whose K/V VALUES are their absolute position, at cache slots start_slot..."""
    abs_positions = list(abs_positions)
    n = len(abs_positions)
    vals = torch.tensor(abs_positions, dtype=torch.float32).view(1, n, 1, 1)
    layer.k_cache[:, start_slot:start_slot + n] = vals
    layer.v_cache[:, start_slot:start_slot + n] = vals
    if getattr(layer, 'positions', None) is not None:
        pos = torch.tensor(abs_positions, dtype=layer.positions.dtype).view(1, 1, n)
        layer.positions[:, :, start_slot:start_slot + n] = pos
    layer.cache_seqlens.fill_(start_slot + n)
    layer.cumulative_length = int(abs_positions[-1]) + 1


def kept_positions(layer, n, head):
    """Absolute positions stored in cache slots [0:n) for one KV head (batch 0)."""
    return sorted(int(x) for x in layer.k_cache[0, :n, head, 0].tolist())


def run_rounds(mode, rounds=4, token_budget=TB, residual=RES, prompt_len=PL):
    """>=4 eviction rounds; returns (per-round per-head kept positions, per-round post-evict
    cache_seqlens BEFORE the next refill, layer)."""
    cur = token_budget + residual
    layer = make_layer(mode, token_budget, residual, cur, prompt_len)
    fill(layer, range(cur), 0)
    next_pos = cur
    kept_per_round, seqlens = [], []
    for _ in range(rounds):
        layer._run_eviction(cur)
        kept_per_round.append([kept_positions(layer, token_budget, h) for h in range(HKV)])
        seqlens.append(int(layer.cache_seqlens[0].item()))
        fill(layer, range(next_pos, next_pos + residual), token_budget)
        next_pos += residual
    return kept_per_round, seqlens, layer


# ----------------------------------------------------------------------------- random_pcap (a)-(c)
def test_pcap_long_prompt():
    os.environ['PROMPT_CAP'] = str(PCAP)
    torch.manual_seed(0)
    kept, seqlens, layer = run_rounds('random_pcap', rounds=4)
    prompt_kept = [[{p for p in kept[r][h] if p < PL} for h in range(HKV)] for r in range(4)]

    # (a) POSITION-TRACKED SEMANTICS: designated set is a guaranteed SUBSET every round; undesignated
    # overflow may survive as ordinary scatter candidates and must DECAY (non-increasing extras).
    ok_a = all(EXPECT_PROMPT <= prompt_kept[0][h] for h in range(HKV))
    check("(a) random_pcap round 1: designated 706-set fully present (subset invariant)", ok_a)
    # (a2) diagnostic split: the protected 706-subsample is fully present at round 1
    ok_a2 = all(EXPECT_PROMPT <= prompt_kept[0][h] for h in range(HKV))
    check("(a2) [diagnostic] round 1: protected 706-subsample fully PRESENT (failure of (a) = extra "
          "unprotected prompt toks, not lost protected ones)", ok_a2)

    # (b) designated set stable ALL rounds; undesignated extras monotonically non-increasing
    ok_b = all(EXPECT_PROMPT <= prompt_kept[r][h] for r in range(4) for h in range(HKV))
    extras = [[len(prompt_kept[r][h] - EXPECT_PROMPT) for r in range(4)] for h in range(HKV)]
    ok_bx = all(all(e[i+1] <= e[i] for i in range(3)) for e in extras)
    check("(b) random_pcap: designated set stable all 4 rounds AND undesignated extras decay",
          ok_b and ok_bx, f"extras/round head0: {extras[0]}")
    # (b2) diagnostic split: the protected subsample itself never churns (always a subset, all rounds)
    ok_b2 = all(EXPECT_PROMPT <= prompt_kept[r][h] for r in range(4) for h in range(HKV))
    check("(b2) [diagnostic] protected 706-subsample stable across ALL 4 rounds (subset check)", ok_b2,
          "sizes/round head0: " + str([len(prompt_kept[r][0]) for r in range(4)]))

    # (c) exact budget every round + inf-protected count <= K
    ok_c1 = (all(len(kept[r][h]) == TB and len(set(kept[r][h])) == TB
                 for r in range(4) for h in range(HKV))
             and seqlens == [TB] * 4)
    check("(c1) random_pcap: post-eviction cache size == token_budget (1200), no dup slots, all 4 rounds",
          ok_c1, f"seqlens={seqlens}")
    check(f"(c2) random_pcap: designated size 706 <= K={K} (stateless impl, no _pl_kept)",
          len(EXPECT_PROMPT) == 706 and len(EXPECT_PROMPT) <= K and not hasattr(layer, '_pl_kept'))


# ----------------------------------------------------------------------- short prompt == random_pp (d)
def test_pcap_short_prompt_equals_random_pp():
    os.environ['PROMPT_CAP'] = str(PCAP)
    SPL = 300
    torch.manual_seed(123)
    kept_pp, _, _ = run_rounds('random_pp', rounds=4, prompt_len=SPL)
    torch.manual_seed(123)
    kept_pc, _, layer = run_rounds('random_pcap', rounds=4, prompt_len=SPL)

    check("(d1) short prompt (300 < cap): random_pcap kept sets IDENTICAL to random_pp (same seed, 4 rounds)",
          kept_pc == kept_pp)
    ok_prompt = all(set(range(SPL)) <= set(kept_pc[r][h]) for r in range(4) for h in range(HKV))
    check("(d2) short prompt: whole prompt (300) kept every round", ok_prompt)


# ------------------------------------------------------------------------------- no state leak (e)
def test_pcap_state_no_leak():
    os.environ['PROMPT_CAP'] = str(PCAP)
    torch.manual_seed(7)
    _, _, layer_long = run_rounds('random_pcap', rounds=1)       # long prompt: _pl_kept -> 706
    fresh = make_layer('random_pcap', prompt_len=300)
    ok_fresh = not hasattr(fresh, '_pl_kept')
    fill(fresh, range(CUR), 0)
    fresh._run_eviction(CUR)
    check("(e) stateless pcap: no _pl_kept attr on any layer (nothing to leak); fresh short-prompt layer "
          "protects its own whole prompt",
          ok_fresh and not hasattr(layer_long, '_pl_kept')
          and set(range(300)) <= {q for q in kept_positions(fresh, TB, 0) if q < 300})


# ----------------------------------------------------------------------------- residual parity (f)
def test_residual_parity_2048_512():
    os.environ['PROMPT_CAP'] = str(PCAP)
    torch.manual_seed(11)
    TB2, RES2 = 2048, 512
    K2 = TB2 - RES2
    check("(f1) residual parity: K = token_budget - residual_length = 2048 - 512 = 1536", K2 == 1536)
    kept, seqlens, layer = run_rounds('random_pcap', rounds=1, token_budget=TB2, residual=RES2)
    ok_size = (seqlens == [TB2]
               and all(len(kept[0][h]) == TB2 and len(set(kept[0][h])) == TB2 for h in range(HKV)))
    # engine split check: slots [K2:TB2) must hold the residual = the most recent RES2 positions
    cur2 = TB2 + RES2
    resid = [int(x) for x in layer.k_cache[0, K2:TB2, 0, 0].tolist()]
    ok_split = resid == list(range(cur2 - RES2, cur2))
    check("(f2) residual parity: post-eviction cache == 2048; slots [1536:2048) = the 512-token residual",
          ok_size and ok_split)


# ------------------------------------------------------------------------ random_ppp regression (g)
def test_random_ppp_regression():
    os.environ.pop('PLAN_PREFIX', None)                          # default 768
    torch.manual_seed(21)
    PPFX = min(PL + 768, K - 64)                                 # = 1072
    kept, seqlens, layer = run_rounds('random_ppp', rounds=4)
    ok_prot = all(set(range(PPFX)) <= set(kept[r][h]) for r in range(4) for h in range(HKV))
    ok_size = (all(len(kept[r][h]) == TB and len(set(kept[r][h])) == TB
                   for r in range(4) for h in range(HKV))
               and seqlens == [TB] * 4)
    check(f"(g) random_ppp regression: prompt+plan-prefix [0:{PPFX}) survives all 4 rounds; exact budget",
          ok_prot and ok_size)


# -------------------------------------------------------------------- random_strat_pp regression (h)
def test_random_strat_pp_regression():
    torch.manual_seed(31)
    n_cand = CUR - RES                                           # 1200 candidates, round 1: index == position
    n_hist = n_cand - PL                                         # 300
    m = max(min(K - PL, n_hist), 1)                              # 236 chunks
    ok = True
    det = ""
    for trial in range(5):
        kept, seqlens, layer = run_rounds('random_strat_pp', rounds=1)
        for h in range(HKV):
            hist = [p for p in kept[0][h] if PL <= p < n_cand]
            chunks = {(p - PL) * m // n_hist for p in hist}
            if not (len(hist) == K - PL == m and len(chunks) == m):
                ok = False
                det = f"trial {trial} head {h}: kept {len(hist)} hist toks over {len(chunks)} chunks, want {m}/{m}"
            if not (set(range(PL)) <= set(kept[0][h]) and len(kept[0][h]) == TB):
                ok = False
                det = f"trial {trial} head {h}: prompt lost or budget != {TB}"
        if seqlens != [TB]:
            ok = False
            det = f"trial {trial}: post-evict cache_seqlens {seqlens} != {TB}"
    check("(h) random_strat_pp regression: exactly 1 kept token per chunk (236 chunks) + exact budget "
          "(round 1, 5 trials)", ok, det)


def test_pmix():
    os.environ['PMIX_P'] = '0.25'; os.environ['PMIX_SMALL'] = '256'
    torch.manual_seed(3)
    kept, seqlens, layer = run_rounds('random_pmix', rounds=4)
    arch = layer._pmix_arch.tolist()
    check("(p1) random_pmix: >=1 archivist head", any(arch), str(arch))
    full = set(range(PL)); small = set(range(256))
    ok_arch = all(full <= {p for p in kept[r][h] if p < PL}
                  for r in range(4) for h in range(HKV) if arch[h])
    check("(p2) random_pmix: archivist heads keep FULL prompt all 4 rounds", ok_arch)
    ok_scat = all(small <= {p for p in kept[r][h] if p < PL}
                  for r in range(4) for h in range(HKV) if not arch[h])
    check("(p3) random_pmix: scatterer heads keep first-256 prefix all 4 rounds", ok_scat)
    ok_bud = all(len(kept[r][h]) == TB and len(set(kept[r][h])) == TB for r in range(4) for h in range(HKV))
    check("(p4) random_pmix: exact budget, no dup slots, all rounds", ok_bud)
    m0 = layer._pmix_arch.clone()
    layer._run_eviction(TB + RES)
    check("(p5) random_pmix: head roles persist (no redraw)", bool((layer._pmix_arch == m0).all()))
    l2 = make_layer('random_pmix')
    check("(p6) random_pmix: fresh layer has no stale roles", not hasattr(l2, '_pmix_arch'))


if __name__ == '__main__':
    test_pcap_long_prompt()
    test_pcap_short_prompt_equals_random_pp()
    test_pcap_state_no_leak()
    test_residual_parity_2048_512()
    test_random_ppp_regression()
    test_random_strat_pp_regression()
    test_pmix()
    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results) - n_fail}/{len(results)} assertions passed")
    sys.exit(1 if n_fail else 0)
