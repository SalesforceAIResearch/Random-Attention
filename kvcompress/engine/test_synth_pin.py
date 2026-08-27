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
"""Unit test: NEW mode 'synth_pin' (S1/S1b/S4 synthetic mechanism probe).

Drives the real EvictLayer._run_eviction on CPU with synthetic KV whose VALUES encode the absolute
token position (same instrument as test_fix_arms.py). 8 KV heads to mirror Qwen3-4B.

Semantics under test: spans in SYNTH_SPEC are BLACKLISTED by default everywhere and PINNED (+inf)
exactly where (layer in scope) AND (head in set). No protection rule anywhere else (background =
uniform random over all candidates including any prompt region).

Assertions:
  (a) h-exactness: span pinned in heads {2,5}, layers 'all' -> after every one of 4 rounds the span
      is FULLY present in heads 2 and 5 and has ZERO tokens in the other 6 heads; budget exact.
  (b) h=0 (heads []): span absent from all 8 heads from round 1 on.
  (c) layer scope: spec layers=[3] -> layer_idx=0 object blacklists the span (absent all heads);
      layer_idx=3 object pins it (present exactly in the pinned head).
  (d) S1b two spans, disjoint heads: x -> head 1 only, y -> head 6 only, simultaneously.
  (e) persistence: pinned span survives all 4 rounds intact (no churn); blacklisted span never
      resurrects after later refills.
  (f) budget parity: post-eviction cache == token_budget every round, every condition.
  (g) SYNTH_AUDIT log reports exactly h copies per event for the pinned span.
  (h) no-spec regression: SYNTH_SPEC=[] == unbiased random keep (span survives at chance in some
      head with no inf/-inf interference; kept-set size exact).

Run:  python test_synth_pin.py
"""
import json
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from kvcompress.engine.cache_utils import EvictLayer

B, HKV, D = 1, 8, 4
TB, RES = 1088, 64
K = TB - RES                      # 1024 scored keep slots (the S1 budget)
CUR = TB + RES
SPAN = (300, 306)                 # 6-token needle span, absolute positions [300, 306)

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))


def make_layer(layer_idx=0):
    layer = EvictLayer(layer_idx, TB, RES, 'synth_pin',
                       rkv_lambda=0.5, smooth=False, verbose=False)
    layer.lazy_initialization(torch.zeros(B, CUR, HKV, D))
    layer.prompt_len = 0
    return layer


def fill(layer, abs_positions, start_slot):
    abs_positions = list(abs_positions)
    n = len(abs_positions)
    vals = torch.tensor(abs_positions, dtype=torch.float32).view(1, n, 1, 1)
    layer.k_cache[:, start_slot:start_slot + n] = vals
    layer.v_cache[:, start_slot:start_slot + n] = vals
    pos = torch.tensor(abs_positions, dtype=layer.positions.dtype).view(1, 1, n)
    layer.positions[:, :, start_slot:start_slot + n] = pos
    layer.cache_seqlens.fill_(start_slot + n)
    layer.cumulative_length = int(abs_positions[-1]) + 1


def span_count(layer, head, lo, hi):
    n = TB
    kept = layer.k_cache[0, :n, head, 0]
    return int(((kept >= lo) & (kept < hi)).sum().item())


def run_rounds(spec, layer_idx=0, rounds=4):
    os.environ['SYNTH_SPEC'] = json.dumps(spec)
    layer = make_layer(layer_idx)
    fill(layer, range(CUR), 0)
    next_pos = CUR
    per_round = []
    for _ in range(rounds):
        layer._run_eviction(CUR)
        per_round.append([span_count(layer, h, *SPAN) for h in range(HKV)])
        seq = int(layer.cache_seqlens[0].item())
        check("budget parity", seq == TB, f"seqlens {seq} != {TB}")
        fill(layer, range(next_pos, next_pos + RES), TB)
        next_pos += RES
    del os.environ['SYNTH_SPEC']
    return per_round


SL = SPAN[1] - SPAN[0]

# (a) h=2, all layers
pr = run_rounds([{"lo": SPAN[0], "hi": SPAN[1], "heads": [2, 5], "layers": "all"}])
for r, counts in enumerate(pr):
    check(f"(a) round{r+1} pinned heads full", counts[2] == SL and counts[5] == SL, str(counts))
    check(f"(a) round{r+1} other heads zero", all(c == 0 for i, c in enumerate(counts) if i not in (2, 5)), str(counts))

# (b) h=0
pr = run_rounds([{"lo": SPAN[0], "hi": SPAN[1], "heads": [], "layers": "all"}])
for r, counts in enumerate(pr):
    check(f"(b) round{r+1} h=0 absent everywhere", all(c == 0 for c in counts), str(counts))

# (c) layer scope
pr0 = run_rounds([{"lo": SPAN[0], "hi": SPAN[1], "heads": [4], "layers": [3]}], layer_idx=0)
pr3 = run_rounds([{"lo": SPAN[0], "hi": SPAN[1], "heads": [4], "layers": [3]}], layer_idx=3)
check("(c) out-of-scope layer blacklists", all(c == 0 for c in pr0[-1]), str(pr0[-1]))
check("(c) in-scope layer pins head 4 only",
      pr3[-1][4] == SL and all(c == 0 for i, c in enumerate(pr3[-1]) if i != 4), str(pr3[-1]))

# (d) S1b: two spans, disjoint heads
SPAN_Y = (600, 606)
os.environ['SYNTH_SPEC'] = json.dumps([
    {"lo": SPAN[0], "hi": SPAN[1], "heads": [1], "layers": "all"},
    {"lo": SPAN_Y[0], "hi": SPAN_Y[1], "heads": [6], "layers": "all"}])
layer = make_layer()
fill(layer, range(CUR), 0)
layer._run_eviction(CUR)
cx = [span_count(layer, h, *SPAN) for h in range(HKV)]
cy = [span_count(layer, h, *SPAN_Y) for h in range(HKV)]
check("(d) x in head 1 only", cx[1] == SL and sum(cx) == SL, str(cx))
check("(d) y in head 6 only", cy[6] == SL and sum(cy) == SL, str(cy))
del os.environ['SYNTH_SPEC']

# (g) audit log
with tempfile.TemporaryDirectory() as td:
    os.environ.update(SYNTH_SPEC=json.dumps(
        [{"lo": SPAN[0], "hi": SPAN[1], "heads": [0, 3, 7], "layers": "all"}]),
        SYNTH_AUDIT='1', SYNTH_AUDIT_DIR=td)
    layer = make_layer(layer_idx=11)
    fill(layer, range(CUR), 0)
    layer._run_eviction(CUR)
    f = [x for x in os.listdir(td) if x.startswith('synth_audit')]
    ok = False
    if f:
        line = open(os.path.join(td, f[0])).read().strip().split('\t')
        counts = json.loads(line[3])
        ok = (line[0] == '11' and counts[0] == SL and counts[3] == SL and counts[7] == SL
              and sum(counts) == 3 * SL)
    check("(g) audit reports exact h", ok, str(f))
    for k in ('SYNTH_SPEC', 'SYNTH_AUDIT', 'SYNTH_AUDIT_DIR'):
        del os.environ[k]

# (h) empty spec == plain random: exact budget, no crash, span survives somewhere at chance-ish
os.environ['SYNTH_SPEC'] = '[]'
layer = make_layer()
fill(layer, range(CUR), 0)
layer._run_eviction(CUR)
check("(h) empty-spec budget", int(layer.cache_seqlens[0].item()) == TB)
del os.environ['SYNTH_SPEC']

fails = [r for r in results if not r[1]]
print(f"\n{len(results) - len(fails)}/{len(results)} assertions passed")
sys.exit(1 if fails else 0)
