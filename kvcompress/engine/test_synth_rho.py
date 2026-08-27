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
"""Unit test: synth_pin rho option (item 3 — correlation dose at matched mass).
(a) budget parity; (b) draw persistence across rounds (same kept sites);
(c) rho=1: per layer all-or-none across site heads; (d) rho=0: sites independent
    (across 200 layer-objects, fraction of mixed layers ~ 1-[s^3+(1-s)^3]);
(e) marginal survival ~ s for both rho values (200 layers, tol 0.07).
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from kvcompress.engine.cache_utils import EvictLayer

B, HKV, D = 1, 8, 4
TB, RES = 1088, 64
CUR = TB + RES
SPAN = (300, 316)
SITE = [2, 3, 5]
results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name + (f" [{detail}]" if detail and not cond else ""))


def one_layer(rho, s=0.5, rounds=2, idx=0):
    os.environ['SYNTH_SPEC'] = json.dumps(
        [{"lo": SPAN[0], "hi": SPAN[1], "rho": rho, "s": s, "site_heads": SITE}])
    layer = EvictLayer(idx, TB, RES, 'synth_pin', rkv_lambda=0.5, smooth=False, verbose=False)
    layer.lazy_initialization(torch.zeros(B, CUR, HKV, D))
    layer.prompt_len = 0
    vals = torch.arange(CUR, dtype=torch.float32).view(1, CUR, 1, 1)
    layer.k_cache[:, :CUR] = vals; layer.v_cache[:, :CUR] = vals
    layer.positions[:, :, :CUR] = torch.arange(CUR).view(1, 1, CUR)
    layer.cache_seqlens.fill_(CUR); layer.cumulative_length = CUR
    kept_rounds = []
    for r in range(rounds):
        layer._run_eviction(CUR)
        check(f"rho{rho} budget", int(layer.cache_seqlens[0].item()) == TB)
        kept = []
        for h in range(HKV):
            pos = layer.k_cache[0, :TB, h, 0]
            kept.append(int(((pos >= SPAN[0]) & (pos < SPAN[1])).sum().item()) > 0)
        kept_rounds.append(kept)
        vals = torch.arange(CUR, CUR + RES, dtype=torch.float32).view(1, RES, 1, 1)
        layer.k_cache[:, TB:TB+RES] = vals; layer.v_cache[:, TB:TB+RES] = vals
        layer.positions[:, :, TB:TB+RES] = torch.arange(CUR, CUR+RES).view(1, 1, RES)
        layer.cache_seqlens.fill_(CUR); layer.cumulative_length = CUR + RES
    del os.environ['SYNTH_SPEC']
    return kept_rounds


k = one_layer(1.0, rounds=3)
check("persistence", k[0] == k[1] == k[2], str(k))
check("non-site always blacklisted", all(not k[0][h] for h in range(HKV) if h not in SITE))

allnone = mixed = 0
marg1 = []
for i in range(200):
    kk = one_layer(1.0, s=0.5, rounds=1, idx=i)[0]
    sv = [kk[h] for h in SITE]
    marg1 += sv
    if all(sv) or not any(sv): allnone += 1
    else: mixed += 1
check("rho=1 all-or-none", mixed == 0, f"mixed={mixed}")
check("rho=1 marginal~s", abs(sum(marg1)/len(marg1) - 0.5) < 0.07, f"{sum(marg1)/len(marg1):.3f}")

mixed0 = 0
marg0 = []
for i in range(200):
    kk = one_layer(0.0, s=0.5, rounds=1, idx=i)[0]
    sv = [kk[h] for h in SITE]
    marg0 += sv
    if any(sv) and not all(sv): mixed0 += 1
check("rho=0 mixed layers exist", 0.5 < mixed0/200 < 0.9, f"{mixed0/200:.2f} (expect ~0.75)")
check("rho=0 marginal~s", abs(sum(marg0)/len(marg0) - 0.5) < 0.07, f"{sum(marg0)/len(marg0):.3f}")

fails = [r for r in results if not r[1]]
print(f"\n{len(results)-len(fails)}/{len(results)} assertions passed")
sys.exit(1 if fails else 0)
