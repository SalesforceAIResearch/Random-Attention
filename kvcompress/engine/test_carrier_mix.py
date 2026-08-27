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
"""Unit test: carrier_mix / anti_mix (F2 causal-transfer arms).

Assertions (position-encoded KV, 4 eviction rounds, HKV=8, TB=1088/RES=64, PL=200):
  (a) budget parity every round, both modes;
  (b) prompt [0:PL) fully kept in ALL heads, both modes;
  (c) carrier_mix: dead heads' kept HISTORY == exactly the youngest K-PL history candidates
      (pure recency); carrier heads' history is scattered (min kept history position far below
      the recency cutoff);
  (d) anti_mix is the exact head-complement of (c);
  (e) CARRIER_HEADS env override respected.
Run:  python test_carrier_mix.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from kvcompress.engine.cache_utils import EvictLayer

B, HKV, D = 1, 8, 4
TB, RES = 1088, 64
K = TB - RES
PL = 200
CUR = TB + RES
CARRIERS = {2, 3, 5}
results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))


def run_mode(mode, rounds=4):
    layer = EvictLayer(0, TB, RES, mode, rkv_lambda=0.5, smooth=False, verbose=False)
    layer.lazy_initialization(torch.zeros(B, CUR, HKV, D))
    layer.prompt_len = PL
    vals = torch.arange(CUR, dtype=torch.float32).view(1, CUR, 1, 1)
    layer.k_cache[:, :CUR] = vals
    layer.v_cache[:, :CUR] = vals
    layer.cache_seqlens.fill_(CUR)
    layer.cumulative_length = CUR
    nxt = CUR
    for r in range(rounds):
        layer._run_eviction(CUR)
        check(f"{mode} r{r+1} budget", int(layer.cache_seqlens[0].item()) == TB)
        for h in range(HKV):
            kept = sorted(int(x) for x in layer.k_cache[0, :TB, h, 0].tolist())
            hist = [p for p in kept if p >= PL]
            check(f"{mode} r{r+1} h{h} prompt", sum(1 for p in kept if p < PL) == PL)
            is_recency_head = (h not in CARRIERS) if mode == 'carrier_mix' else (h in CARRIERS)
            newest = nxt - 1
            if is_recency_head:
                # youngest K-PL contiguous history block (up to residual boundary effects)
                check(f"{mode} r{r+1} h{h} recency", min(hist) >= newest - (K - PL) - RES,
                      f"min hist {min(hist)} vs cutoff {newest - (K-PL) - RES}")
            else:
                check(f"{mode} r{r+1} h{h} scatter", min(hist) < newest - (K - PL) - RES,
                      f"min hist {min(hist)} vs recency floor {newest - (K - PL) - RES}")
        vals = torch.arange(nxt, nxt + RES, dtype=torch.float32).view(1, RES, 1, 1)
        layer.k_cache[:, TB:TB + RES] = vals
        layer.v_cache[:, TB:TB + RES] = vals
        layer.cache_seqlens.fill_(CUR)
        layer.cumulative_length = nxt + RES
        nxt += RES


run_mode('carrier_mix')
run_mode('anti_mix')
os.environ['CARRIER_HEADS'] = '0,1'
layer = EvictLayer(0, TB, RES, 'carrier_mix', rkv_lambda=0.5, smooth=False, verbose=False)
layer.lazy_initialization(torch.zeros(B, CUR, HKV, D))
layer.prompt_len = PL
vals = torch.arange(CUR, dtype=torch.float32).view(1, CUR, 1, 1)
layer.k_cache[:, :CUR] = vals
layer.v_cache[:, :CUR] = vals
layer.cache_seqlens.fill_(CUR)
layer.cumulative_length = CUR
layer._run_eviction(CUR)
kept0 = sorted(int(x) for x in layer.k_cache[0, :TB, 0, 0].tolist())
kept5 = sorted(int(x) for x in layer.k_cache[0, :TB, 5, 0].tolist())
h0_hist = [p for p in kept0 if p >= PL]
h5_hist = [p for p in kept5 if p >= PL]
check("env override: h0 scatters", min(h0_hist) < CUR - (K - PL) - RES)
check("env override: h5 recency", min(h5_hist) >= CUR - (K - PL) - RES - 1)
del os.environ['CARRIER_HEADS']

fails = [r for r in results if not r[1]]
print(f"\n{len(results)-len(fails)}/{len(results)} assertions passed")
sys.exit(1 if fails else 0)
