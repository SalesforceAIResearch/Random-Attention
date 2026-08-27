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
"""Item 0a (WHY_RANDOM_WORKS_PLAN): carrier-restricted survival mass + correlation.

Re-runs the whole-evidence survival-by-age analysis from KEEPLOG_POS dumps, restricted to the
synthetically-identified storage carrier heads of Qwen3-4B (KV heads {2,3,5}; engine indexing —
identical to synth_pin's head indexing). Per method x age bin:

  s_car(a)   mean per-head survival within the carrier set (the functionally relevant mass/3)
  s_dead(a)  mean survival in the 5 non-carrier heads (control: methods shouldn't differ by set)
  u_car      measured any-of-3-carriers coverage
  ceil3      1-(1-s_car)^3   (conditional-independence ceiling over carriers)
  deficit    ceil3 - u_car   (cross-head correlated exclusion, carriers only)

Registered gates (WHY_RANDOM_WORKS_PLAN item 0): rp carrier mass >= 1.8x vase in the 1-2k band;
vase carrier deficit >= 0.10, (triattn <= 0.05 when its dump lands).

Usage: carrier_mass.py <panel_dir> <method> [method...]
"""
import sys
import glob

import numpy as np

CARRIERS = [2, 3, 5]
BINS = [(128, 256), (256, 512), (512, 1024), (1024, 2048), (2048, 4096), (4096, 8192)]
STRIDE, MIN_SPAN, MIN_BIN = 11, 4000, 20


def table(mdir):
    fs = sorted(glob.glob(f"{mdir}/pos/*.npz"))
    nb = len(BINS)
    SC = np.zeros(nb); SD = np.zeros(nb); UC = np.zeros(nb); CE = np.zeros(nb); N = np.zeros(nb)
    used = 0
    for f in fs[::STRIDE]:
        d = np.load(f)
        pos, C, pl = d['pos'], int(d['cum_len']), int(d['prompt_len'])
        span = C - pl
        if span < MIN_SPAN:
            continue
        used += 1
        H = pos.shape[0]
        dead = [h for h in range(H) if h not in CARRIERS]
        cnt_c = np.zeros(span, dtype=np.int16)
        cnt_d = np.zeros(span, dtype=np.int16)
        for h in CARRIERS:
            idx = pos[h][(pos[h] >= pl)] - pl
            cnt_c[idx] += 1
        for h in dead:
            idx = pos[h][(pos[h] >= pl)] - pl
            cnt_d[idx] += 1
        ages = C - (np.arange(span) + pl)
        for i, (lo, hi) in enumerate(BINS):
            m = (ages >= lo) & (ages < hi)
            if m.sum() < MIN_BIN:
                continue
            sc = cnt_c[m].mean() / len(CARRIERS)
            SC[i] += sc
            SD[i] += cnt_d[m].mean() / len(dead)
            UC[i] += (cnt_c[m] > 0).mean()
            CE[i] += 1 - (1 - sc) ** len(CARRIERS)
            N[i] += 1
    name = mdir.rstrip('/').split('/')[-1]
    print(f"== {name} (deep events {used})")
    print(f"{'age':>11} {'s_car':>7} {'s_dead':>7} {'u_car':>7} {'ceil3':>7} {'deficit':>8}")
    rows = {}
    for i, (lo, hi) in enumerate(BINS):
        if N[i]:
            r = (SC[i] / N[i], SD[i] / N[i], UC[i] / N[i], CE[i] / N[i],
                 CE[i] / N[i] - UC[i] / N[i])
            rows[(lo, hi)] = r
            print(f"{lo:>5}-{hi:<5} {r[0]:7.3f} {r[1]:7.3f} {r[2]:7.3f} {r[3]:7.3f} {r[4]:8.3f}")
    return rows


if __name__ == "__main__":
    base = sys.argv[1]
    all_rows = {}
    for m in sys.argv[2:]:
        all_rows[m] = table(f"{base}/{m}")
    band = (1024, 2048)
    if 'random_pp' in all_rows and 'vase_faithful' in all_rows:
        rp = all_rows['random_pp'].get(band)
        vs = all_rows['vase_faithful'].get(band)
        if rp and vs:
            print(f"\nGATE (1-2k band): rp carrier mass / vase = {rp[0]/vs[0]:.2f}x "
                  f"(registered >= 1.8x); vase carrier deficit = {vs[4]:.3f} (registered >= 0.10); "
                  f"rp deficit = {rp[4]:.3f}")
