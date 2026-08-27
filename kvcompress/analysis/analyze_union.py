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
"""E1 union analysis. Reads the LOG_UNION csv and reports, per eviction mode:
  - union/(Hkv*K): cross-head DECORRELATION. 1.0 = heads keep disjoint sets (max breadth);
                   1/Hkv (=0.125 for 8 heads) = all heads identical (random_unified / cusa).
  - union/n_cand:  fraction of the candidate window the model retains SOMEWHERE across heads (coverage).
The union hypothesis predicts: random_pp >> selectors > random_unified on union, tracking accuracy.
Usage: python analyze_union.py logs/union_e1.csv
"""
import sys, csv
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "logs/union_e1.csv"
agg = defaultdict(lambda: {"u_hk": [], "u_nc": [], "Hkv": None, "K": None})
with open(path) as f:
    for row in csv.DictReader(f):
        m = row["mode"]
        agg[m]["u_hk"].append(float(row["union_over_HkvK"]))
        agg[m]["u_nc"].append(float(row["union_over_ncand"]))
        agg[m]["Hkv"], agg[m]["K"] = row["Hkv"], row["K"]

def mean(xs): return sum(xs) / len(xs) if xs else float("nan")

rows = [(m, mean(d["u_hk"]), mean(d["u_nc"]), len(d["u_hk"]), d["Hkv"], d["K"]) for m, d in agg.items()]
rows.sort(key=lambda r: -r[1])  # by decorrelation (union/(Hkv*K)), high -> low

print(f"\n{'mode':<26}{'union/(Hkv*K)':>15}{'union/n_cand':>14}{'n_evict':>9}  (Hkv={rows[0][4] if rows else '?'})")
print("-" * 72)
for m, uhk, unc, n, hkv, K in rows:
    print(f"{m:<26}{uhk:>15.3f}{unc:>14.3f}{n:>9}")
print("\nRead: union/(Hkv*K)=1.0 -> per-head disjoint (max union); =1/Hkv -> all heads identical.")
print("Hypothesis check: random_pp should top this; random_unified should sit at ~1/Hkv (floor).")
