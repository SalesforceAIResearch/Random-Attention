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
"""Whole-evidence union analysis from KEEPLOG_POS dumps (evidence panel).

Per age-of-token bin, per method:
  s(a)   = mean per-head survival (fraction of KV heads holding an age-a history token)
  union  = fraction of age-a tokens held by ANY head (measured multi-head coverage)
  indep  = 1-(1-s(a))^H  (conditional-independence ceiling)
  deficit= indep - union (cross-head correlated exclusion; ~0 => heads independent within age)

Resolves the aggregate-union "paradox": comparing the whole-history union to 1-(1-qbar)^H with an
aggregate qbar overestimates the ceiling via Jensen's inequality; the honest test is per-age.
Age bin (0,128) is excluded: dump-at-eviction boundary artifact.

Usage: python evidence_union.py <panel_dir> <method> [method ...]   (dirs contain pos/*.npz)
       python evidence_union.py --tsv <out.tsv> <panel_dir> <method> [method ...]

With --tsv the per-age rows are also written as method/age_lo/age_hi/s/union/indep, which is
what the paper figure reads (figures/plot_mechanism.py panel d) so no number is transcribed.
"""
import sys, glob
import numpy as np

BINS = [(128,256),(256,512),(512,1024),(1024,2048),(2048,4096),(4096,8192),(8192,1<<30)]
STRIDE, MIN_SPAN, MIN_BIN = 11, 4000, 20

def table(mdir, collect=None):
    fs = sorted(glob.glob(f"{mdir}/pos/*.npz"))
    S=np.zeros(len(BINS)); U=np.zeros(len(BINS)); P=np.zeros(len(BINS)); N=np.zeros(len(BINS)); used=0
    for f in fs[::STRIDE]:
        d=np.load(f); pos,C,pl=d['pos'],int(d['cum_len']),int(d['prompt_len'])
        span=C-pl
        if span<MIN_SPAN: continue
        used+=1; H=pos.shape[0]
        cnt=np.zeros(span,dtype=np.int16)
        for h in range(H):
            idx=pos[h][(pos[h]>=pl)]-pl; cnt[idx]+=1
        ages=C-(np.arange(span)+pl)
        for i,(lo,hi) in enumerate(BINS):
            m=(ages>=lo)&(ages<hi)
            if m.sum()<MIN_BIN: continue
            s=cnt[m].mean()/H
            S[i]+=s; U[i]+=(cnt[m]>0).mean(); P[i]+=1-(1-s)**H; N[i]+=1
    print(f"== {mdir} (deep events {used}, H={H})")
    print(f"{'age':>11} {'s(a)':>7} {'union':>7} {'indep':>7} {'deficit':>8}")
    for i,(lo,hi) in enumerate(BINS):
        if N[i]:
            print(f"{lo:>5}-{min(hi,99999):<5} {S[i]/N[i]:>7.3f} {U[i]/N[i]:>7.3f} {P[i]/N[i]:>7.3f} {P[i]/N[i]-U[i]/N[i]:>8.3f}")
            if collect is not None:
                collect[1].append((collect[0], lo, min(hi, 99999),
                                   S[i]/N[i], U[i]/N[i], P[i]/N[i]))

def _emit_tsv(path, rows):
    with open(path, "w") as f:
        f.write("method\tage_lo\tage_hi\ts\tunion\tindep\n")
        for m, lo, hi, s_, u, p_ in rows:
            f.write(f"{m}\t{lo}\t{hi}\t{s_:.4f}\t{u:.4f}\t{p_:.4f}\n")
    print(f"wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    argv = sys.argv[1:]
    tsv = None
    if argv and argv[0] == "--tsv":
        tsv, argv = argv[1], argv[2:]
    base, rows = argv[0], []
    for m in argv[1:]:
        table(f"{base}/{m}", collect=(m, rows) if tsv else None)
    if tsv:
        _emit_tsv(tsv, rows)
