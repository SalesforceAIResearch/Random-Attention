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
"""Repair coarse/fine shard-file OVERLAPS in a bs1 cell (positional grading safety).

Overlap: two files' ex-ranges intersect (a coarse shard grew across a finer shard's start) => a
problem index appears twice => grader POSITIONAL CORRUPTION. Fix: walk each run's files in
ex_start order; when file B starts inside file A's range, TRUNCATE A to (B.start - A.start)
lines. End-truncation keeps remaining lines positionally valid. Trimmed tails are backed up to
<parent>/_attic_trimmed/ before cutting. Remaining GAPS are printed (finesplit driver fills them).

Usage: trim_shard_overlaps.py <subdir> <nprob> [--apply]
"""
import os, re, sys, glob

sub, nprob = sys.argv[1], int(sys.argv[2])
apply = "--apply" in sys.argv
attic = os.path.join(os.path.dirname(sub.rstrip("/")), "_attic_trimmed")
n_over = n_gap = 0
for rd in sorted(glob.glob(os.path.join(sub, "run_*"))):
    files = []
    for f in glob.glob(os.path.join(rd, "completions_shard*.jsonl")):
        s = int(re.search(r"shard(\d+)\.jsonl", f).group(1))
        n = sum(1 for _ in open(f))
        if n:
            files.append([s, n, f])
    files.sort()
    prev = None
    for cur in files:
        if prev is not None and cur[0] < prev[0] + prev[1]:
            keep = cur[0] - prev[0]
            n_over += 1
            print(f"OVERLAP {os.path.basename(rd)}: shard{prev[0]} ({prev[1]} lines, ex{prev[0]}-{prev[0]+prev[1]-1}) "
                  f"vs shard{cur[0]} -> truncate shard{prev[0]} to {keep}")
            if apply:
                os.makedirs(attic, exist_ok=True)
                lines = open(prev[2]).readlines()
                bak = os.path.join(attic, f"{os.path.basename(rd)}_shard{prev[0]}.tail")
                open(bak, "w").writelines(lines[keep:])
                open(prev[2], "w").writelines(lines[:keep])
                prev[1] = keep
        prev = cur
    # gap report (post-trim coverage walk)
    cov = 0
    for s, n, f in files:
        if s > cov:
            n_gap += 1
            print(f"GAP {os.path.basename(rd)}: ex{cov}-{s-1}")
        cov = max(cov, s + n)
    if cov < nprob:
        n_gap += 1
        print(f"GAP {os.path.basename(rd)}: ex{cov}-{nprob-1}")
print(f"\n{n_over} overlaps, {n_gap} gaps ({'applied' if apply else 'audit only'})")
