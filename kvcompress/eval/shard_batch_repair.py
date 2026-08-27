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
"""Repair 'done-but-short' cells caused by batch-leader resume gating (found 2026-07-27).

Mechanism: eval_hf resumes a (run-group, shard) slot from get_ckpt_start_i computed ONLY from the
group's leader run (run_id multiple of batch_size). Consequences:
  STUCK    : leader complete, follower short  -> resume exits 'Already completed'; the follower's
             missing tail is unreachable by any batch_exist pass.
  MISALIGN : leader short, follower shorter   -> resume appends from the leader's position; the
             follower's file gets ex>=leader_start appended AFTER its own gap, so line index no
             longer equals ex index => grading pairs the wrong gold. WORSE than missing.
Repair unit = the whole (run-group, shard): quarantine every group member's completions_shard and
other_info_shard for that shard (mv to _attic_batchrepair_<ts>/), then rerun the arm; with the
files gone, start_i=0 and the group regenerates that shard cleanly.

Usage: shard_batch_repair.py <subdir> <n_problems> <batch_size> [--apply]
  default = audit only (prints groups + classification); --apply performs the quarantine.
Detects MISALIGN-already-happened by checking follower_lines > leader_start-consistent patterns is
impossible post-hoc from counts alone — any short group is quarantined wholesale, which also heals
prior misalignment as long as the group is regenerated in full.
"""
import glob
import json
import os
import re
import shutil
import sys


def main():
    subdir, nprob, bs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    apply = "--apply" in sys.argv
    runs = sorted(int(d.split("_")[-1]) for d in os.listdir(subdir) if d.startswith("run_"))
    ex_starts = sorted({int(m.group(1)) for f in glob.glob(os.path.join(subdir, "run_*", "completions_shard*.jsonl"))
                        for m in [re.search(r"completions_shard(\d+)\.jsonl", f)] if m})
    expected = {}
    for i, s in enumerate(ex_starts):
        expected[s] = (ex_starts[i + 1] - s) if i + 1 < len(ex_starts) else (nprob - s)
    n_bad = 0
    for leader in range(0, max(runs) + 1, bs):
        group = [r for r in range(leader, leader + bs) if r in runs]
        if not group:
            continue
        for s, exp in expected.items():
            counts = {}
            for r in group:
                f = os.path.join(subdir, f"run_{r}", f"completions_shard{s}.jsonl")
                counts[r] = sum(1 for _ in open(f)) if os.path.exists(f) else 0
            if all(c >= exp for c in counts.values()):
                continue
            lead_c = counts[leader]
            cls = "STUCK" if lead_c >= exp else ("MISALIGN-RISK" if any(c != lead_c for c in counts.values()) else "RESUMABLE")
            print(f"{cls} group_lead=run_{leader} shard{s}: " +
                  " ".join(f"r{r}={c}/{exp}" for r, c in counts.items()))
            n_bad += 1
            if apply and cls != "RESUMABLE":
                # attic OUTSIDE the method dir: an attic inside the subdir pollutes every
                # recursive count (dn gates, cell_integrity, watchers) — found 07-27
                base = os.path.dirname(os.path.dirname(os.path.abspath(subdir)))
                attic = os.path.join(base, "_attic_batchrepair_" + os.path.basename(subdir))
                for r in group:
                    os.makedirs(os.path.join(attic, f"run_{r}"), exist_ok=True)
                    for pat in (f"completions_shard{s}.jsonl", f"other_info_shard{s}.json"):
                        src = os.path.join(subdir, f"run_{r}", pat)
                        if os.path.exists(src):
                            shutil.move(src, os.path.join(attic, f"run_{r}", pat))
                print(f"  -> quarantined group run_{leader}..{group[-1]} shard{s} to _attic_batchrepair/")
    print(f"\n{n_bad} short (group,shard) slots" + (" (quarantine applied where needed)" if apply else " (audit only)"))


if __name__ == "__main__":
    main()
