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
"""Cell integrity auditor — catches the two silent partial-cell pathologies BEFORE grading:

  MIXED  : a method dir contains >1 subdir for the same data_name (e.g. aime25_bs2* AND
           aime25_bs4*): the grader pools them, and if one is a crash-partial its survivors
           bias the pooled grade (documented: phi r-kv aime +6-7pp inflation, 07-24).
  RAGGED : within one subdir, per-shard completion counts deviate from the modal per-shard
           pattern (allowing one remainder shard): a dead worker's shard queue died mid-way,
           and because shard queues process problems IN ORDER (and AIME-style sets order by
           difficulty), missing mass concentrates on the HARDEST problems -> upward bias.
  OVERLAP: within ONE run dir, a shard's line count runs past the next shard's start index, so
           two files claim the same problem indices (grading is positional: gi = shard + line).
           Cause: a cell first run under an N-shard layout and later resumed under a 2N-shard one
           — batch_exist sees the coarse shards as done and writes only the NEW odd shards, whose
           ranges sit inside the coarse ones. Found 08-02 on 14B MATH faithful-VaSE: run_0/run_1
           hold an 8-shard layout (shards 0,62,...,434 at 62 lines = all 500 problems) PLUS a
           16-shard overlay (shards 31,93,...,465 = 252 duplicate lines). grader.py raises
           POSITIONAL CORRUPTION on it, but this auditor — whose job is to catch that BEFORE
           grading — passed the cell clean, because the RAGGED check only compares counts to a
           modal value and never tests the ranges. Overlap is now checked explicitly.

Usage:  cell_integrity.py <gate2_root> [--only substring]   # prints one line per flag + summary
Grading discipline: NO cell is graded/folded unless this auditor reports it CLEAN or the flag is
explicitly annotated in the report.
"""
import collections
import glob
import os
import re
import sys


# aime25 is 30, NOT 29. The old "aime25=29 EFFECTIVE (trailing-newline loader drop)" note was
# wrong: `wc -l` undercounts a file whose last line has no trailing newline, but the loader reads
# 30 non-empty records and grader.py reports n_gold=30. With 29 here, every aime25 run whose
# problem-29 slot had been filled by an exact-slot filler (--ex_start_i 29) looked like a tail
# overrun — 48 false OVERLAP flags on the phi AIME-2x cell alone, which grades clean at n=480.
NPROB_G = {"math": 500, "gpqa": 198, "aime24": 30, "aime25": 30, "aime26": 30,
           "hmmt": 60, "livecodebench": 383, "livecodebench_hard": 350,
           "livecodebench_easy": 322, "livecodebench_sub120": 120}


def audit_method_dir(mdir):
    flags = []
    # group subdirs by data_name prefix (text before _bs)
    subs = [d for d in glob.glob(os.path.join(mdir, "*")) if os.path.isdir(d)]
    byname = collections.defaultdict(list)
    for d in subs:
        base = os.path.basename(d)
        m = re.match(r"([a-z0-9_]+?)_bs\d+", base)
        if m:
            byname[m.group(1)].append(d)
    for name, ds in byname.items():
        if len(ds) > 1:
            tot = {os.path.basename(d): sum(1 for f in glob.glob(os.path.join(d, "**", "completions_shard*.jsonl"), recursive=True) for _ in open(f)) for d in ds}
            flags.append(("MIXED", mdir, name, tot))
    # OVERLAP: must be checked PER RUN DIR (grader.py's seen-set is per run dir) and against shard
    # START INDICES, not against a modal count. Merging runs the way the RAGGED check below does
    # would mask it: shard0 62+62+31+31 and shard31 31+31+31+31 look like plain count spread.
    for d in subs:
        base_d = os.path.basename(d)
        dn = next((k for k in sorted(NPROB_G, key=len, reverse=True) if base_d.startswith(k)), None)
        for run in sorted(glob.glob(os.path.join(d, "run_*"))) or [d]:
            per = {}
            for f in glob.glob(os.path.join(run, "completions_shard*.jsonl")):
                m = re.search(r"completions_shard(\d+)\.jsonl$", os.path.basename(f))
                if m:
                    per[int(m.group(1))] = sum(1 for _ in open(f))
            ks = sorted(per)
            for a, b in zip(ks, ks[1:]):
                if per[a] > b - a:
                    flags.append(("OVERLAP", run, f"shard{a} has {per[a]} lines but shard{b} starts at {b}",
                                  f"{per[a] - (b - a)} problem indices claimed twice -> grader.py raises"))
                    break
            else:
                if ks and dn and ks[-1] + per[ks[-1]] > NPROB_G[dn]:
                    flags.append(("OVERLAP", run, f"tail shard{ks[-1]} + {per[ks[-1]]} lines",
                                  f"overruns n_prob {NPROB_G[dn]}"))
    for d in subs:
        shard_counts = collections.Counter()
        for f in glob.glob(os.path.join(d, "**", "completions_shard*.jsonl"), recursive=True):
            m = re.search(r"completions_shard(\d+)", f)
            shard_counts[int(m.group(1))] += sum(1 for _ in open(f))
        if len(shard_counts) < 2:
            continue
        counts = [shard_counts[s] for s in sorted(shard_counts)]
        modal = collections.Counter(counts[:-1]).most_common(1)[0][0] if len(counts) > 1 else counts[0]
        ragged = [(s, c) for s, c in sorted(shard_counts.items())
                  if c != modal and not (s == max(shard_counts) and c > modal)]
        short = [(s, c) for s, c in ragged if c < modal]
        if short:
            flags.append(("RAGGED", d, f"modal={modal}", short))
        # remainder-shard blind spot (07-25): a larger tail shard can itself be short of its own
        # target (crashed mid-queue on its LAST = hardest problems). Exact check via dataset sizes:
        # infer data_name from the subdir name, R from the modal one-problem shard count.
        NPROB = {"math": 500, "gpqa": 198, "aime24": 30, "aime25": 30, "aime26": 30,  # both 30 non-empty records; the old "aime25=29 EFFECTIVE" note was WRONG (wc -l undercounts a file with no trailing newline) and made every exact-slot-filled aime25 run look like a tail OVERLAP (48 false positives, 08-02). grader.py reports n_gold=30.
                 "hmmt": 60, "livecodebench": 383, "livecodebench_hard": 350,
                 "livecodebench_easy": 322, "livecodebench_sub120": 120}
        base = os.path.basename(d)
        dname = next((k for k in sorted(NPROB, key=len, reverse=True) if base.startswith(k)), None)
        if dname and modal > 0:
            expected = NPROB[dname] * modal  # holds when shards 0..n-2 carry 1 problem each (modal==R)
            n_sh = len(shard_counts)
            # `expected` is only meaningful under its stated precondition: ONE problem per shard, so
            # the shard count must be ~= the problem count. When shards carry RANGES (16 shards over
            # 322 problems, say) modal is lines-per-shard, not R, and expected becomes nonsense --
            # it computed 322*60 = 19320 for a complete 966-line cell. That misfire produced 364 of
            # the 367 flags in the 08-02 tree sweep and buried the 14 real OVERLAPs. Gate on it.
            if n_sh < NPROB[dname] * 0.9:
                continue
            if all(shard_counts[s] == modal for s in sorted(shard_counts)[:-1]):
                total = sum(shard_counts.values())
                if total < expected:
                    flags.append(("TAILSHORT", d,
                                  f"total {total} < expected {expected} (R={modal}); "
                                  f"missing mass is in the tail shard's LAST (hardest) problems", []))
    return flags


def completeness(root, only=None):
    """Blunt invariant: EVERY subdir's total vs its dataset target (n_problems x R inferred from
    max per-problem multiplicity is unreliable — use R from dir count/n_problems rounding).
    Prints every sub-target cell. A quoted number whose cell appears here is NOT reportable
    without a partial flag. This is the check that should have existed from day one."""
    NPROB = {"math": 500, "gpqa": 198, "aime24": 30, "aime25": 30, "aime26": 30,  # both 30 non-empty records; the old "aime25=29 EFFECTIVE" note was WRONG (wc -l undercounts a file with no trailing newline) and made every exact-slot-filled aime25 run look like a tail OVERLAP (48 false positives, 08-02). grader.py reports n_gold=30.
             "hmmt": 60, "livecodebench_sub120": 120, "livecodebench_hard": 350,
             "livecodebench_easy": 322, "livecodebench": 383}
    n_flag = 0
    for sub in sorted(glob.glob(os.path.join(root, "*", "*", "*", "*"))):
        if not os.path.isdir(sub) or "_attic" in sub or "/_" in sub.replace(root, ""):
            continue
        if only and only not in sub:
            continue
        base = os.path.basename(sub)
        dname = next((k for k in sorted(NPROB, key=len, reverse=True) if base.startswith(k)), None)
        if not dname:
            continue
        total = sum(1 for f in glob.glob(os.path.join(sub, "**", "completions_shard*.jsonl"),
                                          recursive=True) for _ in open(f))
        if total == 0:
            continue
        n = NPROB[dname]
        r_est = max(1, round(total / n))
        # nearest plausible R in {1,2,4,8,16,32}; target = smallest standard R >= observed density
        Rs = [1, 2, 4, 8, 16, 32]
        r_target = min([r for r in Rs if r >= total / n] or [32])
        target = n * r_target
        if total < target:
            n_flag += 1
            print(f"INCOMPLETE {sub.replace(root + '/', '')}: {total}/{target} (R~{r_target})")
    print(f"\nCOMPLETENESS: {n_flag} sub-target cells")


def main():
    root = sys.argv[1]
    if "--completeness" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
        completeness(root, only)
        return
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1]
    n_dirs = 0
    all_flags = []
    for mdir in sorted(glob.glob(os.path.join(root, "*", "*", "*"))):
        if not os.path.isdir(mdir) or mdir.count("_attic") or "/_" in mdir.replace(root, ""):
            continue
        if only and only not in mdir:
            continue
        n_dirs += 1
        fl = audit_method_dir(mdir)
        for f in fl:
            print(f[0], f[1].replace(root + "/", ""), f[2], f[3], flush=True)
        all_flags += fl
    print(f"\nAUDITED {n_dirs} method dirs: {len([f for f in all_flags if f[0]=='MIXED'])} MIXED, "
          f"{len([f for f in all_flags if f[0]=='RAGGED'])} RAGGED, "
          f"{len([f for f in all_flags if f[0]=='OVERLAP'])} OVERLAP, "
          f"{len([f for f in all_flags if f[0]=='TAILSHORT'])} TAILSHORT")
    if n_dirs == 0:
        print("WARNING: 0 method dirs matched. This tool globs <root>/*/*/* = model/task/method, so "
              "<root> must be the gate2 ROOT. Pointing it at a single cell silently audits run dirs "
              "and reports everything clean.", file=sys.stderr)
    # Nonzero exit on the two flags that make a cell UNGRADEABLE (both graders now raise on
    # OVERLAP; MIXED silently pools two layouts). This is what makes the auditor usable as a
    # preflight gate in a launcher:
    #     cell_integrity.py gate2 --only <cell> || { echo "repair first"; exit 1; }
    # RAGGED/TAILSHORT are advisory (coverage, not correctness) and do not fail the gate.
    blocking = [f for f in all_flags if f[0] in ("OVERLAP", "MIXED")]
    if blocking:
        print(f"BLOCKING: {len(blocking)} OVERLAP/MIXED flag(s) — these cells cannot be graded "
              f"until repaired (trim_shard_overlaps.py / quarantine the stray subdir).", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
