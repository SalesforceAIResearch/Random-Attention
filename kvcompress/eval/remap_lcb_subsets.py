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
"""Rebuild a subset-generated LCB arm in the FULL-383 problem index space.

Why: 32B LCB rp was generated in two chunks — livecodebench_r263_K3072 (263 problems) and
livecodebench_sub120_K3072_R4 (120) — which together cover the full 383 exactly once (verified by
question_id: 263+120 = 383 distinct, all present in the full gold). Their completions are indexed
WITHIN each subset (gi = shard + line_index against that subset's gold ordering), so a paired test
against a full-set arm finds ZERO shared problems and dies with a ZeroDivisionError.

This writes a merged arm whose shard files are renumbered into full-set indices, so the existing
grader/paired tooling works unmodified. Output is real files (not symlinks) because the shard
NUMBER carries the index — symlinking cannot renumber.

Usage:
  python remap_lcb_subsets.py --out <merged_dir> \
      --src <subset_root>:<subset_gold.jsonl> [--src ...] --full_gold <full.jsonl>
"""
import argparse
import glob
import json
import os
import re


def load_ids(path, key="question_id"):
    return [str(json.loads(l)[key]) for l in open(path) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", action="append", required=True,
                    help="<method_dir>:<subset_gold.jsonl>, repeatable")
    ap.add_argument("--full_gold", required=True)
    ap.add_argument("--data_name", default="livecodebench")
    args = ap.parse_args()

    full_index = {qid: i for i, qid in enumerate(load_ids(args.full_gold))}
    os.makedirs(args.out, exist_ok=True)
    leaf = os.path.join(args.out, f"{args.data_name}_bs4_merged")
    written = collisions = 0
    seen = set()

    for spec in args.src:
        src, gold = spec.rsplit(":", 1)
        sub_ids = load_ids(gold)
        for cf in sorted(glob.glob(os.path.join(src, "**", "completions_shard*.jsonl"), recursive=True)):
            if f"/{args.data_name}_bs" not in cf and "_bs" not in cf:
                continue
            m = re.search(r"completions_shard(\d+)\.jsonl$", os.path.basename(cf))
            if not m:
                continue
            shard = int(m.group(1))
            run = os.path.basename(os.path.dirname(cf))
            lines = [l for l in open(cf) if l.strip()]
            oi_path = cf.replace("completions_shard", "other_info_shard").replace(".jsonl", ".json")
            glens = []
            if os.path.exists(oi_path):
                try:
                    glens = json.load(open(oi_path)).get("generate_lens", [])
                except Exception:
                    glens = []
            for i, line in enumerate(lines):
                sub_gi = shard + i
                if sub_gi >= len(sub_ids):
                    continue
                fi = full_index.get(sub_ids[sub_gi])
                if fi is None:
                    continue
                key = (run, fi)
                if key in seen:
                    collisions += 1
                    continue
                seen.add(key)
                dst_dir = os.path.join(leaf, run)
                os.makedirs(dst_dir, exist_ok=True)
                with open(os.path.join(dst_dir, f"completions_shard{fi}.jsonl"), "w") as fh:
                    fh.write(line if line.endswith("\n") else line + "\n")
                gl = [glens[i]] if i < len(glens) else []
                with open(os.path.join(dst_dir, f"other_info_shard{fi}.json"), "w") as fh:
                    json.dump({"generate_lens": gl}, fh)
                written += 1

    runs = sorted({r for r, _ in seen})
    print(f"[remap] wrote {written} completions over {len(runs)} runs -> {leaf}")
    print(f"[remap] distinct problems per run: "
          f"{ {r: sum(1 for rr, _ in seen if rr == r) for r in runs} }")
    if collisions:
        print(f"[remap] WARNING {collisions} duplicate (run, problem) pairs skipped — subsets overlap?")


if __name__ == "__main__":
    main()
