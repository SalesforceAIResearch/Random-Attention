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
"""Grade VaSE-engine completions with the REPO's MATH grader (parse_ground_truth/extract_answer/
check_is_correct = proper math_equal) PLUS our termination axis. Run with PYTHONPATH from env.sh
(needs kvcompress/harness on the path). Reads completions_shard<S>.jsonl (problem index = S + line), works
on partial runs. Reports flag_acc (correct) / acc_strict (correct AND terminated) / term_rate.
Add --by_level to also break results down by MATH difficulty level (1-5, from each example's
'level' field)."""
import argparse, glob, json, os, re
import numpy as np
from Utils.parser import parse_ground_truth, extract_answer
from Utils.grader import check_is_correct
from Utils.data_loader import load_data


def grade_records(method_dir, golds, data_name, max_tokens):
    """Return (records, n_runs) where records = list of (gi, flag, term) per completion."""
    recs, runs = [], set()
    seen_by_run = {}   # run_dir -> set(gi): a gi seen twice within ONE run dir = positional corruption
    n_fallback = 0     # completions graded without generate_lens (term falls back to "non-empty")
    for cf in sorted(glob.glob(os.path.join(method_dir, "**", "completions_shard*.jsonl"), recursive=True)):
        # Only read the subdir for THIS task (subdirs are named "{task}_bs{bs}_..."). A method dir can
        # hold multiple tasks (e.g. aime_K1024/random_pp has aime25_* and aime26_*); without this filter
        # they'd be pooled against one task's golds. (MATH: "math_bs2..." matches "/math_bs"; excludes "math_l5_bs".)
        if f"/{data_name}_bs" not in cf:
            continue
        m = re.search(r"completions_shard(\d+)\.jsonl$", os.path.basename(cf))
        shard = int(m.group(1)) if m else 0
        run_dir = os.path.dirname(cf)
        runs.add(run_dir)
        seen = seen_by_run.setdefault(run_dir, set())
        comps = [json.loads(l)["completion"] for l in open(cf) if l.strip()]
        oi = cf.replace("completions_shard", "other_info_shard").replace(".jsonl", ".json")
        glens = json.load(open(oi)).get("generate_lens", []) if os.path.exists(oi) else []
        for i, c in enumerate(comps):
            gi = shard + i
            if gi >= len(golds):
                # positional overflow: resumed/duplicated lines push past the gold set — corrupt cell
                raise RuntimeError(f"POSITIONAL CORRUPTION: {cf} line {i} -> problem index {gi} >= "
                                   f"n_gold {len(golds)}. Archive this run dir and re-run fresh.")
            if gi in seen:
                raise RuntimeError(f"POSITIONAL CORRUPTION: problem index {gi} appears twice within run "
                                   f"dir {run_dir} (duplicate/resumed shard lines). Archive + re-run.")
            seen.add(gi)
            try:
                pred = extract_answer(c, data_name)
                f = int(bool(check_is_correct(pred, golds[gi])))
            except Exception:
                f = 0
            # eval_hf caps on max_LENGTH (prompt+gen), so a capped gen has generate_len ~ max_tokens-prompt
            # (MATH prompt < ~600 tok). term = generate_len well below the cap (margin 1000 > max prompt).
            if i < len(glens):
                t = int(glens[i] < max_tokens - 1000)
            else:
                t = int(bool(c.strip()))
                n_fallback += 1
            recs.append((gi, f, t))
    if n_fallback:
        print(f"# WARNING {method_dir}: {n_fallback}/{len(recs)} completions have no generate_lens — "
              f"their term/acc_strict falls back to 'completion non-empty' (NOT a real termination check)")
    return recs, len(runs)


def agg(recs, n_runs):
    n = len(recs)
    g = lambda vals: round(float(np.mean(vals)), 4) if n else None
    return dict(n=n, runs=n_runs,
                flag_acc=g([r[1] for r in recs]),
                acc_strict=g([r[1] * r[2] for r in recs]),
                term_rate=g([r[2] for r in recs]))


def grade(method_dir, golds, data_name, max_tokens):
    """Backwards-compatible: overall aggregate dict (same shape as before)."""
    recs, n_runs = grade_records(method_dir, golds, data_name, max_tokens)
    return agg(recs, n_runs)


def lvl_key(x):
    """Normalize a 'level' field ('Level 5', 5, '5') to an int 1-5, else None."""
    m = re.search(r"(\d)", str(x))
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--methods", default="dense,attn,range_sink_sample_attn,random,recency,caote,cusa")
    ap.add_argument("--data_name", default="math")
    ap.add_argument("--data_dir", default=os.environ.get("RA_DATA_DIR", os.environ.get("KVC_DATA_DIR", "./data")),
                    help="benchmark test-set dir (<data_dir>/<data_name>/test.jsonl); "
                         "defaults to $RA_DATA_DIR, else ./data")
    # DEFAULT MUST MATCH THE RUNS' GENERATION CAP (32768 for every REPORT cell). The old default of
    # 16000 silently graded all non-dense methods with a 15000-token termination threshold while
    # dense used 31768 — deflating acc_strict/term_rate for every compression method (worst on
    # long-trace tasks: 4B HMMT rp strict 0.284@15k vs 0.431@32k). Pass explicitly for 16k-cap runs.
    ap.add_argument("--max_tokens", type=int, default=32768)
    ap.add_argument("--dense_max_tokens", type=int, default=32768)
    ap.add_argument("--by_level", action="store_true",
                    help="also print per-difficulty-level breakdown (MATH 'level' field)")
    args = ap.parse_args()
    examples = load_data(args.data_name, "test", args.data_dir)
    golds = [parse_ground_truth(d, args.data_name)[1] for d in examples]
    levels = [lvl_key(d.get("level")) for d in examples]
    present = sorted({l for l in levels if l is not None})

    print(f"# repo-grader {args.base} data={args.data_name} n_gold={len(golds)} "
          f"term_threshold={args.max_tokens - 1000} (dense {args.dense_max_tokens - 1000})")
    if args.by_level:
        dist = {l: levels.count(l) for l in present}
        print(f"# level distribution (problems): {dist}")
    hdr = f'{"method":24s} {"level":>6} {"flag_acc":>8} {"acc_strict":>10} {"term_rate":>9} {"n":>6} {"runs":>5}'
    print(hdr)
    for mth in args.methods.split(","):
        md = os.path.join(args.base, mth)
        if not os.path.isdir(md):
            print(f"{mth:24s} (missing)"); continue
        mt = args.dense_max_tokens if mth == "dense" else args.max_tokens
        recs, n_runs = grade_records(md, golds, args.data_name, mt)
        s = agg(recs, n_runs)
        print(f'{mth:24s} {"all":>6} {str(s["flag_acc"]):>8} {str(s["acc_strict"]):>10} {str(s["term_rate"]):>9} {s["n"]:>6} {s["runs"]:>5}')
        if args.by_level:
            for l in present:
                sub = [r for r in recs if levels[r[0]] == l]
                sl = agg(sub, n_runs)
                print(f'{"":24s} {("L"+str(l)):>6} {str(sl["flag_acc"]):>8} {str(sl["acc_strict"]):>10} {str(sl["term_rate"]):>9} {sl["n"]:>6} {"":>5}')


if __name__ == "__main__":
    main()
