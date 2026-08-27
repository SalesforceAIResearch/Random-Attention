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
"""Build LCB-v6 difficulty subsets with the SAME pipeline as the medium set (download_tests.py).

datasets>=5 dropped dataset-script support, so this reads the cached raw files of
livecodebench/code_generation_lite directly (release_v6 = test.jsonl..test6.jsonl concatenated
in order — verified against code_generation_lite.py's ALLOWED_FILES/_generate_examples).
Everything else (dataclass parsing incl. compressed private tests, prompt template, md5,
global_id = enumerate index over the FULL v6 set) is imported from download_tests.py verbatim.

Validation gate: `python download_tests_subset.py medium /tmp/x` must reproduce
data/livecodebench/test.jsonl byte-for-byte before any hard/easy output is trusted.

Usage: python download_tests_subset.py <difficulty> [out_root]
       out_root defaults to ../livecodebench_<difficulty>
"""
import glob
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

from download_tests import (
    CodeGenerationProblem,
    Difficulty,
    get_qwen_reasoning_question_template_answer,
    calculate_string_md5,
)

V6_FILES = ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"]


def snapshot_dir():
    pat = os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--livecodebench--code_generation_lite/snapshots/*/")
    dirs = sorted(glob.glob(pat))
    assert dirs, "code_generation_lite cache not found"
    return dirs[-1]


if __name__ == "__main__":
    name = sys.argv[1].lower()
    diff = Difficulty(name)
    out_root = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(f"../livecodebench_{name}")
    tests_dir = out_root / "livecodebench_tests"
    tests_dir.mkdir(parents=True, exist_ok=True)

    snap = snapshot_dir()
    prompt_template = get_qwen_reasoning_question_template_answer

    rows, global_id = [], 0
    for fn in V6_FILES:
        with open(os.path.join(snap, fn)) as f:
            for line in tqdm(f, desc=fn):
                line_data = json.loads(line)
                if line_data["difficulty"] == diff.value:
                    sample = CodeGenerationProblem(**line_data)
                    inputs_outputs = sample.get_evaluation_sample()
                    rows.append({
                        "global_id": global_id,
                        "question_id": sample.question_id,
                        "contest_id": sample.contest_id,
                        "contest_date": sample.contest_date.isoformat(),
                        "prompt": prompt_template(sample),
                        "tests": {
                            "fname": f"{global_id}.json",
                            "md5": calculate_string_md5(json.dumps(inputs_outputs)),
                        },
                    })
                    with open(tests_dir / f"{global_id}.json", "w") as tf:
                        json.dump(inputs_outputs, tf)
                global_id += 1

    with open(out_root / "test.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{name}: {len(rows)}/{global_id} problems -> {out_root}/test.jsonl")
