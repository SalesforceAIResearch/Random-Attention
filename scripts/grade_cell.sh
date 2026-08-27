#!/usr/bin/env bash
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
# Integrity-check and grade one task directory:  scripts/grade_cell.sh <model> <task_K> <method_dir>[,<method_dir>...]
#   e.g.  scripts/grade_cell.sh Qwen3-4B math_K1024 dense,random_pp,attn,vase_faithful
# Math/science tasks -> kvcompress/eval/grader.py (flag_acc / acc_strict / term_rate);
# LiveCodeBench       -> kvcompress/eval/grader_lcb.py (pass@1 against the unit tests).
# Grading is POSITIONAL (problem = shard offset + line): cell_integrity.py runs first and refuses mixed,
# ragged, or overlapping shard layouts -- never grade a cell that fails it.
. "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)/env.sh"
set -u
MODEL=${1:?model}; TDIR=${2:?task_K}; METHODS=${3:?method_dir[,method_dir]}
BASE=$RA_ROOT/results/$MODEL/$TDIR
TASK=${TDIR%_K*}
"$RA_VENV_BIN/python" "$RA_ROOT/kvcompress/eval/cell_integrity.py" "$RA_ROOT/results" || { echo "cell_integrity FAILED -- fix the cell layout before grading" >&2; exit 3; }
cd "$RA_ENGINE"
case $TASK in
  livecodebench*)
    "$RA_VENV_BIN/python" "$RA_ROOT/kvcompress/eval/grader_lcb.py" --base "$BASE" --methods "$METHODS" --data_name "$TASK" \
      --max_tokens 32768 --tests_dir "$RA_DATA_DIR/$TASK/livecodebench_tests" --problems_jsonl "$RA_DATA_DIR/$TASK/test.jsonl" --timeout 20 ;;
  aime)
    for d in aime25 aime26; do "$RA_VENV_BIN/python" "$RA_ROOT/kvcompress/eval/grader.py" --base "$BASE" --methods "$METHODS" --data_name $d --max_tokens 32768; done ;;
  *)
    "$RA_VENV_BIN/python" "$RA_ROOT/kvcompress/eval/grader.py" --base "$BASE" --methods "$METHODS" --data_name "$TASK" --max_tokens 32768 ;;
esac
echo "Paired significance vs random_pp:  python kvcompress/eval/stats_paired.py --base $BASE --data_name $TASK --method_a random_pp --methods_b <dir,dir>"
