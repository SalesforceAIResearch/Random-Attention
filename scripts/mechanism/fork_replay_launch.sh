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
# Fan one fork's 32 replays (2 conds x 16 reps) over this node's 8 GPUs in 4 waves.
# Usage: fork_replay_launch.sh phi_gpqa 86 2 19235 1613:1614
#        fork_replay_launch.sh 4b_gpqa  61 0 11369 335:337
. "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)/env.sh"
set -u
CELL=$1; GI=$2; RUN=$3; FTOK=$4; SPAN=$5; SUFFIX=${6:-}
V=$RA_VENV_BIN/python
RP=$RA_ROOT/kvcompress/analysis/fork_replay.py
OUT=$RA_ROOT/results/fork_replay/${CELL}_gi${GI}${SUFFIX}.jsonl
mkdir -p "$(dirname "$OUT")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
job=0
for cond in control force; do
  for rep in $(seq 0 15); do
    g=$((job % 8))
    if grep -q "\"cond\": \"$cond\", \"rep\": $rep," "$OUT" 2>/dev/null; then
      echo "skip $cond rep$rep (already done)"; continue
    fi
    if [ "$cond" = force ]; then FK="$SPAN"; else FK=""; fi
    CUDA_VISIBLE_DEVICES=$g FORCE_KEEP_RANGE="$FK" $V "$RP" --cell "$CELL" --gi "$GI" \
      --run "$RUN" --f_tok "$FTOK" --cond "$cond" --rep "$rep" --out "$OUT" \
      > $RA_ROOT/logs/freplay_${CELL}${SUFFIX}_${cond}_r${rep}.log 2>&1 &
    job=$((job + 1))
    [ $((job % 8)) -eq 0 ] && wait
  done
done
wait
echo "[$(date)] FORK REPLAY DONE $CELL gi$GI" >> $RA_ROOT/logs/extra_h200.log
