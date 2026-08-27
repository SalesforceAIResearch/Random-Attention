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
# HANDOFF-0818 item 0: 32k MAX-BATCH capacity point (c64 corrected-runtime protocol).
# 512 prompts, 1k in / 32k out; compressed arms at the preemption-safe ceilings (224/224/96 —
# the 8k caps; 32k caches are budget-bound to the same size); dense NOT re-run (its 32k
# steady-state tok/s is pool-bound at the same resident cap regardless of offered count).
# Order: one model fully at a time (4B -> 14B -> 32B). ~20h total single-GPU.
# Required env: WORK, M4, M14, M32, STATS4, STATS14, STATS32.
set -eu
: "${WORK:?}"; : "${M4:?}"; : "${M14:?}"; : "${M32:?}"; : "${STATS4:?}"; : "${STATS14:?}"; : "${STATS32:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
. "$WORK/venv/bin/activate"
export VLLM_LOGGING_CONFIG_PATH="$HERE/logcfg.json"
BASE64="TRIATTN_RUNTIME_DIVIDE_LENGTH=64 TRIATTN_RUNTIME_WINDOW_SIZE=128 TRIATTN_RUNTIME_PROTECT_PREFILL=1 TRIATTN_RUNTIME_LOG_DECISIONS=0"
OUT="$WORK/cap32k_results"; mkdir -p "$OUT"
note(){ echo "[$(date -u '+%m-%d %H:%M')] $*" | tee -a "$OUT/driver.log"; }
arm(){ local MODEL=$1 TAG=$2 STATS=$3 SEQS=$4 A=$5
  local E
  case $A in
    rp)  E="ENABLE_TRIATTENTION=1 $BASE64 TRIATTN_RUNTIME_KV_BUDGET=2048 TRIATTN_RUNTIME_SELECTOR=random_pp TRIATTN_RUNTIME_RANDOM_SEED=1234";;
    tri) E="ENABLE_TRIATTENTION=1 $BASE64 TRIATTN_RUNTIME_KV_BUDGET=2048 TRIATTN_RUNTIME_SPARSE_STATS_PATH=$STATS";;
  esac
  note "out32k_cap $TAG $A start (seqs=$SEQS)"
  env $E vllm bench throughput --model "$MODEL" --dtype bfloat16 --max-model-len 34048 \
    --enforce-eager --no-enable-prefix-caching --random-input-len 1024 --random-output-len 32768 \
    --num-prompts 512 --max-num-seqs $SEQS > "$OUT/out32k_cap_${TAG}_${A}.log" 2>&1 || note "$TAG $A FAILED rc=$?"
  local P=$(grep -ciE "preempt" "$OUT/out32k_cap_${TAG}_${A}.log" || true)
  note "out32k_cap $TAG $A done: $(grep -m1 'Throughput' $OUT/out32k_cap_${TAG}_${A}.log || echo NO-RESULT) | preemption-lines=$P (MUST be 0)"
}
arm "$M4"  4B  "$STATS4"  224 rp;  arm "$M4"  4B  "$STATS4"  224 tri
arm "$M14" 14B "$STATS14" 224 rp;  arm "$M14" 14B "$STATS14" 224 tri
arm "$M32" 32B "$STATS32"  96 rp;  arm "$M32" 32B "$STATS32"  96 tri
tar czf "$WORK/cap32k_results.tgz" -C "$WORK" cap32k_results
note "CAP32K DONE -> $WORK/cap32k_results.tgz (send back)"
