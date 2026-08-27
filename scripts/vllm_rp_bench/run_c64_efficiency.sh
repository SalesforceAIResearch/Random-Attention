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
# CADENCE-64 vLLM efficiency table (SINGLE REP, replaces the K2048/cadence-128 table).
# Required env: WORK (the scratch holding tri/ + venv from the earlier bench),
#               M4, M14, M32 (model dirs), STATS4, STATS14, STATS32 (stats_pl .pt files).
# Dense rows are NOT re-run (event-free -> cadence-invariant; carried over).
# Order: shim refresh -> accuracy companion (probe armed, log mode) -> 32k longdec (3 models)
#        -> 8k capacity (3 models) -> K3072 companion (4B/14B) -> small-batch (4B/14B) -> tar.
set -eu
: "${WORK:?}"; : "${M4:?}"; : "${M14:?}"; : "${M32:?}"; : "${STATS4:?}"; : "${STATS14:?}"; : "${STATS32:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
. "$WORK/venv/bin/activate"
export VLLM_LOGGING_CONFIG_PATH="$HERE/logcfg.json"

# --- 0. refresh shims to v4/v4b (the install predates the anchored-offset fixes) ---
cp "$HERE"/shims/*.py "$WORK/tri/triattention/vllm/runtime/"
cp "$HERE"/selector_random.py "$WORK/tri/triattention/vllm/runtime/"
find "$WORK/tri" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
python -c "import triattention; print('tri OK, shims refreshed')"

BASE64="TRIATTN_RUNTIME_DIVIDE_LENGTH=64 TRIATTN_RUNTIME_WINDOW_SIZE=128 TRIATTN_RUNTIME_PROTECT_PREFILL=1 TRIATTN_RUNTIME_LOG_DECISIONS=0"
OUT="$WORK/c64_results"; mkdir -p "$OUT"
note(){ echo "[$(date -u '+%m-%d %H:%M')] $*" | tee -a "$OUT/driver.log"; }

# --- 1. accuracy companion @ cadence 64 (K1024, 500 problems, 8 shards, probe log mode) ---
note "accuracy companion start"
for i in 0 1 2 3 4 5 6 7; do
  lo=$((i*63)); hi=$((lo+63)); [ $hi -gt 500 ] && hi=500
  env ENABLE_TRIATTENTION=1 TRIATTN_RUNTIME_SELECTOR=random_pp TRIATTN_RUNTIME_RANDOM_SEED=1234 \
    TRIATTN_RUNTIME_KV_BUDGET=1024 $BASE64 TRIATTN_ASSERT_REACHABLE=1 TRIATTN_ASSERT_RAISE=0 TRIATTN_RESERVE_GRACE=3 \
    CUDA_VISIBLE_DEVICES=$i python "$HERE/vllm_math_acc.py" --model "$M4" --runs 1 --top_k -1 \
    --lo $lo --hi $hi --out "$OUT/acc_c64" > "$OUT/acc_c64_$i.log" 2>&1 &
done
wait
note "accuracy companion done; orphan-probe lines: $(grep -h 'TRIATTN-PROBE' $OUT/acc_c64_*.log | wc -l)"

bench(){ # model tag stats budget npro outlen seqs label
  local MODEL=$1 TAG=$2 STATS=$3 BUD=$4 NPRO=$5 OL=$6 SEQS=$7 LBL=$8
  local COMMON="--model $MODEL --dtype bfloat16 --enforce-eager --no-enable-prefix-caching"
  local ML=$((1024+OL+256)); [ $ML -lt 16384 ] && ML=16384
  local THR="--random-input-len 1024 --random-output-len $OL --num-prompts $NPRO --max-num-seqs $SEQS --max-model-len $ML"
  for arm in rp tri; do
    local E
    case $arm in
      rp)  E="ENABLE_TRIATTENTION=1 $BASE64 TRIATTN_RUNTIME_KV_BUDGET=$BUD TRIATTN_RUNTIME_SELECTOR=random_pp TRIATTN_RUNTIME_RANDOM_SEED=1234";;
      tri) E="ENABLE_TRIATTENTION=1 $BASE64 TRIATTN_RUNTIME_KV_BUDGET=$BUD TRIATTN_RUNTIME_SPARSE_STATS_PATH=$STATS";;
    esac
    note "$LBL $TAG $arm start"
    env $E vllm bench throughput $COMMON $THR > "$OUT/${LBL}_${TAG}_${arm}.log" 2>&1 || note "$LBL $TAG $arm FAILED rc=$?"
    note "$LBL $TAG $arm $(grep -m1 'Throughput' $OUT/${LBL}_${TAG}_${arm}.log || echo NO-RESULT)"
  done
}
# --- 2. out32k longdec (128 prompts, K2048) ---
bench "$M4"  4B  "$STATS4"  2048 128 32768 128 out32k
bench "$M14" 14B "$STATS14" 2048 128 32768 128 out32k
bench "$M32" 32B "$STATS32" 2048 128 32768  96 out32k
# --- 3. out8k capacity (512 prompts) ---
bench "$M4"  4B  "$STATS4"  2048 512 8192 224 out8k
bench "$M14" 14B "$STATS14" 2048 512 8192 224 out8k
bench "$M32" 32B "$STATS32" 2048 512 8192  96 out8k
# --- 4. K3072 companion (32k) ---
bench "$M4"  4B  "$STATS4"  3072 128 32768 128 out32k_K3072
bench "$M14" 14B "$STATS14" 3072 128 32768 128 out32k_K3072
# --- 5. small-batch (64-prompt throughput + bs1 latency) ---
bench "$M4"  4B  "$STATS4"  2048  64 8192  64 out8k_64p
bench "$M14" 14B "$STATS14" 2048  64 8192  64 out8k_64p
for m in "$M4:4B:$STATS4" "$M14:14B:$STATS14"; do
  MODEL=${m%%:*}; rest=${m#*:}; TAG=${rest%%:*}; STATS=${rest#*:}
  for arm in rp tri; do
    case $arm in
      rp)  E="ENABLE_TRIATTENTION=1 $BASE64 TRIATTN_RUNTIME_KV_BUDGET=2048 TRIATTN_RUNTIME_SELECTOR=random_pp TRIATTN_RUNTIME_RANDOM_SEED=1234";;
      tri) E="ENABLE_TRIATTENTION=1 $BASE64 TRIATTN_RUNTIME_KV_BUDGET=2048 TRIATTN_RUNTIME_SPARSE_STATS_PATH=$STATS";;
    esac
    note "lat_bs1 $TAG $arm start"
    env $E vllm bench latency --model "$MODEL" --dtype bfloat16 --input-len 1024 --output-len 8192 \
      --batch-size 1 --num-iters 3 --num-iters-warmup 1 --max-model-len 16384 --enforce-eager \
      --no-enable-prefix-caching > "$OUT/lat_bs1_${TAG}_${arm}.log" 2>&1 || note "lat_bs1 $TAG $arm FAILED"
    note "lat_bs1 $TAG $arm $(grep -miE 'avg latency' $OUT/lat_bs1_${TAG}_${arm}.log | head -1 || echo NO-RESULT)"
  done
done
tar czf "$WORK/c64_results.tgz" -C "$WORK" c64_results
note "C64 TABLE DONE -> $WORK/c64_results.tgz (send back)"
