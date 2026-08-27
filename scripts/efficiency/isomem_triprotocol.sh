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
# TriAttention-PROTOCOL iso-memory bench (08-09, answers three questions at once):
#   (a) adds R-KV to the efficiency table (missing everywhere; their Table 5 compares against it),
#   (b) runs at THEIR operating point — 16K decode, budget 3072 — instead of ours (32K, K=1024),
#       so our numbers can be placed next to their Table 4 (2.5x @3072 / 1.9x @4096 / 6.3x @1024),
#   (c) tests the thesis on their own protocol: if rp ~ tri ~ rkv ~ vase at matched budget, their
#       headline gain is a BUDGET/capacity effect that any selector collects, including random.
# Protocol matches theirs in shape: each method at ITS max batch that fits the GPU, aggregate tok/s.
# Ratio vs dense is hardware/model dependent (H200 143GB + 4B here vs their A100 80GB + 8B) — the
# COMPARISON ACROSS SELECTORS at fixed budget is the invariant we care about, not the absolute ratio.
. "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)/env.sh"
set -u
cd "$RA_ENGINE"
V=$RA_VENV_BIN
export PATH="$V:$PATH" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL_TAG=${TRIP_MODEL:-4B}          # 4B (default) or 14B — efficiency is reported for TWO models
case "$MODEL_TAG" in
  4B)  M=$RA_MODELS_DIR/Qwen3-4B;  STATS=stats_pl/qwen3-4b.pt;  BD=30;  BE=200 ;;
  14B) M=$RA_MODELS_DIR/Qwen3-14B; STATS=stats_pl/qwen3-14b.pt; BD=40;  BE=120 ;;
  *) echo "unknown TRIP_MODEL=$MODEL_TAG" >&2; exit 2 ;;
esac
export TRIATTN_STATS="$(pwd)/$STATS"
K=${TRIP_K:-3072}; OUTLEN=${TRIP_OUTLEN:-16384}
OUTDIR=$RA_ROOT/logs/isomem_tri; mkdir -p "$OUTDIR"
TSV=$RA_ROOT/logs/throughput_isomem_tri_${MODEL_TAG}_K${K}_out${OUTLEN}.tsv  # per-config file: a second config must never truncate the first

one(){ # $1=gpu $2=mode $3=label $4=batch [$5=extra args]
  local log="$OUTDIR/${MODEL_TAG}_$3_K${K}_out${OUTLEN}_bs$4.log"
  echo "[$(date)] RUN $3 bs=$4 K=$K outlen=$OUTLEN"
  CUDA_VISIBLE_DEVICES=$1 timeout 14400 $V/python "$RA_ENGINE/vase_protocol_throughput.py" --model_path $M \
    --methods $2 --batch_size $4 --input_len 128 --output_len $OUTLEN --token_budget $K \
    --residual_length 64 --num_warmups 0 --num_runs 1 ${5:-} > "$log" 2>&1
  if grep -qiE "out of memory|OutOfMemory" "$log"; then echo "[$3 bs=$4] OOM"; return 1; fi
  local line; line=$(grep -E '\|.*\|' "$log" | tail -1)
  local tp pk; tp=$(echo "$line"|awk -F'|' '{gsub(/ /,"",$2);print $2}'); pk=$(echo "$line"|awk -F'|' '{gsub(/ /,"",$3);print $3}')
  if [ -n "$tp" ]; then printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$3" "$K" "$OUTLEN" "$4" "$tp" "$pk" >> "$TSV"; echo "[$3 bs=$4] tok/s=$tp peak=${pk}GB"
  else printf "%s\t%s\t%s\t%s\tERR\t—\n" "$3" "$K" "$OUTLEN" "$4" >> "$TSV"; echo "[$3 bs=$4] parse ERR (see $log)"; fi
}

# Resume guard: node churn on 08-09 relocated this program five times; a restart must not redo a
# config that already completed (each is ~1 h). 5 rows = dense + 5 evictors.
if [ -f "$TSV" ] && [ "$(grep -c . "$TSV")" -ge 6 ]; then
  echo "[$(date)] $MODEL_TAG K=$K out=$OUTLEN already complete ($(($(grep -c . "$TSV")-1)) rows) — skipping"
  exit 0
fi
printf "method\tK\toutput_len\tmax_batch\ttok_s\tpeak_GB\n" > "$TSV"
echo "[$(date)] TRI-PROTOCOL ISOMEM START (K=$K, outlen=$OUTLEN, per-method max batch)"
# Evictor per-seq KV at K=3072 is 3x the K=1024 run -> max batch ~1/3 of the 584/552/... values.
# Bisect downward from a generous start; first non-OOM wins. Dense is outlen-bound, unchanged.
( one 0 dense dense $BD || one 0 dense dense $((BD-2)) || one 0 dense dense $((BD-6)) ) &
( one 1 random_pp              random_pp $BE || one 1 random_pp              random_pp $((BE*9/10)) ) &
( one 2 range_sink_sample_attn vase      $((BE*95/100)) || one 2 range_sink_sample_attn vase      $((BE*85/100)) ) &
( one 3 attn                   snapkv    $((BE*93/100)) || one 3 attn                   snapkv    $((BE*85/100)) ) &
( one 4 triattn_ph             triattn   $((BE*91/100)) || one 4 triattn_ph             triattn   $((BE*83/100)) ) &
( one 5 attn_rkv               rkv       $((BE*93/100)) "--rkv_lambda 0.5" || one 5 attn_rkv rkv $((BE*85/100)) "--rkv_lambda 0.5" ) &
wait
rm -f tables/throughput_Qwen3-*_${OUTLEN}.json
echo "[$(date)] TRI-PROTOCOL ISOMEM DONE"; column -t "$TSV"
