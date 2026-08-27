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
# ISO-MEMORY serving throughput (TriAttention-style framing): every method at ITS max batch that fits
# the 143GB H200, output_len=32768, K=1024. Complements the iso-batch (VaSE-protocol) 3.1x number.
# Evictor max batches = bisected values from the max-batch sweep (cache capped at K -> outlen-invariant).
# Dense max batch probed live: 30 -> 28 -> 26 (per-seq 32k KV ~4.5GB + 7.6GB weights).
# Results parsed from EACH JOB'S OWN LOG summary line (post .get-patch) -> no shared tables/*.json race.
# Waits for ksweep + repair to drain first.
. "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)/env.sh"
set -u
cd "$RA_ENGINE"
V=$RA_VENV_BIN
export PATH="$V:$PATH" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRIATTN_STATS="$(pwd)/stats_pl/qwen3-4b.pt"
M=$RA_MODELS_DIR/Qwen3-4B
OUTDIR=$RA_ROOT/logs/isomem; mkdir -p "$OUTDIR"
TSV=$RA_ROOT/logs/throughput_isomem.tsv
echo "[$(date)] isomem staged — waiting for ksweep + repair to drain"
while ps -eo args | grep -qE 'ksweep_throughpu[t]\.sh|bench_repai[r]\.sh|throughput\.p[y]'; do sleep 180; done
sleep 30
one(){ # $1=gpu $2=mode $3=label $4=batch ; returns 0 on success, 1 on OOM
  local log="$OUTDIR/$3_bs$4.log"
  echo "[$(date)] RUN $3 bs=$4"
  CUDA_VISIBLE_DEVICES=$1 timeout 14400 $V/python "$RA_ENGINE/vase_protocol_throughput.py" --model_path $M \
    --methods $2 --batch_size $4 --input_len 128 --output_len 32768 --token_budget 1024 \
    --residual_length 64 --num_warmups 0 --num_runs 1 > "$log" 2>&1
  if grep -qiE "out of memory|OutOfMemory" "$log"; then echo "[$3 bs=$4] OOM"; return 1; fi
  # parse the (patched) summary line: "<name> | <tok/s> | <peak GB>"
  local line; line=$(grep -E '\|.*\|' "$log" | tail -1)
  local tp pk; tp=$(echo "$line"|awk -F'|' '{gsub(/ /,"",$2);print $2}'); pk=$(echo "$line"|awk -F'|' '{gsub(/ /,"",$3);print $3}')
  if [ -n "$tp" ]; then printf "%s\t%s\t%s\t%s\n" "$3" "$4" "$tp" "$pk" >> "$TSV"; echo "[$3 bs=$4] tok/s=$tp peak=${pk}GB"
  else printf "%s\t%s\tERR\t—\n" "$3" "$4" >> "$TSV"; echo "[$3 bs=$4] parse ERR (see $log)"; fi
}
printf "method\tmax_batch\ttok_s\tpeak_GB\n" > "$TSV"
echo "[$(date)] isomem START (outlen=32768, K=1024, per-method max batch)"
( one 0 dense dense 30 || one 0 dense dense 28 || one 0 dense dense 26 ) &
one 1 random_pp              random_pp 584 &
one 2 range_sink_sample_attn vase      552 &
one 3 attn                   snapkv    544 &
one 4 caote                  caote     544 &
one 5 triattn_ph             triattn   536 &
wait
rm -f tables/throughput_Qwen3-4B_32768.json   # scratch from parallel writers; results came from logs
echo "[$(date)] ISOMEM DONE"; column -t "$TSV"
