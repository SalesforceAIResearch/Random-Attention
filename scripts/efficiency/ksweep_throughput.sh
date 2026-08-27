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
# Throughput K-SWEEP at OUR regime-map compression levels: K in {256,512,2048,4096,8192}
# (K=1024 comes from the vase-protocol bench). VaSE-official protocol otherwise: FIXED bs=16,
# input_len=128, output_len=16384, n_large=K/4 hardcoded in throughput.py (faithful at every K).
# Eviction methods only — dense is K-independent (reuse the protocol-bench dense row).
# WAITS for the running vase_protocol_bench to finish before starting.
# Pairs 1:1 with accuracy cells -> accuracy-vs-throughput Pareto figure.
. "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)/env.sh"
set -u
cd "$RA_ENGINE"
V=$RA_VENV_BIN
export PATH="$V:$PATH" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRIATTN_STATS="$(pwd)/stats_pl/qwen3-4b.pt"
M=$RA_MODELS_DIR/Qwen3-4B
OUTDIR=$RA_ROOT/logs/ksweep; mkdir -p "$OUTDIR" tables
TSV=$RA_ROOT/logs/throughput_ksweep.tsv
# wait for the vase-protocol bench (32k round) to fully finish
echo "[$(date)] ksweep staged — waiting for vase_protocol_bench to finish"
while ps -eo args | grep -qE 'vase_protocol_benc[h]\.sh|throughput\.p[y]'; do sleep 120; done
sleep 30
one(){ # $1=gpu $2=eviction_mode $3=label $4=K
  local log="$OUTDIR/$3_K$4.log"
  CUDA_VISIBLE_DEVICES=$1 timeout 10800 $V/python "$RA_ENGINE/vase_protocol_throughput.py" --model_path $M \
    --methods $2 --batch_size 16 --input_len 128 --output_len 16384 --token_budget $4 \
    --residual_length 64 --num_warmups 1 --num_runs 2 > "$log" 2>&1
  local rc=$? j="tables/throughput_Qwen3-4B_16384.json" dst="$OUTDIR/$3_K$4.json"
  [ -f "$j" ] && mv "$j" "$dst"
  if grep -qiE "out of memory|OutOfMemory" "$log"; then
    printf "%s\t%s\t16\tOOM\tOOM\n" "$3" "$4" >> "$TSV"; echo "[$3 K=$4] OOM"; return
  fi
  if [ -f "$dst" ]; then
    $V/python - "$dst" "$3" "$4" >> "$TSV" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); r=list(d.values())[0][0]
print(f"{sys.argv[2]}\t{sys.argv[3]}\t16\t{r['Throughput']}\t{r['Memory']}")
PY
    echo "[$3 K=$4] $(tail -1 "$TSV")"
  else
    printf "%s\t%s\t16\tERR(rc=%s)\t—\n" "$3" "$4" "$rc" >> "$TSV"; echo "[$3 K=$4] ERROR rc=$rc"
  fi
}
printf "method\tK\tbatch\ttok_s\tpeak_GB\n" > "$TSV"
for K in 256 512 2048 4096 8192; do          # our regime-map ladder minus the in-flight K=1024
  echo "[$(date)] === K=$K round (bs=16, outlen=16384, nl=K/4) ==="
  one 0 random_pp              random_pp $K &
  one 1 range_sink_sample_attn vase      $K &
  one 2 attn                   snapkv    $K &
  one 3 caote                  caote     $K &
  one 4 triattn_ph             triattn   $K &
  wait
done
echo "[$(date)] K-SWEEP DONE -> $TSV"; column -t "$TSV"
