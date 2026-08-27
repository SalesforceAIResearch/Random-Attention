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
# SOUND throughput bench = VaSE's OFFICIAL protocol (their benchmark/run_thpt.sh): FIXED batch=16 for every
# method incl. dense, input_len=128, output_len 16384 then 32768, K=1024 (our accuracy budget).
# No max-batch extrapolation. Metric: aggregate tok/s @ fixed bs + peak GB, parsed from the JSON that
# throughput.py writes (its stdout summary KeyErrors on non-dense names; JSON uses .get -> safe).
# One method per GPU in parallel; 16k round first, then 32k round (dense@32k/bs16 ~80GB -> fits H200,
# the row VaSE's 80GB card could NOT run). Round timing: 16k ~40min, 32k ~80min.
. "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)/env.sh"
set -u
cd "$RA_ENGINE"
V=$RA_VENV_BIN
export PATH="$V:$PATH" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRIATTN_STATS="$(pwd)/stats_pl/qwen3-4b.pt"
M=$RA_MODELS_DIR/Qwen3-4B
OUTDIR=$RA_ROOT/logs/vase_protocol_bench; mkdir -p "$OUTDIR" tables
TSV=$RA_ROOT/logs/throughput_vase_protocol.tsv
one(){ # $1=gpu $2=eviction_mode $3=label $4=outlen
  local log="$OUTDIR/$3_$4.log"
  CUDA_VISIBLE_DEVICES=$1 timeout 14400 $V/python "$RA_ENGINE/vase_protocol_throughput.py" --model_path $M \
    --methods $2 --batch_size 16 --input_len 128 --output_len $4 --token_budget 1024 \
    --residual_length 64 --num_warmups 1 --num_runs 2 > "$log" 2>&1
  local rc=$? j="tables/throughput_Qwen3-4B_$4.json" dst="$OUTDIR/$3_$4.json"
  # each invocation overwrites the shared per-outlen JSON -> move it to a per-method name immediately
  [ -f "$j" ] && mv "$j" "$dst"
  if grep -qiE "out of memory|OutOfMemory" "$log"; then
    printf "%s\t%s\t16\tOOM\tOOM\n" "$3" "$4" >> "$TSV"; echo "[$3 @$4] OOM"; return
  fi
  if [ -f "$dst" ]; then
    $V/python - "$dst" "$3" "$4" >> "$TSV" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); r=list(d.values())[0][0]
print(f"{sys.argv[2]}\t{sys.argv[3]}\t16\t{r['Throughput']}\t{r['Memory']}")
PY
    echo "[$3 @$4] done: $(tail -1 "$TSV")"
  else
    printf "%s\t%s\t16\tERR(rc=%s)\t—\n" "$3" "$4" "$rc" >> "$TSV"; echo "[$3 @$4] ERROR rc=$rc (see $log)"
  fi
}
printf "method\toutput_len\tbatch\ttok_s\tpeak_GB\n" > "$TSV"
for OL in 16384 32768; do
  echo "[$(date)] === round output_len=$OL (bs=16, K=1024) ==="
  one 0 dense                  dense     $OL &
  one 1 random_pp              random_pp $OL &
  one 2 range_sink_sample_attn vase      $OL &
  one 3 attn                   snapkv    $OL &
  one 4 caote                  caote     $OL &
  one 5 triattn_ph             triattn   $OL &
  wait
done
echo "[$(date)] VASE-PROTOCOL BENCH DONE -> $TSV"; column -t "$TSV"
