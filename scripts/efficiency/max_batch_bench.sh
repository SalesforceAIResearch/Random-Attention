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
# MAX-BATCH throughput + peak-mem bench (VaSE-protocol bench: kvcompress/harness/vase_protocol_throughput.py). ONE method per GPU, parallel.
# EVICTION methods (cache capped at K -> peak mem output-INDEPENDENT): geometric ramp + BINARY SEARCH the exact
#   max fitting batch @ output_len=4096 (== the 32k max batch, far faster). Record tok/s + peak_GB @ max.
# DENSE (peak mem GROWS with gen length; set by the LONGEST = 32k cap): measure PER-SEQUENCE footprint at
#   bs=1/output_len=32768 (fast, one long decode) + a bs=1/256 baseline (~weights); max batch is then
#   (usable_mem - weights) / per_seq_KV_32k  (dense mem ~linear in batch). 4B, K=1024. Needs idle GPUs 0-5.
. "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)/env.sh"
set -u
cd "$RA_ENGINE"
V=$RA_VENV_BIN
export PATH="$V:$PATH" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRIATTN_STATS="$(pwd)/stats_pl/qwen3-4b.pt"
M=$RA_MODELS_DIR/Qwen3-4B; USABLE=140
OUTDIR=$RA_ROOT/logs/maxbatch; mkdir -p "$OUTDIR"
TSV=$RA_ROOT/logs/throughput_maxbatch.tsv
TP=""; PK=""
run_bs(){ # $1=gpu $2=meth $3=bs $4=outlen $5=nruns $6=logfile ; sets TP/PK ; return 0=fit 1=OOM
  local r; r=$(CUDA_VISIBLE_DEVICES=$1 timeout 2400 $V/python "$RA_ENGINE/vase_protocol_throughput.py" --model_path $M \
      --methods $2 --batch_size $3 --input_len 256 --output_len $4 --token_budget 1024 --residual_length 64 \
      --num_warmups 1 --num_runs $5 2>&1)
  { echo "=== bs=$3 outlen=$4 ==="; echo "$r"|grep -vE "warmup|Loading|Materializing"; } >> "$6"
  echo "$r" | grep -qiE "out of memory|OutOfMemory|CUDA error|CUDA out of" && return 1
  TP=$(echo "$r"|grep -oiE "[0-9.]+ tok/s"|grep -oE "[0-9.]+"|tail -1)
  PK=$(echo "$r"|grep -iE "peak mem|GB"|grep -oE "[0-9]+\.[0-9]+"|tail -1)
  return 0
}
sweep_evict(){ # $1=gpu $2=mode $3=label  (binary-search max batch @ 4096)
  local gpu=$1 meth=$2 lbl=$3 log="$OUTDIR/$3.log"; : > "$log"; local lo=0 hi=0 btp="" bpk=""
  for bs in 8 16 32 64 128 256 512 1024 2048; do
    if run_bs $gpu $meth $bs 4096 2 "$log"; then lo=$bs; btp=$TP; bpk=$PK; echo "[$lbl] bs=$bs FIT tok/s=$TP peak=${PK}GB"
    else hi=$bs; echo "[$lbl] bs=$bs OOM ($lo,$hi)"; break; fi
  done
  if [ "$hi" -gt 0 ] && [ "$lo" -gt 0 ]; then while [ $((hi-lo)) -gt 8 ]; do local mid=$(((lo+hi)/2))
    if run_bs $gpu $meth $mid 4096 2 "$log"; then lo=$mid; btp=$TP; bpk=$PK; else hi=$mid; fi; done; fi
  printf "%s\t%s\t%s\t%s\t4096\n" "$lbl" "$lo" "$btp" "$bpk" >> "$TSV"; echo "[$lbl] MAX bs=$lo tok/s=$btp peak=${bpk}GB"
}
dense_perseq(){ # $1=gpu : per-seq footprint at 32k -> extrapolate max batch
  local gpu=$1 log="$OUTDIR/dense.log"; : > "$log"
  run_bs $gpu dense 1 256   1 "$log"; local P0=$PK                 # ~weights+overhead
  run_bs $gpu dense 1 32768 1 "$log"; local P1=$PK T1=$TP          # weights + 1x KV@32k
  local maxb=$(awk -v u=$USABLE -v p0=$P0 -v p1=$P1 'BEGIN{k=p1-p0; if(k<=0){print 0}else{printf "%d",(u-p0)/k}}')
  printf "dense\t%s\t%s\t%s\t32768(per-seq)\n" "$maxb" "$T1" "$P1" >> "$TSV"
  echo "[dense] per-seq: P0(bs1/256)=${P0}GB P1(bs1/32k)=${P1}GB -> per-seq KV=$(awk -v a=$P1 -v b=$P0 'BEGIN{printf "%.2f",a-b}')GB -> max_bs~$maxb (usable ${USABLE}GB); bs1/32k tok/s=$T1"
}
printf "method\tmax_bs\ttok_s\tpeak_GB\tnote\n" > "$TSV"
echo "[$(date)] bench START (dense=per-seq@32k, eviction=binary-search@4k)"
dense_perseq 0 &
sweep_evict 1 random_pp              random_pp &
sweep_evict 2 range_sink_sample_attn vase      &
sweep_evict 3 attn                   snapkv    &
sweep_evict 4 caote                  caote     &
sweep_evict 5 triattn_ph             triattn   &
wait
echo "[$(date)] BENCH DONE -> $TSV"; cat "$TSV"
