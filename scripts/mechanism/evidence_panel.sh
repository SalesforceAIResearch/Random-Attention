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
# WHOLE-EVIDENCE SURVIVAL PANEL (07-22): position-dump keep-logs (KEEPLOG_POS) for ALL methods on
# 4B MATH K=1024 (P-A), then the matched-protection discriminating cell at 16x (P-B: rp vs vase_pp, K=256).
# Feeds: survival-vs-age curves, cross-head exclusion correlation, all-copies-lost — whole history, per method.
. "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)/env.sh"
set -u
cd "$RA_ENGINE"
VBIN=$RA_VENV_BIN
export PATH="$VBIN:$PATH" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=$RA_ROOT/logs/evidence_panel.log
KL=$RA_ROOT/results/evidence_panel
CB="--model_dir $RA_MODELS_DIR/Qwen3-4B --model_size 4B --attention_implementation flash_attention_2 --num_gpus ${NG:-8} --max_tokens 32768 --sparsity_method eviction --tasks math --batch_size 2 --total_run 2"
run(){ local lbl=$1; shift; echo "[$(date)] RUN $lbl" >>"$LOG"; "$@" >>"$LOG" 2>&1; echo "[$(date)] done $lbl" >>"$LOG"; }
done_n(){ find $1 -name 'completions_shard*.jsonl' 2>/dev/null | xargs -r cat 2>/dev/null | wc -l; }
echo "[$(date)] evidence panel START on $(hostname)" > "$LOG"
grep -q "KEEPLOG_POS" "$RA_ROOT/kvcompress/engine/cache_utils.py" || { echo "FATAL: engine lacks KEEPLOG_POS" | tee -a "$LOG"; exit 1; }
# P-A: 7-method panel, K=1024 (4x)
for spec in "random_pp|--token_budget 1024 --n_large 256 --residual_length 64 --eviction_mode random_pp|" \
            "vase_faithful|--token_budget 1024 --n_large 256 --residual_length 64 --smooth --eviction_mode range_sink_sample_attn|" \
            "snapkv|--token_budget 1024 --n_large 256 --residual_length 64 --smooth --eviction_mode attn|" \
            "rkv_l05|--token_budget 1024 --n_large 256 --residual_length 64 --smooth --rkv_lambda 0.5 --eviction_mode attn_rkv|" \
            "triattn_ph|--token_budget 1024 --n_large 256 --residual_length 64 --eviction_mode triattn_ph|TRIATTN_STATS=$(pwd)/stats_pl/qwen3-4b.pt" \
            "random_unified|--token_budget 1024 --n_large 256 --residual_length 64 --eviction_mode random_unified|" \
            "recency_pp|--token_budget 1024 --n_large 256 --residual_length 64 --eviction_mode recency_pp|"; do
  IFS='|' read -r name flags extra <<< "$spec"
  d=gate2/Qwen3-4B/_evidence_panel/math_K1024_$name
  [ "$(done_n $d)" -ge 1000 ] && { echo "SKIP $name" >>"$LOG"; continue; }
  run panel-$name env WORKERS_PER_GPU=2 NUM_SHARDS=16 KEEPLOG=$KL/$name KEEPLOG_STRIDE=6 KEEPLOG_POS=1 KEEPLOG_POS_STRIDE=16 $extra $VBIN/python parallel_run_hf_mw.py $CB $flags --output_dir $d
done
# P-B: matched-protection discriminating cell at 16x (K=256): rp vs vase_pp
for spec in "rp_K256|--token_budget 256 --n_large 64 --residual_length 64 --eviction_mode random_pp" \
            "vasepp_K256|--token_budget 256 --n_large 64 --residual_length 64 --smooth --eviction_mode range_sink_sample_attn_pp"; do
  IFS='|' read -r name flags <<< "$spec"
  d=gate2/Qwen3-4B/_evidence_panel/math_$name
  [ "$(done_n $d)" -ge 1000 ] && { echo "SKIP $name" >>"$LOG"; continue; }
  run panelB-$name env WORKERS_PER_GPU=2 NUM_SHARDS=16 KEEPLOG=$KL/$name KEEPLOG_STRIDE=6 KEEPLOG_POS=1 KEEPLOG_POS_STRIDE=16 $VBIN/python parallel_run_hf_mw.py $CB $flags --output_dir $d
done
echo "[$(date)] evidence panel DONE" >>"$LOG"
