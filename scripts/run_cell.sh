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
# Canonical accuracy-cell launcher: one (model, task, method[, budget]) cell with the paper's settings.
#   scripts/run_cell.sh <model> <task> <method> [K] [--dry-run]
#   model : a directory under $RA_MODELS_DIR (Qwen3-4B | Qwen3-14B | Qwen3-32B | phi-4-reasoning | DeepSeek-R1-Distill-Llama-8B)
#   task  : math | gpqa | aime25 | aime26 | hmmt | livecodebench | livecodebench_easy | livecodebench_hard
#   method: dense | random_pp | snapkv | rkv | vase | triattn | caote | recency_pp | rkv_official
#           | snapkv_pp | rkv_pp | vase_pp        (matched-protection variants, paper Table 2 / appendix)
#   K     : KV budget in tokens (default = the paper's ~4x point for the task; see docs/METHODS.md)
# Env knobs: NGPU (default 8), WPG (workers per GPU; default per model), NUM_SHARDS (default 16 -- FIXED per
#   cell, never change it for an existing cell), CUDA_VISIBLE_DEVICES, RUNS (override rollouts/problem).
# Re-running the same command resumes the cell (batch_exist); completions land in
#   $RA_ROOT/results/<model>/<task>_K<K>/<method>/<task>_bs<bs>_budget=<K>_<engine-mode>[...]/run_<r>/completions_shard<s>.jsonl
. "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)/env.sh"
set -u
MODEL=${1:?model}; TASK=${2:?task}; METHOD=${3:?method}; K=${4:-}; DRY=0
for a in "$@"; do [ "$a" = "--dry-run" ] && DRY=1; done
[ "$K" = "--dry-run" ] && K=""
case $MODEL in
  Qwen3-4B)  SIZE=4B;  STATS=qwen3-4b.pt;  WPG_D=2 ;;
  Qwen3-14B) SIZE=14B; STATS=qwen3-14b.pt; WPG_D=2 ;;
  Qwen3-32B) SIZE=32B; STATS=qwen3-32b.pt; WPG_D=1 ;;
  phi-4-reasoning) SIZE=14B; STATS=phi-4.pt; WPG_D=2 ;;
  DeepSeek-R1-Distill-Llama-8B) SIZE=8B; STATS=deepseek.pt; WPG_D=2 ;;
  *) echo "unknown model $MODEL (add it here and, for triattn, calibrate stats_pl/<model>.pt)" >&2; exit 1 ;;
esac
# task -> default budget (the paper's ~4x point), rollout batch, rollouts per problem
case $TASK in
  math)               K=${K:-1024}; BS=2;  R=2  ;;
  gpqa)               K=${K:-2048}; BS=4;  R=4  ;;
  aime25|aime26)      K=${K:-4096}; BS=8;  R=16 ;;
  hmmt)               K=${K:-4096}; BS=8;  R=16 ;;
  livecodebench)      K=${K:-3072}; BS=4;  R=4  ;;
  livecodebench_easy) K=${K:-3072}; BS=4;  R=4  ;;
  livecodebench_hard) K=${K:-3072}; BS=1;  R=1  ;;
  *) echo "unknown task $TASK" >&2; exit 1 ;;
esac
R=${RUNS:-$R}; [ "$BS" -gt "$R" ] && BS=$R
NL=$((K/4))   # VaSE's faithful n_large = K/4 (a fixed 256 cripples it at large K)
EV="--sparsity_method eviction --token_budget $K --residual_length 64"
EXTRA_ENV=()
case $METHOD in
  dense)      ARGS="--sparsity_method dense";                                                DIR=dense ;;
  random_pp)  ARGS="$EV --eviction_mode random_pp";                                          DIR=random_pp ;;
  snapkv)     ARGS="$EV --eviction_mode attn --smooth";                                      DIR=attn ;;
  rkv)        ARGS="$EV --eviction_mode attn_rkv --rkv_lambda 0.5 --smooth";                 DIR=attn_rkv_l05 ;;
  vase)       ARGS="$EV --eviction_mode range_sink_sample_attn --n_large $NL --smooth";      DIR=vase_faithful ;;
  caote)      ARGS="$EV --eviction_mode caote --smooth";                                     DIR=caote ;;
  recency_pp) ARGS="$EV --eviction_mode recency_pp";                                         DIR=recency_pp ;;
  rkv_official) ARGS="$EV --eviction_mode rkv_official --rkv_lambda 0.1";                    DIR=rkv_official ;;
  snapkv_pp)  ARGS="$EV --eviction_mode attn_pp";                                            DIR=attn_pp ;;
  rkv_pp)     ARGS="$EV --eviction_mode attn_rkv_pp --rkv_lambda 0.5 --smooth";              DIR=attn_rkv_pp ;;
  vase_pp)    ARGS="$EV --eviction_mode range_sink_sample_attn_pp --n_large $NL --smooth";   DIR=vase_pp ;;
  triattn)    ARGS="$EV --eviction_mode triattn_ph --n_large 256";                           DIR=triattn_ph_memofix
              EXTRA_ENV=(TRIATTN_FIX_MEMO=1 TRIATTN_STATS="$RA_ENGINE/stats_pl/$STATS")
              [ -f "$RA_ENGINE/stats_pl/$STATS" ] || { echo "missing $RA_ENGINE/stats_pl/$STATS (kvcompress/harness/calibrate_triattn_pl.py)" >&2; exit 2; } ;;
  *) echo "unknown method $METHOD" >&2; exit 1 ;;
esac
case $TASK in aime25|aime26) TDIR=aime_K$K ;; *) TDIR=${TASK}_K$K ;; esac
OUT=$RA_ROOT/results/$MODEL/$TDIR/$DIR
MD=$RA_MODELS_DIR/$MODEL
CMD=(env NUM_SHARDS="${NUM_SHARDS:-16}" WORKERS_PER_GPU="${WPG:-$WPG_D}" "${EXTRA_ENV[@]}"
     "$RA_VENV_BIN/python" "$RA_ENGINE/parallel_run_hf_mw.py"
     --model_dir "$MD" --model_size "$SIZE" --attention_implementation flash_attention_2
     --num_gpus "${NGPU:-8}" --max_tokens 32768 --tasks "$TASK" --batch_size "$BS" --total_run "$R"
     $ARGS --output_dir "$OUT")
echo "[run_cell] $MODEL / $TASK / $METHOD / K=$K  ->  $OUT"
printf '  %q' "${CMD[@]}"; echo
[ "$DRY" = 1 ] && exit 0
[ -d "$MD" ] || { echo "no checkpoint at $MD (set RA_MODELS_DIR)" >&2; exit 2; }
mkdir -p "$OUT" "$RA_ROOT/logs"; cd "$RA_ENGINE"
"${CMD[@]}" 2>&1 | tee -a "$RA_ROOT/logs/run_${MODEL}_${TDIR}_${DIR}.log"
echo "[run_cell] done. Grade with: scripts/grade_cell.sh $MODEL $TDIR $DIR"
