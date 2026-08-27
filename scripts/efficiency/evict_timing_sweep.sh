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
# Selection-vs-compaction split: is rp's selection a measurable fraction of an eviction round?
# Answers whether the registered RP_PRECOMPUTE (Tier-3 Step 2) is worth implementing.
# rp's round time IS the compaction floor (its selection is rand+topk); every other mode's
# excess over rp is that mode's scoring cost. Single-stream, steady-state, one process per mode.
. "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)/env.sh"
set -u
cd "$RA_ENGINE"
V=$RA_VENV_BIN
export PATH="$V:$PATH" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 EVICT_TIMING=1
export PYTHONPATH="$RA_ROOT:$(pwd):${PYTHONPATH:-}"   # bench_kv imports kvcompress.engine + the harness
export TRIATTN_STATS="$(pwd)/stats_pl/qwen3-4b.pt"
B=$RA_ROOT/kvcompress/analysis/bench_kv.py
OUT=$RA_ROOT/logs/evict_timing.txt
K=${ET_K:-1024}; DL=${ET_DECODE:-4096}
# Idempotence guard: this sweep can be reached from two paths (appended step in the 121-188
# followon, plus an independent fallback watcher). Whichever arrives first does the work.
if [ "$(grep -c '^RESULT' "$OUT" 2>/dev/null || echo 0)" -ge 6 ]; then
  echo "[evict_timing] already complete ($(grep -c '^RESULT' "$OUT") RESULT lines) — skipping"; exit 0
fi
exec 9>/tmp/evict_timing.lock; flock -n 9 || { echo "[evict_timing] another instance holds the lock"; exit 0; }
: > "$OUT"   # truncate only under the lock; prior 08-09 contents were invalid (no PYTHONPATH, racing instances)
for m in dense random_pp range_sink_sample_attn attn attn_rkv triattn_ph caote; do
  echo "[$(date)] $m" >> "$OUT"
  CUDA_VISIBLE_DEVICES=${ET_GPU:-0} timeout 3600 $V/python "$B" --mode $m --token_budget $K \
    --decode_len $DL 2>&1 | grep -E "^RESULT" >> "$OUT" || echo "  FAILED $m" >> "$OUT"
done
echo "=== evict timing split (K=$K, decode=$DL) ==="; grep RESULT "$OUT"
