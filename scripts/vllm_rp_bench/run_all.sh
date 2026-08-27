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
# Tier-2 driver: setup + smoke + 3 arms x 2 regimes, unattended. See RUNBOOK.md for the registered
# protocol. Results land in $RA_ROOT/results/vllm_bench/ as one log per run plus a
# summary tsv; every failure is loud and non-fatal to later arms.
set -u
OUT=$RA_ROOT/results/vllm_bench
LOG=$OUT/driver.log
mkdir -p "$OUT"
note(){ echo "[$(date -u '+%m-%d %H:%M')] $*" | tee -a "$LOG"; }
WORK=${VLLM_WORK:-$HOME/triattn_vllm}   # local NVMe/scratch preferred (see setup_venv.sh)

note "=== Tier-2 vLLM bench driver on $(hostname) -> $WORK ==="
if [ ! -f "$WORK/venv/bin/activate" ]; then
  note "venv setup starting"
  bash $RA_ROOT/scripts/vllm_rp_bench/setup_venv.sh "$WORK" >> "$LOG" 2>&1 \
    || { note "SETUP FAILED — see driver.log; aborting"; exit 1; }
fi
source "$WORK/venv/bin/activate"

# Tri-arm stats gate (RUNBOOK step 4): inv_freq / resolvable model_name must exist or the tri arm
# silently degrades. Hard-gate the TRI arm on it; rp/off arms don't care.
STATS=$RA_ENGINE/stats_pl/qwen3-14b.pt
TRI_OK=1
python - "$STATS" >> "$LOG" 2>&1 <<'PY' || TRI_OK=0
import sys, torch
d = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
meta = d.get('metadata', d if isinstance(d, dict) else {})
ok = ('inv_freq' in meta) or ('omega' in meta) or bool(meta.get('model_name'))
print('stats metadata keys:', list(meta.keys())[:12], 'TRI-GATE:', 'PASS' if ok else 'FAIL')
assert ok
PY
[ "$TRI_OK" = 1 ] && note "tri stats gate PASS" || note "tri stats gate FAIL — TRI ARM SKIPPED (rp/off still run)"

# Route the handler-less "triattention" logger to stdout in every vllm process — without this the
# activation banner AND LOG_DECISIONS output are silently dropped (08-08 lesson: 6 rc=0 runs, all dense).
export VLLM_LOGGING_CONFIG_PATH=$RA_ROOT/scripts/vllm_rp_bench/logcfg.json
COMMON="--model Qwen/Qwen3-14B --dtype bfloat16 --max-model-len 16384 --enforce-eager --no-enable-prefix-caching"
LAT="--input-len 1024 --output-len 8192 --batch-size 1 --num-iters 3 --num-iters-warmup 1"
# THROUGHPUT MUST use the --random-* forms: `vllm bench throughput` defaults to the random dataset,
# whose --random-input-len/--random-output-len DEFAULTS (1024/128) silently override the deprecated
# --input-len/--output-len ("will be preferred" warning). With --output-len 8192 each request decoded
# only 128 tokens -> never crossed the compression threshold -> zero triattention signals (08-08).
# `vllm bench latency` takes --input-len/--output-len natively (no dataset) and is unaffected.
THR="--random-input-len 1024 --random-output-len 8192 --num-prompts 64"
BASEENV="TRIATTN_RUNTIME_KV_BUDGET=2048 TRIATTN_RUNTIME_DIVIDE_LENGTH=128 TRIATTN_RUNTIME_WINDOW_SIZE=128 TRIATTN_RUNTIME_PROTECT_PREFILL=1 TRIATTN_RUNTIME_LOG_DECISIONS=0"

run(){ # <arm> <envs...> -- <bench> <args>
  local ARM=$1; shift
  local ENVS=(); while [ "$1" != "--" ]; do ENVS+=("$1"); shift; done; shift
  local KIND=$1; shift
  local F=$OUT/${ARM}_${KIND}.log
  note "RUN $ARM/$KIND"
  ( env "${ENVS[@]}" vllm bench "$KIND" $COMMON "$@" ) > "$F" 2>&1
  local rc=$?
  note "DONE $ARM/$KIND rc=$rc $(grep -ioE 'throughput.*|latency.*seconds|tokens/s[^ ]*' "$F" | tail -2 | tr '\n' ' | ')"
}

# off arm (eager baseline)
run off ENABLE_TRIATTENTION=0 -- latency $LAT
run off ENABLE_TRIATTENTION=0 -- throughput $THR
# rp arm
run rp ENABLE_TRIATTENTION=1 $BASEENV TRIATTN_RUNTIME_SELECTOR=random_pp TRIATTN_RUNTIME_RANDOM_SEED=1234 -- latency $LAT
run rp ENABLE_TRIATTENTION=1 $BASEENV TRIATTN_RUNTIME_SELECTOR=random_pp TRIATTN_RUNTIME_RANDOM_SEED=1234 -- throughput $THR
# tri arm (gated)
if [ "$TRI_OK" = 1 ]; then
  run tri ENABLE_TRIATTENTION=1 $BASEENV TRIATTN_RUNTIME_SPARSE_STATS_PATH=$STATS -- latency $LAT
  run tri ENABLE_TRIATTENTION=1 $BASEENV TRIATTN_RUNTIME_SPARSE_STATS_PATH=$STATS -- throughput $THR
fi

note "=== all arms attempted; verifying compression actually applied (rp/tri) ==="
BAD=0
for a in rp tri; do for k in latency throughput; do
  f=$OUT/${a}_${k}.log; [ -f "$f" ] || continue
  n=$(grep -c "compression applied" "$f"); z=$(grep -c "no_compactable_groups" "$f")
  if [ "$n" -gt 0 ] && [ "$z" -eq 0 ]; then V=OK; else V=INVALID; BAD=1; fi
  note "VERIFY $a/$k: applied=$n noop=$z -> $V"
done; done
n_off=$(grep -c "compression applied" $OUT/off_latency.log 2>/dev/null || true); n_off=${n_off:-0}
[ "$n_off" -eq 0 ] && note "VERIFY off: clean baseline (0 compression events)" || { note "VERIFY off: CONTAMINATED ($n_off events)"; BAD=1; }
[ "$BAD" -eq 0 ] && note "=== BENCH SET VALID ===" || note "=== BENCH SET INVALID — do NOT report these numbers ==="
