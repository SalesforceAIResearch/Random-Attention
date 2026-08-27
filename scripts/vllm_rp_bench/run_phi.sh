#!/bin/bash
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
# phi-4-reasoning cadence-64 vLLM efficiency row — same protocol as the c64 sweep
# (one H200, bf16, enforce-eager, no prefix caching, cadence 64, K2048).
# PHI DIFFERENCE: max_position_embeddings=32768, rope_scaling null -> the out32k point
# runs OL=31488 so ML=1024+31488+256=32768 exactly. Report as "out32k (31.5k)".
# DENSE for phi is MEASURED here (off arm) — no carried-over dense exists for phi.
#
# Usage on the remote:
#   MPHI=/path/to/phi-4-reasoning STATSPHI=/path/to/phi-4.pt \
#   VLLM_LOGGING_CONFIG_PATH=/path/to/c64_bundle/logcfg.json \
#   nohup bash run_phi.sh > phi_vllm.log 2>&1 &
# Prereqs: the c64 venv activated (pinned vllm 0.19.0 + shims already installed —
# the same env that produced the c64 table; do NOT reuse a live engine between arms).
set -u
: "${MPHI:?set MPHI=/path/to/phi-4-reasoning weights}"
: "${STATSPHI:?set STATSPHI=/path/to/stats_pl/phi-4.pt (shipped in this bundle)}"
OUT=${OUT:-./out_phi_vllm}; mkdir -p "$OUT"
note(){ echo "[$(date -u +%H:%M:%S)] $*"; }

# tri stats gate (same as run_all.sh)
python - "$STATSPHI" <<'PY' || { note "TRI GATE FAIL — aborting (stats file wrong/missing)"; exit 1; }
import sys, torch
d = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
meta = d.get('metadata', d if isinstance(d, dict) else {})
assert ('inv_freq' in meta) or ('omega' in meta) or bool(meta.get('model_name'))
print('TRI-GATE PASS', list(meta.keys())[:6])
PY
BASE64="TRIATTN_RUNTIME_DIVIDE_LENGTH=64 TRIATTN_RUNTIME_WINDOW_SIZE=128 TRIATTN_RUNTIME_PROTECT_PREFILL=1 TRIATTN_RUNTIME_LOG_DECISIONS=0"

bench(){ # arm npro outlen seqs label
  local ARM=$1 NPRO=$2 OL=$3 SEQS=$4 LBL=$5
  local ML=$((1024+OL+256)); [ $ML -lt 16384 ] && ML=16384
  local COMMON="--model $MPHI --dtype bfloat16 --enforce-eager --no-enable-prefix-caching"
  local THR="--random-input-len 1024 --random-output-len $OL --num-prompts $NPRO --max-num-seqs $SEQS --max-model-len $ML"
  local E
  case $ARM in
    off) E="ENABLE_TRIATTENTION=0";;
    rp)  E="ENABLE_TRIATTENTION=1 $BASE64 TRIATTN_RUNTIME_KV_BUDGET=2048 TRIATTN_RUNTIME_SELECTOR=random_pp TRIATTN_RUNTIME_RANDOM_SEED=1234";;
    tri) E="ENABLE_TRIATTENTION=1 $BASE64 TRIATTN_RUNTIME_KV_BUDGET=2048 TRIATTN_RUNTIME_SPARSE_STATS_PATH=$STATSPHI";;
  esac
  local F=$OUT/phi_${ARM}_${LBL}.log
  note "RUN phi/$ARM/$LBL (OL=$OL seqs=$SEQS ML=$ML)"
  ( env $E vllm bench throughput $COMMON $THR ) > "$F" 2>&1
  note "DONE phi/$ARM/$LBL rc=$? $(grep -ioE 'throughput.*' "$F" | tail -1)"
}

# out32k point first (the headline), then out8k capacity point
for arm in off rp tri; do bench $arm 128 31488 128 out32k; done
for arm in off rp tri; do bench $arm 512  8192 224 out8k;  done

note "=== verify compression applied (rp/tri) and off clean ==="
BAD=0
for a in rp tri; do for l in out32k out8k; do
  f=$OUT/phi_${a}_${l}.log; [ -f "$f" ] || continue
  n=$(grep -c "compression applied" "$f"); z=$(grep -c "no_compactable_groups" "$f")
  [ "$n" -gt 0 ] && [ "$z" -eq 0 ] && V=OK || { V=INVALID; BAD=1; }
  note "VERIFY $a/$l: applied=$n noop=$z -> $V"
done; done
for l in out32k out8k; do
  n=$(grep -c "compression applied" "$OUT/phi_off_${l}.log" 2>/dev/null); n=${n:-0}  # grep -c exits 1 on a 0 count: never `|| echo 0` here (yields "0\n0" -> false CONTAMINATED)
  [ "$n" -eq 0 ] && note "VERIFY off/$l: clean" || { note "VERIFY off/$l: CONTAMINATED ($n events)"; BAD=1; }
done
[ "$BAD" -eq 0 ] && note "=== PHI BENCH SET VALID — tar $OUT and ship ===" || note "=== INVALID — do NOT report ==="
