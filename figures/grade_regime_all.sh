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
# Grade every FULL-CLEAN regime cell -> results.tsv (for plot_compression.py). Skips partial/corrupt/stale.
# Cols: model  task  comp  K  method  flag_acc  acc_strict  n
. "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)/env.sh"
set -u
cd "$RA_ENGINE"
VBIN=$RA_VENV_BIN; export PATH="$VBIN:$PATH"
OUT=$RA_ROOT/results.tsv
printf "model\ttask\tcomp\tK\tmethod\tflag_acc\tacc_strict\tn\n" > "$OUT"
comp_of(){ case "$1:$2" in
  math:2048)echo 2x;;math:1024)echo 4x;;math:512)echo 8x;;math:256)echo 16x;;
  gpqa:4096)echo 2x;;gpqa:2048)echo 4x;;gpqa:1024)echo 8x;;gpqa:512)echo 16x;;
  aime:8192)echo 2x;;aime:4096)echo 4x;;aime:2048)echo 8x;;aime:1024)echo 16x;;
  hmmt:8192)echo 2x;;hmmt:4096)echo 4x;;hmmt:2048)echo 8x;;hmmt:1024)echo 16x;;
  *)echo "?";; esac; }
# grade one method-dir if complete+clean; append row(s)
emit(){ local md=$1 task=$2 K=$3 meth=$4 exp=$5 np=$6 dn=$7
  local d=gate2/$md/${task}_K${K}/$meth
  [ -d "$d" ] || return
  local nsub=$(ls -d "$d"/*/ 2>/dev/null|wc -l)
  [ "$task" != aime ] && [ "$nsub" -gt 1 ] && return   # stale-dup subdir -> skip
  local n=$(cat "$d"/*/run_*/completions_shard*.jsonl 2>/dev/null|wc -l)
  local r0=$(cat "$d"/${task}*_*/run_0/completions_shard*.jsonl 2>/dev/null|wc -l)
  [ "$task" = aime ] && { r0=$(cat "$d"/aime25_*/run_0/completions_shard*.jsonl 2>/dev/null|wc -l); np=29; }
  [ "$r0" -gt "$np" ] && return                        # positional corruption -> skip
  [ "$n" -lt "$exp" ] && return                        # partial -> skip
  local comp=$(comp_of $task $K)
  local line=$($VBIN/python grade_ours.py --base gate2/$md/${task}_K${K} --data_name $dn --methods $meth 2>/dev/null | awk -v m="$meth" '$1==m{print $3"\t"$4"\t"$6; exit}')
  [ -n "$line" ] && printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$md" "$task" "$comp" "$K" "$meth" "$line" >> "$OUT" && echo "  + $md $task $comp $meth"
}
for md in Qwen3-4B phi-4-reasoning Qwen3-32B; do
 for spec in "math:500:1000:1024,512,256,2048" "gpqa:198:792:2048,1024,512,4096" "aime:29:944:4096,2048,1024,8192" "hmmt:60:960:4096,2048,1024,8192"; do
  task=$(echo $spec|cut -d: -f1); np=$(echo $spec|cut -d: -f2); exp=$(echo $spec|cut -d: -f3); Ks=$(echo $spec|cut -d: -f4|tr ',' ' ')
  dn=$task; [ "$task" = aime ] && dn=aime26   # aime graded per-subtask; use aime26 as the regime point (or avg later)
  for K in $Ks; do for meth in random_pp vase_faithful triattn_ph attn dense; do
    # dense/attn live under different dir names; try both faithful conventions
    emit $md $task $K $meth $exp $np $dn
  done; done
 done
done
echo "wrote $OUT ($(($(wc -l < "$OUT")-1)) cells)"
