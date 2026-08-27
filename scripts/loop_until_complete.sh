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
# loop_arm.sh <target> <dir> <sig> <log> -- <cmd...>
# Reruns <cmd> until <dir> holds >= <target> completions (attic-excluded). Waits for any live
# orchestrator matching <sig> to exit first (single-owner). Sleeps 60s between passes.
. "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)/env.sh"
set -u
TARGET=$1; DIR=$2; SIG=$3; LOG=$4; shift 4; [ "$1" = "--" ] && shift
cd "$RA_ENGINE"
dn(){ find "$1" -not -path "*_attic*" -name 'completions_shard*.jsonl' 2>/dev/null | xargs -r cat 2>/dev/null | wc -l; }
echo "[$(date)] loop_arm start target=$TARGET dir=$DIR" >> "$LOG"
while pgrep -f "^[^ ]*python [^ ]*parallel_run_hf_mw.py.*$SIG" >/dev/null; do sleep 120; done  # anchored: only real orchestrators, not keeper/ghost bash argvs
pass=0
while [ "$(dn "$DIR")" -lt "$TARGET" ]; do
  pass=$((pass+1))
  echo "[$(date)] pass $pass (have $(dn "$DIR")/$TARGET)" >> "$LOG"
  "$@" >> "$LOG" 2>&1
  sleep 60
done
echo "[$(date)] loop_arm DONE: $(dn "$DIR")/$TARGET after $pass passes" >> "$LOG"
