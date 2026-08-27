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
# Tier-2 venv — GPU node only, NEVER the shared .venv_vase (July venv-break lesson).
set -euo pipefail
WORK=${1:-${VLLM_WORK:-$HOME/triattn_vllm}}   # a local NVMe/scratch dir is best; any writable path works
mkdir -p "$WORK" && cd "$WORK"
[ -f venv/bin/activate ] || rm -rf venv                 # broken/partial venv from an ensurepip-less node
UVPY=${PYTHON311:-$(command -v python3.11 || command -v python3.12 || command -v python3)}   # >=3.10 for vllm 0.19
if [ ! -d venv ]; then
  if command -v uv >/dev/null;         then uv venv venv --python "$UVPY" --seed
  elif command -v virtualenv >/dev/null; then virtualenv -p "$UVPY" venv
  else for PY in python3.12 python3.11 python3; do command -v $PY >/dev/null && $PY -m venv venv && break; done
  fi
fi
[ -f venv/bin/activate ] || { echo "FATAL: no working venv creator on $(hostname)"; exit 1; }
venv/bin/python - <<'V'
import sys; assert sys.version_info >= (3,10), f'python {sys.version} too old for vllm 0.19'
print('venv python OK:', sys.version.split()[0])
V
source venv/bin/activate
pip install -U pip
pip install vllm==0.19.0                            # brings its own torch + triton
: "${TRIATTN_SRC:?set TRIATTN_SRC=/path/to/a/clone/of https://github.com/WeianMao/triattention}"
rsync -a --exclude .git "$TRIATTN_SRC"/ tri/   # COPY; never modify the clone itself
cp $RA_ROOT/scripts/vllm_rp_bench/selector_random.py tri/triattention/vllm/runtime/
# Apply ALL shims — the rsync above restores a pristine upstream copy, so every shim must be re-copied here.
# (08-09 lesson: state.py was missing from an explicit cp list -> re-setup silently reverted the
# dedup-livelock fix -> KV overgrowth -> pool exhaustion -> silent v1 preemption -> "block reclaim
# prefix mismatch" EngineDeadError in every capacity run. Glob so new shims can't be forgotten.)
cp $RA_ROOT/scripts/vllm_rp_bench/shims/*.py \
   tri/triattention/vllm/runtime/                    # vllm-0.19 API-drift + bookkeeping shims (capture layer ONLY; selection/kernel untouched)
cd tri && python $RA_ROOT/scripts/vllm_rp_bench/apply_hook_edit.py triattention/vllm/runtime/hook_impl.py
pip install -e . --no-deps                          # runtime plugin needs only torch+triton; NEVER [eval]
python - <<'PY'
import vllm, triattention
from importlib.metadata import entry_points
print(vllm.__version__, [e.name for e in entry_points(group="vllm.general_plugins")])
PY
echo "SMOKE: run a short 'vllm bench latency' with ENABLE_TRIATTENTION=1 TRIATTN_RUNTIME_SELECTOR=random_pp"
echo "and expect '[TriAttention] Runtime (V2) plugin activated' in the log (see RUNBOOK.md step 3)."
