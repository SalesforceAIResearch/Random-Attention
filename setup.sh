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
# One-time environment setup for the accuracy + HF-efficiency experiments.
# Tested with Python 3.10, CUDA 12.1, torch 2.4.0, transformers 5.0.0, flash-attn 2.7.3 on H200/A100.
# The vLLM serving benchmark uses a SEPARATE venv (scripts/vllm_rp_bench/setup_venv.sh).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3.10}
[ -d .venv ] || "$PY" -m venv .venv
. .venv/bin/activate
pip install -U pip wheel
pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch==2.4.0
pip install --no-build-isolation flash-attn==2.7.3          # needs torch + nvcc present
pip install -r requirements.txt
# LiveCodeBench difficulty subsets ship with the repo; the graders expect data/<subset>/test.jsonl
for s in easy hard medium r263 sub120; do
  mkdir -p "data/livecodebench_$s"
  [ -e "data/livecodebench_$s/test.jsonl" ] || ln -s "../lcb_subsets/livecodebench_$s.test.jsonl" "data/livecodebench_$s/test.jsonl"
done
echo
echo "Environment ready. Next:"
echo "  1. put the benchmark files under data/<task>/test.jsonl   (data/README.md)"
echo "  2. put the model checkpoints under models/<name>            (or export RA_MODELS_DIR)"
echo "  3. . env.sh && scripts/run_cell.sh Qwen3-4B math random_pp   (docs/METHODS.md for every method)"
