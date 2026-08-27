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
# Source this before running anything: `. env.sh`. Every script and Python entry point reads these.
export RA_ROOT="${RA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export RA_ENGINE="$RA_ROOT/kvcompress/harness"          # eval_hf.py + vendored harness; cwd for the launchers
export RA_DATA_DIR="${RA_DATA_DIR:-$RA_ROOT/data}"       # <task>/test.jsonl per benchmark (see data/README.md)
export RA_MODELS_DIR="${RA_MODELS_DIR:-$RA_ROOT/models}" # HF checkpoints: Qwen3-4B, Qwen3-14B, Qwen3-32B, phi-4-reasoning, ...
export RA_VENV_BIN="${RA_VENV_BIN:-$RA_ROOT/.venv/bin}"  # the accuracy/efficiency venv (setup.sh); vLLM bench uses its own
export PYTHONPATH="$RA_ROOT:$RA_ENGINE${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$RA_VENV_BIN:$PATH"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
