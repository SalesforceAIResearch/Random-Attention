# Third-party notices

Portions of this repository are derived from the projects below. File-level map (upstream commit `bfa2692`
for VaSE, `9c26d14` for TriAttention; every derived file also carries a provenance header):

| file | origin | status |
|---|---|---|
| `kvcompress/engine/cache_utils.py` | VaSE `modified/transformers/cache_utils.py` (itself from HF transformers) | modified: Random Attention, `*_pp` variants, faithful TriAttention, loggers, mechanism modes |
| `kvcompress/engine/modify_forward.py` | VaSE `modified/transformers/modify_forward.py` | modified: + phi3 and llama forward patches |
| `kvcompress/engine/triattn_official.py` | TriAttention `methods/pruning_utils.py` | ported scoring functions, verbatim |
| `kvcompress/harness/eval_hf.py`, `generation_utils.py` | VaSE same paths | import/path lines only |
| `kvcompress/harness/my_utils.py`, `Utils/**`, `prompts/**`, `data/lcb_subsets/download_tests.py` | VaSE same paths | unmodified copies |
| `kvcompress/harness/vase_protocol_throughput.py` | VaSE `benchmark/throughput.py` | one-line fix |
| `kvcompress/harness/parallel_run_hf_mw.py` | descends from VaSE `parallel_run_hf.py` | rewritten (sharding, multi-worker, overrides) |
| `scripts/vllm_rp_bench/shims/*.py` | TriAttention vLLM runtime files | modified copies (`.diff` beside each) |

## VaSE -- MIT License

https://github.com/terarachang/VaSE

MIT License

Copyright (c) 2026 Ting-Yun Chang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## TriAttention -- Apache License 2.0

https://github.com/WeianMao/triattention

Copyright (c) the TriAttention authors. Licensed under the Apache License, Version 2.0 (the "License"); you may
not use those portions except in compliance with the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0. Files derived from this project: `kvcompress/engine/triattn_official.py`
(ported scoring functions) and the vLLM-runtime shims under `scripts/vllm_rp_bench/shims/`, which are modified
copies of files from that repository.

## Hugging Face transformers -- Apache License 2.0

https://github.com/huggingface/transformers -- `kvcompress/engine/cache_utils.py` descends (via VaSE) from
`src/transformers/cache_utils.py`. Copyright the HuggingFace Inc. team, licensed under the Apache License 2.0.

## vLLM -- Apache License 2.0

https://github.com/vllm-project/vllm -- the serving benchmark runs on vLLM 0.19.0 (not redistributed).
