#!/usr/bin/env python3
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
"""vLLM ACCURACY cross-check (addendum 10): 4B MATH under random_pp eviction in TriAttention's
runtime. Does the HF-harness accuracy reproduce when the eviction runs inside vLLM?

Launcher must set: ENABLE_TRIATTENTION=1 TRIATTN_RUNTIME_SELECTOR=random_pp
  TRIATTN_RUNTIME_KV_BUDGET=1024 TRIATTN_RUNTIME_DIVIDE_LENGTH=128
  TRIATTN_RUNTIME_WINDOW_SIZE=128 TRIATTN_RUNTIME_PROTECT_PREFILL=1
Semantics delta vs the HF cell (documented, not hidden): their window=128 vs our residual=64;
their divide-length block compaction vs our per-round topk. This is a does-it-transfer check,
not a bit-identity claim.

Output: gate2-layout cell the repo grader reads directly:
  <out>/math_vllm_bs0_budget=1024_random_pp/run_R/completions_shard0.jsonl  (500 lines, gi order)
"""
import argparse
import json
import os

BASE = os.environ.get("RA_ENGINE", os.path.join(os.environ.get("RA_ROOT", "."), "kvcompress/harness"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(os.environ.get("RA_MODELS_DIR", "models"), "Qwen3-4B"))
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--top_k", type=int, default=-1,
                    help="CORRECTED 08-17 (review): the HF batch_exist sampler applies ONLY "
                         "temperature+top_p (no top_k warper) — matched sampling = top_k off. "
                         "top_k=20 is a stabilizer variant, not the matched condition.")
    ap.add_argument("--allow_dense", action="store_true")
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=500)
    ap.add_argument("--max_tokens", type=int, default=32768)
    ap.add_argument("--max_num_seqs", type=int, default=96,
                    help="pool-safe concurrency cap: their integration corrupts on vLLM "
                         "preemption, so admitted seqs must never outgrow the KV pool "
                         "(A100 ~12.3k blocks; 96 x ~90 compressed blocks leaves margin)")
    ap.add_argument("--out", default=os.path.join(
        BASE, "gate2/Qwen3-4B/math_K1024/rp_vllm0817"))
    args = ap.parse_args()
    assert args.allow_dense or os.environ.get("ENABLE_TRIATTENTION") == "1", "launcher must enable the runtime (or pass --allow_dense for the no-compression control)"

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    data = [json.loads(l) for l in open(os.path.join(BASE, "data/math/test.jsonl"))]
    data = data[args.lo:args.hi]
    prompts = []
    for d in data:
        q = d["problem"].strip()
        msg = [{"role": "user", "content": q + "\nPlease reason step by step, and put your final answer within \\boxed{}."}]
        prompts.append(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))

    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_tokens + 1024,
              enforce_eager=True, enable_prefix_caching=False, max_num_seqs=args.max_num_seqs)
    for r in range(args.runs):
        bud = os.environ.get("TRIATTN_RUNTIME_KV_BUDGET", "dense")
        mode = "dense" if args.allow_dense else "random_pp"
        leaf = os.path.join(args.out, f"math_bs0_budget={bud}_{mode}_vllm", f"run_{r}")
        os.makedirs(leaf, exist_ok=True)
        fp = os.path.join(leaf, f"completions_shard{args.lo}.jsonl")
        if os.path.exists(fp) and sum(1 for _ in open(fp)) == len(data):
            print(f"run_{r} already complete, skip", flush=True)
            continue
        sp = SamplingParams(temperature=0.6, top_p=0.95, top_k=args.top_k,
                            max_tokens=args.max_tokens, seed=20260817 + r)
        outs = llm.generate(prompts, sp)
        with open(fp, "w") as f:
            for p_text, o in zip(prompts, outs):
                # grader convention: "completion" = full decoded prompt+generation text
                f.write(json.dumps({"completion": p_text + o.outputs[0].text}) + "\n")
        with open(os.path.join(leaf, f"other_info_shard{args.lo}.json"), "w") as f:
            json.dump({"generate_lens": [len(o.outputs[0].token_ids) for o in outs]}, f)
        print(f"run_{r} written ({len(outs)} rows)", flush=True)
    print("VLLM MATH ACC DONE", flush=True)


if __name__ == "__main__":
    main()
