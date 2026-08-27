# Modifications Copyright (c) 2026, Salesforce, Inc.  SPDX-License-Identifier: Apache-2.0
# (the original portions remain under their own license -- see the provenance note below and THIRD_PARTY_NOTICES.md)
# VaSE's official decode-throughput benchmark (https://github.com/terarachang/VaSE, MIT; eval/reasoning_tasks/benchmark/throughput.py at
# bfa2692, itself modified from ReasoningPathCompression). Random Attention changes: engine import path and a
# one-line METHOD_RENAME.get() so non-VaSE method names print. Used by scripts/efficiency/*.sh (fixed-batch
# and max-batch protocols).
"""
Decode-throughput benchmark for KV-cache compression methods on Qwen3.
Modified from https://github.com/jiwonsong-dev/ReasoningPathCompression/blob/main/benchmark/throughput.py

Example:
  python benchmark/throughput.py \
    --model_path Qwen/Qwen3-4B \
    --methods dense,attn_rkv \
    --batch_size 16 --input_len 128 --output_len 4096 \
    --token_budget 1024 \
    --num_runs 3
"""
import argparse
import gc
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, DynamicCache

# benchmark/ lives one level below the repo root; make the repo modules importable.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from kvcompress.engine.cache_utils import EvictCache



METHOD_RENAME = {
    "dense": "Full",
    "attn_rkv": "RKV",
    "cur_fixed_gauss": "CUR Fixed",
    "cur_resample_gauss": "CUR Resample",
    "range_sink_sample_attn": "Sample Attn + V",
}


def cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def average_excluding_min_max(numbers):
    if len(numbers) <= 2:
        return sum(numbers) / len(numbers)
    trimmed = numbers.copy()
    trimmed.remove(min(trimmed))
    trimmed.remove(max(trimmed))
    return sum(trimmed) / len(trimmed)


def build_eviction_config(args, eviction_mode):
    """Mirror my_utils.init_evict_configs, but parameterised by eviction_mode."""
    assert args.token_budget > args.residual_length, (
        f"token_budget ({args.token_budget}) must be > residual_length ({args.residual_length})"
    )

    return {
        "token_budget": args.token_budget,
        "residual_length": args.residual_length,
        "rkv_lambda": args.rkv_lambda,
        "eviction_mode": eviction_mode,
        "smooth": ('attn' in eviction_mode),
        "n_large": args.token_budget//4,
        "temperature": 1.0,
        "verbose": args.verbose,
    }


def run_one(model, input_ids, attention_mask, output_len, make_cache):
    """
    Fixed-length greedy decode; returns (decode_sec, num_decode_steps).
    Times the decode loop only (prefill excluded).
    """
    cache = make_cache()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    decode_ms = 0.0

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask,
                        past_key_values=cache, use_cache=True)
        cache = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        for _ in range(output_len):
            start.record()
            outputs = model(input_ids=next_token, past_key_values=cache, use_cache=True)
            end.record()
            torch.cuda.synchronize()
            decode_ms += start.elapsed_time(end)
            cache = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    return decode_ms / 1000.0, output_len


def benchmark_method(args, model, stock_forward, evict_forward, method):
    """Configure attention + cache for `method`, then warm up and time it."""
    is_dense = method == "dense"

    # Toggle the global Qwen3Attention.forward and pick the cache constructor.
    if is_dense:
        _set_qwen3_forward(model, stock_forward)
        make_cache = lambda: DynamicCache()
    else:
        _set_qwen3_forward(model, evict_forward)
        cache_config = build_eviction_config(args, method)
        make_cache = lambda: EvictCache(config=model.config, **cache_config)

    device = model.device
    vocab_size = model.config.vocab_size
    # Random in-vocab prompt; content is irrelevant to throughput, only shape.
    input_ids = torch.randint(0, vocab_size, (args.batch_size, args.input_len),
                              dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    for w in range(args.num_warmups):
        print(f"[{method}] warmup #{w}", flush=True)
        run_one(model, input_ids, attention_mask, args.output_len, make_cache)
        cleanup_memory()

    torch.cuda.reset_peak_memory_stats(device=device)

    throughputs = []
    for r in range(args.num_runs):
        print(f"[{method}] run #{r}", flush=True)
        elapsed, steps = run_one(model, input_ids, attention_mask, args.output_len, make_cache)
        tput = args.batch_size * steps / elapsed
        throughputs.append(tput)
        print(f"[{method}] run #{r}: {steps} decode steps, {elapsed:.2f}s, "
              f"{tput:.2f} tok/s (batched)", flush=True)
        cleanup_memory()

    peak_mem = torch.cuda.max_memory_allocated(device=device)
    return {
        "method": method,
        "throughput_avg": average_excluding_min_max(throughputs),
        "peak_mem_gb": peak_mem / 1024 ** 3,
    }


def _set_qwen3_forward(model, forward_fn):
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
    Qwen3Attention.forward = forward_fn


def main():
    args = parse_args()
    if "Qwen3" not in args.model_path:
        raise ValueError("This benchmark targets Qwen3 (the eviction patch only supports Qwen3).")

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    print(f"Methods: {methods}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
    )
    model.eval()

    # Capture the stock attention forward before any patching, then build the
    # eviction forward by applying the monkeypatch and grabbing it.
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
    stock_forward = Qwen3Attention.forward
    from kvcompress.engine.modify_forward import wrap_evict_attn_forward
    wrap_evict_attn_forward(args.model_path)
    evict_forward = Qwen3Attention.forward
    # Restore stock for now; benchmark_method toggles as needed.
    Qwen3Attention.forward = stock_forward

    results = []
    for method in methods:
        results.append(
            benchmark_method(args, model, stock_forward, evict_forward, method)
        )
        cleanup_memory()

    print("\n" + "=" * 92)
    print(f"Model: {args.model_path}  |  attn={args.attn_implementation}")
    print(f"batch_size={args.batch_size}  input_len={args.input_len}  "
          f"output_len={args.output_len}")
    print(f"warmups={args.num_warmups}  runs={args.num_runs}")
    print(f"eviction: token_budget={args.token_budget}")
    print("-" * 92)
    print(f"{'method':>32} | {'throughput (tok/s)':>18} | {'peak mem (GB)':>14}")
    print("-" * 92)
    for r in results:
        print(f"{METHOD_RENAME.get(r['method'], r['method']):>32} | {r['throughput_avg']:>18.1f} | {r['peak_mem_gb']:>14.2f}")
    print("=" * 92)

    write_results_table(args, results)


def write_results_table(args, results):
    """Append/update tables/throughput_<model>_<output_len>.json, keyed by token_budget."""
    model_name = os.path.basename(args.model_path)
    out_path = os.path.join(REPO_ROOT, "tables", f"throughput_{model_name}_{args.output_len}.json")

    table = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            table = json.load(f)

    table[str(args.token_budget)] = [
        {
            "Method": METHOD_RENAME.get(r["method"], r["method"]),
            "Throughput": round(r["throughput_avg"], 1),
            "Memory": round(r["peak_mem_gb"], 1),
        }
        for r in results
    ]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(table, f, indent=2)
    print(f"Wrote results to {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="Qwen/Qwen3-14B")
    p.add_argument("--methods", type=str, default="dense,attn_rkv,range_sink_sample_attn,cur_resample_gauss")
    p.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--input_len", type=int, default=256)
    p.add_argument("--output_len", type=int, default=16384)
    p.add_argument("--num_warmups", type=int, default=1)
    p.add_argument("--num_runs", type=int, default=5)
    # Eviction params (ignored by 'dense'); mirror my_utils.get_evict_args.
    p.add_argument("--token_budget", type=int, default=4096)
    p.add_argument("--residual_length", type=int, default=64)
    p.add_argument("--rkv_lambda", type=float, default=0.5)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
