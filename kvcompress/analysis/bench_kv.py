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
"""Throughput + peak-memory micro-benchmark for KV eviction (VaSE-paper style).
ONE config per process (so dense loads UNwrapped, eviction loads wrapped — no cross-contamination).
Fixed-length greedy decode (ignores EOS) so per-token throughput is isolated from rambling length.
Prints one RESULT line. Usage:
  python bench_kv.py --mode dense        --decode_len 8192
  python bench_kv.py --mode random_pp    --token_budget 1024 --decode_len 8192
  python bench_kv.py --mode range_sink_sample_attn --token_budget 1024 --decode_len 8192   # = VaSE (top-k)
"""
import argparse, os, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

_EV = []   # (start_event, end_event) per eviction round, filled only when EVICT_TIMING=1


def _instrument_eviction():
    """Wrap EvictLayer._run_eviction with CUDA-event pairs — NO engine edit (the live
    cache_utils.py is in use by concurrent eval jobs, so instrumentation stays external).
    Measures the whole round (selection + K/V compaction). Since compaction is identical
    across modes, rp's round time is the compaction floor and (mode - rp) is that mode's
    selection overhead."""
    from kvcompress.engine.cache_utils import EvictLayer
    orig = EvictLayer._run_eviction

    def timed(self, cur_len):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        out = orig(self, cur_len)
        e.record()
        _EV.append((s, e))
        return out

    EvictLayer._run_eviction = timed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(os.environ.get("RA_MODELS_DIR", "models"), "Qwen3-4B"))
    ap.add_argument("--mode", required=True)              # dense | random_pp | range_sink_sample_attn | caote
    ap.add_argument("--token_budget", type=int, default=1024)
    ap.add_argument("--residual_length", type=int, default=64)
    ap.add_argument("--decode_len", type=int, default=8192)
    ap.add_argument("--n_prefill", type=int, default=256)
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--n_large", type=int, default=None,
                    help="VaSE sampled-attention width; default token_budget//4 (faithful rule). "
                         "The old hardcoded 256 cripples VaSE at large K.")
    args = ap.parse_args()
    device = "cuda:0"
    evict = args.mode != "dense"
    n_large = args.n_large if args.n_large is not None else max(args.token_budget // 4, 1)
    timing = os.environ.get("EVICT_TIMING") == "1"

    if evict:                                            # patch the attention forward BEFORE loading the model
        from kvcompress.engine.modify_forward import wrap_evict_attn_forward
        wrap_evict_attn_forward(args.model)
        if timing:
            _instrument_eviction()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 device_map=device, attn_implementation=args.attn).eval()

    ids = tok("Solve this problem step by step, showing all work. " * 60,
              return_tensors="pt").input_ids[:, :args.n_prefill].to(device)
    am = torch.ones_like(ids)

    if evict:
        from kvcompress.engine.cache_utils import EvictCache
        cfg = dict(token_budget=args.token_budget, residual_length=args.residual_length, rkv_lambda=(0.1 if args.mode == 'attn_rkv' else None),
                   eviction_mode=args.mode, smooth=True, n_large=n_large, temperature=0.6, verbose=False)
        cache = EvictCache(config=model.config, **cfg)
    else:
        cache = DynamicCache()

    def step(cur, am, cache):
        with torch.no_grad():
            out = model(cur, attention_mask=am, past_key_values=cache, use_cache=True, logits_to_keep=1)
        nxt = out.logits[:, -1:, :].argmax(-1)
        am = torch.cat([am, torch.ones((1, 1), device=device, dtype=am.dtype)], 1)
        return nxt, am, out.past_key_values

    torch.cuda.reset_peak_memory_stats(device)
    cur, am, cache = step(ids, am, cache)                # prefill
    for _ in range(8):                                   # warmup (not timed)
        cur, am, cache = step(cur, am, cache)

    torch.cuda.synchronize(device); t0 = time.time()
    for _ in range(args.decode_len):
        cur, am, cache = step(cur, am, cache)
    torch.cuda.synchronize(device); dt = time.time() - t0

    peak = torch.cuda.max_memory_allocated(device) / 1e9
    K = args.token_budget if evict else "full"
    extra = ""
    if timing and _EV:
        ms = sum(s.elapsed_time(e) for s, e in _EV)       # events already synced above
        rounds = len(_EV)
        extra = (f" evict_ms={ms:8.1f} calls={rounds:<6} ms_per_call={ms/rounds:6.3f}"
                 f" pct_of_decode={100*ms/1e3/dt:5.2f}%")
    print(f"RESULT mode={args.mode:24s} K={K:<5} nl={n_large if evict else '-':<5} "
          f"declen={args.decode_len:<6} prefill={args.n_prefill} "
          f"tok/s={args.decode_len/dt:6.1f} sec={dt:6.1f} peak_GB={peak:5.2f}{extra}", flush=True)

main()
