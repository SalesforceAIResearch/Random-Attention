#!/usr/bin/env python
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
"""Calibration for the faithful in-engine TriAttention (`triattn_pl` mode).

Produces the official stats format (per-(layer,head) q_mean_complex + q_abs_mean, ALL heads sampled),
PLUS `omega` (RoPE inv_freq) and `freq_scale` in metadata — because our eviction engine
(cache_utils.py) has no handle on the model's rotary module at eviction time, so we bake those
deterministic RoPE-derived vectors into the stats file.

Capture point = the query ENTERING apply_rotary_pos_emb (post q_proj + q_norm for Qwen3, pre-RoPE),
exactly matching TriAttention's pre-RoPE q statistics.

Usage:
  python calibrate_triattn_pl.py --model_dir <HF dir> --out stats_pl/<name>.pt [--n_tokens 50000] [--corpus <glob>]
"""
import argparse, glob, json, os, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from kvcompress.engine.triattn_official import (
    to_complex_pairs, compute_frequency_scaling, HeadFrequencyStats, save_head_frequency_stats,
)


def load_corpus(path):
    texts = []
    srcs = [path] if path else ['data/math/test.jsonl', 'data/gpqa/test.jsonl', 'data/aime25/test.jsonl']
    for s in srcs:
        for fp in glob.glob(s):
            for line in open(fp):
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                t = d.get('problem') or d.get('question') or d.get('prompt') or ''
                if t:
                    texts.append(t)
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_tokens', type=int, default=50000)
    ap.add_argument('--corpus', default=None)
    ap.add_argument('--max_len', type=int, default=4096)
    args = ap.parse_args()

    dev = 'cuda'
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, attn_implementation='eager').to(dev).eval()
    cfg = model.config
    L = cfg.num_hidden_layers
    Hq = cfg.num_attention_heads
    Hkv = getattr(cfg, 'num_key_value_heads', Hq) or Hq
    d = getattr(cfg, 'head_dim', None) or (cfg.hidden_size // Hq)
    F = d // 2
    print(f"model L={L} Hq={Hq} Hkv={Hkv} head_dim={d}")

    # running accumulators per layer: complex sum [Hq,F], abs sum [Hq,F], token count
    qs = {l: torch.zeros(Hq, F, dtype=torch.cfloat, device=dev) for l in range(L)}
    qa = {l: torch.zeros(Hq, F, device=dev) for l in range(L)}
    nt = {l: 0 for l in range(L)}
    cur = {'l': 0}

    def q_complex_bands(q):
        # q: [B, Hq, S, d] pre-RoPE -> per-head complex bands [tokens, Hq, F] (half-split)
        q = q.permute(0, 2, 1, 3).reshape(-1, Hq, d).float()     # [T, Hq, d]
        return torch.complex(q[..., :F], q[..., F:])             # [T, Hq, F]

    attn_mod = sys.modules[type(model.model.layers[0].self_attn).__module__]
    orig_rope = attn_mod.apply_rotary_pos_emb

    def patched_rope(q, k, *a, **kw):
        l = cur['l']
        with torch.no_grad():
            zq = q_complex_bands(q)                              # [T, Hq, F]
            qs[l] += zq.sum(0); qa[l] += zq.abs().sum(0); nt[l] += zq.shape[0]
        return orig_rope(q, k, *a, **kw)

    attn_mod.apply_rotary_pos_emb = patched_rope
    for i, layer in enumerate(model.model.layers):
        layer.self_attn.register_forward_pre_hook(lambda m, inp, i=i: cur.__setitem__('l', i))

    texts = load_corpus(args.corpus)
    seen = 0
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors='pt', truncation=True, max_length=args.max_len).input_ids.to(dev)
            if ids.numel() < 8:
                continue
            model(ids)
            seen += ids.numel()
            if seen >= args.n_tokens:
                break
    attn_mod.apply_rotary_pos_emb = orig_rope
    print(f"calibrated on ~{seen} tokens")

    # RoPE-derived vectors (baked in for the engine)
    rotary = model.model.rotary_emb
    freq_scale = compute_frequency_scaling(rotary, d, torch.bfloat16, torch.device(dev)).cpu()  # [F]
    omega = rotary.inv_freq.detach().float().cpu()                                               # [F]
    rope_style = getattr(rotary, "_rope_style", "half")

    # per-head stats + validity (MRL = |q_mean|/q_abs_mean, should be high in a dominant band)
    sampled_heads, stats_map = [], {}
    mrls = []
    for l in range(L):
        n = max(nt[l], 1)
        qmc = qs[l] / n                     # [Hq,F] complex
        qam = qa[l] / n                     # [Hq,F]
        R = qmc.abs() / (qam + 1e-8)
        mrls.append(R.max(dim=-1).values.mean().item())
        for h in range(Hq):
            sampled_heads.append((l, h))
            stats_map[(l, h)] = HeadFrequencyStats(q_mean_complex=qmc[h].cpu(), q_abs_mean=qam[h].cpu())
    print(f"VALIDITY: per-head max-band MRL (mean) = {sum(mrls)/len(mrls):.3f}  (paper ~0.977)")

    metadata = {
        'head_dim': d, 'rope_style': rope_style, 'rope_theta': float(getattr(cfg, 'rope_theta', 10000.0)),
        'Hq': Hq, 'Hkv': Hkv, 'n_layers': L,
        'omega': omega, 'freq_scale': freq_scale,   # baked-in RoPE vectors for the engine
    }
    from pathlib import Path
    save_head_frequency_stats(Path(args.out), sampled_heads, stats_map, metadata)
    print(f"saved -> {args.out}  (sampled_heads={len(sampled_heads)}, +omega/freq_scale in metadata)")


if __name__ == '__main__':
    main()
