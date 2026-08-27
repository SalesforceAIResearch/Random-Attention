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
"""random_pp selector for TriAttention's official vLLM runtime plugin.

Uniform random over decode positions per KV head, full prefill + recency window
protected, deterministic per (request, event, layer). Efficiency-benchmark
selector: budget/window/prefill semantics are byte-identical to the TriAttention
selector; only the score is replaced by uniform noise.

Install: copy into <triattention checkout>/triattention/vllm/runtime/ and apply
hook_impl.patch from this directory. Activate with TRIATTN_RUNTIME_SELECTOR=random_pp.
Scoped by EXPERIMENTS_H200_FOLLOWUP.md Tier-2: these numbers are efficiency-only
(their budget semantics: prompt counted, window inside budget) and are never mixed
with the accuracy tables.
"""
from __future__ import annotations

import hashlib
import os
from typing import Any, Callable

import torch

from .config import TriAttentionRuntimeConfig


def _stable_seed(*parts: Any) -> int:
    h = hashlib.blake2b("|".join(str(p) for p in parts).encode(), digest_size=8)
    return int.from_bytes(h.digest(), "little") & 0x7FFF_FFFF_FFFF_FFFF


def build_random_pp_selector(
    config: TriAttentionRuntimeConfig,
    base_runner: Any | None = None,
) -> tuple[Callable[..., dict[str, Any] | None], None, str]:
    base_seed = int(os.environ.get("TRIATTN_RUNTIME_RANDOM_SEED", "1234"))
    generators: dict[torch.device, torch.Generator] = {}

    def _generator(device: torch.device, seed: int) -> torch.Generator:
        gen = generators.get(device)
        if gen is None:
            gen = torch.Generator(device=device)
            generators[device] = gen
        gen.manual_seed(seed)
        return gen

    def _active_req_id() -> str:
        if base_runner is None:
            return "?"
        return str(getattr(base_runner, "_triattention_active_req_id", "?"))

    def _select_keep_indices(
        *,
        keys_dense: torch.Tensor | None = None,
        kv_cache: torch.Tensor | None = None,
        block_ids: "list[int] | torch.Tensor | None" = None,
        block_size: int | None = None,
        total_tokens: int,
        prefill_len: int,
        protect_prefill: bool,
        layer_idx: int,
        round_start: int,
        budget_total: int,
    ) -> dict[str, Any] | None:
        # Fast path + degenerate guard: byte-for-byte the TriAttention behaviour.
        if total_tokens <= budget_total:
            return {"mode": "shared", "indices": list(range(total_tokens))}
        if protect_prefill and config.include_prefill_in_budget and prefill_len > budget_total:
            print(f"[RP-SELECTOR-NONE] req={_active_req_id()} layer={layer_idx} "
                  f"total_tokens={total_tokens} prefill_len={prefill_len} "
                  f"budget_total={budget_total} round_start={round_start} "
                  f"window={config.window_size} include_pf={config.include_prefill_in_budget}",
                  flush=True)
            return None  # planner turns this into a clean "prefill_exceeds_budget" no-op

        if kv_cache is not None:            # paged: [..., block, H, D] -> shape[3] == H_kv
            num_kv_heads = int(kv_cache.shape[3])
            device = kv_cache.device
        elif keys_dense is not None:        # dense: [1, H, T, D]
            num_kv_heads = int(keys_dense.shape[1])
            device = keys_dense.device
        else:
            raise RuntimeError("missing inputs for random selector")

        k = min(int(budget_total), int(total_tokens))
        if k <= 0:
            return {"mode": "shared", "indices": []}

        seed = _stable_seed(base_seed, _active_req_id(), round_start, layer_idx)
        gen = _generator(device, seed)
        scores = torch.rand((num_kv_heads, total_tokens), device=device, generator=gen)

        # Force-keep guards, same semantics as selector_hf._finalize_layer_scores:
        if config.window_size > 0:
            recent = min(config.window_size, total_tokens)
            scores[:, total_tokens - recent:] = float("inf")
        if protect_prefill and prefill_len > 0:
            scores[:, :prefill_len] = float("inf")

        topk = torch.topk(scores, k=k, dim=-1, largest=True, sorted=False).indices
        keep_per_head = torch.sort(topk, dim=-1).values.contiguous()
        return {"mode": "per_head", "indices": keep_per_head}

    # Never reads KV contents -> the paged path is trivially supported (strict mode
    # requires this attr or the planner raises paged_selector_required).
    setattr(_select_keep_indices, "_supports_paged", True)
    # No group selector: the planner falls through to per-layer calls; K is identical
    # across layers so cache_len_after stays consistent.
    return _select_keep_indices, None, "random_pp"
