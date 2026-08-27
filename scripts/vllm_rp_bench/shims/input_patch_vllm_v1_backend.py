# Modifications Copyright (c) 2026, Salesforce, Inc.  SPDX-License-Identifier: Apache-2.0
# (the original portions remain under their own license -- see the provenance note below and THIRD_PARTY_NOTICES.md)
"""Debug-only V1 GPUModelRunner input patch helpers.

This module provides the smallest possible compatibility layer needed to
validate effective override semantics on the legacy/default vLLM V1 runner
path (`vllm.v1.worker.gpu_model_runner.GPUModelRunner`).
"""

from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np

from . import input_patch_state as _patch_state


def _debug_drop_pos_delta() -> bool:
    return os.environ.get("TRIATTN_DEBUG_V1_DROP_POS_DELTA", "0") == "1"


def _debug_drop_seq_base() -> bool:
    return os.environ.get("TRIATTN_DEBUG_V1_DROP_SEQ_BASE", "0") == "1"


def _debug_preserve_rope_positions() -> bool:
    return os.environ.get("TRIATTN_DEBUG_V1_PRESERVE_ROPE_POSITIONS", "0") == "1"


def _build_effective_slot_positions(
    *,
    positions_np: np.ndarray,
    req_indices: np.ndarray,
) -> np.ndarray | None:
    if _debug_drop_pos_delta():
        return None
    if (
        int(req_indices.size) == 0
        or int(positions_np.size) == 0
    ):
        return None

    # Slot positions may follow the compacted KV layout, but decode-time
    # RoPE positions must stay in the original logical sequence space.
    out = positions_np.copy()

    if int(req_indices.max(initial=-1)) + 1 == 1 and _patch_state.ACTIVE_SINGLE_EFFECTIVE_POS_DELTA != 0:
        out += int(_patch_state.ACTIVE_SINGLE_EFFECTIVE_POS_DELTA)
        return out

    sparse_pos_deltas = _patch_state.ACTIVE_EFFECTIVE_POS_DELTA_BY_REQ_IDX
    if not sparse_pos_deltas:
        return None

    row_deltas = np.zeros(int(req_indices.max()) + 1, dtype=positions_np.dtype)
    for req_idx, delta in sparse_pos_deltas.items():
        if 0 <= int(req_idx) < row_deltas.shape[0]:
            row_deltas[int(req_idx)] = int(delta)
    out += row_deltas[req_indices]
    return out


def _apply_sparse_seq_len_overrides_in_place(
    *,
    seq_lens_np: np.ndarray,
    num_computed_tokens_cpu: np.ndarray,
    num_scheduled_tokens: np.ndarray,
    num_reqs: int,
) -> bool:
    if _debug_drop_seq_base():
        return False
    if num_reqs <= 0:
        return False

    applied = False
    if num_reqs == 1 and _patch_state.ACTIVE_SINGLE_EFFECTIVE_SEQ_BASE is not None:
        seq_lens_np[0] = int(_patch_state.ACTIVE_SINGLE_EFFECTIVE_SEQ_BASE) + int(num_scheduled_tokens[0])
        return True

    sparse_bases = _patch_state.ACTIVE_EFFECTIVE_BASE_BY_REQ_IDX
    if not sparse_bases:
        return False

    seq_lens_np[:num_reqs] = num_computed_tokens_cpu[:num_reqs] + num_scheduled_tokens[:num_reqs]
    for req_idx, effective_base in sparse_bases.items():
        idx = int(req_idx)
        if 0 <= idx < num_reqs:
            seq_lens_np[idx] = int(effective_base) + int(num_scheduled_tokens[idx])
            applied = True
    return applied


def _apply_overrides_gpu_resident(
    self: Any,
    num_scheduled_tokens: np.ndarray,
    total_num_scheduled_tokens: int,
    num_reqs: int,
) -> None:
    """SHIM (capture-layer port, 2026-08-08).

    vllm>=0.19 GPUModelRunner keeps `positions`/`seq_lens` as plain GPU tensors
    and computes slot mappings on-GPU via
    `input_batch.block_table.compute_slot_mapping(num_reqs, query_start_loc,
    positions)` (gpu_model_runner.py:1996-2019), so the legacy numpy path in
    `_patched_prepare_inputs` (positions.np / seq_lens.np /
    compute_slot_mapping(req_indices, positions_np)) no longer exists.

    This applies IDENTICAL override semantics on the new buffers:
      - slot positions shifted by the compaction delta (RoPE positions in
        `self.positions` left untouched, exactly as the legacy branch does);
      - per-request seq_lens set to effective_base + num_scheduled_tokens on
        BOTH `self.seq_lens` (GPU) and `self.optimistic_seq_lens_cpu` (the CPU
        source that _build_attention_metadata uses for seq_lens_cpu and
        max_seq_len).
    No selection/kernel semantics are touched.
    """
    import torch

    device = self.positions.device

    if not _debug_drop_pos_delta():
        slot_positions = None
        if num_reqs == 1 and _patch_state.ACTIVE_SINGLE_EFFECTIVE_POS_DELTA != 0:
            slot_positions = (
                self.positions[:total_num_scheduled_tokens]
                + int(_patch_state.ACTIVE_SINGLE_EFFECTIVE_POS_DELTA)
            )
        else:
            sparse_pos_deltas = _patch_state.ACTIVE_EFFECTIVE_POS_DELTA_BY_REQ_IDX
            if sparse_pos_deltas:
                row_deltas = np.zeros(num_reqs, dtype=np.int64)
                for req_idx, delta in sparse_pos_deltas.items():
                    if 0 <= int(req_idx) < num_reqs:
                        row_deltas[int(req_idx)] = int(delta)
                req_indices = np.repeat(
                    np.arange(num_reqs, dtype=np.int64), num_scheduled_tokens
                )
                token_deltas = torch.from_numpy(row_deltas[req_indices]).to(device)
                slot_positions = (
                    self.positions[:total_num_scheduled_tokens] + token_deltas
                )
        if slot_positions is not None:
            query_start_loc = (
                self.query_start_loc.gpu
                if hasattr(self.query_start_loc, "gpu")
                else self.query_start_loc
            )
            self.input_batch.block_table.compute_slot_mapping(
                num_reqs,
                query_start_loc[: num_reqs + 1],
                slot_positions,
            )
            if os.environ.get("TRIATTN_ASSERT_REACHABLE") == "1":
                # RESIDUAL-ISOLATION PROBE (08-18): every overridden request's LAST scheduled
                # token must resolve inside its kept block rows. A raise here = a surviving
                # orphan path (e.g. the prefill/post-forward branch) — the "second bug"
                # hypothesis. Silence across a full run = the residual is retention SEMANTICS.
                bt = self.input_batch.block_table
                tables = bt.block_tables if hasattr(bt, "block_tables") else [bt]
                qsl_cpu = query_start_loc[: num_reqs + 1].cpu().tolist()
                deltas = dict(_patch_state.ACTIVE_EFFECTIVE_POS_DELTA_BY_REQ_IDX or {})
                if num_reqs == 1 and _patch_state.ACTIVE_SINGLE_EFFECTIVE_POS_DELTA != 0:
                    deltas[0] = _patch_state.ACTIVE_SINGLE_EFFECTIVE_POS_DELTA
                for _key in deltas:
                    _dval = deltas[_key]
                    _idx = int(_key)
                    if not (0 <= _idx < num_reqs):
                        continue
                    qend = int(qsl_cpu[_idx + 1]) - 1
                    last_slot_pos = int(slot_positions[qend].item())
                    for tb in tables:
                        bsz = int(getattr(tb, "block_size", 16))
                        nrows = int(tb.num_blocks_per_row[_idx])
                        blk = last_slot_pos // bsz
                        if blk >= nrows:
                            msg = (
                                f"ORPHAN REACHABILITY VIOLATION: req_idx={_idx} "
                                f"last_slot_pos={last_slot_pos} block={blk} "
                                f">= kept_rows={nrows} (block_size={bsz}) "
                                f"delta={_dval} all_rows={[int(t.num_blocks_per_row[_idx]) for t in tables]}"
                            )
                            if os.environ.get("TRIATTN_ASSERT_RAISE") == "1":
                                raise RuntimeError(msg)
                            # Default: rate-limited count-and-log so a full run yields BOTH an
                            # orphan census and an accuracy number (a raise kills the engine
                            # at the first hit and forfeits the lane).
                            _patch_state.ORPHAN_VIOLATION_COUNT = getattr(
                                _patch_state, "ORPHAN_VIOLATION_COUNT", 0) + 1
                            if _patch_state.ORPHAN_VIOLATION_COUNT <= 20 or \
                                    _patch_state.ORPHAN_VIOLATION_COUNT % 100 == 0:
                                print(f"[TRIATTN-PROBE #{_patch_state.ORPHAN_VIOLATION_COUNT}] {msg}",
                                      flush=True)

    if not _debug_drop_seq_base():
        eff_by_idx: dict[int, int] = {}
        if num_reqs == 1 and _patch_state.ACTIVE_SINGLE_EFFECTIVE_SEQ_BASE is not None:
            eff_by_idx[0] = int(_patch_state.ACTIVE_SINGLE_EFFECTIVE_SEQ_BASE) + int(
                num_scheduled_tokens[0]
            )
        else:
            sparse_bases = _patch_state.ACTIVE_EFFECTIVE_BASE_BY_REQ_IDX
            if sparse_bases:
                for req_idx, effective_base in sparse_bases.items():
                    idx = int(req_idx)
                    if 0 <= idx < num_reqs:
                        eff_by_idx[idx] = int(effective_base) + int(
                            num_scheduled_tokens[idx]
                        )
        if eff_by_idx:
            optimistic = getattr(self, "optimistic_seq_lens_cpu", None)
            for idx, eff in eff_by_idx.items():
                self.seq_lens[idx] = eff
                if optimistic is not None:
                    optimistic[idx] = eff


def make_patched_v1_prepare_inputs(
    original_prepare_inputs: Callable[..., Any],
) -> Callable[..., Any]:
    def _patched_prepare_inputs(self, scheduler_output, num_scheduled_tokens):
        out = original_prepare_inputs(self, scheduler_output, num_scheduled_tokens)

        if not _patch_state.ACTIVE_EFFECTIVE_OVERRIDES_ENABLED:
            return out

        _patch_state.mark_active_effective_overrides_consumed()

        total_num_scheduled_tokens = int(getattr(scheduler_output, "total_num_scheduled_tokens", 0))
        num_reqs = int(getattr(self.input_batch, "num_reqs", 0))
        if total_num_scheduled_tokens <= 0 or num_reqs <= 0:
            return out

        # SHIM dispatch (2026-08-08): legacy vLLM kept positions/seq_lens as
        # CpuGpuBuffers with `.np` views; vllm>=0.19 uses plain GPU tensors.
        if not (hasattr(self.positions, "np") and hasattr(self.seq_lens, "np")):
            _apply_overrides_gpu_resident(
                self,
                num_scheduled_tokens,
                total_num_scheduled_tokens,
                num_reqs,
            )
            return out

        req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)
        positions_np = self.positions.np[:total_num_scheduled_tokens]
        original_positions_np = positions_np.copy()

        slot_positions_np = _build_effective_slot_positions(
            positions_np=positions_np,
            req_indices=req_indices,
        )
        if slot_positions_np is not None:
            self.input_batch.block_table.compute_slot_mapping(req_indices, slot_positions_np)
            self.input_batch.block_table.commit_slot_mapping(total_num_scheduled_tokens)
        seq_applied = _apply_sparse_seq_len_overrides_in_place(
            seq_lens_np=self.seq_lens.np,
            num_computed_tokens_cpu=self.input_batch.num_computed_tokens_cpu,
            num_scheduled_tokens=num_scheduled_tokens,
            num_reqs=num_reqs,
        )
        if seq_applied:
            self.seq_lens.np[num_reqs:].fill(0)
            self.seq_lens.copy_to_gpu()

        return out

    return _patched_prepare_inputs
