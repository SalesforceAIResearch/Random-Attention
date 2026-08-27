# Modifications Copyright (c) 2026, Salesforce, Inc.  SPDX-License-Identifier: Apache-2.0
# (the original portions remain under their own license -- see the provenance note below and THIRD_PARTY_NOTICES.md)
"""Helpers for keeping KV allocation state aligned after block reclaim."""

from __future__ import annotations

from typing import Any


EFFECTIVE_KV_OFFSET_ATTR = "_triattention_effective_kv_offset"
EFFECTIVE_NUM_COMPUTED_ATTR = "_triattention_effective_num_computed_tokens"


def _to_non_negative_int(value: Any) -> int | None:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return max(0, out)


def clear_request_allocation_sync_state(request: Any) -> None:
    """Best-effort clear of TriAttention allocation sync attrs from request."""
    if request is None:
        return
    for attr in (EFFECTIVE_KV_OFFSET_ATTR, EFFECTIVE_NUM_COMPUTED_ATTR):
        if hasattr(request, attr):
            try:
                delattr(request, attr)
            except Exception:
                try:
                    setattr(request, attr, None)
                except Exception:
                    continue


def update_request_effective_kv_offset(
    *,
    request: Any,
    cache_len_after: int,
) -> int | None:
    """Update per-request logical-vs-effective offset after a reclaim event."""
    if request is None:
        return None
    logical = _to_non_negative_int(getattr(request, "num_computed_tokens", None))
    effective = _to_non_negative_int(cache_len_after)
    if logical is None or effective is None:
        return None
    if effective > logical:
        effective = logical
    offset = logical - effective
    if offset <= 0:
        if hasattr(request, EFFECTIVE_KV_OFFSET_ATTR):
            try:
                delattr(request, EFFECTIVE_KV_OFFSET_ATTR)
            except Exception:
                setattr(request, EFFECTIVE_KV_OFFSET_ATTR, None)
        return 0
    setattr(request, EFFECTIVE_KV_OFFSET_ATTR, offset)
    return offset


def prepare_request_effective_num_computed(request: Any) -> int | None:
    """Refresh request effective-num-computed marker for the current schedule step."""
    if request is None:
        return None
    logical = _to_non_negative_int(getattr(request, "num_computed_tokens", None))
    if logical is None:
        return None
    offset = _to_non_negative_int(getattr(request, EFFECTIVE_KV_OFFSET_ATTR, None))
    if offset is None or offset <= 0:
        if hasattr(request, EFFECTIVE_NUM_COMPUTED_ATTR):
            try:
                delattr(request, EFFECTIVE_NUM_COMPUTED_ATTR)
            except Exception:
                setattr(request, EFFECTIVE_NUM_COMPUTED_ATTR, None)
        return None
    if logical == 0 or logical < offset:
        clear_request_allocation_sync_state(request)
        return None
    effective = logical - offset
    setattr(request, EFFECTIVE_NUM_COMPUTED_ATTR, effective)
    return effective


def resolve_request_effective_num_computed(request: Any) -> int | None:
    """Resolve safe effective-num-computed marker consumed by allocation patch."""
    if request is None:
        return None
    logical = _to_non_negative_int(getattr(request, "num_computed_tokens", None))
    effective = _to_non_negative_int(getattr(request, EFFECTIVE_NUM_COMPUTED_ATTR, None))
    if logical is None or effective is None:
        return None
    if effective > logical:
        return None
    return effective


def update_request_effective_kv_offset_anchored(
    *, request: Any, cache_len_after: Any, scheduler_nct_pre: Any
) -> int | None:
    """SHIM (08-18): offset = nct_pre - cache_len_after, BOTH pre-forward quantities of
    the compression step, so effective(t) = logical(t) - offset equals the true compacted
    length at every later step, independent of when registration runs relative to
    schedule-time nct advancement or batch-queue lookahead. (Legacy path used live
    num_computed_tokens, overstating the offset by 1 + lookahead_depth -> every
    block-boundary allocation late by that many steps -> orphaned boundary token.)"""
    nct = _to_non_negative_int(scheduler_nct_pre)
    eff = _to_non_negative_int(cache_len_after)
    if nct is None or eff is None:
        return None
    offset = nct - eff
    if offset <= 0:
        clear_request_allocation_sync_state(request)
        return 0
    setattr(request, EFFECTIVE_KV_OFFSET_ATTR, offset)
    return offset
