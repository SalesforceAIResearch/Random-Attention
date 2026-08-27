# Modifications Copyright (c) 2026, Salesforce, Inc.  SPDX-License-Identifier: Apache-2.0
# (the original portions remain under their own license -- see the provenance note below and THIRD_PARTY_NOTICES.md)
"""VERBATIM port of TriAttention (github.com/WeianMao/triattention) scoring functions.

These are copied EXACTLY from the official repo (triattention/methods/pruning_utils.py)
so our in-engine `triattn_pl` mode uses the reference math, not a reconstruction.
Do NOT "clean up" or rename — fidelity to the reference is the point.

Only addition: `unrotate_key_complex` — the standard-RoPE inverse (complex multiply by
exp(-i*pos*omega)), equivalent to the repo's `invert_rope`/`_invert_rope_wan` for the
half-split (GPT-NeoX/Llama/Qwen3/Phi3) convention, since our KV cache stores POST-RoPE keys.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch


# ---------- verbatim: triattention/methods/pruning_utils.py ----------

def to_complex_pairs(tensor: torch.Tensor, *, style: str = "half") -> torch.Tensor:
    if tensor.size(-1) % 2 != 0:
        raise ValueError("Head dimension must be even to form complex pairs")
    real_dtype = torch.float32 if tensor.dtype in (torch.bfloat16, torch.float16) else tensor.dtype
    tensor_real = tensor.to(dtype=real_dtype)
    if style == "interleaved":
        real = tensor_real[..., ::2].contiguous()
        imag = tensor_real[..., 1::2].contiguous()
        return torch.complex(real, imag)
    freq_count = tensor.shape[1] // 2
    real = tensor_real[:, :freq_count].contiguous()
    imag = tensor_real[:, freq_count:].contiguous()
    return torch.complex(real, imag)


def compute_frequency_scaling(rotary, head_dim: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    position_ids = torch.zeros(1, 1, device=device, dtype=torch.long)
    probe = torch.zeros(1, 1, head_dim, device=device, dtype=dtype)
    cos, sin = rotary(probe, position_ids)
    cos0 = cos[0, 0]
    sin0 = sin[0, 0]
    scale = torch.sqrt(cos0[0::2].pow(2) + sin0[0::2].pow(2))
    return scale.to(device=device, dtype=torch.float32)


def build_geometric_offsets(max_length: int, device: torch.device) -> torch.Tensor:
    if max_length < 1:
        raise ValueError("offset_max_length must be >= 1")
    offsets: List[float] = []
    value = 1
    while value <= max_length:
        offsets.append(float(value))
        value *= 2
    return torch.tensor(offsets, device=device, dtype=torch.float32)


def compute_frequency_statistics_from_means(
    q_mean_complex: torch.Tensor,
    q_abs_mean: torch.Tensor,
    k_unrot: torch.Tensor,
    *,
    style: str = "half",
    disable_mlr: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k_complex = to_complex_pairs(k_unrot, style=style)
    q_mean_abs = torch.abs(q_mean_complex)
    k_abs = torch.abs(k_complex)
    relative = q_mean_complex.unsqueeze(0) * torch.conj(k_complex)
    phi = torch.atan2(relative.imag, relative.real)
    amp = q_mean_abs.unsqueeze(0) * k_abs
    if disable_mlr:
        extra = q_abs_mean.unsqueeze(0) * k_abs
    else:
        extra = (q_abs_mean - q_mean_abs).unsqueeze(0) * k_abs
    return amp, phi, extra


def score_keys_for_round(
    key_indices: torch.Tensor,
    round_start: int,
    amp: torch.Tensor,
    phi: torch.Tensor,
    omega: torch.Tensor,
    extra: torch.Tensor,
    offsets: torch.Tensor,
    aggregation: str,
    freq_scale_sq: torch.Tensor,
    disable_trig: bool = False,
) -> torch.Tensor:
    if key_indices.numel() == 0:
        return torch.empty(0, device=amp.device, dtype=torch.float32)

    base_delta = round_start - key_indices.to(device=amp.device, dtype=torch.float32)
    delta_grid = base_delta.unsqueeze(1) + offsets.unsqueeze(0)

    freq_scale_sq = freq_scale_sq.to(device=amp.device, dtype=torch.float32)
    phase = delta_grid.unsqueeze(2) * omega.view(1, 1, -1) + phi.unsqueeze(1)

    cos_phase = torch.cos(phase)

    scale = freq_scale_sq.view(1, 1, -1)

    base_scores = (amp.unsqueeze(1) * scale * cos_phase).sum(dim=2)
    # additive term uses original freq_scale_sq (not affected by high-freq masking)
    additive = (extra * freq_scale_sq.view(1, -1)).sum(dim=1, keepdim=True)
    combined = additive if disable_trig else (base_scores + additive)

    if aggregation == "mean":
        return combined.mean(dim=1)
    return combined.max(dim=1).values


@dataclass
class HeadFrequencyStats:
    q_mean_complex: torch.Tensor
    q_abs_mean: torch.Tensor


def save_head_frequency_stats(
    output_path: Path,
    sampled_heads: Sequence[Tuple[int, int]],
    stats_map: Dict[Tuple[int, int], HeadFrequencyStats],
    metadata: Dict[str, object],
) -> None:
    payload: Dict[str, object] = {
        "metadata": {
            **metadata,
            "sampled_heads": [[int(layer), int(head)] for layer, head in sampled_heads],
        },
        "stats": {},
    }
    for (layer, head), head_stats in stats_map.items():
        key = f"layer{layer:02d}_head{head:02d}"
        payload["stats"][key] = {
            "q_mean_real": head_stats.q_mean_complex.real.cpu(),
            "q_mean_imag": head_stats.q_mean_complex.imag.cpu(),
            "q_abs_mean": head_stats.q_abs_mean.cpu(),
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def load_head_frequency_stats(stats_path, device: torch.device):
    payload = torch.load(stats_path, map_location=device)
    metadata = payload["metadata"]
    stats_raw = payload["stats"]
    sampled_heads = [tuple(item) for item in metadata["sampled_heads"]]
    stats: Dict[Tuple[int, int], HeadFrequencyStats] = {}
    for layer, head in sampled_heads:
        key = f"layer{layer:02d}_head{head:02d}"
        entry = stats_raw.get(key)
        if entry is None:
            continue
        q_mean_complex = torch.complex(
            entry["q_mean_real"].to(device=device, dtype=torch.float32),
            entry["q_mean_imag"].to(device=device, dtype=torch.float32),
        )
        q_abs_mean = entry["q_abs_mean"].to(device=device, dtype=torch.float32)
        stats[(int(layer), int(head))] = HeadFrequencyStats(
            q_mean_complex=q_mean_complex, q_abs_mean=q_abs_mean,
        )
    return metadata, stats


# ---------- our addition: standard-RoPE inverse in complex space ----------

def unrotate_key_complex(k_post: torch.Tensor, positions: torch.Tensor, omega: torch.Tensor,
                         *, style: str = "half") -> torch.Tensor:
    """Recover the PRE-RoPE real key from a POST-RoPE real key.

    Standard half-split RoPE rotates band f of a token at absolute position p by
    angle p*omega[f]:  k_post_complex[f] = k_pre_complex[f] * exp(i*p*omega[f]).
    Inverse: k_pre_complex[f] = k_post_complex[f] * exp(-i*p*omega[f]).

    k_post:    [T, head_dim]  real (post-RoPE key band pairs, half-split)
    positions: [T]            long/float absolute token positions
    omega:     [head_dim//2]  RoPE inv_freq (angular freq per band)
    returns:   [T, head_dim]  real pre-RoPE key in the same half-split layout
    """
    k_c = to_complex_pairs(k_post, style=style)                       # [T, F] complex
    ang = positions.to(k_c.real.dtype).unsqueeze(1) * omega.view(1, -1)  # [T, F]
    rot = torch.polar(torch.ones_like(ang), -ang)                    # exp(-i*ang)
    k_pre = k_c * rot
    return torch.cat([k_pre.real, k_pre.imag], dim=-1)               # half-split real
