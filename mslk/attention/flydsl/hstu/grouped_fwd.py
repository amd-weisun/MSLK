# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""Grouped ragged HSTU self-attention forward (``mslk::hstu_grouped_fwd``) on
the tiled FlyDSL kernel of :mod:`.ragged_fwd`.

G groups of Z sequences share packed Q/K/V and the head count and dims; each
group has its own max length, window, full-attention size and contextual
length (``[G]`` device tensors), and every head is SiLU (T1's
``grouped_ragged_hstu_attention_ref``).

The per-group values stay on the device, so nothing here waits on the host
and the call captures in a CUDA graph:

- Programs to tiles. A sequence of length n has ``ceil(n / BLOCK_M)`` query
  tiles; their exclusive prefix sum and, per tile slot, the owning sequence
  (``searchsorted``) are computed on the device. The grid has
  ``ceil(T / BLOCK_M) + G * Z`` tile slots per head, an upper bound on the
  total tile count; slots past it exit.
- Per-group settings. Each program loads its group's settings, so the kernel
  is compiled with every mask feature the mask type allows and a run-time
  value of 0 turns a feature off.
- Scale modes (T2's rule): no ``attn_scale_tensor`` is
  ``1 / max_seq_len_tensor[g]``; a 1-D ``[G]`` tensor is one scale per group;
  a tensor of 2+ dims ``[T, ...]`` is one scale per packed query token (its
  first column).
"""

from typing import Optional

import torch

from .common import LOWER, MASK_TYPES, MaskSpec, UPPER
from .ragged_fwd import (
    _arch,
    _DTYPES,
    _first_supported,
    _LOG2E,
    build_ragged_fwd,
    check_supported,
    FwdConfig,
    SCALE_GROUP,
    SCALE_INV_N,
    SCALE_TOKEN,
)


def grouped_metadata(
    seq_offsets: torch.Tensor,
    max_seq_len_tensor: torch.Tensor,
    max_attn_len_tensor: torch.Tensor,
    full_attn_size_tensor: torch.Tensor,
    contextual_seq_len_tensor: torch.Tensor,
    num_tokens: int,
    block_m: int,
):
    """The kernel's ``group_meta`` (int32, on the device) and its section
    bases: ``[0, slots)`` each tile slot's sequence (the sequence count past
    the last tile), ``[slots, slots + B + 1)`` the exclusive prefix sum of
    query tiles per sequence, then 4 settings per group."""
    num_seqs = seq_offsets.shape[0] - 1
    slots = (num_tokens + block_m - 1) // block_m + num_seqs
    offsets = seq_offsets.to(torch.int64)
    tiles = (offsets[1:] - offsets[:-1] + (block_m - 1)) // block_m
    prefix = torch.zeros(num_seqs + 1, dtype=torch.int64, device=seq_offsets.device)
    torch.cumsum(tiles, 0, out=prefix[1:])
    slot_ids = torch.arange(slots, dtype=torch.int64, device=seq_offsets.device)
    tile_seq = torch.searchsorted(prefix[1:], slot_ids, right=True)
    settings = torch.stack(
        [
            max_seq_len_tensor.to(torch.int64),
            max_attn_len_tensor.to(torch.int64),
            full_attn_size_tensor.to(torch.int64),
            contextual_seq_len_tensor.to(torch.int64),
        ],
        dim=1,
    ).reshape(-1)
    meta = torch.cat([tile_seq, prefix, settings]).to(torch.int32)
    return meta, slots, slots + num_seqs + 1


def hstu_grouped_fwd(
    Z: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    group_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    max_seq_len_tensor: torch.Tensor,
    max_attn_len_tensor: torch.Tensor,
    full_attn_size_tensor: torch.Tensor,
    contextual_seq_len_tensor: torch.Tensor,
    alpha: float,
    invalid_attn_mask_type: str,
    attn_scale_tensor: Optional[torch.Tensor] = None,
    *,
    config: Optional[FwdConfig] = None,
) -> torch.Tensor:
    """``mslk::hstu_grouped_fwd``; returns the output ``[T, H, Dv]``.

    Arguments as validated by T2 (``validate_grouped_fwd``). As in the ragged
    forward, every sequence of group g must be at most
    ``max_seq_len_tensor[g]`` tokens (checking it would need a host sync), and
    per-group settings T1 rejects are not checked but ignored: a window on an
    upper-triangular mask, a contextual length or full-attention size on a
    mask other than lower-triangular, and a full-attention size without a
    window."""
    from mslk.flydsl.jit import run_compiled

    if invalid_attn_mask_type not in MASK_TYPES:
        raise ValueError(f"unknown mask type {invalid_attn_mask_type!r}")
    num_tokens, num_heads, head_dim = q.shape
    hidden_dim = v.shape[2]
    num_seqs = seq_offsets.shape[0] - 1
    num_groups = group_offsets.shape[0] - 1
    out = torch.empty(
        (num_tokens, num_heads, hidden_dim), dtype=q.dtype, device=q.device
    )
    if num_tokens == 0 or num_seqs == 0:
        return out.zero_()

    arch = _arch()
    cfg = config or _first_supported(q, head_dim, hidden_dim, arch)
    reason = check_supported(q, head_dim, hidden_dim, cfg, arch)
    if reason is not None:
        raise NotImplementedError(f"FlyDSL grouped HSTU forward: {reason}")

    # Every feature T1 allows on the mask type (contextual and full
    # attention: lower-triangular only; a window: not upper-triangular);
    # per-group zeros turn them off.
    lower = invalid_attn_mask_type == LOWER
    spec = MaskSpec(
        invalid_attn_mask_type,
        has_window=invalid_attn_mask_type != UPPER,
        has_contextual=lower,
        has_targets=num_targets is not None,
        has_full_attn=lower,
    )
    if attn_scale_tensor is None:
        scale_mode = SCALE_INV_N
        attn_scale = torch.empty(1, dtype=torch.float32, device=q.device)
    elif attn_scale_tensor.dim() == 1:
        if attn_scale_tensor.shape[0] != num_groups:
            raise ValueError(
                f"a 1-D attn_scale_tensor has one scale per group ({num_groups}), "
                f"got {attn_scale_tensor.shape[0]}"
            )
        scale_mode = SCALE_GROUP
        attn_scale = attn_scale_tensor.to(torch.float32).contiguous()
    else:
        if attn_scale_tensor.shape[0] != num_tokens:
            raise ValueError(
                f"a per-token attn_scale_tensor has one row per token "
                f"({num_tokens}), got {attn_scale_tensor.shape[0]}"
            )
        scale_mode = SCALE_TOKEN
        attn_scale = (
            attn_scale_tensor.reshape(num_tokens, -1)[:, 0]
            .to(torch.float32)
            .contiguous()
        )

    meta, prefix_base, settings_base = grouped_metadata(
        seq_offsets,
        max_seq_len_tensor,
        max_attn_len_tensor,
        full_attn_size_tensor,
        contextual_seq_len_tensor,
        num_tokens,
        cfg.block_m,
    )
    launcher = build_ragged_fwd(
        head_dim,
        hidden_dim,
        _DTYPES[q.dtype],
        spec,
        scale_mode,
        False,
        cfg,
        arch,
        lower,
        False,
        True,
    )
    seq_offsets = seq_offsets.to(torch.int32)
    num_targets = (
        num_targets.to(torch.int32) if num_targets is not None else seq_offsets
    )
    # SiLU in base 2; -1/log2e undoes the base change. The 1/max_seq_len of
    # the default scale is applied per group in the kernel.
    score_scale, out_scale = alpha * -_LOG2E, -1.0 / _LOG2E
    # M is not written for SiLU heads; one placeholder layout for all calls.
    lse = torch.empty((1, 1), dtype=torch.float32, device=q.device)
    run_compiled(
        launcher,
        q,
        k,
        v,
        seq_offsets,
        num_targets,
        attn_scale,
        out,
        lse,
        meta,
        num_seqs,
        num_heads,
        0,
        score_scale,
        out_scale,
        0,
        0,
        0,
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        Z,
        prefix_base,
        settings_base,
        torch.cuda.current_stream(q.device),
    )
    return out
