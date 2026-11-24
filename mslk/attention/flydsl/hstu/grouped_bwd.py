# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""Grouped ragged HSTU self-attention backward (``mslk::hstu_grouped_bwd``)
on gfx950: the ragged SiLU backward of :mod:`.ragged_bwd` with per-group
settings (T1's ``grouped_ragged_hstu_attention_ref``; every head is SiLU).

As in the grouped forward, the per-group values stay on the device and
nothing waits on the host (the call captures in a CUDA graph):

- Programs to KV tiles: the exclusive prefix sum of KV tiles per sequence and,
  per tile slot, the owning sequence (``searchsorted``); the grid has
  ``ceil(T / BLOCK_N) + G * Z`` slots per head and surplus slots do nothing.
- Per-sequence settings (num_targets and the group's window, contextual
  length and full-attention size) are gathered on the device; the kernel is
  compiled with every mask feature the mask type allows.
- The per-row output scale: ``1 / max_seq_len[g]``, a ``[G]`` scale per group,
  or a ``[T, ...]`` scale per token (its first column), as in the forward.
"""

import functools
from typing import Optional

import torch

from .common import LOWER, MASK_TYPES, MaskSpec, UPPER
from .ragged_bwd import _arch, _DIMS, build_ragged_bwd, decline_reason, launch_bwd


def grouped_bwd_metadata(
    Z: int,
    seq_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    max_attn_len_tensor: torch.Tensor,
    contextual_seq_len_tensor: torch.Tensor,
    full_attn_size_tensor: torch.Tensor,
    num_tokens: int,
    block_n: int,
) -> torch.Tensor:
    """The kernel's grouped metadata (int32, see HSTU_GROUPED in
    :mod:`mslk.attention.flydsl.fmha_bwd_mfma_gfx950`)."""
    device = seq_offsets.device
    num_seqs = seq_offsets.shape[0] - 1
    slots = (num_tokens + block_n - 1) // block_n + num_seqs
    offsets = seq_offsets.to(torch.int64)
    tiles = (offsets[1:] - offsets[:-1] + (block_n - 1)) // block_n
    prefix = torch.zeros(num_seqs + 1, dtype=torch.int64, device=device)
    torch.cumsum(tiles, 0, out=prefix[1:])
    slot_ids = torch.arange(slots, dtype=torch.int64, device=device)
    tile_seq = torch.searchsorted(prefix[1:], slot_ids, right=True)
    group = torch.arange(num_seqs, device=device) // Z
    targets = (
        num_targets.to(torch.int64)
        if num_targets is not None
        else torch.zeros(num_seqs, dtype=torch.int64, device=device)
    )
    settings = torch.stack(
        [
            targets,
            max_attn_len_tensor.to(torch.int64)[group],
            contextual_seq_len_tensor.to(torch.int64)[group],
            full_attn_size_tensor.to(torch.int64)[group],
        ],
        dim=1,
    ).reshape(-1)
    return torch.cat([tile_seq, prefix, settings]).to(torch.int32)


def _row_scale(
    Z: int,
    seq_offsets: torch.Tensor,
    max_seq_len_tensor: torch.Tensor,
    attn_scale_tensor: Optional[torch.Tensor],
    num_tokens: int,
    num_heads: int,
) -> torch.Tensor:
    """The per-row output scale in the kernel's LSE layout [H, T]."""
    device = seq_offsets.device
    num_groups = max_seq_len_tensor.numel()
    if attn_scale_tensor is not None and attn_scale_tensor.dim() >= 2:
        if attn_scale_tensor.shape[0] != num_tokens:
            raise ValueError(
                f"a per-token attn_scale_tensor has one row per token "
                f"({num_tokens}), got {attn_scale_tensor.shape[0]}"
            )
        per_token = attn_scale_tensor.reshape(num_tokens, -1)[:, 0].to(torch.float32)
    else:
        token_ids = torch.arange(num_tokens, dtype=torch.int64, device=device)
        token_seq = torch.searchsorted(
            seq_offsets.to(torch.int64)[1:], token_ids, right=True
        )
        token_group = token_seq // Z
        if attn_scale_tensor is None:
            per_group = 1.0 / max_seq_len_tensor.to(torch.float32)
        else:
            if attn_scale_tensor.shape[0] != num_groups:
                raise ValueError(
                    f"a 1-D attn_scale_tensor has one scale per group "
                    f"({num_groups}), got {attn_scale_tensor.shape[0]}"
                )
            per_group = attn_scale_tensor.to(torch.float32)
        per_token = per_group[token_group]
    return per_token.expand(num_heads, num_tokens).contiguous().view(-1, 1)


def hstu_grouped_bwd(
    dout: torch.Tensor,
    Z: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: Optional[torch.Tensor],
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
    Out: Optional[torch.Tensor] = None,
    shared_kv: bool = False,
) -> None:
    """``mslk::hstu_grouped_bwd``: writes dq, dk and dv in place (with
    ``shared_kv``, dK + dV into dk). Arguments as validated by T2; ``Out`` is
    not needed (SiLU heads). Every sequence of group g must be at most
    ``max_seq_len_tensor[g]`` tokens, and per-group settings T1 rejects are
    ignored as in the grouped forward."""
    from mslk.attention.flydsl.fmha_bwd_mfma_gfx950 import gfx950_tile_defaults

    if invalid_attn_mask_type not in MASK_TYPES:
        raise ValueError(f"unknown mask type {invalid_attn_mask_type!r}")
    if shared_kv:
        v = k
    arch = _arch()
    reason = decline_reason(q, v, 0, arch)
    if reason is not None:
        raise NotImplementedError(f"FlyDSL grouped HSTU backward: {reason}")
    num_tokens, num_heads, dim_qk = q.shape
    dim_v = v.shape[2]
    num_seqs = seq_offsets.shape[0] - 1
    if num_tokens == 0 or num_seqs == 0:
        dq.zero_()
        dk.zero_()
        if dv is not None:
            dv.zero_()
        return

    dim = next(d for d in _DIMS if d >= max(dim_qk, dim_v))
    # Tiles from the mean sequence length (the per-group lengths are on the
    # device).
    block_m, block_n = gfx950_tile_defaults(dim, num_tokens // num_seqs, arch)
    lower = invalid_attn_mask_type == LOWER
    spec = MaskSpec(
        invalid_attn_mask_type,
        has_window=invalid_attn_mask_type != UPPER,
        has_contextual=lower,
        has_targets=num_targets is not None,
        has_full_attn=lower,
    )
    dtype_str = "bf16" if q.dtype == torch.bfloat16 else "f16"
    launcher = functools.partial(
        build_ragged_bwd,
        dim,
        dim_qk,
        dim_v,
        dtype_str,
        block_m,
        block_n,
        float(alpha),
        spec,
        arch,
        False,
        True,
    )
    meta = grouped_bwd_metadata(
        Z,
        seq_offsets,
        num_targets,
        max_attn_len_tensor,
        contextual_seq_len_tensor,
        full_attn_size_tensor,
        num_tokens,
        block_n,
    )
    row_scale = _row_scale(
        Z, seq_offsets, max_seq_len_tensor, attn_scale_tensor, num_tokens, num_heads
    )
    launch_bwd(
        launcher,
        dout,
        q,
        k,
        v,
        dq,
        dk,
        dv,
        seq_offsets,
        1,
        block_m,
        row_scale,
        None,
        meta,
        (0, 0, 0),
        shared_kv,
    )
