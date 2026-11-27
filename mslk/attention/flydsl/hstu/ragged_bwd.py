# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""Ragged HSTU self-attention backward (``mslk::hstu_ragged_bwd``) on gfx950,
on the HSTU mode of :mod:`mslk.attention.flydsl.fmha_bwd_mfma_gfx950`."""

import functools
import math
from typing import Optional

import torch

from .common import MaskSpec

_DIMS = (32, 64, 128, 256)
_SMALL_WINDOW = 64


def _arch() -> str:
    from flydsl.runtime.device import get_rocm_arch  # pyre-ignore[21]

    return get_rocm_arch() or ""


def decline_reason(
    q: torch.Tensor, v: torch.Tensor, num_softmax_heads: int, arch: str
) -> Optional[str]:
    if not arch.startswith("gfx950"):
        return f"needs gfx950, got {arch!r}"
    if q.dtype not in (torch.bfloat16, torch.float16):
        return f"unsupported dtype {q.dtype}"
    if num_softmax_heads not in (0, q.shape[1]):
        return f"num_softmax_heads must be 0 or H, got {num_softmax_heads}"
    if max(q.shape[2], v.shape[2]) > _DIMS[-1] or q.shape[2] % 2 or v.shape[2] % 2:
        return f"unsupported head dims {q.shape[2]}, {v.shape[2]}"
    return None


@functools.lru_cache(maxsize=256)
def build_ragged_bwd(
    dim: int,
    dim_qk: int,
    dim_v: int,
    dtype_str: str,
    block_m: int,
    block_n: int,
    alpha: float,
    spec: MaskSpec,
    arch: str,
    softmax: bool = False,
    grouped: bool = False,
    direct: bool = False,
    shared: bool = False,
    sort: bool = False,
    no_dq: bool = False,
):
    from mslk.attention.flydsl.fmha_bwd_mfma_gfx950 import (
        compile_fmha_bwd_dqdkdv_mfma_gfx950,
    )

    return compile_fmha_bwd_dqdkdv_mfma_gfx950(
        D=dim,
        dtype_str=dtype_str,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        scale=alpha,
        varlen=True,
        gpu_arch=arch,
        ck_scope_dvdk=True,
        D_QK=dim_qk,
        D_V=dim_v,
        hstu=spec,
        hstu_softmax=softmax,
        hstu_grouped=grouped,
        hstu_direct_dkv=direct,
        hstu_shared_dkv=shared,
        hstu_sorted=sort,
        hstu_no_dq=no_dq,
    )


@functools.lru_cache(maxsize=16)
def _build_preprocess(dim_v: int, dtype_str: str):
    from mslk.attention.flydsl.fmha_bwd_preprocess import compile_fmha_bwd_preprocess

    return compile_fmha_bwd_preprocess(D=dim_v, dtype_str=dtype_str)


def _softmax_d(dout: torch.Tensor, out: torch.Tensor, dtype_str: str) -> torch.Tensor:
    """D = rowsum(dO * Out) as [T * H, 1] f32."""
    from mslk.flydsl.jit import run_compiled

    num_tokens, num_heads, dim_v = dout.shape
    rows = num_tokens * num_heads
    d_vec = torch.empty(rows, 1, dtype=torch.float32, device=dout.device)
    do2 = dout.contiguous().view(torch.int16).view(rows, dim_v)
    o2 = out.to(dout.dtype).contiguous().view(torch.int16).view(rows, dim_v)
    # The kernel's offsets are 32-bit: chunks stay under 2^31 bytes.
    chunk = (2**31 - 1) // (dim_v * 2)
    launcher = _build_preprocess(dim_v, dtype_str)
    stream = torch.cuda.current_stream(dout.device)
    for lo in range(0, rows, chunk):
        hi = min(lo + chunk, rows)
        run_compiled(launcher, do2[lo:hi], o2[lo:hi], d_vec[lo:hi], hi - lo, stream)
    return d_vec


def _rows(t: torch.Tensor, dim: int) -> torch.Tensor:
    """``t`` with packed heads (the kernel takes a token pitch, not a head pitch)."""
    if t.stride(2) != 1 or t.stride(1) != dim or t.stride(0) % dim:
        t = t.contiguous()
    return t


def hstu_ragged_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: Optional[torch.Tensor],
    seq_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    attn_scale: Optional[torch.Tensor],
    N: int,
    alpha: float,
    max_attn_len: int,
    invalid_attn_mask_type: str,
    contextual_seq_len: int,
    sort_by_length_indices: Optional[torch.Tensor],
    full_attn_size: int,
    num_softmax_heads: int = 0,
    M: Optional[torch.Tensor] = None,
    Out: Optional[torch.Tensor] = None,
    shared_kv: bool = False,
) -> None:
    """``mslk::hstu_ragged_bwd``: writes dq, dk and dv in place (with
    ``shared_kv``, dK + dV into dk). Softmax heads need the forward's ``M``
    and ``Out``. Every sequence must be at most ``N`` tokens."""
    from mslk.attention.flydsl.fmha_bwd_mfma_gfx950 import gfx950_tile_defaults

    if shared_kv:
        v = k
    arch = _arch()
    reason = decline_reason(q, v, num_softmax_heads, arch)
    if reason is not None:
        raise NotImplementedError(f"FlyDSL ragged HSTU backward: {reason}")
    num_tokens, num_heads, dim_qk = q.shape
    dim_v = v.shape[2]
    num_seqs = seq_offsets.shape[0] - 1
    if num_tokens == 0 or num_seqs == 0 or N == 0:
        dq.zero_()
        dk.zero_()
        if dv is not None:
            dv.zero_()
        return

    dim = next(d for d in _DIMS if d >= max(dim_qk, dim_v))
    block_m, block_n = gfx950_tile_defaults(dim, N, arch)
    if dim == 64 and N <= 1024:
        block_m, block_n = 32, 64
    if dim >= 64 and 0 < max_attn_len <= _SMALL_WINDOW:
        block_m, block_n = 32, 64
    split = num_softmax_heads == 0 and split_dq(
        dim,
        dim_qk,
        dim_v,
        N,
        max_attn_len,
        full_attn_size,
        lower=invalid_attn_mask_type == "lower_triangular",
    )
    if split and dim == 128:
        block_m, block_n = 32, 192
    spec = MaskSpec.from_args(
        invalid_attn_mask_type,
        max_attn_len,
        contextual_seq_len,
        full_attn_size,
        has_targets=num_targets is not None,
    )
    dtype_str = "bf16" if q.dtype == torch.bfloat16 else "f16"
    softmax = num_softmax_heads > 0
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
        softmax,
    )

    d_vec = None
    if softmax:
        # Base-2 LSE M [T, H] -> the kernel's natural-log LSE [H, T].
        if M is None or Out is None:
            raise ValueError("softmax heads need the forward's M and Out")
        row_scale = (M.to(torch.float32) * math.log(2.0)).t().contiguous().view(-1, 1)
        d_vec = _softmax_d(dout, Out, dtype_str)
    # SiLU heads: the per-row output scale in the kernel's LSE slot [H, T].
    elif attn_scale is None:
        row_scale = torch.full(
            (num_heads * num_tokens, 1), 1.0 / N, dtype=torch.float32, device=q.device
        )
    else:
        per_token = attn_scale.to(torch.float32).reshape(-1)
        if attn_scale.ndim == 0:
            per_token = per_token.expand(num_tokens)
        elif per_token.numel() != num_tokens:
            raise ValueError(
                "attn_scale must be a 0-dim tensor or have one value per token "
                f"(shape [{num_tokens}]), got {tuple(attn_scale.shape)}"
            )
        row_scale = per_token.expand(num_heads, num_tokens).contiguous().view(-1, 1)

    targets = num_targets.to(torch.int32) if num_targets is not None else None
    dq_targets = targets if targets is not None else seq_offsets.to(torch.int32)
    sort_idx = sort_by_length_indices
    if sort_idx is not None:
        # Kernel metadata: num_targets (zeros without), then the order.
        targets = torch.cat(
            [
                targets
                if targets is not None
                else torch.zeros(num_seqs, dtype=torch.int32, device=q.device),
                sort_idx.to(torch.int32),
            ]
        )
        launcher = functools.partial(launcher, sort=True)
    dq_fn = None
    if split:
        from .ragged_bwd_dq import BLOCK_M as DQ_BLOCK_M, dq_runner

        dq_fn = dq_runner(
            q,
            k,
            v,
            dout,
            row_scale,
            seq_offsets,
            dq_targets,
            (N + DQ_BLOCK_M - 1) // DQ_BLOCK_M,
            alpha,
            spec,
            arch,
            (max_attn_len, contextual_seq_len, full_attn_size),
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
        N,
        block_m,
        row_scale,
        d_vec,
        targets,
        (max_attn_len, contextual_seq_len, full_attn_size),
        shared_kv,
        dq_fn,
    )


def split_dq(
    dim: int,
    dim_qk: int,
    dim_v: int,
    seq_len: int,
    max_attn_len: int = 0,
    full_attn_size: int = 0,
    grouped: bool = False,
    lower: bool = True,
) -> bool:
    """Whether SiLU heads compute dQ with :mod:`.ragged_bwd_dq` and dK / dV
    with the fused kernel minus its dQ step. ``seq_len``: the max sequence
    length (ragged) or the mean (grouped). Needs Dqk % 32 == 0, Dv % 16 == 0."""
    if dim_qk % 32 or dim_v % 16:
        return False
    if dim == 128:
        return (
            not grouped
            and lower
            and full_attn_size == 0
            and (max_attn_len == 0 or max_attn_len >= 256)
        )
    return dim == 64 or (dim == 256 and seq_len >= 1024)


def launch_bwd(
    launcher,
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: Optional[torch.Tensor],
    seq_offsets: torch.Tensor,
    N: int,
    block_m: int,
    row_scale: torch.Tensor,
    d_vec: Optional[torch.Tensor],
    hstu_meta: Optional[torch.Tensor],
    hstu_ints,
    shared_kv: bool,
    dq_fn=None,
) -> None:
    """Run the backward into dq, dk and dv. ``launcher``: :func:`build_ragged_bwd`
    with all but ``direct``/``shared``/``no_dq`` bound. ``row_scale``: the LSE
    slot [H, T]; ``d_vec``: softmax's D (None for SiLU); ``hstu_meta``: targets
    (ragged) or the grouped tile map; ``hstu_ints``: (max_attn_len,
    contextual_seq_len, full_attn_size). ``dq_fn(dq_out)``, when given, writes
    dQ and the kernel computes dK / dV only."""
    from mslk.flydsl.jit import run_compiled

    num_tokens, num_heads, dim_qk = q.shape
    dim_v = v.shape[2]
    dim = next(d for d in _DIMS if d >= max(dim_qk, dim_v))
    num_seqs = seq_offsets.shape[0] - 1
    k, v = _rows(k, dim_qk), _rows(v, dim_v)
    if k.stride(0) // dim_qk != v.stride(0) // dim_v:
        # The kernel takes one KV token pitch.
        k, v = k.contiguous(), v.contiguous()
    q4, k4, v4, do4 = (
        _rows(t, d).unsqueeze(0)
        for t, d in ((q, dim_qk), (k, dim_qk), (v, dim_v), (dout, dim_v))
    )
    as_i16 = lambda t: t.view(torch.int16)  # noqa: E731
    seqstart = seq_offsets.to(torch.int32)
    direct = (
        dim_qk % 4 == 0
        and dim_v % 4 == 0
        and dk.is_contiguous()
        and (shared_kv or dv.is_contiguous())
    )
    # 2-D scratch: FlyDSL passes dims as int32 and T * H * dim can pass 2^31.
    grad_shape = (num_tokens * num_heads, dim)
    if dq_fn is None:
        dq_f32 = torch.zeros(grad_shape, dtype=torch.float32, device=q.device)
    else:
        dq_f32 = torch.zeros(1, dtype=torch.float32, device=q.device)
        launcher = functools.partial(launcher, no_dq=True)
    if direct:
        dk_out = as_i16(dk).view(num_tokens * num_heads, dim_qk)
        dv_out = dk_out if shared_kv else as_i16(dv).view(-1, dim_v)
    else:
        dk_out = torch.zeros(grad_shape, dtype=torch.float32, device=q.device)
        dv_out = torch.zeros(grad_shape, dtype=torch.float32, device=q.device)
    placeholder_f32 = torch.zeros(1, dtype=torch.float32, device=q.device)
    run_compiled(
        launcher(direct=direct, shared=direct and shared_kv),
        as_i16(q4),
        as_i16(k4),
        as_i16(v4),
        as_i16(do4),
        dv_out,
        dk_out,
        dq_f32,
        row_scale,
        row_scale if d_vec is None else d_vec,
        num_seqs,
        N,
        N,
        num_heads,
        (N + block_m - 1) // block_m,
        q4.stride(1) // dim_qk,
        k4.stride(1) // dim_qk,
        do4.stride(1) // dim_v,
        seqstart,
        seqstart,
        num_tokens,
        as_i16(q4),  # Bias: unused
        0,
        0,
        0,
        placeholder_f32,  # dBias: unused
        0,
        seqstart,  # Dropout: unused
        0,
        0,
        0,
        seqstart if hstu_meta is None else hstu_meta,
        *hstu_ints,
        torch.cuda.current_stream(q.device),
    )
    shape = (num_tokens, num_heads, dim)
    if dq_fn is None:
        dq.copy_(dq_f32.view(shape)[..., :dim_qk])
    elif dq.is_contiguous():
        dq_fn(dq)
    else:
        dq_tmp = torch.empty(dq.shape, dtype=dq.dtype, device=dq.device)
        dq_fn(dq_tmp)
        dq.copy_(dq_tmp)
    if direct:
        return
    dk_f32, dv_f32 = dk_out, dv_out
    if shared_kv:
        dk.copy_(dk_f32.view(shape)[..., :dim_qk] + dv_f32.view(shape)[..., :dim_v])
    else:
        dk.copy_(dk_f32.view(shape)[..., :dim_qk])
        dv.copy_(dv_f32.view(shape)[..., :dim_v])
