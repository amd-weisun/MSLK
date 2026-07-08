# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# pyre-unsafe
"""WP-A3: FlyDSL FMHA backward, registered as an opt-in `AttentionBwOpBase`.

This lets FlyDSL's MFMA backward kernel (`mslk.attention.flydsl.fmha_bwd_mfma`)
be exercised through the SAME `test/attention/fmha/test_backward.py` machinery
that judges CK's `ck.BwOp` — see `test/attention/fmha/test_backward_flydsl.py`.

Deliberately NOT added to `ALL_BW_OPS` in `mslk/attention/fmha/__init__.py`:
this is opt-in coverage, not a general-purpose backend.

Stride-aware addressing (sequencing-plan item 7): the kernel supports Q/K/V/dO
with a possibly-non-contiguous per-tensor row pitch — e.g. a packed-`qkv`
tensor sliced/unbound into Q/K/V views (`test_backward.py`'s dominant
non-contiguous case). The kernel still requires each tensor to be flat *within*
a row (last dim D stride-1, H-axis stride exactly D — i.e. only the outer
B/M-or-N row pitch may differ from the contiguous `H*D`); this is checked in
`not_supported_reasons()` below. Real per-axis arbitrary strides (transposed H,
non-unit D stride) remain unsupported — no known test case exercises that.
"""

from typing import List, Optional, Set, Tuple

import torch

from .attn_bias import BlockDiagonalCausalMask, BlockDiagonalMask, LowerTriangularMask
from .common import AttentionBwOpBase, Context, Gradients, Inputs
from .utils.op_common import get_operator, register_operator


def _uniform_row_pitch_reason(
    name: str, t: torch.Tensor, allow_broadcast_heads: bool = False
) -> Optional[str]:
    """Validate the "flat within a row, possibly-non-contiguous row pitch"
    assumption the kernel's stride-aware addressing relies on (sequencing-plan
    item 7): last dim (D) must be stride-1, and the H-axis must be flat
    relative to D (stride(2) == D) — only the outer B/M(or N)-axis row pitch
    may exceed the contiguous H*D. Returns a reason string if unsupported,
    else None. A stride-0 (broadcast/expand) row pitch is explicitly rejected:
    it would alias every row onto row 0 under our `row_pos * stride` formula.

    allow_broadcast_heads (GQA, sequencing-plan item 4): key/value may ALSO
    have stride(-2) == 0 (a `.expand()`-broadcast H axis, e.g.
    `key[:, :, :1].expand(-1, -1, Hq, -1)` -- test_backward_gqa's MQA-via-
    broadcast case) -- this means the tensor's REAL KV-head count is 1, not
    shape[2]. Query never gets this exception (no broadcast-Q case exists).
    """
    D = t.shape[-1]
    if t.stride(-1) != 1:
        return f"{name}'s last dim (head_dim) must be stride-1"
    if allow_broadcast_heads and t.stride(-2) == 0:
        return None
    if t.stride(-2) != D:
        return f"{name}'s head axis must be flat relative to head_dim (stride(-2) == D)"
    if t.stride(-3) % D != 0:
        return f"{name}'s row pitch (stride(-3)) must be a multiple of head_dim"
    if t.stride(-3) == 0:
        return f"{name} has a broadcast (stride-0) row pitch, not supported"
    return None


def _num_kv_heads(key: torch.Tensor) -> int:
    """Real number of distinct KV heads (GQA, sequencing-plan item 4).

    `key.stride(2) == 0` means every logical head slot aliases the same
    underlying head (a `.expand()` broadcast, e.g. MQA-via-broadcast) -- the
    real count is 1, not `key.shape[2]`. Otherwise key is a genuinely
    distinctly-shaped `(B, N, Hkv, D)` tensor (contiguous, stride(2) == D).
    """
    return 1 if key.stride(2) == 0 else key.shape[2]


@torch.library.custom_op(
    "mslk_flydsl::fmha_bwd",
    mutates_args=(),
    device_types=["cuda"],
)
def _flydsl_bwd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    grad: torch.Tensor,
    scale: float,
    causal: bool,
    seqstart_q: Optional[torch.Tensor] = None,
    seqstart_k: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import flydsl.compiler as flyc

    from mslk.attention.flydsl.fmha_bwd_mfma import (
        compile_fmha_bwd_dq_mfma,
        compile_fmha_bwd_dvdk_mfma,
    )

    # Varlen (sequencing-plan item 6, Stage B1, non-causal BlockDiagonalMask
    # only): query/key/value arrive already reshaped to (1, sum_M_i, H, D) --
    # B_logical (the real batch count) is `seqstart_q.shape[0] - 1`, NOT
    # query.shape[0] (always 1 under group-mode). Non-varlen: B == query.shape[0].
    varlen = seqstart_q is not None
    H, D = query.shape[2], query.shape[3]
    total_m = query.shape[1]  # real physical M extent (== sum_M_i under varlen)
    total_n = key.shape[1]    # real physical N extent (== sum_N_i under varlen)
    if varlen:
        B = seqstart_q.shape[0] - 1
        M = int(seqstart_q.diff().max().item())  # max_seqlen_q -- grid/loop sizing only
        N = int(seqstart_k.diff().max().item())  # max_seqlen_k -- grid/loop sizing only
    else:
        B, M = query.shape[0], query.shape[1]
        N = total_n
    device = query.device
    dtype = query.dtype
    dtype_str = "bf16" if dtype == torch.bfloat16 else "f16"

    # GQA (sequencing-plan item 4): H_kv < H is either a genuine distinctly-shaped
    # (B,N,Hkv,D) tensor or a stride-0 `.expand()` broadcast (H_kv=1 in that case,
    # regardless of key.shape[2]) -- see _num_kv_heads. heads_per_kv is passed to
    # the kernel as a COMPILE-TIME constant (like `causal`), not a runtime arg.
    H_kv = _num_kv_heads(key)
    heads_per_kv = H // H_kv

    def _as_i16(t: torch.Tensor) -> torch.Tensor:
        return t.view(torch.int16)

    # Stride-aware addressing (sequencing-plan item 7): Q/K/V/dO may have a
    # non-contiguous row pitch (e.g. a packed-qkv unbind view, stride(1) > H*D)
    # as long as they're flat within a row (stride(-1)==1, stride(2)==D — this
    # is enforced for query/key/value by `not_supported_reasons` before we get
    # here). `grad` isn't visible to `not_supported_reasons` (it's not part of
    # `Inputs`), so validate it here -- this also catches the
    # `grad_out_contiguous=False` broadcast/expand case CK itself doesn't
    # support either (test_backward.py skips it for ckB).
    grad_reason = _uniform_row_pitch_reason("grad", grad)
    if grad_reason is not None:
        raise NotImplementedError(grad_reason)

    # Keep Q/K/V/dO 4D (never collapse (B,M,H,D) -> (B*M*H,D) via `.view()`,
    # which would raise on exactly the non-contiguous case we now want to
    # support) and pass the real row pitch (in row units, i.e. elem_stride //
    # D) as a runtime kernel arg instead.
    Q_4d = _as_i16(query)
    K_4d = _as_i16(key)
    V_4d = _as_i16(value)
    dO_4d = _as_i16(grad)
    q_stride_m = query.stride(1) // D
    kv_stride_n = key.stride(1) // D
    do_stride_m = grad.stride(1) // D

    # `lse` is [B, H, M] float32 (non-varlen) or packed [1, H, sum_M] under
    # varlen (CK's VARLEN_LSE_PACKED=True convention -- see fmha_bwd_mfma.py's
    # _lse_row). `out`/`grad` are BMHK-shaped with total_m as their real M
    # extent either way. These are plain elementwise/reduction ops producing
    # fresh contiguous tensors, so `.view()` on their results is always safe
    # regardless of the strides of `out`/`grad`/`lse` themselves.
    LSE_2d = lse.contiguous().view(H * total_m, 1)
    D_vec = (grad.float() * out.float()).sum(dim=-1).view(total_m * H, 1)

    # dV/dK are shaped [B, N, H_kv, D] under GQA (one gradient per KV head, summed
    # over its heads_per_kv group by the kernel's grid-regroup-by-KV-head -- see
    # fmha_bwd_mfma.py's compile_fmha_bwd_dvdk_mfma docstring); H_kv == H otherwise.
    # total_m/total_n (not B*M/B*N, which under varlen would use max_seqlen*B_logical
    # instead of the real packed extent) size the physical output allocation.
    dV_out = torch.zeros(total_n * H_kv * D, 1, device=device, dtype=torch.float32)
    dK_out = torch.zeros(total_n * H_kv * D, 1, device=device, dtype=torch.float32)
    dQ_out = torch.zeros(total_m * H * D, 1, device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()

    # Split kernels (dvdk + dq) — the faster, production path (see
    # A3_ck_flyDSL_compare.md), not the fully-fused dqdkdv variant.
    # dvdk's Q/dO/P/dS LDS footprint at BLOCK_M=128 is 68/100/164 KB for D=64/128/256
    # -- fits gfx950's 160KB LDS everywhere except D=256, but EXCEEDS gfx942's 64KB
    # LDS at every D we support (even D=64). Drop to BLOCK_M=64 (35/52/83 KB) on
    # gfx942, and on gfx950 only at D=256 — same mitigation CK's own codegen uses
    # (shrinks its M-tile at D>=128, see
    # external/composable_kernel/.../fmha_bwd.py get_dq_dk_dv_tiles()). gfx942 D=256
    # still overflows at BLOCK_M=64 (83KB > 64KB) -- verified via a real LLVM
    # "local memory exceeds limit" error on gfx942 hardware -- drop to BLOCK_M=32
    # there (42.5KB).
    gpu_arch = torch.cuda.get_device_properties(device).gcnArchName
    _is_gfx950 = "gfx950" in gpu_arch
    if _is_gfx950:
        BLOCK_M_DVDK = 64 if D >= 256 else 128
    else:
        BLOCK_M_DVDK = 32 if D >= 256 else 64
    BLOCK_N_DVDK = 64
    launch_dvdk = compile_fmha_bwd_dvdk_mfma(
        D=D,
        dtype_str=dtype_str,
        BLOCK_M=BLOCK_M_DVDK,
        BLOCK_N=BLOCK_N_DVDK,
        scale=scale,
        use_pipeline=True,
        gpu_arch=gpu_arch,
        causal=causal,
        heads_per_kv=heads_per_kv,
        varlen=varlen,
    )
    n_M_tiles = (M + BLOCK_M_DVDK - 1) // BLOCK_M_DVDK
    # Varlen (sequencing-plan item 6, Stage B1): pass the real seqstart tensors;
    # non-varlen passes dummies (unused since varlen=False at compile time --
    # see compile_fmha_bwd_dvdk_mfma's `varlen` kwarg).
    _dummy_seqstart = torch.zeros(1, device=device, dtype=torch.int32)
    _seqstart_q_arg = seqstart_q if varlen else _dummy_seqstart
    _seqstart_k_arg = seqstart_k if varlen else _dummy_seqstart
    args_dvdk = (
        Q_4d, K_4d, V_4d, dO_4d, dV_out, dK_out, LSE_2d, D_vec,
        B, M, N, H, n_M_tiles, q_stride_m, kv_stride_n, do_stride_m,
        _seqstart_q_arg, _seqstart_k_arg, total_m, stream,
    )
    compiled_dvdk = flyc.compile(launch_dvdk, *args_dvdk)
    compiled_dvdk(*args_dvdk)

    # dq's K/V/dS LDS footprint at BLOCK_N=64 is 24/40/72 KB for D=64/128/256 -- fits
    # gfx950's 160KB everywhere but EXCEEDS gfx942's 64KB at D=256 (72KB); drop to
    # BLOCK_N=32 (36KB) there, same style of mitigation as dvdk's BLOCK_M above.
    BLOCK_M_DQ = 64
    BLOCK_N_DQ = 32 if (not _is_gfx950 and D >= 256) else 64
    launch_dq = compile_fmha_bwd_dq_mfma(
        D=D,
        dtype_str=dtype_str,
        BLOCK_M=BLOCK_M_DQ,
        BLOCK_N=BLOCK_N_DQ,
        scale=scale,
        gpu_arch=gpu_arch,
        causal=causal,
        heads_per_kv=heads_per_kv,
        varlen=varlen,
    )
    n_N_tiles = (N + BLOCK_N_DQ - 1) // BLOCK_N_DQ
    args_dq = (
        Q_4d, K_4d, V_4d, dO_4d, dQ_out, LSE_2d, D_vec,
        B, M, N, H, n_N_tiles, q_stride_m, kv_stride_n, do_stride_m,
        _seqstart_q_arg, _seqstart_k_arg, total_m, stream,
    )
    compiled_dq = flyc.compile(launch_dq, *args_dq)
    compiled_dq(*args_dq)

    # Output shapes use total_m/total_n (real packed extent) as the M/N axis --
    # under varlen this is sum_M_i/sum_N_i with B collapsed to 1 (matching
    # Q/K/V's own physical shape); non-varlen: total_m==B*M, total_n==B*N.
    out_b = 1 if varlen else B
    out_m = total_m if varlen else M
    out_n = total_n if varlen else N
    dq = dQ_out.view(out_b, out_m, H, D).to(dtype)
    dk = dK_out.view(out_b, out_n, H_kv, D).to(dtype)
    dv = dV_out.view(out_b, out_n, H_kv, D).to(dtype)
    return dq, dk, dv


@torch.library.register_fake("mslk_flydsl::fmha_bwd")
def _flydsl_bwd_abstract(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    grad: torch.Tensor,
    scale: float,
    causal: bool,
    seqstart_q: Optional[torch.Tensor] = None,
    seqstart_k: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty_like(query),
        torch.empty_like(key),
        torch.empty_like(value),
    )


@register_operator
class BwOp(AttentionBwOpBase):
    """Opt-in backward op wrapping FlyDSL's MFMA dV/dK/dQ kernels (WP-A3).

    Not part of `ALL_BW_OPS` — see `test/attention/fmha/test_backward_flydsl.py`
    for how this is exercised against the shared `test_backward` machinery.
    """

    OPERATOR = get_operator("mslk_flydsl", "fmha_bwd")
    SUPPORTED_DEVICES: Set[str] = {"cuda"}
    SUPPORTED_DTYPES: Set[torch.dtype] = {torch.bfloat16, torch.float16}
    # Kernel wave-tiling requires D a multiple of 32 (sequencing-plan step 2: D=64/128/256
    # done; D=32/96 done via ceil-div wave assignment + out-of-range guards, see
    # fmha_bwd_mfma.py's D_SUBS_PER_WAVE comment).
    SUPPORTED_MAX_K = 256
    SUPPORTED_MIN_K = 32
    SUPPORTS_DROPOUT = False
    SUPPORTS_CUSTOM_SCALE = True
    SUPPORTS_DIFFERENT_VALUE_EMBED = False
    # GQA (sequencing-plan item 4) is supported via the plain 4D BMHK path
    # (key/value with Hkv < Hq, either a genuine (B,N,Hkv,D) tensor or a
    # stride-0 `.expand()` broadcast -- see _num_kv_heads/_uniform_row_pitch_
    # reason), matching how CK's own C++ op and test_backward_gqa use it.
    # SUPPORTS_BMGHK covers a DIFFERENT, unrelated case (true 5D BMGHK
    # tensors) that this kernel does not implement -- stays False.
    SUPPORTS_BMGHK = False
    IS_DETERMINISTIC = True  # split dvdk+dq path, no atomics
    VARLEN_LSE_PACKED = True  # matches ck.BwOp's convention (see fmha_bwd_mfma.py's _lse_row)
    # Simple top-left causal (single fixed-length sequence per batch), non-causal
    # `BlockDiagonalMask` varlen (Stage B1), AND per-block top-left causal
    # `BlockDiagonalCausalMask` varlen (Stage B2) -- sequencing-plan item 6, see
    # fmha_bwd_mfma.py's `varlen`/`causal` kwarg docstrings. The existing
    # `n_row_abs <= m_row_abs` causal term is already tile-relative to each
    # block's own batch-local n_start/m_start (never seqstart-global), which
    # turned out to already implement per-block causal masking correctly once
    # Stage B1's per-batch tiling landed -- verified via CASES_VARLEN_CAUSAL in
    # test_fmha_bwd_{dvdk,dq}_mfma.py before enabling this. NOT other causal
    # block-diagonal variants (e.g. `BlockDiagonalCausalFromBottomRightMask`,
    # different alignment semantics) -- separately scoped, not attempted.
    SUPPORTED_ATTN_BIAS_TYPES = (
        type(None), LowerTriangularMask, BlockDiagonalMask, BlockDiagonalCausalMask,
    )
    _TEST_K: List[int] = [32, 64, 96, 128, 256]
    NAME = "flydslB"

    @classmethod
    def not_supported_reasons(cls, d: Inputs) -> List[str]:
        reasons = super(BwOp, cls).not_supported_reasons(d)
        reason = _uniform_row_pitch_reason("query", d.query)
        if reason is not None:
            reasons.append(reason)
        # key/value get the GQA broadcast exception (allow_broadcast_heads) --
        # query never does (no broadcast-Q case exists).
        for name, t in (("key", d.key), ("value", d.value)):
            reason = _uniform_row_pitch_reason(name, t, allow_broadcast_heads=True)
            if reason is not None:
                reasons.append(reason)
        # K and V share a single kv_stride_n kernel arg -- reject if they differ.
        if d.key.stride(1) != d.value.stride(1):
            reasons.append("key and value must share the same row pitch (stride(1))")
        if d.query.shape[-1] % 32 != 0:
            reasons.append("head_dim must be a multiple of 32")
        # GQA (sequencing-plan item 4): Hq must be a whole multiple of Hkv --
        # the kernel's heads_per_kv = Hq // Hkv grid-regroup requires this.
        h_kv = _num_kv_heads(d.key)
        if d.query.shape[2] % h_kv != 0:
            reasons.append("query head count must be a multiple of the KV head count (GQA)")
        return reasons

    @classmethod
    def apply(cls, ctx: Context, inp: Inputs, grad: torch.Tensor) -> Gradients:
        # LowerTriangularMask (simple causal) and BlockDiagonalCausalMask (per-block
        # top-left causal, Stage B2) both map to the same causal mask math in this
        # kernel -- mirrors ck.py's _custom_mask_type grouping both under
        # CausalFromTopLeft. BlockDiagonalCausalMask subclasses BlockDiagonalMask,
        # so the isinstance check below (seqstart extraction) already covers it.
        causal = isinstance(inp.attn_bias, (LowerTriangularMask, BlockDiagonalCausalMask))
        seqstart_q = seqstart_k = None
        if isinstance(inp.attn_bias, BlockDiagonalMask):
            # Mirrors ck.py's _get_seqlen_info.
            seqstart_q = inp.attn_bias.q_seqinfo.seqstart.to(inp.query.device)
            seqstart_k = inp.attn_bias.k_seqinfo.seqstart.to(inp.query.device)
        dq, dk, dv = cls.OPERATOR(
            inp.query,
            inp.key,
            inp.value,
            ctx.out,
            ctx.lse,
            grad,
            inp.scale_float,
            causal,
            seqstart_q,
            seqstart_k,
        )
        return Gradients(dq=dq, dk=dk, dv=dv)
