# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# pyre-unsafe
"""FlyDSL FMHA backward, registered as an opt-in `AttentionBwOpBase`.

This lets FlyDSL's MFMA backward kernel (`mslk.attention.flydsl.fmha_bwd_mfma`)
be exercised through the SAME `test/attention/fmha/test_backward.py` machinery
that judges CK's `ck.BwOp` — see `test/attention/fmha/test_backward_flydsl.py`.

Part of `ALL_BW_OPS` in `mslk/attention/fmha/__init__.py` (guarded by
`torch.version.hip`) for TEST-ENUMERATION purposes only -- this broadens which
test files (`test_mem_eff_attention.py`, `test_forward.py`) exercise this op.
It is deliberately NOT wired into `dispatch.py`'s live `_dispatch_bw()`
priority list, which still hardcodes `[ck.BwOp]` on ROCm -- this op is not a
live production backend.

Stride-aware addressing: the kernel supports Q/K/V/dO with a possibly-non-
contiguous per-tensor row pitch — e.g. a packed-`qkv` tensor sliced/unbound
into Q/K/V views. The kernel still requires each tensor to be flat *within* a
row (last dim D stride-1, H-axis stride exactly D — i.e. only the outer
B/M-or-N row pitch may differ from the contiguous `H*D`); this is checked in
`not_supported_reasons()` below. Real per-axis arbitrary strides (transposed H,
non-unit D stride) remain unsupported.

Non-goals (intentionally excluded rather than left as silent gaps -- each
entry below is enforced generically, see the cited mechanism, and has its own
negative-path test in `test_backward_flydsl.py`):
- Dropout (`d.p != 0.0`): no MSLK caller passes dropout through this op;
  rejected generically by `AttentionOpBase.not_supported_reasons()`'s
  `(d.p != 0.0) and not cls.SUPPORTS_DROPOUT` check (`SUPPORTS_DROPOUT = False`
  below). Revisit if `flydsl.BwOp` is ever added to a live dispatch path and a
  caller trips this.
- True 5D BMGHK (`d.query.ndim == 5`, multi-query-group layout distinct from
  GQA-via-4D-broadcast, which IS supported): no MSLK caller found; rejected
  generically by the same `not_supported_reasons()`'s
  `not cls.SUPPORTS_BMGHK and d.query.ndim == 5` check (`SUPPORTS_BMGHK = False`
  below). Revisit under the same condition as dropout.
- Paged-KV / gappy-keys backward (`PagedBlockDiagonal*Mask`,
  `BlockDiagonal*GappyKeys*Mask`): these are inference-serving-only shapes in
  MSLK (`tree_attention.py`'s forward-only code path never reaches
  `flydsl.BwOp`); no backward caller found. Rejected generically via
  `SUPPORTED_ATTN_BIAS_TYPES` not listing them (falls through to the base
  class's `type(d.attn_bias) not in cls.SUPPORTED_ATTN_BIAS_TYPES` check).
  Revisit if a backward caller for paged/gappy attention appears.
- Tensor-bias / ALiBi (`LowerTriangularMaskWithTensorBias` and similar): no
  MSLK caller found for this backward op. Rejected the same way, via
  `SUPPORTED_ATTN_BIAS_TYPES` omission.
- Bottom-right / local-window varlen (`BlockDiagonalCausalFromBottomRightMask`
  and other non-top-left alignment variants): only top-left causal alignment
  (`LowerTriangularMask`, `BlockDiagonalCausalMask`) is implemented;
  bottom-right/local-window semantics are separately scoped and not
  implemented. Rejected the same way, via `SUPPORTED_ATTN_BIAS_TYPES`
  omission. Revisit if a caller needs bottom-right or local-window varlen
  causal masking specifically.
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
    assumption the kernel's stride-aware addressing relies on: last dim (D)
    must be stride-1, and the H-axis must be flat relative to D
    (stride(2) == D) — only the outer B/M(or N)-axis row pitch may exceed the
    contiguous H*D. Returns a reason string if unsupported, else None. A
    stride-0 (broadcast/expand) row pitch is explicitly rejected: it would
    alias every row onto row 0 under the `row_pos * stride` formula.

    allow_broadcast_heads (GQA): key/value may ALSO have stride(-2) == 0 (a
    `.expand()`-broadcast H axis, e.g. `key[:, :, :1].expand(-1, -1, Hq, -1)`)
    -- this means the tensor's REAL KV-head count is 1, not shape[2]. Query
    never gets this exception (no broadcast-Q case exists).
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
    """Real number of distinct KV heads (GQA).

    `key.stride(2) == 0` means every logical head slot aliases the same
    underlying head (a `.expand()` broadcast, e.g. MQA-via-broadcast) -- the
    real count is 1, not `key.shape[2]`. Otherwise key is a genuinely
    distinctly-shaped `(B, N, Hkv, D)` tensor (contiguous, stride(2) == D).
    """
    return 1 if key.stride(2) == 0 else key.shape[2]


# Cache of flyc.compile()'d kernels, keyed on the compile-time-constant params
# baked into the kernel body (D, dtype, tile sizes, scale, causal, GQA ratio,
# varlen). `flyc.compile()`'s underlying MLIR/LLVM artifact is itself cached
# (FlyDSL's own on-disk+memory cache, keyed by kernel source + compile-time
# closure values), but `flyc.compile()` ITSELF re-traces/re-binds the Python
# call on every invocation, dominated by `JitFunction.__call__`'s signature
# binding, not by GPU work -- see `CompiledFunction`'s docstring in FlyDSL:
# the whole point of caching the returned `CompiledFunction` object ourselves
# is to skip straight to its cheap `__call__` hot path instead of re-entering
# `flyc.compile()` every backward call. B/M/N/H/n_tiles/data-pointers are all
# runtime `fx.Int32`/tensor args in the kernel signature (not baked at compile
# time), so ONE cached `CompiledFunction` per key correctly serves any
# shape/batch at that (D, dtype, tile, scale, causal, heads_per_kv, varlen,
# arch) combination.
_dvdk_kernel_cache: dict = {}
_dq_kernel_cache: dict = {}


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

    # Varlen (non-causal or causal BlockDiagonalMask): query/key/value arrive
    # already reshaped to (1, sum_M_i, H, D) -- B_logical (the real batch
    # count) is `seqstart_q.shape[0] - 1`, NOT query.shape[0] (always 1 under
    # group-mode). Non-varlen: B == query.shape[0].
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

    # GQA: H_kv < H is either a genuine distinctly-shaped (B,N,Hkv,D) tensor
    # or a stride-0 `.expand()` broadcast (H_kv=1 in that case, regardless of
    # key.shape[2]) -- see _num_kv_heads. heads_per_kv is passed to the
    # kernel as a COMPILE-TIME constant (like `causal`), not a runtime arg.
    H_kv = _num_kv_heads(key)
    heads_per_kv = H // H_kv

    def _as_i16(t: torch.Tensor) -> torch.Tensor:
        return t.view(torch.int16)

    # Stride-aware addressing: Q/K/V/dO may have a non-contiguous row pitch
    # (e.g. a packed-qkv unbind view, stride(1) > H*D) as long as they're
    # flat within a row (stride(-1)==1, stride(2)==D — this is enforced for
    # query/key/value by `not_supported_reasons` before we get here). `grad`
    # isn't visible to `not_supported_reasons` (it's not part of `Inputs`),
    # so validate it here -- this also catches the broadcast/expand dO case
    # CK itself doesn't support either.
    grad_reason = _uniform_row_pitch_reason("grad", grad)
    if grad_reason is not None:
        raise NotImplementedError(grad_reason)

    # Keep Q/K/V/dO 4D (never collapse (B,M,H,D) -> (B*M*H,D) via `.view()`,
    # which would raise on the non-contiguous case) and pass the real row
    # pitch (in row units, i.e. elem_stride // D) as a runtime kernel arg
    # instead.
    Q_4d = _as_i16(query)
    K_4d = _as_i16(key)
    V_4d = _as_i16(value)
    dO_4d = _as_i16(grad)
    q_stride_m = query.stride(1) // D
    kv_stride_n = key.stride(1) // D
    do_stride_m = grad.stride(1) // D

    # `lse` is [B, H, M] float32 (non-varlen, B*H*M elements) or packed
    # [1, H, sum_M] under varlen (CK's VARLEN_LSE_PACKED=True convention, H*
    # total_m elements -- see fmha_bwd_mfma.py's _lse_row). `out`/`grad` are
    # [B, M, H, D] (non-varlen) or [1, total_m, H, D] (varlen), so their
    # per-row-summed D_vec has the same B*H*M vs H*total_m element count
    # split. Non-varlen's `total_m` is already per-batch M (== query.shape[1]),
    # NOT B*M, so the varlen-only element count (H*total_m) would silently
    # undercount by a factor of B here -- must branch on `varlen` explicitly.
    # These are plain elementwise/reduction ops producing fresh contiguous
    # tensors, so `.view()` on their results is always safe regardless of the
    # strides of `out`/`grad`/`lse` themselves.
    _lse_dvec_rows = H * total_m if varlen else B * H * total_m
    LSE_2d = lse.contiguous().view(_lse_dvec_rows, 1)
    D_vec = (grad.float() * out.float()).sum(dim=-1).view(_lse_dvec_rows, 1)

    # dV/dK are shaped [B, N, H_kv, D] under GQA (one gradient per KV head, summed
    # over its heads_per_kv group by the kernel's grid-regroup-by-KV-head -- see
    # fmha_bwd_mfma.py's compile_fmha_bwd_dvdk_mfma docstring); H_kv == H otherwise.
    # Physical output-buffer element count: under varlen, `total_m`/`total_n` are
    # already the real packed extent (B collapsed to 1 in the tensor's own
    # shape); under non-varlen, `total_m`/`total_n` are `query.shape[1]`/
    # `key.shape[1]` -- i.e. PER-BATCH M/N, not the B*M/B*N physical element
    # count -- so B must be multiplied in explicitly here (same trap as
    # `_lse_dvec_rows` above).
    alloc_m = total_m if varlen else B * total_m
    alloc_n = total_n if varlen else B * total_n
    dV_out = torch.zeros(alloc_n * H_kv * D, 1, device=device, dtype=torch.float32)
    dK_out = torch.zeros(alloc_n * H_kv * D, 1, device=device, dtype=torch.float32)
    dQ_out = torch.zeros(alloc_m * H * D, 1, device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()

    # Split kernels (dvdk + dq) — the faster, production path (see
    # A3_ck_flyDSL_compare.md), not the fully-fused dqdkdv variant.
    # dvdk's Q/dO/P/dS LDS footprint at BLOCK_M=128 is 68/100/164 KB for D=64/128/256
    # -- fits gfx950's 160KB LDS everywhere except D=256, but EXCEEDS gfx942's 64KB
    # LDS at every D we support (even D=64). Drop to BLOCK_M=64 (35/52/83 KB) on
    # gfx942, and on gfx950 only at D=256 — same mitigation CK's own codegen uses
    # (shrinks its M-tile at D>=128, see
    # external/composable_kernel/.../fmha_bwd.py get_dq_dk_dv_tiles()). gfx942 D=256
    # still overflows at BLOCK_M=64 (83KB > 64KB) -- confirmed via a real LLVM
    # "local memory exceeds limit" error on gfx942 hardware -- drop to BLOCK_M=32
    # there (42.5KB).
    gpu_arch = torch.cuda.get_device_properties(device).gcnArchName
    _is_gfx950 = "gfx950" in gpu_arch
    if _is_gfx950:
        BLOCK_M_DVDK = 64 if D >= 256 else 128
    else:
        BLOCK_M_DVDK = 32 if D >= 256 else 64
    BLOCK_N_DVDK = 64
    n_M_tiles = (M + BLOCK_M_DVDK - 1) // BLOCK_M_DVDK
    # Varlen: pass the real seqstart tensors; non-varlen passes dummies
    # (unused since varlen=False at compile time -- see
    # compile_fmha_bwd_dvdk_mfma's `varlen` kwarg).
    _dummy_seqstart = torch.zeros(1, device=device, dtype=torch.int32)
    _seqstart_q_arg = seqstart_q if varlen else _dummy_seqstart
    _seqstart_k_arg = seqstart_k if varlen else _dummy_seqstart
    args_dvdk = (
        Q_4d, K_4d, V_4d, dO_4d, dV_out, dK_out, LSE_2d, D_vec,
        B, M, N, H, n_M_tiles, q_stride_m, kv_stride_n, do_stride_m,
        _seqstart_q_arg, _seqstart_k_arg, total_m, stream,
    )
    # Cache the CompiledFunction across calls (see _dvdk_kernel_cache's
    # docstring above): none of these key fields are runtime call args, they
    # are all baked into the kernel body at compile_fmha_bwd_dvdk_mfma() time,
    # so one CompiledFunction correctly serves every B/M/N/H shape at this key.
    dvdk_key = (D, dtype_str, BLOCK_M_DVDK, BLOCK_N_DVDK, scale, gpu_arch,
                causal, heads_per_kv, varlen)
    compiled_dvdk = _dvdk_kernel_cache.get(dvdk_key)
    if compiled_dvdk is None:
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
        compiled_dvdk = flyc.compile(launch_dvdk, *args_dvdk)
        _dvdk_kernel_cache[dvdk_key] = compiled_dvdk
    compiled_dvdk(*args_dvdk)

    # dq's K/V/dS LDS footprint at BLOCK_N=64 is 24/40/72 KB for D=64/128/256 -- fits
    # gfx950's 160KB everywhere but EXCEEDS gfx942's 64KB at D=256 (72KB); drop to
    # BLOCK_N=32 (36KB) there, same style of mitigation as dvdk's BLOCK_M above.
    BLOCK_M_DQ = 64
    BLOCK_N_DQ = 32 if (not _is_gfx950 and D >= 256) else 64
    n_N_tiles = (N + BLOCK_N_DQ - 1) // BLOCK_N_DQ
    args_dq = (
        Q_4d, K_4d, V_4d, dO_4d, dQ_out, LSE_2d, D_vec,
        B, M, N, H, n_N_tiles, q_stride_m, kv_stride_n, do_stride_m,
        _seqstart_q_arg, _seqstart_k_arg, total_m, stream,
    )
    dq_key = (D, dtype_str, BLOCK_M_DQ, BLOCK_N_DQ, scale, gpu_arch,
              causal, heads_per_kv, varlen)
    compiled_dq = _dq_kernel_cache.get(dq_key)
    if compiled_dq is None:
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
        compiled_dq = flyc.compile(launch_dq, *args_dq)
        _dq_kernel_cache[dq_key] = compiled_dq
    compiled_dq(*args_dq)

    # Output shapes: under varlen, B collapses to 1 and total_m/total_n are
    # already the real packed sum_M_i/sum_N_i (matching Q/K/V's own physical
    # shape); under non-varlen, the real batch axis is B and the per-batch
    # extent is M/N (== total_m/total_n, which is per-batch here, not B*M/B*N
    # -- see alloc_m/alloc_n above for why that distinction matters).
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
    """Opt-in backward op wrapping FlyDSL's MFMA dV/dK/dQ kernels.

    Part of `ALL_BW_OPS` (test-enumeration only, ROCm-only) — see
    `test/attention/fmha/test_backward_flydsl.py` for the dedicated matrix, and
    the module docstring above for why this isn't in live dispatch.
    """

    OPERATOR = get_operator("mslk_flydsl", "fmha_bwd")
    SUPPORTED_DEVICES: Set[str] = {"cuda"}
    SUPPORTED_DTYPES: Set[torch.dtype] = {torch.bfloat16, torch.float16}
    # Kernel wave-tiling requires D a multiple of 32 (D=32/96 done via
    # ceil-div wave assignment + out-of-range guards, see fmha_bwd_mfma.py's
    # D_SUBS_PER_WAVE comment).
    SUPPORTED_MAX_K = 256
    SUPPORTED_MIN_K = 32
    SUPPORTS_DROPOUT = False
    SUPPORTS_CUSTOM_SCALE = True
    SUPPORTS_DIFFERENT_VALUE_EMBED = False
    # GQA is supported via the plain 4D BMHK path (key/value with Hkv < Hq,
    # either a genuine (B,N,Hkv,D) tensor or a stride-0 `.expand()` broadcast
    # -- see _num_kv_heads/_uniform_row_pitch_reason), matching how CK's own
    # C++ op and test_backward_gqa use it. SUPPORTS_BMGHK covers a DIFFERENT,
    # unrelated case (true 5D BMGHK tensors) that this kernel does not
    # implement -- stays False.
    SUPPORTS_BMGHK = False
    IS_DETERMINISTIC = True  # split dvdk+dq path, no atomics
    VARLEN_LSE_PACKED = True  # matches ck.BwOp's convention (see fmha_bwd_mfma.py's _lse_row)
    # Simple top-left causal (single fixed-length sequence per batch),
    # non-causal `BlockDiagonalMask` varlen, AND per-block top-left causal
    # `BlockDiagonalCausalMask` varlen -- see fmha_bwd_mfma.py's
    # `varlen`/`causal` kwarg docstrings. The `n_row_abs <= m_row_abs` causal
    # term is tile-relative to each block's own batch-local n_start/m_start
    # (never seqstart-global), which already implements per-block causal
    # masking correctly given per-batch tiling -- confirmed via
    # CASES_VARLEN_CAUSAL in test_fmha_bwd_{dvdk,dq}_mfma.py. NOT other
    # causal block-diagonal variants (e.g.
    # `BlockDiagonalCausalFromBottomRightMask`, different alignment
    # semantics) -- separately scoped, not implemented.
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
        # GQA: Hq must be a whole multiple of Hkv -- the kernel's
        # heads_per_kv = Hq // Hkv grid-regroup requires this.
        h_kv = _num_kv_heads(d.key)
        if d.query.shape[2] % h_kv != 0:
            reasons.append("query head count must be a multiple of the KV head count (GQA)")
        return reasons

    @classmethod
    def apply(cls, ctx: Context, inp: Inputs, grad: torch.Tensor) -> Gradients:
        # LowerTriangularMask (simple causal) and BlockDiagonalCausalMask
        # (per-block top-left causal) both map to the same causal mask math
        # in this kernel -- mirrors ck.py's _custom_mask_type grouping both
        # under CausalFromTopLeft. BlockDiagonalCausalMask subclasses
        # BlockDiagonalMask, so the isinstance check below (seqstart
        # extraction) already covers it.
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
        # GQA-via-broadcast (`key`/`value` genuinely Hkv-headed, exposed to
        # the caller as an H-headed stride-0 `.expand()` view -- see
        # _num_kv_heads): the kernel reduces the per-KV-head gradient
        # internally (grid regrouped by KV-head, no atomics -- see the class
        # docstring's GQA note / fmha_bwd_mfma.py's
        # compile_fmha_bwd_dvdk_mfma docstring), so `dk`/`dv` come back
        # Hkv-shaped here, smaller than `inp.key.shape`.
        # `_memory_efficient_attention_backward`
        # (mslk/attention/fmha/__init__.py) unconditionally reshapes every
        # op's returned dk/dv to the ORIGINAL (broadcast, H-shaped)
        # `inp.key.shape`/`inp.value.shape` -- the same contract every other
        # op follows (e.g. flash.py's BwOp always returns dk/dv pre-reshaped
        # to `inp.key.shape`/`inp.value.shape`). Broadcast back up via
        # `.expand()` (stride-0, no extra memory) to satisfy that contract,
        # WITHOUT changing what the kernel itself computes.
        #
        # Dividing by `heads_per_kv` before expanding is required: under the
        # autograd API, PyTorch's own `ExpandBackward` sums the H broadcast
        # copies when reducing back down to the real small-Hkv leaf tensor --
        # returning the already-reduced value undivided would get summed
        # again, overcounting by exactly `heads_per_kv`. Note `ck.BwOp`'s C++
        # op detects GQA via `key.size(2)` (unaffected by `.expand()`'s
        # stride-0 trick), so it never recognizes this broadcast as GQA and
        # instead computes H independent per-head gradients directly at full
        # H shape (mathematically distinct values per head, since each head
        # has a different Q) -- summed over heads, those equal this op's
        # single reduced value times `heads_per_kv`; both compute the same
        # true gradient, only observable after the same `ExpandBackward`
        # reduction. Called through the non-autograd
        # `memory_efficient_attention_backward` API directly (no autograd
        # graph, no `ExpandBackward` reduction), the two ops' raw returned
        # tensors are therefore NOT directly comparable element-for-element
        # -- this op's contract is "correct after an `ExpandBackward`
        # reduction", matching every other op's broadcast-GQA contract.
        if dk.shape[2] != inp.key.shape[2]:
            heads_per_kv = inp.key.shape[2] // dk.shape[2]
            dk = (dk / heads_per_kv).expand(inp.key.shape)
            dv = (dv / heads_per_kv).expand(inp.value.shape)
        return Gradients(dq=dq, dk=dk, dv=dv)
