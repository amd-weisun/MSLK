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

Zero-copy constraint: the FlyDSL kernel hardcodes flat row-major addressing
(`row = b*(M*H) + m*H + h`, see `fmha_bwd_mfma.py`) and is NOT stride-aware
like CK's C++ op. We use `.view()` only — never `.contiguous()`/`.reshape()` —
so a non-contiguous BMHK input raises rather than silently triggering a copy
kernel (HBM round-trip). `not_supported_reasons()` below checks contiguity
upfront so the test harness skips those cases instead of crashing. Giving the
kernel the same stride-awareness as CK's C++ op is tracked separately
(sequencing-plan item 7) and out of scope here.
"""

from typing import List, Set, Tuple

import torch

from .attn_bias import LowerTriangularMask
from .common import AttentionBwOpBase, Context, Gradients, Inputs
from .utils.op_common import get_operator, register_operator


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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import flydsl.compiler as flyc

    from mslk.attention.flydsl.fmha_bwd_mfma import (
        compile_fmha_bwd_dq_mfma,
        compile_fmha_bwd_dvdk_mfma,
    )

    B, M, H, D = query.shape
    N = key.shape[1]
    device = query.device
    dtype = query.dtype
    dtype_str = "bf16" if dtype == torch.bfloat16 else "f16"

    def _as_i16(t: torch.Tensor) -> torch.Tensor:
        return t.view(torch.int16)

    # `.view()` only (never `.contiguous()`/`.reshape()`) — raises if the
    # input isn't already contiguous in BMHK order, by design (see module
    # docstring). `not_supported_reasons` skips those cases before we get here.
    Q_2d = _as_i16(query).view(B * M * H, D)
    K_2d = _as_i16(key).view(B * N * H, D)
    V_2d = _as_i16(value).view(B * N * H, D)
    dO_2d = _as_i16(grad).view(B * M * H, D)

    # `lse` is [B, H, M] float32 (CK's FwOp convention); `out`/`grad` are BMHK.
    # These are plain elementwise/reduction ops producing fresh contiguous
    # tensors, so `.view()` on their results is always safe regardless of the
    # strides of `out`/`grad`/`lse` themselves.
    LSE_2d = lse.contiguous().view(B * H * M, 1)
    D_vec = (grad.float() * out.float()).sum(dim=-1).view(B * M * H, 1)

    dV_out = torch.zeros(B * N * H * D, 1, device=device, dtype=torch.float32)
    dK_out = torch.zeros(B * N * H * D, 1, device=device, dtype=torch.float32)
    dQ_out = torch.zeros(B * M * H * D, 1, device=device, dtype=torch.float32)

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
    )
    n_M_tiles = (M + BLOCK_M_DVDK - 1) // BLOCK_M_DVDK
    args_dvdk = (
        Q_2d, K_2d, V_2d, dO_2d, dV_out, dK_out, LSE_2d, D_vec,
        B, M, N, H, n_M_tiles, stream,
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
    )
    n_N_tiles = (N + BLOCK_N_DQ - 1) // BLOCK_N_DQ
    args_dq = (
        Q_2d, K_2d, V_2d, dO_2d, dQ_out, LSE_2d, D_vec,
        B, M, N, H, n_N_tiles, stream,
    )
    compiled_dq = flyc.compile(launch_dq, *args_dq)
    compiled_dq(*args_dq)

    dq = dQ_out.view(B, M, H, D).to(dtype)
    dk = dK_out.view(B, N, H, D).to(dtype)
    dv = dV_out.view(B, N, H, D).to(dtype)
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
    SUPPORTS_BMGHK = False  # no GQA yet (sequencing-plan item 4)
    IS_DETERMINISTIC = True  # split dvdk+dq path, no atomics
    # Non-causal + simple top-left causal (single fixed-length sequence per batch)
    # only so far (sequencing-plan item 3). NOT `BlockDiagonalCausalMask` or other
    # block-diagonal/varlen variants: those pack multiple variable-length sequences
    # per batch row with PER-BLOCK causal masking, which this kernel's flat `n<=m`
    # comparison does not implement (confirmed via a real correctness failure —
    # garbage-magnitude gradients, not the usual bf16 precision tail). Real varlen
    # support is sequencing-plan item 6, separately scoped.
    SUPPORTED_ATTN_BIAS_TYPES = (type(None), LowerTriangularMask)
    _TEST_K: List[int] = [32, 64, 96, 128, 256]
    NAME = "flydslB"

    @classmethod
    def not_supported_reasons(cls, d: Inputs) -> List[str]:
        reasons = super(BwOp, cls).not_supported_reasons(d)
        for name, t in (("query", d.query), ("key", d.key), ("value", d.value)):
            if not t.is_contiguous():
                reasons.append(
                    f"{name} is not contiguous in BMHK order — FlyDSL kernel "
                    "addressing is not stride-aware yet (sequencing-plan item 7)"
                )
        if d.query.shape[-1] % 32 != 0:
            reasons.append("head_dim must be a multiple of 32")
        return reasons

    @classmethod
    def apply(cls, ctx: Context, inp: Inputs, grad: torch.Tensor) -> Gradients:
        causal = isinstance(inp.attn_bias, LowerTriangularMask)
        dq, dk, dv = cls.OPERATOR(
            inp.query,
            inp.key,
            inp.value,
            ctx.out,
            ctx.lse,
            grad,
            inp.scale_float,
            causal,
        )
        return Gradients(dq=dq, dk=dk, dv=dv)
