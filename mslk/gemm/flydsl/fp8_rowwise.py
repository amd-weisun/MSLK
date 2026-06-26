"""FlyDSL parallel implementation of the FP8 rowwise GEMM.

Registers ``mslk::f8f8bf16_rowwise_flydsl`` as the ROCm ("CUDA" key) implementation
via ``torch.library.impl``.  This module is imported from ``mslk.gemm.__init__`` only
on ROCm builds (guarded by ``torch.version.hip``).

Development convention (WP-G1):
  - CK remains the default impl of ``f8f8bf16_rowwise`` — untouched.
  - This sibling op is the FlyDSL parallel path, used to diff against CK.
  - Once FlyDSL meets/beats CK on all shapes, point the original op here and
    retire this sibling (schema is identical, callers need no change).

Target: gfx950 (MI350X), CDNA4 MFMA.  Not tested on gfx942.
FlyDSL kernel: kernels/fp8_gemm_8wave.py -- compile_fp8_gemm_8w.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# FlyDSL imports — only available in the container (pip-installed editable).
# ---------------------------------------------------------------------------
try:
    import flydsl.compiler as flyc
    from kernels.fp8_gemm_utils import preshuffle_b

    _FLYDSL_AVAILABLE = True
except ImportError:
    _FLYDSL_AVAILABLE = False


def _as_i8(t: torch.Tensor) -> torch.Tensor:
    """View an fp8 tensor as int8 (required by FlyDSL buffer ops)."""
    return t.view(torch.int8) if "float8" in str(t.dtype) else t


@lru_cache(maxsize=64)
def _get_compiled_kernel(K: int, BLOCK_M: int, BLOCK_N: int):
    """Compile and cache the FlyDSL launch function for a given (K, BLOCK_M, BLOCK_N)."""
    # Import here so the module-level import guard keeps things safe on non-ROCm builds.
    from kernels.fp8_gemm_8wave import compile_fp8_gemm_8w

    launch_fn = compile_fp8_gemm_8w(K=K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, b_preshuffled=False)
    return launch_fn


def _pick_tile(M: int, N: int) -> tuple[int, int]:
    """Pick BLOCK_M / BLOCK_N tile sizes for a given problem shape.

    Constraints from compile_fp8_gemm_8w:
      BLOCK_M >= 128, BLOCK_M % 128 == 0
      BLOCK_N >= 256, BLOCK_N % 256 == 0
    Start with the default (256, 256); add more heuristics as we tune.
    """
    return 256, 256


def f8f8bf16_rowwise_flydsl_impl(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    use_fast_accum: bool = True,
) -> torch.Tensor:
    """FlyDSL fp8 rowwise GEMM: XQ [M,K] x WQ [N,K]^T -> bf16 [M,N].

    Args:
        XQ: fp8 e4m3 activations [M, K], contiguous.
        WQ: fp8 e4m3 weights [N, K], contiguous (B_T orientation, i.e. already transposed).
        x_scale: per-token (per-row) scale [M], fp32.
        w_scale: per-channel (per-column) scale [N], fp32.
        bias: optional [N] bf16/fp32 bias — not yet implemented, raises if provided.
        use_fast_accum: ignored in v0 (FlyDSL kernel uses f32 accumulation).

    Returns:
        bf16 output tensor [M, N].
    """
    if not _FLYDSL_AVAILABLE:
        raise RuntimeError(
            "FlyDSL is not installed. Install from /workspace/FlyDSL inside the container."
        )
    if bias is not None:
        raise NotImplementedError("bias is not yet supported in the FlyDSL rowwise path (v0).")

    M, K = XQ.shape
    N = WQ.shape[0]
    assert WQ.shape[1] == K, f"WQ shape {WQ.shape} incompatible with XQ K={K}"

    BLOCK_M, BLOCK_N = _pick_tile(M, N)
    launch_fn = _get_compiled_kernel(K=K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)

    # Allocate output.
    C = torch.empty((M, N), dtype=torch.bfloat16, device=XQ.device)

    # Flatten tensors to 1-D and view fp8 as int8 (FlyDSL buffer convention).
    a_flat = _as_i8(XQ).contiguous().view(-1)
    b_flat = _as_i8(WQ).contiguous().view(-1)
    c_flat = C.view(-1)
    sa_flat = x_scale.contiguous().view(-1)
    sb_flat = w_scale.contiguous().view(-1)

    # Compile on first call for this arg signature; subsequent calls use the JIT cache.
    compiled = flyc.compile(launch_fn, a_flat, b_flat, c_flat, sa_flat, sb_flat, M, N,
                            torch.cuda.current_stream())
    compiled(a_flat, b_flat, c_flat, sa_flat, sb_flat, M, N, torch.cuda.current_stream())

    return C


# ---------------------------------------------------------------------------
# Torch library registration — mirrors the Triton precedent in mx8mx4_gemm.py.
# ---------------------------------------------------------------------------
if torch.version.hip is not None and hasattr(torch.ops, "mslk"):
    if hasattr(torch.ops.mslk, "f8f8bf16_rowwise_flydsl"):

        @torch.library.impl("mslk::f8f8bf16_rowwise_flydsl", "CUDA")
        def _f8f8bf16_rowwise_flydsl_rocm(
            XQ: torch.Tensor,
            WQ: torch.Tensor,
            x_scale: torch.Tensor,
            w_scale: torch.Tensor,
            bias: Optional[torch.Tensor] = None,
            use_fast_accum: bool = True,
        ) -> torch.Tensor:
            return f8f8bf16_rowwise_flydsl_impl(XQ, WQ, x_scale, w_scale, bias, use_fast_accum)
