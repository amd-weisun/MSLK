"""FlyDSL vs CK backward comparison harness, through MSLK's real API.

Runs, on matched shapes, both `fmha.ck.BwOp` and `fmha.flydsl.BwOp` via MSLK's own
non-autograd direct-call API:
    out, lse = fmha.memory_efficient_attention_forward_requires_grad(..., op=fmha.ck.FwOp)
    dq, dk, dv = fmha.memory_efficient_attention_backward(..., op=op_bw)

Forward is always CK's (`ck.FwOp`) for both rows: `flydsl.BwOp` has no forward kernel of
its own and must be paired with CK's forward everywhere else in the codebase (see
flydsl.py's module docstring, test_backward_flydsl.py) -- this keeps the forward pass
identical for both backward candidates being timed, so only the backward op differs.

Comparison policy: both ops are timed through the identical MSLK entrypoint
(`fmha.memory_efficient_attention_backward`), not CK's standalone C++ benchmark binary,
so the comparison includes the same dispatch/shape-normalization overhead a real caller
of either op actually pays. See A3_ck_flyDSL_compare.md for a PMC/VALU-level root-cause
comparison of the two kernels in isolation.

Correctness is NOT re-checked here: `test_backward_flydsl.py`'s case matrix already covers
accuracy for `flydsl.BwOp` exhaustively. This script is pure timing.

Usage (inside container):
    cd /workspace/MSLK
    HIP_VISIBLE_DEVICES=3 FLYDSL_RUNTIME_ENABLE_CACHE=0 PYTHONPATH=/workspace/FlyDSL:$PYTHONPATH \
    python test/attention/fmha/bench_fmha_bwd_vs_ck.py
"""
import torch
from flydsl.autotune import do_bench

from mslk.attention import fmha
from mslk.attention.fmha import flydsl  # noqa: F401 -- binds fmha.flydsl attribute


def _flop_count(B, Mq, Mkv, H, Dqk, Dv, causal):
    """Standard FMHA backward analytic FLOP count -- matches CK's own internal
    formula (external/composable_kernel/example/ck_tile/01_fmha/fmha_bwd_runner.hpp):
    QK^T + dS^T@Q^T + dS@K^T ("3*2" GEMM group, contracted over Dqk) plus
    dO@V^T + P^T@dO^T ("2*2" GEMM group, contracted over Dv). Independent of
    either op's own self-reporting so both sides use an identical definition.
    Halved for causal shapes (upper-triangular masks ~halve non-zero work).
    """
    flop = B * H * (3 * 2 * Mq * Mkv * Dqk + 2 * 2 * Mq * Mkv * Dv)
    if causal:
        flop //= 2
    return flop


def run_op(op_fw, op_bw, B, Mq, Mkv, H, D, dtype, causal, varlen_seqlens, device):
    """Return (ms, flop) for op_bw's backward pass, through MSLK's real API.

    varlen_seqlens, if given, is (seqlens_q, seqlens_k) and builds a packed
    BlockDiagonalMask input (B collapses to 1, matching its real tensor layout).
    """
    if varlen_seqlens is not None:
        seqlens_q, seqlens_k = varlen_seqlens
        B_logical = len(seqlens_q)
        attn_bias = fmha.attn_bias.BlockDiagonalMask.from_seqlens(seqlens_q, seqlens_k)
        total_m, total_n = sum(seqlens_q), sum(seqlens_k)
        query = torch.randn(1, total_m, H, D, device=device, dtype=dtype)
        key = torch.randn(1, total_n, H, D, device=device, dtype=dtype)
        value = torch.randn(1, total_n, H, D, device=device, dtype=dtype)
        Mq_flop, Mkv_flop, B_flop = max(seqlens_q), max(seqlens_k), B_logical
    else:
        attn_bias = fmha.attn_bias.LowerTriangularMask() if causal else None
        query = torch.randn(B, Mq, H, D, device=device, dtype=dtype)
        key = torch.randn(B, Mkv, H, D, device=device, dtype=dtype)
        value = torch.randn(B, Mkv, H, D, device=device, dtype=dtype)
        Mq_flop, Mkv_flop, B_flop = Mq, Mkv, B

    out, lse = fmha.memory_efficient_attention_forward_requires_grad(
        query, key, value, attn_bias=attn_bias, op=op_fw
    )
    grad = torch.randn_like(out)

    def _bwd():
        return fmha.memory_efficient_attention_backward(
            grad, out, lse, query, key, value, attn_bias=attn_bias, op=op_bw
        )

    # correctness of the CALL (not accuracy) -- fail loudly if not_supported/crashes,
    # rather than silently reporting garbage timing.
    _bwd()
    torch.cuda.synchronize()

    ms = do_bench(_bwd, warmup=10, rep=50)
    flop = _flop_count(B_flop, Mq_flop, Mkv_flop, H, D, D, causal)
    return ms, flop


def _fmt(v):
    return f"{v:9.4f}" if isinstance(v, float) else f"{str(v):>9}"


if __name__ == "__main__":
    device = "cuda"
    # (tag, B, Mq, Mkv, H, D, dtype, causal, varlen_seqlens)
    cases = [
        ("D64  B1H8",   1, 1024, 1024,  8,  64, torch.bfloat16, False, None),
        ("D64  B2H16",  2, 2048, 2048, 16,  64, torch.bfloat16, False, None),
        ("D128 B1H8",   1, 1024, 1024,  8, 128, torch.bfloat16, False, None),
        ("D256 B1H8",   1, 1024, 1024,  8, 256, torch.bfloat16, False, None),
        ("D64  causal", 1, 1024, 1024,  8,  64, torch.bfloat16, True,  None),
        ("D64  f16",    1, 1024, 1024,  8,  64, torch.float16,  False, None),
        ("D64  varlen", 1,    0,    0,  8,  64, torch.bfloat16, False,
         ([64, 128, 32, 512], [64, 128, 32, 512])),
    ]
    print("=== FlyDSL vs CK backward, both through MSLK's memory_efficient_attention_backward (gfx950/gfx942) ===")
    hdr = (f"{'shape':>14} | {'ck_ms':>9} {'ck_TFl':>7} | {'fly_ms':>9} {'fly_TFl':>7} | "
           f"{'ck/fly':>7}")
    print(hdr); print("-" * len(hdr))
    for tag, B, Mq, Mkv, H, D, dtype, causal, varlen_seqlens in cases:
        try:
            ck_ms, flop = run_op(
                fmha.ck.FwOp, fmha.ck.BwOp, B, Mq, Mkv, H, D, dtype, causal,
                varlen_seqlens, device,
            )
            ck_tflops = flop / ck_ms / 1e9
        except Exception as e:
            print(f"{tag:>14} | CK failed: {type(e).__name__}: {e}")
            continue
        try:
            fly_ms, _ = run_op(
                fmha.ck.FwOp, fmha.flydsl.BwOp, B, Mq, Mkv, H, D, dtype, causal,
                varlen_seqlens, device,
            )
            fly_tflops = flop / fly_ms / 1e9
            ratio = ck_ms / fly_ms  # >1 => FlyDSL faster, <1 => CK faster
            print(f"{tag:>14} | {_fmt(ck_ms)} {_fmt(ck_tflops)} | {_fmt(fly_ms)} "
                  f"{_fmt(fly_tflops)} | {ratio:6.2f}x")
        except Exception as e:
            print(f"{tag:>14} | {_fmt(ck_ms)} {_fmt(ck_tflops)} | FlyDSL failed: "
                  f"{type(e).__name__}: {e}")
    print("\nck/fly = ck_ms/fly_ms (>1 => FlyDSL faster, <1 => CK faster). Both ops timed "
          "through the identical fmha.memory_efficient_attention_backward call; forward "
          "is always ck.FwOp for both rows. TFlops uses a shared analytic FLOP formula "
          "(see _flop_count), not either op's own self-reporting.")
