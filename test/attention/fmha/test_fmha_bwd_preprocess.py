"""WP-A3 Phase A — test: D-vector preprocess kernel vs PyTorch reference.

D[b, h, m] = rowsum(dO[b, m, h, :] * O[b, m, h, :])

Usage:
    cd /workspace/MSLK
    PYTHONPATH=/workspace/FlyDSL:$PYTHONPATH \
        python test/attention/fmha/test_fmha_bwd_preprocess.py
"""

import math
import sys
import torch
import flydsl.compiler as flyc

sys.path.insert(0, ".")
from mslk.attention.flydsl.fmha_bwd_preprocess import compile_fmha_bwd_preprocess


def ref_d_vec(dO, O):
    """Reference: D[b,m,h] = rowsum(dO * O) over head_dim.

    Args:
        dO, O : [B, M, H, D]
    Returns:
        D_vec : [B, M, H]  float32   (same layout as kernel output [B*M*H])
    """
    return (dO.float() * O.float()).sum(dim=-1)  # [B, M, H]


def _as_i16(t):
    """View bf16/fp16 tensor as int16 (FlyDSL buffer convention)."""
    return t.view(torch.int16)


def run_case(B, M, H, D, dtype, device="cuda"):
    print(f"  B={B} M={M} H={H} D={D} dtype={dtype}", end=" ... ", flush=True)

    dO = torch.randn(B, M, H, D, device=device, dtype=dtype)
    O  = torch.randn(B, M, H, D, device=device, dtype=dtype)

    # PyTorch reference: [B, M, H]
    D_ref = ref_d_vec(dO, O)

    # FlyDSL kernel
    dtype_str = "bf16" if dtype == torch.bfloat16 else "f16"
    launch_fn = compile_fmha_bwd_preprocess(D=D, dtype_str=dtype_str)

    # dO/O: [B*M*H, D] viewed as int16 (2D — required for fx.slice row indexing)
    dO_2d = _as_i16(dO.contiguous().view(B * M * H, D))
    O_2d  = _as_i16(O.contiguous().view(B * M * H, D))

    # D_out: [B*M*H, 1] float32
    D_out = torch.zeros(B * M * H, 1, device=device, dtype=torch.float32)

    n_rows = B * M * H
    compiled = flyc.compile(launch_fn, dO_2d, O_2d, D_out, n_rows,
                            torch.cuda.current_stream())
    compiled(dO_2d, O_2d, D_out, n_rows, torch.cuda.current_stream())
    torch.cuda.synchronize()

    # Reshape to [B, M, H] for comparison
    D_out_shaped = D_out.view(B, M, H)
    max_err = (D_out_shaped - D_ref).abs().max().item()
    ok = torch.allclose(D_out_shaped, D_ref, rtol=1e-2, atol=1e-2)
    print(f"{'PASS' if ok else 'FAIL'}  max_err={max_err:.5f}")
    return ok


if __name__ == "__main__":
    device = "cuda"
    all_pass = True

    print("=== D-vector preprocess kernel (FlyDSL vs PyTorch ref) ===")
    cases = [
        # B   M    H   D    dtype
        (1,  128,  8,  64,  torch.bfloat16),
        (1,  128,  8,  64,  torch.float16),
        (1,  128,  8, 128,  torch.bfloat16),
        (2,  256,  8, 128,  torch.bfloat16),
        (1,  512,  4, 128,  torch.float16),
        (2,  256, 16,  64,  torch.bfloat16),
    ]
    for args in cases:
        all_pass &= run_case(*args, device=device)

    print(f"\n{'ALL PASSED' if all_pass else 'SOME FAILED'}")
