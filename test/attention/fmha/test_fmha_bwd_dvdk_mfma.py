"""WP-A3 Phase B.4 — test: FUSED dV+dK MFMA kernel vs ref_fmha_bwd.

Usage (inside container):
    cd /workspace/MSLK
    HIP_VISIBLE_DEVICES=3 FLYDSL_RUNTIME_ENABLE_CACHE=0 \
    PYTHONPATH=/workspace/FlyDSL:$PYTHONPATH \
    python test/attention/fmha/test_fmha_bwd_dvdk_mfma.py
"""
import math
import sys
import torch
import flydsl.compiler as flyc

sys.path.insert(0, ".")
from mslk.attention.flydsl.fmha_bwd_mfma import compile_fmha_bwd_dvdk_mfma
sys.path.insert(0, "test/attention/fmha")
from test_fmha_bwd_reference import ref_fmha_bwd, ref_fmha_fwd


def _as_i16(t):
    return t.view(torch.int16)


def run_case(B, M, N, H, D, dtype, device="cuda"):
    print(f"  B={B} M={M} N={N} H={H} D={D} dtype={dtype}", end=" ... ", flush=True)
    scale = 1.0 / math.sqrt(D)

    Q  = torch.randn(B, M, H, D, device=device, dtype=dtype)
    K  = torch.randn(B, N, H, D, device=device, dtype=dtype)
    V  = torch.randn(B, N, H, D, device=device, dtype=dtype)
    O, LSE = ref_fmha_fwd(Q, K, V, scale=scale, causal=False)
    dO = torch.randn_like(O)
    _, dK_ref, dV_ref = ref_fmha_bwd(Q, K, V, O, dO, LSE, scale=scale, causal=False)

    BLOCK_M, BLOCK_N = 128, 64
    dtype_str = "bf16" if dtype == torch.bfloat16 else "f16"

    Q_2d   = _as_i16(Q.contiguous().view(B * M * H, D))
    K_2d   = _as_i16(K.contiguous().view(B * N * H, D))
    V_2d   = _as_i16(V.contiguous().view(B * N * H, D))
    dO_2d  = _as_i16(dO.contiguous().view(B * M * H, D))
    LSE_2d = LSE.contiguous().view(B * H * M, 1)
    D_vec  = (dO.float() * O.float()).sum(dim=-1).contiguous().view(B * M * H, 1)
    dV_out = torch.zeros(B * N * H * D, 1, device=device, dtype=torch.float32)
    dK_out = torch.zeros(B * N * H * D, 1, device=device, dtype=torch.float32)

    launch_fn = compile_fmha_bwd_dvdk_mfma(D=D, dtype_str=dtype_str,
                                            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, scale=scale)
    n_M_tiles = (M + BLOCK_M - 1) // BLOCK_M
    args = (Q_2d, K_2d, V_2d, dO_2d, dV_out, dK_out, LSE_2d, D_vec,
            B, M, N, H, n_M_tiles, torch.cuda.current_stream())
    compiled = flyc.compile(launch_fn, *args)
    compiled(*args)
    torch.cuda.synchronize()

    dV_k = dV_out.view(B, N, H, D).to(dtype)
    dK_k = dK_out.view(B, N, H, D).to(dtype)
    err_v = (dV_k.float() - dV_ref.float()).abs().max().item()
    err_k = (dK_k.float() - dK_ref.float()).abs().max().item()
    ok_v = torch.allclose(dV_k.float(), dV_ref.float(), rtol=0.1, atol=0.1)
    ok_k = torch.allclose(dK_k.float(), dK_ref.float(), rtol=0.1, atol=0.1)
    ok = ok_v and ok_k
    print(f"{'PASS' if ok else 'FAIL'}  dV_err={err_v:.5f} dK_err={err_k:.5f}")
    return ok


if __name__ == "__main__":
    device = "cuda"
    all_pass = True
    print("=== FUSED dV+dK MFMA kernel (Phase B.4) vs ref_fmha_bwd ===")
    cases = [
        (1,  128,  64,  1,  64,  torch.bfloat16),   # M=BLOCK_M
        (1,  256,  64,  1,  64,  torch.bfloat16),   # M=2*BLOCK_M
        (1,  128, 128,  1,  64,  torch.bfloat16),
        (1,  256, 128,  8,  64,  torch.bfloat16),
        (2,  512, 128,  8,  64,  torch.bfloat16),
        (1,  256, 128,  8,  64,  torch.float16),
    ]
    for args in cases:
        all_pass &= run_case(*args, device=device)
    print(f"\n{'ALL PASSED' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)
