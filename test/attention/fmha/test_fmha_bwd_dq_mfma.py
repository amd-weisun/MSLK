"""WP-A3 Phase B.3 — test: MFMA dQ kernel vs ref_fmha_bwd.

Usage (inside container):
    cd /workspace/MSLK
    HIP_VISIBLE_DEVICES=3 FLYDSL_RUNTIME_ENABLE_CACHE=0 \
    PYTHONPATH=/workspace/FlyDSL:$PYTHONPATH \
    python test/attention/fmha/test_fmha_bwd_dq_mfma.py
"""
import math
import sys
import torch
import flydsl.compiler as flyc

sys.path.insert(0, ".")
from mslk.attention.flydsl.fmha_bwd_mfma import compile_fmha_bwd_dq_mfma
sys.path.insert(0, "test/attention/fmha")
from test_fmha_bwd_reference import ref_fmha_bwd, ref_fmha_fwd


def _as_i16(t):
    assert t.dtype in (torch.bfloat16, torch.float16)
    return t.view(torch.int16)


def run_case(B, M, N, H, D, dtype, device="cuda"):
    print(f"  B={B} M={M} N={N} H={H} D={D} dtype={dtype}", end=" ... ", flush=True)
    scale = 1.0 / math.sqrt(D)

    Q  = torch.randn(B, M, H, D, device=device, dtype=dtype)
    K  = torch.randn(B, N, H, D, device=device, dtype=dtype)
    V  = torch.randn(B, N, H, D, device=device, dtype=dtype)
    O, LSE = ref_fmha_fwd(Q, K, V, scale=scale, causal=False)
    dO = torch.randn_like(O)
    dQ_ref, _, _ = ref_fmha_bwd(Q, K, V, O, dO, LSE, scale=scale, causal=False)

    BLOCK_M, BLOCK_N = 64, 64
    dtype_str = "bf16" if dtype == torch.bfloat16 else "f16"

    Q_2d   = _as_i16(Q.contiguous().view(B * M * H, D))
    K_2d   = _as_i16(K.contiguous().view(B * N * H, D))
    V_2d   = _as_i16(V.contiguous().view(B * N * H, D))
    dO_2d  = _as_i16(dO.contiguous().view(B * M * H, D))
    LSE_2d = LSE.contiguous().view(B * H * M, 1)
    D_vec  = (dO.float() * O.float()).sum(dim=-1).contiguous().view(B * M * H, 1)
    dQ_out = torch.zeros(B * M * H * D, 1, device=device, dtype=torch.float32)

    launch_fn = compile_fmha_bwd_dq_mfma(D=D, dtype_str=dtype_str,
                                          BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, scale=scale)
    n_N_tiles = (N + BLOCK_N - 1) // BLOCK_N
    compiled = flyc.compile(launch_fn,
                            Q_2d, K_2d, V_2d, dO_2d, dQ_out, LSE_2d, D_vec,
                            B, M, N, H, n_N_tiles,
                            torch.cuda.current_stream())
    compiled(Q_2d, K_2d, V_2d, dO_2d, dQ_out, LSE_2d, D_vec,
             B, M, N, H, n_N_tiles,
             torch.cuda.current_stream())
    torch.cuda.synchronize()

    dQ_kernel = dQ_out.view(B, M, H, D).to(dtype)
    max_err = (dQ_kernel.float() - dQ_ref.float()).abs().max().item()
    ok = torch.allclose(dQ_kernel.float(), dQ_ref.float(), rtol=0.1, atol=0.1)
    print(f"{'PASS' if ok else 'FAIL'}  max_err={max_err:.5f}")
    if not ok:
        diff = (dQ_kernel.float() - dQ_ref.float()).abs()
        idx  = diff.flatten().argmax().item()
        b_i  = idx // (M * H * D); rem = idx % (M * H * D)
        m_i  = rem // (H * D);     rem = rem % (H * D)
        h_i  = rem // D;           d_i = rem % D
        print(f"    Worst: [b={b_i},m={m_i},h={h_i},d={d_i}]  "
              f"kernel={dQ_kernel[b_i,m_i,h_i,d_i].item():.4f}  "
              f"ref={dQ_ref[b_i,m_i,h_i,d_i].item():.4f}")
    return ok


if __name__ == "__main__":
    device = "cuda"
    all_pass = True

    print("=== dQ MFMA kernel (Phase B.3) vs ref_fmha_bwd ===")
    cases = [
        (1,   64,  64,  1,  64,  torch.bfloat16),
        (1,  128,  64,  1,  64,  torch.bfloat16),
        (1,   64, 128,  1,  64,  torch.bfloat16),
        (1,  128, 128,  8,  64,  torch.bfloat16),
        (2,  256, 128,  8,  64,  torch.bfloat16),
        (1,  128, 128,  8,  64,  torch.float16),
    ]
    for args in cases:
        all_pass &= run_case(*args, device=device)

    print(f"\n{'ALL PASSED' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)
