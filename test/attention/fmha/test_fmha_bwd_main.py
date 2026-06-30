"""WP-A3 Phase A — test: dV and dK standalone kernels vs PyTorch reference.

Compares dV from the FlyDSL kernel against ref_fmha_bwd() (which uses
PyTorch autograd as the ground truth).

Usage (inside container aiter-weisun):
    cd /workspace/MSLK
    FLYDSL_RUNTIME_ENABLE_CACHE=0 PYTHONPATH=/workspace/FlyDSL:$PYTHONPATH \\
        python test/attention/fmha/test_fmha_bwd_main.py
"""

import math
import sys

import torch
import flydsl.compiler as flyc

sys.path.insert(0, ".")
from mslk.attention.flydsl.fmha_bwd_main import (
    compile_fmha_bwd_dv,
    compile_fmha_bwd_dk,
    compile_fmha_bwd_dq,
    compile_fmha_bwd_dqdkdv,
    compile_fmha_bwd_convert_dq,
    compile_fmha_bwd_dvdk,
)
from mslk.attention.flydsl.fmha_bwd_preprocess import compile_fmha_bwd_preprocess

# Import reference implementations
sys.path.insert(0, "test/attention/fmha")
from test_fmha_bwd_reference import ref_fmha_bwd, ref_fmha_fwd


def _as_i16(t):
    """View bf16/fp16 tensor as int16 (FlyDSL buffer convention)."""
    assert t.dtype in (torch.bfloat16, torch.float16)
    return t.view(torch.int16)


def run_case(B, M, N, H, D, dtype, device="cuda"):
    """Run one test case and return True if dV passes."""
    print(f"  B={B} M={M} N={N} H={H} D={D} dtype={dtype}", end=" ... ", flush=True)

    scale = 1.0 / math.sqrt(D)

    # Random inputs
    Q = torch.randn(B, M, H, D, device=device, dtype=dtype)
    K = torch.randn(B, N, H, D, device=device, dtype=dtype)
    V = torch.randn(B, N, H, D, device=device, dtype=dtype)

    # Ground-truth forward pass (to get O and LSE)
    O, LSE = ref_fmha_fwd(Q, K, V, scale=scale, causal=False)

    # Upstream gradient (random)
    dO = torch.randn_like(O)

    # Ground-truth backward pass (PyTorch math)
    dQ_ref, dK_ref, dV_ref = ref_fmha_bwd(Q, K, V, O, dO, LSE, scale=scale, causal=False)

    # ---- FlyDSL dV kernel ----
    BLOCK_M = 64
    BLOCK_N = 64
    assert D == BLOCK_N, f"This kernel requires D == BLOCK_N ({BLOCK_N}), got D={D}"

    dtype_str = "bf16" if dtype == torch.bfloat16 else "f16"
    launch_fn = compile_fmha_bwd_dv(D=D, dtype_str=dtype_str, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, scale=scale)

    # Reshape tensors to 2D [B*seq*H, D] int16 views
    Q_2d  = _as_i16(Q.contiguous().view(B * M * H, D))
    K_2d  = _as_i16(K.contiguous().view(B * N * H, D))
    dO_2d = _as_i16(dO.contiguous().view(B * M * H, D))

    # dV output: [B*N*H*D, 1] float32 (flat 2D for row-based scalar stores)
    dV_out = torch.zeros(B * N * H * D, 1, device=device, dtype=torch.float32)

    # LSE: [B, H, M] -> [B*H*M, 1] float32 (2D: one scalar per row)
    # ref_fmha_fwd returns LSE with shape [B, H, M]
    LSE_2d = LSE.contiguous().view(B * H * M, 1)

    n_M_tiles = (M + BLOCK_M - 1) // BLOCK_M

    compiled = flyc.compile(
        launch_fn,
        Q_2d, K_2d, dO_2d, dV_out, LSE_2d,
        B, M, N, H, n_M_tiles,
        torch.cuda.current_stream(),
    )
    compiled(
        Q_2d, K_2d, dO_2d, dV_out, LSE_2d,
        B, M, N, H, n_M_tiles,
        torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()

    # Reshape dV output to [B, N, H, D] for comparison
    # dV_out is [B*N*H*D, 1] -> view as [B, N, H, D]
    dV_kernel = dV_out.view(B, N, H, D).to(dtype)

    max_err = (dV_kernel.float() - dV_ref.float()).abs().max().item()
    rel_err = max_err / (dV_ref.float().abs().max().item() + 1e-6)

    # Tolerance: match what CK test uses (rtol=0.1, atol=0.1 for bf16 backward)
    ok = torch.allclose(dV_kernel.float(), dV_ref.float(), rtol=0.1, atol=0.1)
    print(f"{'PASS' if ok else 'FAIL'}  max_err={max_err:.5f}  rel_err={rel_err:.4f}")

    if not ok:
        # Print a few samples for debugging
        diff = (dV_kernel.float() - dV_ref.float()).abs()
        idx  = diff.flatten().argmax().item()
        b_i  = idx // (N * H * D)
        rem  = idx % (N * H * D)
        n_i  = rem // (H * D)
        h_i  = (rem % (H * D)) // D
        d_i  = rem % D
        print(f"    Worst: [b={b_i},n={n_i},h={h_i},d={d_i}]  "
              f"kernel={dV_kernel[b_i,n_i,h_i,d_i].item():.4f}  "
              f"ref={dV_ref[b_i,n_i,h_i,d_i].item():.4f}")

    return ok


def _run_preprocess(B, M, H, D, dtype, device):
    """Compute D_vec via the preprocess kernel; return [B*M*H, 1] float32."""
    dO = torch.randn(B, M, H, D, device=device, dtype=dtype)
    O  = torch.randn(B, M, H, D, device=device, dtype=dtype)
    dtype_str = "bf16" if dtype == torch.bfloat16 else "f16"
    launch = compile_fmha_bwd_preprocess(D=D, dtype_str=dtype_str)
    dO_2d = _as_i16(dO.contiguous().view(B * M * H, D))
    O_2d  = _as_i16(O.contiguous().view(B * M * H, D))
    D_out = torch.zeros(B * M * H, 1, device=device, dtype=torch.float32)
    n_rows = B * M * H
    compiled = flyc.compile(launch, dO_2d, O_2d, D_out, n_rows,
                            torch.cuda.current_stream())
    compiled(dO_2d, O_2d, D_out, n_rows, torch.cuda.current_stream())
    torch.cuda.synchronize()
    return dO, O, D_out   # D_out: [B*M*H, 1]


def run_dk_case(B, M, N, H, D, dtype, device="cuda"):
    """Test dK kernel against ref_fmha_bwd."""
    print(f"  B={B} M={M} N={N} H={H} D={D} dtype={dtype}", end=" ... ", flush=True)

    scale = 1.0 / math.sqrt(D)
    Q  = torch.randn(B, M, H, D, device=device, dtype=dtype)
    K  = torch.randn(B, N, H, D, device=device, dtype=dtype)
    V  = torch.randn(B, N, H, D, device=device, dtype=dtype)
    O, LSE = ref_fmha_fwd(Q, K, V, scale=scale, causal=False)
    dO = torch.randn_like(O)
    _, dK_ref, _ = ref_fmha_bwd(Q, K, V, O, dO, LSE, scale=scale, causal=False)

    BLOCK_M, BLOCK_N = 64, 64
    dtype_str = "bf16" if dtype == torch.bfloat16 else "f16"

    # D_vec from preprocess kernel
    _, _, D_vec = _run_preprocess(B, M, H, D, dtype, device)
    # Replace with reference D_vec to isolate dK correctness
    D_vec_ref = (dO.float() * O.float()).sum(dim=-1)   # [B, M, H]
    D_vec_2d  = D_vec_ref.contiguous().view(B * M * H, 1)

    Q_2d  = _as_i16(Q.contiguous().view(B * M * H, D))
    K_2d  = _as_i16(K.contiguous().view(B * N * H, D))
    V_2d  = _as_i16(V.contiguous().view(B * N * H, D))
    dO_2d = _as_i16(dO.contiguous().view(B * M * H, D))
    LSE_2d = LSE.contiguous().view(B * H * M, 1)

    dK_out = torch.zeros(B * N * H * D, 1, device=device, dtype=torch.float32)

    launch_fn = compile_fmha_bwd_dk(D=D, dtype_str=dtype_str,
                                     BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, scale=scale)
    n_M_tiles = (M + BLOCK_M - 1) // BLOCK_M
    compiled = flyc.compile(launch_fn,
                            Q_2d, K_2d, V_2d, dO_2d, dK_out, LSE_2d, D_vec_2d,
                            B, M, N, H, n_M_tiles,
                            torch.cuda.current_stream())
    compiled(Q_2d, K_2d, V_2d, dO_2d, dK_out, LSE_2d, D_vec_2d,
             B, M, N, H, n_M_tiles,
             torch.cuda.current_stream())
    torch.cuda.synchronize()

    dK_kernel = dK_out.view(B, N, H, D).to(dtype)
    max_err = (dK_kernel.float() - dK_ref.float()).abs().max().item()
    ok = torch.allclose(dK_kernel.float(), dK_ref.float(), rtol=0.1, atol=0.1)
    print(f"{'PASS' if ok else 'FAIL'}  max_err={max_err:.5f}")
    if not ok:
        diff = (dK_kernel.float() - dK_ref.float()).abs()
        idx  = diff.flatten().argmax().item()
        b_i  = idx // (N * H * D); rem = idx % (N * H * D)
        n_i  = rem // (H * D);     rem = rem % (H * D)
        h_i  = rem // D;           d_i = rem % D
        print(f"    Worst: [b={b_i},n={n_i},h={h_i},d={d_i}]  "
              f"kernel={dK_kernel[b_i,n_i,h_i,d_i].item():.4f}  "
              f"ref={dK_ref[b_i,n_i,h_i,d_i].item():.4f}")
    return ok


def run_dq_case(B, M, N, H, D, dtype, device="cuda"):
    """Test dQ standalone kernel against ref_fmha_bwd."""
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

    # Use reference D_vec to isolate dQ correctness
    D_vec_ref = (dO.float() * O.float()).sum(dim=-1)   # [B, M, H]
    D_vec_2d  = D_vec_ref.contiguous().view(B * M * H, 1)

    Q_2d  = _as_i16(Q.contiguous().view(B * M * H, D))
    K_2d  = _as_i16(K.contiguous().view(B * N * H, D))
    V_2d  = _as_i16(V.contiguous().view(B * N * H, D))
    dO_2d = _as_i16(dO.contiguous().view(B * M * H, D))
    LSE_2d = LSE.contiguous().view(B * H * M, 1)

    dQ_out = torch.zeros(B * M * H * D, 1, device=device, dtype=torch.float32)

    launch_fn = compile_fmha_bwd_dq(D=D, dtype_str=dtype_str,
                                     BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, scale=scale)
    n_N_tiles = (N + BLOCK_N - 1) // BLOCK_N
    compiled = flyc.compile(launch_fn,
                            Q_2d, K_2d, V_2d, dO_2d, dQ_out, LSE_2d, D_vec_2d,
                            B, M, N, H, n_N_tiles,
                            torch.cuda.current_stream())
    compiled(Q_2d, K_2d, V_2d, dO_2d, dQ_out, LSE_2d, D_vec_2d,
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


def run_dvdk_case(B, M, N, H, D, dtype, device="cuda"):
    """Test fused dV+dK kernel against ref_fmha_bwd (no atomics, no dQ)."""
    print(f"  B={B} M={M} N={N} H={H} D={D} dtype={dtype}", end=" ... ", flush=True)

    scale = 1.0 / math.sqrt(D)
    Q  = torch.randn(B, M, H, D, device=device, dtype=dtype)
    K  = torch.randn(B, N, H, D, device=device, dtype=dtype)
    V  = torch.randn(B, N, H, D, device=device, dtype=dtype)
    O, LSE = ref_fmha_fwd(Q, K, V, scale=scale, causal=False)
    dO = torch.randn_like(O)
    _, dK_ref, dV_ref = ref_fmha_bwd(Q, K, V, O, dO, LSE, scale=scale, causal=False)

    BLOCK_M, BLOCK_N = 64, 64
    dtype_str = "bf16" if dtype == torch.bfloat16 else "f16"
    D_vec_ref = (dO.float() * O.float()).sum(dim=-1).view(B * M * H, 1)

    Q_2d  = _as_i16(Q.contiguous().view(B * M * H, D))
    K_2d  = _as_i16(K.contiguous().view(B * N * H, D))
    V_2d  = _as_i16(V.contiguous().view(B * N * H, D))
    dO_2d = _as_i16(dO.contiguous().view(B * M * H, D))
    LSE_2d = LSE.contiguous().view(B * H * M, 1)

    dV_out = torch.zeros(B * N * H * D, 1, device=device, dtype=torch.float32)
    dK_out = torch.zeros(B * N * H * D, 1, device=device, dtype=torch.float32)

    fn = compile_fmha_bwd_dvdk(D=D, dtype_str=dtype_str,
                                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, scale=scale)
    n_M_tiles = (M + BLOCK_M - 1) // BLOCK_M
    compiled = flyc.compile(fn, Q_2d, K_2d, V_2d, dO_2d, dV_out, dK_out,
                            LSE_2d, D_vec_ref, B, M, N, H, n_M_tiles,
                            torch.cuda.current_stream())
    compiled(Q_2d, K_2d, V_2d, dO_2d, dV_out, dK_out,
             LSE_2d, D_vec_ref, B, M, N, H, n_M_tiles,
             torch.cuda.current_stream())
    torch.cuda.synchronize()

    dV_kernel = dV_out.view(B, N, H, D).to(dtype)
    dK_kernel = dK_out.view(B, N, H, D).to(dtype)
    ok_dv = torch.allclose(dV_kernel.float(), dV_ref.float(), rtol=0.1, atol=0.1)
    ok_dk = torch.allclose(dK_kernel.float(), dK_ref.float(), rtol=0.1, atol=0.1)
    ok = ok_dv and ok_dk
    err_dv = (dV_kernel.float() - dV_ref.float()).abs().max().item()
    err_dk = (dK_kernel.float() - dK_ref.float()).abs().max().item()
    status = "PASS" if ok else f"FAIL(dV={'OK' if ok_dv else 'FAIL'} dK={'OK' if ok_dk else 'FAIL'})"
    print(f"{status}  dV={err_dv:.5f} dK={err_dk:.5f}")
    return ok


def run_fused_case(B, M, N, H, D, dtype, device="cuda"):
    """Test fused dQdKdV + convert_dq kernels against ref_fmha_bwd."""
    print(f"  B={B} M={M} N={N} H={H} D={D} dtype={dtype}", end=" ... ", flush=True)

    scale = 1.0 / math.sqrt(D)
    Q  = torch.randn(B, M, H, D, device=device, dtype=dtype)
    K  = torch.randn(B, N, H, D, device=device, dtype=dtype)
    V  = torch.randn(B, N, H, D, device=device, dtype=dtype)
    O, LSE = ref_fmha_fwd(Q, K, V, scale=scale, causal=False)
    dO = torch.randn_like(O)
    dQ_ref, dK_ref, dV_ref = ref_fmha_bwd(Q, K, V, O, dO, LSE, scale=scale, causal=False)

    BLOCK_M, BLOCK_N = 64, 64
    dtype_str = "bf16" if dtype == torch.bfloat16 else "f16"

    # Reference D_vec to isolate fused kernel math
    D_vec_ref = (dO.float() * O.float()).sum(dim=-1)   # [B, M, H]
    D_vec_2d  = D_vec_ref.contiguous().view(B * M * H, 1)

    Q_2d  = _as_i16(Q.contiguous().view(B * M * H, D))
    K_2d  = _as_i16(K.contiguous().view(B * N * H, D))
    V_2d  = _as_i16(V.contiguous().view(B * N * H, D))
    dO_2d = _as_i16(dO.contiguous().view(B * M * H, D))
    LSE_2d = LSE.contiguous().view(B * H * M, 1)

    # Output buffers
    dV_out   = torch.zeros(B * N * H * D, 1, device=device, dtype=torch.float32)
    dK_out   = torch.zeros(B * N * H * D, 1, device=device, dtype=torch.float32)
    dQ_f32   = torch.zeros(B * M * H * D, 1, device=device, dtype=torch.float32)  # zeroed for atomics
    dQ_out   = torch.zeros(B * M * H, D,  device=device, dtype=torch.int16)        # bf16 output

    # Kernel 2: fused dV + dK + dQ atomic accumulation
    fused_fn  = compile_fmha_bwd_dqdkdv(D=D, dtype_str=dtype_str,
                                          BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, scale=scale)
    n_M_tiles = (M + BLOCK_M - 1) // BLOCK_M
    compiled_fused = flyc.compile(fused_fn,
                                   Q_2d, K_2d, V_2d, dO_2d,
                                   dV_out, dK_out, dQ_f32, LSE_2d, D_vec_2d,
                                   B, M, N, H, n_M_tiles,
                                   torch.cuda.current_stream())
    compiled_fused(Q_2d, K_2d, V_2d, dO_2d,
                   dV_out, dK_out, dQ_f32, LSE_2d, D_vec_2d,
                   B, M, N, H, n_M_tiles,
                   torch.cuda.current_stream())
    torch.cuda.synchronize()

    # Kernel 3: convert dQ f32 -> bf16
    convert_fn = compile_fmha_bwd_convert_dq(D=D, dtype_str=dtype_str, BLOCK_M=BLOCK_M)
    n_M_tiles_cvt = (M + BLOCK_M - 1) // BLOCK_M
    compiled_cvt = flyc.compile(convert_fn,
                                 dQ_f32, dQ_out, B, M, H, n_M_tiles_cvt,
                                 torch.cuda.current_stream())
    compiled_cvt(dQ_f32, dQ_out, B, M, H, n_M_tiles_cvt, torch.cuda.current_stream())
    torch.cuda.synchronize()

    # Compare all three gradients
    dV_kernel = dV_out.view(B, N, H, D).to(dtype)
    dK_kernel = dK_out.view(B, N, H, D).to(dtype)
    dQ_kernel = dQ_out.view(torch.bfloat16 if dtype_str == "bf16" else torch.float16).view(B, M, H, D)

    ok_dv = torch.allclose(dV_kernel.float(), dV_ref.float(), rtol=0.1, atol=0.1)
    ok_dk = torch.allclose(dK_kernel.float(), dK_ref.float(), rtol=0.1, atol=0.1)
    ok_dq = torch.allclose(dQ_kernel.float(), dQ_ref.float(), rtol=0.1, atol=0.1)
    ok = ok_dv and ok_dk and ok_dq

    err_dv = (dV_kernel.float() - dV_ref.float()).abs().max().item()
    err_dk = (dK_kernel.float() - dK_ref.float()).abs().max().item()
    err_dq = (dQ_kernel.float() - dQ_ref.float()).abs().max().item()
    status = "PASS" if ok else f"FAIL(dV={'OK' if ok_dv else 'FAIL'} dK={'OK' if ok_dk else 'FAIL'} dQ={'OK' if ok_dq else 'FAIL'})"
    print(f"{status}  dV={err_dv:.5f} dK={err_dk:.5f} dQ={err_dq:.5f}")
    return ok


if __name__ == "__main__":
    device = "cuda"
    all_pass = True

    print("=== dV kernel (FlyDSL vs ref_fmha_bwd) ===")
    cases = [
        # B    M    N    H    D    dtype
        (1,   64,  64,  1,  64,  torch.bfloat16),
        (1,  128,  64,  1,  64,  torch.bfloat16),
        (1,   64, 128,  1,  64,  torch.bfloat16),
        (1,  128, 128,  8,  64,  torch.bfloat16),
        (2,  256, 128,  8,  64,  torch.bfloat16),
        (1,  128, 128,  8,  64,  torch.float16),
    ]
    for args in cases:
        all_pass &= run_case(*args, device=device)

    print("\n=== dK kernel (FlyDSL vs ref_fmha_bwd) ===")
    for args in cases:
        all_pass &= run_dk_case(*args, device=device)

    print("\n=== dQ kernel (FlyDSL vs ref_fmha_bwd) ===")
    for args in cases:
        all_pass &= run_dq_case(*args, device=device)

    print("\n=== Fused dQdKdV kernel (FlyDSL vs ref_fmha_bwd) ===")
    for args in cases:
        all_pass &= run_dvdk_case(*args, device=device)

    print(f"\n{'ALL PASSED' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)
