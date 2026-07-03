"""WP-A3 Phase B — benchmark MFMA gradients vs Phase A scalar vs PyTorch SDPA autograd.

Measures per-kernel wall time (median, via flydsl do_bench) for dV, dK, dQ:
  - MFMA   : compile_fmha_bwd_{dv,dk,dq}_mfma  (Phase B)
  - scalar : compile_fmha_bwd_{dv,dk,dq}       (Phase A)
  - SDPA   : torch flash SDPA full backward (dQ+dK+dV together) as a reference wall.

Usage (inside container):
    cd /workspace/MSLK
    HIP_VISIBLE_DEVICES=3 PYTHONPATH=/workspace/FlyDSL:$PYTHONPATH \
    python test/attention/fmha/bench_fmha_bwd_mfma.py
"""
import math
import sys
import torch
import flydsl.compiler as flyc
from flydsl.autotune import do_bench

sys.path.insert(0, ".")
from mslk.attention.flydsl.fmha_bwd_mfma import (
    compile_fmha_bwd_dv_mfma,
    compile_fmha_bwd_dk_mfma,
    compile_fmha_bwd_dq_mfma,
)
from mslk.attention.flydsl.fmha_bwd_main import (
    compile_fmha_bwd_dv,
    compile_fmha_bwd_dk,
    compile_fmha_bwd_dq,
)
sys.path.insert(0, "test/attention/fmha")
from test_fmha_bwd_reference import ref_fmha_bwd, ref_fmha_fwd


def _as_i16(t):
    return t.view(torch.int16)


def _prep(B, M, N, H, D, dtype, device):
    scale = 1.0 / math.sqrt(D)
    Q = torch.randn(B, M, H, D, device=device, dtype=dtype)
    K = torch.randn(B, N, H, D, device=device, dtype=dtype)
    V = torch.randn(B, N, H, D, device=device, dtype=dtype)
    O, LSE = ref_fmha_fwd(Q, K, V, scale=scale, causal=False)
    dO = torch.randn_like(O)
    d = dict(
        Q=_as_i16(Q.contiguous().view(B * M * H, D)),
        K=_as_i16(K.contiguous().view(B * N * H, D)),
        V=_as_i16(V.contiguous().view(B * N * H, D)),
        dO=_as_i16(dO.contiguous().view(B * M * H, D)),
        LSE=LSE.contiguous().view(B * H * M, 1),
        Dvec=(dO.float() * O.float()).sum(-1).contiguous().view(B * M * H, 1),
        scale=scale,
    )
    d["raw"] = (Q, K, V, O, dO, LSE)
    return d


def _bench_one(fn):
    try:
        return do_bench(fn, warmup=10, rep=50)
    except Exception as e:
        return f"ERR:{type(e).__name__}"


def bench_case(B, M, N, H, D, dtype, device="cuda"):
    dtype_str = "bf16" if dtype == torch.bfloat16 else "f16"
    p = _prep(B, M, N, H, D, dtype, device)
    stream = torch.cuda.current_stream()
    BM, BN = 64, 64
    nM = (M + BM - 1) // BM
    nN = (N + BN - 1) // BN

    out_dv = torch.zeros(B * N * H * D, 1, device=device, dtype=torch.float32)
    out_dk = torch.zeros(B * N * H * D, 1, device=device, dtype=torch.float32)
    out_dq = torch.zeros(B * M * H * D, 1, device=device, dtype=torch.float32)

    results = {}

    # ---- MFMA ----
    dv_m = compile_fmha_bwd_dv_mfma(D=D, dtype_str=dtype_str, BLOCK_M=BM, BLOCK_N=BN, scale=p["scale"])
    dk_m = compile_fmha_bwd_dk_mfma(D=D, dtype_str=dtype_str, BLOCK_M=BM, BLOCK_N=BN, scale=p["scale"])
    dq_m = compile_fmha_bwd_dq_mfma(D=D, dtype_str=dtype_str, BLOCK_M=BM, BLOCK_N=BN, scale=p["scale"])
    cdv_m = flyc.compile(dv_m, p["Q"], p["K"], p["dO"], out_dv, p["LSE"], B, M, N, H, nM, stream)
    cdk_m = flyc.compile(dk_m, p["Q"], p["K"], p["V"], p["dO"], out_dk, p["LSE"], p["Dvec"], B, M, N, H, nM, stream)
    cdq_m = flyc.compile(dq_m, p["Q"], p["K"], p["V"], p["dO"], out_dq, p["LSE"], p["Dvec"], B, M, N, H, nN, stream)
    results["mfma_dv"] = _bench_one(lambda: cdv_m(p["Q"], p["K"], p["dO"], out_dv, p["LSE"], B, M, N, H, nM, stream))
    results["mfma_dk"] = _bench_one(lambda: cdk_m(p["Q"], p["K"], p["V"], p["dO"], out_dk, p["LSE"], p["Dvec"], B, M, N, H, nM, stream))
    results["mfma_dq"] = _bench_one(lambda: cdq_m(p["Q"], p["K"], p["V"], p["dO"], out_dq, p["LSE"], p["Dvec"], B, M, N, H, nN, stream))

    # ---- Phase A scalar ----
    try:
        dv_s = compile_fmha_bwd_dv(D=D, dtype_str=dtype_str, BLOCK_M=BM, BLOCK_N=BN, scale=p["scale"])
        dk_s = compile_fmha_bwd_dk(D=D, dtype_str=dtype_str, BLOCK_M=BM, BLOCK_N=BN, scale=p["scale"])
        dq_s = compile_fmha_bwd_dq(D=D, dtype_str=dtype_str, BLOCK_M=BM, BLOCK_N=BN, scale=p["scale"])
        cdv_s = flyc.compile(dv_s, p["Q"], p["K"], p["dO"], out_dv, p["LSE"], B, M, N, H, nM, stream)
        cdk_s = flyc.compile(dk_s, p["Q"], p["K"], p["V"], p["dO"], out_dk, p["LSE"], p["Dvec"], B, M, N, H, nM, stream)
        cdq_s = flyc.compile(dq_s, p["Q"], p["K"], p["V"], p["dO"], out_dq, p["LSE"], p["Dvec"], B, M, N, H, nN, stream)
        results["scalar_dv"] = _bench_one(lambda: cdv_s(p["Q"], p["K"], p["dO"], out_dv, p["LSE"], B, M, N, H, nM, stream))
        results["scalar_dk"] = _bench_one(lambda: cdk_s(p["Q"], p["K"], p["V"], p["dO"], out_dk, p["LSE"], p["Dvec"], B, M, N, H, nM, stream))
        results["scalar_dq"] = _bench_one(lambda: cdq_s(p["Q"], p["K"], p["V"], p["dO"], out_dq, p["LSE"], p["Dvec"], B, M, N, H, nN, stream))
    except Exception as e:
        results["scalar_dv"] = results["scalar_dk"] = results["scalar_dq"] = f"ERR:{type(e).__name__}"

    # ---- PyTorch SDPA full backward (dQ+dK+dV fused) ----
    Q, K, V, O, dO, LSE = p["raw"]
    qg = Q.transpose(1, 2).clone().requires_grad_(True)
    kg = K.transpose(1, 2).clone().requires_grad_(True)
    vg = V.transpose(1, 2).clone().requires_grad_(True)
    dOt = dO.transpose(1, 2).contiguous()
    def _sdpa_bwd():
        for g in (qg, kg, vg):
            g.grad = None
        out = torch.nn.functional.scaled_dot_product_attention(qg, kg, vg, scale=p["scale"])
        out.backward(dOt, retain_graph=True)
    results["sdpa_dqkv"] = _bench_one(_sdpa_bwd)

    return results


def _fmt(v):
    return f"{v:8.3f}" if isinstance(v, float) else f"{v:>8}"


if __name__ == "__main__":
    device = "cuda"
    cases = [
        (1,  512,  512,  8, 64, torch.bfloat16),
        (1, 1024, 1024,  8, 64, torch.bfloat16),
        (2, 2048, 2048, 16, 64, torch.bfloat16),
        (1, 1024, 1024,  8, 64, torch.float16),
    ]
    print("=== FMHA backward benchmark (median ms) ===")
    hdr = f"{'shape':>26} | {'mfma_dv':>8} {'scal_dv':>8} | {'mfma_dk':>8} {'scal_dk':>8} | {'mfma_dq':>8} {'scal_dq':>8} | {'sdpa_all':>8}"
    print(hdr)
    print("-" * len(hdr))
    for B, M, N, H, D, dt in cases:
        r = bench_case(B, M, N, H, D, dt, device)
        tag = f"B{B} M{M} N{N} H{H} D{D} {('bf16' if dt==torch.bfloat16 else 'f16')}"
        print(f"{tag:>26} | {_fmt(r['mfma_dv'])} {_fmt(r['scalar_dv'])} | "
              f"{_fmt(r['mfma_dk'])} {_fmt(r['scalar_dk'])} | "
              f"{_fmt(r['mfma_dq'])} {_fmt(r['scalar_dq'])} | {_fmt(r['sdpa_dqkv'])}")
    print("\nNote: mfma/scalar are per-gradient kernels; sdpa_all is fused dQ+dK+dV (one backward).")
