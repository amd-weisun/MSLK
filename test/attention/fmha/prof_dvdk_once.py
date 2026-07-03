"""Minimal single-shot driver for profiling: runs FlyDSL dvdk_fused + dq once
at a fixed large shape, for rocprofv3 --stats / ATT capture.

    HIP_VISIBLE_DEVICES=3 PYTHONPATH=/workspace/FlyDSL:$PYTHONPATH \
    python test/attention/fmha/prof_dvdk_once.py
"""
import math
import os
import sys
import torch
import flydsl.compiler as flyc

sys.path.insert(0, ".")
from mslk.attention.flydsl.fmha_bwd_mfma import (
    compile_fmha_bwd_dvdk_mfma,
    compile_fmha_bwd_dq_mfma,
)
sys.path.insert(0, "test/attention/fmha")
from test_fmha_bwd_reference import ref_fmha_fwd

B, M, N, H, D = 2, 2048, 2048, 16, 64
dtype = torch.bfloat16
ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 20


def _as_i16(t):
    return t.view(torch.int16)


def main():
    dev = "cuda"
    scale = 1.0 / math.sqrt(D)
    Q = torch.randn(B, M, H, D, device=dev, dtype=dtype)
    K = torch.randn(B, N, H, D, device=dev, dtype=dtype)
    V = torch.randn(B, N, H, D, device=dev, dtype=dtype)
    O, LSE = ref_fmha_fwd(Q, K, V, scale=scale, causal=False)
    dO = torch.randn_like(O)

    Q2 = _as_i16(Q.contiguous().view(B * M * H, D))
    K2 = _as_i16(K.contiguous().view(B * N * H, D))
    V2 = _as_i16(V.contiguous().view(B * N * H, D))
    dO2 = _as_i16(dO.contiguous().view(B * M * H, D))
    LSE2 = LSE.contiguous().view(B * H * M, 1)
    Dv = (dO.float() * O.float()).sum(-1).contiguous().view(B * M * H, 1)
    o_dv = torch.zeros(B * N * H * D, 1, device=dev, dtype=torch.float32)
    o_dk = torch.zeros(B * N * H * D, 1, device=dev, dtype=torch.float32)
    o_dq = torch.zeros(B * M * H * D, 1, device=dev, dtype=torch.float32)
    st = torch.cuda.current_stream()
    BM, BN = 128, 64
    nM = (M + BM - 1) // BM
    nN = (N + BN - 1) // BN

    _trload = os.environ.get("PROF_TRLOAD", "0") == "1"
    _pipe = os.environ.get("PROF_PIPELINE", "0") == "1"
    dvdk = compile_fmha_bwd_dvdk_mfma(D=D, dtype_str="bf16", scale=scale, BLOCK_M=BM,
                                      use_trload=_trload, use_pipeline=_pipe)
    dq = compile_fmha_bwd_dq_mfma(D=D, dtype_str="bf16", scale=scale)
    a_dvdk = (Q2, K2, V2, dO2, o_dv, o_dk, LSE2, Dv, B, M, N, H, nM, st)
    a_dq = (Q2, K2, V2, dO2, o_dq, LSE2, Dv, B, M, N, H, nN, st)
    c_dvdk = flyc.compile(dvdk, *a_dvdk)
    c_dq = flyc.compile(dq, *a_dq)

    # warmup
    for _ in range(3):
        c_dvdk(*a_dvdk); c_dq(*a_dq)
    torch.cuda.synchronize()

    for _ in range(ITERS):
        c_dvdk(*a_dvdk)
        c_dq(*a_dq)
    torch.cuda.synchronize()
    print(f"done {ITERS} iters, shape B{B} M{M} N{N} H{H} D{D}")


if __name__ == "__main__":
    main()
