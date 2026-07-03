"""WP-A3 gap table: FlyDSL backward variants vs CK across an M-sweep.

For each shape, times:
  split  = pipelined dvdk (use_pipeline) + pipelined dq (use_pipeline)   [2 kernels]
  fused  = compile_fmha_bwd_dqdkdv_mfma (plain atomic)                    [1 kernel + convert, convert not timed]
  fusedR = compile_fmha_bwd_dqdkdv_mfma (use_lds_reduce)
and CK's tile_example_fmha_bwd (-v=1, self-validated). Reports ms + CK/best ratio.

Each FlyDSL variant is validated vs torch ref (dQ+dK+dV) before timing.
NOTE: flyc.compile runs the kernel once, so we compile against throwaway buffers
and time on fresh zeroed buffers (fused dQ atomic accumulates).

Run inside container:
  HIP_VISIBLE_DEVICES=3 FLYDSL_RUNTIME_ENABLE_CACHE=0 \
  PYTHONPATH=/workspace/FlyDSL:$PYTHONPATH \
  python test/attention/fmha/bench_gap_table.py
"""
import math
import os
import re
import subprocess
import sys

import torch
import flydsl.compiler as flyc
from triton.testing import do_bench

sys.path.insert(0, ".")
from mslk.attention.flydsl.fmha_bwd_mfma import (
    compile_fmha_bwd_dvdk_mfma,
    compile_fmha_bwd_dq_mfma,
    compile_fmha_bwd_dqdkdv_mfma,
)
sys.path.insert(0, "test/attention/fmha")
from test_fmha_bwd_reference import ref_fmha_fwd, ref_fmha_bwd

CK_BWD_BIN = os.environ.get(
    "CK_BWD_BIN", "external/composable_kernel/build/bin/tile_example_fmha_bwd")
GPU = os.environ.get("HIP_VISIBLE_DEVICES", "3")
_CK_RE = re.compile(r"([0-9.]+)\s*ms,\s*([0-9.]+)\s*TFlops")


def _i16(t):
    return t.view(torch.int16)


def _mk(B, M, N, H, D, dtype, device="cuda"):
    scale = 1.0 / math.sqrt(D)
    Q = torch.randn(B, M, H, D, device=device, dtype=dtype)
    K = torch.randn(B, N, H, D, device=device, dtype=dtype)
    V = torch.randn(B, N, H, D, device=device, dtype=dtype)
    O, LSE = ref_fmha_fwd(Q, K, V, scale=scale, causal=False)
    dO = torch.randn_like(O)
    dQr, dKr, dVr = ref_fmha_bwd(Q, K, V, O, dO, LSE, scale=scale, causal=False)
    d = dict(
        Q2=_i16(Q.contiguous().view(B * M * H, D)),
        K2=_i16(K.contiguous().view(B * N * H, D)),
        V2=_i16(V.contiguous().view(B * N * H, D)),
        dO2=_i16(dO.contiguous().view(B * M * H, D)),
        LSE2=LSE.contiguous().view(B * H * M, 1),
        Dv=(dO.float() * O.float()).sum(-1).contiguous().view(B * M * H, 1),
        dQr=dQr, dKr=dKr, dVr=dVr, scale=scale, dtype=dtype,
    )
    return d


def _ok(out, ref, B, S, H, D, dtype):
    return torch.allclose(out.view(B, S, H, D).float(), ref.float(), rtol=0.1, atol=0.1)


def run_split(d, B, M, N, H, D):
    dev = "cuda"; st = torch.cuda.current_stream()
    ds = "bf16" if d["dtype"] == torch.bfloat16 else "f16"
    dv = torch.zeros(B*N*H*D, 1, device=dev); dk = torch.zeros(B*N*H*D, 1, device=dev)
    dq = torch.zeros(B*M*H*D, 1, device=dev)
    BM = 128
    dvdk = compile_fmha_bwd_dvdk_mfma(D=D, dtype_str=ds, scale=d["scale"], BLOCK_M=BM, use_pipeline=True)
    a_dvdk = (d["Q2"], d["K2"], d["V2"], d["dO2"], dv, dk, d["LSE2"], d["Dv"], B, M, N, H, (M+BM-1)//BM, st)
    dqf = compile_fmha_bwd_dq_mfma(D=D, dtype_str=ds, scale=d["scale"], use_pipeline=True)
    a_dq = (d["Q2"], d["K2"], d["V2"], d["dO2"], dq, d["LSE2"], d["Dv"], B, M, N, H, (N+63)//64, st)
    cd = flyc.compile(dvdk, *a_dvdk); cq = flyc.compile(dqf, *a_dq)
    cd(*a_dvdk); cq(*a_dq); torch.cuda.synchronize()
    ok = (_ok(dv, d["dVr"], B, N, H, D, d["dtype"]) and _ok(dk, d["dKr"], B, N, H, D, d["dtype"])
          and _ok(dq, d["dQr"], B, M, H, D, d["dtype"]))
    ms = do_bench(lambda: (cd(*a_dvdk), cq(*a_dq)), warmup=10, rep=50)
    return ms, ok


def run_fused(d, B, M, N, H, D, lds_reduce):
    dev = "cuda"; st = torch.cuda.current_stream()
    ds = "bf16" if d["dtype"] == torch.bfloat16 else "f16"
    dv = torch.zeros(B*N*H*D, 1, device=dev); dk = torch.zeros(B*N*H*D, 1, device=dev)
    dq = torch.zeros(B*M*H*D, 1, device=dev)
    nM = (M + 63) // 64
    fn = compile_fmha_bwd_dqdkdv_mfma(D=D, dtype_str=ds, scale=d["scale"], use_lds_reduce=lds_reduce)
    ac = (d["Q2"], d["K2"], d["V2"], d["dO2"], torch.zeros_like(dv), torch.zeros_like(dk),
          torch.zeros_like(dq), d["LSE2"], d["Dv"], B, M, N, H, nM, st)
    c = flyc.compile(fn, *ac)
    ra = (d["Q2"], d["K2"], d["V2"], d["dO2"], dv, dk, dq, d["LSE2"], d["Dv"], B, M, N, H, nM, st)
    dq.zero_(); c(*ra); torch.cuda.synchronize()
    ok = (_ok(dv, d["dVr"], B, N, H, D, d["dtype"]) and _ok(dk, d["dKr"], B, N, H, D, d["dtype"])
          and _ok(dq, d["dQr"], B, M, H, D, d["dtype"]))
    ms = do_bench(lambda: (dq.zero_(), c(*ra)), warmup=10, rep=50)
    return ms, ok


def run_ck(B, M, N, H, D, dtype):
    if not os.path.exists(CK_BWD_BIN):
        return None
    prec = "bf16" if dtype == torch.bfloat16 else "fp16"
    cmd = [CK_BWD_BIN, f"-prec={prec}", f"-b={B}", f"-h={H}", f"-s={M}", f"-s_k={N}", f"-d={D}", "-v=1"]
    env = dict(os.environ, HIP_VISIBLE_DEVICES=GPU)
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=180, env=env)
    except Exception as e:
        return ("ERR:" + type(e).__name__,)
    m = _CK_RE.search(out.stdout + out.stderr)
    return (float(m.group(1)), float(m.group(2))) if m else ("PARSE?",)


if __name__ == "__main__":
    cases = [
        (1,  256,  256,  8, 64, torch.bfloat16),
        (1,  512,  512,  8, 64, torch.bfloat16),
        (1, 1024, 1024,  8, 64, torch.bfloat16),
        (1, 2048, 2048,  8, 64, torch.bfloat16),
        (2, 2048, 2048, 16, 64, torch.bfloat16),
        (1, 4096, 4096,  8, 64, torch.bfloat16),
    ]
    print(f"CK bin: {'found' if os.path.exists(CK_BWD_BIN) else 'MISSING'}")
    hdr = (f"{'shape':>22} | {'split':>8} {'fused':>8} {'fusedR':>8} | {'best':>8} "
           f"{'oks':>4} | {'CK':>8} | {'CK/best':>8}")
    print(hdr); print("-" * len(hdr))
    for B, M, N, H, D, dt in cases:
        d = _mk(B, M, N, H, D, dt)
        sp, ok0 = run_split(d, B, M, N, H, D)
        fu, ok1 = run_fused(d, B, M, N, H, D, False)
        fr, ok2 = run_fused(d, B, M, N, H, D, True)
        ck = run_ck(B, M, N, H, D, dt)
        best = min(sp, fu, fr)
        oks = f"{int(ok0)}{int(ok1)}{int(ok2)}"
        tag = f"B{B} M{M} H{H}"
        if ck and isinstance(ck[0], float):
            ck_ms = ck[0]
            print(f"{tag:>22} | {sp:8.4f} {fu:8.4f} {fr:8.4f} | {best:8.4f} {oks:>4} | "
                  f"{ck_ms:8.4f} | {ck_ms/best:7.2f}x")
        else:
            print(f"{tag:>22} | {sp:8.4f} {fu:8.4f} {fr:8.4f} | {best:8.4f} {oks:>4} | "
                  f"{str(ck):>8} | {'--':>8}")
    print("\nsplit=pipelined dvdk+dq (2 kernels); fused=dqdkdv atomic; fusedR=+lds_reduce.")
    print("CK/best>1 => CK slower than our best; <1 => CK faster. oks=split,fused,fusedR.")
