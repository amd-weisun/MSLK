"""WP-A3 — FlyDSL vs CK backward comparison harness.

Runs, on matched shapes:
  * FlyDSL MFMA backward  : fused dV+dK (compile_fmha_bwd_dvdk_mfma) + dQ
    (compile_fmha_bwd_dq_mfma), timed with flydsl do_bench, validated vs the
    torch reference (ref_fmha_bwd).
  * CK ck_tile backward   : the standalone `tile_example_fmha_bwd` binary, which
    reports its own ms/TFlops and self-validates on CPU (valid:y/n).

Comparison policy
-----------------
CK and FlyDSL use different memory layouts (CK default iperm=1 -> b*h*s*d; our
kernels/ref use b*s*h*d) and different dropout RNG, so we do NOT numerically
cross-check CK vs FlyDSL. Instead each is INDEPENDENTLY validated against its own
reference, and we compare wall-clock + TFlops. TFlops for FlyDSL is derived using
CK's own FLOP count (backed out of CK's ms*TFlops) so both columns use an
identical FLOP definition.

This script must run INSIDE the container (needs GPU + FlyDSL + the CK binary).
Kept in the repo as the source of truth; push to the remote to run:
  HIP_VISIBLE_DEVICES=3 PYTHONPATH=/workspace/FlyDSL:$PYTHONPATH \
  python test/attention/fmha/bench_fmha_bwd_vs_ck.py

Optional env:
  CK_BWD_BIN  path to tile_example_fmha_bwd
              (default: external/composable_kernel/build/bin/tile_example_fmha_bwd)
  HIP_VISIBLE_DEVICES  which GPU (default 3)
"""
import math
import os
import re
import subprocess
import sys

import torch
import flydsl.compiler as flyc
from flydsl.autotune import do_bench

sys.path.insert(0, ".")
from mslk.attention.flydsl.fmha_bwd_mfma import (
    compile_fmha_bwd_dvdk_mfma,
    compile_fmha_bwd_dq_mfma,
)
sys.path.insert(0, "test/attention/fmha")
from test_fmha_bwd_reference import ref_fmha_fwd, ref_fmha_bwd

CK_BWD_BIN = os.environ.get(
    "CK_BWD_BIN",
    "external/composable_kernel/build/bin/tile_example_fmha_bwd",
)
GPU = os.environ.get("HIP_VISIBLE_DEVICES", "3")


def _as_i16(t):
    return t.view(torch.int16)


# ---------------------------------------------------------------- FlyDSL side
def run_flydsl(B, M, N, H, D, dtype, device="cuda"):
    """Return (total_ms, dvdk_ms, dq_ms, ok) for the FlyDSL backward."""
    dtype_str = "bf16" if dtype == torch.bfloat16 else "f16"
    scale = 1.0 / math.sqrt(D)
    Q = torch.randn(B, M, H, D, device=device, dtype=dtype)
    K = torch.randn(B, N, H, D, device=device, dtype=dtype)
    V = torch.randn(B, N, H, D, device=device, dtype=dtype)
    O, LSE = ref_fmha_fwd(Q, K, V, scale=scale, causal=False)
    dO = torch.randn_like(O)
    dQ_ref, dK_ref, dV_ref = ref_fmha_bwd(Q, K, V, O, dO, LSE, scale=scale, causal=False)

    Q2 = _as_i16(Q.contiguous().view(B * M * H, D))
    K2 = _as_i16(K.contiguous().view(B * N * H, D))
    V2 = _as_i16(V.contiguous().view(B * N * H, D))
    dO2 = _as_i16(dO.contiguous().view(B * M * H, D))
    LSE2 = LSE.contiguous().view(B * H * M, 1)
    Dv = (dO.float() * O.float()).sum(-1).contiguous().view(B * M * H, 1)
    o_dv = torch.zeros(B * N * H * D, 1, device=device, dtype=torch.float32)
    o_dk = torch.zeros(B * N * H * D, 1, device=device, dtype=torch.float32)
    o_dq = torch.zeros(B * M * H * D, 1, device=device, dtype=torch.float32)
    st = torch.cuda.current_stream()
    BM, BN = 128, 64
    nM = (M + BM - 1) // BM
    nN = (N + BN - 1) // BN

    dvdk = compile_fmha_bwd_dvdk_mfma(D=D, dtype_str=dtype_str, scale=scale, BLOCK_M=BM)
    dq = compile_fmha_bwd_dq_mfma(D=D, dtype_str=dtype_str, scale=scale)
    a_dvdk = (Q2, K2, V2, dO2, o_dv, o_dk, LSE2, Dv, B, M, N, H, nM, st)
    a_dq = (Q2, K2, V2, dO2, o_dq, LSE2, Dv, B, M, N, H, nN, st)
    c_dvdk = flyc.compile(dvdk, *a_dvdk)
    c_dq = flyc.compile(dq, *a_dq)

    # correctness
    c_dvdk(*a_dvdk)
    c_dq(*a_dq)
    torch.cuda.synchronize()
    dV_k = o_dv.view(B, N, H, D).to(dtype)
    dK_k = o_dk.view(B, N, H, D).to(dtype)
    dQ_k = o_dq.view(B, M, H, D).to(dtype)
    ok = (torch.allclose(dV_k.float(), dV_ref.float(), rtol=0.1, atol=0.1)
          and torch.allclose(dK_k.float(), dK_ref.float(), rtol=0.1, atol=0.1)
          and torch.allclose(dQ_k.float(), dQ_ref.float(), rtol=0.1, atol=0.1))

    dvdk_ms = do_bench(lambda: c_dvdk(*a_dvdk), warmup=10, rep=50)
    dq_ms = do_bench(lambda: c_dq(*a_dq), warmup=10, rep=50)
    total_ms = do_bench(lambda: (c_dvdk(*a_dvdk), c_dq(*a_dq)), warmup=10, rep=50)
    return total_ms, dvdk_ms, dq_ms, ok


# -------------------------------------------------------------------- CK side
_CK_RE = re.compile(r"([0-9.]+)\s*ms,\s*([0-9.]+)\s*TFlops,\s*([0-9.]+)\s*GB/s,\s*valid:(\w)")


def run_ck(B, M, N, H, D, dtype):
    """Return (ms, tflops, gbps, valid) parsed from the CK bench, or None."""
    prec = "bf16" if dtype == torch.bfloat16 else "fp16"
    if not os.path.exists(CK_BWD_BIN):
        return None
    cmd = [CK_BWD_BIN, f"-prec={prec}", f"-b={B}", f"-h={H}",
           f"-s={M}", f"-s_k={N}", f"-d={D}", "-v=1"]
    env = dict(os.environ, HIP_VISIBLE_DEVICES=GPU)
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
    except Exception as e:
        return ("ERR", str(type(e).__name__), "", "?")
    m = _CK_RE.search(out.stdout + out.stderr)
    if not m:
        return ("PARSE?", (out.stdout + out.stderr).strip().splitlines()[-1:], "", "?")
    ms, tflops, gbps, valid = m.groups()
    return (float(ms), float(tflops), float(gbps), valid)


def _fmt(v):
    return f"{v:8.3f}" if isinstance(v, float) else f"{str(v):>8}"


if __name__ == "__main__":
    device = "cuda"
    cases = [
        (1,  512,  512,  8, 64, torch.bfloat16),
        (1, 1024, 1024,  8, 64, torch.bfloat16),
        (2, 2048, 2048, 16, 64, torch.bfloat16),
        (1, 1024, 1024,  8, 64, torch.float16),
    ]
    print("=== FlyDSL vs CK backward (gfx950) ===")
    print(f"CK bin: {CK_BWD_BIN} ({'found' if os.path.exists(CK_BWD_BIN) else 'MISSING'})")
    hdr = (f"{'shape':>24} | {'fly_tot':>8} {'fly_dvdk':>8} {'fly_dq':>8} {'ok':>3} | "
           f"{'ck_ms':>8} {'ck_TFl':>7} {'ck_v':>4} | {'fly_TFl':>7} {'CK/fly':>7}")
    print(hdr); print("-" * len(hdr))
    for B, M, N, H, D, dt in cases:
        ftot, fdvdk, fdq, ok = run_flydsl(B, M, N, H, D, dt, device)
        ck = run_ck(B, M, N, H, D, dt)
        tag = f"B{B} M{M} N{N} H{H} {('bf16' if dt==torch.bfloat16 else 'f16')}"
        if ck and isinstance(ck[0], float):
            ck_ms, ck_tfl, _, ck_v = ck
            flop = ck_tfl * ck_ms  # flop/1e9  (CK: tflops = flop/1e9/ms)
            fly_tfl = flop / ftot if ftot else 0.0
            ratio = ck_ms / ftot if ftot else 0.0  # >1 => FlyDSL slower
            print(f"{tag:>24} | {_fmt(ftot)} {_fmt(fdvdk)} {_fmt(fdq)} {str(ok):>3} | "
                  f"{_fmt(ck_ms)} {_fmt(ck_tfl)} {ck_v:>4} | {_fmt(fly_tfl)} {ratio:6.2f}x")
        else:
            print(f"{tag:>24} | {_fmt(ftot)} {_fmt(fdvdk)} {_fmt(fdq)} {str(ok):>3} | "
                  f"{'CK:':>8} {str(ck):>20}")
    print("\nfly_tot = dvdk_fused + dq (median ms, ex-preprocess). CK/fly = ck_ms/fly_tot "
          "(<1 => CK faster). fly_TFl uses CK's FLOP count for an identical metric.")
