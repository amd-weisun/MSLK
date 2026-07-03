"""A/B benchmark: dvdk baseline (scalar gather) vs trload (ds_read_tr) B-operand.

Same shapes as bench_fmha_bwd_vs_ck. Reports dvdk_ms for each and the speedup.
"""
import math
import sys

import torch

import flydsl.compiler as flyc
from mslk.attention.flydsl.fmha_bwd_mfma import compile_fmha_bwd_dvdk_mfma

sys.path.insert(0, "/workspace/MSLK/test/attention/fmha")
from test_fmha_bwd_reference import ref_fmha_bwd, ref_fmha_fwd  # noqa: E402

try:
    from triton.testing import do_bench
except Exception:
    do_bench = None


def _as_i16(t):
    return t.view(torch.int16)


def _bench(fn):
    if do_bench is not None:
        return do_bench(fn, warmup=10, rep=50)
    # fallback: crude timing
    import time
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / 50 * 1e3


def run(B, M, N, H, D, dtype, use_trload, use_pipeline=False, device="cuda"):
    dtype_str = "bf16" if dtype == torch.bfloat16 else "f16"
    scale = 1.0 / math.sqrt(D)
    Q = torch.randn(B, M, H, D, device=device, dtype=dtype)
    K = torch.randn(B, N, H, D, device=device, dtype=dtype)
    V = torch.randn(B, N, H, D, device=device, dtype=dtype)
    O, LSE = ref_fmha_fwd(Q, K, V, scale=scale, causal=False)
    dO = torch.randn_like(O)
    _, dK_ref, dV_ref = ref_fmha_bwd(Q, K, V, O, dO, LSE, scale=scale, causal=False)

    Q2 = _as_i16(Q.contiguous().view(B * M * H, D))
    K2 = _as_i16(K.contiguous().view(B * N * H, D))
    V2 = _as_i16(V.contiguous().view(B * N * H, D))
    dO2 = _as_i16(dO.contiguous().view(B * M * H, D))
    LSE2 = LSE.contiguous().view(B * H * M, 1)
    Dv = (dO.float() * O.float()).sum(-1).contiguous().view(B * M * H, 1)
    o_dv = torch.zeros(B * N * H * D, 1, device=device, dtype=torch.float32)
    o_dk = torch.zeros(B * N * H * D, 1, device=device, dtype=torch.float32)
    st = torch.cuda.current_stream()
    BM = 128
    nM = (M + BM - 1) // BM

    dvdk = compile_fmha_bwd_dvdk_mfma(D=D, dtype_str=dtype_str, scale=scale,
                                      BLOCK_M=BM, use_trload=use_trload,
                                      use_pipeline=use_pipeline)
    a = (Q2, K2, V2, dO2, o_dv, o_dk, LSE2, Dv, B, M, N, H, nM, st)
    c = flyc.compile(dvdk, *a)
    c(*a)
    torch.cuda.synchronize()
    ok = (torch.allclose(o_dv.view(B, N, H, D).float(), dV_ref.float(), rtol=0.1, atol=0.1)
          and torch.allclose(o_dk.view(B, N, H, D).float(), dK_ref.float(), rtol=0.1, atol=0.1))
    ms = _bench(lambda: c(*a))
    return ms, ok


if __name__ == "__main__":
    cases = [
        (1, 512, 512, 8, 64, torch.bfloat16),
        (1, 1024, 1024, 8, 64, torch.bfloat16),
        (2, 2048, 2048, 16, 64, torch.bfloat16),
    ]
    hdr = (f"{'shape':>22} | {'base_ms':>9} {'trload':>9} {'pipe':>9} "
           f"{'pipe+tr':>9} | {'best_x':>7} | ok")
    print(hdr)
    for B, M, N, H, D, dt in cases:
        base_ms, ok0 = run(B, M, N, H, D, dt, use_trload=False)
        tr_ms, ok1 = run(B, M, N, H, D, dt, use_trload=True)
        pp_ms, ok2 = run(B, M, N, H, D, dt, use_trload=False, use_pipeline=True)
        pt_ms, ok3 = run(B, M, N, H, D, dt, use_trload=True, use_pipeline=True)
        tag = f"B{B} M{M} N{N} H{H}"
        best = min(tr_ms, pp_ms, pt_ms)
        spd = base_ms / best if best else 0
        ok = ok0 and ok1 and ok2 and ok3
        print(f"{tag:>22} | {base_ms:9.4f} {tr_ms:9.4f} {pp_ms:9.4f} "
              f"{pt_ms:9.4f} | {spd:6.2f}x | {ok}")
