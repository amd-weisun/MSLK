"""WP-A3 Phase B.4 — test: FUSED dV+dK MFMA kernel vs ref_fmha_bwd.

Usage (inside container):
    cd /workspace/MSLK
    HIP_VISIBLE_DEVICES=3 FLYDSL_RUNTIME_ENABLE_CACHE=0 \
    PYTHONPATH=/workspace/FlyDSL:$PYTHONPATH \
    python -m pytest test/attention/fmha/test_fmha_bwd_dvdk_mfma.py -v
"""
import math
import sys
import pytest
import torch
import flydsl.compiler as flyc

sys.path.insert(0, ".")
from mslk.attention.flydsl.fmha_bwd_mfma import compile_fmha_bwd_dvdk_mfma
sys.path.insert(0, "test/attention/fmha")
from test_fmha_bwd_reference import ref_fmha_bwd, ref_fmha_fwd

rocm_only = pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.version.hip, reason="requires ROCm GPU"
)


def _as_i16(t):
    return t.view(torch.int16)


def run_case(B, M, N, H, D, dtype, device="cuda", use_trload=False, use_pipeline=False,
             causal=False, packed_qkv=False, H_kv=None):
    tag = ""
    if use_trload:
        tag += " [trload]"
    if use_pipeline:
        tag += " [pipeline]"
    if causal:
        tag += " [causal]"
    if packed_qkv:
        tag += " [packed_qkv]"
    if H_kv is not None and H_kv != H:
        tag += f" [gqa Hkv={H_kv}]"
    print(f"  B={B} M={M} N={N} H={H} D={D} dtype={dtype}{tag}", end=" ... ", flush=True)
    if use_trload and "gfx950" not in torch.cuda.get_device_properties(device).gcnArchName:
        pytest.skip("use_trload requires gfx950 (ds_read_tr16_b64 is CDNA4-only)")
    if packed_qkv:
        assert M == N, "packed_qkv requires Q and K/V to share seqlen (stacked on a shared dim)"
    if H_kv is not None:
        assert not packed_qkv, "GQA and packed_qkv are independent axes, not tested combined"
        assert H % H_kv == 0, f"H={H} must be a multiple of H_kv={H_kv}"
    scale = 1.0 / math.sqrt(D)
    Hkv = H if H_kv is None else H_kv

    if packed_qkv:
        # Sequencing-plan item 7: qkv = stack([Q,K,V], dim=2), unbind -> non-contiguous
        # Q/K/V views (row pitch 3*H*D instead of H*D), mirrors test_backward.py's
        # dominant non-contiguous-BMHK case.
        qkv = torch.stack(
            [torch.randn(B, M, H, D, device=device, dtype=dtype) for _ in range(3)], dim=2
        )
        Q, K, V = qkv.unbind(2)
        assert not Q.is_contiguous()
    else:
        Q  = torch.randn(B, M, H, D, device=device, dtype=dtype)
        # GQA (sequencing-plan item 4): a genuinely distinctly-shaped (B,N,Hkv,D)
        # tensor (contiguous, stride(2)==D) -- NOT the MQA-via-broadcast case
        # (that's exercised separately by test_backward_gqa itself).
        K  = torch.randn(B, N, Hkv, D, device=device, dtype=dtype)
        V  = torch.randn(B, N, Hkv, D, device=device, dtype=dtype)
    O, LSE = ref_fmha_fwd(Q, K, V, scale=scale, causal=causal)
    dO = torch.randn_like(O)
    _, dK_ref, dV_ref = ref_fmha_bwd(Q, K, V, O, dO, LSE, scale=scale, causal=causal)

    gpu_arch = torch.cuda.get_device_properties(device).gcnArchName
    # BLOCK_M=128's Q/dO/P/dS tiles exceed gfx950's 160KB LDS limit at D=256
    # (measured 164KB vs 83KB at BLOCK_M=64) -- drop BLOCK_M there, same mitigation
    # CK's own codegen uses (shrinks its M-tile at D>=128). gfx942's 64KB LDS is
    # tighter still -- BLOCK_M=64 overflows too at D=256 (83KB), drop to 32 there
    # (see flydsl.py for the same logic, verified on real gfx942 hardware).
    if "gfx950" in gpu_arch:
        BLOCK_M = 64 if D >= 256 else 128
    else:
        BLOCK_M = 32 if D >= 256 else 64
    BLOCK_N = 64
    dtype_str = "bf16" if dtype == torch.bfloat16 else "f16"

    Q_4d  = _as_i16(Q)
    K_4d  = _as_i16(K)
    V_4d  = _as_i16(V)
    dO_4d = _as_i16(dO.contiguous())
    q_stride_m  = Q.stride(1) // D
    kv_stride_n = K.stride(1) // D
    do_stride_m = dO_4d.stride(1) // D
    LSE_2d = LSE.contiguous().view(B * H * M, 1)
    D_vec  = (dO.float() * O.float()).sum(dim=-1).contiguous().view(B * M * H, 1)
    dV_out = torch.zeros(B * N * Hkv * D, 1, device=device, dtype=torch.float32)
    dK_out = torch.zeros(B * N * Hkv * D, 1, device=device, dtype=torch.float32)

    launch_fn = compile_fmha_bwd_dvdk_mfma(D=D, dtype_str=dtype_str,
                                            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, scale=scale,
                                            use_trload=use_trload, use_pipeline=use_pipeline,
                                            heads_per_kv=H // Hkv,
                                            gpu_arch=gpu_arch, causal=causal)
    n_M_tiles = (M + BLOCK_M - 1) // BLOCK_M
    args = (Q_4d, K_4d, V_4d, dO_4d, dV_out, dK_out, LSE_2d, D_vec,
            B, M, N, H, n_M_tiles, q_stride_m, kv_stride_n, do_stride_m,
            torch.cuda.current_stream())
    compiled = flyc.compile(launch_fn, *args)
    compiled(*args)
    torch.cuda.synchronize()

    dV_k = dV_out.view(B, N, Hkv, D).to(dtype)
    dK_k = dK_out.view(B, N, Hkv, D).to(dtype)
    err_v = (dV_k.float() - dV_ref.float()).abs().max().item()
    err_k = (dK_k.float() - dK_ref.float()).abs().max().item()
    ok_v = torch.allclose(dV_k.float(), dV_ref.float(), rtol=0.1, atol=0.1)
    ok_k = torch.allclose(dK_k.float(), dK_ref.float(), rtol=0.1, atol=0.1)
    ok = ok_v and ok_k
    print(f"{'PASS' if ok else 'FAIL'}  dV_err={err_v:.5f} dK_err={err_k:.5f}")
    return ok


CASES = [
    (1,  128,  64,  1,  64,  torch.bfloat16),   # M=BLOCK_M
    (1,  256,  64,  1,  64,  torch.bfloat16),   # M=2*BLOCK_M
    (1,  128, 128,  1,  64,  torch.bfloat16),
    (1,  256, 128,  8,  64,  torch.bfloat16),
    (2,  512, 128,  8,  64,  torch.bfloat16),
    (1,  256, 128,  8,  64,  torch.float16),
]

# D=128/256 (sequencing-plan step 2, head-dim generalization -- D a multiple of 64;
# each wave sequentially loops D_SUBS_PER_WAVE 32-col subtiles, see fmha_bwd_mfma.py).
CASES_WIDE_D = [
    (1,  128,  64,  1,  128, torch.bfloat16),
    (1,  256, 128,  8,  128, torch.bfloat16),
    (2,  512, 128,  8,  128, torch.bfloat16),
    (1,  256, 128,  8,  128, torch.float16),
    (1,  128,  64,  1,  256, torch.bfloat16),
    (1,  256, 128,  8,  256, torch.bfloat16),
    (2,  512, 128,  8,  256, torch.bfloat16),
    (1,  256, 128,  8,  256, torch.float16),
]

# D=32/96 (sequencing-plan step 2 follow-up -- D a multiple of 32 but not 64;
# ceil-div wave assignment + out-of-range guards, see fmha_bwd_mfma.py's
# D_SUBS_PER_WAVE comment).
CASES_ODD_D = [
    (1,  128,  64,  1,  32,  torch.bfloat16),
    (1,  256, 128,  8,  32,  torch.bfloat16),
    (1,  256, 128,  8,  32,  torch.float16),
    (1,  128,  64,  1,  96,  torch.bfloat16),
    (1,  256, 128,  8,  96,  torch.bfloat16),
    (1,  256, 128,  8,  96,  torch.float16),
]

# Causal masking (sequencing-plan item 3), across the D matrix.
CASES_CAUSAL = [
    (1,  128,  64,  1,  64,  torch.bfloat16),
    (1,  256, 128,  8,  64,  torch.bfloat16),
    (1,  256, 128,  8,  128, torch.bfloat16),
    (1,  256, 128,  8,  256, torch.bfloat16),
    (1,  256, 128,  8,  32,  torch.bfloat16),
    (1,  256, 128,  8,  96,  torch.bfloat16),
]

# Stride-aware addressing (sequencing-plan item 7) -- packed-qkv unbind view
# (M==N required: Q/K/V share seqlen when stacked on a shared dim).
CASES_PACKED_QKV = [
    (1,  128, 128,  1,  64,  torch.bfloat16),
    (1,  256, 256,  8,  64,  torch.bfloat16),
    (1,  256, 256,  8,  128, torch.bfloat16),
    (1,  256, 256,  8,  32,  torch.bfloat16),
    (1,  256, 256,  8,  96,  torch.bfloat16),
    (1,  256, 256,  8,  64,  torch.float16),
]

# GQA (sequencing-plan item 4) -- genuinely distinctly-shaped (B,N,Hkv,D) K/V
# (contiguous, Hkv < Hq, Hq % Hkv == 0). (B, M, N, H=Hq, D, dtype, H_kv).
CASES_GQA = [
    (1,  128,  64,  2,  64,  torch.bfloat16, 1),   # MQA (Hkv=1)
    (1,  128,  64,  8,  64,  torch.bfloat16, 2),
    (1,  256, 128,  8,  64,  torch.bfloat16, 4),
    (2,  512, 128,  8,  64,  torch.bfloat16, 2),
    (1,  256, 128,  8,  128, torch.bfloat16, 2),
    (1,  256, 128,  8,  32,  torch.bfloat16, 2),
    (1,  256, 128,  8,  96,  torch.bfloat16, 4),
    (1,  256, 128,  8,  64,  torch.float16,  2),
    # heads_per_kv==1 (Hkv==Hq) degenerate check: must match non-GQA behavior.
    (1,  256, 128,  8,  64,  torch.bfloat16, 8),
]


@rocm_only
@pytest.mark.parametrize("use_trload,use_pipeline", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("B,M,N,H,D,dtype", CASES)
def test_dvdk_mfma(B, M, N, H, D, dtype, use_trload, use_pipeline):
    assert run_case(B, M, N, H, D, dtype, device="cuda", use_trload=use_trload, use_pipeline=use_pipeline)


@rocm_only
@pytest.mark.parametrize("use_trload,use_pipeline", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("B,M,N,H,D,dtype", CASES_WIDE_D)
def test_dvdk_mfma_wide_d(B, M, N, H, D, dtype, use_trload, use_pipeline):
    assert run_case(B, M, N, H, D, dtype, device="cuda", use_trload=use_trload, use_pipeline=use_pipeline)


@rocm_only
@pytest.mark.parametrize("use_pipeline", [False, True])
@pytest.mark.parametrize("B,M,N,H,D,dtype", CASES_ODD_D)
def test_dvdk_mfma_odd_d(B, M, N, H, D, dtype, use_pipeline):
    # use_trload not exercised here: it's a separate HW-transpose feature, orthogonal
    # to the D=32/96 remainder-tile logic under test.
    assert run_case(B, M, N, H, D, dtype, device="cuda", use_pipeline=use_pipeline)


@rocm_only
@pytest.mark.parametrize("use_pipeline", [False, True])
@pytest.mark.parametrize("B,M,N,H,D,dtype", CASES_CAUSAL)
def test_dvdk_mfma_causal(B, M, N, H, D, dtype, use_pipeline):
    assert run_case(B, M, N, H, D, dtype, device="cuda", use_pipeline=use_pipeline, causal=True)


@rocm_only
@pytest.mark.parametrize("use_pipeline", [False, True])
@pytest.mark.parametrize("B,M,N,H,D,dtype", CASES_PACKED_QKV)
def test_dvdk_mfma_packed_qkv(B, M, N, H, D, dtype, use_pipeline):
    assert run_case(B, M, N, H, D, dtype, device="cuda", use_pipeline=use_pipeline, packed_qkv=True)


@rocm_only
@pytest.mark.parametrize("use_pipeline", [False, True])
@pytest.mark.parametrize("B,M,N,H,D,dtype,H_kv", CASES_GQA)
def test_dvdk_mfma_gqa(B, M, N, H, D, dtype, H_kv, use_pipeline):
    assert run_case(B, M, N, H, D, dtype, device="cuda", use_pipeline=use_pipeline, H_kv=H_kv)


if __name__ == "__main__":
    device = "cuda"
    all_pass = True
    print("=== FUSED dV+dK MFMA kernel (Phase B.4) vs ref_fmha_bwd ===")
    for args in CASES:
        all_pass &= run_case(*args, device=device)
    trload = "--trload" in sys.argv or "--all" in sys.argv
    if trload:
        print("\n=== trload (ds_read_tr) variant ===")
        for args in CASES:
            all_pass &= run_case(*args, device=device, use_trload=True)
    pipeline = "--pipeline" in sys.argv or "--all" in sys.argv
    if pipeline:
        print("\n=== pipeline (lane-distributed coop load) variant ===")
        for args in CASES:
            all_pass &= run_case(*args, device=device, use_pipeline=True)
        print("\n=== pipeline + trload variant ===")
        for args in CASES:
            all_pass &= run_case(*args, device=device, use_trload=True, use_pipeline=True)
    print(f"\n{'ALL PASSED' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)
