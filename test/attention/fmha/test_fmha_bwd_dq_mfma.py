"""WP-A3 Phase B.3 — test: MFMA dQ kernel vs ref_fmha_bwd.

Usage (inside container):
    cd /workspace/MSLK
    HIP_VISIBLE_DEVICES=3 FLYDSL_RUNTIME_ENABLE_CACHE=0 \
    PYTHONPATH=/workspace/FlyDSL:$PYTHONPATH \
    python -m pytest test/attention/fmha/test_fmha_bwd_dq_mfma.py -v
"""
import math
import sys
import pytest
import torch
import flydsl.compiler as flyc

sys.path.insert(0, ".")
from mslk.attention.flydsl.fmha_bwd_mfma import compile_fmha_bwd_dq_mfma
sys.path.insert(0, "test/attention/fmha")
from test_fmha_bwd_reference import ref_fmha_bwd, ref_fmha_fwd

rocm_only = pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.version.hip, reason="requires ROCm GPU"
)


def _as_i16(t):
    assert t.dtype in (torch.bfloat16, torch.float16)
    return t.view(torch.int16)


def run_case(B, M, N, H, D, dtype, device="cuda", use_pipeline=False, causal=False, packed_qkv=False,
             H_kv=None):
    tag = " [pipeline]" if use_pipeline else ""
    if causal:
        tag += " [causal]"
    if packed_qkv:
        tag += " [packed_qkv]"
    if H_kv is not None and H_kv != H:
        tag += f" [gqa Hkv={H_kv}]"
    print(f"  B={B} M={M} N={N} H={H} D={D} dtype={dtype}{tag}", end=" ... ", flush=True)
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
        # GQA (sequencing-plan item 4): genuinely distinctly-shaped (B,N,Hkv,D) K/V.
        K  = torch.randn(B, N, Hkv, D, device=device, dtype=dtype)
        V  = torch.randn(B, N, Hkv, D, device=device, dtype=dtype)
    O, LSE = ref_fmha_fwd(Q, K, V, scale=scale, causal=causal)
    dO = torch.randn_like(O)
    dQ_ref, _, _ = ref_fmha_bwd(Q, K, V, O, dO, LSE, scale=scale, causal=causal)

    gpu_arch = torch.cuda.get_device_properties(device).gcnArchName
    # K/V/dS LDS footprint at BLOCK_N=64 is 24/40/72 KB for D=64/128/256 -- fits
    # gfx950's 160KB everywhere but overflows gfx942's 64KB at D=256 (72KB); drop to
    # BLOCK_N=32 (36KB) there, verified on real gfx942 hardware (see flydsl.py).
    BLOCK_M = 64
    BLOCK_N = 32 if ("gfx950" not in gpu_arch and D >= 256) else 64
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
    dQ_out = torch.zeros(B * M * H * D, 1, device=device, dtype=torch.float32)

    launch_fn = compile_fmha_bwd_dq_mfma(D=D, dtype_str=dtype_str,
                                          BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, scale=scale,
                                          use_pipeline=use_pipeline,
                                          heads_per_kv=H // Hkv,
                                          gpu_arch=gpu_arch, causal=causal)
    n_N_tiles = (N + BLOCK_N - 1) // BLOCK_N
    _dummy_seqstart = torch.zeros(1, device=device, dtype=torch.int32)
    args = (Q_4d, K_4d, V_4d, dO_4d, dQ_out, LSE_2d, D_vec,
            B, M, N, H, n_N_tiles, q_stride_m, kv_stride_n, do_stride_m,
            _dummy_seqstart, _dummy_seqstart, M,
            torch.cuda.current_stream())
    compiled = flyc.compile(launch_fn, *args)
    compiled(*args)
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


CASES = [
    (1,   64,  64,  1,  64,  torch.bfloat16),
    (1,  128,  64,  1,  64,  torch.bfloat16),
    (1,   64, 128,  1,  64,  torch.bfloat16),
    (1,  128, 128,  8,  64,  torch.bfloat16),
    (2,  256, 128,  8,  64,  torch.bfloat16),
    (1,  128, 128,  8,  64,  torch.float16),
]

# D=128/256 (sequencing-plan step 2, head-dim generalization -- D a multiple of 64;
# each wave sequentially loops D_SUBS_PER_WAVE 32-col subtiles, see fmha_bwd_mfma.py).
CASES_WIDE_D = [
    (1,   64,  64,  1,  128, torch.bfloat16),
    (1,  128, 128,  8,  128, torch.bfloat16),
    (2,  256, 128,  8,  128, torch.bfloat16),
    (1,  128, 128,  8,  128, torch.float16),
    (1,   64,  64,  1,  256, torch.bfloat16),
    (1,  128, 128,  8,  256, torch.bfloat16),
    (2,  256, 128,  8,  256, torch.bfloat16),
    (1,  128, 128,  8,  256, torch.float16),
]

# D=32/96 (sequencing-plan step 2 follow-up -- D a multiple of 32 but not 64;
# ceil-div wave assignment + out-of-range guards, see fmha_bwd_mfma.py's
# D_SUBS_PER_WAVE comment).
CASES_ODD_D = [
    (1,   64,  64,  1,  32,  torch.bfloat16),
    (1,  128, 128,  8,  32,  torch.bfloat16),
    (1,  128, 128,  8,  32,  torch.float16),
    (1,   64,  64,  1,  96,  torch.bfloat16),
    (1,  128, 128,  8,  96,  torch.bfloat16),
    (1,  128, 128,  8,  96,  torch.float16),
]

# Causal masking (sequencing-plan item 3), across the D matrix.
CASES_CAUSAL = [
    (1,   64,  64,  1,  64,  torch.bfloat16),
    (1,  128, 128,  8,  64,  torch.bfloat16),
    (1,  128, 128,  8,  128, torch.bfloat16),
    (1,  128, 128,  8,  256, torch.bfloat16),
    (1,  128, 128,  8,  32,  torch.bfloat16),
    (1,  128, 128,  8,  96,  torch.bfloat16),
]

# Stride-aware addressing (sequencing-plan item 7) -- packed-qkv unbind view
# (M==N required: Q/K/V share seqlen when stacked on a shared dim).
CASES_PACKED_QKV = [
    (1,  128, 128,  1,  64,  torch.bfloat16),
    (1,  128, 128,  8,  64,  torch.bfloat16),
    (1,  128, 128,  8,  128, torch.bfloat16),
    (1,  128, 128,  8,  32,  torch.bfloat16),
    (1,  128, 128,  8,  96,  torch.bfloat16),
    (1,  128, 128,  8,  64,  torch.float16),
]

# GQA (sequencing-plan item 4) -- genuinely distinctly-shaped (B,N,Hkv,D) K/V
# (contiguous, Hkv < Hq, Hq % Hkv == 0). (B, M, N, H=Hq, D, dtype, H_kv).
CASES_GQA = [
    (1,   64,  64,  2,  64,  torch.bfloat16, 1),   # MQA (Hkv=1)
    (1,  128, 128,  8,  64,  torch.bfloat16, 2),
    (2,  256, 128,  8,  64,  torch.bfloat16, 4),
    (1,  128, 128,  8,  128, torch.bfloat16, 2),
    (1,  128, 128,  8,  32,  torch.bfloat16, 2),
    (1,  128, 128,  8,  96,  torch.bfloat16, 4),
    (1,  128, 128,  8,  64,  torch.float16,  2),
    # heads_per_kv==1 (Hkv==Hq) degenerate check: must match non-GQA behavior.
    (1,  128, 128,  8,  64,  torch.bfloat16, 8),
]


@rocm_only
@pytest.mark.parametrize("use_pipeline", [False, True])
@pytest.mark.parametrize("B,M,N,H,D,dtype", CASES)
def test_dq_mfma(B, M, N, H, D, dtype, use_pipeline):
    assert run_case(B, M, N, H, D, dtype, device="cuda", use_pipeline=use_pipeline)


@rocm_only
@pytest.mark.parametrize("use_pipeline", [False, True])
@pytest.mark.parametrize("B,M,N,H,D,dtype", CASES_WIDE_D)
def test_dq_mfma_wide_d(B, M, N, H, D, dtype, use_pipeline):
    assert run_case(B, M, N, H, D, dtype, device="cuda", use_pipeline=use_pipeline)


@rocm_only
@pytest.mark.parametrize("use_pipeline", [False, True])
@pytest.mark.parametrize("B,M,N,H,D,dtype", CASES_ODD_D)
def test_dq_mfma_odd_d(B, M, N, H, D, dtype, use_pipeline):
    assert run_case(B, M, N, H, D, dtype, device="cuda", use_pipeline=use_pipeline)


@rocm_only
@pytest.mark.parametrize("use_pipeline", [False, True])
@pytest.mark.parametrize("B,M,N,H,D,dtype", CASES_CAUSAL)
def test_dq_mfma_causal(B, M, N, H, D, dtype, use_pipeline):
    assert run_case(B, M, N, H, D, dtype, device="cuda", use_pipeline=use_pipeline, causal=True)


@rocm_only
@pytest.mark.parametrize("use_pipeline", [False, True])
@pytest.mark.parametrize("B,M,N,H,D,dtype", CASES_PACKED_QKV)
def test_dq_mfma_packed_qkv(B, M, N, H, D, dtype, use_pipeline):
    assert run_case(B, M, N, H, D, dtype, device="cuda", use_pipeline=use_pipeline, packed_qkv=True)


@rocm_only
@pytest.mark.parametrize("use_pipeline", [False, True])
@pytest.mark.parametrize("B,M,N,H,D,dtype,H_kv", CASES_GQA)
def test_dq_mfma_gqa(B, M, N, H, D, dtype, H_kv, use_pipeline):
    assert run_case(B, M, N, H, D, dtype, device="cuda", use_pipeline=use_pipeline, H_kv=H_kv)


def run_case_varlen(seqlens_q, seqlens_k, H, D, dtype, device="cuda", use_pipeline=False,
                     causal=False):
    """Sequencing-plan item 6: BlockDiagonalMask-style varlen (Stage B1 non-causal,
    Stage B2 causal). Mirrors test_fmha_bwd_dvdk_mfma.py's run_case_varlen -- see
    there for design notes.
    """
    assert len(seqlens_q) == len(seqlens_k)
    B_logical = len(seqlens_q)
    tag = " [pipeline]" if use_pipeline else ""
    if causal:
        tag += " [causal]"
    print(f"  varlen B={B_logical} seqlens_q={seqlens_q} seqlens_k={seqlens_k} "
          f"H={H} D={D} dtype={dtype}{tag} ... ", end="", flush=True)
    scale = 1.0 / math.sqrt(D)

    Qs, Ks, Vs, Os, LSEs, dOs, dQ_refs = [], [], [], [], [], [], []
    for m, n in zip(seqlens_q, seqlens_k):
        q = torch.randn(1, m, H, D, device=device, dtype=dtype)
        k = torch.randn(1, n, H, D, device=device, dtype=dtype)
        v = torch.randn(1, n, H, D, device=device, dtype=dtype)
        o, lse = ref_fmha_fwd(q, k, v, scale=scale, causal=causal)
        do = torch.randn_like(o)
        dq_ref, _, _ = ref_fmha_bwd(q, k, v, o, do, lse, scale=scale, causal=causal)
        Qs.append(q); Ks.append(k); Vs.append(v); Os.append(o); LSEs.append(lse); dOs.append(do)
        dQ_refs.append(dq_ref)

    total_m = sum(seqlens_q)
    max_seqlen_q = max(seqlens_q)
    max_seqlen_k = max(seqlens_k)

    Q = torch.cat(Qs, dim=1)
    K = torch.cat(Ks, dim=1)
    V = torch.cat(Vs, dim=1)
    dO = torch.cat(dOs, dim=1)
    LSE = torch.cat(LSEs, dim=2)
    D_vec = torch.cat(
        [(do.float() * o.float()).sum(dim=-1) for do, o in zip(dOs, Os)], dim=1
    ).contiguous().view(total_m * H, 1)

    seqstart_q = torch.tensor(
        [0] + list(torch.tensor(seqlens_q).cumsum(0).tolist()), device=device, dtype=torch.int32
    )
    seqstart_k = torch.tensor(
        [0] + list(torch.tensor(seqlens_k).cumsum(0).tolist()), device=device, dtype=torch.int32
    )

    gpu_arch = torch.cuda.get_device_properties(device).gcnArchName
    BLOCK_M = 64
    BLOCK_N = 32 if ("gfx950" not in gpu_arch and D >= 256) else 64
    dtype_str = "bf16" if dtype == torch.bfloat16 else "f16"

    Q_4d = _as_i16(Q)
    K_4d = _as_i16(K)
    V_4d = _as_i16(V)
    dO_4d = _as_i16(dO.contiguous())
    q_stride_m = Q.stride(1) // D
    kv_stride_n = K.stride(1) // D
    do_stride_m = dO_4d.stride(1) // D
    LSE_2d = LSE.contiguous().view(H * total_m, 1)
    dQ_out = torch.zeros(total_m * H * D, 1, device=device, dtype=torch.float32)

    launch_fn = compile_fmha_bwd_dq_mfma(
        D=D, dtype_str=dtype_str, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, scale=scale,
        use_pipeline=use_pipeline, gpu_arch=gpu_arch, causal=causal, varlen=True,
    )
    n_N_tiles = (max_seqlen_k + BLOCK_N - 1) // BLOCK_N
    args = (
        Q_4d, K_4d, V_4d, dO_4d, dQ_out, LSE_2d, D_vec,
        B_logical, max_seqlen_q, max_seqlen_k, H, n_N_tiles,
        q_stride_m, kv_stride_n, do_stride_m,
        seqstart_q, seqstart_k, total_m,
        torch.cuda.current_stream(),
    )
    compiled = flyc.compile(launch_fn, *args)
    compiled(*args)
    torch.cuda.synchronize()

    dQ_kernel = dQ_out.view(1, total_m, H, D).to(dtype)
    dQ_ref = torch.cat(dQ_refs, dim=1)
    max_err = (dQ_kernel.float() - dQ_ref.float()).abs().max().item()
    ok = torch.allclose(dQ_kernel.float(), dQ_ref.float(), rtol=0.1, atol=0.1)
    print(f"{'PASS' if ok else 'FAIL'}  max_err={max_err:.5f}")
    return ok


# Varlen / group-mode (sequencing-plan item 6, Stage B1) -- non-causal
# BlockDiagonalMask only, hand-picked small seqlen sets before broadening.
CASES_VARLEN = [
    ([64, 128, 32], [64, 128, 32], 1, 64, torch.bfloat16),
    ([64, 128, 32], [64, 128, 32], 8, 64, torch.bfloat16),
    ([100, 50, 200], [100, 50, 200], 8, 64, torch.bfloat16),
    ([64, 128, 32], [64, 128, 32], 8, 128, torch.bfloat16),
]


@rocm_only
@pytest.mark.parametrize("use_pipeline", [False, True])
@pytest.mark.parametrize("seqlens_q,seqlens_k,H,D,dtype", CASES_VARLEN)
def test_dq_mfma_varlen(seqlens_q, seqlens_k, H, D, dtype, use_pipeline):
    assert run_case_varlen(seqlens_q, seqlens_k, H, D, dtype, device="cuda", use_pipeline=use_pipeline)


# Varlen + causal (sequencing-plan item 6, Stage B2) -- BlockDiagonalCausalMask,
# per-block top-left causal. See test_fmha_bwd_dvdk_mfma.py's matching cases for
# the design hypothesis this is verifying.
CASES_VARLEN_CAUSAL = [
    ([64, 128, 32], [64, 128, 32], 1, 64, torch.bfloat16),
    ([64, 128, 32], [64, 128, 32], 8, 64, torch.bfloat16),
    ([100, 50, 200], [100, 50, 200], 8, 64, torch.bfloat16),
    ([64, 128, 32], [64, 128, 32], 8, 128, torch.bfloat16),
]


@rocm_only
@pytest.mark.parametrize("use_pipeline", [False, True])
@pytest.mark.parametrize("seqlens_q,seqlens_k,H,D,dtype", CASES_VARLEN_CAUSAL)
def test_dq_mfma_varlen_causal(seqlens_q, seqlens_k, H, D, dtype, use_pipeline):
    assert run_case_varlen(seqlens_q, seqlens_k, H, D, dtype, device="cuda",
                            use_pipeline=use_pipeline, causal=True)


if __name__ == "__main__":
    device = "cuda"
    all_pass = True

    print("=== dQ MFMA kernel (Phase B.3) vs ref_fmha_bwd ===")
    for args in CASES:
        all_pass &= run_case(*args, device=device)

    if "--pipeline" in sys.argv or "--all" in sys.argv:
        print("\n=== pipeline (lane-distributed K/V load) variant ===")
        for args in CASES:
            all_pass &= run_case(*args, device=device, use_pipeline=True)

    print(f"\n{'ALL PASSED' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)
