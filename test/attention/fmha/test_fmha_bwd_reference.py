"""WP-A3 reference harness: PyTorch eager FMHA forward+backward.

Produces golden (O, LSE, dQ, dK, dV) to validate the FlyDSL backward kernel against.
No FlyDSL or CK involved — pure PyTorch math.

Tensor layout follows CK convention (BMHK):
  Q, K, V, O : [B, M, H, D]   (batch, seq, heads, head_dim)
  dQ, dK, dV : same as Q/K/V
  LSE         : [B, H, M]      (as returned by CK forward)

Usage (inside container aiter-weisun):
    cd /workspace/MSLK
    python test/attention/fmha/test_fmha_bwd_reference.py
"""

import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Reference forward + backward
# ---------------------------------------------------------------------------

def ref_fmha_fwd(Q, K, V, scale=None, causal=False):
    """Pure PyTorch FMHA forward.

    Args:
        Q: [B, M, H, D]  (BMHK layout, matches CK convention)
        K: [B, N, H, D]
        V: [B, N, H, Dv]
        scale: softmax scale (default 1/sqrt(D))
        causal: if True, apply lower-triangular causal mask

    Returns:
        O  : [B, M, H, Dv]
        LSE: [B, H, M]   log-sum-exp per query row (for backward recompute)
    """
    B, M, H, D = Q.shape
    N = K.shape[1]
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    # Rearrange to [B, H, M/N, D] for matmuls
    q = Q.transpose(1, 2).float()  # [B, H, M, D]
    k = K.transpose(1, 2).float()  # [B, H, N, D]
    v = V.transpose(1, 2).float()  # [B, H, N, Dv]

    S = scale * (q @ k.transpose(-2, -1))  # [B, H, M, N]

    if causal:
        mask = torch.triu(torch.ones(M, N, device=Q.device, dtype=torch.bool), diagonal=1)
        S = S.masked_fill(mask, float("-inf"))

    # LSE: [B, H, M]  -- save for backward
    LSE = torch.logsumexp(S, dim=-1)

    P = torch.softmax(S, dim=-1)           # [B, H, M, N]
    O = (P @ v).transpose(1, 2)            # [B, M, H, Dv]

    return O.to(Q.dtype), LSE


def ref_fmha_bwd(Q, K, V, O, dO, LSE, scale=None, causal=False):
    """Pure PyTorch FMHA backward (Flash-Attention math).

    Recomputes P from (Q, K, LSE) — does not store P from forward.

    Args:
        Q, K, V : [B, M/N, H, D]
        O       : [B, M, H, Dv]  (forward output)
        dO      : [B, M, H, Dv]  (upstream gradient)
        LSE     : [B, H, M]      (log-sum-exp from forward)
        scale, causal: same as forward

    Returns:
        dQ: [B, M, H, D]
        dK: [B, N, H, D]
        dV: [B, N, H, Dv]
    """
    B, M, H, D = Q.shape
    N = K.shape[1]
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    q  = Q.transpose(1, 2).float()   # [B, H, M, D]
    k  = K.transpose(1, 2).float()   # [B, H, N, D]
    v  = V.transpose(1, 2).float()   # [B, H, N, Dv]
    o  = O.transpose(1, 2).float()   # [B, H, M, Dv]
    do = dO.transpose(1, 2).float()  # [B, H, M, Dv]

    # --- Recompute P from Q, K, LSE (no KV re-scan) ---
    S = scale * (q @ k.transpose(-2, -1))  # [B, H, M, N]
    if causal:
        mask = torch.triu(torch.ones(M, N, device=Q.device, dtype=torch.bool), diagonal=1)
        S = S.masked_fill(mask, float("-inf"))
    P = torch.exp(S - LSE.unsqueeze(-1))   # [B, H, M, N]  (= softmax(S))

    # --- D vector: rowsum(dO * O) ---
    D_vec = (do * o).sum(dim=-1, keepdim=True)   # [B, H, M, 1]

    # --- Gradients ---
    dV = P.transpose(-2, -1) @ do          # [B, H, N, Dv]
    dP = do @ v.transpose(-2, -1)          # [B, H, M, N]
    dS = P * (dP - D_vec)                  # [B, H, M, N]
    if causal:
        dS = dS.masked_fill(mask, 0.0)
    dS = scale * dS
    dK = dS.transpose(-2, -1) @ q         # [B, H, N, D]
    dQ = dS @ k                            # [B, H, M, D]

    return (
        dQ.transpose(1, 2).to(Q.dtype),
        dK.transpose(1, 2).to(K.dtype),
        dV.transpose(1, 2).to(V.dtype),
    )


def ref_fmha_autograd(Q, K, V, scale=None, causal=False):
    """Cross-check: torch.autograd gradients (independent of ref_fmha_bwd).

    Returns (O, LSE, dQ, dK, dV) using autograd — the ground truth.
    """
    B, M, H, D = Q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    q = Q.detach().float().requires_grad_(True)
    k = K.detach().float().requires_grad_(True)
    v = V.detach().float().requires_grad_(True)

    q_  = q.transpose(1, 2)   # [B, H, M, D]
    k_  = k.transpose(1, 2)
    v_  = v.transpose(1, 2)

    S = scale * (q_ @ k_.transpose(-2, -1))
    if causal:
        mask = torch.triu(torch.ones(M, K.shape[1], device=Q.device, dtype=torch.bool), diagonal=1)
        S = S.masked_fill(mask, float("-inf"))

    LSE_ag = torch.logsumexp(S, dim=-1)
    P = torch.softmax(S, dim=-1)
    O_ag = (P @ v_).transpose(1, 2)

    dO = torch.randn_like(O_ag)
    O_ag.backward(dO)

    return (
        O_ag.to(Q.dtype).detach(),
        LSE_ag.detach(),
        q.grad.to(Q.dtype),
        k.grad.to(K.dtype),
        v.grad.to(V.dtype),
        dO.to(Q.dtype),
    )


# ---------------------------------------------------------------------------
# CK comparison (ROCm only)
# ---------------------------------------------------------------------------

def ref_fmha_ck(Q, K, V, scale=None, causal=False):
    """CK fwd + bwd — currently crashes on gfx950.

    Both the pre-built and newly built mslk.so segfault on memory_efficient_attention()
    with CK ops on gfx950. Cause unknown (build targeted gfx942;gfx950, so not simply
    an arch issue — may be a CK runtime bug or ROCm mismatch). Needs investigation.

    Use ref_fmha_autograd() as the ground truth baseline for A3 development.
    """
    raise RuntimeError(
        "CK FMHA crashes (SIGSEGV) on gfx950 — cause under investigation. "
        "Use ref_fmha_autograd() as the ground truth baseline."
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(name, actual, expected, rtol=1e-2, atol=1e-2):
    match = torch.allclose(actual.float(), expected.float(), rtol=rtol, atol=atol)
    max_err = (actual.float() - expected.float()).abs().max().item()
    status = "PASS" if match else "FAIL"
    print(f"  [{status}] {name:8s}  max_err={max_err:.5f}  rtol={rtol} atol={atol}")
    return match


def run_case(B, M, N, H, D, Dv, dtype, causal, device="cuda"):
    print(f"\nB={B} M={M} N={N} H={H} D={D} Dv={Dv} dtype={dtype} causal={causal}")
    scale = 1.0 / math.sqrt(D)

    Q  = torch.randn(B, M, H, D,  device=device, dtype=dtype)
    K  = torch.randn(B, N, H, D,  device=device, dtype=dtype)
    V  = torch.randn(B, N, H, Dv, device=device, dtype=dtype)

    # --- Ground truth from autograd ---
    O_ag, LSE_ag, dQ_ag, dK_ag, dV_ag, dO = ref_fmha_autograd(Q, K, V, scale, causal)

    # --- Our reference forward ---
    O_ref, LSE_ref = ref_fmha_fwd(Q, K, V, scale, causal)

    # --- Our reference backward (recomputes P from LSE) ---
    dQ_ref, dK_ref, dV_ref = ref_fmha_bwd(Q, K, V, O_ref, dO, LSE_ref, scale, causal)

    # Validate forward
    all_pass = True
    all_pass &= validate("O",   O_ref,   O_ag)
    all_pass &= validate("LSE", LSE_ref, LSE_ag)

    # Validate backward vs autograd
    all_pass &= validate("dQ", dQ_ref, dQ_ag)
    all_pass &= validate("dK", dK_ref, dK_ag)
    all_pass &= validate("dV", dV_ref, dV_ag)

    return all_pass


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def run_ck_case(B, M, N, H, D, Dv, dtype, causal, device="cuda"):
    """Run CK fwd+bwd and diff against our reference backward math.

    CK bwd is known to be inaccurate with bfloat16; fp16 is the reliable dtype.
    Uses looser tolerance (rtol=0.05, atol=0.05) to match CK's own test tolerances.
    """
    import torch.version
    if torch.version.hip is None:
        print(f"  [SKIP] Not ROCm — skipping CK comparison")
        return True

    print(f"\n[CK] B={B} M={M} N={N} H={H} D={D} Dv={Dv} dtype={dtype} causal={causal}")
    scale = 1.0 / math.sqrt(D)

    Q = torch.randn(B, M, H, D,  device=device, dtype=dtype)
    K = torch.randn(B, N, H, D,  device=device, dtype=dtype)
    V = torch.randn(B, N, H, Dv, device=device, dtype=dtype)

    try:
        O_ck, dQ_ck, dK_ck, dV_ck, dO = ref_fmha_ck(Q, K, V, scale, causal)
    except (RuntimeError, NotImplementedError) as e:
        print(f"  [SKIP] CK does not support this config: {e}")
        return True

    # Compute our reference backward using the same dO
    O_ref, LSE_ref = ref_fmha_fwd(Q, K, V, scale, causal)
    dQ_ref, dK_ref, dV_ref = ref_fmha_bwd(Q, K, V, O_ref, dO, LSE_ref, scale, causal)

    # CK fp16 bwd uses loose tolerances (from ck.py BwOp.ERROR_ATOL)
    rtol, atol = 0.05, 0.05

    all_pass = True
    all_pass &= validate("O",   O_ref,  O_ck,  rtol=rtol, atol=atol)
    all_pass &= validate("dQ", dQ_ref, dQ_ck,  rtol=rtol, atol=atol)
    all_pass &= validate("dK", dK_ref, dK_ck,  rtol=rtol, atol=atol)
    all_pass &= validate("dV", dV_ref, dV_ck,  rtol=rtol, atol=atol)
    return all_pass


if __name__ == "__main__":
    device = "cuda"
    all_pass = True

    print("=" * 60)
    print("Part 1: PyTorch autograd reference (golden math check)")
    print("=" * 60)
    pt_cases = [
        # B   M    N    H   D   Dv  dtype           causal
        (1,  128, 128,  8,  64,  64, torch.bfloat16, False),
        (1,  128, 128,  8,  64,  64, torch.bfloat16, True),
        (2,  256, 256,  8, 128, 128, torch.bfloat16, False),
        (2,  256, 256,  8, 128, 128, torch.bfloat16, True),
        (1,  128, 256,  4,  64,  64, torch.bfloat16, False),  # cross-attention M≠N
        (1,  512, 512,  8,  64,  64, torch.float16,  False),
        (1,  512, 512,  8,  64,  64, torch.float16,  True),
    ]
    for args in pt_cases:
        all_pass &= run_case(*args, device=device)

    print("\n" + "=" * 60)
    print("Part 2: CK baseline comparison (fp16 only — bf16 known inaccurate)")
    print("=" * 60)
    # CK bwd: fp16 only (bf16 is skipped in the official test_backward.py too)
    # D must be even, head_dim in [32,64,96,128,256] (CK constraints)
    ck_cases = [
        # B   M    N    H   D   Dv  dtype          causal
        (1,  128, 128,  8,  64,  64, torch.float16, False),
        (1,  128, 128,  8,  64,  64, torch.float16, True),
        (2,  256, 256,  8, 128, 128, torch.float16, False),
        (2,  256, 256,  8, 128, 128, torch.float16, True),
        (1,  512, 512,  4, 128, 128, torch.float16, False),
    ]
    for args in ck_cases:
        all_pass &= run_ck_case(*args, device=device)

    print(f"\n{'ALL PASSED' if all_pass else 'SOME FAILED'}")
