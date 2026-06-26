"""WP-A3 Phase A — FMHA backward kernels: dV, dK, dQ (standalone), then fused dQdKdV.

Standalone kernels validate each gradient independently against ref_fmha_bwd
before they are merged into the single fused kernel that matches CK's structure.

Kernel sequence (matches CK exactly):
  1. compile_fmha_bwd_preprocess   (fmha_bwd_preprocess.py) — D-vector
  2. compile_fmha_bwd_dqdkdv       (this file, fused)        — dV+dK+dQ main
  3. compile_fmha_bwd_convert_dq   (this file)               — fp32→bf16 for dQ

Standalone (for debugging/validation only):
  compile_fmha_bwd_dv   — dV only, grid over N-tiles  ✅ PASSED
  compile_fmha_bwd_dk   — dK only, grid over N-tiles
  compile_fmha_bwd_dq   — dQ only, grid over M-tiles (no atomics)
"""

# ---------------------------------------------------------------------------
# compile_fmha_bwd_dv — standalone dV kernel (validated, kept as reference).

import math as _math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, range_constexpr
from flydsl.expr import math as fly_math
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.vector import ReductionOp
from flydsl.expr.utils.arith import _to_raw as _raw
from kernels.kernels_common import dtype_to_elem_type, get_warp_size

WARP_SIZE = get_warp_size()  # 64 on CDNA

_LOG2E = _math.log2(_math.e)


def compile_fmha_bwd_dv(
    *,
    D: int,
    dtype_str: str = "bf16",
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    scale: float = None,  # softmax scale; defaults to 1/sqrt(D)
):
    """Compile the dV kernel for the FMHA backward pass.

    Args:
        D        : head dimension (multiple of 8, D == BLOCK_N for this kernel)
        dtype_str: "bf16" or "f16"
        BLOCK_M  : M-tile size (compile-time unrolled inner loop)
        BLOCK_N  : N-tile size = block size (number of threads)
        scale    : softmax scale (default 1/sqrt(D)); baked into kernel at compile time

    Returns:
        launch_fn(Q, K, dO, dV, LSE, B, M, N, H, n_M_tiles, stream)
          Q, K, dO : [B*seq*H, D] int16
          dV       : [B*N*H, D]   float32 (output)
          LSE      : [B*H*M]      float32
    """
    import math as _pymath
    if scale is None:
        scale = 1.0 / _pymath.sqrt(D)
    assert D % 8 == 0, f"D={D} must be a multiple of 8"

    elem_dtype = dtype_to_elem_type(dtype_str)
    elem_bits  = 16
    VEC_WIDTH  = 128 // elem_bits   # 8 elements per 128-bit load
    N_COL_VECS = D // VEC_WIDTH     # number of vec-columns per row

    fm = arith.FastMathFlags.fast

    @flyc.kernel
    def fmha_bwd_dv_kernel(
        Q:         fx.Tensor,   # [B*M*H, D] int16 (bf16/fp16 view)
        K:         fx.Tensor,   # [B*N*H, D] int16
        dO:        fx.Tensor,   # [B*M*H, D] int16
        dV:        fx.Tensor,   # [B*N*H, D] float32 (output accumulator)
        LSE:       fx.Tensor,   # [B*H*M] float32
        seq_M:     fx.Int32,
        seq_N:     fx.Int32,
        n_heads:   fx.Int32,
        n_M_tiles: fx.Int32,    # ceil(M / BLOCK_M)
    ):
        """Main kernel: one block per (batch, head, N-tile).

        thread_idx.x = n_in_tile — the specific KV column this thread handles.
        All D float32 accumulators for dV[n, :] live in registers across M-tiles.
        """
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        n_heads_idx   = fx.Index(n_heads)
        seq_M_idx     = fx.Index(seq_M)
        seq_N_idx     = fx.Index(seq_N)
        n_M_tiles_idx = fx.Index(n_M_tiles)
        num_N_tiles   = (seq_N_idx + BLOCK_N - 1) // BLOCK_N

        bid_idx   = fx.Index(bid)
        n_tile    = bid_idx % num_N_tiles
        bh_idx    = bid_idx // num_N_tiles    # flat batch*H index
        batch_idx = bh_idx // n_heads_idx
        head_idx  = bh_idx % n_heads_idx

        n_start   = n_tile * BLOCK_N
        n_in_tile = fx.Index(tid)             # 0 .. BLOCK_N-1
        n_global  = n_start + n_in_tile

        # ---- Buffer tensors (2D for row slicing) ----
        Q_buf   = fx.rocdl.make_buffer_tensor(Q)
        K_buf   = fx.rocdl.make_buffer_tensor(K)
        dO_buf  = fx.rocdl.make_buffer_tensor(dO)
        dV_buf  = fx.rocdl.make_buffer_tensor(dV)
        LSE_buf = fx.rocdl.make_buffer_tensor(LSE)

        copy_16b  = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
        copy_f32  = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        store_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)

        def _load_elem_vec(raw_buf, row, col_vec):
            """128-bit load: VEC_WIDTH elements from (row, col_vec) of a 2D buffer.
            Follows preprocess pattern: slice row first, then divide, then slice col.
            """
            row_sl  = fx.slice(raw_buf, (row, None))
            div_row = fx.logical_divide(row_sl, fx.make_layout(VEC_WIDTH, 1))
            r = fx.make_rmem_tensor(VEC_WIDTH, elem_dtype)
            fx.copy_atom_call(copy_16b, fx.slice(div_row, (None, col_vec)), r)
            return fx.memref_load_vec(r)

        def _load_f32_row(buf, row_idx):
            """32-bit load: one float32 from row row_idx of a raw 2D buffer [N, 1].
            Matches the D_out store pattern in fmha_bwd_preprocess.py:
              row_sl = slice(buf, (row, None)) -> 1D [1] view
              div    = logical_divide(row_sl, (1, 1))
              copy_atom -> reg -> load
            """
            row_sl = fx.slice(buf, (row_idx, None))
            div_1  = fx.logical_divide(row_sl, fx.make_layout(1, 1))
            r = fx.make_rmem_tensor(1, fx.Float32)
            fx.copy_atom_call(copy_f32, fx.slice(div_1, (None, 0)), r)
            return fx.memref_load(r, 0)

        def _store_f32_row(buf, row_idx, val):
            """32-bit store: one float32 to row row_idx of a raw 2D buffer [N, 1]."""
            row_sl = fx.slice(buf, (row_idx, None))
            div_1  = fx.logical_divide(row_sl, fx.make_layout(1, 1))
            r = fx.make_rmem_tensor(1, fx.Float32)
            fx.memref_store(val, r, 0)
            fx.copy_atom_call(store_f32, r, fx.slice(div_1, (None, 0)))

        # ---- Global-row helpers (BMHK layout: row = b*seq*H + pos*H + h) ----
        # Returns Int32 because fx.slice() expects i32/i64 coordinates (not index).
        def _q_row(q_pos):
            return fx.Int32(batch_idx * (seq_M_idx * n_heads_idx) + q_pos * n_heads_idx + head_idx)

        def _kv_row(kv_pos):
            return fx.Int32(batch_idx * (seq_N_idx * n_heads_idx) + kv_pos * n_heads_idx + head_idx)

        def _lse_flat(q_pos):
            # LSE layout: [B, H, M] -> flat = b*H*M + h*M + m = bh_idx*M + m
            return fx.Int32(bh_idx * seq_M_idx + q_pos)

        # ---- Bounds check for this thread's N position ----
        n_valid    = n_global < seq_N_idx
        n_safe     = n_valid.select(n_global, seq_N_idx - fx.Index(1))
        kv_row_idx = _kv_row(n_safe)

        # Pre-load K row for this thread's N position (constant across M-tiles)
        k_vecs = []
        for cv in range_constexpr(N_COL_VECS):
            k_vecs.append(_load_elem_vec(K_buf, kv_row_idx, cv))

        # ---- dV accumulator: D float32 scalars ----
        init_dv = [fx.Float32(0.0)] * D

        # ---- Outer loop over M-tiles (scf.for with loop-carried dV accumulator) ----
        loop_results = init_dv
        for m_tile, iter_args in range(fx.Index(0), n_M_tiles_idx, fx.Index(1), init=init_dv):
            # Extract D carried accumulators
            dv_acc = [iter_args[d] for d in range(D)]

            # ---- Inner unrolled loop over BLOCK_M query rows ----
            for m_in_tile in range_constexpr(BLOCK_M):
                m_global = m_tile * BLOCK_M + m_in_tile
                m_valid  = m_global < seq_M_idx
                m_safe   = m_valid.select(m_global, seq_M_idx - fx.Index(1))

                q_row_idx  = _q_row(m_safe)
                do_row_idx = _q_row(m_safe)

                # Scalar dot product S[m, n] = Q[m] · K[n]
                s_acc = fx.Float32(0.0)
                for cv in range_constexpr(N_COL_VECS):
                    q_vec = _load_elem_vec(Q_buf, q_row_idx, cv)
                    prod  = q_vec.to(fx.Float32) * k_vecs[cv].to(fx.Float32)
                    s_acc = s_acc.addf(prod.reduce(ReductionOp.ADD, fastmath=fm), fastmath=fm)

                # S = scale * dot  (scale is baked in at compile time)
                scale_cst = fx.Float32(scale)
                s_val = fx.Float32(arith.mulf(_raw(s_acc), _raw(scale_cst), fastmath=fm))

                # P = exp2((S - LSE) * log2e)
                lse_val = _load_f32_row(LSE_buf, _lse_flat(m_safe))
                log2e   = fx.Float32(_LOG2E)
                s_sub   = fx.Float32(arith.subf(_raw(s_val), _raw(lse_val), fastmath=fm))
                p_arg   = fx.Float32(arith.mulf(_raw(s_sub), _raw(log2e), fastmath=fm))
                p_val   = fx.Float32(fly_math.exp2(p_arg, fastmath=fm))
                # Zero out OOB rows
                valid_mn = m_valid & n_valid
                p_val    = valid_mn.select(p_val, fx.Float32(0.0))

                # dV[n, d] += P[m, n] * dO[m, d]
                for cv in range_constexpr(N_COL_VECS):
                    do_vec = _load_elem_vec(dO_buf, do_row_idx, cv)
                    do_f32 = do_vec.to(fx.Float32)
                    for e in range_constexpr(VEC_WIDTH):
                        d = cv * VEC_WIDTH + e
                        contrib  = fx.Float32(arith.mulf(_raw(p_val), _raw(do_f32[e]), fastmath=fm))
                        dv_acc[d] = dv_acc[d].addf(contrib, fastmath=fm)

            # Yield accumulated dV back as loop-carried state
            loop_results = yield dv_acc

        # ---- Store dV to global (float32) ----
        # dV is passed as flat [B*N*H*D, 1] 2D buffer.
        # Row index for (kv_row, d) = kv_row * D + d.
        if n_valid:
            for d in range_constexpr(D):
                flat_dv_row = kv_row_idx * D + d
                _store_f32_row(dV_buf, flat_dv_row, loop_results[d])

    @flyc.jit
    def launch_fn(
        Q:         fx.Tensor,
        K:         fx.Tensor,
        dO:        fx.Tensor,
        dV:        fx.Tensor,
        LSE:       fx.Tensor,
        B:         fx.Int32,
        M:         fx.Int32,
        N:         fx.Int32,
        H:         fx.Int32,
        n_M_tiles: fx.Int32,
        stream:    fx.Stream,
    ):
        num_N_tiles = (fx.Index(N) + BLOCK_N - 1) // BLOCK_N
        grid_x = fx.Int32(fx.Index(B) * fx.Index(H) * num_N_tiles)
        fmha_bwd_dv_kernel(
            Q, K, dO, dV, LSE,
            M, N, H, n_M_tiles,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_N, 1, 1),
            stream=stream,
        )

    return launch_fn


# ---------------------------------------------------------------------------
# compile_fmha_bwd_dk — standalone dK kernel (for validation).
#
# dK[n, d] = sum_m dS[m, n] * Q[m, d]
# where dS[m, n] = scale * P[m, n] * (dP[m, n] - D_vec[m])
#       dP[m, n] = dot(dO[m], V[n])
#       P[m, n]  = exp2((scale * Q[m]@K[n] - LSE[m]) * log2e)
#
# Grid/block: identical to dV — grid=(B*H*num_N_tiles,), block=(BLOCK_N,).
# Each thread owns one n_in_tile, accumulates dK[n,:] across all M-tiles.
#
# Extra inputs vs dV:
#   V     : [B*N*H, D] int16  (needed for dP = dot(dO, V))
#   D_vec : [B*M*H, 1] float32  (precomputed rowsum(dO*O), same row order as Q)
# ---------------------------------------------------------------------------

def compile_fmha_bwd_dk(
    *,
    D: int,
    dtype_str: str = "bf16",
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    scale: float = None,
):
    """Compile the standalone dK kernel.

    Returns:
        launch_fn(Q, K, V, dO, dK, LSE, D_vec, B, M, N, H, n_M_tiles, stream)
          Q, K, V, dO : [B*seq*H, D] int16
          dK          : [B*N*H*D, 1] float32 output (flat 2D)
          LSE         : [B*H*M, 1]   float32
          D_vec       : [B*M*H, 1]   float32  (from preprocess kernel)
    """
    import math as _pymath
    if scale is None:
        scale = 1.0 / _pymath.sqrt(D)
    assert D % 8 == 0

    elem_dtype = dtype_to_elem_type(dtype_str)
    elem_bits  = 16
    VEC_WIDTH  = 128 // elem_bits
    N_COL_VECS = D // VEC_WIDTH
    fm = arith.FastMathFlags.fast

    @flyc.kernel
    def fmha_bwd_dk_kernel(
        Q:     fx.Tensor,   # [B*M*H, D] int16
        K:     fx.Tensor,   # [B*N*H, D] int16
        V:     fx.Tensor,   # [B*N*H, D] int16
        dO:    fx.Tensor,   # [B*M*H, D] int16
        dK:    fx.Tensor,   # [B*N*H*D, 1] float32
        LSE:   fx.Tensor,   # [B*H*M, 1] float32
        D_vec: fx.Tensor,   # [B*M*H, 1] float32
        seq_M:     fx.Int32,
        seq_N:     fx.Int32,
        n_heads:   fx.Int32,
        n_M_tiles: fx.Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        n_heads_idx   = fx.Index(n_heads)
        seq_M_idx     = fx.Index(seq_M)
        seq_N_idx     = fx.Index(seq_N)
        n_M_tiles_idx = fx.Index(n_M_tiles)
        num_N_tiles   = (seq_N_idx + BLOCK_N - 1) // BLOCK_N

        bid_idx   = fx.Index(bid)
        n_tile    = bid_idx % num_N_tiles
        bh_idx    = bid_idx // num_N_tiles
        batch_idx = bh_idx // n_heads_idx
        head_idx  = bh_idx % n_heads_idx

        n_start   = n_tile * BLOCK_N
        n_in_tile = fx.Index(tid)
        n_global  = n_start + n_in_tile

        Q_buf    = fx.rocdl.make_buffer_tensor(Q)
        K_buf    = fx.rocdl.make_buffer_tensor(K)
        V_buf    = fx.rocdl.make_buffer_tensor(V)
        dO_buf   = fx.rocdl.make_buffer_tensor(dO)
        dK_buf   = fx.rocdl.make_buffer_tensor(dK)
        LSE_buf  = fx.rocdl.make_buffer_tensor(LSE)
        Dvec_buf = fx.rocdl.make_buffer_tensor(D_vec)

        copy_16b  = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
        copy_f32  = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        store_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)

        def _load_elem_vec(raw_buf, row, col_vec):
            row_sl  = fx.slice(raw_buf, (row, None))
            div_row = fx.logical_divide(row_sl, fx.make_layout(VEC_WIDTH, 1))
            r = fx.make_rmem_tensor(VEC_WIDTH, elem_dtype)
            fx.copy_atom_call(copy_16b, fx.slice(div_row, (None, col_vec)), r)
            return fx.memref_load_vec(r)

        def _load_f32_row(buf, row_idx):
            row_sl = fx.slice(buf, (row_idx, None))
            div_1  = fx.logical_divide(row_sl, fx.make_layout(1, 1))
            r = fx.make_rmem_tensor(1, fx.Float32)
            fx.copy_atom_call(copy_f32, fx.slice(div_1, (None, 0)), r)
            return fx.memref_load(r, 0)

        def _store_f32_row(buf, row_idx, val):
            row_sl = fx.slice(buf, (row_idx, None))
            div_1  = fx.logical_divide(row_sl, fx.make_layout(1, 1))
            r = fx.make_rmem_tensor(1, fx.Float32)
            fx.memref_store(val, r, 0)
            fx.copy_atom_call(store_f32, r, fx.slice(div_1, (None, 0)))

        def _q_row(q_pos):
            return fx.Int32(batch_idx * (seq_M_idx * n_heads_idx) + q_pos * n_heads_idx + head_idx)

        def _kv_row(kv_pos):
            return fx.Int32(batch_idx * (seq_N_idx * n_heads_idx) + kv_pos * n_heads_idx + head_idx)

        def _lse_flat(q_pos):
            return fx.Int32(bh_idx * seq_M_idx + q_pos)

        def _dvec_flat(q_pos):
            # D_vec layout [B*M*H, 1]: row = b*M*H + m*H + h = same as _q_row index
            return _q_row(q_pos)

        n_valid    = n_global < seq_N_idx
        n_safe     = n_valid.select(n_global, seq_N_idx - fx.Index(1))
        kv_row_idx = _kv_row(n_safe)

        # Pre-load K and V rows (constant across M-tiles)
        k_vecs = []
        v_vecs = []
        for cv in range_constexpr(N_COL_VECS):
            k_vecs.append(_load_elem_vec(K_buf, kv_row_idx, cv))
            v_vecs.append(_load_elem_vec(V_buf, kv_row_idx, cv))

        init_dk = [fx.Float32(0.0)] * D

        loop_results = init_dk
        for m_tile, iter_args in range(fx.Index(0), n_M_tiles_idx, fx.Index(1), init=init_dk):
            dk_acc = [iter_args[d] for d in range(D)]

            for m_in_tile in range_constexpr(BLOCK_M):
                m_global = m_tile * BLOCK_M + m_in_tile
                m_valid  = m_global < seq_M_idx
                m_safe   = m_valid.select(m_global, seq_M_idx - fx.Index(1))

                q_row_idx  = _q_row(m_safe)
                do_row_idx = _q_row(m_safe)

                # S[m, n] = scale * Q[m] · K[n]
                s_acc = fx.Float32(0.0)
                for cv in range_constexpr(N_COL_VECS):
                    q_vec = _load_elem_vec(Q_buf, q_row_idx, cv)
                    prod  = q_vec.to(fx.Float32) * k_vecs[cv].to(fx.Float32)
                    s_acc = s_acc.addf(prod.reduce(ReductionOp.ADD, fastmath=fm), fastmath=fm)

                scale_cst = fx.Float32(scale)
                s_val = fx.Float32(arith.mulf(_raw(s_acc), _raw(scale_cst), fastmath=fm))

                # P[m, n] = exp2((S - LSE) * log2e)
                lse_val = _load_f32_row(LSE_buf, _lse_flat(m_safe))
                log2e   = fx.Float32(_LOG2E)
                s_sub   = fx.Float32(arith.subf(_raw(s_val), _raw(lse_val), fastmath=fm))
                p_arg   = fx.Float32(arith.mulf(_raw(s_sub), _raw(log2e), fastmath=fm))
                p_val   = fx.Float32(fly_math.exp2(p_arg, fastmath=fm))
                valid_mn = m_valid & n_valid
                p_val    = valid_mn.select(p_val, fx.Float32(0.0))

                # dP[m, n] = dO[m] · V[n]
                dp_acc = fx.Float32(0.0)
                for cv in range_constexpr(N_COL_VECS):
                    do_vec = _load_elem_vec(dO_buf, do_row_idx, cv)
                    prod   = do_vec.to(fx.Float32) * v_vecs[cv].to(fx.Float32)
                    dp_acc = dp_acc.addf(prod.reduce(ReductionOp.ADD, fastmath=fm), fastmath=fm)

                # D_vec[m] = rowsum(dO[m] * O[m])  (precomputed)
                d_val = _load_f32_row(Dvec_buf, _dvec_flat(m_safe))
                d_val = valid_mn.select(d_val, fx.Float32(0.0))

                # dS[m, n] = scale * P[m, n] * (dP[m, n] - D_vec[m])
                dp_sub = fx.Float32(arith.subf(_raw(dp_acc), _raw(d_val), fastmath=fm))
                ds_val = fx.Float32(arith.mulf(_raw(scale_cst), _raw(p_val), fastmath=fm))
                ds_val = fx.Float32(arith.mulf(_raw(ds_val), _raw(dp_sub), fastmath=fm))

                # dK[n, d] += dS[m, n] * Q[m, d]
                for cv in range_constexpr(N_COL_VECS):
                    q_vec = _load_elem_vec(Q_buf, q_row_idx, cv)
                    q_f32 = q_vec.to(fx.Float32)
                    for e in range_constexpr(VEC_WIDTH):
                        d = cv * VEC_WIDTH + e
                        contrib  = fx.Float32(arith.mulf(_raw(ds_val), _raw(q_f32[e]), fastmath=fm))
                        dk_acc[d] = dk_acc[d].addf(contrib, fastmath=fm)

            loop_results = yield dk_acc

        if n_valid:
            for d in range_constexpr(D):
                flat_dk_row = kv_row_idx * D + d
                _store_f32_row(dK_buf, flat_dk_row, loop_results[d])

    @flyc.jit
    def launch_fn(
        Q:         fx.Tensor,
        K:         fx.Tensor,
        V:         fx.Tensor,
        dO:        fx.Tensor,
        dK:        fx.Tensor,
        LSE:       fx.Tensor,
        D_vec:     fx.Tensor,
        B:         fx.Int32,
        M:         fx.Int32,
        N:         fx.Int32,
        H:         fx.Int32,
        n_M_tiles: fx.Int32,
        stream:    fx.Stream,
    ):
        num_N_tiles = (fx.Index(N) + BLOCK_N - 1) // BLOCK_N
        grid_x = fx.Int32(fx.Index(B) * fx.Index(H) * num_N_tiles)
        fmha_bwd_dk_kernel(
            Q, K, V, dO, dK, LSE, D_vec,
            M, N, H, n_M_tiles,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_N, 1, 1),
            stream=stream,
        )

    return launch_fn


# ---------------------------------------------------------------------------
# compile_fmha_bwd_dq — standalone dQ kernel (no atomics, for validation).
#
# dQ[m, d] = sum_n dS[m, n] * K[n, d]
# where dS[m, n] = scale * P[m, n] * (dP[m, n] - D_vec[m])
#       dP[m, n] = dot(dO[m], V[n])
#
# Grid: (B * H * ceil(M / BLOCK_M),) — one block per M-tile.
# Block: (BLOCK_M,) — one thread per m_in_tile row.
# Each thread owns one m_in_tile, accumulates dQ[m,:] across all N-tiles,
# stores once. No atomics because M-rows are partitioned across blocks.
# ---------------------------------------------------------------------------

def compile_fmha_bwd_dq(
    *,
    D: int,
    dtype_str: str = "bf16",
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    scale: float = None,
):
    """Compile the standalone dQ kernel (no atomics, grid over M-tiles).

    Returns:
        launch_fn(Q, K, V, dO, dQ, LSE, D_vec, B, M, N, H, n_N_tiles, stream)
    """
    import math as _pymath
    if scale is None:
        scale = 1.0 / _pymath.sqrt(D)
    assert D % 8 == 0

    elem_dtype = dtype_to_elem_type(dtype_str)
    elem_bits  = 16
    VEC_WIDTH  = 128 // elem_bits
    N_COL_VECS = D // VEC_WIDTH
    fm = arith.FastMathFlags.fast

    @flyc.kernel
    def fmha_bwd_dq_kernel(
        Q:     fx.Tensor,
        K:     fx.Tensor,
        V:     fx.Tensor,
        dO:    fx.Tensor,
        dQ:    fx.Tensor,   # [B*M*H*D, 1] float32
        LSE:   fx.Tensor,   # [B*H*M, 1] float32
        D_vec: fx.Tensor,   # [B*M*H, 1] float32
        seq_M:     fx.Int32,
        seq_N:     fx.Int32,
        n_heads:   fx.Int32,
        n_N_tiles: fx.Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        n_heads_idx   = fx.Index(n_heads)
        seq_M_idx     = fx.Index(seq_M)
        seq_N_idx     = fx.Index(seq_N)
        n_N_tiles_idx = fx.Index(n_N_tiles)
        num_M_tiles   = (seq_M_idx + BLOCK_M - 1) // BLOCK_M

        bid_idx   = fx.Index(bid)
        m_tile    = bid_idx % num_M_tiles
        bh_idx    = bid_idx // num_M_tiles
        batch_idx = bh_idx // n_heads_idx
        head_idx  = bh_idx % n_heads_idx

        m_start   = m_tile * BLOCK_M
        m_in_tile = fx.Index(tid)
        m_global  = m_start + m_in_tile

        Q_buf    = fx.rocdl.make_buffer_tensor(Q)
        K_buf    = fx.rocdl.make_buffer_tensor(K)
        V_buf    = fx.rocdl.make_buffer_tensor(V)
        dO_buf   = fx.rocdl.make_buffer_tensor(dO)
        dQ_buf   = fx.rocdl.make_buffer_tensor(dQ)
        LSE_buf  = fx.rocdl.make_buffer_tensor(LSE)
        Dvec_buf = fx.rocdl.make_buffer_tensor(D_vec)

        copy_16b  = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
        copy_f32  = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        store_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)

        def _load_elem_vec(raw_buf, row, col_vec):
            row_sl  = fx.slice(raw_buf, (row, None))
            div_row = fx.logical_divide(row_sl, fx.make_layout(VEC_WIDTH, 1))
            r = fx.make_rmem_tensor(VEC_WIDTH, elem_dtype)
            fx.copy_atom_call(copy_16b, fx.slice(div_row, (None, col_vec)), r)
            return fx.memref_load_vec(r)

        def _load_f32_row(buf, row_idx):
            row_sl = fx.slice(buf, (row_idx, None))
            div_1  = fx.logical_divide(row_sl, fx.make_layout(1, 1))
            r = fx.make_rmem_tensor(1, fx.Float32)
            fx.copy_atom_call(copy_f32, fx.slice(div_1, (None, 0)), r)
            return fx.memref_load(r, 0)

        def _store_f32_row(buf, row_idx, val):
            row_sl = fx.slice(buf, (row_idx, None))
            div_1  = fx.logical_divide(row_sl, fx.make_layout(1, 1))
            r = fx.make_rmem_tensor(1, fx.Float32)
            fx.memref_store(val, r, 0)
            fx.copy_atom_call(store_f32, r, fx.slice(div_1, (None, 0)))

        def _q_row(q_pos):
            return fx.Int32(batch_idx * (seq_M_idx * n_heads_idx) + q_pos * n_heads_idx + head_idx)

        def _kv_row(kv_pos):
            return fx.Int32(batch_idx * (seq_N_idx * n_heads_idx) + kv_pos * n_heads_idx + head_idx)

        def _lse_flat(q_pos):
            return fx.Int32(bh_idx * seq_M_idx + q_pos)

        m_valid    = m_global < seq_M_idx
        m_safe     = m_valid.select(m_global, seq_M_idx - fx.Index(1))
        q_row_idx  = _q_row(m_safe)
        do_row_idx = _q_row(m_safe)

        # Pre-load Q, dO, LSE, D_vec for this row (constant across N-tiles)
        q_vecs  = []
        do_vecs = []
        for cv in range_constexpr(N_COL_VECS):
            q_vecs.append(_load_elem_vec(Q_buf,  q_row_idx,  cv))
            do_vecs.append(_load_elem_vec(dO_buf, do_row_idx, cv))

        lse_val   = _load_f32_row(LSE_buf,  _lse_flat(m_safe))
        d_val     = _load_f32_row(Dvec_buf, q_row_idx)
        scale_cst = fx.Float32(scale)
        log2e     = fx.Float32(_LOG2E)

        init_dq = [fx.Float32(0.0)] * D

        loop_results = init_dq
        for n_tile, iter_args in range(fx.Index(0), n_N_tiles_idx, fx.Index(1), init=init_dq):
            dq_acc = [iter_args[d] for d in range(D)]

            for n_in_tile in range_constexpr(BLOCK_N):
                n_global   = n_tile * BLOCK_N + n_in_tile
                n_valid    = n_global < seq_N_idx
                n_safe     = n_valid.select(n_global, seq_N_idx - fx.Index(1))
                kv_row_idx = _kv_row(n_safe)

                # S[m, n] = scale * Q[m] . K[n]
                s_acc = fx.Float32(0.0)
                for cv in range_constexpr(N_COL_VECS):
                    k_vec = _load_elem_vec(K_buf, kv_row_idx, cv)
                    prod  = q_vecs[cv].to(fx.Float32) * k_vec.to(fx.Float32)
                    s_acc = s_acc.addf(prod.reduce(ReductionOp.ADD, fastmath=fm), fastmath=fm)

                s_val = fx.Float32(arith.mulf(_raw(s_acc), _raw(scale_cst), fastmath=fm))

                # P[m, n] = exp2((S - LSE) * log2e)
                s_sub = fx.Float32(arith.subf(_raw(s_val), _raw(lse_val), fastmath=fm))
                p_arg = fx.Float32(arith.mulf(_raw(s_sub), _raw(log2e), fastmath=fm))
                p_val = fx.Float32(fly_math.exp2(p_arg, fastmath=fm))
                valid_mn = m_valid & n_valid
                p_val    = valid_mn.select(p_val, fx.Float32(0.0))

                # dP[m, n] = dO[m] . V[n]
                dp_acc = fx.Float32(0.0)
                for cv in range_constexpr(N_COL_VECS):
                    v_vec  = _load_elem_vec(V_buf, kv_row_idx, cv)
                    prod   = do_vecs[cv].to(fx.Float32) * v_vec.to(fx.Float32)
                    dp_acc = dp_acc.addf(prod.reduce(ReductionOp.ADD, fastmath=fm), fastmath=fm)

                # dS[m, n] = scale * P[m, n] * (dP[m, n] - D_vec[m])
                dp_sub = fx.Float32(arith.subf(_raw(dp_acc), _raw(d_val), fastmath=fm))
                ds_val = fx.Float32(arith.mulf(_raw(scale_cst), _raw(p_val), fastmath=fm))
                ds_val = fx.Float32(arith.mulf(_raw(ds_val), _raw(dp_sub), fastmath=fm))

                # dQ[m, d] += dS[m, n] * K[n, d]
                for cv in range_constexpr(N_COL_VECS):
                    k_vec = _load_elem_vec(K_buf, kv_row_idx, cv)
                    k_f32 = k_vec.to(fx.Float32)
                    for e in range_constexpr(VEC_WIDTH):
                        d = cv * VEC_WIDTH + e
                        contrib   = fx.Float32(arith.mulf(_raw(ds_val), _raw(k_f32[e]), fastmath=fm))
                        dq_acc[d] = dq_acc[d].addf(contrib, fastmath=fm)

            loop_results = yield dq_acc

        if m_valid:
            for d in range_constexpr(D):
                flat_dq_row = q_row_idx * D + d
                _store_f32_row(dQ_buf, flat_dq_row, loop_results[d])

    @flyc.jit
    def launch_fn(
        Q:         fx.Tensor,
        K:         fx.Tensor,
        V:         fx.Tensor,
        dO:        fx.Tensor,
        dQ:        fx.Tensor,
        LSE:       fx.Tensor,
        D_vec:     fx.Tensor,
        B:         fx.Int32,
        M:         fx.Int32,
        N:         fx.Int32,
        H:         fx.Int32,
        n_N_tiles: fx.Int32,
        stream:    fx.Stream,
    ):
        num_M_tiles = (fx.Index(M) + BLOCK_M - 1) // BLOCK_M
        grid_x = fx.Int32(fx.Index(B) * fx.Index(H) * num_M_tiles)
        fmha_bwd_dq_kernel(
            Q, K, V, dO, dQ, LSE, D_vec,
            M, N, H, n_N_tiles,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_M, 1, 1),
            stream=stream,
        )

    return launch_fn


# ---------------------------------------------------------------------------
# compile_fmha_bwd_dqdkdv — fused main backward kernel (matches CK Kernel 2).
#
# Computes dV, dK, dQ in a single kernel, grid over N-tiles:
#   dV[n] = sum_m P[m,n] * dO[m]           — accumulate in registers, store once
#   dK[n] = sum_m dS[m,n] * Q[m]           — accumulate in registers, store once
#   dQ[m] += sum_n dS[m,n] * K[n]          — atomic add (multiple N-tile blocks)
#
# dV and dK are N-tile-unique -> simple store, no atomics.
# dQ is M-row-shared across N-tile blocks -> float32 atomic add to scratch.
#
# Grid:  (B * H * ceil(N / BLOCK_N),)  one block per KV-tile
# Block: (BLOCK_N,)                    one thread per KV column
#
# dQ_f32 scratch must be zero-initialised before launch (caller's responsibility).
# After this kernel, call compile_fmha_bwd_convert_dq to cast f32 -> bf16.
# ---------------------------------------------------------------------------

def compile_fmha_bwd_dqdkdv(
    *,
    D: int,
    dtype_str: str = "bf16",
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    scale: float = None,
):
    """Compile the fused dV+dK+dQ main backward kernel.

    Returns:
        launch_fn(Q, K, V, dO, dV, dK, dQ_f32, LSE, D_vec,
                  B, M, N, H, n_M_tiles, stream)
          Q, K, V, dO : [B*seq*H, D]   int16
          dV, dK      : [B*N*H*D, 1]   float32 output (flat 2D)
          dQ_f32      : [B*M*H*D, 1]   float32 scratch (caller zeros before call)
          LSE         : [B*H*M, 1]     float32
          D_vec       : [B*M*H, 1]     float32
    """
    import math as _pymath
    if scale is None:
        scale = 1.0 / _pymath.sqrt(D)
    assert D % 8 == 0

    elem_dtype = dtype_to_elem_type(dtype_str)
    elem_bits  = 16
    VEC_WIDTH  = 128 // elem_bits
    N_COL_VECS = D // VEC_WIDTH
    fm = arith.FastMathFlags.fast

    @flyc.kernel
    def fmha_bwd_dqdkdv_kernel(
        Q:      fx.Tensor,   # [B*M*H, D] int16
        K:      fx.Tensor,   # [B*N*H, D] int16
        V:      fx.Tensor,   # [B*N*H, D] int16
        dO:     fx.Tensor,   # [B*M*H, D] int16
        dV:     fx.Tensor,   # [B*N*H*D, 1] float32
        dK:     fx.Tensor,   # [B*N*H*D, 1] float32
        dQ_f32: fx.Tensor,   # [B*M*H*D, 1] float32 (zeroed by caller)
        LSE:    fx.Tensor,   # [B*H*M, 1] float32
        D_vec:  fx.Tensor,   # [B*M*H, 1] float32
        seq_M:     fx.Int32,
        seq_N:     fx.Int32,
        n_heads:   fx.Int32,
        n_M_tiles: fx.Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        n_heads_idx   = fx.Index(n_heads)
        seq_M_idx     = fx.Index(seq_M)
        seq_N_idx     = fx.Index(seq_N)
        n_M_tiles_idx = fx.Index(n_M_tiles)
        num_N_tiles   = (seq_N_idx + BLOCK_N - 1) // BLOCK_N

        bid_idx   = fx.Index(bid)
        n_tile    = bid_idx % num_N_tiles
        bh_idx    = bid_idx // num_N_tiles
        batch_idx = bh_idx // n_heads_idx
        head_idx  = bh_idx % n_heads_idx

        n_start   = n_tile * BLOCK_N
        n_in_tile = fx.Index(tid)
        n_global  = n_start + n_in_tile

        Q_buf    = fx.rocdl.make_buffer_tensor(Q)
        K_buf    = fx.rocdl.make_buffer_tensor(K)
        V_buf    = fx.rocdl.make_buffer_tensor(V)
        dO_buf   = fx.rocdl.make_buffer_tensor(dO)
        dV_buf   = fx.rocdl.make_buffer_tensor(dV)
        dK_buf   = fx.rocdl.make_buffer_tensor(dK)
        # dQ_f32: plain tensor (NOT buffer-wrapped) — UniversalAtomic requires generic memory
        LSE_buf  = fx.rocdl.make_buffer_tensor(LSE)
        Dvec_buf = fx.rocdl.make_buffer_tensor(D_vec)

        copy_16b  = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
        copy_f32  = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        store_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        # UniversalCopy/Atomic for dQ_f32 (plain tensor, not buffer descriptor)
        ucopy_f32  = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
        atomic_add_f32 = fx.make_copy_atom(
            fx.UniversalAtomic(fx.AtomicOp.Add, fx.Float32), fx.Float32
        )

        def _load_elem_vec(raw_buf, row, col_vec):
            row_sl  = fx.slice(raw_buf, (row, None))
            div_row = fx.logical_divide(row_sl, fx.make_layout(VEC_WIDTH, 1))
            r = fx.make_rmem_tensor(VEC_WIDTH, elem_dtype)
            fx.copy_atom_call(copy_16b, fx.slice(div_row, (None, col_vec)), r)
            return fx.memref_load_vec(r)

        def _load_f32_row(buf, row_idx):
            row_sl = fx.slice(buf, (row_idx, None))
            div_1  = fx.logical_divide(row_sl, fx.make_layout(1, 1))
            r = fx.make_rmem_tensor(1, fx.Float32)
            fx.copy_atom_call(copy_f32, fx.slice(div_1, (None, 0)), r)
            return fx.memref_load(r, 0)

        def _store_f32_row(buf, row_idx, val):
            row_sl = fx.slice(buf, (row_idx, None))
            div_1  = fx.logical_divide(row_sl, fx.make_layout(1, 1))
            r = fx.make_rmem_tensor(1, fx.Float32)
            fx.memref_store(val, r, 0)
            fx.copy_atom_call(store_f32, r, fx.slice(div_1, (None, 0)))

        # dQ_f32 uses UniversalAtomic — divide the plain tensor directly (no make_buffer_tensor)
        dQ_div = fx.logical_divide(dQ_f32, fx.make_layout(1, 1))

        def _atomic_add_dq(flat_row, val):
            """Float32 atomic add into dQ_f32 at flat row index flat_row."""
            r = fx.make_rmem_tensor(1, fx.Float32)
            fx.memref_store(val, r, 0)
            fx.copy_atom_call(atomic_add_f32, r,
                              fx.slice(dQ_div, (None, fx.Int32(flat_row))))

        def _q_row(q_pos):
            return fx.Int32(batch_idx * (seq_M_idx * n_heads_idx) + q_pos * n_heads_idx + head_idx)

        def _kv_row(kv_pos):
            return fx.Int32(batch_idx * (seq_N_idx * n_heads_idx) + kv_pos * n_heads_idx + head_idx)

        def _lse_flat(q_pos):
            return fx.Int32(bh_idx * seq_M_idx + q_pos)

        n_valid    = n_global < seq_N_idx
        n_safe     = n_valid.select(n_global, seq_N_idx - fx.Index(1))
        kv_row_idx = _kv_row(n_safe)

        # Pre-load K and V rows (constant across all M-tiles)
        k_vecs = []
        v_vecs = []
        for cv in range_constexpr(N_COL_VECS):
            k_vecs.append(_load_elem_vec(K_buf, kv_row_idx, cv))
            v_vecs.append(_load_elem_vec(V_buf, kv_row_idx, cv))

        # dV and dK accumulators in registers (D float32 each)
        init_dv = [fx.Float32(0.0)] * D
        init_dk = [fx.Float32(0.0)] * D
        init_all = init_dv + init_dk  # 2*D carried values

        loop_results = init_all
        for m_tile, iter_args in range(fx.Index(0), n_M_tiles_idx, fx.Index(1), init=init_all):
            dv_acc = [iter_args[d]     for d in range(D)]
            dk_acc = [iter_args[D + d] for d in range(D)]

            for m_in_tile in range_constexpr(BLOCK_M):
                m_global   = m_tile * BLOCK_M + m_in_tile
                m_valid    = m_global < seq_M_idx
                m_safe     = m_valid.select(m_global, seq_M_idx - fx.Index(1))
                q_row_idx  = _q_row(m_safe)
                do_row_idx = _q_row(m_safe)

                # S[m, n] = scale * Q[m] . K[n]
                s_acc = fx.Float32(0.0)
                for cv in range_constexpr(N_COL_VECS):
                    q_vec = _load_elem_vec(Q_buf, q_row_idx, cv)
                    prod  = q_vec.to(fx.Float32) * k_vecs[cv].to(fx.Float32)
                    s_acc = s_acc.addf(prod.reduce(ReductionOp.ADD, fastmath=fm), fastmath=fm)

                scale_cst = fx.Float32(scale)
                s_val = fx.Float32(arith.mulf(_raw(s_acc), _raw(scale_cst), fastmath=fm))

                # P[m, n] = exp2((S - LSE) * log2e)
                lse_val  = _load_f32_row(LSE_buf, _lse_flat(m_safe))
                log2e    = fx.Float32(_LOG2E)
                s_sub    = fx.Float32(arith.subf(_raw(s_val),  _raw(lse_val),  fastmath=fm))
                p_arg    = fx.Float32(arith.mulf(_raw(s_sub),  _raw(log2e),    fastmath=fm))
                p_val    = fx.Float32(fly_math.exp2(p_arg, fastmath=fm))
                valid_mn = m_valid & n_valid
                p_val    = valid_mn.select(p_val, fx.Float32(0.0))

                # dP[m, n] = dO[m] . V[n]
                dp_acc = fx.Float32(0.0)
                for cv in range_constexpr(N_COL_VECS):
                    do_vec = _load_elem_vec(dO_buf, do_row_idx, cv)
                    prod   = do_vec.to(fx.Float32) * v_vecs[cv].to(fx.Float32)
                    dp_acc = dp_acc.addf(prod.reduce(ReductionOp.ADD, fastmath=fm), fastmath=fm)

                # D_vec[m] = rowsum(dO * O)  (precomputed by Kernel 1)
                d_val = _load_f32_row(Dvec_buf, q_row_idx)
                d_val = valid_mn.select(d_val, fx.Float32(0.0))

                # dS[m, n] = scale * P[m, n] * (dP[m, n] - D_vec[m])
                dp_sub = fx.Float32(arith.subf(_raw(dp_acc),  _raw(d_val),    fastmath=fm))
                ds_val = fx.Float32(arith.mulf(_raw(scale_cst), _raw(p_val),  fastmath=fm))
                ds_val = fx.Float32(arith.mulf(_raw(ds_val),  _raw(dp_sub),   fastmath=fm))

                # dV[n, d] += P[m, n] * dO[m, d]
                for cv in range_constexpr(N_COL_VECS):
                    do_vec = _load_elem_vec(dO_buf, do_row_idx, cv)
                    do_f32 = do_vec.to(fx.Float32)
                    for e in range_constexpr(VEC_WIDTH):
                        d = cv * VEC_WIDTH + e
                        contrib  = fx.Float32(arith.mulf(_raw(p_val), _raw(do_f32[e]), fastmath=fm))
                        dv_acc[d] = dv_acc[d].addf(contrib, fastmath=fm)

                # dK[n, d] += dS[m, n] * Q[m, d]
                for cv in range_constexpr(N_COL_VECS):
                    q_vec = _load_elem_vec(Q_buf, q_row_idx, cv)
                    q_f32 = q_vec.to(fx.Float32)
                    for e in range_constexpr(VEC_WIDTH):
                        d = cv * VEC_WIDTH + e
                        contrib  = fx.Float32(arith.mulf(_raw(ds_val), _raw(q_f32[e]), fastmath=fm))
                        dk_acc[d] = dk_acc[d].addf(contrib, fastmath=fm)

                # dQ[m, d] += dS[m, n] * K[n, d]  — atomic (shared across N-tile blocks)
                if m_valid:
                    for cv in range_constexpr(N_COL_VECS):
                        k_f32 = k_vecs[cv].to(fx.Float32)
                        for e in range_constexpr(VEC_WIDTH):
                            d = cv * VEC_WIDTH + e
                            contrib     = fx.Float32(arith.mulf(_raw(ds_val), _raw(k_f32[e]), fastmath=fm))
                            flat_dq_row = q_row_idx * D + d
                            _atomic_add_dq(flat_dq_row, contrib)

            loop_results = yield dv_acc + dk_acc

        # Store dV and dK (unique per N-tile, no atomics)
        if n_valid:
            for d in range_constexpr(D):
                flat_row = kv_row_idx * D + d
                _store_f32_row(dV_buf, flat_row, loop_results[d])
                _store_f32_row(dK_buf, flat_row, loop_results[D + d])

    @flyc.jit
    def launch_fn(
        Q:         fx.Tensor,
        K:         fx.Tensor,
        V:         fx.Tensor,
        dO:        fx.Tensor,
        dV:        fx.Tensor,
        dK:        fx.Tensor,
        dQ_f32:    fx.Tensor,
        LSE:       fx.Tensor,
        D_vec:     fx.Tensor,
        B:         fx.Int32,
        M:         fx.Int32,
        N:         fx.Int32,
        H:         fx.Int32,
        n_M_tiles: fx.Int32,
        stream:    fx.Stream,
    ):
        num_N_tiles = (fx.Index(N) + BLOCK_N - 1) // BLOCK_N
        grid_x = fx.Int32(fx.Index(B) * fx.Index(H) * num_N_tiles)
        fmha_bwd_dqdkdv_kernel(
            Q, K, V, dO, dV, dK, dQ_f32, LSE, D_vec,
            M, N, H, n_M_tiles,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_N, 1, 1),
            stream=stream,
        )

    return launch_fn


# ---------------------------------------------------------------------------
# compile_fmha_bwd_convert_dq — dQ dtype convert kernel (matches CK Kernel 3).
#
# Converts the float32 dQ scratch buffer -> bf16/fp16 output.
# Grid:  (B * H * ceil(M / BLOCK_M),)
# Block: (BLOCK_M,)  one thread per M row
# ---------------------------------------------------------------------------

def compile_fmha_bwd_convert_dq(
    *,
    D: int,
    dtype_str: str = "bf16",
    BLOCK_M: int = 64,
):
    """Convert float32 dQ accumulator to bf16/fp16.

    Returns:
        launch_fn(dQ_f32, dQ, B, M, H, n_M_tiles, stream)
          dQ_f32 : [B*M*H*D, 1] float32 (scratch from dqdkdv kernel)
          dQ     : [B*M*H, D]   int16 view of bf16/fp16 output
    """
    assert D % 8 == 0
    elem_bits = 16
    VEC_WIDTH = 128 // elem_bits

    # dQ output dtype
    elem_dtype = dtype_to_elem_type(dtype_str)
    N_COL_VECS_CVT = D // VEC_WIDTH

    @flyc.kernel
    def convert_dq_kernel(
        dQ_f32: fx.Tensor,   # [B*M*H*D, 1] float32 (flat)
        dQ:     fx.Tensor,   # [B*M*H, D]   int16
        seq_M:   fx.Int32,
        n_heads: fx.Int32,
    ):
        """Each thread handles one M-row: loads D f32 values, converts to bf16, stores."""
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        n_heads_idx = fx.Index(n_heads)
        seq_M_idx   = fx.Index(seq_M)
        num_M_tiles = (seq_M_idx + BLOCK_M - 1) // BLOCK_M

        bid_idx   = fx.Index(bid)
        m_tile    = bid_idx % num_M_tiles
        bh_idx    = bid_idx // num_M_tiles
        batch_idx = bh_idx // n_heads_idx
        head_idx  = bh_idx % n_heads_idx

        m_start   = m_tile * BLOCK_M
        m_in_tile = fx.Index(tid)
        m_global  = m_start + m_in_tile
        m_valid   = m_global < seq_M_idx
        m_safe    = m_valid.select(m_global, seq_M_idx - fx.Index(1))

        q_row = fx.Int32(batch_idx * (seq_M_idx * n_heads_idx) + m_safe * n_heads_idx + head_idx)

        # Use plain tensors with UniversalCopy for dQ_f32 (flat f32 buffer)
        # Use buffer tensor + slice pattern for dQ output (int16)
        dQ_buf = fx.rocdl.make_buffer_tensor(dQ)
        copy_f32  = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
        store_16b = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), elem_bits)

        # dQ_f32 divided into flat scalar elements: [B*M*H*D, 1]
        dQ_f32_div = fx.logical_divide(dQ_f32, fx.make_layout(1, 1))

        if m_valid:
            for cv in range_constexpr(N_COL_VECS_CVT):
                # Load VEC_WIDTH consecutive f32 values and pack into one bf16 vec store
                bf16_vals = []
                for e in range_constexpr(VEC_WIDTH):
                    d = cv * VEC_WIDTH + e
                    flat_src = fx.Int32(q_row * D + d)
                    r_f32 = fx.make_rmem_tensor(1, fx.Float32)
                    fx.copy_atom_call(copy_f32, fx.slice(dQ_f32_div, (None, flat_src)), r_f32)
                    val_f32 = fx.memref_load(r_f32, 0)
                    # f32 -> bf16: take upper 16 bits of IEEE754 representation
                    val_i32 = fx.Int32(Vec.from_elements([val_f32], fx.Float32).bitcast(fx.Int32)[0])
                    val_i16 = fx.Int16(val_i32.shrui(fx.Int32(16)))
                    bf16_vals.append(val_i16)

                # Pack VEC_WIDTH bf16 values into a vector and store via 128-bit write
                bf16_vec = Vec.from_elements(bf16_vals, fx.Int16)
                r_vec = fx.make_rmem_tensor(VEC_WIDTH, fx.Int16)
                fx.memref_store_vec(bf16_vec, r_vec)

                # Store to dQ[q_row, cv*VEC_WIDTH .. (cv+1)*VEC_WIDTH]
                row_sl  = fx.slice(dQ_buf, (q_row, None))
                div_row = fx.logical_divide(row_sl, fx.make_layout(VEC_WIDTH, 1))
                store_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
                fx.copy_atom_call(store_atom, r_vec, fx.slice(div_row, (None, cv)))

    @flyc.jit
    def launch_fn(
        dQ_f32:    fx.Tensor,
        dQ:        fx.Tensor,
        B:         fx.Int32,
        M:         fx.Int32,
        H:         fx.Int32,
        n_M_tiles: fx.Int32,
        stream:    fx.Stream,
    ):
        num_M_tiles = (fx.Index(M) + BLOCK_M - 1) // BLOCK_M
        grid_x = fx.Int32(fx.Index(B) * fx.Index(H) * num_M_tiles)
        convert_dq_kernel(dQ_f32, dQ, M, H).launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_M, 1, 1),
            stream=stream,
        )

    return launch_fn
