"""WP-A3 Phase B — FMHA backward with MFMA tiling.

Replaces scalar dot-product loops with mfma_f32_32x32x16_{bf16,f16}.

Phase B.1 (this file): dV only with MFMA, D=128. Validates MFMA layout.
Phase B.2: add dK.
Phase B.3: add dQ, fuse all.

MFMA32x32x16 register layout (CDNA4/gfx950, wave64) — EMPIRICALLY PROBED
(see test/attention/fmha/probe_mfma_layout.py):
  Lane j (0..63): j_mod = j%32, j_div = j//32
  INPUT operand[free, k]:  free = j_mod,  k = j_div*8 + e  (e = 0..7, MFMA_LK=8)
  OUTPUT C[m, n] for reg r (0..15):
    m = j_div*4 + (r//4)*8 + (r%4)   (A-operand free dim; varies with r)
    n = j_mod                        (B-operand free dim; fixed across r)
  NOTE: input free-dim = j_mod, but output row(m) uses the scrambled
  j_div*4+(r//4)*8+(r%4) mapping. The two are NOT the same axis — this is
  the register-layout mismatch that the GEMM1->GEMM2 bridge must reconcile.

For S = Q @ K^T with BLOCK_N=64, D=128, BLOCK_SIZE=256 (4 waves):
  - wave_n_sub = wave // 2  -> which 32 N-rows (of BLOCK_N=64)
  - wave_m_sub = wave  % 2  -> which 32 M-rows (of BLOCK_M=64)
  Each wave computes a [32M, 32N] S sub-tile.
  K_STEPS = D/16 = 8 MFMA calls.

For dV = P^T @ dO with BLOCK_N=64, D=128:
  - dV has shape [BLOCK_N, D] = [64, 128]
  - wave_n_sub -> which 32 N-rows of dV (same as for S)
  - wave_d_sub -> which 32 D-cols of dV
  But we only have 4 waves, and need 2×2=4 sub-tiles of dV... but each wave
  needs BOTH S (which determines wave_m_sub and wave_n_sub) AND dV sub-tile.
  Since BLOCK_M=64=BLOCK_N=64, we can set wave_m_sub=wave_n_sub for S
  and wave_d_sub separately. Actually, let's use:
    wave_n_sub = wave // 2  -> N-rows for both S and dV
    wave_m_sub = wave %  2  -> M-rows for S (inner dim of P)
    wave_d_sub = wave %  2  -> D-cols for dV (output dim)
  This means S[wave_m_sub*32:+32, wave_n_sub*32:+32]
  and dV[wave_n_sub*32:+32, wave_d_sub*32:+32]
  which requires: dV covers the same D-range as M-rows of Q.
  With D=128=BLOCK_M*2: wave_d_sub*32 covers cols 0..31 or 32..63, but D=128!
  So we need 4 D-sub-tiles for D=128. But we only have 4 waves total.

  Correct decomposition for D=128, BLOCK_N=64:
    BLOCK_SIZE = 4 * WARP_SIZE = 256
    dV output: [BLOCK_N, D] = [64, 128] = 4 × [32, 32] sub-tiles
    dK output: [BLOCK_N, D] = same
    Need: 4 sub-tiles per output, 4 waves -> 1 wave per sub-tile

    Wave assignment:
      wave 0: n_sub=0, d_sub=0  -> dV[0:32,   0:32]
      wave 1: n_sub=0, d_sub=1  -> dV[0:32,  32:64]
      wave 2: n_sub=1, d_sub=0  -> dV[32:64,  0:32]
      wave 3: n_sub=1, d_sub=1  -> dV[32:64, 32:64]
    BUT D=128 needs d_sub=0..3 (4 D-tiles), not 0..1.

  For D=128 with 4 waves: each wave must cover 2 D-sub-tiles (sequential).
  OR: use BLOCK_SIZE=512 (8 waves) for D=128.

  SIMPLEST APPROACH for Phase B.1:
    Use BLOCK_SIZE=256, BLOCK_N=64, D=64 (not 128).
    -> wave_n_sub = wave // 2, wave_d_sub = wave % 2
    -> 4 waves cover [64N, 64D] exactly (2×2 = 4 × [32,32] sub-tiles) ✓

  For D=128: use BLOCK_SIZE=512 (8 waves), or do 2 MFMA rounds per wave.
  Phase B.1 targets D=64 for clean validation, then extend.

Target: gfx950 (CDNA4, wave64).
"""

import math as _math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl
from flydsl.expr import math as fly_math
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue
from flydsl.expr.utils.arith import _to_raw as _raw
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from kernels.kernels_common import dtype_to_elem_type, get_warp_size

WARP_SIZE = get_warp_size()  # 64 on CDNA4
_LOG2E   = _math.log2(_math.e)


def _ds_read_tr_v4(v4_type, lds_elem_idx, lds_byte_base):
    """gfx950 hardware-transpose LDS read (ds_read_b64_tr_b16), rocdl wrapper.

    Reads a 4x16 bf16 tile cooperatively across a 16-lane group and returns the
    transposed 16x4 (4 bf16 per lane). Pair two calls + shuffle for a v8 operand.
    """
    byte_i64 = fx.Int64(lds_elem_idx * 2 + lds_byte_base)
    ptr = buffer_ops.create_llvm_ptr(byte_i64, address_space=3)
    return rocdl.ds_read_tr16_b64(v4_type, ptr).result


def compile_fmha_bwd_dv_mfma(
    *,
    D: int = 64,
    dtype_str: str = "bf16",
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    scale: float = None,
):
    """Phase B.1: dV only with MFMA, validates MFMA register layout.

    D must equal BLOCK_N (= 64) for the clean 4-wave decomposition:
      4 waves × [32N, 32D] sub-tiles = [64N, 64D] = [BLOCK_N, D].

    Returns:
        launch_fn(Q, K, dO, dV, LSE, B, M, N, H, n_M_tiles, stream)
          Q, K, dO  : [B*seq*H, D]   int16
          dV        : [B*N*H*D, 1]   float32
          LSE       : [B*H*M, 1]     float32
    """
    import math as _pm
    if scale is None:
        scale = 1.0 / _pm.sqrt(D)
    assert D == BLOCK_N, f"Phase B.1 requires D == BLOCK_N, got D={D}, BLOCK_N={BLOCK_N}"
    assert D % 16 == 0

    elem_dtype = dtype_to_elem_type(dtype_str)
    elem_bits  = 16
    MFMA_K     = 16              # K-dimension per MFMA call
    MFMA_LK    = 8               # elements per lane per MFMA (MFMA_K / 2)
    K_STEPS    = D // MFMA_K    # number of MFMA calls to cover D
    fm         = arith.FastMathFlags.fast

    # Block layout: 4 waves for [BLOCK_N, D] = [64, 64] dV output
    BLOCK_SIZE = 256
    NUM_WAVES  = BLOCK_SIZE // WARP_SIZE  # 4
    # wave_n_sub = wave // 2  -> N sub-tile (0 or 1, each covers 32 N-rows)
    # wave_d_sub = wave  % 2  -> D sub-tile (0 or 1, each covers 32 D-cols)
    WAVE_N_TILES = BLOCK_N // 32  # 2
    WAVE_D_TILES = D // 32        # 2
    assert WAVE_N_TILES * WAVE_D_TILES == NUM_WAVES

    # LDS: Q tile [BLOCK_M, D] + dO tile [BLOCK_M, D]
    LDS_Q_ELEMS   = BLOCK_M * D
    LDS_DO_ELEMS  = BLOCK_M * D
    LDS_LSE_ELEMS = BLOCK_M   # f32 LSE tile, staged once per m_tile

    gpu_arch = "gfx950"
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="fmha_bwd_dv_mfma_smem")
    lds_q_off  = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_q_off + LDS_Q_ELEMS * 2
    lds_do_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_do_off + LDS_DO_ELEMS * 2
    lds_lse_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_lse_off + LDS_LSE_ELEMS * 4   # f32

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def fmha_bwd_dv_mfma_kernel(  # noqa: F811
        Q:         fx.Tensor,   # [B*M*H, D] int16
        K:         fx.Tensor,   # [B*N*H, D] int16
        dO:        fx.Tensor,   # [B*M*H, D] int16
        dV:        fx.Tensor,   # [B*N*H*D, 1] float32
        LSE:       fx.Tensor,   # [B*H*M, 1] float32
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

        n_start = n_tile * BLOCK_N

        wave        = fx.Index(tid // WARP_SIZE)
        lane        = fx.Index(tid % WARP_SIZE)
        lane_mod_32 = fx.Index(lane % 32)
        lane_div_32 = fx.Index(lane // 32)

        wave_n_sub  = fx.Index(wave // WAVE_D_TILES)
        wave_d_sub  = fx.Index(wave % WAVE_D_TILES)

        # ---- Buffers ----
        dV_buf  = fx.rocdl.make_buffer_tensor(dV)
        LSE_buf = fx.rocdl.make_buffer_tensor(LSE)

        copy_16b  = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
        copy_f32  = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        store_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)

        # MLIR types — must be created inside kernel context
        v16f32_type = Vec.make_type(16, fx.Float32)
        v8elem_type = Vec.make_type(MFMA_LK, elem_dtype)

        # ---- LDS ----
        base_ptr = allocator.get_base()
        lds_q   = SmemPtr(base_ptr, lds_q_off,   elem_dtype.ir_type, shape=(LDS_Q_ELEMS,)).get()
        lds_do  = SmemPtr(base_ptr, lds_do_off,  elem_dtype.ir_type, shape=(LDS_DO_ELEMS,)).get()
        lds_lse = SmemPtr(base_ptr, lds_lse_off, fx.Float32.ir_type, shape=(LDS_LSE_ELEMS,)).get()

        # ---- Helpers ----
        def _q_row(q_pos):
            return fx.Int32(batch_idx * (seq_M_idx * n_heads_idx) + q_pos * n_heads_idx + head_idx)

        def _kv_row(kv_pos):
            return fx.Int32(batch_idx * (seq_N_idx * n_heads_idx) + kv_pos * n_heads_idx + head_idx)

        def _lse_row(q_pos):
            return fx.Int32(bh_idx * seq_M_idx + q_pos)

        def _load_global_vec_cv(rsrc, row_i32, col_offset_idx):
            """128-bit load: MFMA_LK=8 bf16 via buffer_load (flat index).
            rsrc: buffer resource from buffer_ops.create_buffer_resource
            row_i32: i32 row index
            col_offset_idx: index column offset (elements, not bytes)
            Flat element index = row * D + col_offset.
            """
            from flydsl.expr import buffer_ops as _bops
            stride_idx = fx.Index(D)
            flat_elem  = fx.Index(row_i32) * stride_idx + col_offset_idx
            return _bops.buffer_load(rsrc, flat_elem, vec_width=MFMA_LK, dtype=elem_dtype)

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

        # Select MFMA function at compile time (Python string comparison, not runtime if)
        _mfma_fn = fx.rocdl.mfma_f32_32x32x16_bf16 if dtype_str == "bf16" \
                   else fx.rocdl.mfma_f32_32x32x16_f16

        def mfma(a_pack, b_pack, c_acc):
            """Call MFMA32x32x16 — _mfma_fn selected at Python compile time."""
            return _mfma_fn(v16f32_type, [a_pack, b_pack, c_acc])

        def _lds_load_pack_a(lds_arr, base_row_in_tile, k_step):
            """Load v8bf16 A-operand pack for MFMA from LDS.
            MFMA32x32x16 A-layout: lane j -> A[j%32, (j//32)*8 + 0..7]
            """
            lds_row = fx.Index(base_row_in_tile) + lane_mod_32
            lds_col = fx.Index(k_step * MFMA_K) + lane_div_32 * MFMA_LK
            lds_idx = lds_row * D + lds_col
            return Vec.load(v8elem_type, lds_arr, [lds_idx]).ir_value()

        # ---- Pre-load K packs for this wave's N sub-tile ----
        # K[wave_n_sub*32 + lane_mod_32, k_step*16 + lane_div_32*8 + 0..7]
        # Same A-operand layout: K^T used as B-operand in S = Q @ K^T
        # (MFMA: C = A*B, A=[M,K], B=[N,K] -> C=[M,N]; so K acts as B^T=[D,N] -> use K as B)
        # Create buffer resources for flat element-indexed loads
        from flydsl.expr import buffer_ops as _bops
        q_rsrc  = _bops.create_buffer_resource(Q)
        k_rsrc  = _bops.create_buffer_resource(K)
        do_rsrc = _bops.create_buffer_resource(dO)

        n_global_wave_base = n_start + wave_n_sub * 32
        n_row_abs_kv = n_global_wave_base + lane_mod_32
        n_valid_kv   = n_row_abs_kv < seq_N_idx
        n_safe_kv    = n_valid_kv.select(n_row_abs_kv, seq_N_idx - fx.Index(1))
        kv_row_g_pre = _kv_row(n_safe_kv)   # i32 global row

        # Pre-load K and V packs (one per K-step, lane_div_32 selects which half)
        # col_offset = ks * MFMA_K + lane_div_32 * MFMA_LK  (all index arithmetic)
        # Pre-load K packs for this wave's N sub-tile (K^T is B-operand in S = Q @ K^T)
        k_packs = []
        for ks in range_constexpr(K_STEPS):
            col_off = fx.Index(ks * MFMA_K) + lane_div_32 * MFMA_LK
            k_packs.append(_load_global_vec_cv(k_rsrc, kv_row_g_pre, col_off))

        # N bounds check for this wave's N sub-tile
        n_row_check = n_global_wave_base + lane_mod_32
        n_valid     = n_row_check < seq_N_idx

        # ---- dV accumulator: Vec[16, f32] zero vector ----
        # Use Vec.filled inside the kernel (MLIR context active).
        # flash_attn_generic uses this same pattern successfully.
        # Use a dummy scalar to ensure init is multi-element (single-element may not return list)
        dv_init   = Vec.filled(16, 0.0, fx.Float32)
        dummy_val = fx.Float32(0.0)
        init_dv   = [dv_init, dummy_val]

        loop_results = init_dv
        for m_tile, iter_args in range(fx.Index(0), n_M_tiles_idx, fx.Index(1), init=init_dv):
            dv_acc = iter_args[0]
            # iter_args[1] is dummy, unused

            m_start = m_tile * BLOCK_M

            # ---- Cooperative LDS load: Q and dO tiles ----
            # BLOCK_SIZE=256 threads, BLOCK_M*D/(MFMA_LK) vec-loads per tile
            # Each thread loads one row worth of vecs per iteration
            VEC_COLS    = D // MFMA_LK     # = D/8, number of vec-columns per row
            ROWS_PER_WAVE_LD = BLOCK_M // NUM_WAVES  # rows per wave for loading
            for row_off in range_constexpr(ROWS_PER_WAVE_LD):
                row_in_tile = wave * ROWS_PER_WAVE_LD + row_off
                m_global_ld = m_start + row_in_tile
                m_valid_ld  = m_global_ld < seq_M_idx
                m_safe_ld   = m_valid_ld.select(m_global_ld, seq_M_idx - fx.Index(1))
                q_row_g     = _q_row(m_safe_ld)
                for cv in range_constexpr(VEC_COLS):
                    col_off_ld = fx.Index(cv * MFMA_LK)   # compile-time offset, no lane_div_32
                    q_vec  = _load_global_vec_cv(q_rsrc,  q_row_g, col_off_ld)
                    do_vec = _load_global_vec_cv(do_rsrc, q_row_g, col_off_ld)
                    lds_base = row_in_tile * D + cv * MFMA_LK
                    Vec(q_vec).store(lds_q,  [lds_base])
                    Vec(do_vec).store(lds_do, [lds_base])

            # ---- Cooperative LSE tile stage: BLOCK_M f32 values, once per m_tile ----
            # Avoids the 32x redundant global LSE reload (all lanes sharing lane_mod_32
            # otherwise fetch the same LSE[m]). First BLOCK_M threads each load one row.
            tid_idx = fx.Index(tid)
            if tid_idx < fx.Index(BLOCK_M):
                m_g_lse  = m_start + tid_idx
                m_ok_lse = m_g_lse < seq_M_idx
                m_sf_lse = m_ok_lse.select(m_g_lse, seq_M_idx - fx.Index(1))
                lse_g    = _load_f32_row(LSE_buf, _lse_row(m_sf_lse))
                Vec.from_elements([lse_g], fx.Float32).store(lds_lse, [tid_idx])

            gpu.barrier()

            # ---- Outer loop over M sub-tiles (2 sub-tiles of 32 rows each) ----
            # Each wave processes all M sub-tiles for its fixed [n_sub, d_sub] output.
            # For S = Q[m_sub*32:+32, :] @ K[n_sub*32:+32, :]^T:
            #   -> produces s_acc for [m_sub*32:+32, n_sub*32:+32]
            # For dV += P^T @ dO[m_sub*32:+32, d_sub*32:+32]:
            #   -> accumulates into dv_acc
            # MFMA32x32x16 layouts (CDNA4/gfx950, wave64), verified vs flash_attn_generic:
            #   INPUT operand[free, k]:  free = lane%32,  k = (lane//32)*8 + e  (e=0..7)
            #   OUTPUT C[m, n]        :  m = lane%32,     n = (lane//32)*4 + (r//4)*8 + (r%4)
            M_SUBTILES = BLOCK_M // 32   # = 2
            for m_sub in range_constexpr(M_SUBTILES):
                # ---- S = Q[m_sub, :] @ K[n_sub, :]^T ----
                # A=Q[m,d]: m=lane%32, d=ks*16+lane//32*8+e
                # B=K[n,d]: n=lane%32, d=ks*16+lane//32*8+e   (K^T -> K as B-operand)
                s_acc = Vec.filled(16, 0.0, fx.Float32)
                for ks in range_constexpr(K_STEPS):
                    q_pack = _lds_load_pack_a(lds_q, m_sub * 32, ks)
                    s_acc  = mfma(q_pack, k_packs[ks], s_acc)

                # ---- P = exp2((scale*S - LSE) * log2e), stored to LDS[m_local, n_local] ----
                # MFMA output decode (probed): C[M,N], M = (lane//32)*4+(r//4)*8+(r%4), N = lane%32.
                # For S = Q @ K^T: M = query row (m), N = key col (n).
                #   m_within = (lane//32)*4 + (r//4)*8 + (r%4)   (varies with r)
                #   n_within = lane%32                            (fixed)
                scale_cst = fx.Float32(scale)
                log2e_cst = fx.Float32(_LOG2E)

                n_within  = lane_mod_32
                n_row_abs = n_within + wave_n_sub * 32 + n_start
                n_ok      = n_row_abs < seq_N_idx

                for r in range_constexpr(16):
                    m_within  = lane_div_32 * 4 + ((r // 4) * 8 + (r % 4))
                    m_row_abs = m_within + (m_sub * 32) + m_start
                    m_valid   = m_row_abs < seq_M_idx
                    m_local_lse = m_within + (m_sub * 32)   # 0..BLOCK_M-1 within tile
                    lse_val   = Vec.load(Vec.make_type(1, fx.Float32), lds_lse, [m_local_lse])[0]
                    s_val     = Vec(s_acc)[r]
                    s_scaled  = fx.Float32(arith.mulf(_raw(s_val), _raw(scale_cst), fastmath=fm))
                    s_sub_lse = fx.Float32(arith.subf(_raw(s_scaled), _raw(lse_val), fastmath=fm))
                    p_arg     = fx.Float32(arith.mulf(_raw(s_sub_lse), _raw(log2e_cst), fastmath=fm))
                    p_val     = fx.Float32(fly_math.exp2(p_arg, fastmath=fm))
                    p_val     = (m_valid & n_ok).select(p_val, fx.Float32(0.0))
                    # Store P[m_local, n_local] in [BLOCK_M, BLOCK_N] LDS layout.
                    m_local   = m_within + (m_sub * 32)
                    n_local   = n_within + wave_n_sub * 32
                    lds_idx   = m_local * BLOCK_N + n_local
                    p_vec     = Vec.from_elements([p_val], fx.Float32).to(elem_dtype)
                    p_vec.store(lds_q, [lds_idx])

                gpu.barrier()

                # ---- dV += P^T @ dO  (contract over this m_sub's 32 query rows) ----
                # A=P^T[n,m]=P[m,n]: free=n=lane%32, k=m=ks*16+lane//32*8+e
                # B=dO[m,d]        : free=d=lane%32, k=m=ks*16+lane//32*8+e
                for ks in range_constexpr(32 // MFMA_K):  # 2 steps for 32 m-rows
                    pt_r = fx.make_rmem_tensor(MFMA_LK, elem_dtype)
                    do_r = fx.make_rmem_tensor(MFMA_LK, elem_dtype)
                    for e in range_constexpr(MFMA_LK):
                        m_local = lane_div_32 * MFMA_LK + (m_sub * 32 + ks * MFMA_K + e)
                        # A: P[m_local, n_local], n_local = wave_n_sub*32 + lane%32
                        n_local = lane_mod_32 + wave_n_sub * 32
                        p_sc    = Vec.load(Vec.make_type(1, elem_dtype), lds_q,  [m_local * BLOCK_N + n_local])[0]
                        fx.memref_store(p_sc, pt_r, e)
                        # B: dO[m_local, d_local], d_local = wave_d_sub*32 + lane%32
                        d_local = lane_mod_32 + wave_d_sub * 32
                        do_sc   = Vec.load(Vec.make_type(1, elem_dtype), lds_do, [m_local * D + d_local])[0]
                        fx.memref_store(do_sc, do_r, e)
                    dv_acc = mfma(fx.memref_load_vec(pt_r), fx.memref_load_vec(do_r), dv_acc)

                gpu.barrier()

            loop_results = yield [dv_acc, dummy_val]

        # ---- Store dV ----
        # dV MFMA output C[m,n]: m = free-dim of A=P^T = N-key (dV row),
        #                        n = free-dim of B=dO  = D    (dV col).
        # Output layout: lane j, reg r -> C[m, n] with
        #   m (dV row) = lane%32                                    (fixed across r)
        #   n (dV col) = (lane//32)*4 + (r//4)*8 + (r%4)            (varies with r)
        # dV = P^T @ dO: A=P^T (free=n_key), B=dO (free=d). Output C[M,N]:
        #   M = n_key = (lane//32)*4+(r//4)*8+(r%4)   (varies with r)
        #   N = d     = lane%32                        (fixed)
        dv_final  = loop_results[0]   # vector<16xf32>
        d_col_abs = lane_mod_32 + wave_d_sub * 32
        for r in range_constexpr(16):
            n_within  = lane_div_32 * 4 + ((r // 4) * 8 + (r % 4))
            n_row_abs = n_within + wave_n_sub * 32 + n_start
            n_ok      = n_row_abs < seq_N_idx
            n_safe    = n_ok.select(n_row_abs, seq_N_idx - fx.Index(1))
            kv_row_g  = _kv_row(n_safe)
            flat_dv   = fx.Int32(fx.Index(kv_row_g) * fx.Index(D) + d_col_abs)
            val_f32   = Vec(dv_final)[r]
            if n_ok:
                _store_f32_row(dV_buf, flat_dv, val_f32)

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
        from flydsl.compiler.kernel_function import CompilationContext
        from flydsl._mlir import ir
        allocator.finalized = False
        _ctx = CompilationContext.get_current()
        with ir.InsertionPoint(_ctx.gpu_module_body):
            allocator.finalize()

        num_N_tiles = (fx.Index(N) + BLOCK_N - 1) // BLOCK_N
        grid_x = fx.Int32(fx.Index(B) * fx.Index(H) * num_N_tiles)
        fmha_bwd_dv_mfma_kernel(
            Q, K, dO, dV, LSE,
            M, N, H, n_M_tiles,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    return launch_fn


def compile_fmha_bwd_dk_mfma(
    *,
    D: int = 64,
    dtype_str: str = "bf16",
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    scale: float = None,
):
    """Phase B.2: dK with MFMA. Grid/block identical to dV.

    dK[n,d] = sum_m dS[m,n] * Q[m,d],  dS = scale * P * (dP - D_vec[m])
      S  = Q @ K^T          (GEMM1a, output P after softmax)
      dP = dO @ V^T         (GEMM1b, same output layout as S)
      dS = scale*P*(dP-Dm)  (elementwise on the shared output register layout)
      dK = dS^T @ Q         (GEMM2, transpose bridge via LDS like dV)

    Returns:
        launch_fn(Q, K, V, dO, dK, LSE, D_vec, B, M, N, H, n_M_tiles, stream)
          Q, K, V, dO : [B*seq*H, D]   int16
          dK          : [B*N*H*D, 1]   float32
          LSE, D_vec  : [B*H*M, 1] / [B*M*H, 1] float32
    """
    import math as _pm
    if scale is None:
        scale = 1.0 / _pm.sqrt(D)
    assert D == BLOCK_N, f"Phase B.2 requires D == BLOCK_N, got D={D}, BLOCK_N={BLOCK_N}"
    assert D % 16 == 0

    elem_dtype = dtype_to_elem_type(dtype_str)
    elem_bits  = 16
    MFMA_K     = 16
    MFMA_LK    = 8
    K_STEPS    = D // MFMA_K
    fm         = arith.FastMathFlags.fast

    BLOCK_SIZE = 256
    NUM_WAVES  = BLOCK_SIZE // WARP_SIZE
    WAVE_N_TILES = BLOCK_N // 32
    WAVE_D_TILES = D // 32
    assert WAVE_N_TILES * WAVE_D_TILES == NUM_WAVES

    # LDS: Q tile + dO tile + dS scratch [BLOCK_M, BLOCK_N] (Q must survive GEMM2)
    # + LSE and D_vec f32 tiles staged once per m_tile (avoid 32x redundant global reload).
    LDS_Q_ELEMS   = BLOCK_M * D
    LDS_DO_ELEMS  = BLOCK_M * D
    LDS_DS_ELEMS  = BLOCK_M * BLOCK_N
    LDS_LSE_ELEMS = BLOCK_M
    LDS_DM_ELEMS  = BLOCK_M

    gpu_arch = "gfx950"
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="fmha_bwd_dk_mfma_smem")
    lds_q_off  = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_q_off + LDS_Q_ELEMS * 2
    lds_do_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_do_off + LDS_DO_ELEMS * 2
    lds_ds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_ds_off + LDS_DS_ELEMS * 2
    lds_lse_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_lse_off + LDS_LSE_ELEMS * 4
    lds_dm_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_dm_off + LDS_DM_ELEMS * 4

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def fmha_bwd_dk_mfma_kernel(  # noqa: F811
        Q:         fx.Tensor,
        K:         fx.Tensor,
        V:         fx.Tensor,
        dO:        fx.Tensor,
        dK:        fx.Tensor,
        LSE:       fx.Tensor,
        D_vec:     fx.Tensor,
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

        n_start = n_tile * BLOCK_N

        wave        = fx.Index(tid // WARP_SIZE)
        lane        = fx.Index(tid % WARP_SIZE)
        lane_mod_32 = fx.Index(lane % 32)
        lane_div_32 = fx.Index(lane // 32)
        wave_n_sub  = fx.Index(wave // WAVE_D_TILES)
        wave_d_sub  = fx.Index(wave % WAVE_D_TILES)

        dK_buf   = fx.rocdl.make_buffer_tensor(dK)
        LSE_buf  = fx.rocdl.make_buffer_tensor(LSE)
        Dvec_buf = fx.rocdl.make_buffer_tensor(D_vec)

        copy_f32  = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        store_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)

        v8elem_type = Vec.make_type(MFMA_LK, elem_dtype)
        v16f32_type = Vec.make_type(16, fx.Float32)

        base_ptr = allocator.get_base()
        lds_q   = SmemPtr(base_ptr, lds_q_off,   elem_dtype.ir_type, shape=(LDS_Q_ELEMS,)).get()
        lds_do  = SmemPtr(base_ptr, lds_do_off,  elem_dtype.ir_type, shape=(LDS_DO_ELEMS,)).get()
        lds_ds  = SmemPtr(base_ptr, lds_ds_off,  elem_dtype.ir_type, shape=(LDS_DS_ELEMS,)).get()
        lds_lse = SmemPtr(base_ptr, lds_lse_off, fx.Float32.ir_type, shape=(LDS_LSE_ELEMS,)).get()
        lds_dm  = SmemPtr(base_ptr, lds_dm_off,  fx.Float32.ir_type, shape=(LDS_DM_ELEMS,)).get()

        def _q_row(q_pos):
            return fx.Int32(batch_idx * (seq_M_idx * n_heads_idx) + q_pos * n_heads_idx + head_idx)

        def _kv_row(kv_pos):
            return fx.Int32(batch_idx * (seq_N_idx * n_heads_idx) + kv_pos * n_heads_idx + head_idx)

        def _lse_row(q_pos):
            return fx.Int32(bh_idx * seq_M_idx + q_pos)

        def _dvec_row(q_pos):
            # D_vec layout [B*M*H, 1]: same row order as Q.
            return _q_row(q_pos)

        from flydsl.expr import buffer_ops as _bops
        q_rsrc  = _bops.create_buffer_resource(Q)
        k_rsrc  = _bops.create_buffer_resource(K)
        v_rsrc  = _bops.create_buffer_resource(V)
        do_rsrc = _bops.create_buffer_resource(dO)

        def _load_global_vec_cv(rsrc, row_i32, col_offset_idx):
            stride_idx = fx.Index(D)
            flat_elem  = fx.Index(row_i32) * stride_idx + col_offset_idx
            return _bops.buffer_load(rsrc, flat_elem, vec_width=MFMA_LK, dtype=elem_dtype)

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

        def _lds_load_pack_a(lds_arr, base_row_in_tile, k_step):
            lds_row = fx.Index(base_row_in_tile) + lane_mod_32
            lds_col = fx.Index(k_step * MFMA_K) + lane_div_32 * MFMA_LK
            return Vec.load(v8elem_type, lds_arr, [lds_row * D + lds_col]).ir_value()

        _mfma_fn = fx.rocdl.mfma_f32_32x32x16_bf16 if dtype_str == "bf16" \
                   else fx.rocdl.mfma_f32_32x32x16_f16

        def mfma(a_pack, b_pack, c_acc):
            return _mfma_fn(v16f32_type, [a_pack, b_pack, c_acc])

        # ---- Pre-load K and V packs for this wave's N sub-tile ----
        # K is B-operand of S=Q@K^T; V is B-operand of dP=dO@V^T. Same layout.
        n_global_wave_base = n_start + wave_n_sub * 32
        n_row_abs_kv = n_global_wave_base + lane_mod_32
        n_valid_kv   = n_row_abs_kv < seq_N_idx
        n_safe_kv    = n_valid_kv.select(n_row_abs_kv, seq_N_idx - fx.Index(1))
        kv_row_g_pre = _kv_row(n_safe_kv)

        k_packs = []
        v_packs = []
        for ks in range_constexpr(K_STEPS):
            col_off = fx.Index(ks * MFMA_K) + lane_div_32 * MFMA_LK
            k_packs.append(_load_global_vec_cv(k_rsrc, kv_row_g_pre, col_off))
            v_packs.append(_load_global_vec_cv(v_rsrc, kv_row_g_pre, col_off))

        dk_init   = Vec.filled(16, 0.0, fx.Float32)
        dummy_val = fx.Float32(0.0)
        init_dk   = [dk_init, dummy_val]

        loop_results = init_dk
        for m_tile, iter_args in range(fx.Index(0), n_M_tiles_idx, fx.Index(1), init=init_dk):
            dk_acc = iter_args[0]
            m_start = m_tile * BLOCK_M

            # ---- Cooperative LDS load: Q and dO tiles ----
            VEC_COLS         = D // MFMA_LK
            ROWS_PER_WAVE_LD = BLOCK_M // NUM_WAVES
            for row_off in range_constexpr(ROWS_PER_WAVE_LD):
                row_in_tile = wave * ROWS_PER_WAVE_LD + row_off
                m_global_ld = m_start + row_in_tile
                m_valid_ld  = m_global_ld < seq_M_idx
                m_safe_ld   = m_valid_ld.select(m_global_ld, seq_M_idx - fx.Index(1))
                q_row_g     = _q_row(m_safe_ld)
                for cv in range_constexpr(VEC_COLS):
                    col_off_ld = fx.Index(cv * MFMA_LK)
                    q_vec  = _load_global_vec_cv(q_rsrc,  q_row_g, col_off_ld)
                    do_vec = _load_global_vec_cv(do_rsrc, q_row_g, col_off_ld)
                    lds_base = row_in_tile * D + cv * MFMA_LK
                    Vec(q_vec).store(lds_q,  [lds_base])
                    Vec(do_vec).store(lds_do, [lds_base])

            # ---- Cooperative LSE + D_vec tile stage (once per m_tile) ----
            tid_idx = fx.Index(tid)
            if tid_idx < fx.Index(BLOCK_M):
                m_g_ls   = m_start + tid_idx
                m_ok_ls  = m_g_ls < seq_M_idx
                m_sf_ls  = m_ok_ls.select(m_g_ls, seq_M_idx - fx.Index(1))
                lse_g    = _load_f32_row(LSE_buf,  _lse_row(m_sf_ls))
                dm_g     = _load_f32_row(Dvec_buf, _dvec_row(m_sf_ls))
                Vec.from_elements([lse_g], fx.Float32).store(lds_lse, [tid_idx])
                Vec.from_elements([dm_g],  fx.Float32).store(lds_dm,  [tid_idx])

            gpu.barrier()

            scale_cst = fx.Float32(scale)
            log2e_cst = fx.Float32(_LOG2E)
            M_SUBTILES = BLOCK_M // 32
            for m_sub in range_constexpr(M_SUBTILES):
                # ---- GEMM1a: S = Q @ K^T ; GEMM1b: dP = dO @ V^T ----
                s_acc  = Vec.filled(16, 0.0, fx.Float32)
                dp_acc = Vec.filled(16, 0.0, fx.Float32)
                for ks in range_constexpr(K_STEPS):
                    q_pack  = _lds_load_pack_a(lds_q,  m_sub * 32, ks)
                    do_pack = _lds_load_pack_a(lds_do, m_sub * 32, ks)
                    s_acc   = mfma(q_pack,  k_packs[ks], s_acc)
                    dp_acc  = mfma(do_pack, v_packs[ks], dp_acc)

                # ---- dS = scale * P * (dP - D_vec[m]) ; store to LDS[m,n] ----
                # P and dP share output layout: M=(lane//32)*4+(r//4)*8+(r%4), N=lane%32.
                n_within  = lane_mod_32
                n_row_abs = n_within + wave_n_sub * 32 + n_start
                n_ok      = n_row_abs < seq_N_idx
                for r in range_constexpr(16):
                    m_within  = lane_div_32 * 4 + ((r // 4) * 8 + (r % 4))
                    m_row_abs = m_within + (m_sub * 32) + m_start
                    m_valid   = m_row_abs < seq_M_idx
                    m_local_f = m_within + (m_sub * 32)   # 0..BLOCK_M-1 within tile
                    lse_val   = Vec.load(Vec.make_type(1, fx.Float32), lds_lse, [m_local_f])[0]
                    dm_val    = Vec.load(Vec.make_type(1, fx.Float32), lds_dm,  [m_local_f])[0]
                    s_val     = Vec(s_acc)[r]
                    dp_val    = Vec(dp_acc)[r]
                    s_scaled  = fx.Float32(arith.mulf(_raw(s_val), _raw(scale_cst), fastmath=fm))
                    s_sub_lse = fx.Float32(arith.subf(_raw(s_scaled), _raw(lse_val), fastmath=fm))
                    p_arg     = fx.Float32(arith.mulf(_raw(s_sub_lse), _raw(log2e_cst), fastmath=fm))
                    p_val     = fx.Float32(fly_math.exp2(p_arg, fastmath=fm))
                    valid_mn  = m_valid & n_ok
                    p_val     = valid_mn.select(p_val, fx.Float32(0.0))
                    # dS = scale * P * (dP - D_vec)
                    dp_sub    = fx.Float32(arith.subf(_raw(dp_val), _raw(dm_val), fastmath=fm))
                    ds_val    = fx.Float32(arith.mulf(_raw(scale_cst), _raw(p_val), fastmath=fm))
                    ds_val    = fx.Float32(arith.mulf(_raw(ds_val), _raw(dp_sub), fastmath=fm))
                    ds_val    = valid_mn.select(ds_val, fx.Float32(0.0))
                    m_local   = m_within + (m_sub * 32)
                    n_local   = n_within + wave_n_sub * 32
                    ds_vec    = Vec.from_elements([ds_val], fx.Float32).to(elem_dtype)
                    ds_vec.store(lds_ds, [m_local * BLOCK_N + n_local])

                gpu.barrier()

                # ---- dK += dS^T @ Q  (contract over this m_sub's 32 query rows) ----
                # A=dS^T[n,m]=dS[m,n]: free=n=lane%32, k=m=ks*16+lane//32*8+e
                # B=Q[m,d]           : free=d=lane%32, k=m=ks*16+lane//32*8+e
                for ks in range_constexpr(32 // MFMA_K):
                    dst_r = fx.make_rmem_tensor(MFMA_LK, elem_dtype)
                    q_r   = fx.make_rmem_tensor(MFMA_LK, elem_dtype)
                    for e in range_constexpr(MFMA_LK):
                        m_local = lane_div_32 * MFMA_LK + (m_sub * 32 + ks * MFMA_K + e)
                        n_local = lane_mod_32 + wave_n_sub * 32
                        ds_sc   = Vec.load(Vec.make_type(1, elem_dtype), lds_ds, [m_local * BLOCK_N + n_local])[0]
                        fx.memref_store(ds_sc, dst_r, e)
                        d_local = lane_mod_32 + wave_d_sub * 32
                        q_sc    = Vec.load(Vec.make_type(1, elem_dtype), lds_q, [m_local * D + d_local])[0]
                        fx.memref_store(q_sc, q_r, e)
                    dk_acc = mfma(fx.memref_load_vec(dst_r), fx.memref_load_vec(q_r), dk_acc)

                gpu.barrier()

            loop_results = yield [dk_acc, dummy_val]

        # ---- Store dK ----  (same output decode as dV: M=n_key varies, N=d fixed)
        dk_final  = loop_results[0]
        d_col_abs = lane_mod_32 + wave_d_sub * 32
        for r in range_constexpr(16):
            n_within  = lane_div_32 * 4 + ((r // 4) * 8 + (r % 4))
            n_row_abs = n_within + wave_n_sub * 32 + n_start
            n_ok      = n_row_abs < seq_N_idx
            n_safe    = n_ok.select(n_row_abs, seq_N_idx - fx.Index(1))
            kv_row_g  = _kv_row(n_safe)
            flat_dk   = fx.Int32(fx.Index(kv_row_g) * fx.Index(D) + d_col_abs)
            val_f32   = Vec(dk_final)[r]
            if n_ok:
                _store_f32_row(dK_buf, flat_dk, val_f32)

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
        from flydsl.compiler.kernel_function import CompilationContext
        from flydsl._mlir import ir
        allocator.finalized = False
        _ctx = CompilationContext.get_current()
        with ir.InsertionPoint(_ctx.gpu_module_body):
            allocator.finalize()

        num_N_tiles = (fx.Index(N) + BLOCK_N - 1) // BLOCK_N
        grid_x = fx.Int32(fx.Index(B) * fx.Index(H) * num_N_tiles)
        fmha_bwd_dk_mfma_kernel(
            Q, K, V, dO, dK, LSE, D_vec,
            M, N, H, n_M_tiles,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    return launch_fn


def compile_fmha_bwd_dvdk_mfma(
    *,
    D: int = 64,
    dtype_str: str = "bf16",
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    scale: float = None,
    use_trload: bool = False,
    use_pipeline: bool = False,
):
    """Phase B.4: FUSED dV + dK with MFMA (matches CK's dV/dK fusion).

    use_trload=True: GEMM2 B-operand (dO/Q) is read via the hardware LDS-transpose
    ds_read_b64_tr_b16 instead of the 8x scalar gather. Requires EVEN LDS_Q_STRIDE
    (odd stride breaks the tr 64-bit column alignment). The A-operand (P^T/dS^T) is
    re-ordered to the transpose's P8 k-permutation so contraction stays aligned.

    Both dV and dK grid over N-tiles and share S, dP, P, dS. Computing them in
    one kernel eliminates a redundant S/dP GEMM pass vs running dv+dk separately.
      S  = Q @ K^T,  dP = dO @ V^T          (GEMM1a/b, shared)
      P  = softmax(S);  dS = scale*P*(dP-Dm) (both stored to LDS)
      dV = P^T  @ dO                         (GEMM2a)
      dK = dS^T @ Q                          (GEMM2b)
    Neither output needs atomics (both N-tile-unique). dQ stays a separate kernel
    (grid over M-tiles; fusing it would need atomic-add, blocked by the f32-atomic
    2x bug — same reason CK / Phase A keep dQ split).

    Returns:
        launch_fn(Q, K, V, dO, dV, dK, LSE, D_vec, B, M, N, H, n_M_tiles, stream)
          Q, K, V, dO : [B*seq*H, D]   int16
          dV, dK      : [B*N*H*D, 1]   float32
          LSE, D_vec  : [B*H*M,1] / [B*M*H,1] float32
    """
    import math as _pm
    if scale is None:
        scale = 1.0 / _pm.sqrt(D)
    assert D == BLOCK_N, f"Phase B.4 requires D == BLOCK_N, got D={D}, BLOCK_N={BLOCK_N}"
    assert D % 16 == 0

    elem_dtype = dtype_to_elem_type(dtype_str)
    MFMA_K     = 16
    MFMA_LK    = 8
    K_STEPS    = D // MFMA_K
    fm         = arith.FastMathFlags.fast

    BLOCK_SIZE = 256
    NUM_WAVES  = BLOCK_SIZE // WARP_SIZE
    WAVE_N_TILES = BLOCK_N // 32
    WAVE_D_TILES = D // 32
    assert WAVE_N_TILES * WAVE_D_TILES == NUM_WAVES

    # LDS layout:
    # Q/dO: [M, LDS_Q_STRIDE] row-major, padded stride for bank-conflict-free scatter.
    # P/dS: TRANSPOSED [N, LDS_MPAD] with padded stride for vectorized GEMM2 A-reads.
    # LSE, D_vec: [BLOCK_M] f32 scalars.
    # Bank analysis for Q/dO scatter (GEMM2): (m*LDS_Q_STRIDE+d)/2%32.
    # S=D+2=66: 16 consecutive m-rows map to 16 distinct banks — zero conflicts.
    # S=66 is 4-byte aligned (m*132%4=0), enabling ds_read_b64 (v4 f16) for GEMM1.
    LDS_MPAD      = BLOCK_M + 8   # P/dS transposed stride padding
    # Q/dO row stride: baseline uses D+2 (odd for D=64) for bank-conflict-free scalar scatter;
    # trload needs EVEN stride (D+8) so ds_read_b64_tr keeps 64-bit column alignment.
    LDS_Q_STRIDE  = (D + 8) if use_trload else (D + 2)
    LDS_Q_ELEMS   = BLOCK_M * LDS_Q_STRIDE
    LDS_DO_ELEMS  = BLOCK_M * LDS_Q_STRIDE
    LDS_DS_ELEMS  = BLOCK_N * LDS_MPAD
    LDS_P_ELEMS   = BLOCK_N * LDS_MPAD
    LDS_LSE_ELEMS = BLOCK_M
    LDS_DM_ELEMS  = BLOCK_M

    gpu_arch = "gfx950"
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="fmha_bwd_dvdk_mfma_smem")
    lds_q_off  = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_q_off + LDS_Q_ELEMS * 2
    lds_do_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_do_off + LDS_DO_ELEMS * 2
    lds_ds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_ds_off + LDS_DS_ELEMS * 2
    lds_p_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_p_off + LDS_P_ELEMS * 2
    lds_lse_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_lse_off + LDS_LSE_ELEMS * 4
    lds_dm_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_dm_off + LDS_DM_ELEMS * 4

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def fmha_bwd_dvdk_mfma_kernel(  # noqa: F811
        Q:         fx.Tensor,
        K:         fx.Tensor,
        V:         fx.Tensor,
        dO:        fx.Tensor,
        dV:        fx.Tensor,
        dK:        fx.Tensor,
        LSE:       fx.Tensor,
        D_vec:     fx.Tensor,
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

        n_start = n_tile * BLOCK_N

        wave        = fx.Index(tid // WARP_SIZE)
        lane        = fx.Index(tid % WARP_SIZE)
        lane_mod_32 = fx.Index(lane % 32)
        lane_div_32 = fx.Index(lane // 32)
        wave_n_sub  = fx.Index(wave // WAVE_D_TILES)
        wave_d_sub  = fx.Index(wave % WAVE_D_TILES)

        dV_buf   = fx.rocdl.make_buffer_tensor(dV)
        dK_buf   = fx.rocdl.make_buffer_tensor(dK)
        LSE_buf  = fx.rocdl.make_buffer_tensor(LSE)
        Dvec_buf = fx.rocdl.make_buffer_tensor(D_vec)

        copy_f32  = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        store_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)

        v8elem_type = Vec.make_type(MFMA_LK, elem_dtype)
        v16f32_type = Vec.make_type(16, fx.Float32)

        base_ptr = allocator.get_base()
        lds_q   = SmemPtr(base_ptr, lds_q_off,   elem_dtype.ir_type, shape=(LDS_Q_ELEMS,)).get()
        lds_do  = SmemPtr(base_ptr, lds_do_off,  elem_dtype.ir_type, shape=(LDS_DO_ELEMS,)).get()
        lds_ds  = SmemPtr(base_ptr, lds_ds_off,  elem_dtype.ir_type, shape=(LDS_DS_ELEMS,)).get()
        lds_p   = SmemPtr(base_ptr, lds_p_off,   elem_dtype.ir_type, shape=(LDS_P_ELEMS,)).get()
        lds_lse = SmemPtr(base_ptr, lds_lse_off, fx.Float32.ir_type, shape=(LDS_LSE_ELEMS,)).get()
        lds_dm  = SmemPtr(base_ptr, lds_dm_off,  fx.Float32.ir_type, shape=(LDS_DM_ELEMS,)).get()

        def _q_row(q_pos):
            return fx.Int32(batch_idx * (seq_M_idx * n_heads_idx) + q_pos * n_heads_idx + head_idx)

        def _kv_row(kv_pos):
            return fx.Int32(batch_idx * (seq_N_idx * n_heads_idx) + kv_pos * n_heads_idx + head_idx)

        def _lse_row(q_pos):
            return fx.Int32(bh_idx * seq_M_idx + q_pos)

        def _dvec_row(q_pos):
            return _q_row(q_pos)

        from flydsl.expr import buffer_ops as _bops
        q_rsrc  = _bops.create_buffer_resource(Q)
        k_rsrc  = _bops.create_buffer_resource(K)
        v_rsrc  = _bops.create_buffer_resource(V)
        do_rsrc = _bops.create_buffer_resource(dO)

        def _load_global_vec_cv(rsrc, row_i32, col_offset_idx):
            flat_elem = fx.Index(row_i32) * fx.Index(D) + col_offset_idx
            return _bops.buffer_load(rsrc, flat_elem, vec_width=MFMA_LK, dtype=elem_dtype)

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

        v4elem_type = Vec.make_type(MFMA_LK // 2, elem_dtype)

        def _lds_load_pack_a(lds_arr, base_row_in_tile, k_step):
            # Q/dO stored with padded stride LDS_Q_STRIDE=66. Use 2×v4 loads (ds_read_b64,
            # 4-byte aligned) rather than 1 v8 (ds_read_b128, needs 16-byte alignment).
            lds_row = fx.Index(base_row_in_tile) + lane_mod_32
            lds_col_lo = fx.Index(k_step * MFMA_K) + lane_div_32 * MFMA_LK
            lds_col_hi = lds_col_lo + fx.Index(MFMA_LK // 2)
            lo = Vec.load(v4elem_type, lds_arr, [lds_row * LDS_Q_STRIDE + lds_col_lo])
            hi = Vec.load(v4elem_type, lds_arr, [lds_row * LDS_Q_STRIDE + lds_col_hi])
            return Vec(lo).shuffle(Vec(hi), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()

        _mfma_fn = fx.rocdl.mfma_f32_32x32x16_bf16 if dtype_str == "bf16" \
                   else fx.rocdl.mfma_f32_32x32x16_f16

        def mfma(a_pack, b_pack, c_acc):
            return _mfma_fn(v16f32_type, [a_pack, b_pack, c_acc])

        # ---- Pre-load K and V packs for this wave's N sub-tile ----
        n_global_wave_base = n_start + wave_n_sub * 32
        n_row_abs_kv = n_global_wave_base + lane_mod_32
        n_valid_kv   = n_row_abs_kv < seq_N_idx
        n_safe_kv    = n_valid_kv.select(n_row_abs_kv, seq_N_idx - fx.Index(1))
        kv_row_g_pre = _kv_row(n_safe_kv)

        k_packs = []
        v_packs = []
        for ks in range_constexpr(K_STEPS):
            col_off = fx.Index(ks * MFMA_K) + lane_div_32 * MFMA_LK
            k_packs.append(_load_global_vec_cv(k_rsrc, kv_row_g_pre, col_off))
            v_packs.append(_load_global_vec_cv(v_rsrc, kv_row_g_pre, col_off))

        dk_init   = Vec.filled(16, 0.0, fx.Float32)
        dv_init   = Vec.filled(16, 0.0, fx.Float32)
        dummy_val = fx.Float32(0.0)
        init_st   = [dk_init, dv_init, dummy_val]

        loop_results = init_st
        for m_tile, iter_args in range(fx.Index(0), n_M_tiles_idx, fx.Index(1), init=init_st):
            dk_acc = iter_args[0]
            dv_acc = iter_args[1]
            m_start = m_tile * BLOCK_M

            # ---- Cooperative LDS load: Q and dO tiles ----
            VEC_COLS         = D // MFMA_LK
            ROWS_PER_WAVE_LD = BLOCK_M // NUM_WAVES
            if use_pipeline:
                # Lane-distributed cooperative load: the (row_off, cv) work items of
                # this wave are spread across its 64 lanes (baseline had every lane
                # redundantly issue ALL items -> 64x redundant global loads + same-
                # address LDS stores). 32 rows * 8 cvs = 256 items / 64 lanes = 4/lane.
                N_ITEMS_LD     = ROWS_PER_WAVE_LD * VEC_COLS
                ITEMS_PER_LANE = N_ITEMS_LD // WARP_SIZE
                for it in range_constexpr(ITEMS_PER_LANE):
                    item        = lane + fx.Index(it * WARP_SIZE)
                    row_off_i   = item // fx.Index(VEC_COLS)
                    cv_i        = item % fx.Index(VEC_COLS)
                    row_in_tile = wave * ROWS_PER_WAVE_LD + row_off_i
                    m_global_ld = m_start + row_in_tile
                    m_valid_ld  = m_global_ld < seq_M_idx
                    m_safe_ld   = m_valid_ld.select(m_global_ld, seq_M_idx - fx.Index(1))
                    q_row_g     = _q_row(m_safe_ld)
                    col_off_ld  = cv_i * fx.Index(MFMA_LK)
                    q_vec  = _load_global_vec_cv(q_rsrc,  q_row_g, col_off_ld)
                    do_vec = _load_global_vec_cv(do_rsrc, q_row_g, col_off_ld)
                    lds_base = row_in_tile * LDS_Q_STRIDE + col_off_ld
                    Vec(q_vec).store(lds_q,  [lds_base])
                    Vec(do_vec).store(lds_do, [lds_base])
            else:
                for row_off in range_constexpr(ROWS_PER_WAVE_LD):
                    row_in_tile = wave * ROWS_PER_WAVE_LD + row_off
                    m_global_ld = m_start + row_in_tile
                    m_valid_ld  = m_global_ld < seq_M_idx
                    m_safe_ld   = m_valid_ld.select(m_global_ld, seq_M_idx - fx.Index(1))
                    q_row_g     = _q_row(m_safe_ld)
                    for cv in range_constexpr(VEC_COLS):
                        col_off_ld = fx.Index(cv * MFMA_LK)
                        q_vec  = _load_global_vec_cv(q_rsrc,  q_row_g, col_off_ld)
                        do_vec = _load_global_vec_cv(do_rsrc, q_row_g, col_off_ld)
                        lds_base = row_in_tile * LDS_Q_STRIDE + cv * MFMA_LK
                        Vec(q_vec).store(lds_q,  [lds_base])
                        Vec(do_vec).store(lds_do, [lds_base])

            # ---- Cooperative LSE + D_vec tile stage ----
            tid_idx = fx.Index(tid)
            if tid_idx < fx.Index(BLOCK_M):
                m_g_ls   = m_start + tid_idx
                m_ok_ls  = m_g_ls < seq_M_idx
                m_sf_ls  = m_ok_ls.select(m_g_ls, seq_M_idx - fx.Index(1))
                lse_g    = _load_f32_row(LSE_buf,  _lse_row(m_sf_ls))
                dm_g     = _load_f32_row(Dvec_buf, _dvec_row(m_sf_ls))
                Vec.from_elements([lse_g], fx.Float32).store(lds_lse, [tid_idx])
                Vec.from_elements([dm_g],  fx.Float32).store(lds_dm,  [tid_idx])

            gpu.barrier()

            scale_cst = fx.Float32(scale)
            log2e_cst = fx.Float32(_LOG2E)
            M_SUBTILES = BLOCK_M // 32
            for m_sub in range_constexpr(M_SUBTILES):
                # ---- GEMM1a: S = Q @ K^T ; GEMM1b: dP = dO @ V^T ----
                s_acc  = Vec.filled(16, 0.0, fx.Float32)
                dp_acc = Vec.filled(16, 0.0, fx.Float32)
                for ks in range_constexpr(K_STEPS):
                    q_pack  = _lds_load_pack_a(lds_q,  m_sub * 32, ks)
                    do_pack = _lds_load_pack_a(lds_do, m_sub * 32, ks)
                    s_acc   = mfma(q_pack,  k_packs[ks], s_acc)
                    dp_acc  = mfma(do_pack, v_packs[ks], dp_acc)

                # ---- P (for dV) and dS (for dK), both stored to LDS[m,n] ----
                n_within  = lane_mod_32
                n_row_abs = n_within + wave_n_sub * 32 + n_start
                n_ok      = n_row_abs < seq_N_idx
                for r in range_constexpr(16):
                    m_within  = lane_div_32 * 4 + ((r // 4) * 8 + (r % 4))
                    m_row_abs = m_within + (m_sub * 32) + m_start
                    m_valid   = m_row_abs < seq_M_idx
                    m_local_f = m_within + (m_sub * 32)
                    lse_val   = Vec.load(Vec.make_type(1, fx.Float32), lds_lse, [m_local_f])[0]
                    dm_val    = Vec.load(Vec.make_type(1, fx.Float32), lds_dm,  [m_local_f])[0]
                    s_val     = Vec(s_acc)[r]
                    dp_val    = Vec(dp_acc)[r]
                    s_scaled  = fx.Float32(arith.mulf(_raw(s_val), _raw(scale_cst), fastmath=fm))
                    s_sub_lse = fx.Float32(arith.subf(_raw(s_scaled), _raw(lse_val), fastmath=fm))
                    p_arg     = fx.Float32(arith.mulf(_raw(s_sub_lse), _raw(log2e_cst), fastmath=fm))
                    p_val     = fx.Float32(fly_math.exp2(p_arg, fastmath=fm))
                    valid_mn  = m_valid & n_ok
                    p_val     = valid_mn.select(p_val, fx.Float32(0.0))
                    dp_sub    = fx.Float32(arith.subf(_raw(dp_val), _raw(dm_val), fastmath=fm))
                    ds_val    = fx.Float32(arith.mulf(_raw(scale_cst), _raw(p_val), fastmath=fm))
                    ds_val    = fx.Float32(arith.mulf(_raw(ds_val), _raw(dp_sub), fastmath=fm))
                    ds_val    = valid_mn.select(ds_val, fx.Float32(0.0))
                    m_local   = m_within + (m_sub * 32)
                    n_local   = n_within + wave_n_sub * 32
                    Vec.from_elements([p_val],  fx.Float32).to(elem_dtype).store(lds_p,  [n_local * LDS_MPAD + m_local])
                    Vec.from_elements([ds_val], fx.Float32).to(elem_dtype).store(lds_ds, [n_local * LDS_MPAD + m_local])

                gpu.barrier()

                # ---- dV += P^T @ dO ; dK += dS^T @ Q (shared operand loads) ----
                # A=P^T/dS^T[n,m]: free=n=lane%32, k=m; B=dO/Q[m,d]: free=d=lane%32, k=m
                MFMA_KS   = 32 // MFMA_K
                d_local   = lane_mod_32 + wave_d_sub * 32
                n_local_g = lane_mod_32 + wave_n_sub * 32
                for ks in range_constexpr(MFMA_KS):
                    n_local = lane_mod_32 + wave_n_sub * 32
                    if use_trload:
                        # B-operand (dO/Q) via HW transpose. tr yields, per lane, contract
                        # m = P8 = {0,1,2,3,8,9,10,11} + lane_div_32*4 (relative to m_row base),
                        # free d = lane%32. Read row-major [m,d] with EVEN LDS_Q_STRIDE.
                        # tr lane decomposition:
                        tr_k_group  = (lane_mod_32 % 16) // 4   # lane%16 //4 within 32-lane half
                        tr_col_sub  = lane % 4
                        tr_col_half = lane_mod_32 // 16
                        d_col = wave_d_sub * 32 + tr_col_half * 16 + tr_col_sub * 4
                        m_base = m_sub * 32 + ks * MFMA_K + lane_div_32 * 4 + tr_k_group
                        lo = m_base * LDS_Q_STRIDE + d_col
                        hi = lo + 8 * LDS_Q_STRIDE
                        do_a = _ds_read_tr_v4(v4elem_type, lo, lds_do_off)
                        do_b = _ds_read_tr_v4(v4elem_type, hi, lds_do_off)
                        do_pack = Vec(do_a).shuffle(Vec(do_b), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()
                        q_a = _ds_read_tr_v4(v4elem_type, lo, lds_q_off)
                        q_b = _ds_read_tr_v4(v4elem_type, hi, lds_q_off)
                        q_pack = Vec(q_a).shuffle(Vec(q_b), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()
                        # A-operand (P^T/dS^T): must hold the SAME m as B at each hardware slot.
                        # B (tr) gives m = (m_sub*32+ks*16) + lane_div_32*4 + P8[e], P8={0,1,2,3,8,9,10,11}.
                        # So load P^T[n, base_a + {0,1,2,3}] ++ P^T[n, base_a + {8,9,10,11}].
                        base_a = lane_div_32 * 4 + (m_sub * 32 + ks * MFMA_K)
                        p_lo  = Vec.load(v4elem_type, lds_p,  [n_local * LDS_MPAD + base_a])
                        p_hi  = Vec.load(v4elem_type, lds_p,  [n_local * LDS_MPAD + base_a + 8])
                        p_pack = Vec(p_lo).shuffle(Vec(p_hi), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()
                        ds_lo = Vec.load(v4elem_type, lds_ds, [n_local * LDS_MPAD + base_a])
                        ds_hi = Vec.load(v4elem_type, lds_ds, [n_local * LDS_MPAD + base_a + 8])
                        ds_pack = Vec(ds_lo).shuffle(Vec(ds_hi), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()
                        dv_acc = mfma(p_pack,  do_pack, dv_acc)
                        dk_acc = mfma(ds_pack, q_pack,  dk_acc)
                    else:
                        base_m  = lane_div_32 * MFMA_LK + (m_sub * 32 + ks * MFMA_K)
                        p_pack  = Vec.load(v8elem_type, lds_p,  [n_local * LDS_MPAD + base_m]).ir_value()
                        ds_pack = Vec.load(v8elem_type, lds_ds, [n_local * LDS_MPAD + base_m]).ir_value()
                        # B-operand (dO / Q): scatter load with padded stride LDS_Q_STRIDE=D+2
                        # for bank-conflict-free access (16 consecutive m-rows hit 16 distinct banks).
                        do_r  = fx.make_rmem_tensor(MFMA_LK, elem_dtype)
                        q_r   = fx.make_rmem_tensor(MFMA_LK, elem_dtype)
                        for e in range_constexpr(MFMA_LK):
                            m_local = lane_div_32 * MFMA_LK + (m_sub * 32 + ks * MFMA_K + e)
                            do_sc = Vec.load(Vec.make_type(1, elem_dtype), lds_do, [m_local * LDS_Q_STRIDE + d_local])[0]
                            q_sc  = Vec.load(Vec.make_type(1, elem_dtype), lds_q,  [m_local * LDS_Q_STRIDE + d_local])[0]
                            fx.memref_store(do_sc, do_r,  e)
                            fx.memref_store(q_sc,  q_r,   e)
                        dv_acc = mfma(p_pack,  fx.memref_load_vec(do_r), dv_acc)
                        dk_acc = mfma(ds_pack, fx.memref_load_vec(q_r),  dk_acc)

                gpu.barrier()

            loop_results = yield [dk_acc, dv_acc, dummy_val]

        # ---- Store dV and dK ----  (same output decode: M=n_key varies with r, N=d fixed)
        dk_final  = loop_results[0]
        dv_final  = loop_results[1]
        d_col_abs = lane_mod_32 + wave_d_sub * 32
        for r in range_constexpr(16):
            n_within  = lane_div_32 * 4 + ((r // 4) * 8 + (r % 4))
            n_row_abs = n_within + wave_n_sub * 32 + n_start
            n_ok      = n_row_abs < seq_N_idx
            n_safe    = n_ok.select(n_row_abs, seq_N_idx - fx.Index(1))
            kv_row_g  = _kv_row(n_safe)
            flat_col  = fx.Int32(fx.Index(kv_row_g) * fx.Index(D) + d_col_abs)
            if n_ok:
                _store_f32_row(dV_buf, flat_col, Vec(dv_final)[r])
                _store_f32_row(dK_buf, flat_col, Vec(dk_final)[r])

    @flyc.jit
    def launch_fn(
        Q:         fx.Tensor,
        K:         fx.Tensor,
        V:         fx.Tensor,
        dO:        fx.Tensor,
        dV:        fx.Tensor,
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
        from flydsl.compiler.kernel_function import CompilationContext
        from flydsl._mlir import ir
        allocator.finalized = False
        _ctx = CompilationContext.get_current()
        with ir.InsertionPoint(_ctx.gpu_module_body):
            allocator.finalize()

        num_N_tiles = (fx.Index(N) + BLOCK_N - 1) // BLOCK_N
        grid_x = fx.Int32(fx.Index(B) * fx.Index(H) * num_N_tiles)
        fmha_bwd_dvdk_mfma_kernel(
            Q, K, V, dO, dV, dK, LSE, D_vec,
            M, N, H, n_M_tiles,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    return launch_fn


def compile_fmha_bwd_dq_mfma(
    *,
    D: int = 64,
    dtype_str: str = "bf16",
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    scale: float = None,
    use_pipeline: bool = False,
):
    """Phase B.3: dQ with MFMA. Grid over M-tiles, runtime loop over N-tiles.

    dQ[m,d] = sum_n dS[m,n] * K[n,d],  dS = scale * P * (dP - D_vec[m])
      S  = Q @ K^T          (GEMM1a)
      dP = dO @ V^T         (GEMM1b)
      dS = scale*P*(dP-Dm)  (elementwise; store to LDS[m,n])
      dQ = dS @ K           (GEMM2; contract over n)

    Mirror of dK: Q/dO are the fixed register A-packs; K/V stream into LDS each
    N-tile (K used in two layouts -> must be LDS). Each block owns a full M-tile
    and accumulates over all N-tiles in registers -> no atomics.

    Returns:
        launch_fn(Q, K, V, dO, dQ, LSE, D_vec, B, M, N, H, n_N_tiles, stream)
          dQ : [B*M*H*D, 1] float32
    """
    import math as _pm
    if scale is None:
        scale = 1.0 / _pm.sqrt(D)
    assert D == BLOCK_M, f"Phase B.3 requires D == BLOCK_M, got D={D}, BLOCK_M={BLOCK_M}"
    assert D % 16 == 0

    elem_dtype = dtype_to_elem_type(dtype_str)
    MFMA_K     = 16
    MFMA_LK    = 8
    K_STEPS    = D // MFMA_K
    fm         = arith.FastMathFlags.fast

    BLOCK_SIZE = 256
    NUM_WAVES  = BLOCK_SIZE // WARP_SIZE
    WAVE_M_TILES = BLOCK_M // 32
    WAVE_D_TILES = D // 32
    assert WAVE_M_TILES * WAVE_D_TILES == NUM_WAVES

    # LDS: K tile + V tile [BLOCK_N, D] + dS scratch [BLOCK_M, BLOCK_N].
    LDS_K_ELEMS  = BLOCK_N * D
    LDS_V_ELEMS  = BLOCK_N * D
    LDS_DS_ELEMS = BLOCK_M * BLOCK_N

    gpu_arch = "gfx950"
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="fmha_bwd_dq_mfma_smem")
    lds_k_off  = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_k_off + LDS_K_ELEMS * 2
    lds_v_off  = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_v_off + LDS_V_ELEMS * 2
    lds_ds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_ds_off + LDS_DS_ELEMS * 2

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def fmha_bwd_dq_mfma_kernel(  # noqa: F811
        Q:         fx.Tensor,
        K:         fx.Tensor,
        V:         fx.Tensor,
        dO:        fx.Tensor,
        dQ:        fx.Tensor,
        LSE:       fx.Tensor,
        D_vec:     fx.Tensor,
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

        m_start = m_tile * BLOCK_M

        wave        = fx.Index(tid // WARP_SIZE)
        lane        = fx.Index(tid % WARP_SIZE)
        lane_mod_32 = fx.Index(lane % 32)
        lane_div_32 = fx.Index(lane // 32)
        wave_m_sub  = fx.Index(wave // WAVE_D_TILES)
        wave_d_sub  = fx.Index(wave % WAVE_D_TILES)

        dQ_buf   = fx.rocdl.make_buffer_tensor(dQ)
        LSE_buf  = fx.rocdl.make_buffer_tensor(LSE)
        Dvec_buf = fx.rocdl.make_buffer_tensor(D_vec)

        copy_f32  = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        store_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)

        v8elem_type = Vec.make_type(MFMA_LK, elem_dtype)
        v16f32_type = Vec.make_type(16, fx.Float32)

        base_ptr = allocator.get_base()
        lds_k  = SmemPtr(base_ptr, lds_k_off,  elem_dtype.ir_type, shape=(LDS_K_ELEMS,)).get()
        lds_v  = SmemPtr(base_ptr, lds_v_off,  elem_dtype.ir_type, shape=(LDS_V_ELEMS,)).get()
        lds_ds = SmemPtr(base_ptr, lds_ds_off, elem_dtype.ir_type, shape=(LDS_DS_ELEMS,)).get()

        def _q_row(q_pos):
            return fx.Int32(batch_idx * (seq_M_idx * n_heads_idx) + q_pos * n_heads_idx + head_idx)

        def _kv_row(kv_pos):
            return fx.Int32(batch_idx * (seq_N_idx * n_heads_idx) + kv_pos * n_heads_idx + head_idx)

        def _lse_row(q_pos):
            return fx.Int32(bh_idx * seq_M_idx + q_pos)

        def _dvec_row(q_pos):
            return _q_row(q_pos)

        from flydsl.expr import buffer_ops as _bops
        q_rsrc  = _bops.create_buffer_resource(Q)
        k_rsrc  = _bops.create_buffer_resource(K)
        v_rsrc  = _bops.create_buffer_resource(V)
        do_rsrc = _bops.create_buffer_resource(dO)

        def _load_global_vec_cv(rsrc, row_i32, col_offset_idx):
            flat_elem = fx.Index(row_i32) * fx.Index(D) + col_offset_idx
            return _bops.buffer_load(rsrc, flat_elem, vec_width=MFMA_LK, dtype=elem_dtype)

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

        def _lds_load_pack_a(lds_arr, base_row_in_tile, k_step):
            lds_row = fx.Index(base_row_in_tile) + lane_mod_32
            lds_col = fx.Index(k_step * MFMA_K) + lane_div_32 * MFMA_LK
            return Vec.load(v8elem_type, lds_arr, [lds_row * D + lds_col]).ir_value()

        _mfma_fn = fx.rocdl.mfma_f32_32x32x16_bf16 if dtype_str == "bf16" \
                   else fx.rocdl.mfma_f32_32x32x16_f16

        def mfma(a_pack, b_pack, c_acc):
            return _mfma_fn(v16f32_type, [a_pack, b_pack, c_acc])

        # ---- Pre-load Q, dO A-packs for this wave's M sub-tile (constant across N-loop) ----
        # A-operand of S=Q@K^T and dP=dO@V^T: free=m=wave_m_sub*32+lane%32, contract=d.
        m_wave_base = m_start + wave_m_sub * 32
        m_row_abs_q = m_wave_base + lane_mod_32
        m_valid_q   = m_row_abs_q < seq_M_idx
        m_safe_q    = m_valid_q.select(m_row_abs_q, seq_M_idx - fx.Index(1))
        q_row_g_pre = _q_row(m_safe_q)

        q_packs  = []
        do_packs = []
        for ks in range_constexpr(K_STEPS):
            col_off = fx.Index(ks * MFMA_K) + lane_div_32 * MFMA_LK
            q_packs.append(_load_global_vec_cv(q_rsrc,  q_row_g_pre, col_off))
            do_packs.append(_load_global_vec_cv(do_rsrc, q_row_g_pre, col_off))

        # ---- Pre-load per-r LSE, D_vec, m-validity (m varies with r, const across N) ----
        lse_vals   = []
        dvec_vals  = []
        m_valids   = []
        for r in range_constexpr(16):
            m_within  = lane_div_32 * 4 + ((r // 4) * 8 + (r % 4))
            m_row_abs = m_within + wave_m_sub * 32 + m_start
            m_valid   = m_row_abs < seq_M_idx
            m_safe    = m_valid.select(m_row_abs, seq_M_idx - fx.Index(1))
            lse_vals.append(_load_f32_row(LSE_buf, _lse_row(m_safe)))
            dvec_vals.append(_load_f32_row(Dvec_buf, _dvec_row(m_safe)))
            m_valids.append(m_valid)

        scale_cst = fx.Float32(scale)
        log2e_cst = fx.Float32(_LOG2E)

        dq_init   = Vec.filled(16, 0.0, fx.Float32)
        dummy_val = fx.Float32(0.0)
        init_dq   = [dq_init, dummy_val]

        loop_results = init_dq
        for n_tile, iter_args in range(fx.Index(0), n_N_tiles_idx, fx.Index(1), init=init_dq):
            dq_acc = iter_args[0]
            n_start = n_tile * BLOCK_N

            # ---- Cooperative LDS load: K and V tiles [BLOCK_N, D] ----
            VEC_COLS         = D // MFMA_LK
            ROWS_PER_WAVE_LD = BLOCK_N // NUM_WAVES
            if use_pipeline:
                # Lane-distributed: spread (row_off, cv) items across the wave's 64
                # lanes (baseline had every lane redundantly issue ALL items -> 64x
                # redundant global loads + same-address LDS stores).
                N_ITEMS_LD     = ROWS_PER_WAVE_LD * VEC_COLS
                ITEMS_PER_LANE = N_ITEMS_LD // WARP_SIZE
                if ITEMS_PER_LANE < 1:
                    ITEMS_PER_LANE = 1
                for it in range_constexpr(ITEMS_PER_LANE):
                    item        = lane + fx.Index(it * WARP_SIZE)
                    item_ok     = item < fx.Index(N_ITEMS_LD)
                    item_s      = item_ok.select(item, fx.Index(0))
                    row_off_i   = item_s // fx.Index(VEC_COLS)
                    cv_i        = item_s % fx.Index(VEC_COLS)
                    row_in_tile = wave * ROWS_PER_WAVE_LD + row_off_i
                    n_global_ld = n_start + row_in_tile
                    n_valid_ld  = n_global_ld < seq_N_idx
                    n_safe_ld   = n_valid_ld.select(n_global_ld, seq_N_idx - fx.Index(1))
                    kv_row_g    = _kv_row(n_safe_ld)
                    col_off_ld  = cv_i * fx.Index(MFMA_LK)
                    k_vec = _load_global_vec_cv(k_rsrc, kv_row_g, col_off_ld)
                    v_vec = _load_global_vec_cv(v_rsrc, kv_row_g, col_off_ld)
                    lds_base = row_in_tile * D + col_off_ld
                    Vec(k_vec).store(lds_k, [lds_base])
                    Vec(v_vec).store(lds_v, [lds_base])
            else:
                for row_off in range_constexpr(ROWS_PER_WAVE_LD):
                    row_in_tile = wave * ROWS_PER_WAVE_LD + row_off
                    n_global_ld = n_start + row_in_tile
                    n_valid_ld  = n_global_ld < seq_N_idx
                    n_safe_ld   = n_valid_ld.select(n_global_ld, seq_N_idx - fx.Index(1))
                    kv_row_g    = _kv_row(n_safe_ld)
                    for cv in range_constexpr(VEC_COLS):
                        col_off_ld = fx.Index(cv * MFMA_LK)
                        k_vec = _load_global_vec_cv(k_rsrc, kv_row_g, col_off_ld)
                        v_vec = _load_global_vec_cv(v_rsrc, kv_row_g, col_off_ld)
                        lds_base = row_in_tile * D + cv * MFMA_LK
                        Vec(k_vec).store(lds_k, [lds_base])
                        Vec(v_vec).store(lds_v, [lds_base])

            gpu.barrier()

            N_SUBTILES = BLOCK_N // 32
            for n_sub in range_constexpr(N_SUBTILES):
                # ---- GEMM1a: S = Q @ K^T ; GEMM1b: dP = dO @ V^T ----
                # A=Q/dO (free=m), B=K/V (free=n=n_sub*32+lane%32). Output C[m,n].
                s_acc  = Vec.filled(16, 0.0, fx.Float32)
                dp_acc = Vec.filled(16, 0.0, fx.Float32)
                for ks in range_constexpr(K_STEPS):
                    k_pack = _lds_load_pack_a(lds_k, n_sub * 32, ks)
                    v_pack = _lds_load_pack_a(lds_v, n_sub * 32, ks)
                    s_acc  = mfma(q_packs[ks],  k_pack, s_acc)
                    dp_acc = mfma(do_packs[ks], v_pack, dp_acc)

                # ---- dS = scale * P * (dP - D_vec[m]) ; store to LDS[m,n] ----
                n_within  = lane_mod_32
                n_row_abs = n_within + n_sub * 32 + n_start
                n_ok      = n_row_abs < seq_N_idx
                for r in range_constexpr(16):
                    m_within  = lane_div_32 * 4 + ((r // 4) * 8 + (r % 4))
                    lse_val   = lse_vals[r]
                    dm_val    = dvec_vals[r]
                    s_val     = Vec(s_acc)[r]
                    dp_val    = Vec(dp_acc)[r]
                    s_scaled  = fx.Float32(arith.mulf(_raw(s_val), _raw(scale_cst), fastmath=fm))
                    s_sub_lse = fx.Float32(arith.subf(_raw(s_scaled), _raw(lse_val), fastmath=fm))
                    p_arg     = fx.Float32(arith.mulf(_raw(s_sub_lse), _raw(log2e_cst), fastmath=fm))
                    p_val     = fx.Float32(fly_math.exp2(p_arg, fastmath=fm))
                    valid_mn  = m_valids[r] & n_ok
                    p_val     = valid_mn.select(p_val, fx.Float32(0.0))
                    dp_sub    = fx.Float32(arith.subf(_raw(dp_val), _raw(dm_val), fastmath=fm))
                    ds_val    = fx.Float32(arith.mulf(_raw(scale_cst), _raw(p_val), fastmath=fm))
                    ds_val    = fx.Float32(arith.mulf(_raw(ds_val), _raw(dp_sub), fastmath=fm))
                    ds_val    = valid_mn.select(ds_val, fx.Float32(0.0))
                    m_local   = m_within + wave_m_sub * 32
                    n_local   = n_within + n_sub * 32
                    ds_vec    = Vec.from_elements([ds_val], fx.Float32).to(elem_dtype)
                    ds_vec.store(lds_ds, [m_local * BLOCK_N + n_local])

                gpu.barrier()

                # ---- dQ += dS @ K  (contract over this n_sub's 32 key rows) ----
                # A=dS[m,n]: free=m=lane%32, k=n=ks*16+lane//32*8+e
                # B=K[n,d] : free=d=lane%32, k=n=ks*16+lane//32*8+e
                for ks in range_constexpr(32 // MFMA_K):
                    dst_r = fx.make_rmem_tensor(MFMA_LK, elem_dtype)
                    k_r   = fx.make_rmem_tensor(MFMA_LK, elem_dtype)
                    for e in range_constexpr(MFMA_LK):
                        n_local = lane_div_32 * MFMA_LK + (n_sub * 32 + ks * MFMA_K + e)
                        m_local = lane_mod_32 + wave_m_sub * 32
                        ds_sc   = Vec.load(Vec.make_type(1, elem_dtype), lds_ds, [m_local * BLOCK_N + n_local])[0]
                        fx.memref_store(ds_sc, dst_r, e)
                        d_local = lane_mod_32 + wave_d_sub * 32
                        k_sc    = Vec.load(Vec.make_type(1, elem_dtype), lds_k, [n_local * D + d_local])[0]
                        fx.memref_store(k_sc, k_r, e)
                    dq_acc = mfma(fx.memref_load_vec(dst_r), fx.memref_load_vec(k_r), dq_acc)

                gpu.barrier()

            loop_results = yield [dq_acc, dummy_val]

        # ---- Store dQ ----  output C[M,N]: M=m (varies with r), N=d (fixed)
        dq_final  = loop_results[0]
        d_col_abs = lane_mod_32 + wave_d_sub * 32
        for r in range_constexpr(16):
            m_within  = lane_div_32 * 4 + ((r // 4) * 8 + (r % 4))
            m_row_abs = m_within + wave_m_sub * 32 + m_start
            m_ok      = m_row_abs < seq_M_idx
            m_safe    = m_ok.select(m_row_abs, seq_M_idx - fx.Index(1))
            q_row_g   = _q_row(m_safe)
            flat_dq   = fx.Int32(fx.Index(q_row_g) * fx.Index(D) + d_col_abs)
            val_f32   = Vec(dq_final)[r]
            if m_ok:
                _store_f32_row(dQ_buf, flat_dq, val_f32)

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
        from flydsl.compiler.kernel_function import CompilationContext
        from flydsl._mlir import ir
        allocator.finalized = False
        _ctx = CompilationContext.get_current()
        with ir.InsertionPoint(_ctx.gpu_module_body):
            allocator.finalize()

        num_M_tiles = (fx.Index(M) + BLOCK_M - 1) // BLOCK_M
        grid_x = fx.Int32(fx.Index(B) * fx.Index(H) * num_M_tiles)
        fmha_bwd_dq_mfma_kernel(
            Q, K, V, dO, dQ, LSE, D_vec,
            M, N, H, n_N_tiles,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    return launch_fn


def compile_fmha_bwd_dqdkdv_mfma(
    *,
    D: int = 64,
    dtype_str: str = "bf16",
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    scale: float = None,
    use_pipeline: bool = True,
):
    """Phase B.5: FUSED dQ + dV + dK in one N-tile-gridded kernel (matches CK).

    This is compile_fmha_bwd_dvdk_mfma's body (lane-distributed pipeline path)
    plus the dQ GEMM grafted in, so Q/K/V/dO are loaded once and S/dP/P/dS are
    computed once for all three gradients (the fusion win vs running dvdk + dq).

    dV/dK are N-tile-unique -> register-accumulated, plain store (no atomics).
    dQ[m,d] = sum_n dS[m,n]*K[n,d] is SHARED across N-tile blocks -> each block
    contributes a partial via atomic-add into an f32 dQ scratch. (The old belief
    that f32 atomics "double" was a flyc.compile test-harness artifact; atomics
    are correct — see probe_atomic_sweep.py.) A separate convert kernel casts the
    f32 scratch to the output dtype (same as the standalone dq path uses f32).

    Within a block (4 waves, wave=(wave_n_sub,wave_d_sub)): dS[n,m] lives in LDS
    for ALL n. For a given d_sub, the two waves with n_sub in {0,1} each contract
    their own 32 n-rows of dQ[m, d_sub-cols]; atomic-add sums the two halves (and
    across blocks). K is streamed into an [n,d] LDS buffer (the QK k_packs have the
    wrong layout for the dQ contraction).

    Returns:
        launch_fn(Q, K, V, dO, dV, dK, dQ_f32, LSE, D_vec, B, M, N, H, n_M_tiles, stream)
          dQ_f32 : [B*M*H*D, 1] float32 scratch (zero it before launch; convert after)
    """
    import math as _pm
    if scale is None:
        scale = 1.0 / _pm.sqrt(D)
    assert D == BLOCK_N, f"requires D == BLOCK_N, got D={D}, BLOCK_N={BLOCK_N}"
    assert D % 16 == 0

    elem_dtype = dtype_to_elem_type(dtype_str)
    MFMA_K     = 16
    MFMA_LK    = 8
    K_STEPS    = D // MFMA_K
    fm         = arith.FastMathFlags.fast

    BLOCK_SIZE = 256
    NUM_WAVES  = BLOCK_SIZE // WARP_SIZE
    WAVE_N_TILES = BLOCK_N // 32
    WAVE_D_TILES = D // 32
    assert WAVE_N_TILES * WAVE_D_TILES == NUM_WAVES

    LDS_MPAD      = BLOCK_M + 8
    LDS_Q_STRIDE  = (D + 2)          # odd stride: bank-conflict-free scalar scatter
    LDS_Q_ELEMS   = BLOCK_M * LDS_Q_STRIDE
    LDS_DO_ELEMS  = BLOCK_M * LDS_Q_STRIDE
    LDS_DS_ELEMS  = BLOCK_N * LDS_MPAD
    LDS_P_ELEMS   = BLOCK_N * LDS_MPAD
    LDS_K_ELEMS   = BLOCK_N * D       # K in [n,d] layout for the dQ contraction
    LDS_LSE_ELEMS = BLOCK_M
    LDS_DM_ELEMS  = BLOCK_M

    gpu_arch = "gfx950"
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="fmha_bwd_dqdkdv_mfma_smem")
    lds_q_off  = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_q_off + LDS_Q_ELEMS * 2
    lds_do_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_do_off + LDS_DO_ELEMS * 2
    lds_ds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_ds_off + LDS_DS_ELEMS * 2
    lds_p_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_p_off + LDS_P_ELEMS * 2
    lds_k_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_k_off + LDS_K_ELEMS * 2
    lds_lse_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_lse_off + LDS_LSE_ELEMS * 4
    lds_dm_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_dm_off + LDS_DM_ELEMS * 4

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def fmha_bwd_dqdkdv_mfma_kernel(  # noqa: F811
        Q:         fx.Tensor,
        K:         fx.Tensor,
        V:         fx.Tensor,
        dO:        fx.Tensor,
        dV:        fx.Tensor,
        dK:        fx.Tensor,
        dQ_f32:    fx.Tensor,
        LSE:       fx.Tensor,
        D_vec:     fx.Tensor,
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

        n_start = n_tile * BLOCK_N

        wave        = fx.Index(tid // WARP_SIZE)
        lane        = fx.Index(tid % WARP_SIZE)
        lane_mod_32 = fx.Index(lane % 32)
        lane_div_32 = fx.Index(lane // 32)
        wave_n_sub  = fx.Index(wave // WAVE_D_TILES)
        wave_d_sub  = fx.Index(wave % WAVE_D_TILES)

        dV_buf   = fx.rocdl.make_buffer_tensor(dV)
        dK_buf   = fx.rocdl.make_buffer_tensor(dK)
        LSE_buf  = fx.rocdl.make_buffer_tensor(LSE)
        Dvec_buf = fx.rocdl.make_buffer_tensor(D_vec)

        copy_f32  = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        store_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)

        v8elem_type = Vec.make_type(MFMA_LK, elem_dtype)
        v16f32_type = Vec.make_type(16, fx.Float32)

        base_ptr = allocator.get_base()
        lds_q   = SmemPtr(base_ptr, lds_q_off,   elem_dtype.ir_type, shape=(LDS_Q_ELEMS,)).get()
        lds_do  = SmemPtr(base_ptr, lds_do_off,  elem_dtype.ir_type, shape=(LDS_DO_ELEMS,)).get()
        lds_ds  = SmemPtr(base_ptr, lds_ds_off,  elem_dtype.ir_type, shape=(LDS_DS_ELEMS,)).get()
        lds_p   = SmemPtr(base_ptr, lds_p_off,   elem_dtype.ir_type, shape=(LDS_P_ELEMS,)).get()
        lds_k   = SmemPtr(base_ptr, lds_k_off,   elem_dtype.ir_type, shape=(LDS_K_ELEMS,)).get()
        lds_lse = SmemPtr(base_ptr, lds_lse_off, fx.Float32.ir_type, shape=(LDS_LSE_ELEMS,)).get()
        lds_dm  = SmemPtr(base_ptr, lds_dm_off,  fx.Float32.ir_type, shape=(LDS_DM_ELEMS,)).get()

        def _q_row(q_pos):
            return fx.Int32(batch_idx * (seq_M_idx * n_heads_idx) + q_pos * n_heads_idx + head_idx)

        def _kv_row(kv_pos):
            return fx.Int32(batch_idx * (seq_N_idx * n_heads_idx) + kv_pos * n_heads_idx + head_idx)

        def _lse_row(q_pos):
            return fx.Int32(bh_idx * seq_M_idx + q_pos)

        def _dvec_row(q_pos):
            return _q_row(q_pos)

        from flydsl.expr import buffer_ops as _bops
        q_rsrc  = _bops.create_buffer_resource(Q)
        k_rsrc  = _bops.create_buffer_resource(K)
        v_rsrc  = _bops.create_buffer_resource(V)
        do_rsrc = _bops.create_buffer_resource(dO)

        # Raw <4xi32> buffer resource for dQ atomics (raw_buffer_atomic_fadd wants
        # a vector<4xi32> rsrc, NOT the ptr<8> descriptor create_buffer_resource
        # returns). Build the descriptor manually (flash_attn_gfx950.py:741 recipe).
        from flydsl._mlir.dialects import fly as _fly_d
        from flydsl._mlir.dialects import llvm as _llvm_d
        from flydsl._mlir import ir as _ir_d
        from flydsl.expr.typing import T as _T
        _dq_base_ptr = _fly_d.extract_aligned_pointer_as_index(
            _ir_d.Type.parse("!llvm.ptr"), dQ_f32)
        _dq_base_i64 = _llvm_d.PtrToIntOp(_T.i64, _dq_base_ptr).result
        _dq_lo = ArithValue(_dq_base_i64).trunci(_T.i32)
        _dq_hi = ArithValue(ArithValue(_dq_base_i64).shrui(fx.Int64(32))).trunci(_T.i32)
        dq_rsrc = Vec.from_elements(
            [_dq_lo, _dq_hi,
             _bops._create_i32_constant(0xFFFFFFFF),
             _bops._create_i32_constant(_bops._get_buffer_flags())],
            fx.Int32,
        ).ir_value()

        def _load_global_vec_cv(rsrc, row_i32, col_offset_idx):
            flat_elem = fx.Index(row_i32) * fx.Index(D) + col_offset_idx
            return _bops.buffer_load(rsrc, flat_elem, vec_width=MFMA_LK, dtype=elem_dtype)

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

        def _atomic_add_dq(flat_elem, val_f32):
            # f32 atomic add into dQ_f32[flat_elem]; offset is in BYTES.
            rocdl.raw_buffer_atomic_fadd(
                _raw(val_f32), dq_rsrc,
                _raw(fx.Int32(fx.Index(flat_elem) * 4)),
                _raw(fx.Int32(0)), _raw(fx.Int32(0)),
            )

        v4elem_type = Vec.make_type(MFMA_LK // 2, elem_dtype)

        def _lds_load_pack_a(lds_arr, base_row_in_tile, k_step):
            lds_row = fx.Index(base_row_in_tile) + lane_mod_32
            lds_col_lo = fx.Index(k_step * MFMA_K) + lane_div_32 * MFMA_LK
            lds_col_hi = lds_col_lo + fx.Index(MFMA_LK // 2)
            lo = Vec.load(v4elem_type, lds_arr, [lds_row * LDS_Q_STRIDE + lds_col_lo])
            hi = Vec.load(v4elem_type, lds_arr, [lds_row * LDS_Q_STRIDE + lds_col_hi])
            return Vec(lo).shuffle(Vec(hi), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()

        _mfma_fn = fx.rocdl.mfma_f32_32x32x16_bf16 if dtype_str == "bf16" \
                   else fx.rocdl.mfma_f32_32x32x16_f16

        def mfma(a_pack, b_pack, c_acc):
            return _mfma_fn(v16f32_type, [a_pack, b_pack, c_acc])

        # ---- Pre-load K and V packs for this wave's N sub-tile (QK/dP layout) ----
        n_global_wave_base = n_start + wave_n_sub * 32
        n_row_abs_kv = n_global_wave_base + lane_mod_32
        n_valid_kv   = n_row_abs_kv < seq_N_idx
        n_safe_kv    = n_valid_kv.select(n_row_abs_kv, seq_N_idx - fx.Index(1))
        kv_row_g_pre = _kv_row(n_safe_kv)

        k_packs = []
        v_packs = []
        for ks in range_constexpr(K_STEPS):
            col_off = fx.Index(ks * MFMA_K) + lane_div_32 * MFMA_LK
            k_packs.append(_load_global_vec_cv(k_rsrc, kv_row_g_pre, col_off))
            v_packs.append(_load_global_vec_cv(v_rsrc, kv_row_g_pre, col_off))

        # ---- Cooperative LDS load: K tile [BLOCK_N, D] (once per block, for dQ) ----
        VEC_COLS_KV = D // MFMA_LK
        ROWS_PER_WAVE_KV = BLOCK_N // NUM_WAVES
        N_ITEMS_KV = ROWS_PER_WAVE_KV * VEC_COLS_KV
        ITEMS_PER_LANE_KV = N_ITEMS_KV // WARP_SIZE
        if ITEMS_PER_LANE_KV < 1:
            ITEMS_PER_LANE_KV = 1
        for it in range_constexpr(ITEMS_PER_LANE_KV):
            item        = lane + fx.Index(it * WARP_SIZE)
            item_ok     = item < fx.Index(N_ITEMS_KV)
            item_s      = item_ok.select(item, fx.Index(0))
            row_off_i   = item_s // fx.Index(VEC_COLS_KV)
            cv_i        = item_s % fx.Index(VEC_COLS_KV)
            row_in_tile = wave * ROWS_PER_WAVE_KV + row_off_i
            n_global_ld = n_start + row_in_tile
            n_valid_ld  = n_global_ld < seq_N_idx
            n_safe_ld   = n_valid_ld.select(n_global_ld, seq_N_idx - fx.Index(1))
            kv_row_g    = _kv_row(n_safe_ld)
            col_off_ld  = cv_i * fx.Index(MFMA_LK)
            k_vec = _load_global_vec_cv(k_rsrc, kv_row_g, col_off_ld)
            Vec(k_vec).store(lds_k, [row_in_tile * D + col_off_ld])

        dk_init   = Vec.filled(16, 0.0, fx.Float32)
        dv_init   = Vec.filled(16, 0.0, fx.Float32)
        dummy_val = fx.Float32(0.0)
        init_st   = [dk_init, dv_init, dummy_val]

        loop_results = init_st
        for m_tile, iter_args in range(fx.Index(0), n_M_tiles_idx, fx.Index(1), init=init_st):
            dk_acc = iter_args[0]
            dv_acc = iter_args[1]
            m_start = m_tile * BLOCK_M

            # ---- Cooperative LDS load: Q and dO tiles (lane-distributed) ----
            VEC_COLS         = D // MFMA_LK
            ROWS_PER_WAVE_LD = BLOCK_M // NUM_WAVES
            N_ITEMS_LD     = ROWS_PER_WAVE_LD * VEC_COLS
            ITEMS_PER_LANE = N_ITEMS_LD // WARP_SIZE
            for it in range_constexpr(ITEMS_PER_LANE):
                item        = lane + fx.Index(it * WARP_SIZE)
                row_off_i   = item // fx.Index(VEC_COLS)
                cv_i        = item % fx.Index(VEC_COLS)
                row_in_tile = wave * ROWS_PER_WAVE_LD + row_off_i
                m_global_ld = m_start + row_in_tile
                m_valid_ld  = m_global_ld < seq_M_idx
                m_safe_ld   = m_valid_ld.select(m_global_ld, seq_M_idx - fx.Index(1))
                q_row_g     = _q_row(m_safe_ld)
                col_off_ld  = cv_i * fx.Index(MFMA_LK)
                q_vec  = _load_global_vec_cv(q_rsrc,  q_row_g, col_off_ld)
                do_vec = _load_global_vec_cv(do_rsrc, q_row_g, col_off_ld)
                lds_base = row_in_tile * LDS_Q_STRIDE + col_off_ld
                Vec(q_vec).store(lds_q,  [lds_base])
                Vec(do_vec).store(lds_do, [lds_base])

            # ---- Cooperative LSE + D_vec tile stage ----
            tid_idx = fx.Index(tid)
            if tid_idx < fx.Index(BLOCK_M):
                m_g_ls   = m_start + tid_idx
                m_ok_ls  = m_g_ls < seq_M_idx
                m_sf_ls  = m_ok_ls.select(m_g_ls, seq_M_idx - fx.Index(1))
                lse_g    = _load_f32_row(LSE_buf,  _lse_row(m_sf_ls))
                dm_g     = _load_f32_row(Dvec_buf, _dvec_row(m_sf_ls))
                Vec.from_elements([lse_g], fx.Float32).store(lds_lse, [tid_idx])
                Vec.from_elements([dm_g],  fx.Float32).store(lds_dm,  [tid_idx])

            gpu.barrier()

            scale_cst = fx.Float32(scale)
            log2e_cst = fx.Float32(_LOG2E)
            M_SUBTILES = BLOCK_M // 32
            for m_sub in range_constexpr(M_SUBTILES):
                # ---- GEMM1a: S = Q @ K^T ; GEMM1b: dP = dO @ V^T ----
                s_acc  = Vec.filled(16, 0.0, fx.Float32)
                dp_acc = Vec.filled(16, 0.0, fx.Float32)
                for ks in range_constexpr(K_STEPS):
                    q_pack  = _lds_load_pack_a(lds_q,  m_sub * 32, ks)
                    do_pack = _lds_load_pack_a(lds_do, m_sub * 32, ks)
                    s_acc   = mfma(q_pack,  k_packs[ks], s_acc)
                    dp_acc  = mfma(do_pack, v_packs[ks], dp_acc)

                # ---- P (for dV) and dS (for dK/dQ), both stored TRANSPOSED [n,m] ----
                n_within  = lane_mod_32
                n_row_abs = n_within + wave_n_sub * 32 + n_start
                n_ok      = n_row_abs < seq_N_idx
                for r in range_constexpr(16):
                    m_within  = lane_div_32 * 4 + ((r // 4) * 8 + (r % 4))
                    m_row_abs = m_within + (m_sub * 32) + m_start
                    m_valid   = m_row_abs < seq_M_idx
                    m_local_f = m_within + (m_sub * 32)
                    lse_val   = Vec.load(Vec.make_type(1, fx.Float32), lds_lse, [m_local_f])[0]
                    dm_val    = Vec.load(Vec.make_type(1, fx.Float32), lds_dm,  [m_local_f])[0]
                    s_val     = Vec(s_acc)[r]
                    dp_val    = Vec(dp_acc)[r]
                    s_scaled  = fx.Float32(arith.mulf(_raw(s_val), _raw(scale_cst), fastmath=fm))
                    s_sub_lse = fx.Float32(arith.subf(_raw(s_scaled), _raw(lse_val), fastmath=fm))
                    p_arg     = fx.Float32(arith.mulf(_raw(s_sub_lse), _raw(log2e_cst), fastmath=fm))
                    p_val     = fx.Float32(fly_math.exp2(p_arg, fastmath=fm))
                    valid_mn  = m_valid & n_ok
                    p_val     = valid_mn.select(p_val, fx.Float32(0.0))
                    dp_sub    = fx.Float32(arith.subf(_raw(dp_val), _raw(dm_val), fastmath=fm))
                    ds_val    = fx.Float32(arith.mulf(_raw(scale_cst), _raw(p_val), fastmath=fm))
                    ds_val    = fx.Float32(arith.mulf(_raw(ds_val), _raw(dp_sub), fastmath=fm))
                    ds_val    = valid_mn.select(ds_val, fx.Float32(0.0))
                    m_local   = m_within + (m_sub * 32)
                    n_local   = n_within + wave_n_sub * 32
                    Vec.from_elements([p_val],  fx.Float32).to(elem_dtype).store(lds_p,  [n_local * LDS_MPAD + m_local])
                    Vec.from_elements([ds_val], fx.Float32).to(elem_dtype).store(lds_ds, [n_local * LDS_MPAD + m_local])

                gpu.barrier()

                # ---- dV += P^T @ dO ; dK += dS^T @ Q (scalar-gather B-operand) ----
                MFMA_KS   = 32 // MFMA_K
                d_local   = lane_mod_32 + wave_d_sub * 32
                for ks in range_constexpr(MFMA_KS):
                    n_local = lane_mod_32 + wave_n_sub * 32
                    base_m  = lane_div_32 * MFMA_LK + (m_sub * 32 + ks * MFMA_K)
                    p_pack  = Vec.load(v8elem_type, lds_p,  [n_local * LDS_MPAD + base_m]).ir_value()
                    ds_pack = Vec.load(v8elem_type, lds_ds, [n_local * LDS_MPAD + base_m]).ir_value()
                    do_r  = fx.make_rmem_tensor(MFMA_LK, elem_dtype)
                    q_r   = fx.make_rmem_tensor(MFMA_LK, elem_dtype)
                    for e in range_constexpr(MFMA_LK):
                        m_local = lane_div_32 * MFMA_LK + (m_sub * 32 + ks * MFMA_K + e)
                        do_sc = Vec.load(Vec.make_type(1, elem_dtype), lds_do, [m_local * LDS_Q_STRIDE + d_local])[0]
                        q_sc  = Vec.load(Vec.make_type(1, elem_dtype), lds_q,  [m_local * LDS_Q_STRIDE + d_local])[0]
                        fx.memref_store(do_sc, do_r,  e)
                        fx.memref_store(q_sc,  q_r,   e)
                    dv_acc = mfma(p_pack,  fx.memref_load_vec(do_r), dv_acc)
                    dk_acc = mfma(ds_pack, fx.memref_load_vec(q_r),  dk_acc)

                # ---- dQ = dS @ K  (contract over this wave's 32 n-rows) ----
                # A=dS[m,n]: free=m=lane%32, k=n. dS in lds_ds as [n,m] (n_local*MPAD+m).
                # B=K[n,d] : free=d=lane%32, k=n. K in lds_k as [n,d] (n_local*D+d).
                # Output C[m,d]: m varies with r (scrambled), d=lane%32 fixed.
                dq_acc = Vec.filled(16, 0.0, fx.Float32)
                for ks in range_constexpr(MFMA_KS):
                    ds_r = fx.make_rmem_tensor(MFMA_LK, elem_dtype)
                    k_r  = fx.make_rmem_tensor(MFMA_LK, elem_dtype)
                    m_free = lane_mod_32 + (m_sub * 32)
                    d_free = lane_mod_32 + wave_d_sub * 32
                    for e in range_constexpr(MFMA_LK):
                        n_local = lane_div_32 * MFMA_LK + (wave_n_sub * 32 + ks * MFMA_K + e)
                        ds_sc = Vec.load(Vec.make_type(1, elem_dtype), lds_ds, [n_local * LDS_MPAD + m_free])[0]
                        k_sc  = Vec.load(Vec.make_type(1, elem_dtype), lds_k,  [n_local * D + d_free])[0]
                        fx.memref_store(ds_sc, ds_r, e)
                        fx.memref_store(k_sc,  k_r,  e)
                    dq_acc = mfma(fx.memref_load_vec(ds_r), fx.memref_load_vec(k_r), dq_acc)

                # atomic-add dQ partial (m varies with r; d=lane%32 fixed per wave_d_sub)
                d_col_abs_dq = lane_mod_32 + wave_d_sub * 32
                for r in range_constexpr(16):
                    m_within  = lane_div_32 * 4 + ((r // 4) * 8 + (r % 4))
                    m_row_abs = m_within + (m_sub * 32) + m_start
                    m_ok      = m_row_abs < seq_M_idx
                    if m_ok:
                        q_row_g = _q_row(m_row_abs)
                        flat_dq = fx.Int32(fx.Index(q_row_g) * fx.Index(D) + d_col_abs_dq)
                        _atomic_add_dq(flat_dq, Vec(dq_acc)[r])

                gpu.barrier()

            loop_results = yield [dk_acc, dv_acc, dummy_val]

        # ---- Store dV and dK ----
        dk_final  = loop_results[0]
        dv_final  = loop_results[1]
        d_col_abs = lane_mod_32 + wave_d_sub * 32
        for r in range_constexpr(16):
            n_within  = lane_div_32 * 4 + ((r // 4) * 8 + (r % 4))
            n_row_abs = n_within + wave_n_sub * 32 + n_start
            n_ok      = n_row_abs < seq_N_idx
            n_safe    = n_ok.select(n_row_abs, seq_N_idx - fx.Index(1))
            kv_row_g  = _kv_row(n_safe)
            flat_col  = fx.Int32(fx.Index(kv_row_g) * fx.Index(D) + d_col_abs)
            if n_ok:
                _store_f32_row(dV_buf, flat_col, Vec(dv_final)[r])
                _store_f32_row(dK_buf, flat_col, Vec(dk_final)[r])

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
        from flydsl.compiler.kernel_function import CompilationContext
        from flydsl._mlir import ir
        allocator.finalized = False
        _ctx = CompilationContext.get_current()
        with ir.InsertionPoint(_ctx.gpu_module_body):
            allocator.finalize()

        num_N_tiles = (fx.Index(N) + BLOCK_N - 1) // BLOCK_N
        grid_x = fx.Int32(fx.Index(B) * fx.Index(H) * num_N_tiles)
        fmha_bwd_dqdkdv_mfma_kernel(
            Q, K, V, dO, dV, dK, dQ_f32, LSE, D_vec,
            M, N, H, n_M_tiles,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    return launch_fn
