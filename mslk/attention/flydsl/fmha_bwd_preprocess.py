"""WP-A3 Phase A — FMHA backward preprocess: D-vector kernel.

Computes D[b, m, h] = rowsum(dO[b, m, h, :] * O[b, m, h, :])
per query row. This is the first of 3 backward phases.

Layout convention (matches CK / ref_fmha_bwd_reference.py):
  dO, O  : [B, M, H, D]  (BMHK, contiguous) — passed as [B*M*H, D] 2D tensor
  D_out  : [B*M*H]       float32 — one scalar per row

Kernel strategy:
  - Grid: (B*M*H,) — one block per row.
  - Pass dO/O as 2D [n_rows, D] tensors so fx.slice(buf, (bid, None)) works.
  - Warp reduce via shuffle_xor, cross-warp via LDS.

Target: gfx950 (CDNA4, wave64).
"""

import math as _math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, gpu, range_constexpr
from flydsl.expr.vector import ReductionOp
from kernels.kernels_common import dtype_to_elem_type, get_warp_size

WARP_SIZE     = get_warp_size()   # 64 on CDNA
BLOCK_THREADS = 256               # threads per block


def compile_fmha_bwd_preprocess(*, D: int, dtype_str: str = "bf16"):
    """Compile the D-vector preprocess kernel.

    Args:
        D        : head dimension (must be multiple of 8 and of WARP_SIZE)
        dtype_str: "bf16" or "f16"

    Returns:
        launch_fn(dO_2d, O_2d, D_out, stream)
          dO_2d, O_2d : [B*M*H, D]  int16 view of bf16/fp16  (2D, contiguous)
          D_out       : [B*M*H, 1]  float32 (2D, one scalar per row)
    """
    assert D % 8 == 0, f"D={D} must be a multiple of 8 (128-bit loads)"
    assert D % WARP_SIZE == 0, f"D={D} must be a multiple of WARP_SIZE={WARP_SIZE}"

    elem_dtype = dtype_to_elem_type(dtype_str)
    elem_bits  = 16               # bf16 / fp16
    VEC_WIDTH  = 128 // elem_bits  # 8 elements per 128-bit load
    N_VECS     = D // VEC_WIDTH    # number of vec-columns per row
    N_THREADS  = min(BLOCK_THREADS, N_VECS)
    N_TILES    = N_VECS // N_THREADS   # tiles per thread (≥ 1)
    RED_SLOTS  = max(1, N_THREADS // WARP_SIZE)

    @fx.struct
    class SharedStorage:
        s_red: fx.Array[fx.Float32, RED_SLOTS, 16]

    @flyc.kernel
    def d_vec_kernel(
        dO:    fx.Tensor,   # [n_rows, D]  int16
        O:     fx.Tensor,   # [n_rows, D]  int16
        D_out: fx.Tensor,   # [n_rows, 1]  float32
    ):
        bid = fx.block_idx.x    # row index (= b*M*H + m*H + h)
        tid = fx.thread_idx.x
        fm  = arith.FastMathFlags.fast

        # LDS
        lds   = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_red = lds.s_red.view(fx.make_layout(RED_SLOTS, 1))

        # Slice this block's row
        dO_buf = fx.rocdl.make_buffer_tensor(dO)
        O_buf  = fx.rocdl.make_buffer_tensor(O)
        D_buf  = fx.rocdl.make_buffer_tensor(D_out)

        dO_row = fx.slice(dO_buf, (bid, None))   # [D] int16
        O_row  = fx.slice(O_buf,  (bid, None))   # [D] int16
        D_row  = fx.slice(D_buf,  (bid, None))   # [1] float32

        # Divide row into vec-columns
        copy_atom  = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
        store_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)

        dO_div = fx.logical_divide(dO_row, fx.make_layout(VEC_WIDTH, 1))
        O_div  = fx.logical_divide(O_row,  fx.make_layout(VEC_WIDTH, 1))
        D_div  = fx.logical_divide(D_row,  fx.make_layout(1, 1))

        def _load_vec(div_tensor, col):
            r = fx.make_rmem_tensor(VEC_WIDTH, elem_dtype)
            fx.copy_atom_call(copy_atom, fx.slice(div_tensor, (None, col)), r)
            return fx.memref_load_vec(r)

        # ---- Accumulate partial dot ----
        acc = fx.Float32(0.0)

        for tile in range_constexpr(N_TILES):
            col = tid + tile * N_THREADS
            do_vec = _load_vec(dO_div, col)
            o_vec  = _load_vec(O_div,  col)
            do_f32 = do_vec.to(fx.Float32)
            o_f32  = o_vec.to(fx.Float32)
            prod   = do_f32 * o_f32                         # element-wise multiply
            acc    = acc.addf(prod.reduce(ReductionOp.ADD, fastmath=fm), fastmath=fm)

        # ---- Warp reduce (within N_THREADS, which may be < WARP_SIZE) ----
        w = acc
        for _sh in range_constexpr(int(_math.log2(N_THREADS))):
            off  = N_THREADS // (2 << _sh)
            peer = w.shuffle_xor(off, WARP_SIZE)
            w    = w.addf(peer, fastmath=fm)

        # ---- Cross-warp reduce via LDS ----
        lane = tid % WARP_SIZE
        wave = tid // WARP_SIZE

        if lane == 0:
            fx.memref_store(w, s_red, wave)
        gpu.barrier()

        if wave == 0:
            in_range  = lane < RED_SLOTS
            lane_safe = in_range.select(lane, 0)
            v = in_range.select(fx.memref_load(s_red, lane_safe), 0.0)
            # Reduce across the RED_SLOTS warp results (only when RED_SLOTS > 1)
            for _sh in range_constexpr(max(0, int(_math.log2(max(1, RED_SLOTS))))):
                off  = max(1, RED_SLOTS) // (2 << _sh)
                peer = v.shuffle_xor(off, WARP_SIZE)
                v    = v.addf(peer, fastmath=fm)
            if lane == 0:
                fx.memref_store(v, s_red, 0)
        gpu.barrier()

        # ---- Store result ----
        if tid == 0:
            result  = fx.memref_load(s_red, 0)
            out_reg = fx.make_rmem_tensor(1, fx.Float32)
            fx.memref_store(result, out_reg, 0)
            fx.copy_atom_call(store_atom, out_reg,
                              fx.slice(D_div, (None, 0)))

    @flyc.jit
    def launch_fn(
        dO:    fx.Tensor,
        O:     fx.Tensor,
        D_out: fx.Tensor,
        n_rows: fx.Int32,
        stream: fx.Stream,
    ):
        d_vec_kernel(dO, O, D_out).launch(
            grid=(n_rows, 1, 1),
            block=(N_THREADS, 1, 1),
            stream=stream,
        )

    return launch_fn
