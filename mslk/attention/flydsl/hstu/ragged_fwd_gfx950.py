# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# The software pipeline (two wave groups, K/V DMA into a double-buffered LDS
# ring, ds_read_tr16_b64 V reads, eight-cluster schedule) is the gfx950
# flash-attention kernel's, mslk/attention/flydsl/flash_attn_gfx950.py:
#
#   SPDX-License-Identifier: Apache-2.0
#   Copyright (c) 2025 FlyDSL Project Contributors

# pyre-unsafe

"""Ragged HSTU self-attention forward on gfx950's flash-attention pipeline.

Same results as :mod:`mslk.attention.flydsl.hstu.ragged_fwd` for the shapes it
takes (D 64 or 128, V's dim equal to D, bf16; SiLU also fp16): it reuses the
dual-wave software pipeline of MSLK's gfx950 flash attention and replaces
its softmax and causal mask with HSTU's:

- KV tiles. A query block (256 rows) visits logical tiles ``0 .. n``; logical
  tile ``t`` is physical tile ``t`` inside the contextual prefix and
  ``t + (main_start - prefix_end)`` after it, so the pipeline walks the
  contextual columns and then the window (or causal range) as one sequence.
  The pipeline needs an even count of at least four; padding tiles past the
  range hold no visible key and are masked.
- Mask. T1's mask (:mod:`.common`) per score, on tiles that are not fully
  visible to the wave's rows.
- Activation. SiLU heads: ``silu(alpha * s)`` in place of the softmax exp2,
  with no running max or sum, scaled by 1/N, a scalar or a per-token scale in
  the epilogue. Softmax heads: the pipeline's online softmax with
  ``alpha * log2(e)`` as the score scale, and M = m + log2(l).
"""

import functools
import math as host_math
from typing import Optional

import torch

from .common import LOWER, MaskSpec

_LOG2E = host_math.log2(host_math.e)

BLOCK_M = 256
BLOCK_N = 64

SCALE_INV_N = "inv_n"
SCALE_SCALAR = "scalar"
SCALE_TOKEN = "token"


def decline_reason(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax: bool,
    arch: str,
) -> Optional[str]:
    """Why this kernel cannot take a call, or None."""
    if not arch.startswith("gfx950"):
        return f"needs gfx950, got {arch!r}"
    if q.dtype not in (torch.bfloat16, torch.float16):
        return f"unsupported dtype {q.dtype}"
    if softmax and q.dtype != torch.bfloat16:
        # The pipeline's online softmax overflows in fp16 at large logits.
        return "softmax heads need bf16"
    head_dim = q.shape[2]
    if head_dim not in (64, 128) or v.shape[2] != head_dim:
        return (
            f"needs D in (64, 128) and V's dim equal to D, got {head_dim}, {v.shape[2]}"
        )
    return None


@functools.lru_cache(maxsize=256)
def build_ragged_fwd_gfx950(
    num_heads: int,
    head_dim: int,
    dtype_str: str,
    mask_type: str,
    has_window: bool,
    has_contextual: bool,
    has_targets: bool,
    has_full_attn: bool,
    scale_mode: str,
    softmax: bool,
    sorted_order: bool = False,
    reverse_q_blocks: bool = False,
):
    """Compile-time specialization; returns ``launch(...)``.

    ``sorted_order``: sequences run longest first (through the ``SortIdx``
    order), and on a lower-triangular mask a sequence's query blocks run
    last first, so the heaviest blocks start earliest."""
    import flydsl.compiler as flyc  # pyre-ignore[21]
    import flydsl.expr as fx  # pyre-ignore[21]
    from flydsl.compiler.kernel_function import CompilationContext  # pyre-ignore[21]
    from flydsl.expr import const_expr, gpu, range_constexpr, rocdl  # pyre-ignore[21]
    from flydsl.expr.typing import T, Vector as Vec  # pyre-ignore[21]
    from mslk.attention.flydsl._common import tensor_cache_sig as _tensor_cache_sig
    from mslk.attention.flydsl.flash_attn_utils import (
        _anchor_v_o,
        _anchor_v_p,
        _dualwave_sync_barrier,
        _make_dualwave_swp_traits,
        _s_barrier,
        _s_nop,
        _s_setprio,
        _s_waitcnt,
        _scale_o_accs,
        _sched_barrier,
        _sched_barrier_exp_pairs,
        _sched_barrier_pairs,
        _score_lists_to_vecs,
        _seq_pad_score_threshold,
        _stagger_extra_barrier_if_one,
        _stagger_extra_barrier_if_zero,
        _v_pair_to_vec32,
        _v_vec32_to_pair,
        _waitcnt_vm_n,
        as_mlir_value,
        DualwaveGemmHelper,
        DualwaveKernelContext,
        DualwaveKvGmemToLdsLoader,
        DualwaveKvLdsToVgprLoader,
        DualwaveQLoader,
        DualwaveSoftmaxHelper,
        DualwaveStoreHelper,
    )

    from .common import device_ops, seq_mask

    traits = _make_dualwave_swp_traits(
        num_heads,
        num_heads,
        head_dim,
        causal=False,
        dtype_str=dtype_str,
        waves_per_eu=2,
        daz=True,
        dualwave_swp_lazy_rescale=True,
        dualwave_swp_setprio=True,
        dualwave_swp_enable_stagger=True,
        varlen=True,
    )
    assert traits.BLOCK_M == BLOCK_M and traits.BLOCK_N == BLOCK_N

    # Scalars only in the kernels' closures: FlyDSL's compile cache keys on
    # scalar closure values (and tuples), not on objects such as the traits.
    MASK_TYPE = mask_type
    HAS_WINDOW = has_window
    HAS_CONTEXTUAL = has_contextual
    HAS_TARGETS = has_targets
    HAS_FULL_ATTN = has_full_attn
    SCALE_MODE = scale_mode
    SOFTMAX = softmax
    SORTED = sorted_order
    REVERSE_Q_BLOCKS = reverse_q_blocks
    ROWS_PER_WAVE = traits.ROWS_PER_WAVE
    # Tiles can skip the per-score mask when the whole tile is visible to the
    # wave's rows; worked out for the lower-triangular mask without a
    # contextual prefix (window and targets allowed). Others mask every tile.
    CAN_SKIP_MASK = MASK_TYPE == LOWER and not HAS_CONTEXTUAL
    _cache_tag = (
        traits.cache_tag,
        "hstu",
        MASK_TYPE,
        HAS_WINDOW,
        HAS_CONTEXTUAL,
        HAS_TARGETS,
        HAS_FULL_ATTN,
        SCALE_MODE,
        SOFTMAX,
        SORTED,
        REVERSE_Q_BLOCKS,
    )

    _lds_elem_dtype = fx.BFloat16 if dtype_str == "bf16" else fx.Float16

    @fx.struct
    class SharedStorage:
        kv: fx.Array[_lds_elem_dtype, traits.LDS_KV_TOTAL_SIZE, 16]

    class TileMappedKvLoader(DualwaveKvGmemToLdsLoader):
        """KV DMA through the logical -> physical tile map."""

        def tile_start(self, tile_idx):
            return self.hstu_phys_tile(tile_idx) * traits.BLOCK_N

    class SiluActivation:
        """The softmax helper's interface for SiLU heads: no running max, no
        row sum, no rescale. Scores arrive raw; p = g * sigmoid(-g) style base
        change: with g = s * alpha * -log2e, g / (1 + exp2(g)) is
        (-log2e) * silu(alpha * s); the epilogue scale undoes -log2e."""

        def __init__(self, ctx, g_scale):
            self.ctx = ctx
            self.g_scale = g_scale
            self.cast_helper = DualwaveSoftmaxHelper(ctx)

        def _silu(self, x):
            g = fx.Float32(x) * self.g_scale
            e = fx.Float32(rocdl.exp2(T.f32, as_mlir_value(g)))
            r = fx.Float32(rocdl.rcp(T.f32, as_mlir_value(e + fx.Float32(1.0))))
            return as_mlir_value(g * r)

        def prepare(self, v_s):
            return v_s

        def reduce_max(self, v_s):
            return self.ctx.c_zero_f

        def floor_masked_max(self, m):
            return m

        def sub_m(self, v_s, row_max):
            return v_s

        def exp2(self, v_s, start, length):
            # Mirrors _exp2_score_slice: start 0 activates the lo half and
            # keeps hi raw; start 16 activates hi and returns both as lists.
            if const_expr(start == 0):
                lo = [self._silu(Vec(v_s[0])[r]) for r in range_constexpr(16)]
                return Vec.from_elements(lo, fx.Float32).ir_value(), v_s[1]
            lo = [Vec(v_s[0])[r] for r in range_constexpr(16)]
            hi = [self._silu(Vec(v_s[1])[r]) for r in range_constexpr(16)]
            return lo, hi

        def reduce_sum(self, l_row, v_p):
            return l_row

        def cast_p(self, v_p):
            return self.cast_helper.cast_p(v_p)

        def lazy_rescale_o(self, v_o, m_row, l_row, m_tile_max, v_p):
            return v_o, m_row, l_row, v_p

        def rescale_from_tile_max(self, m_row, m_tile_max):
            return m_row, None

        def apply_l_rescale(self, l_row, rescale):
            return l_row

        def scale_o(self, v_o, rescale):
            return None

    class SoftmaxActivation(DualwaveSoftmaxHelper):
        def prepare(self, v_s):
            return self.v_s_vec_to_lists(v_s)

        def floor_masked_max(self, m):
            # A row can see no key in a tile, and its first tiles may all be
            # masked (windows): keep the running max finite.
            return super().floor_masked_max(m)

    @flyc.kernel(known_block_size=[traits.BLOCK_SIZE, 1, 1])
    def hstu_ragged_fwd_gfx950_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        LseOut: fx.Tensor,
        CuSeq: fx.Tensor,
        NumTargets: fx.Tensor,
        AttnScale: fx.Tensor,
        SortIdx: fx.Tensor,
        stride_q_n: fx.Int32,
        stride_kv_n: fx.Int32,
        score_scale: fx.Float32,
        out_scale: fx.Float32,
        max_attn_len: fx.Int32,
        contextual_seq_len: fx.Int32,
        full_attn_size: fx.Int32,
    ):
        ctx = DualwaveKernelContext(
            traits,
            Q,
            K,
            V,
            O,
            O,
            CuSeq,
            CuSeq,
            O,
            fx.Int32(0),
            fx.Int32(0),
            stride_q_n,
            stride_kv_n,
            fx.Int32(head_dim),
            fx.Int32(0),
            LSE=LseOut,
        )
        ctx.init_types_and_constants()
        ctx.init_runtime_indices()
        ctx.init_lds(SharedStorage)
        ctx.init_thread_mapping()
        if const_expr(SORTED):
            ctx.batch_idx = fx.Index(fx.Int32(SortIdx[fx.Int32(ctx.batch_idx)]))
        if const_expr(REVERSE_Q_BLOCKS):
            ctx.q_block_idx = fx.Index(gpu.grid_dim.y) - fx.Index(1) - ctx.q_block_idx
            ctx.q_start = ctx.q_block_idx * traits.BLOCK_M
        ctx.init_sequence_lengths()
        ctx.init_descriptors()
        ctx.init_workspace()
        ctx.init_atoms_and_lds_ptrs()
        ctx.init_dma_thread_offsets()
        ctx.init_q_row()

        # ---- HSTU mask state and the logical -> physical KV tile map ----
        ops = device_ops()
        seq_len_i32 = fx.Int32(ctx.seqlen_q_v)
        num_target = fx.Int32(0)
        if const_expr(HAS_TARGETS):
            num_target = fx.Int32(NumTargets[fx.Int32(ctx.batch_idx)])
        sm = seq_mask(
            MaskSpec(MASK_TYPE, HAS_WINDOW, HAS_CONTEXTUAL, HAS_TARGETS, HAS_FULL_ATTN),
            ops,
            seq_len_i32,
            num_targets=num_target,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
            full_attn_size=full_attn_size,
        )
        q_start_i32 = fx.Int32(ctx.q_start)
        prefix_end, main_start, main_end = sm.kv_tiles(q_start_i32, BLOCK_M, BLOCK_N)
        n_logical = prefix_end + (main_end - main_start)
        # Even, and at least four (the pipeline's prologue and epilogue).
        n_logical = ((n_logical + fx.Int32(1)) // fx.Int32(2)) * fx.Int32(2)
        n_logical = ops.max(n_logical, fx.Int32(4))
        tile_gap = main_start - prefix_end

        def phys_tile_i32(tile_idx):
            t = fx.Int32(tile_idx)
            return (t < prefix_end).select(t, t + tile_gap)

        ctx.hstu_phys_tile = lambda tile_idx: fx.Index(phys_tile_i32(tile_idx))
        ctx.kv_tile_size = traits.BLOCK_N
        ctx.split_t0 = 0
        ctx.split_t_end = fx.Index(n_logical)
        ctx.max_num_tiles = ctx.split_t_end
        ctx.causal_end_raw_i32 = None
        ctx.init_active_guard()
        ctx.init_lds_read_bases()
        ctx.init_dma_m0_tables()

        # Score scale: softmax uses alpha * log2(e) where the pipeline has
        # 1/sqrt(D) * log2(e); set before the helpers copy the context.
        if const_expr(SOFTMAX):
            ctx.c_sm_scale_log2e = score_scale

        q_row_i32 = ctx.q_row_i32
        q_row_id = sm.to_id(q_row_i32)
        wave_row0 = ctx.q_start_pos_i32
        wave_row_last = wave_row0 + fx.Int32(ROWS_PER_WAVE - 1)
        col_lane_base = fx.Int32(ctx.lane_div_32) * fx.Int32(4)
        masked_score = ctx.c_neg_inf if SOFTMAX else ctx.c_zero_f

        def tile_needs_mask(c0):
            """True unless every score of the tile is visible to every row of
            this wave (uniform across the wave)."""
            c_last = c0 + fx.Int32(BLOCK_N - 1)
            need = c_last >= seq_len_i32
            need = need | (sm.to_id(c_last) >= sm.to_id(wave_row0))
            if const_expr(HAS_WINDOW):
                need = need | (sm.to_id(c0) < sm.to_id(wave_row_last) - sm.max_attn_len)
            return need

        def mask_scores(v_s, c0):
            lo = [Vec(v_s[0])[r] for r in range_constexpr(16)]
            hi = [Vec(v_s[1])[r] for r in range_constexpr(16)]
            base = c0 + col_lane_base
            for r in range_constexpr(16):
                for half, scores in ((0, lo), (1, hi)):
                    col = base + fx.Int32(
                        _seq_pad_score_threshold(traits, r) + 32 * half
                    )
                    keep = sm.visible(q_row_i32, col, q_row_id, sm.to_id(col)) & (
                        col < seq_len_i32
                    )
                    scores[r] = keep.select(scores[r], masked_score)
            return _score_lists_to_vecs((lo, hi))

        def hstu_mask(v_s, tile_idx):
            c0 = phys_tile_i32(tile_idx) * fx.Int32(BLOCK_N)
            if const_expr(not CAN_SKIP_MASK):
                return mask_scores(v_s, c0)
            needs = tile_needs_mask(c0)

            @flyc.jit
            def _mask_if_needed(v_s, c0, needs):
                s_lo, s_hi = v_s
                if needs:
                    s_lo, s_hi = mask_scores((s_lo, s_hi), c0)
                return s_lo, s_hi

            return _mask_if_needed(v_s, c0, needs)

        active = ctx.active
        elem_dtype = ctx.elem_dtype
        stagger_i32 = ctx.stagger_i32
        l_row_init = ctx.c_zero_f
        split_t_end = ctx.split_t_end
        v_o_zero = ctx.c_zero_v16f32

        kv_gmem_to_lds = TileMappedKvLoader(ctx)
        kv_lds_to_regs = DualwaveKvLdsToVgprLoader(ctx)
        output_store = DualwaveStoreHelper(ctx)
        q_loader = DualwaveQLoader(ctx)
        gemm_helper = DualwaveGemmHelper(ctx)
        if const_expr(SOFTMAX):
            act = SoftmaxActivation(ctx)
        else:
            act = SiluActivation(ctx, score_scale)

        def _main_body():
            kv_gmem_to_lds.load_k_split(0, 0)
            _s_waitcnt(0)
            _sched_barrier(0)
            _s_barrier()

            q_all_bf16 = q_loader.load_all()

            kv_gmem_to_lds.load_k_split(1, 1)
            kv_gmem_to_lds.load_v_split(0, 0)
            v_k = kv_lds_to_regs.load_k(0)
            _sched_barrier(0)
            _s_waitcnt(traits.LGKMCNT_0_ONLY)
            _waitcnt_vm_n(ctx.NUM_DMA_V)

            if const_expr(traits.DUALWAVE_SWP_ENABLE_STAGGER):
                _stagger_extra_barrier_if_one(stagger_i32)
            else:
                _sched_barrier(0)
                _s_barrier()

            v_s_0 = gemm_helper.qk(v_k, q_all_bf16)
            _sched_barrier(0)
            v_s_0 = act.prepare(hstu_mask(v_s_0, fx.Index(0)))
            m_row_pro = act.floor_masked_max(act.reduce_max(v_s_0))
            v_s_0 = act.sub_m(v_s_0, m_row_pro)
            v_p_0 = act.exp2(v_s_0, 0, 16)
            _dualwave_sync_barrier()

            loop_lb = fx.Index(3)
            kv_gmem_to_lds.load_k_split(2, 0)

            init_args = [m_row_pro, l_row_init]
            for _ in range_constexpr(traits.D_CHUNKS):
                init_args.append(v_o_zero)
            init_args.append(_v_pair_to_vec32(v_p_0))
            loop_results = init_args
            for j, loop_args in range(
                loop_lb,
                split_t_end - fx.Index(1),
                fx.Index(2),
                init=init_args,
            ):
                m_row = loop_args[0]
                l_row = loop_args[1]
                v_o = [loop_args[2 + i] for i in range_constexpr(traits.D_CHUNKS)]
                v_p_0 = _v_vec32_to_pair(loop_args[2 + traits.D_CHUNKS])
                j_idx = j

                # Cluster 0: prefetch V buf1, read K for MMA0.
                _s_nop(7)
                _sched_barrier(0)
                kv_gmem_to_lds.load_v_tile(j_idx - 2, 1)
                v_k = kv_lds_to_regs.load_k(1)
                _s_waitcnt(traits.LGKMCNT_0_ONLY)
                _waitcnt_vm_n(ctx.NUM_DMA_K + ctx.NUM_DMA_V)
                _dualwave_sync_barrier()

                # Cluster 1: MMA0, finish v_p_0, cast P.
                v_s_1 = gemm_helper.qk(v_k, q_all_bf16)
                v_p_0 = act.exp2(v_p_0, 16, 16)
                l_row = act.reduce_sum(l_row, v_p_0)
                v_p_0 = act.cast_p(v_p_0)
                v_p_0 = _anchor_v_p(traits, v_p_0, elem_dtype=elem_dtype)
                _sched_barrier_exp_pairs(traits, 6, 3, 1)
                _sched_barrier_pairs(traits, 10, 5, 1)
                _dualwave_sync_barrier()

                # Cluster 2: prefetch next K, read V, mask v_s_1.
                _s_nop(7)
                _sched_barrier(0)
                kv_gmem_to_lds.load_k_tile(j_idx, 1)
                v_v = kv_lds_to_regs.load_v(0)
                v_s_1 = act.prepare(hstu_mask(v_s_1, j_idx - 2))
                _s_waitcnt(traits.LGKMCNT_0_ONLY)
                _waitcnt_vm_n(ctx.NUM_DMA_K + ctx.NUM_DMA_V)
                _dualwave_sync_barrier()

                # Cluster 3: P*V, row max / rescale (softmax), first-half activation.
                _s_setprio(1)
                v_o = gemm_helper.pv_step_k(0, v_p_0, v_v, v_o)
                m_tile_max_a = act.reduce_max(v_s_1)
                _sched_barrier_pairs(traits, 4, 6, 2)
                v_o, m_row, l_row, v_p_0 = act.lazy_rescale_o(
                    v_o, m_row, l_row, m_tile_max_a, v_p_0
                )
                v_o = gemm_helper.pv_step_k(1, v_p_0, v_v, v_o)
                v_o = gemm_helper.pv_step_k(2, v_p_0, v_v, v_o)
                v_o = gemm_helper.pv_step_k(3, v_p_0, v_v, v_o)
                v_s_1 = act.sub_m(v_s_1, m_row)
                v_p_1 = act.exp2(v_s_1, 0, 16)
                _sched_barrier_pairs(traits, 6, 6, 2)
                _sched_barrier_exp_pairs(traits, 6, 3, 2)
                _s_setprio(0)
                _dualwave_sync_barrier()

                # Cluster 4: prefetch V buf0, read K.
                _s_nop(7)
                _sched_barrier(0)
                kv_gmem_to_lds.load_v_tile(j_idx - 1, 0)
                v_k = kv_lds_to_regs.load_k(0)
                _s_waitcnt(traits.LGKMCNT_0_ONLY)
                _waitcnt_vm_n(ctx.NUM_DMA_K + ctx.NUM_DMA_V)
                _dualwave_sync_barrier()

                # Cluster 5: MMA0, finish v_p_1, cast P.
                v_s_0 = gemm_helper.qk(v_k, q_all_bf16)
                v_p_1 = act.exp2(v_p_1, 16, 16)
                l_row = act.reduce_sum(l_row, v_p_1)
                v_p_1 = act.cast_p(v_p_1)
                v_p_1 = _anchor_v_p(traits, v_p_1, elem_dtype=elem_dtype)
                _sched_barrier_exp_pairs(traits, 6, 3, 3)
                _sched_barrier_pairs(traits, 10, 5, 3)
                _dualwave_sync_barrier()

                # Cluster 6: prefetch next K, read V, mask v_s_0.
                _s_nop(7)
                _sched_barrier(0)
                kv_gmem_to_lds.load_k_tile(j_idx + 1, 0)
                v_v = kv_lds_to_regs.load_v(1)
                v_s_0 = act.prepare(hstu_mask(v_s_0, j_idx - 1))
                _s_waitcnt(traits.LGKMCNT_0_ONLY)
                _waitcnt_vm_n(ctx.NUM_DMA_K + ctx.NUM_DMA_V)
                _dualwave_sync_barrier()

                # Cluster 7: mirrors cluster 3.
                _s_setprio(1)
                v_o = gemm_helper.pv_step_k(0, v_p_1, v_v, v_o)
                m_tile_max_b = act.reduce_max(v_s_0)
                _sched_barrier_pairs(traits, 4, 6, 4)
                v_o, m_row, l_row, v_p_1 = act.lazy_rescale_o(
                    v_o, m_row, l_row, m_tile_max_b, v_p_1
                )
                v_o = gemm_helper.pv_step_k(1, v_p_1, v_v, v_o)
                v_o = gemm_helper.pv_step_k(2, v_p_1, v_v, v_o)
                v_o = gemm_helper.pv_step_k(3, v_p_1, v_v, v_o)
                v_s_0 = act.sub_m(v_s_0, m_row)
                v_p_0 = act.exp2(v_s_0, 0, 16)
                _sched_barrier_pairs(traits, 6, 5, 4)
                _sched_barrier_exp_pairs(traits, 6, 3, 4)
                _s_setprio(0)
                _dualwave_sync_barrier()

                yield_args = [m_row, l_row] + v_o + [_v_pair_to_vec32(v_p_0)]
                loop_results = yield yield_args

            m_row = loop_results[0]
            l_row = loop_results[1]
            v_o = [loop_results[2 + i] for i in range_constexpr(traits.D_CHUNKS)]
            v_p_0 = _v_vec32_to_pair(loop_results[2 + traits.D_CHUNKS])

            max_m3 = split_t_end - 3
            max_m2 = split_t_end - 2
            max_m1 = split_t_end - 1

            # Epilogue C0.
            _s_nop(7)
            _sched_barrier(0)
            kv_gmem_to_lds.load_v_tile(max_m3, 1)
            v_k = kv_lds_to_regs.load_k(1)
            _s_waitcnt(traits.LGKMCNT_0_ONLY)
            _waitcnt_vm_n(ctx.NUM_DMA_K + ctx.NUM_DMA_V)
            _dualwave_sync_barrier()

            # Epilogue C1.
            v_s_1 = gemm_helper.qk(v_k, q_all_bf16)
            v_p_0 = act.exp2(v_p_0, 16, 16)
            l_row = act.reduce_sum(l_row, v_p_0)
            v_p_0 = act.cast_p(v_p_0)
            v_p_0 = _anchor_v_p(traits, v_p_0, elem_dtype=elem_dtype)
            _sched_barrier_exp_pairs(traits, 6, 3, 5)
            _sched_barrier_pairs(traits, 10, 5, 5)
            _dualwave_sync_barrier()

            # Epilogue C2.
            _s_nop(7)
            _sched_barrier(0)
            kv_gmem_to_lds.load_k_tile(max_m1, 1)
            v_packs_e3 = kv_lds_to_regs.load_v(0)
            v_s_1 = act.prepare(hstu_mask(v_s_1, max_m3))
            _s_waitcnt(traits.LGKMCNT_0_ONLY)
            _waitcnt_vm_n(ctx.NUM_DMA_K + ctx.NUM_DMA_V)
            _dualwave_sync_barrier()

            # Epilogue C3.
            _s_setprio(1)
            v_o = gemm_helper.pv(v_p_0, v_packs_e3, v_o)
            m_tile_max_e3 = act.reduce_max(v_s_1)
            row_max_e3, rescale_e3 = act.rescale_from_tile_max(m_row, m_tile_max_e3)
            m_row = row_max_e3
            v_s_1 = act.sub_m(v_s_1, row_max_e3)
            v_p_1 = act.exp2(v_s_1, 0, 16)
            _sched_barrier_pairs(traits, 10, 5, 6)
            _sched_barrier_exp_pairs(traits, 6, 3, 6)
            _sched_barrier(0)
            if const_expr(SOFTMAX):
                act.scale_o(v_o, rescale_e3)
            v_o = _anchor_v_o(traits, v_o)
            _s_setprio(0)
            _dualwave_sync_barrier()

            # Epilogue C4.
            _s_nop(7)
            _sched_barrier(0)
            kv_gmem_to_lds.load_v_tile(max_m2, 0)
            v_k = kv_lds_to_regs.load_k(0)
            _s_waitcnt(traits.LGKMCNT_0_ONLY)
            _waitcnt_vm_n(ctx.NUM_DMA_K + ctx.NUM_DMA_V)
            _dualwave_sync_barrier()

            # Epilogue C5.
            v_s_0 = gemm_helper.qk(v_k, q_all_bf16)
            l_row = act.apply_l_rescale(l_row, rescale_e3)
            v_p_1 = act.exp2(v_p_1, 16, 16)
            l_row = act.reduce_sum(l_row, v_p_1)
            v_p_1 = act.cast_p(v_p_1)
            v_p_1 = _anchor_v_p(traits, v_p_1, elem_dtype=elem_dtype)
            _sched_barrier_exp_pairs(traits, 6, 3, 7)
            _sched_barrier_pairs(traits, 10, 5, 7)
            _dualwave_sync_barrier()

            # Epilogue C6.
            v_packs_e7 = kv_lds_to_regs.load_v(1)
            v_s_0 = act.prepare(hstu_mask(v_s_0, max_m2))
            _s_waitcnt(traits.LGKMCNT_0_ONLY)
            _waitcnt_vm_n(ctx.NUM_DMA_V)
            _dualwave_sync_barrier()

            # Epilogue C7.
            _s_setprio(1)
            v_o = gemm_helper.pv(v_p_1, v_packs_e7, v_o)
            m_tile_max_e7 = act.reduce_max(v_s_0)
            row_max_e7, rescale_e7 = act.rescale_from_tile_max(m_row, m_tile_max_e7)
            m_row = row_max_e7
            v_s_0 = act.sub_m(v_s_0, row_max_e7)
            v_p_0 = act.exp2(v_s_0, 0, 16)
            _sched_barrier_pairs(traits, 10, 5, 8)
            _sched_barrier_exp_pairs(traits, 6, 3, 8)
            _sched_barrier(0)
            if const_expr(SOFTMAX):
                act.scale_o(v_o, rescale_e7)
            v_o = _anchor_v_o(traits, v_o)
            _s_setprio(0)
            _dualwave_sync_barrier()

            # Epilogue C8.
            _s_nop(7)
            _sched_barrier(0)
            kv_gmem_to_lds.load_v_tile(max_m1, 1)
            v_k = kv_lds_to_regs.load_k(1)
            _s_waitcnt(traits.LGKMCNT_0_ONLY)
            _waitcnt_vm_n(ctx.NUM_DMA_V)
            _dualwave_sync_barrier()

            # Epilogue C9.
            v_s_1 = gemm_helper.qk(v_k, q_all_bf16)
            l_row = act.apply_l_rescale(l_row, rescale_e7)
            v_p_0 = act.exp2(v_p_0, 16, 16)
            l_row = act.reduce_sum(l_row, v_p_0)
            v_p_0 = act.cast_p(v_p_0)
            v_p_0 = _anchor_v_p(traits, v_p_0, elem_dtype=elem_dtype)
            _sched_barrier_exp_pairs(traits, 6, 3, 9)
            _sched_barrier_pairs(traits, 10, 5, 9)
            _dualwave_sync_barrier()

            # Epilogue C10.
            v_packs_e11 = kv_lds_to_regs.load_v(0)
            v_s_1 = act.prepare(hstu_mask(v_s_1, max_m1))
            _s_waitcnt(traits.LGKMCNT_0_ONLY)
            _waitcnt_vm_n(0)
            _dualwave_sync_barrier()

            # Epilogue C11.
            v_o = gemm_helper.pv(v_p_0, v_packs_e11, v_o)
            m_tile_max_e11 = act.reduce_max(v_s_1)
            row_max_e11, rescale_e11 = act.rescale_from_tile_max(m_row, m_tile_max_e11)
            m_row = row_max_e11
            v_s_1 = act.sub_m(v_s_1, row_max_e11)
            v_p_1 = act.exp2(v_s_1, 0, 16)
            _sched_barrier_pairs(traits, 9, 6, 10)
            _sched_barrier_exp_pairs(traits, 7, 3, 10)
            _sched_barrier(0)
            v_p_1 = act.exp2(v_p_1, 16, 16)
            l_row = act.apply_l_rescale(l_row, rescale_e11)
            l_row = act.reduce_sum(l_row, v_p_1)
            v_p_1 = act.cast_p(v_p_1)
            v_p_1 = _anchor_v_p(traits, v_p_1, elem_dtype=elem_dtype)
            _sched_barrier(0)
            if const_expr(SOFTMAX):
                act.scale_o(v_o, rescale_e11)
            v_o = _anchor_v_o(traits, v_o)
            _s_barrier()
            _sched_barrier(0)

            # Epilogue C12.
            v_packs_e13 = kv_lds_to_regs.load_v(1)
            _s_waitcnt(traits.LGKMCNT_0_ONLY)
            _dualwave_sync_barrier()

            # Epilogue C13: final P*V.
            v_o = gemm_helper.pv(v_p_1, v_packs_e13, v_o)

            # Output scale.
            row_in_seq = q_row_i32 < seq_len_i32
            if const_expr(SOFTMAX):
                l_inv = act.safe_l_inv(l_row)
                _scale_o_accs(v_o, l_inv, traits, ctx.fm_fast)
            else:
                row_scale = out_scale
                if const_expr(SCALE_MODE == SCALE_SCALAR):
                    row_scale = out_scale * fx.Float32(AttnScale[fx.Int32(0)])
                if const_expr(SCALE_MODE == SCALE_TOKEN):
                    tok = fx.Int32(ctx.q_tok_base) + row_in_seq.select(
                        q_row_i32, fx.Int32(0)
                    )
                    row_scale = out_scale * fx.Float32(AttnScale[tok])
                _scale_o_accs(v_o, row_scale, traits, ctx.fm_fast)

            if const_expr(traits.DUALWAVE_SWP_ENABLE_STAGGER):
                _stagger_extra_barrier_if_zero(stagger_i32)
            else:
                _s_barrier()

            output_store.store_final_o(v_o, ctx.q_row)

            if const_expr(SOFTMAX):
                # M = m + log2(l) (base 2; m is already alpha*log2e-scaled),
                # -inf through its bit pattern on rows with no key.
                l_val = fx.Float32(l_row)
                has_key = l_val > fx.Float32(0.0)
                safe_l = has_key.select(l_val, fx.Float32(1.0))
                lse = fx.Float32(m_row) + fx.log2(safe_l, fastmath="afn")
                lse_bits = has_key.select(lse.bitcast(fx.Int32), fx.Int32(-8388608))
                writer = row_in_seq & (fx.Int32(ctx.lane) < fx.Int32(32))
                lse_row = fx.Int64(fx.Int32(ctx.q_tok_base) + q_row_i32)
                head = fx.Int32(ctx.q_head_idx)

                @flyc.jit
                def _store_lse(lse_out, writer, lse_row, head, lse_bits):
                    if writer:
                        lse_out[lse_row, head] = lse_bits.bitcast(fx.Float32)

                _store_lse(LseOut, writer, lse_row, head, lse_bits)

        if active is None:
            _main_body()
        else:

            @flyc.jit
            def _run_body_if_active():
                if active:
                    _main_body()

            _run_body_if_active()

    @flyc.jit
    def launch_hstu_ragged_fwd_gfx950(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        LseOut: fx.Tensor,
        CuSeq: fx.Tensor,
        NumTargets: fx.Tensor,
        AttnScale: fx.Tensor,
        SortIdx: fx.Tensor,
        num_seqs: fx.Int32,
        max_seq_len: fx.Int32,
        stride_q_n: fx.Int32,
        stride_kv_n: fx.Int32,
        score_scale: fx.Float32,
        out_scale: fx.Float32,
        max_attn_len: fx.Int32,
        contextual_seq_len: fx.Int32,
        full_attn_size: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        # The cache tag carries every compile-time choice into FlyDSL's key.
        _ = _cache_tag
        num_q_blocks = (fx.Index(max_seq_len) + BLOCK_M - 1) // BLOCK_M
        hstu_ragged_fwd_gfx950_kernel(
            Q,
            K,
            V,
            O,
            LseOut,
            CuSeq,
            NumTargets,
            AttnScale,
            SortIdx,
            stride_q_n,
            stride_kv_n,
            score_scale,
            out_scale,
            max_attn_len,
            contextual_seq_len,
            full_attn_size,
            value_attrs={
                "rocdl.waves_per_eu": traits.WAVES_PER_EU,
                "rocdl.flat_work_group_size": f"{traits.BLOCK_SIZE},{traits.BLOCK_SIZE}",
                "passthrough": [
                    ["denormal-fp-math-f32", "preserve-sign,preserve-sign"],
                    ["no-nans-fp-math", "true"],
                    ["unsafe-fp-math", "true"],
                ],
            },
        ).launch(
            grid=(traits.NUM_HEADS_Q, num_q_blocks, fx.Index(num_seqs)),
            block=(traits.BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    compile_hints = {
        "fast_fp_math": True,
        "unsafe_fp_math": True,
        "llvm_options": {
            "enable-post-misched": False,
            "lsr-drop-solution": True,
        },
    }
    cf_cache = {}

    def launch(*args):
        tensors = args[:9]
        key = tuple(_tensor_cache_sig(t) for t in tensors)
        cf = cf_cache.get(key)
        if cf is not None:
            return cf(*args)
        with CompilationContext.compile_hints(compile_hints):
            cf = flyc.compile(launch_hstu_ragged_fwd_gfx950, *args)
        cf_cache[key] = cf
        return None

    return launch


def hstu_ragged_fwd_gfx950(
    N: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    spec: MaskSpec,
    num_targets: Optional[torch.Tensor],
    attn_scale: Optional[torch.Tensor],
    scale_mode: str,
    max_attn_len: int,
    contextual_seq_len: int,
    full_attn_size: int,
    softmax: bool,
    out: torch.Tensor,
    M: torch.Tensor,
    sort_idx: Optional[torch.Tensor] = None,
    reverse_q_blocks: bool = False,
) -> None:
    """Run the kernel into ``out`` and ``M``; ``sort_idx`` (int32, sequences
    longest first) turns on the sorted schedule. Arguments as normalized by
    :func:`mslk.attention.flydsl.hstu.ragged_fwd.hstu_ragged_fwd`."""
    num_tokens, num_heads, head_dim = q.shape
    # The pipeline addresses heads as head * D and shares one token stride
    # between Q and O, and one between K and V.
    if q.stride() != out.stride():
        q = q.contiguous()
    if q.stride(1) != head_dim or q.stride(2) != 1:
        q = q.contiguous()
    if (
        k.stride(1) != head_dim
        or v.stride(1) != head_dim
        or k.stride(0) != v.stride(0)
        or k.stride(2) != 1
        or v.stride(2) != 1
    ):
        k, v = k.contiguous(), v.contiguous()
    launch = build_ragged_fwd_gfx950(
        num_heads,
        head_dim,
        "bf16" if q.dtype == torch.bfloat16 else "f16",
        spec.mask_type,
        spec.has_window,
        spec.has_contextual,
        spec.has_targets,
        spec.has_full_attn,
        scale_mode,
        softmax,
        sort_idx is not None,
        reverse_q_blocks,
    )
    num_seqs = seq_offsets.shape[0] - 1
    if softmax:
        score_scale, out_scale = alpha * _LOG2E, 1.0
    else:
        score_scale, out_scale = alpha * -_LOG2E, -1.0 / _LOG2E
        if scale_mode == SCALE_INV_N:
            out_scale /= N
    launch(
        q,
        k,
        v,
        out,
        M,
        seq_offsets,
        num_targets,
        attn_scale,
        seq_offsets if sort_idx is None else sort_idx,
        num_seqs,
        N,
        q.stride(0),
        k.stride(0),
        score_scale,
        out_scale,
        max_attn_len,
        contextual_seq_len,
        full_attn_size,
        torch.cuda.current_stream(q.device),
    )
