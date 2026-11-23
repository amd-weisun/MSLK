# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# The KV loop (K through LDS, V prefetched to registers then LDS, MFMA32
# register layouts, 128-bit permlane O stores) is the N32 path of MSLK's
# generic flash attention, mslk/attention/flydsl/flash_attn_generic.py:
#
#   SPDX-License-Identifier: Apache-2.0
#   Copyright (c) 2025 FlyDSL Project Contributors

# pyre-unsafe

"""Ragged HSTU self-attention forward on the generic flash-attention loop.

Covers what the gfx950 dual-wave pipeline (:mod:`.ragged_fwd_gfx950`) does
not, and is faster where its 256-row blocks waste work: head dim 256 (any
multiple of 32 from 64 to 256), short sequences and short windows, with
64-row query blocks of two waves. The HSTU pieces are the same: a logical ->
physical KV tile map (contextual prefix, then the window or causal range),
T1's mask per score on tiles that are not fully visible, SiLU without
running statistics or the online softmax with ``alpha * log2(e)``.

Q/K/V/O token strides are compile-time here (``H * D``); the caller passes
contiguous tensors.
"""

import functools
import math as host_math
from typing import Optional

import torch

from .common import LOWER, MaskSpec

_LOG2E = host_math.log2(host_math.e)

BLOCK_M = 64
BLOCK_N = 64

SCALE_INV_N = "inv_n"
SCALE_SCALAR = "scalar"
SCALE_TOKEN = "token"

# Window value that turns on the generic helpers' guards for rows whose keys
# are all masked (finite running max, 0 output) without bounding anything:
# this kernel computes its own KV range and mask.
_GUARDS_ONLY_WINDOW = 1 << 30


def decline_reason(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax: bool,
    arch: str,
) -> Optional[str]:
    """Why this kernel cannot take a call, or None."""
    if not arch.startswith("gfx950"):
        # gfx942 builds of the generic loop take code paths (K/V global
        # prefetch) this kernel does not carry.
        return f"needs gfx950, got {arch!r}"
    if q.dtype not in (torch.bfloat16, torch.float16):
        return f"unsupported dtype {q.dtype}"
    head_dim = q.shape[2]
    if head_dim % 32 or head_dim < 64 or head_dim > 256 or v.shape[2] != head_dim:
        return (
            "needs D a multiple of 32 in [64, 256] and V's dim equal to D, "
            f"got {head_dim}, {v.shape[2]}"
        )
    return None


@functools.lru_cache(maxsize=256)
def build_ragged_fwd_generic(
    num_heads: int,
    head_dim: int,
    dtype_str: str,
    arch: str,
    mask_type: str,
    has_window: bool,
    has_contextual: bool,
    has_targets: bool,
    has_full_attn: bool,
    scale_mode: str,
    softmax: bool,
    sorted_order: bool = False,
    reverse_q_blocks: bool = False,
    block_m: int = BLOCK_M,
    waves_per_eu: int = 0,
):
    """Compile-time specialization; returns ``launch(...)``. ``sorted_order``
    as in :func:`.ragged_fwd_gfx950.build_ragged_fwd_gfx950`."""
    import flydsl.compiler as flyc  # pyre-ignore[21]
    import flydsl.expr as fx  # pyre-ignore[21]
    from flydsl._mlir import ir  # pyre-ignore[21]
    from flydsl.compiler.kernel_function import CompilationContext  # pyre-ignore[21]
    from flydsl.expr import const_expr, gpu, range_constexpr, rocdl  # pyre-ignore[21]
    from flydsl.expr.typing import T, Vector as Vec  # pyre-ignore[21]
    from flydsl.utils.smem_allocator import SmemAllocator  # pyre-ignore[21]
    from mslk.attention.flydsl._common import tensor_cache_sig as _tensor_cache_sig
    from mslk.attention.flydsl.flash_attn_utils import (
        _make_flash_attn_generic_traits,
        _waitcnt_vm_n,
        as_mlir_value,
        GenericFlashAttnContext,
        GenericGemmHelper,
        GenericKvGmemToLdsLoader,
        GenericKvLdsToVgprLoader,
        GenericQLoader,
        GenericSoftmaxHelper,
        GenericStoreHelper,
        scf_if_dispatch,
    )

    from .common import device_ops, seq_mask

    # D 256 is VGPR-bound; one wave per EU lets the compiler avoid spills.
    if waves_per_eu == 0:
        waves_per_eu = 1 if head_dim >= 256 else 2
    BLOCK_M_ = block_m
    traits = _make_flash_attn_generic_traits(
        num_heads,
        num_heads,
        head_dim,
        arch,
        causal=False,
        dtype_str=dtype_str,
        flat_work_group_size=block_m * 2,
        block_m=block_m,
        path_tag="N32",
        varlen=True,
        cross_seqlen=False,
        paged=False,
        kv_cache_layout="linear",
        waves_per_eu=waves_per_eu,
        daz=True,
        fast_fp_math=True,
        unsafe_fp_math=True,
        sm_scale=None,
        skip_kv_pad_mask=None,
        return_lse=False,
        window_left=_GUARDS_ONLY_WINDOW,
        has_bias=False,
        causal_top_left=False,
        has_dropout=False,
        gappy_kv=False,
        num_kv_splits=1,
        q_pack_qlen=0,
        q_pack_group=0,
        kv_lens=False,
        paged_kv_dma=False,
        paged_k_prefetch=False,
        high_concurrency=False,
        fused_combine=False,
        paged_pid_regs=False,
        head_dim_qk=None,
        head_dim_v=None,
    )
    assert traits.BLOCK_M == block_m and traits.BLOCK_N == BLOCK_N
    assert traits.N_SUBTILES == 1 and not traits.ENABLE_DMA
    assert not traits.ENABLE_GFX942_KV_GPFETCH and not traits.ENABLE_PREFETCH_3BUF

    # Scalars only in the kernels' closures (FlyDSL keys its compile cache on
    # scalar closure values); the traits ride in through _cache_tag.
    MASK_TYPE = mask_type
    HAS_WINDOW = has_window
    HAS_CONTEXTUAL = has_contextual
    HAS_TARGETS = has_targets
    HAS_FULL_ATTN = has_full_attn
    SCALE_MODE = scale_mode
    SOFTMAX = softmax
    SORTED = sorted_order
    REVERSE_Q_BLOCKS = reverse_q_blocks
    K_SUB_N = traits.K_SUB_N
    D_CHUNKS = traits.D_CHUNKS
    CAN_SKIP_MASK = MASK_TYPE == LOWER and not HAS_CONTEXTUAL
    _cache_tag = (
        traits.cache_tag,
        "hstu_generic",
        MASK_TYPE,
        HAS_WINDOW,
        HAS_CONTEXTUAL,
        HAS_TARGETS,
        HAS_FULL_ATTN,
        SCALE_MODE,
        SOFTMAX,
        SORTED,
        REVERSE_Q_BLOCKS,
        BLOCK_M_,
        waves_per_eu,
    )
    # Non-vectorized MFMA32 score layout: lane column offsets per register.
    MOFF = (0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27)

    allocator = SmemAllocator(
        None, arch=arch, global_sym_name="hstu_ragged_fwd_generic_smem"
    )
    lds_kv_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_kv_offset + traits.LDS_KV_TOTAL_SIZE * 2

    @flyc.kernel(known_block_size=[traits.BLOCK_SIZE, 1, 1])
    def hstu_ragged_fwd_generic_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        LseOut: fx.Tensor,
        CuSeq: fx.Tensor,
        NumTargets: fx.Tensor,
        AttnScale: fx.Tensor,
        SortIdx: fx.Tensor,
        max_seq_len: fx.Int32,
        score_scale: fx.Float32,
        out_scale: fx.Float32,
        max_attn_len: fx.Int32,
        contextual_seq_len: fx.Int32,
        full_attn_size: fx.Int32,
    ):
        _ = _cache_tag
        ctx = GenericFlashAttnContext(
            traits, K, V, max_seq_len, max_seq_len, allocator, lds_kv_offset
        )
        ctx.init_types_and_pointers()
        gemm_helper = GenericGemmHelper(ctx)
        softmax_helper = GenericSoftmaxHelper(ctx)
        kv_gmem_to_lds = GenericKvGmemToLdsLoader(ctx)
        kv_lds_to_vgpr = GenericKvLdsToVgprLoader(ctx)
        q_loader = GenericQLoader(ctx)
        store_helper = GenericStoreHelper(ctx)

        ctx.pin_kernel_args(Q, CuSeq, CuSeq, CuSeq, CuSeq, fx.Int32(0))
        ctx.init_sequence_indices()
        ctx.init_lds_view()
        ctx.init_thread_mapping()
        ctx.init_block_mapping()
        if const_expr(SORTED):
            ctx.batch_idx = fx.Index(fx.Int32(SortIdx[fx.Int32(ctx.batch_idx)]))
        if const_expr(REVERSE_Q_BLOCKS):
            ctx.q_tile_idx = ctx.num_q_tiles - fx.Index(1) - ctx.q_tile_idx
            ctx.q_start = ctx.q_tile_idx * traits.BLOCK_M
        ctx.init_sequence_lengths(CuSeq, CuSeq, None)
        ctx.init_load_mapping()
        ctx.init_kv_batch_pointers()
        ctx.init_descriptors(Q, O)
        ctx.init_q_row()
        q_b_packs = q_loader.load_all(ctx.q_rsrc, ctx.q_row)
        ctx.init_constants(1.0)
        if const_expr(SOFTMAX):
            # The softmax helper scales raw scores by this (alpha * log2 e).
            ctx.c_sm_scale_log2e = score_scale

        # ---- HSTU mask state and the logical -> physical KV tile map ----
        ops = device_ops()
        seq_len_i32 = fx.Int32(ctx.seqlen_q_b)
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
        prefix_end, main_start, main_end = sm.kv_tiles(q_start_i32, BLOCK_M_, BLOCK_N)
        n_logical = prefix_end + (main_end - main_start)
        tile_gap = main_start - prefix_end

        q_row_i32 = ctx.q_row_i32
        q_row_id = sm.to_id(q_row_i32)
        block_row_last = q_start_i32 + fx.Int32(BLOCK_M_ - 1)
        col_lane_base = fx.Int32(ctx.lane_div_32) * fx.Int32(4)
        masked_score = ctx.c_neg_inf if SOFTMAX else ctx.c_zero_f

        def tile_needs_mask(c0):
            """False only when every score of the tile is visible to every row
            of the block (uniform across the block)."""
            c_last = c0 + fx.Int32(BLOCK_N - 1)
            need = c_last >= seq_len_i32
            need = need | (sm.to_id(c_last) >= sm.to_id(q_start_i32))
            if const_expr(HAS_WINDOW):
                need = need | (
                    sm.to_id(c0) < sm.to_id(block_row_last) - sm.max_attn_len
                )
            return need

        def mask_one(score, col):
            keep = sm.visible(q_row_i32, col, q_row_id, sm.to_id(col)) & (
                col < seq_len_i32
            )
            return keep.select(score, masked_score)

        def hstu_mask(s_lo, s_hi, c0):
            base = c0 + col_lane_base

            def _apply(_names, *scores):
                out = []
                for r in range_constexpr(16):
                    col = base + fx.Int32(MOFF[r])
                    out.append(mask_one(scores[2 * r], col))
                    out.append(mask_one(scores[2 * r + 1], col + fx.Int32(K_SUB_N)))
                return out

            interleaved = [x for r in range(16) for x in (s_lo[r], s_hi[r])]
            if const_expr(CAN_SKIP_MASK):
                masked = scf_if_dispatch(
                    tile_needs_mask(c0),
                    _apply,
                    state_names=tuple("_hm%d" % i for i in range(32)),
                    state_values=interleaved,
                )
            else:
                masked = _apply(None, *interleaved)
            return [masked[2 * r] for r in range(16)], [
                masked[2 * r + 1] for r in range(16)
            ]

        def silu_vals(scores):
            # g = s * alpha * -log2e; g / (1 + exp2(g)) = (-log2e) silu(alpha s).
            out = []
            for x in scores:
                g = fx.Float32(x) * score_scale
                e = fx.Float32(rocdl.exp2(T.f32, as_mlir_value(g)))
                r = fx.Float32(rocdl.rcp(T.f32, as_mlir_value(e + fx.Float32(1.0))))
                out.append(as_mlir_value(g * r))
            return out

        def p_packs_nearest(p_vals):
            # Softmax probabilities round to nearest (the helper's bf16 packs
            # truncate; the bias shows on rows that see few keys).
            vals = [fx.Float32(x).to(ctx.elem_dtype) for x in p_vals]
            step = traits.MFMA_LANE_K
            return [
                Vec.from_elements(
                    vals[i * step : (i + 1) * step], ctx.elem_dtype
                ).ir_value()
                for i in range_constexpr(traits.PV_K_STEPS)
            ]

        def p_packs(p_vals):
            if const_expr(SOFTMAX):
                return p_packs_nearest(p_vals)
            return softmax_helper.build_p_packs(p_vals)

        init_args = [ctx.c_neg_inf, ctx.c_zero_f]
        for _ in range_constexpr(D_CHUNKS):
            init_args.append(ctx.c_zero_v16f32)
        ones_vec = Vec.filled(16, 1.0, fx.Float32)

        loop_results = init_args
        for t, it in range(
            fx.Index(0), fx.Index(n_logical), fx.Index(1), init=init_args
        ):
            m_running = it[0]
            l_running = it[1]
            o_accs = [it[2 + i] for i in range_constexpr(D_CHUNKS)]
            t_i32 = fx.Int32(t)
            phys = (t_i32 < prefix_end).select(t_i32, t_i32 + tile_gap)
            c0 = phys * fx.Int32(BLOCK_N)
            kv_start = fx.Index(c0)

            kv_gmem_to_lds.coop_load_k(kv_start, 0, pid=None)
            gpu.barrier()
            k_base = kv_gmem_to_lds.k_buf_base(0)
            v_vecs = kv_gmem_to_lds.coop_load_v_global(kv_start, pid=None)

            k_lo, k_hi = kv_lds_to_vgpr.load_k_packs(k_base)
            s_acc_lo, s_acc_hi = gemm_helper.gemm1_accumulate(
                kv_lds_to_vgpr, k_lo, k_hi, q_b_packs, kv_gmem_to_lds, kv_start
            )
            s_lo, s_hi = softmax_helper.split_scores(s_acc_lo, s_acc_hi)
            s_lo, s_hi = hstu_mask(s_lo, s_hi, c0)

            if const_expr(SOFTMAX):
                m_new, l_new, corr, p_lo, p_hi = softmax_helper.online_softmax(
                    m_running, l_running, s_lo, s_hi
                )
                o_accs, corr_vec = softmax_helper.rescale_o_accs(o_accs, corr)
            else:
                m_new, l_new = m_running, l_running
                p_lo, p_hi = silu_vals(s_lo), silu_vals(s_hi)
                corr_vec = ones_vec

            v_base = kv_gmem_to_lds.v_buf_base(0)
            _waitcnt_vm_n(0)
            kv_gmem_to_lds.coop_store_v_lds(v_vecs, 0)
            rocdl.sched_group_barrier(rocdl.mask_dswr, 1, 0)
            gpu.barrier()

            o_accs = gemm_helper.gemm2_pv(
                kv_lds_to_vgpr,
                o_accs,
                p_packs(p_lo),
                p_packs(p_hi),
                v_base,
                corr_vec,
            )
            loop_results = yield [m_new, l_new] + o_accs

        row_in_seq = q_row_i32 < seq_len_i32
        if const_expr(SOFTMAX):
            m_final = fx.Float32(loop_results[0])
            l_final = fx.Float32(loop_results[1])
            # M = m * alpha * log2e + log2(l), base 2; -inf (bit pattern) on
            # rows with no key.
            has_key = l_final > fx.Float32(0.0)
            safe_l = has_key.select(l_final, fx.Float32(1.0))
            lse = m_final * score_scale + fx.log2(safe_l, fastmath="afn")
            lse_bits = has_key.select(lse.bitcast(fx.Int32), fx.Int32(-8388608))
            writer = row_in_seq & (fx.Int32(ctx.lane_div_32) == fx.Int32(0))
            lse_row = fx.Int64(fx.Int32(ctx.q_tok_base) + q_row_i32)
            head = fx.Int32(ctx.q_head_idx)

            @flyc.jit
            def _store_lse(lse_out, writer, lse_row, head, lse_bits):
                if writer:
                    lse_out[lse_row, head] = lse_bits.bitcast(fx.Float32)

            _store_lse(LseOut, writer, lse_row, head, lse_bits)
            store_helper.normalize_and_store_o(loop_results, ctx.o_rsrc, ctx.q_row)
        else:
            row_scale = out_scale
            if const_expr(SCALE_MODE == SCALE_SCALAR):
                row_scale = out_scale * fx.Float32(AttnScale[fx.Int32(0)])
            if const_expr(SCALE_MODE == SCALE_TOKEN):
                tok = fx.Int32(ctx.q_tok_base) + row_in_seq.select(
                    q_row_i32, fx.Int32(0)
                )
                row_scale = out_scale * fx.Float32(AttnScale[tok])
            scale_vec = Vec.from_elements([row_scale], fx.Float32).broadcast_to(16)
            v_o = [
                Vec(loop_results[2 + dc]) * scale_vec
                for dc in range_constexpr(D_CHUNKS)
            ]
            store_helper._store_permlane_o(v_o, ctx.o_rsrc, ctx.q_row)

    @flyc.jit
    def launch_hstu_ragged_fwd_generic(
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
        score_scale: fx.Float32,
        out_scale: fx.Float32,
        max_attn_len: fx.Int32,
        contextual_seq_len: fx.Int32,
        full_attn_size: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        allocator.finalized = False
        cctx = CompilationContext.get_current()
        with ir.InsertionPoint(cctx.gpu_module_body):
            allocator.finalize()
        num_q_tiles = (fx.Index(max_seq_len) + BLOCK_M_ - 1) // BLOCK_M_
        grid_x = fx.Index(num_seqs) * num_q_tiles * traits.NUM_HEADS_Q
        hstu_ragged_fwd_generic_kernel(
            Q,
            K,
            V,
            O,
            LseOut,
            CuSeq,
            NumTargets,
            AttnScale,
            SortIdx,
            max_seq_len,
            score_scale,
            out_scale,
            max_attn_len,
            contextual_seq_len,
            full_attn_size,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu,
                "rocdl.flat_work_group_size": f"{traits.BLOCK_SIZE},{traits.BLOCK_SIZE}",
                "passthrough": [
                    ["denormal-fp-math-f32", "preserve-sign,preserve-sign"],
                    ["no-nans-fp-math", "true"],
                    ["unsafe-fp-math", "true"],
                ],
            },
        ).launch(
            grid=(grid_x, 1, 1),
            block=(traits.BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    compile_hints = {
        "fast_fp_math": True,
        "unsafe_fp_math": True,
        "llvm_options": {"enable-post-misched": False, "lsr-drop-solution": True},
    }
    cf_cache = {}

    def launch(*args):
        key = tuple(_tensor_cache_sig(t) for t in args[:9])
        cf = cf_cache.get(key)
        if cf is not None:
            return cf(*args)
        with CompilationContext.compile_hints(compile_hints):
            cf = flyc.compile(launch_hstu_ragged_fwd_generic, *args)
        cf_cache[key] = cf
        return None

    return launch


def hstu_ragged_fwd_generic(
    N: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    spec: MaskSpec,
    num_targets: torch.Tensor,
    attn_scale: torch.Tensor,
    scale_mode: str,
    max_attn_len: int,
    contextual_seq_len: int,
    full_attn_size: int,
    softmax: bool,
    out: torch.Tensor,
    M: torch.Tensor,
    arch: str,
    sort_idx: Optional[torch.Tensor] = None,
    reverse_q_blocks: bool = False,
    block_m: int = BLOCK_M,
    waves_per_eu: int = 0,
) -> None:
    """Run the kernel into ``out`` (contiguous) and ``M``. Arguments as
    normalized by :func:`mslk.attention.flydsl.hstu.ragged_fwd.hstu_ragged_fwd`."""
    num_tokens, num_heads, head_dim = q.shape
    # Token strides are compile-time (H * D) in this kernel.
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    assert out.is_contiguous()
    launch = build_ragged_fwd_generic(
        num_heads,
        head_dim,
        "bf16" if q.dtype == torch.bfloat16 else "f16",
        arch,
        spec.mask_type,
        spec.has_window,
        spec.has_contextual,
        spec.has_targets,
        spec.has_full_attn,
        scale_mode,
        softmax,
        sort_idx is not None,
        reverse_q_blocks,
        block_m,
        waves_per_eu,
    )
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
        seq_offsets.shape[0] - 1,
        N,
        score_scale,
        out_scale,
        max_attn_len,
        contextual_seq_len,
        full_attn_size,
        torch.cuda.current_stream(q.device),
    )
