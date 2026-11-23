# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# The tile pipeline (K/V register prefetch, swizzled K and transposed V LDS
# tiles, the operand-swapped GEMM1 whose result feeds GEMM2 from registers,
# the SiLU base change and the group-major grid) is adapted from AITER's
# FlyDSL HSTU forward, aiter/ops/flydsl/kernels/hstu/hstu_attention_fwd.py:
#
#   SPDX-License-Identifier: MIT
#   Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# The mask, KV-tile ranges, scale modes and softmax heads follow T1
# (mslk.attention.hstu._torch_ref) and the TLX gfx950 ragged HSTU kernel
# (facebookexperimental/triton third_party/tlx/ops/kernels/hstu_attn/gfx950.py).

# pyre-unsafe

"""FlyDSL ragged HSTU self-attention forward for CDNA3/CDNA4.

``out_i = sum_j valid(i, j) * silu(alpha * q_i . k_j) * scale_i * v_j`` on SiLU
heads, with ``scale`` ``1/N``, one scalar or one value per query token;
softmax heads use ``softmax(alpha * q_i . k_j)`` over the valid keys and also
write the base-2 LSE ``M``. ``valid`` is T1's ragged mask
(:mod:`mslk.attention.flydsl.hstu.common`).

One program computes one (sequence, head, BLOCK_M query tile). It visits only
the KV tiles that can hold a visible key (window, causal triangle, contextual
columns, full-attention rows), and skips the per-element mask on tiles that
are fully visible. Which features are present is compiled in; their values
are run-time arguments.

Tensors are packed ``[tokens, heads, dim]`` with any token and head strides;
the last dim must be contiguous. Row addresses are 64-bit.
"""

import functools
import math as host_math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .common import LOWER, MaskSpec

_LOG2E = host_math.log2(host_math.e)

WARP_SIZE = 64
# Grid decoded group-major across the XCDs for L2 locality.
NUM_GRID_GROUPS = 8
MFMA_M = 16
MFMA_N = 16
# P is reused from the four-value C fragment, so P*V stays 16 deep.
MFMA_K = 16
MFMA_LANE_K = 4
MFMA_ELEMS_PER_LANE = (MFMA_M * MFMA_N) // WARP_SIZE

SCALE_INV_N = "inv_n"
SCALE_SCALAR = "scalar"
SCALE_TOKEN = "token"

_DTYPES = {torch.bfloat16: "bf16", torch.float16: "f16"}


@dataclass(frozen=True)
class FwdConfig:
    """Tile configuration."""

    block_m: int = 128
    block_n: int = 32
    num_waves: int = 4
    waves_per_eu: int = 2


def default_config(head_dim: int, hidden_dim: int) -> FwdConfig:
    """Heuristic tiles until the tuned defaults land (AITER's MI300X sweep)."""
    dim = max(hidden_dim, ((head_dim + 63) // 64) * 64)
    if dim >= 256:
        return FwdConfig(128, 16, 4, 0)
    if dim >= 192:
        return FwdConfig(128, 32, 4, 2)
    if dim >= 128:
        return FwdConfig(128, 64, 4, 2)
    return FwdConfig(128, 32, 4, 2)


# Fallbacks for head dims the heuristic tile does not divide (80, 96, 160 ...):
# fewer threads per load pass, in the order tried.
_FALLBACK_CONFIGS = tuple(
    FwdConfig(block_m, block_n, waves, 0)
    for waves in (4, 2, 1)
    for block_m in (128, 64)
    for block_n in (32, 16, 64)
    if block_m % (waves * MFMA_M) == 0
) + (
    # Load passes of 3 and 5 waves, for dims with a factor of 3 or 5 (AITER's
    # map for 96, 160 and 192).
    FwdConfig(96, 48, 3, 0),
    FwdConfig(160, 80, 5, 0),
)


def _first_supported(
    q: torch.Tensor, head_dim: int, hidden_dim: int, arch: str
) -> FwdConfig:
    cfg = default_config(head_dim, hidden_dim)
    if check_supported(q, head_dim, hidden_dim, cfg, arch) is None:
        return cfg
    for cfg in _FALLBACK_CONFIGS:
        if check_supported(q, head_dim, hidden_dim, cfg, arch) is None:
            return cfg
    return default_config(head_dim, hidden_dim)


def _arch() -> str:
    from flydsl.runtime.device import get_rocm_arch  # pyre-ignore[21]

    return get_rocm_arch() or ""


def _dim_mfma(head_dim: int, arch: str) -> Tuple[int, int]:
    """QK contraction depth and operand elements per lane: gfx950's 16x16x32
    where the head dim allows, else 16x16x16."""
    mfma_k = 16 if arch.startswith("gfx942") else 32
    if head_dim % mfma_k != 0:
        mfma_k = 16
    return mfma_k, (MFMA_M * mfma_k) // WARP_SIZE


def _dma_params(arch: str) -> Tuple[int, int, int]:
    """(DMA elements, K swizzle rows, K swizzle shift); see AITER's
    ``_arch_dma_params``: gfx942 has 32 LDS banks, gfx950 64."""
    if arch.startswith("gfx942"):
        return 2, 16, 2
    return 8, 8, 3


def _lds_capacity(arch: str) -> int:
    return 160 * 1024 if arch.startswith("gfx950") else 64 * 1024


def check_supported(
    q: torch.Tensor,
    head_dim: int,
    hidden_dim: int,
    cfg: FwdConfig,
    arch: Optional[str] = None,
) -> Optional[str]:
    """Why this kernel cannot take a call, or None if it can."""
    arch = _arch() if arch is None else arch
    if not (arch.startswith("gfx942") or arch.startswith("gfx950")):
        return f"unsupported architecture {arch!r}"
    if q.dtype not in _DTYPES:
        return f"unsupported dtype {q.dtype}"
    if head_dim % 16 or hidden_dim % 16 or head_dim <= 0 or hidden_dim <= 0:
        return (
            f"head dims must be positive multiples of 16, got {head_dim}, {hidden_dim}"
        )
    threads = cfg.num_waves * WARP_SIZE
    if cfg.block_m % (cfg.num_waves * MFMA_M) or cfg.block_n % MFMA_N:
        return f"bad tile {cfg}"
    dma_elems, _, _ = _dma_params(arch)
    head_dim_k = ((head_dim + 63) // 64) * 64
    for dim in (head_dim_k, hidden_dim):
        vec = (
            8
            if dim % 8 == 0 and (cfg.block_n * dim) % (threads * 8) == 0
            else dma_elems
        )
        if (cfg.block_n * dim) % (threads * dma_elems) or threads % (dim // vec):
            return f"tile {cfg} does not divide the load pass for dim {dim}"
        rows = threads // (dim // vec)
        if not (cfg.block_n % rows == 0 or rows > cfg.block_n):
            return f"tile {cfg} does not divide the load rows for dim {dim}"
    lds = cfg.block_n * head_dim_k * 2 + hidden_dim * (cfg.block_n + 8) * 2
    if lds > _lds_capacity(arch):
        return f"LDS tile of {lds} B too large"
    return None


@functools.lru_cache(maxsize=1024)
def build_ragged_fwd(
    head_dim: int,
    hidden_dim: int,
    dtype_str: str,
    spec: MaskSpec,
    scale_mode: str,
    softmax: bool,
    cfg: FwdConfig,
    arch: str,
):
    """Compile-time specialization of the kernel; returns its host launcher."""
    import flydsl.compiler as flyc  # pyre-ignore[21]
    import flydsl.expr as fx  # pyre-ignore[21]
    from flydsl.expr import const_expr, gpu, range_constexpr, rocdl  # pyre-ignore[21]
    from flydsl.expr.typing import Vector as Vec  # pyre-ignore[21]

    from .common import device_ops, seq_mask

    if softmax:
        raise NotImplementedError("softmax heads are not implemented yet")

    # FlyDSL keys its compile cache on the kernel's *scalar* closure values
    # (and types) only: a dataclass captured by the kernel is left out of the
    # key, so two variants would share one cached binary. Unpack every
    # compile-time parameter into a scalar before the kernel captures it.
    BLOCK_M, BLOCK_N = cfg.block_m, cfg.block_n
    NUM_WAVES = cfg.num_waves
    WAVES_PER_EU = cfg.waves_per_eu
    MASK_TYPE = spec.mask_type
    HAS_WINDOW = spec.has_window
    HAS_CONTEXTUAL = spec.has_contextual
    HAS_TARGETS = spec.has_targets
    HAS_FULL_ATTN = spec.has_full_attn
    SCALE_MODE = scale_mode
    BLOCK_THREADS = NUM_WAVES * WARP_SIZE
    ROWS_PER_WAVE = BLOCK_M // NUM_WAVES
    Q_SUBTILES = ROWS_PER_WAVE // MFMA_M
    KV_SUBTILES = BLOCK_N // MFMA_N
    MFMA_DIM_K, MFMA_DIM_LANE_K = _dim_mfma(head_dim, arch)
    MFMA_DIM_LANE_K_LOG2 = MFMA_DIM_LANE_K.bit_length() - 1
    DMA_ELEMS, K_SWZ_ROWS, K_SWZ_SHIFT = _dma_params(arch)
    assert K_SWZ_SHIFT >= MFMA_DIM_LANE_K_LOG2

    elem_dtype = fx.BFloat16 if dtype_str == "bf16" else fx.Float16
    is_bf16 = dtype_str == "bf16"

    K_STEPS = head_dim // MFMA_DIM_K
    # K's LDS row pitch rounds up to 64 so the XOR swizzle stays in-row; the
    # padding columns load as zero against a zero Q operand.
    HEAD_DIM_K = ((head_dim + 63) // 64) * 64
    K_COL_GUARD = head_dim < HEAD_DIM_K
    K_STEPS_K = HEAD_DIM_K // MFMA_DIM_K
    D_CHUNKS = hidden_dim // MFMA_M
    # V transposed in LDS ([d, kv]); +8 pads the row against bank conflicts.
    V_T_STRIDE = BLOCK_N + 8

    def _load_tiling(dim):
        wide = dim % 8 == 0 and (BLOCK_N * dim) % (BLOCK_THREADS * 8) == 0
        vec = 8 if wide else DMA_ELEMS
        threads_per_row = dim // vec
        rows_per_batch = BLOCK_THREADS // threads_per_row
        return (
            vec,
            threads_per_row,
            rows_per_batch,
            max(1, BLOCK_N // rows_per_batch),
            rows_per_batch > BLOCK_N,
        )

    VEC_K, THREADS_PER_ROW_K, ROWS_PER_BATCH_K, NUM_BATCHES_K, K_NEEDS_GUARD = (
        _load_tiling(HEAD_DIM_K)
    )
    VEC_V, THREADS_PER_ROW_V, ROWS_PER_BATCH_V, NUM_BATCHES_V, V_NEEDS_GUARD = (
        _load_tiling(hidden_dim)
    )
    assert VEC_K % MFMA_DIM_LANE_K == 0

    # Tiles fully below the diagonal need no per-element mask under a plain
    # causal mask (targets allowed): see the split in the kernel body.
    CAUSAL_SPLIT = MASK_TYPE == LOWER and not HAS_WINDOW and not HAS_CONTEXTUAL

    @fx.struct
    class SharedStorage:
        k: fx.Array[elem_dtype, BLOCK_N * HEAD_DIM_K, 16]
        v: fx.Array[elem_dtype, hidden_dim * V_T_STRIDE, 16]

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def hstu_ragged_fwd_kernel(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        seq_offsets: fx.Tensor,
        num_targets: fx.Tensor,
        attn_scale: fx.Tensor,
        out: fx.Tensor,
        num_heads: fx.Int32,
        num_q_tiles: fx.Int32,
        hz_per_group: fx.Int32,
        hz_total: fx.Int32,
        alpha_neg_log2e: fx.Float32,
        out_scale: fx.Float32,
        max_attn_len: fx.Int32,
        contextual_seq_len: fx.Int32,
        full_attn_size: fx.Int32,
    ) -> None:
        ops = device_ops()
        compute_type = fx.Float32.ir_type
        c_zero_dim_pack = Vec.filled(MFMA_DIM_LANE_K, 0.0, elem_dtype).ir_value()

        dim_mfma_acc, mfma_acc = _bind_mfma_accs(
            fx, Vec, elem_dtype, (MFMA_DIM_K, MFMA_DIM_LANE_K), (MFMA_K, MFMA_LANE_K)
        )

        tid = fx.Int32(gpu.thread_idx.x)
        wave_id = tid // fx.Int32(WARP_SIZE)
        lane = tid % fx.Int32(WARP_SIZE)
        lane_mod_16 = lane % fx.Int32(MFMA_N)
        lane_div_16 = lane // fx.Int32(MFMA_N)

        # ---- Group-major grid decode -> (sequence, head, query tile) ----
        block_id = fx.Int32(gpu.block_idx.x)
        grid_group = block_id % fx.Int32(NUM_GRID_GROUPS)
        pos_in_group = block_id // fx.Int32(NUM_GRID_GROUPS)
        local_hz_idx = pos_in_group // num_q_tiles
        q_tile_idx = pos_in_group % num_q_tiles
        hz_idx = grid_group * hz_per_group + local_hz_idx
        # The last group is padded past Z*H; padding programs get seq_len 0.
        block_valid = hz_idx < hz_total
        hz_idx = block_valid.select(hz_idx, fx.Int32(0))
        batch_idx = hz_idx // num_heads
        head_idx = hz_idx % num_heads

        seq_start = fx.Int32(seq_offsets[batch_idx])
        seq_len = fx.Int32(seq_offsets[batch_idx + fx.Int32(1)]) - seq_start
        seq_len = block_valid.select(seq_len, fx.Int32(0))
        num_target = fx.Int32(0)
        if const_expr(HAS_TARGETS):
            num_target = fx.Int32(num_targets[batch_idx])
        sm = seq_mask(
            MaskSpec(MASK_TYPE, HAS_WINDOW, HAS_CONTEXTUAL, HAS_TARGETS, HAS_FULL_ATTN),
            ops,
            seq_len,
            num_targets=num_target,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
            full_attn_size=full_attn_size,
        )

        def grouped_loader(t, dim, g):
            # (row, head) goes through the tensor's own i64-strided layout, so
            # row * stride cannot wrap; the in-row access is a small i32 one.
            in_row = fx.make_layout((dim // g, g), (g, 1))

            def load(row_i64, head_val, colgrp):
                sub = t[row_i64, head_val, None]
                return fx.make_view(fx.get_iter(sub), in_row)[colgrp, None].load()

            return load

        q_load = grouped_loader(q, head_dim, MFMA_DIM_LANE_K)
        v_load = grouped_loader(v, hidden_dim, VEC_V)
        k_load = grouped_loader(k, head_dim, VEC_K)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        k_smem = lds.k.view(
            fx.make_layout(
                (BLOCK_N, HEAD_DIM_K // MFMA_DIM_LANE_K, MFMA_DIM_LANE_K),
                (HEAD_DIM_K, MFMA_DIM_LANE_K, 1),
            )
        )
        v_smem = lds.v.view(
            fx.make_layout(
                (hidden_dim, V_T_STRIDE // MFMA_LANE_K, MFMA_LANE_K),
                (V_T_STRIDE, MFMA_LANE_K, 1),
            )
        )

        def k_swz_grp(tile_row, col_grp):
            # col ^ ((row & (ROWS-1)) << SHIFT), in MFMA_DIM_LANE_K-group units.
            return col_grp ^ (
                (tile_row & fx.Int32(K_SWZ_ROWS - 1))
                << fx.Int32(K_SWZ_SHIFT - MFMA_DIM_LANE_K_LOG2)
            )

        q_wave_base = q_tile_idx * fx.Int32(BLOCK_M) + wave_id * fx.Int32(ROWS_PER_WAVE)

        # ---- Q rows and register-resident Q packs (B operand) ----
        q_rows = []
        q_in_bounds = []
        for qg in range_constexpr(Q_SUBTILES):
            local = q_wave_base + fx.Int32(qg * MFMA_M) + lane_mod_16
            q_rows.append(local)
            q_in_bounds.append(local < seq_len)
        q_row_ids = [sm.to_id(q_rows[qg]) for qg in range_constexpr(Q_SUBTILES)]

        q_packs = []
        for ks in range_constexpr(K_STEPS):
            q_col = fx.Int32(ks * MFMA_DIM_K) + lane_div_16 * fx.Int32(MFMA_DIM_LANE_K)
            per_qg = []
            for qg in range_constexpr(Q_SUBTILES):
                safe = q_in_bounds[qg].select(seq_start + q_rows[qg], seq_start)
                raw = q_load(
                    fx.Int64(safe), head_idx, q_col // fx.Int32(MFMA_DIM_LANE_K)
                ).ir_value()
                per_qg.append(q_in_bounds[qg].select(raw, c_zero_dim_pack))
            q_packs.append(per_qg)

        # ---- SiLU gate ----
        # exp2(x * -log2e) == e^-x. With g = s * (alpha * -log2e), p = g * sigmoid
        # = (-log2e) * silu(alpha * s); out_scale carries the -1/log2e back.
        c_one_f = fx.Float32(1.0)
        c_zero_f = fx.Float32(0.0)

        def _rcp(x):
            return fx.Float32(fx.rocdl.rcp(compute_type, x.ir_value()))

        def silu_scale_batch(s_list):
            g = [s * alpha_neg_log2e for s in s_list]
            emu = [fx.exp2(gi, fastmath="afn") for gi in g]
            sig = [_rcp(c_one_f + e) for e in emu]
            return [g[i] * sig[i] for i in range(len(s_list))]

        def pack_p(vals):
            """Pack 4 f32 into a bf16/f16 GEMM2 A-operand fragment (bf16 truncates)."""
            if const_expr(is_bf16):
                c16 = fx.Int32(16)
                cmask = fx.Int32(0xFFFF0000)

                def bf16_pair(lo_f32, hi_f32):
                    lo_i32 = lo_f32.bitcast(fx.Int32)
                    hi_i32 = hi_f32.bitcast(fx.Int32)
                    return (hi_i32 & cmask) | lo_i32.shrui(c16)

                pairs = [bf16_pair(vals[0], vals[1]), bf16_pair(vals[2], vals[3])]
                return Vec.from_elements(pairs, fx.Int32).bitcast(elem_dtype).ir_value()
            elems = [x.to(elem_dtype) for x in vals]
            return Vec.from_elements(elems, elem_dtype).ir_value()

        # ---- KV tiles to visit: [0, prefix_end) then [main_start, main_end) ----
        q_start = q_tile_idx * fx.Int32(BLOCK_M)
        prefix_end, main_start, main_end = sm.kv_tiles(q_start, BLOCK_M, BLOCK_N)
        active = q_start < seq_len

        N_ACC = D_CHUNKS * Q_SUBTILES
        c_zero_v4f32 = Vec.filled(MFMA_ELEMS_PER_LANE, 0.0, fx.Float32).ir_value()

        k_load_row_in_batch = tid // fx.Int32(THREADS_PER_ROW_K)
        k_load_lane_in_row = tid % fx.Int32(THREADS_PER_ROW_K)
        k_load_col = k_load_lane_in_row * fx.Int32(VEC_K)
        k_load_col_grp = k_load_lane_in_row * fx.Int32(VEC_K // MFMA_DIM_LANE_K)

        def async_load_k_regs(kv_start, full_tile=False):
            """Issue K[kv_start] loads to registers; full_tile drops the
            in-sequence guard for tiles known to be inside the sequence."""
            vecs = []
            for b in range_constexpr(NUM_BATCHES_K):
                row = k_load_row_in_batch + fx.Int32(b * ROWS_PER_BATCH_K)
                tok = kv_start + row
                guard = None
                if const_expr(not full_tile):
                    guard = tok < seq_len
                if const_expr(K_NEEDS_GUARD):
                    rowg = row < fx.Int32(BLOCK_N)
                    guard = rowg if guard is None else (guard & rowg)
                if const_expr(K_COL_GUARD):
                    colg = k_load_col < fx.Int32(head_dim)
                    guard = colg if guard is None else (guard & colg)
                if const_expr(guard is None):
                    vecs.append(
                        k_load(
                            fx.Int64(seq_start + tok),
                            head_idx,
                            k_load_col // fx.Int32(VEC_K),
                        ).ir_value()
                    )
                else:
                    safe = guard.select(seq_start + tok, seq_start)
                    raw = k_load(
                        fx.Int64(safe), head_idx, k_load_col // fx.Int32(VEC_K)
                    ).ir_value()
                    vecs.append(
                        guard.select(raw, Vec.filled(VEC_K, 0.0, elem_dtype).ir_value())
                    )
            return vecs

        def store_k_regs_to_lds(vecs):
            for b in range_constexpr(NUM_BATCHES_K):
                row = k_load_row_in_batch + fx.Int32(b * ROWS_PER_BATCH_K)
                for h in range_constexpr(VEC_K // MFMA_DIM_LANE_K):
                    col_grp = k_load_col_grp + fx.Int32(h)
                    half = Vec.from_elements(
                        [
                            Vec(vecs[b])[h * MFMA_DIM_LANE_K + j]
                            for j in range_constexpr(MFMA_DIM_LANE_K)
                        ],
                        elem_dtype,
                    )
                    k_smem[row, k_swz_grp(row, col_grp), None].store(half)

        v_load_row_in_batch = tid // fx.Int32(THREADS_PER_ROW_V)
        v_load_lane_in_row = tid % fx.Int32(THREADS_PER_ROW_V)
        v_load_col = v_load_lane_in_row * fx.Int32(VEC_V)

        def async_load_v_regs(kv_start, full_tile=False):
            vecs = []
            for b in range_constexpr(NUM_BATCHES_V):
                row = v_load_row_in_batch + fx.Int32(b * ROWS_PER_BATCH_V)
                tok = kv_start + row
                guard = None
                if const_expr(not full_tile):
                    guard = tok < seq_len
                if const_expr(V_NEEDS_GUARD):
                    rowg = row < fx.Int32(BLOCK_N)
                    guard = rowg if guard is None else (guard & rowg)
                if const_expr(guard is None):
                    vecs.append(
                        v_load(
                            fx.Int64(seq_start + tok),
                            head_idx,
                            v_load_col // fx.Int32(VEC_V),
                        ).ir_value()
                    )
                else:
                    safe_tok = guard.select(seq_start + tok, seq_start)
                    raw = v_load(
                        fx.Int64(safe_tok), head_idx, v_load_col // fx.Int32(VEC_V)
                    ).ir_value()
                    vecs.append(
                        guard.select(raw, Vec.filled(VEC_V, 0.0, elem_dtype).ir_value())
                    )
            return vecs

        def store_v_regs_to_lds(vecs):
            """V scattered transposed to [d, kv] so GEMM2 reads 4 kv per ds_read_b64."""
            for b in range_constexpr(NUM_BATCHES_V):
                kv_row = v_load_row_in_batch + fx.Int32(b * ROWS_PER_BATCH_V)
                kv_grp = kv_row // fx.Int32(MFMA_LANE_K)
                kv_lane = kv_row % fx.Int32(MFMA_LANE_K)
                vv = Vec(vecs[b])
                for j in range_constexpr(VEC_V):
                    v_smem[v_load_col + fx.Int32(j), kv_grp, kv_lane] = vv[j]

        def read_k_packs(ng):
            local_k_row = fx.Int32(ng * MFMA_M) + lane_mod_16
            packs = []
            for ks in range_constexpr(K_STEPS_K):
                k_col_grp = fx.Int32(ks * (MFMA_DIM_K // MFMA_DIM_LANE_K)) + lane_div_16
                packs.append(
                    k_smem[local_k_row, k_swz_grp(local_k_row, k_col_grp), None].load()
                )
            return packs

        def compute_p_tile(kv_start, k_packs_by_ng, apply_mask=True):
            """GEMM1 with A = K, B = Q, so the C fragment (lane: query
            lane_mod_16, keys lane_div_16*4 + i) is GEMM2's A operand."""
            p_packs = [
                [None for _ in range_constexpr(Q_SUBTILES)]
                for _ in range_constexpr(KV_SUBTILES)
            ]
            for ng in range_constexpr(KV_SUBTILES):
                k_packs = [
                    Vec(k_packs_by_ng[ng][ks]) for ks in range_constexpr(K_STEPS_K)
                ]
                if const_expr(apply_mask):
                    kv_base = (
                        kv_start
                        + fx.Int32(ng * MFMA_M)
                        + lane_div_16 * fx.Int32(MFMA_LANE_K)
                    )
                    col_raw = [
                        kv_base + fx.Int32(i)
                        for i in range_constexpr(MFMA_ELEMS_PER_LANE)
                    ]
                    col_in_seq = [
                        col_raw[i] < seq_len
                        for i in range_constexpr(MFMA_ELEMS_PER_LANE)
                    ]
                    col_id = [
                        sm.to_id(col_raw[i])
                        for i in range_constexpr(MFMA_ELEMS_PER_LANE)
                    ]
                for qg in range_constexpr(Q_SUBTILES):
                    cur = c_zero_v4f32
                    for ks in range_constexpr(K_STEPS_K):
                        q_op = q_packs[ks][qg] if ks < K_STEPS else c_zero_dim_pack
                        cur = dim_mfma_acc(k_packs[ks].ir_value(), q_op, cur)
                    s_vals = [Vec(cur)[i] for i in range_constexpr(MFMA_ELEMS_PER_LANE)]
                    if const_expr(apply_mask):
                        keep = [
                            sm.visible(q_rows[qg], col_raw[i], q_row_ids[qg], col_id[i])
                            & col_in_seq[i]
                            for i in range_constexpr(MFMA_ELEMS_PER_LANE)
                        ]
                        s_vals = [
                            keep[i].select(s_vals[i], c_zero_f)
                            for i in range_constexpr(MFMA_ELEMS_PER_LANE)
                        ]
                    # A masked score of 0 gates to silu(0) = 0.
                    p_packs[ng][qg] = pack_p(silu_scale_batch(s_vals))
            return p_packs

        def accum_o_tile(o_acc, p_packs):
            """O[m, d] += P[m, n] V[n, d] with A = P, B = V[kv, d]."""

            def read_v_pack(c, ng):
                d_col = fx.Int32(c * MFMA_M) + lane_mod_16
                kv_grp = fx.Int32(ng * (MFMA_M // MFMA_LANE_K)) + lane_div_16
                return v_smem[d_col, kv_grp, None].load().ir_value()

            for c in range_constexpr(D_CHUNKS):
                v_packs = [read_v_pack(c, ng) for ng in range_constexpr(KV_SUBTILES)]
                for qg in range_constexpr(Q_SUBTILES):
                    acc_off = c * Q_SUBTILES + qg
                    cur = o_acc[acc_off]
                    for ng in range_constexpr(KV_SUBTILES):
                        cur = mfma_acc(p_packs[ng][qg], v_packs[ng], cur)
                    o_acc[acc_off] = cur
            return o_acc

        def run_kv_tile(o_acc, kv_start, apply_mask=True, full_tile=False):
            """K staged through registers to swizzled LDS; V's global load
            stays in flight under GEMM1 (counted vmcnt)."""
            k_vecs = async_load_k_regs(kv_start, full_tile=full_tile)
            v_vecs = async_load_v_regs(kv_start, full_tile=full_tile)
            rocdl.s_waitcnt(vmcnt=NUM_BATCHES_V)
            store_k_regs_to_lds(k_vecs)
            gpu.barrier()
            k_packs = [read_k_packs(ng) for ng in range_constexpr(KV_SUBTILES)]
            p_packs = compute_p_tile(kv_start, k_packs, apply_mask=apply_mask)
            rocdl.s_waitcnt(vmcnt=0)
            store_v_regs_to_lds(v_vecs)
            rocdl.sched_group_barrier(rocdl.mask_dswr, NUM_BATCHES_V, 0)
            gpu.barrier()
            # The next tile's K barrier fences these V reads from its V
            # store, and this tile's V barrier fenced the K reads above.
            o_acc = accum_o_tile(o_acc, p_packs)
            return o_acc

        # Unmasked tiles [main_start, unmasked_end): under a plain causal mask
        # every key below min(q_start, max_ids) is visible to every row of the
        # tile (targets clamp ids at max_ids). Computed outside the dynamic
        # `if` below: FlyDSL carries any local whose method a region calls.
        unmasked_end = main_start
        if const_expr(CAUSAL_SPLIT):
            split_pos = q_start
            if const_expr(HAS_TARGETS):
                split_pos = ops.min(q_start, sm.max_ids)
            unmasked_end = ops.max(split_pos // fx.Int32(BLOCK_N), main_start)
            unmasked_end = ops.min(unmasked_end, main_end)

        if active:
            acc_init = [c_zero_v4f32 for _ in range(N_ACC)]

            loop_results = acc_init
            # Contextual columns a window would skip.
            for kv_tile, it in range(
                fx.Int32(0),
                fx.Int32(prefix_end),
                fx.Int32(1),
                init=acc_init,
            ):  # ty: ignore
                it_list = list(it) if isinstance(it, (list, tuple)) else [it]
                o_acc = [it_list[i] for i in range(N_ACC)]
                kv_start = fx.Int32(kv_tile) * fx.Int32(BLOCK_N)
                o_acc = run_kv_tile(o_acc, kv_start, apply_mask=True)
                loop_results = yield o_acc

            for kv_tile, it in range(
                fx.Int32(main_start),
                fx.Int32(unmasked_end),
                fx.Int32(1),
                init=loop_results,
            ):  # ty: ignore
                it_list = list(it) if isinstance(it, (list, tuple)) else [it]
                o_acc = [it_list[i] for i in range(N_ACC)]
                kv_start = fx.Int32(kv_tile) * fx.Int32(BLOCK_N)
                o_acc = run_kv_tile(o_acc, kv_start, apply_mask=False, full_tile=True)
                loop_results = yield o_acc

            for kv_tile, it in range(
                fx.Int32(unmasked_end),
                fx.Int32(main_end),
                fx.Int32(1),
                init=loop_results,
            ):  # ty: ignore
                it_list = list(it) if isinstance(it, (list, tuple)) else [it]
                o_acc = [it_list[i] for i in range(N_ACC)]
                kv_start = fx.Int32(kv_tile) * fx.Int32(BLOCK_N)
                o_acc = run_kv_tile(o_acc, kv_start, apply_mask=True)
                loop_results = yield o_acc

            # ---- Epilogue. GEMM2's C fragment is [d, query]: d = lane_mod_16,
            # query = qg*16 + lane_div_16*4 + e. ----
            results = (
                list(loop_results)
                if isinstance(loop_results, (list, tuple))
                else [loop_results]
            )
            row_scale = out_scale
            if const_expr(SCALE_MODE == SCALE_SCALAR):
                row_scale = out_scale * fx.Float32(attn_scale[fx.Int32(0)])
            for qg in range_constexpr(Q_SUBTILES):
                q_row_base = (
                    q_wave_base
                    + fx.Int32(qg * MFMA_M)
                    + lane_div_16 * fx.Int32(MFMA_LANE_K)
                )
                for e in range_constexpr(MFMA_ELEMS_PER_LANE):
                    q_row_e = q_row_base + fx.Int32(e)
                    if q_row_e < seq_len:
                        scale_e = row_scale
                        if const_expr(SCALE_MODE == SCALE_TOKEN):
                            scale_e = out_scale * fx.Float32(
                                attn_scale[seq_start + q_row_e]
                            )
                        for c in range_constexpr(D_CHUNKS):
                            ov = results[c * Q_SUBTILES + qg]
                            d_col = fx.Int32(c * MFMA_M) + lane_mod_16
                            val = (Vec(ov)[e] * scale_e).to(elem_dtype)
                            out[fx.Int64(seq_start + q_row_e), head_idx, d_col] = val

    @flyc.jit
    def launch_hstu_ragged_fwd(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        seq_offsets: fx.Tensor,
        num_targets: fx.Tensor,
        attn_scale: fx.Tensor,
        out: fx.Tensor,
        num_seqs: fx.Int32,
        num_heads: fx.Int32,
        max_seq_len: fx.Int32,
        alpha_neg_log2e: fx.Float32,
        out_scale: fx.Float32,
        max_attn_len: fx.Int32,
        contextual_seq_len: fx.Int32,
        full_attn_size: fx.Int32,
        stream: fx.Stream,
    ) -> None:
        c_ngg = fx.Int32(NUM_GRID_GROUPS)
        num_q_tiles = (max_seq_len + fx.Int32(BLOCK_M - 1)) // fx.Int32(BLOCK_M)
        hz_total = num_seqs * num_heads
        hz_per_group = (hz_total + fx.Int32(NUM_GRID_GROUPS - 1)) // c_ngg
        grid_blocks = num_q_tiles * hz_per_group * c_ngg
        hstu_ragged_fwd_kernel(
            q,
            k,
            v,
            seq_offsets,
            num_targets,
            attn_scale,
            out,
            num_heads,
            num_q_tiles,
            hz_per_group,
            hz_total,
            alpha_neg_log2e,
            out_scale,
            max_attn_len,
            contextual_seq_len,
            full_attn_size,
            value_attrs={
                "rocdl.waves_per_eu": WAVES_PER_EU,
                "rocdl.flat_work_group_size": f"{BLOCK_THREADS},{BLOCK_THREADS}",
            },
        ).launch(grid=(grid_blocks, 1, 1), block=BLOCK_THREADS, smem=0, stream=stream)

    launch_hstu_ragged_fwd.compile_hints = {
        "fast_fp_math": True,
        "unsafe_fp_math": True,
    }
    return launch_hstu_ragged_fwd


def _bind_mfma_accs(fx, Vec, elem_dtype, *shapes):
    """One ``acc(a, b, c) -> c'`` per MFMA (depth, lane elements); equal
    shapes share an atom. Call inside a kernel body."""
    c_rmem = fx.make_rmem_tensor(MFMA_ELEMS_PER_LANE, fx.Float32)
    cache = {}
    accs = []
    for mfma_k, lane_k in shapes:
        if (mfma_k, lane_k) not in cache:
            atom = fx.make_mma_atom(fx.rocdl.MFMA(MFMA_M, MFMA_M, mfma_k, elem_dtype))
            a_rmem = fx.make_rmem_tensor(lane_k, elem_dtype)
            b_rmem = fx.make_rmem_tensor(lane_k, elem_dtype)

            def acc(a_pack, b_pack, c, atom=atom, a_rmem=a_rmem, b_rmem=b_rmem):
                a_rmem.store(Vec(a_pack))
                b_rmem.store(Vec(b_pack))
                c_rmem.store(Vec(c))
                fx.mma_atom_call(atom, c_rmem, a_rmem, b_rmem, c_rmem)
                return c_rmem.load().ir_value()

            cache[(mfma_k, lane_k)] = acc
        accs.append(cache[(mfma_k, lane_k)])
    return accs


def hstu_ragged_fwd(
    N: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    invalid_attn_mask_type: str,
    num_targets: Optional[torch.Tensor],
    attn_scale: Optional[torch.Tensor],
    max_attn_len: int,
    contextual_seq_len: int,
    full_attn_size: int,
    num_softmax_heads: int = 0,
    shared_kv: bool = False,
    *,
    config: Optional[FwdConfig] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``mslk::hstu_ragged_fwd`` on the FlyDSL kernel; returns ``(out, M)``."""
    from mslk.flydsl.jit import run_compiled

    if shared_kv:
        v = k
    num_tokens, num_heads, head_dim = q.shape
    hidden_dim = v.shape[2]
    arch = _arch()
    cfg = config or _first_supported(q, head_dim, hidden_dim, arch)
    reason = check_supported(q, head_dim, hidden_dim, cfg, arch)
    if reason is not None:
        raise NotImplementedError(f"FlyDSL ragged HSTU forward: {reason}")
    softmax = num_softmax_heads > 0
    spec = MaskSpec.from_args(
        invalid_attn_mask_type,
        max_attn_len,
        contextual_seq_len,
        full_attn_size,
        has_targets=num_targets is not None,
    )
    if attn_scale is None:
        scale_mode = SCALE_INV_N
    elif attn_scale.ndim == 0:
        scale_mode = SCALE_SCALAR
    else:
        scale_mode = SCALE_TOKEN

    out = torch.empty(v.shape, dtype=q.dtype, device=q.device)
    M = torch.empty((num_tokens, num_heads), dtype=torch.float32, device=q.device)
    num_seqs = seq_offsets.shape[0] - 1
    if num_tokens == 0 or num_seqs == 0 or N == 0:
        out.zero_()
        M.fill_(float("-inf"))
        return out, M

    launcher = build_ragged_fwd(
        head_dim,
        hidden_dim,
        _DTYPES[q.dtype],
        spec,
        scale_mode,
        softmax,
        cfg,
        arch,
    )
    seq_offsets = seq_offsets.to(torch.int32)
    if num_targets is None:
        num_targets = seq_offsets
    else:
        num_targets = num_targets.to(torch.int32)
    if attn_scale is None:
        attn_scale = torch.empty(1, dtype=torch.float32, device=q.device)
    else:
        attn_scale = attn_scale.reshape(-1).to(torch.float32)
    out_scale = -1.0 / _LOG2E
    if scale_mode == SCALE_INV_N:
        out_scale /= N
    run_compiled(
        launcher,
        q,
        k,
        v,
        seq_offsets,
        num_targets,
        attn_scale,
        out,
        num_seqs,
        num_heads,
        N,
        alpha * -_LOG2E,
        out_scale,
        max_attn_len,
        contextual_seq_len,
        full_attn_size,
        torch.cuda.current_stream(q.device),
    )
    return out, M
