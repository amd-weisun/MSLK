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
# Grouped calls only: one scale per group.
SCALE_GROUP = "group"

_DTYPES = {torch.bfloat16: "bf16", torch.float16: "f16"}

# ``kernel="auto"`` takes the pipelined kernel for lower-triangular masks when
# the max sequence length is above _PIPELINED_MIN_N (_PIPELINED_MIN_N_SILU for
# SiLU heads) and any window is longer than _PIPELINED_MIN_WINDOW, and the
# tiled kernel otherwise. MI350 bf16, B 256-512 at 0.75 sparsity (pipelined /
# tiled time): causal softmax 0.84-0.99 at N 4096 and 0.68-0.89 from N 8192;
# causal SiLU 0.98-1.07 at N 4096 and 0.81-0.95 from N 8192; upper-
# triangular 1.4-1.7 (the tiled kernel skips tiles on any mask); the tiled
# kernel is faster at N <= 2048 and on windows up to about 2k. D 256 is tiled
# only.
_PIPELINED_MIN_N = 2048
_PIPELINED_MIN_N_SILU = 4096
_PIPELINED_MIN_WINDOW = 2048


@dataclass(frozen=True)
class FwdConfig:
    """Tile configuration."""

    block_m: int = 128
    block_n: int = 32
    num_waves: int = 4
    waves_per_eu: int = 2
    # gfx950 DMA path: 2 double-buffers K, so the next tile's K is issued a
    # whole tile ahead (V stays single-buffered).
    k_stages: int = 1


# Windows up to this many keys take 32-key tiles on gfx950: a query tile then
# re-reads fewer K/V rows around its window, which is what bounds such calls
# (memory-bound: a 16-key window at D 128 runs at about 35 TFLOPS).
_SMALL_WINDOW = 64


def default_config(
    head_dim: int,
    hidden_dim: int,
    arch: str = "",
    max_attn_len: int = 0,
) -> FwdConfig:
    """Default tiles. gfx950 (K/V by DMA): from an MI350 sweep. Elsewhere
    AITER's MI300X heuristic."""
    dim = max(hidden_dim, ((head_dim + 63) // 64) * 64)
    if arch.startswith("gfx950"):
        # K double-buffered throughout: the next tile's K is issued a whole
        # tile ahead, and two K buffers plus V still fit three blocks per CU
        # (48 KB at 32-key tiles, D 256). MI350, B512 H4: D 256 N 512 0.560
        # -> 0.525 ms, N 4096 16.7 -> 15.7 ms; D 128 N 2048 2.31 -> 2.27 ms;
        # 16-key softmax window, D 128: 3.85 -> 3.67 ms.
        # Small windows: 32-key tiles (a query tile re-reads fewer K/V rows).
        if 0 < max_attn_len <= _SMALL_WINDOW:
            return FwdConfig(64, 32, 4, 0, k_stages=2)
        # D 128 and up: 32-key tiles keep D 256 at three waves per SIMD and
        # are 2-8% faster than 64-key tiles at every length.
        if dim >= 128:
            return FwdConfig(64, 32, 4, 3, k_stages=2)
        return FwdConfig(64, 64, 4, 3, k_stages=2)
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
    q: torch.Tensor,
    head_dim: int,
    hidden_dim: int,
    arch: str,
    max_attn_len: int = 0,
) -> FwdConfig:
    cfg = default_config(head_dim, hidden_dim, arch, max_attn_len)
    if check_supported(q, head_dim, hidden_dim, cfg, arch) is None:
        return cfg
    for cfg in _FALLBACK_CONFIGS:
        if check_supported(q, head_dim, hidden_dim, cfg, arch) is None:
            return cfg
    return default_config(head_dim, hidden_dim, arch, max_attn_len)


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
    lds = (
        cfg.block_n * head_dim_k * 2 * max(1, cfg.k_stages)
        + hidden_dim * (cfg.block_n + 8) * 2
    )
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
    reverse_q_tiles: bool = False,
    single_head: bool = False,
    grouped: bool = False,
):
    """Compile-time specialization of the kernel; returns its host launcher.

    ``single_head`` (H == 1) keys a launcher of its own: FlyDSL fixes which
    dim of a tensor has unit stride at compile time, and M ([T, H]) has
    strides (1, 1) at H = 1 but (H, 1) otherwise. A launcher keeps its first
    compiled binary (``run_compiled``), so sharing one across the two
    layouts would write M through the wrong one."""
    import flydsl.compiler as flyc  # pyre-ignore[21]
    import flydsl.expr as fx  # pyre-ignore[21]
    from flydsl.expr import const_expr, gpu, range_constexpr, rocdl  # pyre-ignore[21]
    from flydsl.expr.typing import (  # pyre-ignore[21]  # pyre-ignore[21]
        T,
        Vector as Vec,
    )
    from mslk.attention.flydsl.flash_attn_utils import (
        _make_rebased_view,
        as_mlir_value,
        scf_if_dispatch,
    )
    from mslk.flydsl import buffer_ops

    from .common import device_ops, seq_mask

    # FlyDSL keys its compile cache on the kernel's *scalar* closure values
    # (and types) only: a dataclass captured by the kernel is left out of the
    # key, so two variants would share one cached binary. Unpack every
    # compile-time parameter into a scalar before the kernel captures it.
    BLOCK_M, BLOCK_N = cfg.block_m, cfg.block_n
    NUM_WAVES = cfg.num_waves
    WAVES_PER_EU = cfg.waves_per_eu
    K_STAGES = cfg.k_stages
    MASK_TYPE = spec.mask_type
    HAS_WINDOW = spec.has_window
    HAS_CONTEXTUAL = spec.has_contextual
    HAS_TARGETS = spec.has_targets
    HAS_FULL_ATTN = spec.has_full_attn
    SCALE_MODE = scale_mode
    SOFTMAX = softmax
    # Lazy-rescale threshold (log2 domain): p stays below 2^8.
    LAZY_RESCALE = 8.0
    REVERSE_Q_TILES = reverse_q_tiles
    # Grouped calls (T11): programs map to (sequence, query tile) through
    # group_meta, and each program loads its group's settings.
    GROUPED = grouped
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
    # A 16-lane ds_read_b128 phase reads 16 K rows; with K rows a multiple of
    # 256 B apart, XOR-ing the 16 B column group with row % 16 (when there
    # are 16+ groups) puts them in 16 distinct bank slots, where row % 8
    # leaves a 2-way conflict.
    # The XOR only stays inside the row when the chunk count is a multiple
    # of the swizzle span.
    if arch.startswith("gfx950") and (HEAD_DIM_K // 8) % 16 == 0:
        K_SWZ_ROWS = 16
    K_COL_GUARD = head_dim < HEAD_DIM_K
    K_STEPS_K = HEAD_DIM_K // MFMA_DIM_K
    D_CHUNKS = hidden_dim // MFMA_M
    # gfx950 keeps V in its natural [kv, d] layout in LDS (one vector store
    # per lane) and reads GEMM2's B operand with the transposing
    # ds_read_b64_tr_b16. Elsewhere V is scattered transposed ([d, kv]).
    V_TR = arch.startswith("gfx950")
    # Transposed layout: +8 pads the [d, kv] row against bank conflicts.
    V_T_STRIDE = BLOCK_N + 8
    # Natural layout row padding: a 16-lane ds_read_b64_tr phase reads 8
    # rows x 32 B, conflict-free when the row stride is an odd multiple of
    # 32 B modulo 256 B; 16 elements of padding gives that for every D that
    # is a multiple of 128 (64 B multiples otherwise still halve conflicts).
    V_ROW_STRIDE = hidden_dim + 16
    # P*V on gfx950's full-rate 16x16x32 MFMA, pairing 16-key subtiles.
    PV_K32 = V_TR and (BLOCK_N // MFMA_N) % 2 == 0
    # Scheduling fences, gfx950 DMA path: the compiler otherwise issues the
    # LDS reads of later QK subtiles and P*V column chunks early, keeping
    # their operands live. With them, D 256 at 64x64x4 waves fits 166 VGPRs
    # (three waves per SIMD; 214 without) and runs 0.30 ms instead of 0.32.
    QK_FENCE = V_TR
    PV_FENCE_CHUNKS = 8 if V_TR else 0

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

    # Natural V rows, covering load batches that run past BLOCK_N (their
    # rows hold zeros and are never read).
    V_ROWS = max(BLOCK_N, ROWS_PER_BATCH_V * NUM_BATCHES_V)
    # gfx950: K and V tiles go global -> LDS by DMA (buffer_load ... lds, 16 B
    # per lane, one contiguous KB per wave instruction), so no registers stage
    # them. Tiles are unpadded; bank conflicts are avoided by XOR swizzles of
    # 16 B chunks applied through the source addresses instead.
    DMA_INSTR_BYTES = WARP_SIZE * 16
    K_DMA_INSTR = BLOCK_N * HEAD_DIM_K * 2 // DMA_INSTR_BYTES
    V_DMA_INSTR = BLOCK_N * hidden_dim * 2 // DMA_INSTR_BYTES
    KV_DMA = (
        V_TR
        and hidden_dim % 8 == 0
        and (BLOCK_N * HEAD_DIM_K * 2) % (DMA_INSTR_BYTES * NUM_WAVES) == 0
        and (BLOCK_N * hidden_dim * 2) % (DMA_INSTR_BYTES * NUM_WAVES) == 0
    )
    # V: chunk ^ 2 * (row % V_SWZ_ROWS) keeps each 32 B segment a
    # ds_read_b64_tr_b16 phase reads contiguous and puts 8 rows in distinct
    # bank slots when a row has 16+ chunks.
    # The XOR (up to 2 * V_SWZ_ROWS - 1) must stay inside the row: 2 *
    # V_SWZ_ROWS divides the row's chunk count.
    V_SWZ_ROWS = 8
    while V_SWZ_ROWS > 1 and (hidden_dim // 8) % (2 * V_SWZ_ROWS):
        V_SWZ_ROWS //= 2
    if KV_DMA:
        V_ROW_STRIDE = hidden_dim
        V_ROWS = BLOCK_N
    V_LDS_ELEMS = V_ROWS * V_ROW_STRIDE if V_TR else hidden_dim * V_T_STRIDE
    # K double-buffer (gfx950 DMA path): stage t % 2 holds tile t's K.
    KDB = KV_DMA and K_STAGES == 2
    K_LDS_STAGES = 2 if KDB else 1
    K_STAGE_ELEMS = BLOCK_N * HEAD_DIM_K
    # K DMA instructions per wave per tile: the vmcnt that leaves K(t + 1)
    # in flight while waiting for V(t).
    K_VM_PER_TILE = K_DMA_INSTR // NUM_WAVES

    @fx.struct
    class SharedStorage:
        k: fx.Array[elem_dtype, K_LDS_STAGES * K_STAGE_ELEMS, 16]
        v: fx.Array[elem_dtype, V_LDS_ELEMS, 16]

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def hstu_ragged_fwd_kernel(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        seq_offsets: fx.Tensor,
        num_targets: fx.Tensor,
        attn_scale: fx.Tensor,
        out: fx.Tensor,
        lse_out: fx.Tensor,
        group_meta: fx.Tensor,
        num_heads: fx.Int32,
        num_q_tiles: fx.Int32,
        hz_per_group: fx.Int32,
        hz_total: fx.Int32,
        score_scale: fx.Float32,
        out_scale: fx.Float32,
        max_attn_len: fx.Int32,
        contextual_seq_len: fx.Int32,
        full_attn_size: fx.Int32,
        k_tok_stride: fx.Int32,
        k_head_stride: fx.Int32,
        v_tok_stride: fx.Int32,
        v_head_stride: fx.Int32,
        group_size: fx.Int32,
        meta_prefix_base: fx.Int32,
        meta_settings_base: fx.Int32,
    ) -> None:
        ops = device_ops()
        compute_type = fx.Float32.ir_type
        c_zero_dim_pack = Vec.filled(MFMA_DIM_LANE_K, 0.0, elem_dtype).ir_value()

        dim_mfma_acc, mfma_acc, mfma_acc_k32 = _bind_mfma_accs(
            fx,
            Vec,
            elem_dtype,
            (MFMA_DIM_K, MFMA_DIM_LANE_K),
            (MFMA_K, MFMA_LANE_K),
            (2 * MFMA_K, 2 * MFMA_LANE_K),
        )

        tid = fx.Int32(gpu.thread_idx.x)
        wave_id = tid // fx.Int32(WARP_SIZE)
        lane = tid % fx.Int32(WARP_SIZE)
        lane_mod_16 = lane % fx.Int32(MFMA_N)
        lane_div_16 = lane // fx.Int32(MFMA_N)

        block_id = fx.Int32(gpu.block_idx.x)
        group_idx = fx.Int32(0)
        group_max_len = fx.Int32(1)
        if const_expr(GROUPED):
            # ---- Grouped decode: group_meta holds, from 0, each program's
            # sequence (num_seqs past the last tile); from meta_prefix_base,
            # the exclusive prefix sum of query tiles per sequence; from
            # meta_settings_base, per group: max_seq_len, max_attn_len,
            # full_attn_size, contextual_seq_len. ----
            # Programs run as (head, tile slot), slots fastest, so a
            # sequence's tiles are consecutive; each XCD (block_id % 8)
            # takes a contiguous share, keeping a sequence's K/V in one L2.
            n_progs = meta_prefix_base * num_heads
            per_xcd = (n_progs + fx.Int32(NUM_GRID_GROUPS - 1)) // fx.Int32(
                NUM_GRID_GROUPS
            )
            order = (block_id % fx.Int32(NUM_GRID_GROUPS)) * per_xcd + (
                block_id // fx.Int32(NUM_GRID_GROUPS)
            )
            in_range = order < n_progs
            order = in_range.select(order, fx.Int32(0))
            head_idx = order // meta_prefix_base
            prog = order % meta_prefix_base
            seq = fx.Int32(group_meta[prog])
            block_valid = in_range & (seq < hz_total)
            batch_idx = block_valid.select(seq, fx.Int32(0))
            tile0 = fx.Int32(group_meta[meta_prefix_base + batch_idx])
            q_tile_idx = prog - tile0
            if const_expr(REVERSE_Q_TILES):
                n_tiles = (
                    fx.Int32(group_meta[meta_prefix_base + batch_idx + fx.Int32(1)])
                    - tile0
                )
                q_tile_idx = n_tiles - fx.Int32(1) - q_tile_idx
            # Surplus slots decode against sequence 0 with a meaningless
            # (possibly negative) tile; tile 0 with seq_len 0 keeps them
            # inactive.
            q_tile_idx = block_valid.select(q_tile_idx, fx.Int32(0))
            group_idx = batch_idx // group_size
            settings = meta_settings_base + group_idx * fx.Int32(4)
            group_max_len = fx.Int32(group_meta[settings])
            max_attn_len = fx.Int32(group_meta[settings + fx.Int32(1)])
            full_attn_size = fx.Int32(group_meta[settings + fx.Int32(2)])
            contextual_seq_len = fx.Int32(group_meta[settings + fx.Int32(3)])
        else:
            # ---- Group-major grid decode -> (sequence, head, query tile) ----
            grid_group = block_id % fx.Int32(NUM_GRID_GROUPS)
            pos_in_group = block_id // fx.Int32(NUM_GRID_GROUPS)
            local_hz_idx = pos_in_group // num_q_tiles
            q_tile_idx = pos_in_group % num_q_tiles
            if const_expr(REVERSE_Q_TILES):
                # Heaviest causal query tiles first.
                q_tile_idx = num_q_tiles - fx.Int32(1) - q_tile_idx
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
                (
                    K_LDS_STAGES * BLOCK_N,
                    HEAD_DIM_K // MFMA_DIM_LANE_K,
                    MFMA_DIM_LANE_K,
                ),
                (HEAD_DIM_K, MFMA_DIM_LANE_K, 1),
            )
        )
        if const_expr(V_TR):
            v_smem = lds.v.view(
                fx.make_layout(
                    (V_ROWS, V_ROW_STRIDE // VEC_V, VEC_V),
                    (V_ROW_STRIDE, VEC_V, 1),
                )
            )
            v_lds_base = fx.Int64(fx.ptrtoint(lds.v.ptr))
            # ds_read_b64_tr_b16, per group of 16 lanes: lane i addresses row
            # i // 4 and columns (i % 4) * 4 .. +3 of a 4 x 16 block and gets
            # column i's four rows back, which is the 16x16x16 MFMA's B
            # operand (lane: d = lane % 16, kv = (lane // 16) * 4 .. +3).
            v_tr_row = lane_div_16 * fx.Int32(MFMA_LANE_K) + lane_mod_16 // fx.Int32(4)
            v_tr_lane_off = v_tr_row * fx.Int32(V_ROW_STRIDE) + (
                lane_mod_16 % fx.Int32(4)
            ) * fx.Int32(4)
            v_tr_type = Vec.make_type(MFMA_LANE_K, elem_dtype)
            if const_expr(KV_DMA):
                # Swizzled chunk of column c*16 + (lane % 4)*4: chunk 2c + j,
                # j = (lane % 4) // 2, XOR 2 * (row % V_SWZ_ROWS); the row's low
                # bits are lane-constant (subtiles start at multiples of 16).
                # (2c + j) ^ 2r = 2 (c ^ r) + j, and with c = c_hi * R + c_lo
                # (R = V_SWZ_ROWS, r < R): c ^ r = c_hi * R + (c_lo ^ r). So a
                # read is one of R lane-dependent pointers (one per c_lo) plus
                # a constant (c_hi, and the 16-row subtile ng, which leaves
                # row % R unchanged) that folds into the DS offset field:
                # R address registers live instead of one per read.
                v_tr_r = v_tr_row % fx.Int32(V_SWZ_ROWS)
                v_tr_lane_byte = v_lds_base + fx.Int64(
                    (
                        v_tr_row * fx.Int32(V_ROW_STRIDE)
                        + ((lane_mod_16 % fx.Int32(4)) // fx.Int32(2)) * fx.Int32(8)
                        + (lane_mod_16 % fx.Int32(2)) * fx.Int32(4)
                    )
                    * fx.Int32(2)
                )
                v_tr_ptrs = [
                    buffer_ops.create_llvm_ptr(
                        v_tr_lane_byte
                        + fx.Int64(
                            ((fx.Int32(c_lo) ^ v_tr_r) * fx.Int32(16)) * fx.Int32(2)
                        ),
                        address_space=3,
                    )
                    for c_lo in range(V_SWZ_ROWS)
                ]
        else:
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
            # SiLU heads: score_scale = alpha * -log2e.
            g = [s * score_scale for s in s_list]
            emu = [fx.exp2(gi, fastmath="afn") for gi in g]
            sig = [_rcp(c_one_f + e) for e in emu]
            return [g[i] * sig[i] for i in range(len(s_list))]

        def pack_p(vals, round_nearest=False):
            """Pack 4 f32 into a bf16/f16 GEMM2 A-operand fragment. bf16
            truncates unless round_nearest: cheap, and its bias stays inside
            tolerance for SiLU weights, but not for softmax probabilities."""
            if const_expr(is_bf16 and not round_nearest):
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

        # ---- Online softmax (softmax heads; score_scale = alpha * log2e) ----
        # Scores are base 2. Each lane holds one query (lane_mod_16) and four
        # keys per 16-key subtile; the four lanes sharing a query differ in
        # lane bits 4 and 5. The row max m is kept reduced across them; the
        # row sum l is a per-lane partial, reduced once in the epilogue.
        # Masked scores use a finite sentinel and are zeroed after exp2: the
        # kernel compiles with fast math, which assumes no infinities.
        c_neg_big = fx.Float32(-1.0e30)

        def fmax(a, b):
            return (a > b).select(a, b)

        def lanes_max(x):
            x = fmax(x, x.shuffle_xor(16, WARP_SIZE))
            return fmax(x, x.shuffle_xor(32, WARP_SIZE))

        def lanes_sum(x):
            x = x + x.shuffle_xor(16, WARP_SIZE)
            return x + x.shuffle_xor(32, WARP_SIZE)

        def acc_layout(x, e):
            # GEMM2's C fragment holds query lane_div_16*4 + e on this lane;
            # read that query's value from the lane holding it in GEMM1 layout.
            src = lane_div_16 * fx.Int32(MFMA_LANE_K) + fx.Int32(e)
            return fx.Float32(gpu.shuffle_idx(x, src, WARP_SIZE))

        def scale_o_rows(o_acc, row_vals):
            # o_acc[c*Q + qg] *= row_vals[qg] per query, in GEMM2 layout.
            for qg in range_constexpr(Q_SUBTILES):
                per_e = [
                    acc_layout(row_vals[qg], e)
                    for e in range_constexpr(MFMA_ELEMS_PER_LANE)
                ]
                for c in range_constexpr(D_CHUNKS):
                    ov = Vec(o_acc[c * Q_SUBTILES + qg])
                    o_acc[c * Q_SUBTILES + qg] = Vec.from_elements(
                        [
                            ov[e] * per_e[e]
                            for e in range_constexpr(MFMA_ELEMS_PER_LANE)
                        ],
                        fx.Float32,
                    ).ir_value()
            return o_acc

        def lazy_scale_o_rows(o_acc, corr_rescale):
            # Rescale O only on tiles where the wave's reference max moved.
            corr, rescale = corr_rescale

            def _then(_names, *state):
                return scale_o_rows(list(state), corr)

            def _else(_names, *state):
                return list(state)

            return list(
                scf_if_dispatch(
                    rescale,
                    _then,
                    _else,
                    state_names=tuple("_ro%d" % i for i in range(len(o_acc))),
                    state_values=list(o_acc),
                )
            )

        # ---- KV tiles to visit: [0, prefix_end) then [main_start, main_end) ----
        q_start = q_tile_idx * fx.Int32(BLOCK_M)
        prefix_end, main_start, main_end = sm.kv_tiles(q_start, BLOCK_M, BLOCK_N)
        active = q_start < seq_len

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
            """gfx950: one vector store per lane into natural [kv, d] rows.
            Otherwise V scattered transposed to [d, kv] so GEMM2 reads 4 kv
            per ds_read_b64."""
            if const_expr(V_TR):
                for b in range_constexpr(NUM_BATCHES_V):
                    kv_row = v_load_row_in_batch + fx.Int32(b * ROWS_PER_BATCH_V)
                    v_smem[kv_row, v_load_lane_in_row, None].store(Vec(vecs[b]))
                return
            for b in range_constexpr(NUM_BATCHES_V):
                kv_row = v_load_row_in_batch + fx.Int32(b * ROWS_PER_BATCH_V)
                kv_grp = kv_row // fx.Int32(MFMA_LANE_K)
                kv_lane = kv_row % fx.Int32(MFMA_LANE_K)
                vv = Vec(vecs[b])
                for j in range_constexpr(VEC_V):
                    v_smem[v_load_col + fx.Int32(j), kv_grp, kv_lane] = vv[j]

        def read_k_packs(ng, stage):
            local_k_row = fx.Int32(ng * MFMA_M) + lane_mod_16
            if const_expr(KDB):
                # Stages are BLOCK_N rows apart, a multiple of the swizzle
                # span, so the swizzle is unchanged.
                local_k_row = local_k_row + stage * fx.Int32(BLOCK_N)
            packs = []
            for ks in range_constexpr(K_STEPS_K):
                k_col_grp = fx.Int32(ks * (MFMA_DIM_K // MFMA_DIM_LANE_K)) + lane_div_16
                packs.append(
                    k_smem[local_k_row, k_swz_grp(local_k_row, k_col_grp), None].load()
                )
            return packs

        def tile_scores(stage):
            """GEMM1 with A = K, B = Q, so the C fragment (lane: query
            lane_mod_16, keys lane_div_16*4 + i) is GEMM2's A operand.
            Returns the scores flat, index (ng * Q_SUBTILES + qg) * 4 + i."""
            flat = []
            for ng in range_constexpr(KV_SUBTILES):
                # Read this subtile's K packs just before its MFMAs, so only
                # one subtile's packs are live (D 256: 32 VGPRs, not 128).
                k_packs = [Vec(x) for x in read_k_packs(ng, stage)]
                for qg in range_constexpr(Q_SUBTILES):
                    cur = c_zero_v4f32
                    for ks in range_constexpr(K_STEPS_K):
                        q_op = q_packs[ks][qg] if ks < K_STEPS else c_zero_dim_pack
                        cur = dim_mfma_acc(k_packs[ks].ir_value(), q_op, cur)
                    for i in range_constexpr(MFMA_ELEMS_PER_LANE):
                        flat.append(Vec(cur)[i])
                if const_expr(QK_FENCE):
                    rocdl.sched_barrier(0)
            return flat

        # Masked pre-activation value: SiLU scores gate to silu(0) = 0;
        # softmax's scaled scores take a finite sentinel (fast math assumes
        # no infinities) and their p is zeroed below.
        masked_value = c_neg_big if SOFTMAX else c_zero_f

        def mask_values(kv_start, values):
            out = list(values)
            for ng in range_constexpr(KV_SUBTILES):
                kv_base = (
                    kv_start
                    + fx.Int32(ng * MFMA_M)
                    + lane_div_16 * fx.Int32(MFMA_LANE_K)
                )
                for i in range_constexpr(MFMA_ELEMS_PER_LANE):
                    col = kv_base + fx.Int32(i)
                    col_id = sm.to_id(col)
                    for qg in range_constexpr(Q_SUBTILES):
                        keep = sm.visible(q_rows[qg], col, q_row_ids[qg], col_id) & (
                            col < seq_len
                        )
                        idx = (ng * Q_SUBTILES + qg) * MFMA_ELEMS_PER_LANE + i
                        out[idx] = keep.select(fx.Float32(out[idx]), masked_value)
            return [fx.Float32(x).ir_value() for x in out]

        block_row_last = q_start + fx.Int32(BLOCK_M - 1)

        def tile_needs_mask(kv_start):
            """False only when every key of the tile is visible to every row
            of the block (uniform across the block)."""
            return sm.tile_needs_mask(
                q_start, block_row_last, kv_start, kv_start + fx.Int32(BLOCK_N - 1)
            )

        N_SCORES = KV_SUBTILES * Q_SUBTILES * MFMA_ELEMS_PER_LANE
        # A run-time contextual length > 0 makes tile_needs_mask true, so a
        # contextual spec can skip only in grouped calls (where a group's
        # length may be 0); a ragged one has contextual only when > 0.
        CAN_SKIP_MASK = MASK_TYPE == LOWER and (GROUPED or not HAS_CONTEXTUAL)

        def masked_values(kv_start, values):
            if const_expr(not CAN_SKIP_MASK):
                return mask_values(kv_start, values)

            def _apply(_names, *vals):
                return mask_values(kv_start, vals)

            return list(
                scf_if_dispatch(
                    tile_needs_mask(kv_start),
                    _apply,
                    state_names=tuple("_hm%d" % i for i in range(N_SCORES)),
                    state_values=[fx.Float32(x).ir_value() for x in values],
                )
            )

        def compute_p_tile(kv_start, m_old, l_old, stage=None):
            """Scores, mask, activation. Returns P packs and, for softmax heads,
            the new row max, row-sum partials and the output rescale factor."""
            flat = tile_scores(stage)
            if const_expr(SOFTMAX):
                flat = [fx.Float32(x) * score_scale for x in flat]
            flat = [fx.Float32(x) for x in masked_values(kv_start, flat)]

            def vals(ng, qg):
                base = (ng * Q_SUBTILES + qg) * MFMA_ELEMS_PER_LANE
                return flat[base : base + MFMA_ELEMS_PER_LANE]

            p_packs = [
                [None for _ in range_constexpr(Q_SUBTILES)]
                for _ in range_constexpr(KV_SUBTILES)
            ]
            if const_expr(not SOFTMAX):
                for ng in range_constexpr(KV_SUBTILES):
                    for qg in range_constexpr(Q_SUBTILES):
                        p_packs[ng][qg] = pack_p(silu_scale_batch(vals(ng, qg)))
                return p_packs, m_old, l_old, None

            # Lazy rescale (as in MSLK's dual-wave flash attention): the
            # running max is a reference that moves only when some row of
            # the wave sees a tile max more than LAZY_RESCALE above it, so the
            # output accumulator is rescaled on few tiles (a wave-uniform
            # branch in the caller). p then reaches at most 2^LAZY_RESCALE.
            # A row with no valid key yet uses reference 0, so masked scores
            # (the finite sentinel) give p = 0 without a per-score check.
            c_live = fx.Float32(-1.0e29)
            c_lazy = fx.Float32(LAZY_RESCALE)
            tile_maxes = []
            need_any = None
            for qg in range_constexpr(Q_SUBTILES):
                tile_max = c_neg_big
                for ng in range_constexpr(KV_SUBTILES):
                    for x in vals(ng, qg):
                        tile_max = fmax(tile_max, x)
                tile_max = lanes_max(tile_max)
                tile_maxes.append(tile_max)
                need = tile_max > fx.Float32(m_old[qg]) + c_lazy
                need_any = need if need_any is None else (need_any | need)
            ballot = rocdl.ballot(T.i64, as_mlir_value(need_any))
            rescale = fx.Int64(ballot) != fx.Int64(0)
            m_new, l_new, corr = [], [], []
            for qg in range_constexpr(Q_SUBTILES):
                m_o = fx.Float32(m_old[qg])
                m_q = rescale.select(fmax(m_o, tile_maxes[qg]), m_o)
                corr_q = fx.exp2(m_o - m_q, fastmath="afn")
                m_ref = (m_q > c_live).select(m_q, c_zero_f)
                l_q = fx.Float32(l_old[qg]) * corr_q
                for ng in range_constexpr(KV_SUBTILES):
                    p_vals = []
                    for x in vals(ng, qg):
                        p = fx.exp2(x - m_ref, fastmath="afn")
                        l_q = l_q + p
                        p_vals.append(p)
                    p_packs[ng][qg] = pack_p(p_vals, round_nearest=True)
                m_new.append(m_q.ir_value())
                l_new.append(l_q.ir_value())
                corr.append(corr_q)
            return p_packs, m_new, l_new, (corr, rescale)

        def accum_o_tile(o_acc, p_packs):
            """O[m, d] += P[m, n] V[n, d] with A = P, B = V[kv, d]."""

            def read_v_pack(c, ng):
                if const_expr(KV_DMA):
                    c_hi, c_lo = divmod(c, V_SWZ_ROWS)
                    ptr = buffer_ops.get_element_ptr(
                        v_tr_ptrs[c_lo],
                        static_byte_offset=(
                            c_hi * V_SWZ_ROWS * 16 + ng * MFMA_M * V_ROW_STRIDE
                        )
                        * 2,
                    )
                    return rocdl.ds_read_tr16_b64(v_tr_type, ptr).result
                if const_expr(V_TR):
                    elem = v_tr_lane_off + fx.Int32(
                        ng * MFMA_M * V_ROW_STRIDE + c * MFMA_M
                    )
                    ptr = buffer_ops.create_llvm_ptr(
                        v_lds_base + fx.Int64(elem) * fx.Int64(2), address_space=3
                    )
                    return rocdl.ds_read_tr16_b64(v_tr_type, ptr).result
                d_col = fx.Int32(c * MFMA_M) + lane_mod_16
                kv_grp = fx.Int32(ng * (MFMA_M // MFMA_LANE_K)) + lane_div_16
                return v_smem[d_col, kv_grp, None].load().ir_value()

            if const_expr(PV_K32):
                # 16x16x32 MFMAs over pairs of 16-key subtiles: A is the two
                # P fragments back to back (keys g*4.. and 16 + g*4.. for lane
                # group g), B the V rows of the same keys from two transposing
                # reads. The K order only has to agree between A and B.
                for c in range_constexpr(D_CHUNKS):
                    v_pairs = []
                    for j in range_constexpr(KV_SUBTILES // 2):
                        lo = read_v_pack(c, 2 * j)
                        hi = read_v_pack(c, 2 * j + 1)
                        v_pairs.append(
                            Vec(lo)
                            .shuffle(Vec(hi), list(range(2 * MFMA_LANE_K)))
                            .ir_value()
                        )
                    for qg in range_constexpr(Q_SUBTILES):
                        acc_off = c * Q_SUBTILES + qg
                        cur = o_acc[acc_off]
                        for j in range_constexpr(KV_SUBTILES // 2):
                            a = Vec(p_packs[2 * j][qg]).shuffle(
                                Vec(p_packs[2 * j + 1][qg]),
                                list(range(2 * MFMA_LANE_K)),
                            )
                            cur = mfma_acc_k32(a.ir_value(), v_pairs[j], cur)
                        o_acc[acc_off] = cur
                    if const_expr(PV_FENCE_CHUNKS and (c + 1) % PV_FENCE_CHUNKS == 0):
                        rocdl.sched_barrier(0)
                return o_acc

            for c in range_constexpr(D_CHUNKS):
                v_packs = [read_v_pack(c, ng) for ng in range_constexpr(KV_SUBTILES)]
                for qg in range_constexpr(Q_SUBTILES):
                    acc_off = c * Q_SUBTILES + qg
                    cur = o_acc[acc_off]
                    for ng in range_constexpr(KV_SUBTILES):
                        cur = mfma_acc(p_packs[ng][qg], v_packs[ng], cur)
                    o_acc[acc_off] = cur
            return o_acc

        N_ACC = D_CHUNKS * Q_SUBTILES
        # Accumulator state: O, then the softmax row max and sum partials.
        N_STATE = N_ACC + (2 * Q_SUBTILES if SOFTMAX else 0)

        def split_state(it):
            it_list = list(it) if isinstance(it, (list, tuple)) else [it]
            o_acc = [it_list[i] for i in range(N_ACC)]
            m_old = [it_list[N_ACC + qg] for qg in range(Q_SUBTILES)] if SOFTMAX else []
            l_old = (
                [it_list[N_ACC + Q_SUBTILES + qg] for qg in range(Q_SUBTILES)]
                if SOFTMAX
                else []
            )
            return o_acc, m_old, l_old

        # Logical KV tiles: the contextual prefix [0, prefix_end), then the
        # main range from main_start.
        n_logical = prefix_end + (main_end - main_start)
        tile_gap = main_start - prefix_end

        def tile_start(t):
            return (t < prefix_end).select(t, t + tile_gap) * fx.Int32(BLOCK_N)

        if const_expr(KV_DMA):
            dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS(128), 128)
            lds_ptr_ty = fx.PointerType.get(elem_dtype.ir_type, 2, 16)
            wave_uni = fx.Int32(rocdl.readfirstlane(T.i32, as_mlir_value(wave_id)))

            buf_flags = fx.Int32(buffer_ops._get_buffer_flags())

            def rebased_div(t, byte_off, n_elems):
                # Fold the (64-bit) sequence/head base into the descriptor so
                # per-lane offsets stay 32-bit, with num_records bounded to the
                # sequence: rows past it read as zeros in hardware instead of
                # running off the end of the allocation (the plain buffer
                # pointer carries a 4 GiB bound).
                return _make_rebased_view(
                    fx.get_iter(t),
                    byte_off,
                    fx.Int64(n_elems) * fx.Int64(2),
                    fx.make_layout(n_elems, fx.Int32(1)),
                    _buf_flags_i32=buf_flags,
                    _elem_ir=elem_dtype.ir_type,
                )

            rows_in_seq = ops.max(seq_len, fx.Int32(1))
            k_div = rebased_div(
                k,
                (
                    fx.Int64(seq_start) * fx.Int64(k_tok_stride)
                    + fx.Int64(head_idx) * fx.Int64(k_head_stride)
                )
                * fx.Int64(2),
                rows_in_seq * k_tok_stride,
            )
            v_div = rebased_div(
                v,
                (
                    fx.Int64(seq_start) * fx.Int64(v_tok_stride)
                    + fx.Int64(head_idx) * fx.Int64(v_head_stride)
                )
                * fx.Int64(2),
                rows_in_seq * v_tok_stride,
            )
            k_lds_byte = fx.Int32(fx.ptrtoint(lds.k.ptr))
            v_lds_byte = fx.Int32(v_lds_base)

            def dma_tile(
                div,
                lds_byte,
                row_elems,
                n_instr,
                swz_rows,
                swz_mul,
                tok_stride,
                col_limit,
                kv_start,
            ):
                """Tile rows kv_start.. into LDS; wave w issues instructions
                w, w + NUM_WAVES, ...; lane i of instruction n lands at byte
                n * 1 KB + 16 i, i.e. row e // row_elems, chunk slot s, which
                holds global chunk s ^ swz_mul * (row % swz_rows). Rows past
                the sequence fall outside the descriptor (bounded to the
                sequence) and read as zeros: their scores are masked and P is
                0. The offset is a uniform tile base plus a per-lane constant,
                so nothing per row stays live across the loop."""
                tile_base = kv_start * tok_stride
                # Instruction j's rows are row0 + j * rows_per_j when a wave's
                # stride over instructions is whole rows; its swizzle then
                # depends on j only through (j * rows_per_j) % swz_rows, so the
                # per-lane offset takes at most a few values and the rest is a
                # uniform (scalar) j * rows_per_j * stride.
                elems_per_j = NUM_WAVES * DMA_INSTR_BYTES // 2
                whole_rows = elems_per_j % row_elems == 0
                rows_per_j = elems_per_j // row_elems
                e0 = wave_uni * fx.Int32(DMA_INSTR_BYTES // 2) + lane * fx.Int32(8)
                row0 = e0 // fx.Int32(row_elems)
                slot0 = (e0 % fx.Int32(row_elems)) // fx.Int32(8)
                lane_offs = {}
                for j in range_constexpr(n_instr // NUM_WAVES):
                    n = fx.Int32(j * NUM_WAVES) + wave_uni
                    if const_expr(whole_rows):
                        jr = (j * rows_per_j) % swz_rows
                        if jr not in lane_offs:
                            row_sw = (row0 + fx.Int32(jr)) % fx.Int32(swz_rows)
                            lane_offs[jr] = row0 * tok_stride + (
                                slot0 ^ (row_sw * fx.Int32(swz_mul))
                            ) * fx.Int32(8)
                        col = None
                        src_elem = (
                            tile_base
                            + fx.Int32(j * rows_per_j) * tok_stride
                            + lane_offs[jr]
                        )
                    else:
                        e = n * fx.Int32(DMA_INSTR_BYTES // 2) + lane * fx.Int32(8)
                        row = e // fx.Int32(row_elems)
                        slot = (e % fx.Int32(row_elems)) // fx.Int32(8)
                        col = (
                            slot ^ ((row % fx.Int32(swz_rows)) * fx.Int32(swz_mul))
                        ) * fx.Int32(8)
                        src_elem = tile_base + (row * tok_stride + col)
                    if const_expr(col_limit < row_elems):
                        if const_expr(col is None):
                            row_sw = (row0 + fx.Int32(jr)) % fx.Int32(swz_rows)
                            col = (slot0 ^ (row_sw * fx.Int32(swz_mul))) * fx.Int32(8)
                        src_elem = (col < fx.Int32(col_limit)).select(
                            src_elem, fx.Int32(0)
                        )
                    dst_byte = lds_byte + n * fx.Int32(DMA_INSTR_BYTES)
                    dst = fx.make_view(
                        fx.inttoptr(lds_ptr_ty, dst_byte), fx.make_layout(1, 1)
                    )
                    fx.copy(dma_atom, fx.slice(div, (None, src_elem)), dst)

            def dma_k(kv_start, stage=None):
                dma_tile(
                    k_div,
                    k_lds_byte
                    if stage is None
                    else k_lds_byte + stage * fx.Int32(K_STAGE_ELEMS * 2),
                    HEAD_DIM_K,
                    K_DMA_INSTR,
                    K_SWZ_ROWS,
                    1,
                    k_tok_stride,
                    head_dim,
                    kv_start,
                )

            def dma_v(kv_start):
                dma_tile(
                    v_div,
                    v_lds_byte,
                    hidden_dim,
                    V_DMA_INSTR,
                    V_SWZ_ROWS,
                    2,
                    v_tok_stride,
                    hidden_dim,
                    kv_start,
                )

        def run_kv_tile_dma(state, kv_start, next_start):
            """One KV tile, K and V by DMA into single LDS buffers: K(t) was
            issued by the previous tile; V(t) is issued once K(t) is in and
            overlaps GEMM1; K(t+1) is issued once V(t) is in and overlaps
            GEMM2. The barriers also fence each buffer's reads from the next
            write."""
            o_acc, m_old, l_old = split_state(state)
            rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()
            dma_v(kv_start)
            p_packs, m_new, l_new, corr = compute_p_tile(kv_start, m_old, l_old)
            if const_expr(SOFTMAX):
                o_acc = lazy_scale_o_rows(o_acc, corr)
            rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()
            dma_k(next_start)
            o_acc = accum_o_tile(o_acc, p_packs)
            return o_acc + list(m_new) + list(l_new)

        def run_kv_tile_kdb(state, t):
            """One KV tile with K double-buffered: K(t) was issued a whole
            tile ahead. The first barrier makes K(t) visible and frees the V
            buffer (GEMM2 of t - 1 is done) and K's other stage (GEMM1 of
            t - 1 is done), so V(t) and K(t + 1) go out; GEMM1 overlaps V(t),
            and the second barrier makes V(t) visible with K(t + 1) still in
            flight."""
            o_acc, m_old, l_old = split_state(state)
            rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()
            kv_start = tile_start(t)
            dma_v(kv_start)
            nxt = t + fx.Int32(1)
            dma_k(tile_start(nxt), nxt % fx.Int32(2))
            stage = t % fx.Int32(2)
            p_packs, m_new, l_new, corr = compute_p_tile(kv_start, m_old, l_old, stage)
            if const_expr(SOFTMAX):
                o_acc = lazy_scale_o_rows(o_acc, corr)
            rocdl.s_waitcnt(vmcnt=K_VM_PER_TILE)
            gpu.barrier()
            o_acc = accum_o_tile(o_acc, p_packs)
            return o_acc + list(m_new) + list(l_new)

        def run_kv_tile(state, kv_start, next_start):
            """One KV tile, software-pipelined: this tile's K and V arrive in
            registers from the previous iteration; the next tile's K load is
            issued once K is in LDS and its V load once V is, so both overlap
            this tile's GEMMs. LLVM places the waits at first use."""
            o_acc, m_old, l_old = split_state(state)
            k_vecs = state[N_STATE : N_STATE + NUM_BATCHES_K]
            v_vecs = state[N_STATE + NUM_BATCHES_K :]
            store_k_regs_to_lds(k_vecs)
            gpu.barrier()
            k_next = async_load_k_regs(next_start)
            p_packs, m_new, l_new, corr = compute_p_tile(kv_start, m_old, l_old)
            if const_expr(SOFTMAX):
                o_acc = lazy_scale_o_rows(o_acc, corr)
            store_v_regs_to_lds(v_vecs)
            rocdl.sched_group_barrier(rocdl.mask_dswr, NUM_BATCHES_V, 0)
            gpu.barrier()
            v_next = async_load_v_regs(next_start)
            # The next tile's K barrier fences these V reads from its V
            # store, and this tile's V barrier fenced the K reads above.
            o_acc = accum_o_tile(o_acc, p_packs)
            return o_acc + list(m_new) + list(l_new) + list(k_next) + list(v_next)

        if active:
            acc_init = [c_zero_v4f32 for _ in range(N_ACC)]
            if const_expr(SOFTMAX):
                acc_init = (
                    acc_init
                    + [c_neg_big.ir_value() for _ in range(Q_SUBTILES)]
                    + [c_zero_f.ir_value() for _ in range(Q_SUBTILES)]
                )
            first = tile_start(fx.Int32(0))
            if const_expr(KDB):
                dma_k(first, fx.Int32(0))
            elif const_expr(KV_DMA):
                dma_k(first)
            else:
                acc_init = (
                    acc_init
                    + list(async_load_k_regs(first))
                    + list(async_load_v_regs(first))
                )

            loop_results = acc_init
            for kv_tile, it in range(
                fx.Int32(0),
                fx.Int32(n_logical),
                fx.Int32(1),
                init=acc_init,
            ):  # ty: ignore
                state = list(it) if isinstance(it, (list, tuple)) else [it]
                t = fx.Int32(kv_tile)
                if const_expr(KDB):
                    state = run_kv_tile_kdb(state, t)
                elif const_expr(KV_DMA):
                    state = run_kv_tile_dma(
                        state, tile_start(t), tile_start(t + fx.Int32(1))
                    )
                else:
                    state = run_kv_tile(
                        state, tile_start(t), tile_start(t + fx.Int32(1))
                    )
                loop_results = yield state

            # ---- Epilogue. GEMM2's C fragment is [d, query]: d = lane_mod_16,
            # query = qg*16 + lane_div_16*4 + e. ----
            results, m_fin, l_fin = split_state(loop_results)
            if const_expr(SOFTMAX):
                # out = o / l; M = m + log2(l), -inf on rows with no key.
                inv_l = []
                for qg in range_constexpr(Q_SUBTILES):
                    l_tot = lanes_sum(fx.Float32(l_fin[qg]))
                    has_key = l_tot > c_zero_f
                    safe_l = has_key.select(l_tot, fx.Float32(1.0))
                    inv_l.append(has_key.select(fx.Float32(1.0) / safe_l, c_zero_f))
                    lse = fx.Float32(m_fin[qg]) + fx.log2(safe_l, fastmath="afn")
                    # -inf through its bit pattern; fast math assumes no infs.
                    lse_bits = has_key.select(lse.bitcast(fx.Int32), fx.Int32(-8388608))
                    if (lane_div_16 == fx.Int32(0)) & q_in_bounds[qg]:
                        lse_out[fx.Int64(seq_start + q_rows[qg]), head_idx] = (
                            lse_bits.bitcast(fx.Float32)
                        )
                results = scale_o_rows(results, inv_l)
            row_scale = out_scale
            if const_expr(SCALE_MODE == SCALE_SCALAR):
                row_scale = out_scale * fx.Float32(attn_scale[fx.Int32(0)])
            if const_expr(GROUPED and SCALE_MODE == SCALE_INV_N):
                row_scale = out_scale / fx.Float32(group_max_len)
            if const_expr(SCALE_MODE == SCALE_GROUP):
                row_scale = out_scale * fx.Float32(attn_scale[group_idx])
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
        lse_out: fx.Tensor,
        group_meta: fx.Tensor,
        num_seqs: fx.Int32,
        num_heads: fx.Int32,
        max_seq_len: fx.Int32,
        score_scale: fx.Float32,
        out_scale: fx.Float32,
        max_attn_len: fx.Int32,
        contextual_seq_len: fx.Int32,
        full_attn_size: fx.Int32,
        k_tok_stride: fx.Int32,
        k_head_stride: fx.Int32,
        v_tok_stride: fx.Int32,
        v_head_stride: fx.Int32,
        group_size: fx.Int32,
        meta_prefix_base: fx.Int32,
        meta_settings_base: fx.Int32,
        stream: fx.Stream,
    ) -> None:
        c_ngg = fx.Int32(NUM_GRID_GROUPS)
        num_q_tiles = (max_seq_len + fx.Int32(BLOCK_M - 1)) // fx.Int32(BLOCK_M)
        hz_total = num_seqs * num_heads
        hz_per_group = (hz_total + fx.Int32(NUM_GRID_GROUPS - 1)) // c_ngg
        grid_blocks = num_q_tiles * hz_per_group * c_ngg
        if const_expr(GROUPED):
            # One program per (tile slot, head), rounded up to the XCD count;
            # meta_prefix_base is the number of tile slots. hz_total carries
            # the sequence count.
            hz_total = num_seqs
            grid_blocks = (
                (meta_prefix_base * num_heads + fx.Int32(NUM_GRID_GROUPS - 1))
                // c_ngg
                * c_ngg
            )
        hstu_ragged_fwd_kernel(
            q,
            k,
            v,
            seq_offsets,
            num_targets,
            attn_scale,
            out,
            lse_out,
            group_meta,
            num_heads,
            num_q_tiles,
            hz_per_group,
            hz_total,
            score_scale,
            out_scale,
            max_attn_len,
            contextual_seq_len,
            full_attn_size,
            k_tok_stride,
            k_head_stride,
            v_tok_stride,
            v_head_stride,
            group_size,
            meta_prefix_base,
            meta_settings_base,
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
    kernel: str = "auto",
    schedule: str = "auto",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``mslk::hstu_ragged_fwd`` on a FlyDSL kernel; returns ``(out, M)``.

    Every sequence must be at most ``N`` tokens (T1 checks this; checking it
    here would need a host sync): rows past ``N`` are not written. ``M``
    ([T, H] float32) is written for softmax heads only; for SiLU heads its
    contents are unspecified (the backward does not read it).
    ``num_softmax_heads`` is 0 (all SiLU) or H (all softmax). The tiled kernel
    reads strided q/k/v (e.g. slices of a packed qkv) in place; the pipelined
    one copies them to contiguous when their strides do not fit it.

    ``kernel``: ``"pipelined"`` is the gfx950 flash-attention pipeline
    (:mod:`.ragged_fwd_gfx950`), ``"tiled"`` the kernel of this module.
    ``"auto"`` (no tile ``config``) takes the pipelined kernel for long
    causal or long-window calls it accepts, and the tiled kernel otherwise.

    ``schedule``: ``"reverse"`` runs each sequence's query blocks last first
    on a lower-triangular mask, so the heaviest causal blocks start earliest;
    ``"sorted"`` (pipelined kernel only) runs sequences longest first (an
    on-device argsort); ``"sorted_reverse"`` both; ``"plain"``
    neither. ``"auto"`` is ``"reverse"``: free, and 0.99-1.10x in the
    interim benchmark, while sorting gained up to 1.17x on some causal batches
    but lost up to 1.3x elsewhere to its extra launches; revisit with
    production length distributions."""
    from mslk.flydsl.jit import run_compiled

    from . import ragged_fwd_gfx950 as pipelined

    if kernel not in ("auto", "pipelined", "tiled"):
        raise ValueError(f"unknown kernel {kernel!r}")
    if schedule not in ("auto", "plain", "sorted", "reverse", "sorted_reverse"):
        raise ValueError(f"unknown schedule {schedule!r}")
    if shared_kv:
        v = k
    num_tokens, num_heads, head_dim = q.shape
    hidden_dim = v.shape[2]
    arch = _arch()
    if num_softmax_heads not in (0, num_heads):
        raise ValueError(
            f"num_softmax_heads must be 0 (SiLU) or H={num_heads} (softmax), "
            f"got {num_softmax_heads}"
        )
    softmax = num_softmax_heads > 0
    use_pipelined = False
    if kernel in ("auto", "pipelined") and config is None:
        decline = pipelined.decline_reason(q, k, v, softmax, arch)
        if kernel == "pipelined" and decline is not None:
            raise NotImplementedError(f"FlyDSL pipelined HSTU forward: {decline}")
        # The pipeline works on 256-row blocks of at least four 64-key
        # tiles; short sequences go to the tiled kernel's 64-row blocks.
        use_pipelined = decline is None and (
            kernel == "pipelined"
            or (
                invalid_attn_mask_type == "lower_triangular"
                and N > (_PIPELINED_MIN_N if softmax else _PIPELINED_MIN_N_SILU)
                and (max_attn_len == 0 or max_attn_len > _PIPELINED_MIN_WINDOW)
            )
        )
    if not use_pipelined:
        cfg = config or _first_supported(q, head_dim, hidden_dim, arch, max_attn_len)
        reason = check_supported(q, head_dim, hidden_dim, cfg, arch)
        if reason is not None:
            raise NotImplementedError(f"FlyDSL ragged HSTU forward: {reason}")
    spec = MaskSpec.from_args(
        invalid_attn_mask_type,
        max_attn_len,
        contextual_seq_len,
        full_attn_size,
        has_targets=num_targets is not None,
    )
    if attn_scale is None or softmax:
        # attn_scale scales SiLU heads only (T1).
        scale_mode = SCALE_INV_N
    elif attn_scale.ndim == 0:
        scale_mode = SCALE_SCALAR
    else:
        # As T1: one value per token (the kernel indexes it by token).
        if attn_scale.ndim != 1 or attn_scale.numel() != num_tokens:
            raise ValueError(
                "attn_scale must be a 0-dim tensor or have one value per token "
                f"(shape [{num_tokens}]), got {tuple(attn_scale.shape)}"
            )
        scale_mode = SCALE_TOKEN

    out = torch.empty(v.shape, dtype=q.dtype, device=q.device)
    M = torch.empty((num_tokens, num_heads), dtype=torch.float32, device=q.device)
    num_seqs = seq_offsets.shape[0] - 1
    if num_tokens == 0 or num_seqs == 0 or N == 0:
        out.zero_()
        M.fill_(float("-inf"))
        return out, M

    seq_offsets = seq_offsets.to(torch.int32)
    if num_targets is not None:
        num_targets = num_targets.to(torch.int32)
    if scale_mode == SCALE_INV_N:
        attn_scale = torch.empty(1, dtype=torch.float32, device=q.device)
    else:
        attn_scale = attn_scale.reshape(-1).to(torch.float32)

    sort_idx = None
    reverse = (
        schedule in ("auto", "reverse", "sorted_reverse") and spec.mask_type == LOWER
    )
    if schedule in ("sorted", "sorted_reverse") and use_pipelined:
        # Longest sequence first; computed on the device (no host sync).
        lengths = seq_offsets[1:] - seq_offsets[:-1]
        sort_idx = torch.argsort(lengths, descending=True).to(torch.int32)
    if use_pipelined:
        pipelined.hstu_ragged_fwd_gfx950(
            N,
            alpha,
            q,
            k,
            v,
            seq_offsets,
            spec,
            seq_offsets if num_targets is None else num_targets,
            attn_scale,
            scale_mode,
            max_attn_len,
            contextual_seq_len,
            full_attn_size,
            softmax,
            out,
            M,
            sort_idx,
            reverse,
        )
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
        reverse,
        num_heads == 1,
    )
    if num_targets is None:
        num_targets = seq_offsets
    if softmax:
        score_scale, out_scale = alpha * _LOG2E, 1.0
    else:
        # The SiLU gate runs in base 2: -1/log2e undoes the base change.
        score_scale, out_scale = alpha * -_LOG2E, -1.0 / _LOG2E
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
        M,
        seq_offsets,  # group_meta: unused by ragged calls
        num_seqs,
        num_heads,
        N,
        score_scale,
        out_scale,
        max_attn_len,
        contextual_seq_len,
        full_attn_size,
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        1,
        0,
        0,
        torch.cuda.current_stream(q.device),
    )
    return out, M
