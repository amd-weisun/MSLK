# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""HSTU ragged backward, SiLU heads, on gfx950: a KV-tile kernel built around
32x32x16 MFMAs (EXPERIMENTAL; see :mod:`.ragged_bwd` for the production path).

A block owns 128 keys of one (sequence, head); wave w owns keys 32w..32w+31.
It loops over the 32-row query tiles that see its keys (``SeqMask.q_tiles``):

- S = Q K_w^T and dP = dO V_w^T: A = Q / dO rows from LDS, B = the wave's K /
  V held in registers for the whole block. C: lane n = lane % 32 (key),
  m = 8 (r // 4) + 4 (lane // 32) + r % 4 (query row).
- With x = alpha S: P = silu(x) c on the mask, dS = alpha dP c silu'(x) (c
  the per-row output scale; LDS keeps alpha c, so P = S sigmoid(x) (alpha c)
  and dS carries the alpha that dK and dQ need).
- dV_w^T += dO^T P and dK_w^T += Q^T dS: A = dO / Q from transposing LDS
  reads (lane d = lane % 32), B = P / dS straight from the S / dP
  accumulators; both take K slots {8g + 4 (lane // 32) + e}, so their K
  orders agree without a shuffle. C: lane n = lane % 32, 4 contiguous d per
  row group.
- dQ: dS goes to LDS as [n][m]; wave w takes d slice w (32 columns), sums over
  the block's 128 keys (A = dS, B = K, both transposing reads) and adds to
  f32 dQ with atomics.
"""

import functools
from typing import Optional

import torch

from .common import MaskSpec

BLOCK_N = 128
BLOCK_M = 32
NUM_WAVES = 4
WARP_SIZE = 64
_LOG2E = 1.4426950408889634
import os as _os  # EXPERIMENT

KV_AHEAD = int(_os.environ.get("KV_AHEAD", "4"))
KV_ORDER = int(_os.environ.get("KV_ORDER", "0"))
KV_KLDS = int(_os.environ.get("KV_KLDS", "0"))
KV_PROF = int(_os.environ.get("KV_PROF", "0"))
KV_NOPROC = int(_os.environ.get("KV_NOPROC", "0"))
_PROF = None


@functools.lru_cache(maxsize=128)
def build_ragged_bwd_kv(dim: int, dtype_str: str, spec: MaskSpec, arch: str):
    import flydsl.compiler as flyc  # pyre-ignore[21]
    import flydsl.expr as fx  # pyre-ignore[21]
    from flydsl._mlir import ir as _ir  # pyre-ignore[21]
    from flydsl._mlir.dialects import fly as _fly_d, llvm as _llvm_d  # pyre-ignore[21]
    from flydsl.expr import const_expr, gpu, range_constexpr, rocdl  # pyre-ignore[21]
    from flydsl.expr.typing import T, Vector as Vec  # pyre-ignore[21]
    from flydsl.expr.utils.arith import (  # pyre-ignore[21]
        _to_raw as _raw,
        ArithValue,
        xori,
    )
    from mslk.attention.flydsl.flash_attn_utils import scf_if_dispatch
    from mslk.flydsl import buffer_ops

    from .common import device_ops, seq_mask

    D = dim
    assert D in (64, 128)
    MASK = (
        spec.mask_type,
        spec.has_window,
        spec.has_contextual,
        spec.has_targets,
        spec.has_full_attn,
    )
    elem_dtype = fx.BFloat16 if dtype_str == "bf16" else fx.Float16
    K_STEPS = D // 16  # S / dP over d
    D_TILES = D // 32  # dV / dK / dQ d tiles
    # LDS rows are unpadded and XOR-swizzled in 16-byte groups (see swz), so
    # the row reads (b128) and the transposing reads (4 rows x 4 groups per
    # 32 lanes) are both free of bank conflicts.
    KS = D  # LDS row stride of K / Q / dO (elements)
    DSS = BLOCK_M  # LDS row stride of dS^T [n][m]
    NG = D // 8  # 16-byte groups per K / Q / dO row
    BLOCK_THREADS = NUM_WAVES * WARP_SIZE
    GROUPS = D // 8  # 8-element groups per row
    ROWS_PER_PASS = BLOCK_THREADS // GROUPS
    QDO_PASSES = -(-BLOCK_M // ROWS_PER_PASS)

    @fx.struct
    class SharedStorage:
        k: fx.Array[elem_dtype, BLOCK_N * KS, 16]
        q: fx.Array[elem_dtype, 2 * BLOCK_M * KS, 16]
        do: fx.Array[elem_dtype, 2 * BLOCK_M * KS, 16]
        ds: fx.Array[elem_dtype, 2 * BLOCK_N * DSS, 16]
        rs: fx.Array[fx.Float32, 2 * BLOCK_M, 16]
        item: fx.Array[fx.Int32, 4, 16]

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def hstu_bwd_kv_kernel(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        dout: fx.Tensor,
        dk_out: fx.Tensor,
        dv_out: fx.Tensor,
        dq_f32: fx.Tensor,
        row_scale: fx.Tensor,
        prof: fx.Tensor,
        work: fx.Tensor,
        seq_offsets: fx.Tensor,
        num_targets: fx.Tensor,
        num_seqs: fx.Int32,
        num_heads: fx.Int32,
        n_kv_tiles: fx.Int32,
        total_tokens: fx.Int32,
        alpha: fx.Float32,
        max_attn_len: fx.Int32,
        contextual_seq_len: fx.Int32,
        full_attn_size: fx.Int32,
    ) -> None:
        ops = device_ops()
        tid = fx.Int32(gpu.thread_idx.x)
        wave = tid // fx.Int32(WARP_SIZE)
        lane = tid % fx.Int32(WARP_SIZE)
        l32 = lane % fx.Int32(32)
        h = lane // fx.Int32(32)
        l16 = lane % fx.Int32(16)
        g16 = lane // fx.Int32(16)
        v4_type = Vec.make_type(4, elem_dtype)
        v16f32 = Vec.make_type(16, fx.Float32)
        zero16 = Vec.filled(16, 0.0, fx.Float32).ir_value()
        zero8 = Vec.filled(8, 0.0, elem_dtype).ir_value()

        def mfma(a, b, c):
            if const_expr(dtype_str == "bf16"):
                return rocdl.mfma_f32_32x32x16_bf16(v16f32, [a, b, c, 0, 0, 0])
            return rocdl.mfma_f32_32x32x16_f16(v16f32, [a, b, c, 0, 0, 0])

        # ---- (sequence, head)-major, KV tiles consecutive, XCD-contiguous ----
        bid = fx.Int32(gpu.block_idx.x)

        def _rt():
            v = _llvm_d.inline_asm(
                T.i64,
                [],
                "s_memrealtime $0\ns_waitcnt lgkmcnt(0)",
                "=s",
                has_side_effects=True,
            )
            return fx.Int32(ArithValue(v).trunci(T.i32))

        def _getreg(name):
            return fx.Int32(
                _llvm_d.inline_asm(
                    T.i32,
                    [],
                    f"s_getreg_b32 $0, hwreg({name})",
                    "=s",
                    has_side_effects=True,
                )
            )

        def _prof_store(idx, i, val):
            prof[idx * fx.Int32(4) + fx.Int32(i)] = val

        total = num_seqs * num_heads * n_kv_tiles
        per_xcd = (total + fx.Int32(7)) // fx.Int32(8)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        # Persistent: a block per CU takes (sequence, head, KV tile) items
        # from its XCD's counter until they run out. Block lengths vary a lot,
        # and with a block per item the in-order dispatcher left ~17% of the
        # CUs idle.
        xcd = bid % fx.Int32(8)
        work_base = fx.Int64(
            _llvm_d.PtrToIntOp(
                T.i64,
                _fly_d.extract_aligned_pointer_as_index(
                    _ir.Type.parse("!llvm.ptr"), work
                ),
            ).result
        )
        item_view = lds.item.view(fx.make_layout(4, 1))

        # Every thread runs the claim (lane 0 adds 1, the rest 0): a branch on
        # tid == 0 here gets jump-threaded into the loop latch, the loop turns
        # irreducible and the barriers below hang. The atomic optimizer makes
        # it one atomic per wave.
        def fetch():
            ptr = _llvm_d.IntToPtrOp(
                _ir.Type.parse("!llvm.ptr<1>"),
                _raw(work_base + fx.Int64(xcd * fx.Int32(4))),
            ).result
            old = _llvm_d.AtomicRMWOp(
                _llvm_d.AtomicBinOp.add,
                ptr,
                _raw((tid == fx.Int32(0)).select(fx.Int32(1), fx.Int32(0))),
                _llvm_d.AtomicOrdering.monotonic,
            ).result
            item_view[wave] = fx.Int32(rocdl.readfirstlane(T.i32, old))
            gpu.barrier()
            # Wave-uniform (a scalar), so the loop exit is uniform.
            got = fx.Int32(rocdl.readfirstlane(T.i32, _raw(fx.Int32(item_view[0]))))
            gpu.barrier()
            return got

        def _uni(x):
            return fx.Int32(rocdl.readfirstlane(T.i32, _raw(x)))

        def process(order):
            item_id = order
            if const_expr(KV_PROF):
                t_start = _rt()
            in_range = order < total
            order = in_range.select(order, fx.Int32(0))
            if const_expr(KV_ORDER == 1):
                n_bh = num_seqs * num_heads
                kv_tile = order // n_bh
                bh = order % n_bh
            else:
                kv_tile = order % n_kv_tiles
                bh = order // n_kv_tiles
            seq = bh // num_heads
            head = bh % num_heads
            # Per-item scalars, made wave-uniform: after the last item's
            # stores these are vector loads, and a dQ descriptor built from
            # them would need a waterfall loop on every atomic.
            seq_start = _uni(fx.Int32(seq_offsets[seq]))
            seq_len = _uni(fx.Int32(seq_offsets[seq + fx.Int32(1)])) - seq_start
            seq_len = in_range.select(seq_len, fx.Int32(0))
            n_tgt = fx.Int32(0)
            if const_expr(MASK[3]):
                n_tgt = _uni(fx.Int32(num_targets[seq]))
            sm = seq_mask(
                MaskSpec(*MASK),
                ops,
                seq_len,
                num_targets=n_tgt,
                max_attn_len=max_attn_len,
                contextual_seq_len=contextual_seq_len,
                full_attn_size=full_attn_size,
            )
            n0 = kv_tile * fx.Int32(BLOCK_N)
            nw0 = n0 + wave * fx.Int32(32)

            # The AST rewriter carries locals whose methods are called inside a
            # dynamic if/for as loop state; reach the mask only through closures.
            def _q_tiles():
                return sm.q_tiles(n0, BLOCK_N, BLOCK_M)

            def _to_id(x):
                return sm.to_id(x)

            def _visible(m_abs, n_abs, n_id):
                return sm.visible(m_abs, n_abs, _to_id(m_abs), n_id)

            def _needs_mask(m0, nk0):
                return sm.tile_needs_mask(
                    m0, m0 + fx.Int32(BLOCK_M - 1), nk0, nk0 + fx.Int32(31)
                )

            in_row8 = fx.make_layout((D // 8, 8), (8, 1))
            in_row4 = fx.make_layout((D // 4, 4), (4, 1))

            def load8(t, row, colgrp):
                ok = row < seq_len
                safe = ok.select(seq_start + row, seq_start)
                sub = t[fx.Int64(safe), head, None]
                raw = fx.make_view(fx.get_iter(sub), in_row8)[colgrp, None].load()
                return ok.select(Vec(raw).ir_value(), zero8)

            k_view = lds.k.view(fx.make_layout((BLOCK_N, KS // 8, 8), (KS, 8, 1)))
            q_view = lds.q.view(fx.make_layout((2 * BLOCK_M, KS // 8, 8), (KS, 8, 1)))
            do_view = lds.do.view(fx.make_layout((2 * BLOCK_M, KS // 8, 8), (KS, 8, 1)))
            rs_view = lds.rs.view(fx.make_layout(2 * BLOCK_M, 1))

            def _rs_store(idx, val):
                # A subscript store inside a dynamic if reads as reassigning the
                # view; keep it in a function.
                rs_view[idx] = val

            ds_view = lds.ds.view(
                fx.make_layout((2 * BLOCK_N, DSS // 4, 4), (DSS, 4, 1))
            )

            def _lds_addr(arr):
                return fx.Int32(
                    ArithValue(_raw(fx.Int64(fx.ptrtoint(arr.ptr)))).trunci(T.i32)
                )

            k_base = _lds_addr(lds.k)
            q_base = _lds_addr(lds.q)
            do_base = _lds_addr(lds.do)
            ds_base = _lds_addr(lds.ds)

            # Swizzles: a row's 16-byte group g is stored at g ^ swz(row) (K / Q /
            # dO) or g ^ ds_swz(row) (dS^T, 4 groups a row). Both depend on
            # row % 16 only, so row offsets that are multiples of 16 (buffers, K
            # steps) stay constants.
            def swz(r):
                if const_expr(NG == 16):
                    return (r % fx.Int32(4)) * fx.Int32(4) + (
                        r // fx.Int32(4)
                    ) % fx.Int32(4)
                return ((r // fx.Int32(2)) % fx.Int32(2)) * fx.Int32(4) + (
                    r // fx.Int32(4)
                ) % fx.Int32(4)

            def sgrp(r, g):
                return fx.Int32(xori(g, swz(r)))

            def ds_sgrp(r, g):
                return fx.Int32(xori(g, (r // fx.Int32(4)) % fx.Int32(4)))

            # ds_read_b64_tr_b16 of a 4 x 16 block: per 16-lane group, lane i
            # addresses row + i // 4, col + (i % 4) * 4 and gets column col + i's
            # four rows. A lane's byte address is a base fixed for the kernel
            # (lr: its row below 16, col = 8 g + e) plus a constant, in 32 bits,
            # so the constant folds into the instruction's offset field.
            def tr_lane(stride, lr, g, sw):
                e = (l16 % fx.Int32(2)) * fx.Int32(4)
                return (lr * fx.Int32(stride) + sw(lr, g) * fx.Int32(8) + e) * fx.Int32(
                    2
                )

            def tr4(addr, const_elems):
                ptr = _llvm_d.IntToPtrOp(
                    _ir.Type.parse("!llvm.ptr<3>"),
                    _raw(addr + fx.Int32(2 * const_elems)),
                ).result
                return rocdl.ds_read_tr16_b64(v4_type, ptr).result

            def tr8(addr_lo, addr_hi, const_elems):
                a = tr4(addr_lo, const_elems)
                b = tr4(addr_hi, const_elems)
                return Vec(a).shuffle(Vec(b), list(range(8))).ir_value()

            g_lane = (g16 % fx.Int32(2)) * fx.Int32(2) + (
                l16 % fx.Int32(4)
            ) // fx.Int32(2)
            # dV / dK A operands: Q / dO rows 16 ks + 4 h + 8 half, d tile dt.
            qdo_lane = {
                (half, dt): tr_lane(
                    KS,
                    fx.Int32(8 * half) + h * fx.Int32(4) + l16 // fx.Int32(4),
                    fx.Int32(4 * dt) + g_lane,
                    sgrp,
                )
                for half in range(2)
                for dt in range(D_TILES)
            }
            # dQ: dS rows (keys) 16 kk + 8 h + 4 half, m cols; K the same rows, d
            # cols 32 wave + 16 (g16 % 2).
            dq_lr = [
                fx.Int32(4 * half) + h * fx.Int32(8) + l16 // fx.Int32(4)
                for half in range(2)
            ]
            ds_lane = [tr_lane(DSS, lr, g_lane, ds_sgrp) for lr in dq_lr]
            kd_lane = [
                tr_lane(KS, lr, wave * fx.Int32(4) + g_lane, sgrp) for lr in dq_lr
            ]

            # f32 atomics into dQ, rebased to the sequence (32-bit offsets).
            def _raw_f32_rsrc(tensor, base_byte_offset, num_records_bytes):
                base_ptr = _fly_d.extract_aligned_pointer_as_index(
                    _ir.Type.parse("!llvm.ptr"), tensor
                )
                base_i64 = fx.Int64(
                    _llvm_d.PtrToIntOp(T.i64, base_ptr).result
                ) + fx.Int64(base_byte_offset)
                lo = ArithValue(_raw(base_i64)).trunci(T.i32)
                hi = ArithValue(ArithValue(_raw(base_i64)).shrui(fx.Int64(32))).trunci(
                    T.i32
                )
                return Vec.from_elements(
                    [
                        lo,
                        hi,
                        _raw(num_records_bytes),
                        buffer_ops._create_i32_constant(buffer_ops._get_buffer_flags()),
                    ],
                    fx.Int32,
                ).ir_value()

            # Bounded to the sequence's rows: atomics past it are dropped in
            # hardware, so the per-row check is free.
            row_bytes = num_heads * fx.Int32(D * 4)
            dq_rsrc = _raw_f32_rsrc(
                dq_f32,
                fx.Int64(seq_start) * fx.Int64(row_bytes),
                seq_len * row_bytes,
            )

            def atomic_dq(off, val):
                rocdl.raw_buffer_atomic_fadd(
                    T.f32,
                    _raw(val),
                    dq_rsrc,
                    _raw(off),
                    _raw(fx.Int32(0)),
                    aux=_ir.IntegerAttr.get(T.i32, 0),
                )

            active = n0 < seq_len
            if active:
                # The wave's K / V rows as 32x32x16 B operands (lane n, 8 d per
                # K step); K also goes to LDS for dQ.
                k_regs = []
                v_regs = []
                for s in range_constexpr(K_STEPS):
                    grp = fx.Int32(2 * s) + h
                    kv8 = load8(k, nw0 + l32, grp)
                    if const_expr(not KV_KLDS):
                        k_regs.append(kv8)
                    v_regs.append(load8(v, nw0 + l32, grp))
                    k_view[wave * fx.Int32(32) + l32, sgrp(l32, grp), None].store(
                        Vec(kv8)
                    )

                pre, lo, hi, tail, end = _q_tiles()
                main_n = hi - lo
                n_logical = pre + main_n + (end - tail)

                def phys(t):
                    in_main = t - pre
                    in_tail = in_main - main_n
                    return (t < pre).select(
                        t, (in_main < main_n).select(lo + in_main, tail + in_tail)
                    )

                def load_tile(m0):
                    """This thread's share of a query tile: Q / dO 8-element
                    groups and, for threads < BLOCK_M, the row's output scale."""
                    qs, dos = [], []
                    for ps in range_constexpr(QDO_PASSES):
                        r = fx.Int32(ps * ROWS_PER_PASS) + tid // fx.Int32(GROUPS)
                        r = (r < fx.Int32(BLOCK_M)).select(r, fx.Int32(0))
                        grp = tid % fx.Int32(GROUPS)
                        qs.append(load8(q, m0 + r, grp))
                        dos.append(load8(dout, m0 + r, grp))
                    rm = m0 + (tid % fx.Int32(BLOCK_M))
                    rm = (rm < seq_len).select(rm, fx.Int32(0))
                    rsv = fx.Float32(
                        row_scale[fx.Int64(head * total_tokens + seq_start + rm)]
                    )
                    return qs, dos, rsv

                def store_tile(buf, qs, dos, rsv):
                    row0 = buf * fx.Int32(BLOCK_M)
                    for ps in range_constexpr(QDO_PASSES):
                        r = fx.Int32(ps * ROWS_PER_PASS) + tid // fx.Int32(GROUPS)
                        grp = tid % fx.Int32(GROUPS)
                        if r < fx.Int32(BLOCK_M):
                            q_view[row0 + r, sgrp(r, grp), None].store(Vec(qs[ps]))
                            do_view[row0 + r, sgrp(r, grp), None].store(Vec(dos[ps]))
                    if tid < fx.Int32(BLOCK_M):
                        _rs_store(row0 + tid, rsv * alpha)

                def tile_m0(t):
                    t = (t < n_logical).select(t, n_logical - fx.Int32(1))
                    return phys(t) * fx.Int32(BLOCK_M)

                # Prologue: tile 0 into buffer 0; tile 1 in registers. Each
                # iteration then issues tile t + 2's loads, a whole iteration
                # before they are stored (one tile of compute did not cover the
                # global load latency at one wave per SIMD).
                qs0, dos0, rs0 = load_tile(tile_m0(fx.Int32(0)))
                store_tile(fx.Int32(0), qs0, dos0, rs0)
                qs1, dos1, rs1 = load_tile(tile_m0(fx.Int32(1)))
                gpu.barrier()

                # Tile t's dQ atomics are issued during tile t + 1's S / dP MFMAs
                # (back to back they stall on the memory queue). The first
                # iteration has none: its base points past the sequence, where
                # the bounded descriptor drops them.
                no_dq_base = seq_len * row_bytes
                init = (
                    [zero16 for _ in range(2 * D_TILES)]
                    + list(qs1)
                    + list(dos1)
                    + [rs1, zero16, no_dq_base]
                )
                for t_idx, acc in range(fx.Int32(0), n_logical, fx.Int32(1), init=init):  # ty: ignore
                    acc = list(acc) if isinstance(acc, (list, tuple)) else [acc]
                    dv_acc = acc[:D_TILES]
                    dk_acc = acc[D_TILES : 2 * D_TILES]
                    pf = acc[2 * D_TILES :]
                    qs_n = pf[:QDO_PASSES]
                    dos_n = pf[QDO_PASSES : 2 * QDO_PASSES]
                    rs_n = pf[2 * QDO_PASSES]
                    dq_prev = pf[2 * QDO_PASSES + 1]
                    dq_prev_base = fx.Int32(pf[2 * QDO_PASSES + 2])
                    t_cur = fx.Int32(t_idx)
                    m0 = phys(t_cur) * fx.Int32(BLOCK_M)
                    buf = t_cur % fx.Int32(2)
                    row0 = buf * fx.Int32(BLOCK_M)
                    # Tile t + 2 into registers under this tile's compute.
                    qs_nn, dos_nn, rs_nn = load_tile(tile_m0(t_cur + fx.Int32(2)))

                    # ---- S = Q K_w^T, dP = dO V_w^T ----
                    # All A operands first (registers are plentiful at one wave
                    # per SIMD), so the LDS reads overlap instead of each MFMA
                    # waiting on its own.
                    qas, das = [], []
                    if const_expr(KV_KLDS):
                        k_regs = []
                    for s in range_constexpr(K_STEPS):
                        grp = fx.Int32(2 * s) + h
                        qas.append(
                            q_view[row0 + l32, sgrp(l32, grp), None].load().ir_value()
                        )
                        das.append(
                            do_view[row0 + l32, sgrp(l32, grp), None].load().ir_value()
                        )
                        if const_expr(KV_KLDS):
                            k_regs.append(
                                k_view[wave * fx.Int32(32) + l32, sgrp(l32, grp), None]
                                .load()
                                .ir_value()
                            )
                    s_acc = zero16
                    dp_acc = zero16
                    per_step = -(-16 // K_STEPS)
                    for s in range_constexpr(K_STEPS):
                        s_acc = mfma(qas[s], k_regs[s], s_acc)
                        dp_acc = mfma(das[s], v_regs[s], dp_acc)
                        for r in range_constexpr(
                            per_step * s, min(16, per_step * (s + 1))
                        ):
                            atomic_dq(
                                dq_prev_base
                                + fx.Int32(8 * (r // 4) + r % 4) * row_bytes,
                                fx.Float32(Vec(dq_prev)[r]),
                            )
                        # Keep the atomics spread over the MFMAs: left alone, the
                        # scheduler bunches them, and they stall on the queue.
                        rocdl.sched_barrier(0)

                    # ---- P, dS (C layout: n = lane % 32, m = 8 (r // 4) + 4 h + r % 4) ----
                    def pack8(vals, ks):
                        return (
                            Vec.from_elements(
                                [vals[4 * (2 * ks) + e] for e in range(4)]
                                + [vals[4 * (2 * ks + 1) + e] for e in range(4)],
                                fx.Float32,
                            )
                            .to(elem_dtype)
                            .ir_value()
                        )

                    # Rows / keys past the sequence have zero Q / dO / K / V, so
                    # their P and dS are already 0: only the HSTU mask matters,
                    # and tiles it fully covers (uniform per wave) skip it.
                    def elementwise(masked):
                        n_abs = nw0 + l32
                        n_id = _to_id(n_abs)
                        p_vals, ds_vals = [], []
                        for r in range_constexpr(16):
                            m_loc = fx.Int32(8 * (r // 4) + r % 4) + h * fx.Int32(4)
                            c = fx.Float32(rs_view[row0 + m_loc])
                            s_raw = fx.Float32(Vec(s_acc)[r])
                            x = s_raw * alpha
                            e = fx.exp2(x * fx.Float32(-_LOG2E), fastmath="afn")
                            sig = fx.Float32(
                                fx.rocdl.rcp(
                                    fx.Float32.ir_type, (fx.Float32(1.0) + e).ir_value()
                                )
                            )
                            p = s_raw * sig * c
                            g = (
                                c
                                * sig
                                * (fx.Float32(1.0) + x * (fx.Float32(1.0) - sig))
                            )
                            dsv = fx.Float32(Vec(dp_acc)[r]) * g
                            if const_expr(masked):
                                ok = _visible(m0 + m_loc, n_abs, n_id)
                                p = ok.select(p, fx.Float32(0.0))
                                dsv = ok.select(dsv, fx.Float32(0.0))
                            p_vals.append(p)
                            ds_vals.append(dsv)
                        return [pack8(p_vals, ks) for ks in range(2)] + [
                            pack8(ds_vals, ks) for ks in range(2)
                        ]

                    packed = scf_if_dispatch(
                        _needs_mask(m0, nw0),
                        lambda *_: elementwise(True),
                        lambda *_: elementwise(False),
                        result_names=("_p0", "_p1", "_ds0", "_ds1"),
                        result_values=[zero8] * 4,
                    )
                    p8 = [packed[0], packed[1]]
                    ds8 = [packed[2], packed[3]]

                    # dS^T -> LDS [n][m] for dQ, double-buffered: its readers two
                    # tiles back passed the last tile's barrier. Group gq is half
                    # gq % 2 of ds8[gq // 2].
                    for gq in range_constexpr(4):
                        ds_view[
                            buf * fx.Int32(BLOCK_N) + wave * fx.Int32(32) + l32,
                            ds_sgrp(l32, fx.Int32(gq)) * fx.Int32(2) + h,
                            None,
                        ].store(
                            Vec(ds8[gq // 2]).shuffle(
                                Vec(ds8[gq // 2]), [4 * (gq % 2) + e for e in range(4)]
                            )
                        )

                    # ---- dV^T += dO^T P, dK^T += Q^T dS ----
                    rocdl.sched_barrier(0)
                    q_tile = q_base + row0 * fx.Int32(KS * 2)
                    do_tile = do_base + row0 * fx.Int32(KS * 2)
                    do_ts, q_ts = {}, {}
                    for dt in range_constexpr(D_TILES):
                        lo, hi = qdo_lane[(0, dt)], qdo_lane[(1, dt)]
                        for ks in range_constexpr(2):
                            c = 16 * ks * KS
                            do_ts[(dt, ks)] = tr8(do_tile + lo, do_tile + hi, c)
                            q_ts[(dt, ks)] = tr8(q_tile + lo, q_tile + hi, c)
                    for dt in range_constexpr(D_TILES):
                        for ks in range_constexpr(2):
                            dv_acc[dt] = mfma(do_ts[(dt, ks)], p8[ks], dv_acc[dt])
                            dk_acc[dt] = mfma(q_ts[(dt, ks)], ds8[ks], dk_acc[dt])
                    # Reads run KV_AHEAD MFMAs ahead (left alone, the scheduler
                    # puts each right before its MFMA and waits on it).
                    if const_expr(KV_AHEAD > 0):
                        n_mf = 4 * D_TILES
                        rocdl.sched_group_barrier(rocdl.mask_dsrd, 2 * KV_AHEAD, 0)
                        for i in range_constexpr(n_mf):
                            rocdl.sched_group_barrier(rocdl.mask_mfma, 1, 0)
                            if const_expr(i < n_mf - KV_AHEAD):
                                rocdl.sched_group_barrier(rocdl.mask_dsrd, 2, 0)
                        rocdl.sched_barrier(0)

                    # The next tile into the other buffer (free since the last
                    # iteration's closing barrier), before the dQ atomics: the
                    # memory counter is in order, so waiting for these loads after
                    # the atomics would also wait for the atomics.
                    store_tile(fx.Int32(1) - buf, qs_n, dos_n, rs_n)
                    gpu.barrier()  # dS (and the next tile) visible to every wave
                    dq_acc = zero16
                    dq_cur_base = no_dq_base
                    ds_tile = ds_base + buf * fx.Int32(BLOCK_N * DSS * 2)
                    if wave < fx.Int32(D_TILES):
                        ds_as, k_bs = [], []
                        for kk in range_constexpr(BLOCK_N // 16):
                            ds_as.append(
                                tr8(
                                    ds_tile + ds_lane[0],
                                    ds_tile + ds_lane[1],
                                    16 * kk * DSS,
                                )
                            )
                            k_bs.append(
                                tr8(
                                    k_base + kd_lane[0],
                                    k_base + kd_lane[1],
                                    16 * kk * KS,
                                )
                            )
                        for kk in range_constexpr(BLOCK_N // 16):
                            dq_acc = mfma(ds_as[kk], k_bs[kk], dq_acc)
                        # One base per lane; row r adds a constant row count.
                        dq_cur_base = (m0 + h * fx.Int32(4)) * row_bytes + (
                            head * fx.Int32(D) + wave * fx.Int32(32) + l32
                        ) * fx.Int32(4)
                    # No closing barrier: the next tile writes the other dS buffer
                    # and, after its own barrier, this tile's Q / dO buffer.
                    loop_out = yield (
                        dv_acc
                        + dk_acc
                        + list(qs_nn)
                        + list(dos_nn)
                        + [rs_nn, dq_acc, dq_cur_base]
                    )

                res = (
                    list(loop_out)
                    if isinstance(loop_out, (list, tuple))
                    else [loop_out]
                )
                dv_fin = res[:D_TILES]
                dk_fin = res[D_TILES : 2 * D_TILES]
                dq_last = res[2 * D_TILES + 2 * QDO_PASSES + 1]
                dq_last_base = fx.Int32(res[2 * D_TILES + 2 * QDO_PASSES + 2])
                for r in range_constexpr(16):
                    atomic_dq(
                        dq_last_base + fx.Int32(8 * (r // 4) + r % 4) * row_bytes,
                        fx.Float32(Vec(dq_last)[r]),
                    )
                # ---- dK / dV: lane n = lane % 32, d = 32 dt + 8 g + 4 h + e ----
                n_abs = nw0 + l32
                if n_abs < seq_len:
                    row = fx.Int64(seq_start + n_abs)
                    for dt in range_constexpr(D_TILES):
                        for gq in range_constexpr(4):
                            grp4 = fx.Int32(dt * 8 + 2 * gq) + h
                            vk = [
                                fx.Float32(Vec(dk_fin[dt])[4 * gq + e])
                                for e in range(4)
                            ]
                            vv = [
                                fx.Float32(Vec(dv_fin[dt])[4 * gq + e])
                                for e in range(4)
                            ]
                            fx.make_view(fx.get_iter(dk_out[row, head, None]), in_row4)[
                                grp4, None
                            ].store(Vec.from_elements(vk, fx.Float32).to(elem_dtype))
                            fx.make_view(fx.get_iter(dv_out[row, head, None]), in_row4)[
                                grp4, None
                            ].store(Vec.from_elements(vv, fx.Float32).to(elem_dtype))

            if const_expr(KV_PROF):
                # Branch-free (every thread stores the same values).
                gpu.barrier()
                t_end = _rt()
                _prof_store(item_id, 0, t_start)
                _prof_store(item_id, 1, t_end)
                _prof_store(item_id, 2, _getreg("HW_REG_HW_ID"))
                _prof_store(item_id, 3, _getreg("HW_REG_XCC_ID"))

        item = fetch()
        while item < per_xcd:
            if const_expr(not KV_NOPROC):
                process(xcd * per_xcd + item)
            item = fetch()

    @flyc.jit
    def launch(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        dout: fx.Tensor,
        dk_out: fx.Tensor,
        dv_out: fx.Tensor,
        dq_f32: fx.Tensor,
        row_scale: fx.Tensor,
        prof: fx.Tensor,
        work: fx.Tensor,
        seq_offsets: fx.Tensor,
        num_targets: fx.Tensor,
        num_seqs: fx.Int32,
        num_heads: fx.Int32,
        n_kv_tiles: fx.Int32,
        total_tokens: fx.Int32,
        alpha: fx.Float32,
        max_attn_len: fx.Int32,
        contextual_seq_len: fx.Int32,
        full_attn_size: fx.Int32,
        n_blocks: fx.Int32,
        stream: fx.Stream,
    ) -> None:
        grid = n_blocks
        hstu_bwd_kv_kernel(
            q,
            k,
            v,
            dout,
            dk_out,
            dv_out,
            dq_f32,
            row_scale,
            prof,
            work,
            seq_offsets,
            num_targets,
            num_seqs,
            num_heads,
            n_kv_tiles,
            total_tokens,
            alpha,
            max_attn_len,
            contextual_seq_len,
            full_attn_size,
            value_attrs={
                "rocdl.waves_per_eu": 1,
                "rocdl.flat_work_group_size": f"{BLOCK_THREADS},{BLOCK_THREADS}",
            },
        ).launch(grid=(grid, 1, 1), block=BLOCK_THREADS, smem=0, stream=stream)

    import json  # EXPERIMENT
    import os  # EXPERIMENT

    launch.compile_hints = {
        "fast_fp_math": True,
        "unsafe_fp_math": True,
        "llvm_options": json.loads(os.environ.get("KV_LLVM", "{}")),
    }
    return launch


@functools.lru_cache(maxsize=None)
def _num_blocks(device) -> int:
    # One block per CU (one wave per SIMD), a multiple of the 8 XCDs.
    return torch.cuda.get_device_properties(device).multi_processor_count // 8 * 8


def hstu_ragged_bwd_kv(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    row_scale: torch.Tensor,
    N: int,
    alpha: float,
    spec: MaskSpec,
    max_attn_len: int,
    contextual_seq_len: int,
    full_attn_size: int,
) -> None:
    """SiLU heads, contiguous [T, H, D] q/k/v/dout/dq/dk/dv; ``row_scale`` is
    the per-row output scale as [H, T] f32."""
    from mslk.flydsl.jit import run_compiled

    from .ragged_bwd import _arch

    num_tokens, num_heads, dim = q.shape
    num_seqs = seq_offsets.shape[0] - 1
    launcher = build_ragged_bwd_kv(
        dim, "bf16" if q.dtype == torch.bfloat16 else "f16", spec, _arch()
    )
    dq_f32 = torch.zeros(
        (num_tokens * num_heads, dim), dtype=torch.float32, device=q.device
    )
    seqs = seq_offsets.to(torch.int32)
    run_compiled(
        launcher,
        q,
        k,
        v,
        dout,
        dk,
        dv,
        dq_f32,
        row_scale.view(-1),
        _PROF if _PROF is not None else row_scale.view(-1),
        torch.zeros(8, dtype=torch.int32, device=q.device),
        seqs,
        num_targets.to(torch.int32) if num_targets is not None else seqs,
        num_seqs,
        num_heads,
        (N + BLOCK_N - 1) // BLOCK_N,
        num_tokens,
        float(alpha),
        max_attn_len,
        contextual_seq_len,
        full_attn_size,
        _num_blocks(q.device),
        torch.cuda.current_stream(q.device),
    )
    dq.copy_(dq_f32.view(num_tokens, num_heads, dim))
