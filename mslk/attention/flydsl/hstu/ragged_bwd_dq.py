# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""HSTU backward dQ for SiLU heads on gfx950 (ragged or grouped), no atomics.

A block owns 128 query rows of one (sequence, head), 32 per wave, with Q and
dO in registers as MFMA B operands; it loops over the visible 32-key K/V tiles
in LDS: S^T = K Q^T, dP^T = V dO^T, dS^T = dP^T c silu'(alpha S^T), then
dQ^T += K^T dS^T with dS^T fed straight from the accumulators.
"""

import functools

import torch

from .common import MaskSpec

NUM_WAVES = 4
BLOCK_M = 32 * NUM_WAVES
BLOCK_N = 32
WARP_SIZE = 64
_LOG2E = 1.4426950408889634


@functools.lru_cache(maxsize=128)
def build_ragged_bwd_dq(
    dim_qk: int,
    dim_v: int,
    dtype_str: str,
    spec: MaskSpec,
    arch: str,
    grouped: bool = False,
):
    import flydsl.compiler as flyc  # pyre-ignore[21]
    import flydsl.expr as fx  # pyre-ignore[21]
    from flydsl._mlir import ir as _ir  # pyre-ignore[21]
    from flydsl._mlir.dialects import llvm as _llvm_d  # pyre-ignore[21]
    from flydsl.expr import const_expr, gpu, range_constexpr, rocdl  # pyre-ignore[21]
    from flydsl.expr.typing import T, Vector as Vec  # pyre-ignore[21]
    from flydsl.expr.utils.arith import _to_raw as _raw, ArithValue  # pyre-ignore[21]
    from mslk.attention.flydsl.flash_attn_utils import (
        _make_rebased_view,
        scf_if_dispatch,
    )
    from mslk.flydsl import buffer_ops

    from .common import device_ops, seq_mask

    DQK, DV = dim_qk, dim_v
    assert DQK % 32 == 0 and DV % 16 == 0
    MASK = (
        spec.mask_type,
        spec.has_window,
        spec.has_contextual,
        spec.has_targets,
        spec.has_full_attn,
    )
    GROUPED = grouped
    elem_dtype = fx.BFloat16 if dtype_str == "bf16" else fx.Float16
    QK_STEPS = DQK // 16  # S over d
    V_STEPS = DV // 16  # dP over d
    D_TILES = DQK // 32  # dQ d tiles
    # 32 B pad: the four rows of a transposing read hit disjoint banks.
    KS = DQK + 16
    VS = DV + 8  # LDS row stride of V
    BLOCK_THREADS = NUM_WAVES * WARP_SIZE
    K_GROUPS, V_GROUPS = DQK // 8, DV // 8  # 8-element groups per row
    K_PASSES = -(-(BLOCK_N * K_GROUPS) // BLOCK_THREADS)
    V_PASSES = -(-(BLOCK_N * V_GROUPS) // BLOCK_THREADS)

    # DMA path: unpadded LDS, 16 B chunk c of row r in slot c ^ (r % 8).
    DMA_BYTES = WARP_SIZE * 16
    DMA = (
        arch.startswith("gfx950")
        and DQK % 64 == 0
        and DV % 64 == 0
        and (BLOCK_N * DQK * 2) % (DMA_BYTES * NUM_WAVES) == 0
        and (BLOCK_N * DV * 2) % (DMA_BYTES * NUM_WAVES) == 0
    )
    NST = 2 if (not DMA or max(DQK, DV) > 128) else 3
    K_ROW = DQK if DMA else KS
    V_ROW = DV if DMA else VS
    K_INSTR = BLOCK_N * DQK * 2 // DMA_BYTES
    V_INSTR = BLOCK_N * DV * 2 // DMA_BYTES

    @fx.struct
    class SharedStorage:
        k: fx.Array[elem_dtype, NST * BLOCK_N * K_ROW, 16]
        v: fx.Array[elem_dtype, NST * BLOCK_N * V_ROW, 16]

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def hstu_bwd_dq_kernel(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        dout: fx.Tensor,
        dq_out: fx.Tensor,
        row_scale: fx.Tensor,
        seq_offsets: fx.Tensor,
        num_targets: fx.Tensor,
        num_seqs: fx.Int32,
        num_heads: fx.Int32,
        n_q_tiles: fx.Int32,
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

        # Grouped: num_targets holds grouped_bwd_metadata, n_q_tiles its slot count.
        bid = fx.Int32(gpu.block_idx.x)
        total = num_seqs * num_heads * n_q_tiles
        if const_expr(GROUPED):
            total = num_heads * n_q_tiles
        per_xcd = (total + fx.Int32(7)) // fx.Int32(8)
        order = (bid % fx.Int32(8)) * per_xcd + bid // fx.Int32(8)
        in_range = order < total
        order = in_range.select(order, fx.Int32(0))
        n_tgt = fx.Int32(0)
        if const_expr(GROUPED):
            head = order // n_q_tiles
            slot = order % n_q_tiles
            seq_raw = fx.Int32(num_targets[slot])
            in_range = in_range & (seq_raw < num_seqs)
            seq = in_range.select(seq_raw, fx.Int32(0))
            q_tile = slot - fx.Int32(num_targets[n_q_tiles + seq])
            settings = n_q_tiles + num_seqs + fx.Int32(1) + seq * fx.Int32(4)
            if const_expr(MASK[3]):
                n_tgt = fx.Int32(num_targets[settings])
            max_attn_len = fx.Int32(num_targets[settings + fx.Int32(1)])
            contextual_seq_len = fx.Int32(num_targets[settings + fx.Int32(2)])
            full_attn_size = fx.Int32(num_targets[settings + fx.Int32(3)])
        else:
            q_tile = order % n_q_tiles
            bh = order // n_q_tiles
            seq = bh // num_heads
            head = bh % num_heads
            if const_expr(MASK[3]):
                n_tgt = fx.Int32(num_targets[seq])
        seq_start = fx.Int32(seq_offsets[seq])
        seq_len = fx.Int32(seq_offsets[seq + fx.Int32(1)]) - seq_start
        seq_len = in_range.select(seq_len, fx.Int32(0))
        sm = seq_mask(
            MaskSpec(*MASK),
            ops,
            seq_len,
            num_targets=n_tgt,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
            full_attn_size=full_attn_size,
        )
        m0 = q_tile * fx.Int32(BLOCK_M)
        mw0 = m0 + wave * fx.Int32(32)

        # Reach sm only through closures: the AST rewriter would carry it as loop state.
        def _kv_tiles():
            return sm.kv_tiles(m0, BLOCK_M, BLOCK_N)

        def _to_id(x):
            return sm.to_id(x)

        def _visible(m_abs, m_id, n_abs):
            return sm.visible(m_abs, n_abs, m_id, _to_id(n_abs))

        def _hidden(nk0):
            return sm.tile_hidden(
                mw0, mw0 + fx.Int32(31), nk0, nk0 + fx.Int32(BLOCK_N - 1)
            )

        def _needs_mask(nk0):
            return sm.tile_needs_mask(
                mw0, mw0 + fx.Int32(31), nk0, nk0 + fx.Int32(BLOCK_N - 1)
            )

        def load8(t, layout, row, colgrp):
            ok = (row >= fx.Int32(0)) & (row < seq_len)
            safe = ok.select(seq_start + row, seq_start)
            sub = t[fx.Int64(safe), head, None]
            raw = fx.make_view(fx.get_iter(sub), layout)[colgrp, None].load()
            return ok.select(Vec(raw).ir_value(), zero8)

        row8_qk = fx.make_layout((DQK // 8, 8), (8, 1))
        row8_v = fx.make_layout((DV // 8, 8), (8, 1))
        row4_qk = fx.make_layout((DQK // 4, 4), (4, 1))

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        k_view = lds.k.view(
            fx.make_layout((NST * BLOCK_N, K_ROW // 8, 8), (K_ROW, 8, 1))
        )
        v_view = lds.v.view(
            fx.make_layout((NST * BLOCK_N, V_ROW // 8, 8), (V_ROW, 8, 1))
        )

        def _lds_addr(arr):
            return fx.Int32(
                ArithValue(_raw(fx.Int64(fx.ptrtoint(arr.ptr)))).trunci(T.i32)
            )

        k_base = _lds_addr(lds.k)
        v_base = _lds_addr(lds.v)

        # ds_read_b64_tr_b16: lane i of a 16-lane group gets column i of a 4x16 block.
        def tr4(addr, const_elems):
            ptr = _llvm_d.IntToPtrOp(
                _ir.Type.parse("!llvm.ptr<3>"),
                _raw(addr + fx.Int32(2 * const_elems)),
            ).result
            return rocdl.ds_read_tr16_b64(v4_type, ptr).result

        def tr8(addr, row_lo, row_hi, col):
            a = tr4(addr, row_lo * K_ROW + col)
            b = tr4(addr, row_hi * K_ROW + col)
            return Vec(a).shuffle(Vec(b), list(range(8))).ir_value()

        # A reads use the view: a raw-pointer LDS load waits on in-flight DMA.
        tr_j = fx.Int32(2) * (g16 % fx.Int32(2)) + (l16 % fx.Int32(4)) // fx.Int32(2)
        tr_lane_base = (h * fx.Int32(4) + l16 // fx.Int32(4)) * fx.Int32(K_ROW * 2) + (
            l16 % fx.Int32(2)
        ) * fx.Int32(8)
        tr_lane_off = [
            tr_lane_base
            + (fx.Int32(4) * (fx.Int32(dl) ^ h) + (tr_j ^ (l16 // fx.Int32(4))))
            * fx.Int32(16)
            for dl in range(2)
        ]

        a_swz = l32 % fx.Int32(8)

        def a_read(view, row0, s):
            chunk = (fx.Int32(2 * s) + h) ^ a_swz
            return view[row0 + l32, chunk, None].load().ir_value()

        k_lds_ptr = fx.to_llvm_ptr(lds.k.ptr)

        def tr4_gep(off, const_bytes):
            ptr = buffer_ops.get_element_ptr(
                k_lds_ptr, byte_offset=_raw(off), static_byte_offset=const_bytes
            )
            return rocdl.ds_read_tr16_b64(v4_type, ptr).result

        def kt_read(row0, dt, ks):
            # GEP off the LDS buffer (an inttoptr address makes LLVM wait on
            # every DMA in flight).
            off = row0 * fx.Int32(K_ROW * 2) + tr_lane_off[dt % 2]
            c = (dt // 2) * 128 + 16 * ks * K_ROW * 2
            lo = tr4_gep(off, c)
            hi = tr4_gep(off, c + 16 * K_ROW)
            return Vec(lo).shuffle(Vec(hi), list(range(8))).ir_value()

        kt_lane = (
            (h * fx.Int32(4) + l16 // fx.Int32(4)) * fx.Int32(K_ROW)
            + (g16 % fx.Int32(2)) * fx.Int32(16)
            + (l16 % fx.Int32(4)) * fx.Int32(4)
        ) * fx.Int32(2)

        # alpha is folded into c_lane: dQ carries alpha.
        m_lane = mw0 + l32
        q_regs, do_regs = [], []
        for s in range_constexpr(QK_STEPS):
            q_regs.append(load8(q, row8_qk, m_lane, fx.Int32(2 * s) + h))
        for s in range_constexpr(V_STEPS):
            do_regs.append(load8(dout, row8_v, m_lane, fx.Int32(2 * s) + h))
        m_safe = (m_lane < seq_len).select(m_lane, fx.Int32(0))
        c_lane = (
            fx.Float32(row_scale[fx.Int64(head * total_tokens + seq_start + m_safe)])
            * alpha
        )
        m_id = _to_id(m_lane)

        pre_end, main_start, main_end = _kv_tiles()
        n_tiles = pre_end + (main_end - main_start)

        def phys(t):
            return (t < pre_end).select(t, main_start + (t - pre_end))

        def tile_n0(t):
            t = (t < n_tiles).select(t, n_tiles - fx.Int32(1))
            return phys(t) * fx.Int32(BLOCK_N)

        # Descriptors bounded to the sequence: rows outside it read as zeros.
        buf_flags = fx.Int32(buffer_ops._get_buffer_flags())
        k_tok = num_heads * fx.Int32(DQK)
        v_tok = num_heads * fx.Int32(DV)

        def seq_div(t, tok, dim):
            return _make_rebased_view(
                fx.get_iter(t),
                (fx.Int64(seq_start) * fx.Int64(tok) + fx.Int64(head) * fx.Int64(dim))
                * fx.Int64(2),
                fx.Int64(seq_len) * fx.Int64(tok) * fx.Int64(2),
                fx.make_layout(seq_len * tok, fx.Int32(1)),
                _buf_flags_i32=buf_flags,
                _elem_ir=elem_dtype.ir_type,
            )

        k_div = seq_div(k, k_tok, DQK)
        v_div = seq_div(v, v_tok, DV)
        load_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_dtype)

        def bload(div, off):
            r = fx.make_rmem_tensor(8, elem_dtype)
            fx.copy_atom_call(load_atom, fx.slice(div, (None, off)), r)
            return fx.memref_load_vec(r)

        k_lane_row = [
            (fx.Int32(ps * BLOCK_THREADS) + tid) // fx.Int32(K_GROUPS)
            for ps in range(K_PASSES)
        ]
        v_lane_row = [
            (fx.Int32(ps * BLOCK_THREADS) + tid) // fx.Int32(V_GROUPS)
            for ps in range(V_PASSES)
        ]
        k_lane_off = [
            k_lane_row[ps] * k_tok
            + ((fx.Int32(ps * BLOCK_THREADS) + tid) % fx.Int32(K_GROUPS)) * fx.Int32(8)
            for ps in range(K_PASSES)
        ]
        v_lane_off = [
            v_lane_row[ps] * v_tok
            + ((fx.Int32(ps * BLOCK_THREADS) + tid) % fx.Int32(V_GROUPS)) * fx.Int32(8)
            for ps in range(V_PASSES)
        ]

        def load_tile(n0):
            ks = [bload(k_div, n0 * k_tok + k_lane_off[ps]) for ps in range(K_PASSES)]
            vs = [bload(v_div, n0 * v_tok + v_lane_off[ps]) for ps in range(V_PASSES)]
            return ks, vs

        dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS(128), 128)
        lds_ptr_ty = fx.PointerType.get(elem_dtype.ir_type, 2, 16)
        wave_uni = fx.Int32(rocdl.readfirstlane(T.i32, _raw(wave)))

        def dma_tile(div, lds_byte, row_elems, tok, n_instr, n0):
            cpr = row_elems // 8
            rows_per = WARP_SIZE // cpr
            for j in range_constexpr(n_instr // NUM_WAVES):
                n = fx.Int32(j * NUM_WAVES) + wave_uni
                row = n * fx.Int32(rows_per) + lane // fx.Int32(cpr)
                slot = lane % fx.Int32(cpr)
                src = (n0 + row) * tok + (slot ^ (row % fx.Int32(8))) * fx.Int32(8)
                dst = fx.make_view(
                    fx.inttoptr(lds_ptr_ty, lds_byte + n * fx.Int32(DMA_BYTES)),
                    fx.make_layout(1, 1),
                )
                fx.copy(dma_atom, fx.slice(div, (None, src)), dst)

        def dma_kv(stage, n0):
            dma_tile(
                k_div,
                k_base + stage * fx.Int32(BLOCK_N * DQK * 2),
                DQK,
                k_tok,
                K_INSTR,
                n0,
            )
            dma_tile(
                v_div,
                v_base + stage * fx.Int32(BLOCK_N * DV * 2),
                DV,
                v_tok,
                V_INSTR,
                n0,
            )

        VM_PER_TILE = (K_INSTR + V_INSTR) // NUM_WAVES

        def store_tile(buf, ks, vs):
            row0 = buf * fx.Int32(BLOCK_N)
            for ps in range_constexpr(K_PASSES):
                it = fx.Int32(ps * BLOCK_THREADS) + tid
                r = it // fx.Int32(K_GROUPS)
                if r < fx.Int32(BLOCK_N):
                    k_view[row0 + r, it % fx.Int32(K_GROUPS), None].store(Vec(ks[ps]))
            for ps in range_constexpr(V_PASSES):
                it = fx.Int32(ps * BLOCK_THREADS) + tid
                r = it // fx.Int32(V_GROUPS)
                if r < fx.Int32(BLOCK_N):
                    v_view[row0 + r, it % fx.Int32(V_GROUPS), None].store(Vec(vs[ps]))

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

        def tile_compute(row0, n0, dq_in):
            dq_in = list(dq_in)
            s_acc = zero16
            for s in range_constexpr(QK_STEPS):
                if const_expr(DMA):
                    ka = a_read(k_view, row0, s)
                else:
                    ka = k_view[row0 + l32, fx.Int32(2 * s) + h, None].load().ir_value()
                s_acc = mfma(ka, q_regs[s], s_acc)
            dp_acc = zero16
            for s in range_constexpr(V_STEPS):
                if const_expr(DMA):
                    va = a_read(v_view, row0, s)
                else:
                    va = v_view[row0 + l32, fx.Int32(2 * s) + h, None].load().ir_value()
                dp_acc = mfma(va, do_regs[s], dp_acc)

            # C layout: m = lane % 32, n = 8 (r // 4) + 4 h + r % 4.
            def elementwise(masked):
                ds_vals = []
                for r in range_constexpr(16):
                    n_abs = n0 + fx.Int32(8 * (r // 4) + r % 4) + h * fx.Int32(4)
                    x = fx.Float32(Vec(s_acc)[r]) * alpha
                    e = fx.exp2(x * fx.Float32(-_LOG2E), fastmath="afn")
                    sig = fx.Float32(
                        fx.rocdl.rcp(
                            fx.Float32.ir_type, (fx.Float32(1.0) + e).ir_value()
                        )
                    )
                    g = c_lane * sig * (fx.Float32(1.0) + x * (fx.Float32(1.0) - sig))
                    dsv = fx.Float32(Vec(dp_acc)[r]) * g
                    if const_expr(masked):
                        dsv = _visible(m_lane, m_id, n_abs).select(dsv, fx.Float32(0.0))
                    ds_vals.append(dsv)
                return [pack8(ds_vals, ks) for ks in range(2)]

            packed = scf_if_dispatch(
                _needs_mask(n0),
                lambda *_: elementwise(True),
                lambda *_: elementwise(False),
                result_names=("_ds0", "_ds1"),
                result_values=[zero8] * 2,
            )

            k_tile = k_base + row0 * fx.Int32(K_ROW * 2) + kt_lane
            for dt in range_constexpr(D_TILES):
                for ks in range_constexpr(2):
                    if const_expr(DMA):
                        kt = kt_read(row0, dt, ks)
                    else:
                        kt = tr8(k_tile, 16 * ks, 16 * ks + 8, 32 * dt)
                    dq_in[dt] = mfma(kt, packed[ks], dq_in[dt])

            return dq_in

        if const_expr(DMA):
            for st in range_constexpr(NST - 1):
                dma_kv(fx.Int32(st), tile_n0(fx.Int32(st)))
            init = [zero16 for _ in range(D_TILES)]
            for t_idx, acc in range(fx.Int32(0), n_tiles, fx.Int32(1), init=init):  # ty: ignore
                acc = list(acc) if isinstance(acc, (list, tuple)) else [acc]
                dq_acc = acc[:D_TILES]
                t_cur = fx.Int32(t_idx)
                n0 = phys(t_cur) * fx.Int32(BLOCK_N)
                # Tile t landed; the barrier also frees tile t - 1's stage.
                rocdl.s_waitcnt(vmcnt=VM_PER_TILE * (NST - 2))
                gpu.barrier()
                t_next = t_cur + fx.Int32(NST - 1)
                dma_kv(t_next % fx.Int32(NST), tile_n0(t_next))
                row0 = (t_cur % fx.Int32(NST)) * fx.Int32(BLOCK_N)
                dq_acc = scf_if_dispatch(
                    _hidden(n0),
                    lambda *_: list(dq_acc),
                    lambda *_: tile_compute(row0, n0, dq_acc),
                    result_names=tuple(f"_dq{i}" for i in range(D_TILES)),
                    result_values=list(dq_acc),
                )
                dq_acc = list(dq_acc) if isinstance(dq_acc, (list, tuple)) else [dq_acc]
                loop_out = yield dq_acc
            rocdl.s_waitcnt(vmcnt=0)
        else:
            ks0, vs0 = load_tile(tile_n0(fx.Int32(0)))
            store_tile(fx.Int32(0), ks0, vs0)
            ks1, vs1 = load_tile(tile_n0(fx.Int32(1)))
            gpu.barrier()

            init = [zero16 for _ in range(D_TILES)] + list(ks1) + list(vs1)
            for t_idx, acc in range(fx.Int32(0), n_tiles, fx.Int32(1), init=init):  # ty: ignore
                acc = list(acc) if isinstance(acc, (list, tuple)) else [acc]
                dq_acc = acc[:D_TILES]
                ks_n = acc[D_TILES : D_TILES + K_PASSES]
                vs_n = acc[D_TILES + K_PASSES :]
                t_cur = fx.Int32(t_idx)
                n0 = phys(t_cur) * fx.Int32(BLOCK_N)
                buf = t_cur % fx.Int32(2)
                row0 = buf * fx.Int32(BLOCK_N)
                ks_nn, vs_nn = load_tile(tile_n0(t_cur + fx.Int32(2)))

                dq_acc = scf_if_dispatch(
                    _hidden(n0),
                    lambda *_: list(dq_acc),
                    lambda *_: tile_compute(row0, n0, dq_acc),
                    result_names=tuple(f"_dq{i}" for i in range(D_TILES)),
                    result_values=list(dq_acc),
                )
                dq_acc = list(dq_acc) if isinstance(dq_acc, (list, tuple)) else [dq_acc]

                store_tile(fx.Int32(1) - buf, ks_n, vs_n)
                gpu.barrier()
                loop_out = yield dq_acc + list(ks_nn) + list(vs_nn)

        res = list(loop_out) if isinstance(loop_out, (list, tuple)) else [loop_out]
        dq_fin = res[:D_TILES]
        # Lane m = lane % 32, d = 32 dt + 8 g + 4 h + e.
        if m_lane < seq_len:
            row = fx.Int64(seq_start + m_lane)
            for dt in range_constexpr(D_TILES):
                for gq in range_constexpr(4):
                    grp4 = fx.Int32(dt * 8 + 2 * gq) + h
                    vals = [fx.Float32(Vec(dq_fin[dt])[4 * gq + e]) for e in range(4)]
                    fx.make_view(fx.get_iter(dq_out[row, head, None]), row4_qk)[
                        grp4, None
                    ].store(Vec.from_elements(vals, fx.Float32).to(elem_dtype))

    @flyc.jit
    def launch(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        dout: fx.Tensor,
        dq_out: fx.Tensor,
        row_scale: fx.Tensor,
        seq_offsets: fx.Tensor,
        num_targets: fx.Tensor,
        num_seqs: fx.Int32,
        num_heads: fx.Int32,
        n_q_tiles: fx.Int32,
        total_tokens: fx.Int32,
        alpha: fx.Float32,
        max_attn_len: fx.Int32,
        contextual_seq_len: fx.Int32,
        full_attn_size: fx.Int32,
        stream: fx.Stream,
    ) -> None:
        total = num_seqs * num_heads * n_q_tiles
        if const_expr(GROUPED):
            total = num_heads * n_q_tiles
        grid = (total + fx.Int32(7)) // fx.Int32(8) * fx.Int32(8)
        hstu_bwd_dq_kernel(
            q,
            k,
            v,
            dout,
            dq_out,
            row_scale,
            seq_offsets,
            num_targets,
            num_seqs,
            num_heads,
            n_q_tiles,
            total_tokens,
            alpha,
            max_attn_len,
            contextual_seq_len,
            full_attn_size,
            value_attrs={
                "rocdl.flat_work_group_size": f"{BLOCK_THREADS},{BLOCK_THREADS}",
            },
        ).launch(grid=(grid, 1, 1), block=BLOCK_THREADS, smem=0, stream=stream)

    launch.compile_hints = {"fast_fp_math": True, "unsafe_fp_math": True}
    return launch


def dq_runner(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    row_scale: torch.Tensor,
    seq_offsets: torch.Tensor,
    meta: torch.Tensor,
    num_tiles: int,
    alpha: float,
    spec: MaskSpec,
    arch: str,
    mask_ints=(0, 0, 0),
    grouped: bool = False,
):
    """Return ``run(dq_out)`` writing dQ into contiguous ``dq_out``. Ragged:
    ``meta`` is int32 num_targets, ``num_tiles`` query tiles per sequence.
    Grouped: ``meta`` is grouped_bwd_metadata at BLOCK_M, ``num_tiles`` its
    slot count."""
    from mslk.flydsl.jit import run_compiled

    num_tokens, num_heads, dim_qk = q.shape
    launcher = build_ragged_bwd_dq(
        dim_qk,
        v.shape[2],
        "bf16" if q.dtype == torch.bfloat16 else "f16",
        spec,
        arch,
        grouped,
    )
    seqs = seq_offsets.to(torch.int32)
    tensors = [t.contiguous() for t in (q, k, v, dout)]

    def run(dq_out: torch.Tensor) -> None:
        run_compiled(
            launcher,
            *tensors,
            dq_out,
            row_scale.view(-1),
            seqs,
            meta,
            seq_offsets.shape[0] - 1,
            num_heads,
            num_tiles,
            num_tokens,
            float(alpha),
            *mask_ints,
            torch.cuda.current_stream(q.device),
        )

    return run
