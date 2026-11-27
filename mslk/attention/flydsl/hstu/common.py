# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""Mask, position-remap and KV-tile-range helpers shared by the FlyDSL HSTU kernels.

Semantics follow :func:`mslk.attention.hstu._torch_ref.ragged_hstu_valid_mask`.
Helpers are written against :class:`Ops` so they run on host ints (:data:`HOST`)
and on FlyDSL ``Int32`` (:func:`device_ops`). Features are compile-time
(:class:`MaskSpec`); their values are run-time, and a value of 0 turns a
feature off.
"""

from dataclasses import dataclass
from typing import Any, Callable, Tuple

# Sequences must be shorter than this so id +/- _UNBOUNDED cannot overflow i32.
_UNBOUNDED = 1 << 30

LOWER = "lower_triangular"
UPPER = "upper_triangular"
NONE = "none"
MASK_TYPES = (LOWER, UPPER, NONE)


@dataclass(frozen=True)
class MaskSpec:
    """Compile-time mask features. Values are passed separately at run time."""

    mask_type: str = LOWER
    has_window: bool = False
    has_contextual: bool = False
    has_targets: bool = False
    has_full_attn: bool = False

    def __post_init__(self) -> None:
        if self.mask_type not in MASK_TYPES:
            raise ValueError(f"unknown mask type {self.mask_type!r}")
        if self.has_full_attn and not (self.mask_type == LOWER and self.has_window):
            raise ValueError("has_full_attn requires a windowed lower_triangular mask")
        if self.mask_type == UPPER and self.has_window:
            raise ValueError("upper_triangular does not take a window")

    @classmethod
    def from_args(
        cls,
        invalid_attn_mask_type: str,
        max_attn_len: int,
        contextual_seq_len: int,
        full_attn_size: int,
        has_targets: bool,
    ) -> "MaskSpec":
        """Features present in one call; ``full_attn_size`` needs a window."""
        has_window = max_attn_len > 0
        return cls(
            mask_type=invalid_attn_mask_type,
            has_window=has_window,
            has_contextual=contextual_seq_len > 0,
            has_targets=has_targets,
            has_full_attn=(
                full_attn_size > 0 and has_window and invalid_attn_mask_type == LOWER
            ),
        )


@dataclass(frozen=True)
class Ops:
    """The arithmetic the helpers need: lift an int, select, min, max."""

    const: Callable[[int], Any]
    select: Callable[[Any, Any, Any], Any]

    def min(self, a: Any, b: Any) -> Any:
        return self.select(a < b, a, b)

    def max(self, a: Any, b: Any) -> Any:
        return self.select(a > b, a, b)

    def cdiv(self, a: Any, b: int) -> Any:
        return (a + self.const(b - 1)) // self.const(b)


HOST = Ops(const=int, select=lambda c, a, b: a if c else b)


def device_ops() -> Ops:
    """:class:`Ops` over FlyDSL ``Int32``; call inside a kernel body."""
    import flydsl.expr as fx  # pyre-ignore[21]

    def lift(x: Any) -> Any:
        return fx.Int32(x) if isinstance(x, int) else x

    return Ops(const=fx.Int32, select=lambda c, a, b: c.select(lift(a), lift(b)))


@dataclass(frozen=True)
class SeqMask:
    """One sequence's mask state. Build with :func:`seq_mask`."""

    spec: MaskSpec
    ops: Ops
    seq_len: Any
    max_ids: Any
    # _UNBOUNDED when the window is off.
    max_attn_len: Any
    contextual_seq_len: Any
    ctx_shift: Any
    # None when not compiled in.
    ctx_on: Any
    # -_UNBOUNDED when the rule is off.
    full_attn_size: Any

    def to_id(self, raw: Any) -> Any:
        """Raw position -> id: contextual prefix to 0, targets clamp to ``max_ids``."""
        o, pos = self.ops, raw
        if self.spec.has_contextual:
            pos = o.max(pos - self.ctx_shift, o.const(0))
        if self.spec.has_targets:
            pos = o.min(pos, self.max_ids)
        return pos

    def _first_raw_with_id(self, ident: Any) -> Any:
        """Smallest raw position whose id is >= ``ident`` (``ident <= max_ids``)."""
        o = self.ops
        if not self.spec.has_contextual:
            return o.max(ident, o.const(0))
        return o.select(ident > o.const(0), ident + self.ctx_shift, o.const(0))

    def visible(self, m_raw: Any, n_raw: Any, m_id: Any, n_id: Any) -> Any:
        """True where query ``m`` sees key ``n`` (both in-sequence)."""
        spec, o = self.spec, self.ops
        if spec.mask_type == LOWER:
            vis = n_id < m_id
            if spec.has_window:
                vis = vis & (n_id >= m_id - self.max_attn_len)
                if spec.has_full_attn:
                    full_row = m_id >= self.max_ids - self.full_attn_size
                    vis = vis | (full_row & (n_id < m_id))
        elif spec.mask_type == UPPER:
            vis = n_id > m_id
        else:
            vis = n_id < self.max_ids
            if spec.has_window:
                d = n_id - m_id
                vis = (
                    vis
                    & (d <= self.max_attn_len)
                    & (d >= o.const(0) - self.max_attn_len)
                )
        if spec.has_contextual:
            ctx = ((m_id == o.const(0)) & (n_id < self.max_ids)) | (n_id == o.const(0))
            vis = vis | (ctx & self.ctx_on)
        return vis | (m_raw == n_raw)

    def tile_needs_mask(
        self, row0: Any, row_last: Any, col0: Any, col_last: Any
    ) -> Any:
        """False only when every key in ``[col0, col_last]`` is visible to every
        row in ``[row0, row_last]``; uniform across the rows."""
        spec = self.spec
        need = (col_last >= self.seq_len) | (self.to_id(col_last) >= self.to_id(row0))
        if spec.has_window:
            outside = self.to_id(col0) < self.to_id(row_last) - self.max_attn_len
            if spec.has_full_attn:
                outside = outside & (
                    self.to_id(row0) < self.max_ids - self.full_attn_size
                )
            need = need | outside
        if spec.has_contextual:
            need = need | self.ctx_on
        if spec.mask_type == UPPER:
            need = need | (col_last >= col0)
        return need

    def tile_hidden(self, row0: Any, row_last: Any, col0: Any, col_last: Any) -> Any:
        """True only when no key in ``[col0, col_last]`` is visible to any row in
        ``[row0, row_last]`` (lower-triangular only); uniform across the rows."""
        spec, o = self.spec, self.ops
        if spec.mask_type != LOWER:
            return col0 > col_last  # never
        hidden = (col0 > row_last) & (self.to_id(col0) >= self.to_id(row_last))
        if spec.has_window:
            before = self.to_id(col_last) < self.to_id(row0) - self.max_attn_len
            before = before & (col_last < row0)
            if spec.has_full_attn:
                before = before & (
                    self.to_id(row_last) < self.max_ids - self.full_attn_size
                )
            hidden = hidden | before
        if spec.has_contextual:
            hidden = hidden & (self.contextual_seq_len <= o.const(0))
        return hidden

    def kv_tiles(
        self, q_start: Any, block_m: int, block_n: int
    ) -> Tuple[Any, Any, Any]:
        """KV tiles a query tile must visit: ``[0, prefix_end)`` (contextual
        columns) then ``[main_start, main_end)``, with ``prefix_end <= main_start``."""
        spec, o = self.spec, self.ops
        seq_len = self.seq_len
        q_end = o.min(q_start + o.const(block_m), seq_len)
        active = q_start < seq_len
        ctx_row = q_start < self.contextual_seq_len if spec.has_contextual else None

        lo = o.const(0)
        hi = seq_len
        if spec.mask_type == LOWER:
            hi = q_end
            if spec.has_window:
                lo = self._first_raw_with_id(self.to_id(q_start) - self.max_attn_len)
                if spec.has_full_attn:
                    last_id = self.to_id(q_end - o.const(1))
                    full = last_id >= self.max_ids - self.full_attn_size
                    lo = o.select(full, o.const(0), lo)
        elif spec.mask_type == UPPER:
            lo = q_start
        elif spec.has_window:
            lo = self._first_raw_with_id(self.to_id(q_start) - self.max_attn_len)
            hi_id = self.to_id(q_end - o.const(1)) + self.max_attn_len
            hi = o.select(
                hi_id >= self.max_ids,
                seq_len,
                self._first_raw_with_id(hi_id + o.const(1)),
            )
            hi = o.min(o.max(hi, q_end), seq_len)
        if ctx_row is not None:
            lo = o.select(ctx_row, o.const(0), lo)
            hi = o.select(ctx_row, seq_len, hi)

        main_start = lo // o.const(block_n)
        main_end = o.select(active, o.cdiv(hi, block_n), o.const(0))
        main_start = o.min(main_start, main_end)
        prefix_end = o.const(0)
        if spec.has_contextual and (spec.has_window or spec.mask_type == UPPER):
            ctx_tiles = o.cdiv(o.min(self.contextual_seq_len, seq_len), block_n)
            prefix_end = o.min(ctx_tiles, main_start)
            prefix_end = o.select(active, prefix_end, o.const(0))
        return prefix_end, main_start, main_end

    def q_tiles(
        self, k_start: Any, block_n: int, block_m: int
    ) -> Tuple[Any, Any, Any, Any, Any]:
        """Query tiles that see some key of the KV tile at ``k_start``:
        ``[0, prefix_end)`` (contextual rows), ``[main_start, main_end)`` and
        ``[tail_start, tail_end)`` (full-attention rows), ordered and disjoint."""
        spec, o = self.spec, self.ops
        seq_len = self.seq_len
        all_tiles = o.cdiv(seq_len, block_m)
        zero = o.const(0)
        k_last = o.min(k_start + o.const(block_n), seq_len) - o.const(1)
        active = k_start < seq_len
        if spec.mask_type != LOWER:
            end = o.select(active, all_tiles, zero)
            if spec.mask_type == UPPER and not spec.has_contextual:
                end = o.select(active, o.cdiv(k_last + o.const(1), block_m), zero)
            return zero, zero, end, end, end
        hi = seq_len
        if spec.has_window:
            reach = self.to_id(k_last) + self.max_attn_len
            hi = o.select(
                reach >= self.max_ids,
                seq_len,
                o.min(self._first_raw_with_id(reach + o.const(1)), seq_len),
            )
        main_start = k_start // o.const(block_m)
        main_end = o.cdiv(hi, block_m)
        prefix_end = zero
        tail_start = all_tiles
        if spec.has_contextual:
            ctx_rows = o.min(self.contextual_seq_len, seq_len)
            prefix_end = o.cdiv(ctx_rows, block_m)
            ctx_keys = k_start < self.contextual_seq_len
            main_start = o.select(ctx_keys, zero, main_start)
            main_end = o.select(ctx_keys, all_tiles, main_end)
        if spec.has_full_attn:
            first_full = self.max_ids - self.full_attn_size
            tail_row = o.select(
                first_full >= self.max_ids + o.const(1),
                seq_len,
                o.min(self._first_raw_with_id(o.max(first_full, zero)), seq_len),
            )
            tail_start = tail_row // o.const(block_m)
        main_end = o.select(active, main_end, zero)
        prefix_end = o.select(active, prefix_end, zero)
        main_start = o.min(o.max(main_start, prefix_end), main_end)
        tail_start = o.max(tail_start, main_end)
        tail_end = o.select(active, all_tiles, zero)
        tail_start = o.min(tail_start, tail_end)
        return prefix_end, main_start, main_end, tail_start, tail_end


def seq_mask(
    spec: MaskSpec,
    ops: Ops,
    seq_len: Any,
    num_targets: Any = 0,
    max_attn_len: Any = 0,
    contextual_seq_len: Any = 0,
    full_attn_size: Any = 0,
) -> SeqMask:
    """Mask state of one sequence; values of features not in ``spec`` are ignored."""
    o = ops
    zero = o.const(0)
    max_ids = seq_len
    ctx_shift, ctx_on = zero, None
    if spec.has_contextual:
        ctx_shift = o.max(contextual_seq_len - o.const(1), zero)
        ctx_on = contextual_seq_len > zero
        max_ids = max_ids - ctx_shift
    if spec.has_targets:
        max_ids = max_ids - num_targets
    window = o.const(_UNBOUNDED)
    if spec.has_window:
        window = o.select(max_attn_len > zero, max_attn_len, o.const(_UNBOUNDED))
    full = o.const(-_UNBOUNDED)
    if spec.has_full_attn:
        full = o.select(full_attn_size > zero, full_attn_size, o.const(-_UNBOUNDED))
    return SeqMask(
        spec=spec,
        ops=ops,
        seq_len=seq_len,
        max_ids=max_ids,
        max_attn_len=window,
        contextual_seq_len=contextual_seq_len if spec.has_contextual else zero,
        ctx_shift=ctx_shift,
        ctx_on=ctx_on,
        full_attn_size=full,
    )
