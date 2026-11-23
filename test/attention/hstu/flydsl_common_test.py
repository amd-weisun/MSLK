# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""Host checks of the FlyDSL HSTU mask helpers against T1's mask.

``mslk.attention.flydsl.hstu.common`` runs the same code on host integers and
inside kernels, so these exhaustive checks over small sequences cover the mask
and tile-range logic the kernels use. They need no GPU and no FlyDSL.
"""

import itertools

import pytest
import torch
from mslk.attention.flydsl.hstu.common import (
    HOST,
    LOWER,
    MaskSpec,
    NONE,
    seq_mask,
    UPPER,
)
from mslk.attention.hstu._torch_ref import ragged_hstu_valid_mask

# (mask type, max_attn_len, contextual_seq_len, full_attn_size): every
# combination T1 accepts, over small values.
_OPTIONS = (
    [(LOWER, w, c, f) for w in (0, 1, 3, 6) for c in (0, 1, 2, 5) for f in (0, 1, 4)]
    + [(UPPER, 0, 0, 0)]
    + [(NONE, w, 0, 0) for w in (0, 1, 3, 6)]
)
_LENGTHS = (0, 1, 2, 5, 9, 17)


def _cases():
    for (mask, w, c, f), n in itertools.product(_OPTIONS, _LENGTHS):
        for t in (None, 0, 1, 3):
            if t is not None and t > n:
                continue
            yield mask, w, c, f, n, t


def _kernel_mask(spec, sm, n):
    out = torch.zeros(n, n, dtype=torch.bool)
    for m in range(n):
        for k in range(n):
            out[m, k] = bool(sm.visible(m, k, sm.to_id(m), sm.to_id(k)))
    return out


def _host_seq_mask(mask, w, c, f, n, t, spec=None):
    spec = spec or MaskSpec.from_args(mask, w, c, f, has_targets=t is not None)
    return spec, seq_mask(
        spec,
        HOST,
        n,
        num_targets=t or 0,
        max_attn_len=w,
        contextual_seq_len=c,
        full_attn_size=f,
    )


@pytest.mark.parametrize("mask,w,c,f,n,t", list(_cases()))
def test_visible_matches_reference(mask, w, c, f, n, t):
    spec, sm = _host_seq_mask(mask, w, c, f, n, t)
    ref = ragged_hstu_valid_mask(n, mask, t, w, c, f)
    assert torch.equal(_kernel_mask(spec, sm, n), ref)


@pytest.mark.parametrize("mask,w,c,f,n,t", list(_cases()))
def test_zero_values_turn_compiled_features_off(mask, w, c, f, n, t):
    """The grouped kernel compiles a feature in when any group uses it, and
    passes 0 for the others; 0 must then match T1 with the feature off."""
    if mask == UPPER:
        return
    spec = MaskSpec(
        mask_type=mask,
        has_window=True,
        has_contextual=mask == LOWER,
        # Targets are compiled in exactly when num_targets is passed, which
        # differs from 0 targets in T1 when a sequence is shorter than its
        # contextual prefix.
        has_targets=t is not None,
        has_full_attn=mask == LOWER,
    )
    _, sm = _host_seq_mask(mask, w, c, f, n, t, spec=spec)
    ref = ragged_hstu_valid_mask(n, mask, t, w, c, f)
    assert torch.equal(_kernel_mask(spec, sm, n), ref)


@pytest.mark.parametrize("block_m,block_n", [(1, 1), (2, 3), (4, 2), (8, 4), (16, 16)])
def test_kv_tiles_cover_every_visible_key(block_m, block_n):
    """Every visible key of a query tile lies in the tiles it visits, and the
    tiles it skips hold no visible key."""
    skipped = 0
    for mask, w, c, f, n, t in _cases():
        spec, sm = _host_seq_mask(mask, w, c, f, n, t)
        ref = ragged_hstu_valid_mask(n, mask, t, w, c, f)
        for q_start in range(0, n + block_m, block_m):
            pre, lo, hi = sm.kv_tiles(q_start, block_m, block_n)
            assert 0 <= pre <= lo <= hi, (pre, lo, hi)
            if q_start >= n:
                assert hi == 0
                continue
            visited = torch.zeros(n, dtype=torch.bool)
            for tile in itertools.chain(range(pre), range(lo, hi)):
                visited[tile * block_n : (tile + 1) * block_n] = True
            rows = ref[q_start : q_start + block_m].any(dim=0)
            missing = rows & ~visited
            assert not missing.any(), (mask, w, c, f, n, t, q_start, pre, lo, hi)
            skipped += int((~visited).sum())
    # The ranges must actually skip work somewhere (windows, causal).
    assert skipped > 0


@pytest.mark.parametrize("block_m,block_n", [(1, 1), (2, 3), (4, 2), (8, 4), (16, 16)])
def test_tile_needs_mask_is_safe(block_m, block_n):
    """A tile declared unmasked has every key visible to every row (lower
    triangular, no contextual prefix), and the check does skip tiles,
    including full-attention blocks outside their window."""
    skipped = skipped_full = 0
    for mask, w, c, f, n, t in _cases():
        if mask != LOWER or c:
            continue
        spec, sm = _host_seq_mask(mask, w, c, f, n, t)
        ref = ragged_hstu_valid_mask(n, mask, t, w, c, f)
        for r0 in range(0, n, block_m):
            r1 = min(r0 + block_m, n) - 1
            for c0 in range(0, n + block_n, block_n):
                c1 = c0 + block_n - 1
                if sm.tile_needs_mask(r0, r1, c0, c1):
                    continue
                assert c1 < n and ref[r0 : r1 + 1, c0 : c1 + 1].all(), (
                    mask,
                    w,
                    c,
                    f,
                    n,
                    t,
                    r0,
                    c0,
                )
                skipped += 1
                if spec.has_full_attn and sm.to_id(c0) < sm.to_id(r1) - w:
                    skipped_full += 1
    assert skipped > 0 and skipped_full > 0
