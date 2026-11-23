# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""The FlyDSL ragged HSTU forward against T1's reference
(``mslk.attention.hstu._torch_ref.ragged_hstu_attention_ref``)."""

import pytest
import torch
from mslk.attention.hstu._torch_ref import ragged_hstu_attention_ref
from mslk.flydsl.common import is_flydsl_available

from .hstu_test_utils import (
    assert_hstu_close,
    HEAD_DIMS,
    lengths_to_offsets,
    make_ragged_inputs,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.hip is None
    or not is_flydsl_available(),
    reason="needs a ROCm GPU and FlyDSL",
)

_CUDA = torch.device("cuda")

# Each mask feature alone, then combinations.
_MASKS = {
    "causal": {},
    "upper": {"invalid_attn_mask_type": "upper_triangular"},
    "window": {"max_attn_len": 37},
    "window_short": {"max_attn_len": 3},
    "contextual": {"contextual_seq_len": 11},
    "full_attn_alone": {"full_attn_size": 9},
    "window_contextual": {"max_attn_len": 21, "contextual_seq_len": 40},
    "window_full": {"max_attn_len": 16, "full_attn_size": 24},
    "window_contextual_full": {
        "max_attn_len": 30,
        "contextual_seq_len": 5,
        "full_attn_size": 50,
    },
}


def _run(inputs, **options):
    from mslk.attention.flydsl.hstu.ragged_fwd import hstu_ragged_fwd

    args = {
        "invalid_attn_mask_type": "lower_triangular",
        "num_targets": None,
        "attn_scale": None,
        "max_attn_len": 0,
        "contextual_seq_len": 0,
        "full_attn_size": 0,
    }
    args.update(options)
    call = dict(inputs)
    call.update(args)
    config = call.pop("config", None)
    out, _ = hstu_ragged_fwd(**call, config=config)
    call.pop("num_softmax_heads", None)
    ref, _ = ragged_hstu_attention_ref(**call)
    return out, ref


def _inputs(batch, max_len, heads=2, dim=64, hidden=None, sparsity=0.6, **kw):
    return make_ragged_inputs(
        batch,
        max_len,
        heads,
        dim,
        hidden or dim,
        sparsity,
        device=_CUDA,
        **kw,
    )


@pytest.mark.parametrize("mask", list(_MASKS))
@pytest.mark.parametrize("targets", [False, True])
def test_mask_features(mask, targets):
    inputs = _inputs(6, 300, max_targets=20 if targets else 0, seed=1)
    out, ref = _run(inputs, **_MASKS[mask])
    assert_hstu_close(out, ref, "silu")


@pytest.mark.parametrize(
    "dim,hidden",
    [(d, d) for d in HEAD_DIMS] + [(64, 128), (128, 64), (48, 96), (64, 160)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_head_dims(dim, hidden, dtype):
    inputs = _inputs(4, 200, heads=3, dim=dim, hidden=hidden, dtype=dtype, seed=2)
    out, ref = _run(inputs, max_attn_len=50, contextual_seq_len=3)
    assert_hstu_close(out, ref, "silu")


@pytest.mark.parametrize("form", ["scalar", "token"])
def test_attn_scale(form):
    inputs = _inputs(5, 160, seed=3)
    total = inputs["q"].shape[0]
    if form == "scalar":
        scale = torch.tensor(0.37, device=_CUDA)
    else:
        scale = torch.rand(total, device=_CUDA, dtype=torch.bfloat16) + 0.5
    out, ref = _run(inputs, attn_scale=scale)
    assert_hstu_close(out, ref, "silu")


def test_alpha_is_a_runtime_value():
    inputs = _inputs(3, 100, seed=4)
    for alpha in (0.05, 0.5, 2.0):
        inputs["alpha"] = alpha
        out, ref = _run(inputs)
        assert_hstu_close(out, ref, "silu")


@pytest.mark.parametrize(
    "lengths",
    [
        [0],
        [0, 0, 0],
        [1],
        [0, 5, 0, 1, 2],
        [127, 128, 129, 1, 255, 256, 257],
        [2048, 3, 900],
    ],
)
def test_lengths(lengths):
    gen = torch.Generator().manual_seed(5)
    lengths_t = torch.tensor(lengths)
    total = int(lengths_t.sum())
    q, k, v = (
        (torch.rand(total, 2, 64, generator=gen) * 2 - 1).to(_CUDA, torch.bfloat16)
        for _ in range(3)
    )
    inputs = {
        "N": max(max(lengths), 1),
        "alpha": 0.125,
        "q": q,
        "k": k,
        "v": v,
        "seq_offsets": lengths_to_offsets(lengths_t, _CUDA),
    }
    for options in (
        {},
        {"max_attn_len": 7},
        {"max_attn_len": 7, "contextual_seq_len": 3},
    ):
        out, ref = _run(inputs, **options)
        assert_hstu_close(out, ref, "silu")


def test_n_larger_than_longest_sequence():
    """N sets the default 1/N scale and the grid, not the lengths."""
    inputs = _inputs(4, 100, seed=6)
    inputs["N"] = 333
    out, ref = _run(inputs)
    assert_hstu_close(out, ref, "silu")


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_index_dtypes(index_dtype):
    inputs = _inputs(4, 120, max_targets=5, seed=7)
    inputs["seq_offsets"] = inputs["seq_offsets"].to(index_dtype)
    targets = inputs["num_targets"].to(index_dtype)
    out, ref = _run(inputs, num_targets=targets, max_attn_len=9)
    assert_hstu_close(out, ref, "silu")


def test_strided_views():
    """q, k and v slices of one packed qkv tensor, so token strides differ
    from the head-dim extent."""
    base = _inputs(5, 150, seed=8)
    total, heads, dim = base["q"].shape
    qkv = torch.rand(total, heads, 3 * dim + 16, device=_CUDA, dtype=torch.bfloat16)
    inputs = dict(base)
    inputs["q"] = qkv[:, :, :dim]
    inputs["k"] = qkv[:, :, dim : 2 * dim]
    inputs["v"] = qkv[:, :, 2 * dim + 16 :]
    out, ref = _run(inputs, max_attn_len=20)
    assert_hstu_close(out, ref, "silu")


def test_shared_kv():
    inputs = _inputs(4, 140, shared_kv=True, seed=9)
    out, ref = _run(inputs)
    assert_hstu_close(out, ref, "silu")


@pytest.mark.parametrize(
    "config",
    [(64, 32, 4, 2), (128, 32, 4, 0), (128, 64, 4, 2), (64, 16, 2, 0), (32, 32, 2, 2)],
)
def test_tile_configs(config):
    from mslk.attention.flydsl.hstu.ragged_fwd import FwdConfig

    inputs = _inputs(5, 260, max_targets=10, seed=10)
    inputs["config"] = FwdConfig(*config)
    for options in (
        {},
        {"max_attn_len": 33, "contextual_seq_len": 17, "full_attn_size": 20},
    ):
        out, ref = _run(inputs, **options)
        assert_hstu_close(out, ref, "silu")


def test_variants_do_not_share_a_compiled_kernel():
    """FlyDSL keys its compile cache on scalar closure values only. Variants
    that differ only in compile-time mask features must each get their own
    binary (a regression served the upper-triangular binary to a windowed
    lower-triangular call)."""
    inputs = _inputs(2, 64, seed=11)
    for options in (
        {"invalid_attn_mask_type": "upper_triangular"},
        {"max_attn_len": 1000},
        {},
        {"max_attn_len": 1000, "contextual_seq_len": 1},
    ):
        out, ref = _run(inputs, **options)
        assert_hstu_close(out, ref, "silu")


def test_unsupported_dims_are_declined():
    """No tile divides the load pass for hidden dim 80; the kernel must
    refuse rather than compute a wrong result."""
    inputs = _inputs(2, 32, dim=48, hidden=80, seed=12)
    with pytest.raises(NotImplementedError, match="FlyDSL ragged HSTU forward"):
        _run(inputs)
