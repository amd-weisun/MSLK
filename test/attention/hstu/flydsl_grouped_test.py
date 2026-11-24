# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""The FlyDSL grouped ragged HSTU forward against T1's grouped reference."""

import pytest
import torch
from mslk.attention.hstu._torch_ref import grouped_ragged_hstu_attention_ref

from .hstu_test_utils import assert_hstu_close, lengths_to_offsets, make_grouped_inputs

_CUDA = torch.device("cuda")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="needs a ROCm GPU",
)


def _settings(inputs, windows=None, contexts=None, fulls=None):
    G = inputs["max_seq_len_tensor"].numel()
    for name, values in (
        ("max_attn_len_tensor", windows),
        ("contextual_seq_len_tensor", contexts),
        ("full_attn_size_tensor", fulls),
    ):
        if values is not None:
            assert len(values) == G
            inputs[name] = torch.tensor(values, device=_CUDA)
    return inputs


def _ref_args(inputs, attn_scale=None):
    """T1 takes a per-token scale as [T]; T2's grouped API as [T, ...]."""
    args = dict(inputs)
    if attn_scale is not None and attn_scale.dim() > 1:
        attn_scale = attn_scale.reshape(attn_scale.shape[0], -1)[:, 0]
    args["attn_scale_tensor"] = attn_scale
    return args


def _check(inputs, attn_scale=None, via_op=False):
    args = dict(inputs, attn_scale_tensor=attn_scale)
    if via_op:
        out = torch.ops.mslk.hstu_grouped_fwd(**args)
    else:
        from mslk.attention.flydsl.hstu.grouped_fwd import hstu_grouped_fwd

        out = hstu_grouped_fwd(**args)
    ref = grouped_ragged_hstu_attention_ref(**_ref_args(inputs, attn_scale))
    assert_hstu_close(out, ref, "silu")
    return out


@pytest.mark.parametrize("dim", [64, 128, 256])
def test_groups_with_different_settings(dim):
    inputs = make_grouped_inputs(
        3, [200, 64, 330], 2, dim, dim, 0.6, device=_CUDA, seed=1, max_targets=10
    )
    _settings(inputs, windows=[0, 20, 7], contexts=[5, 0, 12], fulls=[0, 9, 30])
    _check(inputs)


@pytest.mark.parametrize("mask", ["upper_triangular", "none"])
def test_mask_types(mask):
    inputs = make_grouped_inputs(2, [150, 90], 2, 64, 64, 0.6, device=_CUDA, seed=2)
    inputs["invalid_attn_mask_type"] = mask
    # T1: no contextual length or full attention off lower-triangular masks,
    # and no window on upper-triangular ones.
    windows = [0, 0] if mask == "upper_triangular" else [0, 17]
    _settings(inputs, windows=windows)
    _check(inputs)


@pytest.mark.parametrize("mode", ["inv_n", "group", "token"])
def test_scale_modes(mode):
    inputs = make_grouped_inputs(
        2, [180, 120, 60], 2, 128, 128, 0.6, device=_CUDA, seed=3
    )
    T = inputs["q"].shape[0]
    scale = {
        "inv_n": None,
        "group": torch.tensor([0.5, 2.0, 1.5], device=_CUDA),
        "token": torch.rand(T, 1, device=_CUDA) + 0.5,
    }[mode]
    _check(inputs, scale)


def test_single_group_and_through_the_op():
    inputs = make_grouped_inputs(5, [256], 3, 128, 128, 0.5, device=_CUDA, seed=4)
    _settings(inputs, windows=[33], contexts=[4], fulls=[10])
    _check(inputs, via_op=True)


def test_empty_groups_and_sequences():
    # Group 1 has only empty sequences; others mix empty and non-empty ones.
    lengths = torch.tensor([0, 37, 1, 0, 0, 0, 64, 0, 129])
    Z = 3
    total = int(lengths.sum())
    gen = torch.Generator().manual_seed(5)
    q, k, v = (
        (torch.rand(total, 2, 64, generator=gen) * 2 - 1).to(_CUDA, torch.bfloat16)
        for _ in range(3)
    )
    offsets = lengths_to_offsets(lengths, _CUDA)
    inputs = {
        "Z": Z,
        "q": q,
        "k": k,
        "v": v,
        "seq_offsets": offsets,
        "group_offsets": offsets[::Z].clone(),
        "num_targets": None,
        "max_seq_len_tensor": torch.tensor([64, 16, 160], device=_CUDA),
        "max_attn_len_tensor": torch.tensor([0, 5, 40], device=_CUDA),
        "full_attn_size_tensor": torch.zeros(3, dtype=torch.int64, device=_CUDA),
        "contextual_seq_len_tensor": torch.tensor([2, 0, 0], device=_CUDA),
        "alpha": 0.125,
        "invalid_attn_mask_type": "lower_triangular",
    }
    _check(inputs)


def test_no_tokens():
    from mslk.attention.flydsl.hstu.grouped_fwd import hstu_grouped_fwd

    inputs = make_grouped_inputs(2, [8, 8], 2, 64, 64, 0.5, device=_CUDA, seed=6)
    for name in ("q", "k", "v"):
        inputs[name] = inputs[name][:0]
    inputs["seq_offsets"] = torch.zeros(5, dtype=torch.int64, device=_CUDA)
    inputs["group_offsets"] = torch.zeros(3, dtype=torch.int64, device=_CUDA)
    out = hstu_grouped_fwd(**inputs)
    assert out.shape == (0, 2, 64)


def test_cuda_graph_capture():
    """No host synchronization: the tile map is built on the device."""
    from mslk.attention.flydsl.hstu.grouped_fwd import hstu_grouped_fwd

    inputs = make_grouped_inputs(
        3, [150, 300], 2, 128, 128, 0.6, device=_CUDA, seed=7, max_targets=5
    )
    _settings(inputs, windows=[0, 25], contexts=[3, 0], fulls=[0, 12])
    scale = torch.tensor([1.5, 0.75], device=_CUDA)
    args = dict(inputs, attn_scale_tensor=scale)
    hstu_grouped_fwd(**args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = hstu_grouped_fwd(**args)
    for name in ("q", "k", "v"):
        inputs[name].copy_(torch.rand_like(inputs[name]) * 2 - 1)
    scale.copy_(torch.tensor([0.5, 2.0], device=_CUDA))
    inputs["max_attn_len_tensor"].copy_(torch.tensor([10, 0], device=_CUDA))
    graph.replay()
    torch.cuda.synchronize()
    ref = grouped_ragged_hstu_attention_ref(**_ref_args(inputs, scale))
    assert_hstu_close(out, ref, "silu")


def test_rejects_bad_scale():
    from mslk.attention.flydsl.hstu.grouped_fwd import hstu_grouped_fwd

    inputs = make_grouped_inputs(2, [64, 64], 2, 64, 64, 0.5, device=_CUDA, seed=8)
    with pytest.raises(ValueError, match="per group"):
        hstu_grouped_fwd(**inputs, attn_scale_tensor=torch.ones(5, device=_CUDA))


_AOT_COMPILE = """
from mslk.attention.flydsl.hstu import ragged_fwd_aot as A
for c in A.AOT_CONFIGS:
    if (c["kernel"], c["head_dim"], c["mask"], c["scale_mode"]) == (
        "grouped", 64, (True, True, False, True), "inv_n"
    ):
        A.compile_aot_config(c, "gfx950")
"""

_AOT_RUN = """
import torch
from mslk.attention.flydsl.hstu.grouped_fwd import hstu_grouped_fwd
q = torch.rand(400, 3, 64, device="cuda", dtype=torch.bfloat16)
off = torch.tensor([0, 150, 160, 400], device="cuda", dtype=torch.int64)
g = lambda *x: torch.tensor(x, device="cuda")
hstu_grouped_fwd(
    1, q, q, q, off, off, None, g(150, 10, 300), g(0, 0, 40), g(0, 0, 8), g(3, 0, 0),
    0.1, "lower_triangular",
)
torch.cuda.synchronize()
"""


def test_aot_cache_serves_real_calls(tmp_path):
    """A grouped config pre-compiled under COMPILE_ONLY with CPU placeholders
    serves a real call (other head count) with JIT disabled."""
    import os
    import subprocess
    import sys

    from mslk.attention.flydsl.hstu.ragged_fwd import _arch

    if not _arch().startswith("gfx950"):
        pytest.skip("AOT configs target gfx950")
    env = dict(os.environ, FLYDSL_RUNTIME_CACHE_DIR=str(tmp_path))
    env.pop("FLYDSL_RUNTIME_RUN_ONLY", None)
    env.pop("FLYDSL_RUNTIME_ENABLE_CACHE", None)
    subprocess.run(
        [sys.executable, "-c", _AOT_COMPILE],
        env=dict(env, COMPILE_ONLY="1"),
        check=True,
    )
    subprocess.run(
        [sys.executable, "-c", _AOT_RUN],
        env=dict(env, FLYDSL_RUNTIME_RUN_ONLY="1"),
        check=True,
    )
