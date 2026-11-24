# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""The FlyDSL grouped ragged HSTU backward against autograd on T1's grouped
reference."""

import pytest
import torch
from mslk.attention.hstu._torch_ref import grouped_ragged_hstu_attention_ref

from .hstu_test_utils import assert_hstu_close, lengths_to_offsets, make_grouped_inputs

_CUDA = torch.device("cuda")


def _gfx950() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is None:
        return False
    from mslk.attention.flydsl.hstu.ragged_bwd import _arch

    return _arch().startswith("gfx950")


pytestmark = pytest.mark.skipif(not _gfx950(), reason="needs gfx950")

_ARGS = (
    "Z",
    "seq_offsets",
    "group_offsets",
    "num_targets",
    "max_seq_len_tensor",
    "max_attn_len_tensor",
    "full_attn_size_tensor",
    "contextual_seq_len_tensor",
    "alpha",
    "invalid_attn_mask_type",
)


def _settings(inputs, windows=None, contexts=None, fulls=None):
    for name, values in (
        ("max_attn_len_tensor", windows),
        ("contextual_seq_len_tensor", contexts),
        ("full_attn_size_tensor", fulls),
    ):
        if values is not None:
            inputs[name] = torch.tensor(values, device=_CUDA)
    return inputs


def _ref_scale(attn_scale):
    """T1 takes a per-token scale as [T]; T2's grouped API as [T, ...]."""
    if attn_scale is not None and attn_scale.dim() > 1:
        return attn_scale.reshape(attn_scale.shape[0], -1)[:, 0]
    return attn_scale


def _reference_grads(inputs, dout, attn_scale=None, shared_kv=False):
    q, k, v = (inputs[n].detach().float().requires_grad_() for n in ("q", "k", "v"))
    if shared_kv:
        v = k
    args = {n: inputs.get(n) for n in _ARGS}
    out = grouped_ragged_hstu_attention_ref(
        q=q, k=k, v=v, attn_scale_tensor=_ref_scale(attn_scale), **args
    )
    out.backward(dout.float())
    return q.grad, k.grad, None if shared_kv else v.grad


def _assert_grads(got, want):
    for name, g, w in zip(("dq", "dk", "dv"), got, want):
        if w is not None:
            assert_hstu_close(g, w, "silu", name)


def _check(inputs, attn_scale=None, shared_kv=False):
    from mslk.attention.flydsl.hstu.grouped_bwd import hstu_grouped_bwd

    q, k, v = inputs["q"], inputs["k"], inputs["v"]
    gen = torch.Generator(device=_CUDA).manual_seed(0)
    dout = torch.randn(
        q.shape[0], q.shape[1], v.shape[2], device=_CUDA, generator=gen
    ).to(q.dtype)
    want = _reference_grads(inputs, dout, attn_scale, shared_kv)
    dq, dk = torch.empty_like(q), torch.empty_like(k)
    dv = None if shared_kv else torch.empty_like(v)
    args = {n: inputs.get(n) for n in _ARGS}
    hstu_grouped_bwd(
        dout,
        q=q,
        k=k,
        v=k if shared_kv else v,
        dq=dq,
        dk=dk,
        dv=dv,
        attn_scale_tensor=attn_scale,
        shared_kv=shared_kv,
        **args,
    )
    _assert_grads((dq, dk, dv), want)


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
    _settings(inputs, windows=[0, 0] if mask == "upper_triangular" else [0, 17])
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


def test_shared_kv():
    inputs = make_grouped_inputs(3, [160, 80], 2, 64, 64, 0.6, device=_CUDA, seed=4)
    _settings(inputs, windows=[12, 0])
    _check(inputs, shared_kv=True)


def test_empty_groups_and_sequences():
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


def test_through_autograd():
    """T2's grouped autograd: the FlyDSL grouped forward, then this backward."""
    import mslk.attention.hstu as api

    inputs = make_grouped_inputs(
        3, [150, 90], 2, 64, 64, 0.6, device=_CUDA, seed=6, max_targets=5
    )
    _settings(inputs, windows=[0, 11], contexts=[3, 0])
    q, k, v = (inputs[n].detach().clone().requires_grad_() for n in ("q", "k", "v"))
    out = api.grouped_ragged_hstu_attention(
        inputs["Z"],
        q,
        k,
        v,
        inputs["seq_offsets"],
        inputs["group_offsets"],
        inputs["num_targets"],
        inputs["max_seq_len_tensor"],
        inputs["max_attn_len_tensor"],
        inputs["full_attn_size_tensor"],
        inputs["contextual_seq_len_tensor"],
        inputs["alpha"],
        inputs["invalid_attn_mask_type"],
    )
    dout = torch.randn_like(out)
    out.backward(dout)
    want = _reference_grads(inputs, dout)
    _assert_grads((q.grad, k.grad, v.grad), want)


def test_cuda_graph_capture():
    """No host synchronization: replay with new lengths and settings."""
    from mslk.attention.flydsl.hstu.grouped_bwd import hstu_grouped_bwd

    inputs = make_grouped_inputs(
        3, [150, 300], 2, 64, 64, 0.6, device=_CUDA, seed=7, max_targets=5
    )
    _settings(inputs, windows=[0, 25], contexts=[3, 0], fulls=[0, 12])
    q, k, v = inputs["q"], inputs["k"], inputs["v"]
    dout = torch.randn_like(v)
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    args = {n: inputs.get(n) for n in _ARGS}

    def call():
        hstu_grouped_bwd(dout, q=q, k=k, v=v, dq=dq, dk=dk, dv=dv, **args)

    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for t in (q, k, v, dout):
        t.copy_(torch.rand_like(t) * 2 - 1)
    T = q.shape[0]
    lengths = torch.tensor([0, 1, T // 3, 0, T - T // 3 - 6, 5])
    offsets = lengths_to_offsets(lengths, _CUDA)
    inputs["seq_offsets"].copy_(offsets)
    inputs["group_offsets"].copy_(offsets[::3])
    inputs["max_seq_len_tensor"].copy_(torch.tensor([T // 3, T], device=_CUDA))
    inputs["max_attn_len_tensor"].copy_(torch.tensor([10, 0], device=_CUDA))
    inputs["num_targets"].copy_(torch.minimum(inputs["num_targets"], lengths.to(_CUDA)))
    graph.replay()
    torch.cuda.synchronize()
    _assert_grads((dq, dk, dv), _reference_grads(inputs, dout))
