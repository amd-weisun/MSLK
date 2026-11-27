# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""The FlyDSL ragged HSTU backward (SiLU heads) against autograd on the reference."""

import pytest
import torch
from mslk.attention.hstu._torch_ref import ragged_hstu_attention_ref

from .hstu_test_utils import assert_hstu_close, lengths_to_offsets, make_ragged_inputs

_CUDA = torch.device("cuda")


def _gfx950() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is None:
        return False
    from mslk.attention.flydsl.hstu.ragged_bwd import _arch

    return _arch().startswith("gfx950")


pytestmark = pytest.mark.skipif(not _gfx950(), reason="needs gfx950")

_MASKS = {
    "causal": {},
    "window": {"max_attn_len": 37},
    "window_short": {"max_attn_len": 3},
    "contextual": {"contextual_seq_len": 11},
    "window_contextual": {"max_attn_len": 21, "contextual_seq_len": 40},
    "window_full": {"max_attn_len": 16, "full_attn_size": 24},
    "window_contextual_full": {
        "max_attn_len": 30,
        "contextual_seq_len": 5,
        "full_attn_size": 50,
    },
}


def _args(inputs, **options):
    args = {
        "N": inputs["N"],
        "alpha": inputs["alpha"],
        "seq_offsets": inputs["seq_offsets"],
        "invalid_attn_mask_type": "lower_triangular",
        "num_targets": inputs.get("num_targets"),
        "attn_scale": None,
        "max_attn_len": 0,
        "contextual_seq_len": 0,
        "full_attn_size": 0,
    }
    args.update(options)
    return args


def _reference_grads(inputs, dout, args, shared_kv=False):
    q, k, v = (inputs[n].detach().float().requires_grad_() for n in ("q", "k", "v"))
    if shared_kv:
        v = k
    out, _ = ragged_hstu_attention_ref(q=q, k=k, v=v, **args)
    out.backward(dout.float())
    return q.grad, k.grad, None if shared_kv else v.grad


def _check(inputs, shared_kv=False, sort_idx=None, **options):
    from mslk.attention.flydsl.hstu.ragged_bwd import hstu_ragged_bwd

    args = _args(inputs, **options)
    q, k, v = inputs["q"], inputs["k"], inputs["v"]
    gen = torch.Generator(device=_CUDA).manual_seed(0)
    dout = torch.randn(
        q.shape[0], q.shape[1], v.shape[2], device=_CUDA, generator=gen
    ).to(q.dtype)
    ref = _reference_grads(inputs, dout, args, shared_kv)
    dq, dk = torch.empty_like(q), torch.empty_like(k)
    dv = None if shared_kv else torch.empty_like(v)
    hstu_ragged_bwd(
        dout,
        q,
        k,
        k if shared_kv else v,
        dq,
        dk,
        dv,
        args["seq_offsets"],
        args["num_targets"],
        args["attn_scale"],
        args["N"],
        args["alpha"],
        args["max_attn_len"],
        args["invalid_attn_mask_type"],
        args["contextual_seq_len"],
        sort_idx,
        args["full_attn_size"],
        shared_kv=shared_kv,
    )
    for name, got, want in (("dq", dq, ref[0]), ("dk", dk, ref[1]), ("dv", dv, ref[2])):
        if want is not None:
            assert_hstu_close(got, want, "silu", name)


@pytest.mark.parametrize("mask", list(_MASKS))
@pytest.mark.parametrize("targets", [False, True])
def test_mask_features(mask, targets):
    inputs = make_ragged_inputs(
        6, 300, 2, 64, 64, 0.6, device=_CUDA, seed=1, max_targets=20 if targets else 0
    )
    _check(inputs, **_MASKS[mask])


@pytest.mark.parametrize("mask_type", ["upper_triangular", "none"])
def test_mask_types(mask_type):
    inputs = make_ragged_inputs(4, 200, 2, 64, 64, 0.6, device=_CUDA, seed=2)
    options = {"invalid_attn_mask_type": mask_type}
    if mask_type == "none":
        options["max_attn_len"] = 25
    _check(inputs, **options)


@pytest.mark.parametrize(
    "dim,hidden",
    [
        (32, 32),
        (64, 64),
        (128, 128),
        (256, 256),
        (64, 128),
        (128, 64),
        (32, 64),
        (64, 256),
        (128, 256),
        (80, 128),
        # The fused kernel at D 64 / 256 for SiLU heads (Dqk not a multiple
        # of 32, or Dv not a multiple of 16: no split dQ).
        (16, 64),
        (48, 256),
        (64, 40),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_head_dims(dim, hidden, dtype):
    inputs = make_ragged_inputs(
        4, 260, 3, dim, hidden, 0.6, dtype=dtype, device=_CUDA, seed=3
    )
    _check(inputs, max_attn_len=50, contextual_seq_len=3)


@pytest.mark.parametrize(
    "dim,hidden,max_len,batch,heads,options",
    [
        (128, 128, 300, 4, 2, {}),
        (128, 128, 600, 3, 2, {"max_attn_len": 300, "contextual_seq_len": 7}),
        (96, 128, 600, 3, 2, {}),
        (96, 64, 600, 3, 2, {}),
        (256, 256, 1100, 2, 1, {}),
        (96, 256, 1100, 2, 1, {}),
        (160, 256, 1100, 2, 1, {}),
    ],
)
def test_split_dq_paths(dim, hidden, max_len, batch, heads, options):
    from mslk.attention.flydsl.hstu.ragged_bwd import split_dq

    compute_dim = next(d for d in (32, 64, 128, 256) if d >= max(dim, hidden))
    assert split_dq(
        compute_dim,
        dim,
        hidden,
        max_len,
        options.get("max_attn_len", 0),
        options.get("full_attn_size", 0),
    )
    inputs = make_ragged_inputs(
        batch, max_len, heads, dim, hidden, 0.3, device=_CUDA, seed=5, max_targets=20
    )
    _check(inputs, **options)


@pytest.mark.parametrize("mode", ["scalar", "token"])
def test_scale_modes(mode):
    inputs = make_ragged_inputs(4, 220, 2, 128, 128, 0.6, device=_CUDA, seed=4)
    T = inputs["q"].shape[0]
    scale = (
        torch.tensor(0.37, device=_CUDA)
        if mode == "scalar"
        else torch.rand(T, device=_CUDA) + 0.5
    )
    _check(inputs, attn_scale=scale)


def test_shared_kv():
    inputs = make_ragged_inputs(4, 180, 2, 64, 64, 0.6, device=_CUDA, seed=5)
    _check(inputs, shared_kv=True, max_attn_len=20)


def test_strided_and_short_sequences():
    lengths = torch.tensor([0, 1, 2, 37, 0, 129, 64])
    total = int(lengths.sum())
    gen = torch.Generator().manual_seed(6)
    qkv = (torch.rand(total, 2, 3 * 64 + 16, generator=gen) * 2 - 1).to(
        _CUDA, torch.bfloat16
    )
    inputs = {
        "N": 129,
        "alpha": 0.125,
        "q": qkv[:, :, :64],
        "k": qkv[:, :, 64:128],
        "v": qkv[:, :, 144:],
        "seq_offsets": lengths_to_offsets(lengths, _CUDA),
    }
    _check(inputs, max_attn_len=9, contextual_seq_len=2)


@pytest.mark.parametrize("dim", [32, 64])
@pytest.mark.parametrize("window", [0, 16])
def test_small_head_dim_short_sequences(dim, window):
    """Short sequences and small windows pick 32x64 tiles at D 64; D 32
    must keep its default tiles."""
    inputs = make_ragged_inputs(3, 300, 2, dim, dim, 0.6, device=_CUDA, seed=8)
    _check(inputs, max_attn_len=window)


@pytest.mark.parametrize("layout", ["fused_split", "v_wide_pitch"])
def test_kv_token_pitches_differ(layout):
    """K and V with different token pitches: split from a fused [T, H * (Dq
    + Dk + Dv)] projection with Dqk != Dv, or V sliced from a wider tensor."""
    lengths = torch.tensor([45, 0, 130, 77])
    total, heads = int(lengths.sum()), 2
    gen = torch.Generator().manual_seed(9)
    rand = lambda *s: (torch.rand(*s, generator=gen) * 2 - 1).to(  # noqa: E731
        _CUDA, torch.bfloat16
    )
    if layout == "fused_split":
        dqk, dv = 64, 128
        q, k, v = torch.split(
            rand(total, heads * (2 * dqk + dv)),
            [heads * dqk, heads * dqk, heads * dv],
            dim=1,
        )
        q, k, v = (t.view(total, heads, -1) for t in (q, k, v))
    else:
        q, k = rand(total, heads, 64), rand(total, heads, 64)
        v = rand(total, 2 * heads, 64)[:, :heads]
    inputs = {
        "N": 130,
        "alpha": 0.125,
        "q": q,
        "k": k,
        "v": v,
        "seq_offsets": lengths_to_offsets(lengths, _CUDA),
    }
    _check(inputs, max_attn_len=20)


@pytest.mark.parametrize("order", ["by_length", "permuted"])
def test_sort_by_length_indices(order):
    """Blocks follow sort_by_length_indices; any permutation of the
    sequences gives the same gradients (11 sequences: not a multiple of 8)."""
    inputs = make_ragged_inputs(
        11, 300, 2, 64, 64, 0.6, device=_CUDA, seed=13, max_targets=6
    )
    offsets = inputs["seq_offsets"]
    lengths = offsets[1:] - offsets[:-1]
    if order == "by_length":
        sort_idx = torch.argsort(lengths, descending=True)
    else:
        sort_idx = torch.randperm(
            lengths.numel(), generator=torch.Generator().manual_seed(1)
        ).to(_CUDA)
    _check(inputs, sort_idx=sort_idx, max_attn_len=40, contextual_seq_len=3)


def test_through_autograd():
    """The autograd path: the FlyDSL forward, then this backward."""
    import mslk.attention.hstu as api

    inputs = make_ragged_inputs(
        3, 150, 2, 64, 64, 0.6, device=_CUDA, seed=7, max_targets=5
    )
    args = _args(inputs, max_attn_len=12)
    q, k, v = (inputs[n].detach().clone().requires_grad_() for n in ("q", "k", "v"))
    out = api.ragged_hstu_attention(
        args["N"],
        args["alpha"],
        q,
        k,
        v,
        args["seq_offsets"],
        args["invalid_attn_mask_type"],
        args["num_targets"],
        None,
        args["max_attn_len"],
        0,
        0,
    )
    if isinstance(out, tuple):
        out = out[0]
    dout = torch.randn_like(out)
    out.backward(dout)
    ref = _reference_grads(inputs, dout, args)
    for name, got, want in (
        ("dq", q.grad, ref[0]),
        ("dk", k.grad, ref[1]),
        ("dv", v.grad, ref[2]),
    ):
        assert_hstu_close(got, want, "silu", name)


def test_cuda_graph_capture():
    from mslk.attention.flydsl.hstu.ragged_bwd import hstu_ragged_bwd

    inputs = make_ragged_inputs(3, 200, 2, 64, 64, 0.6, device=_CUDA, seed=8)
    args = _args(inputs, max_attn_len=30)
    q, k, v = inputs["q"], inputs["k"], inputs["v"]
    dout = torch.randn_like(v)
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)

    def call():
        hstu_ragged_bwd(
            dout,
            q,
            k,
            v,
            dq,
            dk,
            dv,
            args["seq_offsets"],
            None,
            None,
            args["N"],
            args["alpha"],
            args["max_attn_len"],
            "lower_triangular",
            0,
            None,
            0,
        )

    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for t in (q, k, v, dout):
        t.copy_(torch.rand_like(t) * 2 - 1)
    graph.replay()
    torch.cuda.synchronize()
    ref = _reference_grads(inputs, dout, args)
    for name, got, want in (("dq", dq, ref[0]), ("dk", dk, ref[1]), ("dv", dv, ref[2])):
        assert_hstu_close(got, want, "silu", name)


def _check_softmax(inputs, **options):
    """Softmax heads: M and Out from the reference forward, gradients against
    autograd on the reference."""
    from mslk.attention.flydsl.hstu.ragged_bwd import hstu_ragged_bwd

    args = _args(inputs, **options)
    q, k, v = inputs["q"], inputs["k"], inputs["v"]
    H = q.shape[1]
    with torch.no_grad():
        out, lse = ragged_hstu_attention_ref(q=q, k=k, v=v, num_softmax_heads=H, **args)
    gen = torch.Generator(device=_CUDA).manual_seed(0)
    dout = torch.randn(out.shape, device=_CUDA, generator=gen).to(q.dtype)
    qr, kr, vr = (x.detach().float().requires_grad_() for x in (q, k, v))
    ref_out, _ = ragged_hstu_attention_ref(
        q=qr, k=kr, v=vr, num_softmax_heads=H, **args
    )
    ref_out.backward(dout.float())
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    hstu_ragged_bwd(
        dout,
        q,
        k,
        v,
        dq,
        dk,
        dv,
        args["seq_offsets"],
        args["num_targets"],
        None,
        args["N"],
        args["alpha"],
        args["max_attn_len"],
        args["invalid_attn_mask_type"],
        args["contextual_seq_len"],
        None,
        args["full_attn_size"],
        num_softmax_heads=H,
        M=lse,
        Out=out,
    )
    # Softmax gradients are normalized; scale the SiLU bound by each max.
    for name, got, want in (
        ("dq", dq, qr.grad),
        ("dk", dk, kr.grad),
        ("dv", dv, vr.grad),
    ):
        assert_hstu_close(got, want, "silu", name)


@pytest.mark.parametrize(
    "mask", ["causal", "window", "contextual", "window_contextual_full"]
)
@pytest.mark.parametrize("dim", [64, 128, 256])
def test_softmax_heads(mask, dim):
    inputs = make_ragged_inputs(
        4, 260, 2, dim, dim, 0.6, device=_CUDA, seed=11, max_targets=8
    )
    inputs["alpha"] = dim**-0.5
    _check_softmax(inputs, **_MASKS[mask])


@pytest.mark.parametrize("dim,hidden", [(64, 128), (64, 256), (128, 256)])
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"max_attn_len": 16, "full_attn_size": 40},
        {"max_attn_len": 64, "full_attn_size": 40},
    ],
    ids=["causal", "window16_full", "window64_full"],
)
def test_softmax_heads_dqk_ne_dv(dim, hidden, options):
    """Softmax heads with Dqk != Dv, on windows with full-attention rows and
    no targets."""
    inputs = make_ragged_inputs(4, 260, 4, dim, hidden, 0.6, device=_CUDA, seed=12)
    inputs["alpha"] = dim**-0.5
    _check_softmax(inputs, **options)


def test_softmax_short_and_empty_sequences():
    lengths = torch.tensor([0, 1, 2, 127, 300, 0, 64])
    total = int(lengths.sum())
    gen = torch.Generator().manual_seed(12)
    q, k, v = (
        (torch.rand(total, 2, 64, generator=gen) * 2 - 1).to(_CUDA, torch.bfloat16)
        for _ in range(3)
    )
    inputs = {
        "N": 300,
        "alpha": 0.5,
        "q": q,
        "k": k,
        "v": v,
        "seq_offsets": lengths_to_offsets(lengths, _CUDA),
    }
    _check_softmax(inputs, max_attn_len=20, contextual_seq_len=4)


def test_softmax_through_autograd():
    """The autograd path with softmax heads: the FlyDSL forward's M and Out."""
    import mslk.attention.hstu as api

    inputs = make_ragged_inputs(3, 150, 2, 64, 64, 0.6, device=_CUDA, seed=13)
    args = _args(inputs, max_attn_len=12)
    q, k, v = (inputs[n].detach().clone().requires_grad_() for n in ("q", "k", "v"))
    out = api.ragged_hstu_attention(
        args["N"],
        64**-0.5,
        q,
        k,
        v,
        args["seq_offsets"],
        args["invalid_attn_mask_type"],
        None,
        None,
        args["max_attn_len"],
        0,
        0,
        2,
    )
    if isinstance(out, tuple):
        out = out[0]
    dout = torch.randn_like(out)
    out.backward(dout)
    qr, kr, vr = (inputs[n].detach().float().requires_grad_() for n in ("q", "k", "v"))
    ref, _ = ragged_hstu_attention_ref(
        q=qr, k=kr, v=vr, num_softmax_heads=2, **dict(args, alpha=64**-0.5)
    )
    ref.backward(dout.float())
    for name, got, want in (
        ("dq", q.grad, qr.grad),
        ("dk", k.grad, kr.grad),
        ("dv", v.grad, vr.grad),
    ):
        assert_hstu_close(got, want, "silu", name)


def test_packed_gradients_past_4gib():
    """f32 dQ/dK/dV scratch past 4 GiB: the last sequences (one empty) match
    a call on just them (the descriptors are rebased per sequence)."""
    from mslk.attention.flydsl.hstu.ragged_bwd import hstu_ragged_bwd

    H, D, N = 8, 128, 4096
    lengths = torch.full((262,), N)
    lengths[-2], lengths[-1] = 37, 0
    off = torch.zeros(lengths.numel() + 1, dtype=torch.int64)
    off[1:] = lengths.cumsum(0)
    T = int(off[-1])
    assert T * H * D * 4 > 2**32
    gen = torch.Generator(device=_CUDA).manual_seed(10)
    q, k, dout = (
        (torch.rand(T, H, D, device=_CUDA, generator=gen) * 0.2 - 0.1).to(
            torch.bfloat16
        )
        for _ in range(3)
    )

    def run(q, k, dout, offsets):
        dq, dk = torch.empty_like(q), torch.empty_like(k)
        hstu_ragged_bwd(
            dout,
            q,
            k,
            k,
            dq,
            dk,
            None,
            offsets.to(_CUDA),
            None,
            None,
            N,
            1.0 / D,
            16,
            "lower_triangular",
            0,
            None,
            0,
            shared_kv=True,
        )
        return dq, dk

    dq, dk = run(q, k, dout, off)
    lo = int(off[-4])
    dq_t, dk_t = run(q[lo:].clone(), k[lo:].clone(), dout[lo:].clone(), off[-4:] - lo)
    torch.testing.assert_close(dq[lo:], dq_t, atol=0, rtol=0)
    torch.testing.assert_close(dk[lo:], dk_t, atol=0, rtol=0)


def test_strided_gradient_outputs():
    """Non-contiguous dq/dk/dv take the f32-scratch path (contiguous ones
    are stored by the kernel directly)."""
    from mslk.attention.flydsl.hstu.ragged_bwd import hstu_ragged_bwd

    inputs = make_ragged_inputs(3, 160, 2, 64, 64, 0.6, device=_CUDA, seed=14)
    args = _args(inputs, max_attn_len=20)
    q, k, v = inputs["q"], inputs["k"], inputs["v"]
    dout = torch.randn_like(v)
    T = q.shape[0]
    grads = torch.empty(T, 2, 3 * 64, device=_CUDA, dtype=q.dtype)
    dq, dk, dv = grads[..., :64], grads[..., 64:128], grads[..., 128:]
    hstu_ragged_bwd(
        dout,
        q,
        k,
        v,
        dq,
        dk,
        dv,
        args["seq_offsets"],
        None,
        None,
        args["N"],
        args["alpha"],
        args["max_attn_len"],
        "lower_triangular",
        0,
        None,
        0,
    )
    ref = _reference_grads(inputs, dout, args)
    for name, got, want in (("dq", dq, ref[0]), ("dk", dk, ref[1]), ("dv", dv, ref[2])):
        assert_hstu_close(got, want, "silu", name)


def test_softmax_d_past_4gib():
    """D = rowsum(dO * Out) for inputs past 4 GiB (the preprocess kernel's
    offsets are 32-bit; it runs on row chunks)."""
    from mslk.attention.flydsl.hstu.ragged_bwd import _softmax_d

    T, D = (2**32) // (128 * 2) + 4096, 128
    gen = torch.Generator(device=_CUDA).manual_seed(15)
    dout, out = (
        torch.randn(T, 1, D, device=_CUDA, generator=gen).to(torch.bfloat16)
        for _ in range(2)
    )
    got = _softmax_d(dout, out, "bf16").view(-1)
    for lo, hi in ((0, 4096), (T - 8192, T)):
        want = (dout[lo:hi].float() * out[lo:hi].float()).sum(-1).view(-1)
        torch.testing.assert_close(got[lo:hi], want, atol=1e-3, rtol=1e-4)
