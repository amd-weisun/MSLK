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
from mslk.attention.hstu._torch_ref import (
    ragged_hstu_attention_ref,
    ragged_hstu_valid_mask,
)
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
        "num_targets": inputs.get("num_targets"),
        "attn_scale": None,
        "max_attn_len": 0,
        "contextual_seq_len": 0,
        "full_attn_size": 0,
    }
    args.update(options)
    call = dict(inputs)
    call.update(args)
    config = call.pop("config", None)
    kernel = call.pop("kernel", "auto")
    _skip_if_declined(kernel, call)
    out, lse = hstu_ragged_fwd(**call, config=config, kernel=kernel)
    ref, ref_lse = ragged_hstu_attention_ref(**call)
    if call.get("num_softmax_heads", 0):
        return out, ref, lse, ref_lse
    return out, ref


# Every kernel: the gfx950 dual-wave pipeline (D 64/128), the generic
# flash-attention loop (D 64-256) and the tiled kernel.
KERNELS = ["pipelined", "generic", "tiled"]


def _skip_if_declined(kernel, call):
    if kernel not in ("pipelined", "generic"):
        return
    from mslk.attention.flydsl.hstu import ragged_fwd_generic, ragged_fwd_gfx950
    from mslk.attention.flydsl.hstu.ragged_fwd import _arch

    module = ragged_fwd_gfx950 if kernel == "pipelined" else ragged_fwd_generic
    v = call["k"] if call.get("shared_kv") else call["v"]
    reason = module.decline_reason(
        call["q"], call["k"], v, call.get("num_softmax_heads", 0) > 0, _arch()
    )
    if reason is not None:
        pytest.skip(f"pipelined kernel declines: {reason}")


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
@pytest.mark.parametrize("kernel", KERNELS)
def test_mask_features(mask, targets, kernel):
    inputs = _inputs(6, 300, max_targets=20 if targets else 0, seed=1)
    out, ref = _run(inputs, kernel=kernel, **_MASKS[mask])
    assert_hstu_close(out, ref, "silu")


@pytest.mark.parametrize(
    "dim,hidden",
    [(d, d) for d in HEAD_DIMS] + [(64, 128), (128, 64), (48, 96), (64, 160)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("kernel", KERNELS)
def test_head_dims(dim, hidden, dtype, kernel):
    inputs = _inputs(4, 200, heads=3, dim=dim, hidden=hidden, dtype=dtype, seed=2)
    out, ref = _run(inputs, kernel=kernel, max_attn_len=50, contextual_seq_len=3)
    assert_hstu_close(out, ref, "silu")


@pytest.mark.parametrize("form", ["scalar", "token"])
@pytest.mark.parametrize("kernel", KERNELS)
def test_attn_scale(form, kernel):
    inputs = _inputs(5, 160, seed=3)
    total = inputs["q"].shape[0]
    if form == "scalar":
        scale = torch.tensor(0.37, device=_CUDA)
    else:
        scale = torch.rand(total, device=_CUDA, dtype=torch.bfloat16) + 0.5
    out, ref = _run(inputs, kernel=kernel, attn_scale=scale)
    assert_hstu_close(out, ref, "silu")


@pytest.mark.parametrize("kernel", KERNELS)
def test_alpha_is_a_runtime_value(kernel):
    inputs = _inputs(3, 100, seed=4)
    for alpha in (0.05, 0.5, 2.0):
        inputs["alpha"] = alpha
        out, ref = _run(inputs, kernel=kernel)
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
@pytest.mark.parametrize("kernel", KERNELS)
def test_lengths(lengths, kernel):
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
        out, ref = _run(inputs, kernel=kernel, **options)
        assert_hstu_close(out, ref, "silu")


def test_n_larger_than_longest_sequence():
    """N sets the default 1/N scale and the grid, not the lengths."""
    inputs = _inputs(4, 100, seed=6)
    inputs["N"] = 333
    out, ref = _run(inputs)
    assert_hstu_close(out, ref, "silu")


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("kernel", KERNELS)
def test_index_dtypes(index_dtype, kernel):
    inputs = _inputs(4, 120, max_targets=5, seed=7)
    inputs["seq_offsets"] = inputs["seq_offsets"].to(index_dtype)
    targets = inputs["num_targets"].to(index_dtype)
    out, ref = _run(inputs, kernel=kernel, num_targets=targets, max_attn_len=9)
    assert_hstu_close(out, ref, "silu")


@pytest.mark.parametrize("kernel", KERNELS)
def test_strided_views(kernel):
    """q, k and v slices of one packed qkv tensor, so token strides differ
    from the head-dim extent."""
    base = _inputs(5, 150, seed=8)
    total, heads, dim = base["q"].shape
    qkv = torch.rand(total, heads, 3 * dim + 16, device=_CUDA, dtype=torch.bfloat16)
    inputs = dict(base)
    inputs["q"] = qkv[:, :, :dim]
    inputs["k"] = qkv[:, :, dim : 2 * dim]
    inputs["v"] = qkv[:, :, 2 * dim + 16 :]
    out, ref = _run(inputs, kernel=kernel, max_attn_len=20)
    assert_hstu_close(out, ref, "silu")


@pytest.mark.parametrize("kernel", KERNELS)
def test_shared_kv(kernel):
    inputs = _inputs(4, 140, shared_kv=True, seed=9)
    out, ref = _run(inputs, kernel=kernel)
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


@pytest.mark.parametrize("kernel", KERNELS)
def test_variants_do_not_share_a_compiled_kernel(kernel):
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
        out, ref = _run(inputs, kernel=kernel, **options)
        assert_hstu_close(out, ref, "silu")


def test_unsupported_dims_are_declined():
    """No tile divides the load pass for hidden dim 80; the kernel must
    refuse rather than compute a wrong result."""
    inputs = _inputs(2, 32, dim=48, hidden=80, seed=12)
    with pytest.raises(NotImplementedError, match="FlyDSL ragged HSTU forward"):
        _run(inputs)


def _bf16_p_softmax(inputs, options):
    """Softmax attention as a kernel with bf16 P computes it: fp32 scores and
    softmax, P rounded to the input dtype, fp32 P @ V, 1/l in fp32."""
    offsets = inputs["seq_offsets"].tolist()
    targets = inputs.get("num_targets")
    q, k, v = (inputs[n].float() for n in ("q", "k", "v"))
    out = torch.zeros_like(v)
    for b in range(len(offsets) - 1):
        s0, e0 = offsets[b], offsets[b + 1]
        if e0 == s0:
            continue
        valid = ragged_hstu_valid_mask(
            e0 - s0,
            options.get("invalid_attn_mask_type", "lower_triangular"),
            None if targets is None else int(targets[b]),
            options.get("max_attn_len", 0),
            options.get("contextual_seq_len", 0),
            options.get("full_attn_size", 0),
            device=q.device,
        )
        scores = inputs["alpha"] * torch.einsum("qhd,khd->hqk", q[s0:e0], k[s0:e0])
        scores = scores.masked_fill(~valid, float("-inf"))
        p = torch.exp(scores - scores.amax(-1, keepdim=True))
        l_sum = p.sum(-1, keepdim=True)
        p_rounded = p.to(inputs["q"].dtype).float()
        out[s0:e0] = torch.einsum("hqk,khv->qhv", p_rounded, v[s0:e0]) / l_sum.permute(
            1, 0, 2
        )
    return out.to(inputs["q"].dtype)


def _check_softmax(inputs, options):
    out, ref, lse, ref_lse = _run(inputs, **options)
    # Against a kernel-precision model (P rounded to bf16/fp16 for P @ V):
    # within about one output ulp (accumulation orders differ).
    emulated = _bf16_p_softmax(inputs, options)
    torch.testing.assert_close(out.float(), emulated.float(), atol=2e-3, rtol=8e-3)
    # Against T1. Its softmax bound (atol 8e-4, rtol 1.6e-2) is tighter than
    # rounding P to bf16 allows: the emulation above misses it by up to 1.8x
    # on rows that see only a few keys. Hence twice T1's bound.
    torch.testing.assert_close(out.float(), ref.float(), atol=1.6e-3, rtol=3.2e-2)
    torch.testing.assert_close(lse, ref_lse, atol=2e-3, rtol=1e-4)


@pytest.mark.parametrize("mask", list(_MASKS))
@pytest.mark.parametrize("dim", HEAD_DIMS)
@pytest.mark.parametrize("kernel", KERNELS)
def test_softmax_heads(mask, dim, kernel):
    """All heads softmax (T2 takes num_softmax_heads 0 or H): out and the
    base-2 LSE M."""
    inputs = _inputs(5, 260, heads=2, dim=dim, max_targets=8, seed=13)
    inputs["num_softmax_heads"] = 2
    _check_softmax(inputs, dict(_MASKS[mask], kernel=kernel))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("kernel", KERNELS)
def test_softmax_lengths_and_dtypes(dtype, kernel):
    gen = torch.Generator().manual_seed(15)
    lengths_t = torch.tensor([0, 1, 2, 127, 300, 0, 64])
    total = int(lengths_t.sum())
    q, k, v = (
        (torch.rand(total, 2, 64, generator=gen) * 2 - 1).to(_CUDA, dtype)
        for _ in range(3)
    )
    inputs = {
        "N": 300,
        "alpha": 0.5,
        "q": q,
        "k": k,
        "v": v,
        "seq_offsets": lengths_to_offsets(lengths_t, _CUDA),
        "num_softmax_heads": 2,
    }
    _check_softmax(
        inputs, {"max_attn_len": 20, "contextual_seq_len": 4, "kernel": kernel}
    )


def test_softmax_ignores_attn_scale():
    inputs = _inputs(3, 100, seed=14)
    inputs["num_softmax_heads"] = 2
    _check_softmax(inputs, {"attn_scale": torch.tensor(5.0, device=_CUDA)})


def _op_args(inputs, **options):
    args = {
        "N": inputs["N"],
        "alpha": inputs["alpha"],
        "q": inputs["q"],
        "k": inputs["k"],
        "v": inputs["v"],
        "seq_offsets": inputs["seq_offsets"],
        "invalid_attn_mask_type": "lower_triangular",
        "num_targets": inputs.get("num_targets"),
        "attn_scale": None,
        "max_attn_len": 0,
        "contextual_seq_len": 0,
        "full_attn_size": 0,
        "num_softmax_heads": 0,
        "shared_kv": inputs.get("shared_kv", False),
    }
    args.update(options)
    return args


@pytest.mark.parametrize("softmax", [False, True])
def test_through_the_op(softmax):
    """mslk::hstu_ragged_fwd and the public wrapper route to this kernel."""
    import mslk.attention.hstu as hstu

    inputs = _inputs(4, 180, max_targets=6, seed=16)
    args = _op_args(
        inputs,
        max_attn_len=25,
        contextual_seq_len=3,
        num_softmax_heads=2 if softmax else 0,
    )
    ref, ref_lse = ragged_hstu_attention_ref(**args)
    for out, lse in (
        torch.ops.mslk.hstu_ragged_fwd(**args),
        hstu.ragged_hstu_attention_fwd(**args),
    ):
        if softmax:
            torch.testing.assert_close(
                out.float(), ref.float(), atol=1.6e-3, rtol=3.2e-2
            )
            torch.testing.assert_close(lse, ref_lse, atol=2e-3, rtol=1e-4)
        else:
            assert_hstu_close(out, ref, "silu")


def test_cuda_graph_capture():
    """After a warm-up call, the forward captures and replays in a CUDA
    graph: no host synchronization and no allocation the graph cannot own."""
    inputs = _inputs(4, 200, max_targets=5, seed=17)
    scale = torch.rand(inputs["q"].shape[0], device=_CUDA) + 0.5
    args = _op_args(inputs, max_attn_len=30, attn_scale=scale)
    torch.ops.mslk.hstu_ragged_fwd(**args)  # warm up: compile and route
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, _ = torch.ops.mslk.hstu_ragged_fwd(**args)
    # New values in the same buffers, then replay.
    for name in ("q", "k", "v"):
        inputs[name].copy_(torch.rand_like(inputs[name]) * 2 - 1)
    scale.copy_(torch.rand_like(scale) + 0.5)
    graph.replay()
    torch.cuda.synchronize()
    ref, _ = ragged_hstu_attention_ref(**args)
    assert_hstu_close(out, ref, "silu")


def _rows_reference(inputs, b, rows, options):
    """T1's output for some rows of sequence b, without materializing the
    other rows' scores (T1's reference forms [H, n, n])."""
    from mslk.attention.hstu._torch_ref import _attend

    s0, e0 = (int(x) for x in inputs["seq_offsets"][b : b + 2])
    targets = inputs.get("num_targets")
    valid = ragged_hstu_valid_mask(
        e0 - s0,
        options.get("invalid_attn_mask_type", "lower_triangular"),
        None if targets is None else int(targets[b]),
        options.get("max_attn_len", 0),
        options.get("contextual_seq_len", 0),
        options.get("full_attn_size", 0),
        device=_CUDA,
    )[rows]
    q = inputs["q"][s0:e0][rows].float()
    k = inputs["k"][s0:e0].float()
    v = inputs["v"][s0:e0].float()
    out, _ = _attend(q, k, v, valid, inputs["alpha"], 1.0 / inputs["N"], 0)
    return out.to(inputs["q"].dtype)


def _free_gpu_bytes():
    free, _ = torch.cuda.mem_get_info()
    return free


@pytest.mark.parametrize("kernel", KERNELS)
def test_offsets_beyond_2_31_elements(kernel):
    """Packed row offset x token stride crosses 2^31 elements (A9): the last
    sequences start beyond it, so 32-bit address arithmetic would wrap."""
    heads, dim = (16, 128) if kernel == "pipelined" else (8, 256)
    per_token = heads * dim
    total = (2**31) // per_token + 4096  # ~1.05M tokens, ~4.3 GB per tensor
    if _free_gpu_bytes() < 5 * total * per_token * 2:
        pytest.skip("needs about 25 GB of free GPU memory")
    gen = torch.Generator(device=_CUDA).manual_seed(18)

    def rand(*shape):
        return (torch.rand(*shape, generator=gen, device=_CUDA) * 2 - 1).to(
            torch.bfloat16
        )

    q, k, v = rand(total, heads, dim), rand(total, heads, dim), rand(total, heads, dim)
    # Long sequences up to the end; the last two start past 2^31 elements.
    lengths = [4096] * (total // 4096 - 2)
    lengths += [total - sum(lengths) - 1000, 1000]
    offsets = lengths_to_offsets(torch.tensor(lengths), _CUDA)
    assert int(offsets[-2]) * per_token > 2**31
    inputs = {
        "N": max(lengths),
        "alpha": dim**-0.5,
        "q": q,
        "k": k,
        "v": v,
        "seq_offsets": offsets,
    }
    options = {"max_attn_len": 512}
    out, _ = _run_kernel_only(inputs, kernel=kernel, **options)
    last = len(lengths) - 1
    for b in (0, last - 1, last):
        n = lengths[b]
        rows = torch.tensor([0, 1, n // 2, n - 2, n - 1], device=_CUDA)
        s0 = int(offsets[b])
        ref = _rows_reference(inputs, b, rows, options)
        assert_hstu_close(out[s0 + rows], ref, "silu")


@pytest.mark.parametrize("kernel", KERNELS)
def test_64k_sequence(kernel):
    """A single sequence of 64k tokens at head dim 256 (A9)."""
    n, heads, dim = 65536, 1, (128 if kernel == "pipelined" else 256)
    gen = torch.Generator(device=_CUDA).manual_seed(19)
    q, k, v = (
        (torch.rand(n, heads, dim, generator=gen, device=_CUDA) * 2 - 1).to(
            torch.bfloat16
        )
        for _ in range(3)
    )
    inputs = {
        "N": n,
        "alpha": dim**-0.5,
        "q": q,
        "k": k,
        "v": v,
        "seq_offsets": torch.tensor([0, n], device=_CUDA),
        "num_targets": torch.tensor([300], device=_CUDA),
    }
    rows = torch.tensor([0, 7, 4095, 32768, n - 301, n - 300, n - 1], device=_CUDA)
    for options in ({}, {"max_attn_len": 2048, "full_attn_size": 1000}):
        out, _ = _run_kernel_only(inputs, kernel=kernel, **options)
        ref = _rows_reference(inputs, 0, rows, options)
        assert_hstu_close(out[rows], ref, "silu")


def _run_kernel_only(inputs, kernel="auto", **options):
    from mslk.attention.flydsl.hstu.ragged_fwd import hstu_ragged_fwd

    args = _op_args(inputs, **options)
    return hstu_ragged_fwd(**args, kernel=kernel)


@pytest.mark.parametrize("schedule", ["plain", "reverse", "sorted", "sorted_reverse"])
@pytest.mark.parametrize("kernel", ["pipelined", "generic"])
@pytest.mark.parametrize("mask", ["causal", "upper", "window_contextual_full"])
def test_schedules(schedule, kernel, mask):
    """Block orders change only the schedule, not the result."""
    from mslk.attention.flydsl.hstu.ragged_fwd import hstu_ragged_fwd

    lengths = torch.tensor([30, 900, 0, 1500, 64, 700, 1200, 5])
    gen = torch.Generator().manual_seed(20)
    total = int(lengths.sum())
    q, k, v = (
        (torch.rand(total, 2, 128, generator=gen) * 2 - 1).to(_CUDA, torch.bfloat16)
        for _ in range(3)
    )
    inputs = {
        "N": 1500,
        "alpha": 128**-0.5,
        "q": q,
        "k": k,
        "v": v,
        "seq_offsets": lengths_to_offsets(lengths, _CUDA),
        "num_targets": torch.tensor([3, 10, 0, 40, 0, 1, 7, 2], device=_CUDA),
    }
    args = _op_args(inputs, **_MASKS[mask])
    out, _ = hstu_ragged_fwd(**args, kernel=kernel, schedule=schedule)
    ref, _ = ragged_hstu_attention_ref(**args)
    assert_hstu_close(out, ref, "silu")
