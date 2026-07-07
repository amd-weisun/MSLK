# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# pyre-unsafe
# pyre-ignore-all-errors[29]
"""WP-A3: opt-in coverage test for `fmha.flydsl.BwOp`.

Deliberately NOT merged into `test_backward.py`'s `ALL_BW_OPS` parametrization
(`fmha.flydsl.BwOp` is not added to `ALL_BW_OPS` in `mslk/attention/fmha/__init__.py`
— see `mslk/attention/fmha/flydsl.py` docstring). Instead this file reuses the
SAME parametrization generator (`case_generation._generate_op_device_dtype_biasT_B_Mq_Mkv_H_K_Kv`)
and the SAME helpers (`create_tensors`, `ref_attention_for_test`, `assert_allclose`)
that `test_backward.py` uses for `ck.BwOp` et al., scoped to `[fmha.flydsl.BwOp]`
only, so FlyDSL's functional coverage is measured identically to CK's.

Usage (inside container):
    cd /workspace/MSLK
    HIP_VISIBLE_DEVICES=3 FLYDSL_RUNTIME_ENABLE_CACHE=0 \
    PYTHONPATH=/workspace/FlyDSL:$PYTHONPATH \
    python -m pytest test/attention/fmha/test_backward_flydsl.py -v
"""

import logging
import random

import pytest
import torch

from mslk.attention import fmha
from mslk.attention.fmha import flydsl  # noqa: F401 -- binds fmha.flydsl attribute
from mslk.attention.fmha.unbind import unbind

from .case_generation import (
    _generate_op_device_dtype_biasT_B_Mq_Mkv_H_K_Kv,
    create_tensors,
    get_bias_grad,
)
from .utils import (
    assert_allclose,
    disable_tf32,
    ref_attention_for_test,
    UNSUPPORTED_OP_PASSES,
)

logger = logging.getLogger(__file__)

FLYDSL_BW_OPS = [fmha.flydsl.BwOp]

parametrize_opBW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv = pytest.mark.parametrize(
    "opBW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv",
    **_generate_op_device_dtype_biasT_B_Mq_Mkv_H_K_Kv(FLYDSL_BW_OPS),
)


@disable_tf32
@pytest.mark.parametrize("fmt", ["BMK", "BMHK"])
@pytest.mark.parametrize("grad_out_contiguous", [False, True])
@parametrize_opBW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv
def test_backward_flydsl(  # noqa: C901
    opBW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv,
    grad_out_contiguous,
    fmt,
):
    (
        op_bw,
        device,
        dtype,
        bias_type,
        batch_size,
        q_len,
        kv_len,
        h,
        k,
        kv,
    ) = opBW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv

    # FlyDSL's kernel is not stride-aware; skip the qkv-fused-storage /
    # non-contiguous path rather than falling back to a copy (see
    # mslk/attention/fmha/flydsl.py docstring).
    if not grad_out_contiguous:
        pytest.skip("FlyDSL BwOp requires contiguous grad_out (no stride-aware addressing yet)")

    attn_bias_requires_grad = (
        random.Random(q_len + kv_len * batch_size).randint(0, 1) > 0
    )
    try:
        query, key, value, attn_bias = create_tensors(
            *opBW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv,
            attn_bias_requires_grad=attn_bias_requires_grad,
            fmt=fmt,
        )
    except pytest.skip.Exception as e:
        if UNSUPPORTED_OP_PASSES:
            logger.warning(f"Skipping {opBW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv}: {e}")
            return
        raise

    scale = None
    if op_bw.SUPPORTS_CUSTOM_SCALE and query.shape[-1] < 32:
        scale = (1 / 32) ** 0.5
    # Pin the forward op to CK, per WP-A3 decision: FlyDSL has no forward
    # kernel, and CK's forward is verified working on both gfx942 and gfx950
    # (mirrors test_backward.py's `if op_bw == fmha.ck.BwOp: op_fw = fmha.ck.FwOp`).
    op_fw = fmha.ck.FwOp

    qkv = None

    if (
        fmt == "BMHK"
        and query.shape[3] == value.shape[3]
        and query.shape[1] == value.shape[1]
    ):
        qkv = torch.stack([query, key, value], 2)
        qkv.requires_grad_(True)
        # bm3hk -> 3 x bmhk
        query, key, value = unbind(qkv, 2)
        assert not query.is_contiguous()

    query.requires_grad_(True)
    key.requires_grad_(True)
    value.requires_grad_(True)

    if not op_bw.supports(fmha.Inputs(query, key, value, attn_bias)):
        if UNSUPPORTED_OP_PASSES:
            return
        pytest.skip("inputs not supported")

    out = fmha.memory_efficient_attention(
        query, key, value, attn_bias, scale=scale, op=(op_fw, op_bw)
    )

    grad_out = torch.randn_like(out)
    out.backward(grad_out)

    grads = []
    if qkv is None:
        grads = [query.grad, key.grad, value.grad]
        query.grad = None
        key.grad = None
        value.grad = None
    else:
        grads = [qkv.grad]
        qkv.grad = None
    if attn_bias_requires_grad:
        attn_bias_grad = get_bias_grad(attn_bias, clear=True)
        if attn_bias_grad is not None:
            grads.append(attn_bias_grad)

    ref = ref_attention_for_test(query, key, value, attn_bias, scale=scale)
    ref.backward(grad_out)

    assert_allclose(
        out.float().to(ref.device),
        ref.float(),
        "fw pass",
        atol=op_fw.ERROR_ATOL[dtype],
        rtol=op_fw.ERROR_RTOL[dtype],
    )

    del out
    del grad_out
    del ref

    atol = op_bw.ERROR_ATOL[dtype]
    rtol = op_bw.ERROR_RTOL[dtype]

    grads_ref = []
    grads_name = []
    if qkv is None:
        assert isinstance(query.grad, torch.Tensor)
        assert isinstance(key.grad, torch.Tensor)
        assert isinstance(value.grad, torch.Tensor)
        grads_ref = [query.grad, key.grad, value.grad]
        grads_name = ["query", "key", "value"]
    else:
        assert isinstance(qkv.grad, torch.Tensor)
        grads_ref = [qkv.grad]
        grads_name = ["qkv"]

    if attn_bias_requires_grad:
        attn_bias_grad = get_bias_grad(attn_bias)
        if attn_bias_grad is not None:
            grads_ref.append(attn_bias.grad)
            grads_name.append("bias")

    del query
    del key
    del value
    del qkv

    assert len(grads_ref) == len(grads), (
        "Wrong number of gradients (maybe bias grad didn't backprop?)"
    )
    for name, calc_grad, ref_grad in zip(grads_name, grads, grads_ref):
        assert_allclose(
            calc_grad.to(ref_grad.device),
            ref_grad,
            msg=f"{op_fw.NAME}+{op_bw.NAME}:{name}",
            atol=atol,
            rtol=rtol,
        )
