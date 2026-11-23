# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""AOT pre-compilation of the ragged HSTU forward kernels
(see :mod:`mslk.flydsl.aot`).

Covers what ``hstu_ragged_fwd(kernel="auto")`` routes to on gfx950, for
bf16 and the common mask features; other variants JIT on first use:

- the tiled kernel: D 64/128/256. It takes the head count at run time,
  but FlyDSL keys a tensor's layout on which dim has unit stride, and the
  [T, H] M output has two layouts: H = 1 (strides (1, 1)) and H >= 2; one
  entry each;
- the pipelined kernel, which compiles the head count in: D 64/128 at
  H 1/2/4/8.

Each config compiles under FlyDSL's ``COMPILE_ONLY`` with CPU placeholder
tensors (FlyDSL treats non-device tensors as compile placeholders; the
kernels take dynamic layouts, so the cache key matches real calls).
"""

from typing import Any, Dict, List, Tuple

import torch

from .common import LOWER, MaskSpec

AOT_ARCHS: List[str] = ["gfx950"]

# (has_window, has_contextual, has_targets, has_full_attn) on a
# lower-triangular mask.
_MASKS: Tuple[Tuple[bool, bool, bool, bool], ...] = (
    (False, False, False, False),
    (False, False, True, False),
    (True, False, False, False),
    (True, False, True, False),
    (True, True, True, False),
    (True, True, True, True),
)
# (scale_mode, softmax)
_HEADS: Tuple[Tuple[str, bool], ...] = (
    ("inv_n", False),
    ("token", False),
    ("inv_n", True),
)
_PIPELINED_HEADS = (1, 2, 4, 8)

AOT_CONFIGS: List[Dict[str, Any]] = [
    {
        "kernel": "tiled",
        "head_dim": d,
        "num_heads": h,
        "mask": m,
        "scale_mode": sc,
        "softmax": sm,
    }
    for d in (64, 128, 256)
    for h in (1, 2)
    for m in _MASKS
    for sc, sm in _HEADS
] + [
    {
        "kernel": "pipelined",
        "head_dim": d,
        "num_heads": h,
        "mask": m,
        "scale_mode": sc,
        "softmax": sm,
    }
    for d in (64, 128)
    for h in _PIPELINED_HEADS
    for m in _MASKS
    for sc, sm in _HEADS
]


def _spec(mask: Tuple[bool, bool, bool, bool]) -> MaskSpec:
    has_window, has_contextual, has_targets, has_full_attn = mask
    return MaskSpec(LOWER, has_window, has_contextual, has_targets, has_full_attn)


def compile_aot_config(config: Dict[str, Any], arch: str) -> None:
    import flydsl.compiler as flyc  # pyre-ignore[21]
    import flydsl.expr as fx  # pyre-ignore[21]
    from flydsl.compiler.kernel_function import CompilationContext  # pyre-ignore[21]

    d = config["head_dim"]
    spec = _spec(tuple(config["mask"]))
    scale_mode, softmax = config["scale_mode"], config["softmax"]
    h = config["num_heads"]
    cpu = torch.device("cpu")
    qkv = torch.empty(1, h, d, dtype=torch.bfloat16, device=cpu)
    out = torch.empty(1, h, d, dtype=torch.bfloat16, device=cpu)
    lse = torch.empty(1, h, dtype=torch.float32, device=cpu)
    idx = torch.empty(2, dtype=torch.int32, device=cpu)
    scale = torch.empty(1, dtype=torch.float32, device=cpu)
    stream = fx.Stream(None)

    if config["kernel"] == "tiled":
        from . import ragged_fwd as tiled

        # Windowed masks also get the small-window tiles.
        cfgs = {tiled.default_config(d, d, arch)}
        if spec.has_window:
            cfgs.add(tiled.default_config(d, d, arch, tiled._SMALL_WINDOW))
        for cfg in sorted(cfgs, key=repr):
            _compile_tiled(
                tiled,
                flyc,
                CompilationContext,
                d,
                h,
                spec,
                scale_mode,
                softmax,
                cfg,
                arch,
                qkv,
                idx,
                scale,
                out,
                lse,
                stream,
            )
        return

    from . import ragged_fwd_gfx950 as pipelined

    launch = pipelined.build_ragged_fwd_gfx950(
        h,
        d,
        "bf16",
        spec.mask_type,
        spec.has_window,
        spec.has_contextual,
        spec.has_targets,
        spec.has_full_attn,
        scale_mode,
        softmax,
        False,
        True,
    )
    launch(
        qkv,
        qkv,
        qkv,
        out,
        lse,
        idx,
        idx,
        scale,
        idx,
        1,
        1,
        h * d,
        h * d,
        1.0,
        1.0,
        0,
        0,
        0,
        stream,
    )


def _compile_tiled(
    tiled,
    flyc,
    CompilationContext,
    d,
    h,
    spec,
    scale_mode,
    softmax,
    cfg,
    arch,
    qkv,
    idx,
    scale,
    out,
    lse,
    stream,
):
    launcher = tiled.build_ragged_fwd(
        d, d, "bf16", spec, scale_mode, softmax, cfg, arch, True
    )
    with CompilationContext.compile_hints(launcher.compile_hints):
        flyc.compile(
            launcher,
            qkv,
            qkv,
            qkv,
            idx,
            idx,
            scale,
            out,
            lse,
            1,
            h,
            1,
            1.0,
            1.0,
            0,
            0,
            0,
            h * d,
            d,
            h * d,
            d,
            stream,
        )
