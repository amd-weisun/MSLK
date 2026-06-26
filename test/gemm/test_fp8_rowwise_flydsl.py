"""WP-G1 prototype test: FlyDSL f8f8bf16_rowwise_flydsl vs CK baseline.

Runs on gfx950 (ROCm) only.  The FlyDSL op is a parallel sibling of the CK
``f8f8bf16_rowwise`` op — same schema, same inputs, output diffed for parity.

Usage (inside container aiter-weisun):
    cd /workspace/MSLK
    PYTHONPATH=/workspace/FlyDSL:$PYTHONPATH python -m pytest \
        test/gemm/test_fp8_rowwise_flydsl.py -v
"""

import unittest

import mslk.gemm  # noqa: F401 — triggers FlyDSL op registration
import torch
from mslk.testing.device import skipUnlessGfxArch, skipUnlessRocm


FP8_DTYPE = torch.float8_e4m3fnuz  # ROCm gfx950 fp8 dtype


def _make_inputs(M: int, N: int, K: int, device: str = "cuda"):
    """Build identical fp8 inputs and row/col scales."""
    XQ = torch.randn(M, K, device=device).to(FP8_DTYPE)
    WQ = torch.randn(N, K, device=device).to(FP8_DTYPE)
    x_scale = torch.rand(M, device=device, dtype=torch.float32) + 0.1
    w_scale = torch.rand(N, device=device, dtype=torch.float32) + 0.1
    return XQ, WQ, x_scale, w_scale


@skipUnlessRocm()
@skipUnlessGfxArch("gfx950")
class TestFp8RowwiseFlyDSLvsCK(unittest.TestCase):
    """Diff FlyDSL f8f8bf16_rowwise_flydsl against the CK f8f8bf16_rowwise baseline."""

    # Shapes to exercise: (M, N, K).
    # K must be a multiple of 128 (FlyDSL BLOCK_K=128).
    # M/N are rounded up internally — keep them multiples of 256 for v0.
    SHAPES = [
        (256, 256, 256),
        (512, 512, 512),
        (1024, 4096, 4096),   # typical LLM projection shape
        (2048, 4096, 4096),
    ]

    def _run_both(self, M, N, K):
        XQ, WQ, x_scale, w_scale = _make_inputs(M, N, K)

        out_ck = torch.ops.mslk.f8f8bf16_rowwise(XQ, WQ, x_scale, w_scale)
        out_flydsl = torch.ops.mslk.f8f8bf16_rowwise_flydsl(XQ, WQ, x_scale, w_scale)
        return out_ck, out_flydsl

    def _assert_close(self, out_ck, out_flydsl, M, N, K):
        ok = torch.allclose(
            out_flydsl.to(torch.float32),
            out_ck.to(torch.float32),
            rtol=0.1,
            atol=0.1,
        )
        if not ok:
            max_diff = (out_flydsl.float() - out_ck.float()).abs().max().item()
            self.fail(
                f"FlyDSL vs CK mismatch (M={M} N={N} K={K}): max_diff={max_diff:.4f}"
            )

    def test_smoke_small(self):
        """Quick smoke test: smallest valid shape."""
        M, N, K = 256, 256, 256
        out_ck, out_flydsl = self._run_both(M, N, K)
        self._assert_close(out_ck, out_flydsl, M, N, K)

    def test_shapes(self):
        """Correctness across a range of shapes."""
        for M, N, K in self.SHAPES:
            with self.subTest(M=M, N=N, K=K):
                out_ck, out_flydsl = self._run_both(M, N, K)
                self._assert_close(out_ck, out_flydsl, M, N, K)

    def test_output_dtype(self):
        """Output must be bfloat16."""
        M, N, K = 256, 256, 256
        _, out_flydsl = self._run_both(M, N, K)
        self.assertEqual(out_flydsl.dtype, torch.bfloat16)
        self.assertEqual(out_flydsl.shape, (M, N))

    def test_bias_raises(self):
        """Bias is not yet implemented in v0 — must raise NotImplementedError."""
        M, N, K = 256, 256, 256
        XQ, WQ, x_scale, w_scale = _make_inputs(M, N, K)
        bias = torch.zeros(N, device="cuda", dtype=torch.bfloat16)
        with self.assertRaises(NotImplementedError):
            torch.ops.mslk.f8f8bf16_rowwise_flydsl(XQ, WQ, x_scale, w_scale, bias=bias)


if __name__ == "__main__":
    unittest.main()
