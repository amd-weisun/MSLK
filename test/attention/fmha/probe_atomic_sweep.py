"""Resolve whether the f32 atomic doubling is deterministic (per-instruction x2)
or contention-dependent (race). Fixed add=1.0; sweep G (block count). Read ratio.

  ratio == 2.0 for all G  -> deterministic x2  -> 0.5 prescale is safe
  ratio drifts with G      -> race/lost-adds   -> 0.5 prescale NOT safe

Run:
  HIP_VISIBLE_DEVICES=3 FLYDSL_RUNTIME_ENABLE_CACHE=0 \
  PYTHONPATH=/workspace/FlyDSL:$PYTHONPATH \
  python test/attention/fmha/probe_atomic_sweep.py
"""
import sys
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, rocdl, buffer_ops
from flydsl._mlir.dialects import llvm
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly as _fly
from flydsl.expr.typing import T

sys.path.insert(0, ".")
Vec = fx.Vector


def _raw(v):
    return v.ir_value() if hasattr(v, "ir_value") else v


def _rsrc(out):
    base_ptr = _fly.extract_aligned_pointer_as_index(ir.Type.parse("!llvm.ptr"), out)
    base_i64 = llvm.PtrToIntOp(T.i64, base_ptr).result
    base_lo = fx.ArithValue(base_i64).trunci(T.i32)
    base_hi = fx.ArithValue(fx.ArithValue(base_i64).shrui(fx.Int64(32))).trunci(T.i32)
    return Vec.from_elements(
        [base_lo, base_hi,
         buffer_ops._create_i32_constant(0xFFFFFFFF),
         buffer_ops._create_i32_constant(buffer_ops._get_buffer_flags())],
        fx.Int32,
    ).ir_value()


def build():
    @flyc.kernel(known_block_size=[64, 1, 1])
    def k(out: fx.Tensor):
        rsrc = _rsrc(out)
        rocdl.raw_buffer_atomic_fadd(
            _raw(fx.Float32(1.0)), rsrc,
            _raw(fx.Int32(0)), _raw(fx.Int32(0)), _raw(fx.Int32(0)),
        )

    @flyc.jit
    def launch(out: fx.Tensor, G: fx.Int32, stream: fx.Stream):
        k(out).launch(grid=(fx.Int32(G), 1, 1), block=(64, 1, 1), stream=stream)

    return launch


def main():
    dev = "cuda"
    st = torch.cuda.current_stream()
    c = flyc.compile(build(), torch.zeros(1, 1, device=dev), 1, st)
    print("fixed add=1.0, block=64; sweep G:")
    for G in [1, 4, 16, 64, 256, 512, 1024]:
        out = torch.zeros(1, 1, device=dev, dtype=torch.float32)
        c(out, G, st)
        torch.cuda.synchronize()
        got = out[0, 0].item()
        expect = G * 64
        print(f"    G={G:5d}  expect={expect:7d}  got={got:9.1f}  ratio={got/expect:.4f}")


if __name__ == "__main__":
    main()
