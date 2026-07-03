"""Probe the gfx950 f32 buffer-atomic doubling.

Findings so far: rocdl.raw_buffer_atomic_fadd doubles EXACTLY 2x even for a
single thread/single block (1 atomic add of 1.0 -> 2.0). ISA shows one
buffer_atomic_add_f32, so the instruction itself executes twice on this arch.

This probe checks:
  (A) does a 0.5 pre-scale cleanly compensate (2x * 0.5 = 1x)?  -> workaround
  (B) does the i32 atomic (atomicrmw add) also double?          -> f32-specific?

Run:
  HIP_VISIBLE_DEVICES=3 FLYDSL_RUNTIME_ENABLE_CACHE=0 \
  PYTHONPATH=/workspace/FlyDSL:$PYTHONPATH \
  python test/attention/fmha/probe_atomic_fadd.py
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


def build(add_val):
    @flyc.kernel(known_block_size=[64, 1, 1])
    def k(out: fx.Tensor):
        rsrc = _rsrc(out)
        rocdl.raw_buffer_atomic_fadd(
            _raw(fx.Float32(add_val)), rsrc,
            _raw(fx.Int32(0)), _raw(fx.Int32(0)), _raw(fx.Int32(0)),
        )

    @flyc.jit
    def launch(out: fx.Tensor, G: fx.Int32, stream: fx.Stream):
        k(out).launch(grid=(fx.Int32(G), 1, 1), block=(64, 1, 1), stream=stream)

    return launch


def main():
    dev = "cuda"
    st = torch.cuda.current_stream()
    G, T_ = 8, 64
    print("(A) f32 atomic add-value compensation test:")
    for add_val in [1.0, 0.5]:
        out = torch.zeros(1, 1, device=dev, dtype=torch.float32)
        c = flyc.compile(build(add_val), out, G, st)
        c(out, G, st)
        torch.cuda.synchronize()
        got = out[0, 0].item()
        expect = G * T_ * add_val
        print(f"    add={add_val}  expect={expect:7.1f}  got={got:8.1f}  ratio={got/expect:.4f}")


if __name__ == "__main__":
    main()
