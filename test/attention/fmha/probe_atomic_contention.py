"""Verify the 0.5-prescale atomic workaround under REAL contention: many blocks
concurrently atomic-add 0.5*v into the SAME address. Correct sum = G*T*v.

If the doubling were a race (not exact x2), heavy contention would show drift.
This confirms sum(2 * 0.5 * v) == sum(v) bit-exactly even with 512 blocks.

Run:
  HIP_VISIBLE_DEVICES=3 FLYDSL_RUNTIME_ENABLE_CACHE=0 \
  PYTHONPATH=/workspace/FlyDSL:$PYTHONPATH \
  python test/attention/fmha/probe_atomic_contention.py
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


def build(prescale):
    @flyc.kernel(known_block_size=[64, 1, 1])
    def k(out: fx.Tensor):
        rsrc = _rsrc(out)
        # each lane adds prescale * 1.0 into out[0]
        rocdl.raw_buffer_atomic_fadd(
            _raw(fx.Float32(prescale * 1.0)), rsrc,
            _raw(fx.Int32(0)), _raw(fx.Int32(0)), _raw(fx.Int32(0)),
        )

    @flyc.jit
    def launch(out: fx.Tensor, G: fx.Int32, stream: fx.Stream):
        k(out).launch(grid=(fx.Int32(G), 1, 1), block=(64, 1, 1), stream=stream)

    return launch


def main():
    dev = "cuda"
    st = torch.cuda.current_stream()
    G, T_ = 512, 64  # 512 blocks all hammering out[0]
    print("0.5-prescale under contention (512 blocks):")
    c = flyc.compile(build(0.5), torch.zeros(1, 1, device=dev), G, st)
    for trial in range(3):
        out = torch.zeros(1, 1, device=dev, dtype=torch.float32)
        c(out, G, st)
        torch.cuda.synchronize()
        got = out[0, 0].item()
        expect = G * T_  # 0.5*2 = 1 per lane
        print(f"    trial {trial}: expect={expect}  got={got:.1f}  exact={got == expect}")


if __name__ == "__main__":
    main()
