"""Probe mfma_f32_16x16x32_bf16 register layout on gfx950.

Feed A[lane] = lane%32 repeated 16 times, B = 1.0 repeated 16 times, C = 0.
Output C[lane,r] = (lane%32) * 32 if m-coord = lane%32,
                 = (lane//32)*16 + r if m-coord depends on r.
Divide by 32 to recover the m-coordinate per (lane, r).
"""
import sys
sys.setrecursionlimit(20000)
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec

WARP_SIZE = 64


def build():
    @flyc.kernel(known_block_size=[WARP_SIZE, 1, 1])
    def kern(C: fx.Tensor):
        tid = fx.thread_idx.x
        C_buf = fx.rocdl.make_buffer_tensor(C)
        bf16 = fx.BFloat16
        f32  = fx.Float32
        v4f32   = Vec.make_type(4, f32)    # 16x16x32 output: 4 f32 per lane
        v8bf16  = Vec.make_type(8, bf16)   # 8 bf16 per lane for A/B input (MFMA_LK=8)

        lane = fx.Index(tid % WARP_SIZE)
        lane_mod_32 = fx.Index(lane % 32)

        # A[lane] = lane%32 (as bf16) repeated 8 times
        a_val  = Vec.from_elements([lane_mod_32], fx.Int32).to(bf16)[0]
        a_pack = Vec.from_elements([a_val] * 8, bf16).ir_value()

        # B[lane] = 1.0 repeated 8 times
        b_one  = Vec.from_elements([fx.Int32(1)], fx.Int32).to(bf16)[0]
        b_pack = Vec.from_elements([b_one] * 8, bf16).ir_value()

        zero_c = Vec.from_elements([fx.Float32(0.0)] * 4, f32).ir_value()
        c_out = rocdl.mfma_f32_16x16x32_bf16(v4f32, [a_pack, b_pack, zero_c])

        # Store all 4 output values: [lane*4+r] = output[r]
        catom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), f32)
        lane_i32 = fx.Int32(lane)
        for r in range_constexpr(4):
            val = Vec(c_out)[r]
            out_idx = fx.Int32(lane_i32 * 4 + r)
            rr = fx.make_rmem_tensor(1, f32)
            fx.memref_store(val, rr, 0)
            row_sl = fx.slice(C_buf, (out_idx, None))
            div_1 = fx.logical_divide(row_sl, fx.make_layout(1, 1))
            fx.copy_atom_call(catom, rr, fx.slice(div_1, (None, 0)))

    @flyc.jit
    def launch(C: fx.Tensor, stream: fx.Stream):
        kern(C).launch(grid=(1, 1, 1), block=(WARP_SIZE, 1, 1), stream=stream)

    return launch


if __name__ == "__main__":
    dev = "cuda"
    C = torch.zeros(WARP_SIZE * 4, 1, dtype=torch.float32, device=dev)
    launch = build()
    comp = flyc.compile(launch, C, torch.cuda.current_stream())
    comp(C, torch.cuda.current_stream())
    torch.cuda.synchronize()
    Cv = C.view(WARP_SIZE, 4)
    print("mfma_f32_16x16x32_bf16: A[lane]=lane%32 x16, B=1.0 x16, C_init=0")
    print("Output[lane, r]: if m-coord = lane%32, output = lane%32 * 32")
    print("                 if m-coord depends on (lane,r), output = m_coord * 32")
    print("lane | r0     r1     r2     r3   | r0/32  r1/32  r2/32  r3/32 (m-coord?)")
    for lane in range(WARP_SIZE):
        vals = [Cv[lane, r].item() for r in range(4)]
        div32 = [v/32 for v in vals]
        print(f"  {lane:2d}: {vals[0]:6.1f} {vals[1]:6.1f} {vals[2]:6.1f} {vals[3]:6.1f} | "
              f"{div32[0]:5.1f}  {div32[1]:5.1f}  {div32[2]:5.1f}  {div32[3]:5.1f}")
        if lane == 31:
            print("  --- (lane 32 boundary) ---")
