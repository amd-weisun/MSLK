"""Systematic probe for mfma_f32_16x16x32_bf16 register layout on gfx950.

Run 1: A[lane, e] = lane_mod_16, B = 1 → output/K = m-coord
Run 2: A = 1, B[lane, e] = lane_mod_16 → output/K = n-coord
"""
import sys
sys.setrecursionlimit(20000)
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec

WARP_SIZE = 64
K_TOTAL   = 32


def build_m():
    @flyc.kernel(known_block_size=[WARP_SIZE, 1, 1])
    def kern(C: fx.Tensor):
        tid = fx.thread_idx.x
        C_buf = fx.rocdl.make_buffer_tensor(C)
        bf16 = fx.BFloat16
        f32  = fx.Float32
        v4f32  = Vec.make_type(4, f32)
        lane = fx.Index(tid % WARP_SIZE)
        lane_mod_16 = fx.Index(lane % 16)
        a_val = Vec.from_elements([lane_mod_16], fx.Int32).to(bf16)[0]
        one   = Vec.from_elements([fx.Int32(1)], fx.Int32).to(bf16)[0]
        a_pack = Vec.from_elements([a_val] * 8, bf16).ir_value()
        b_pack = Vec.from_elements([one] * 8, bf16).ir_value()
        zero_c = Vec.from_elements([fx.Float32(0.0)] * 4, f32).ir_value()
        c_out  = rocdl.mfma_f32_16x16x32_bf16(v4f32, [a_pack, b_pack, zero_c])
        catom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), f32)
        lane_i32 = fx.Int32(lane)
        for r in range_constexpr(4):
            rr = fx.make_rmem_tensor(1, f32)
            fx.memref_store(Vec(c_out)[r], rr, 0)
            fx.copy_atom_call(catom, rr, fx.slice(fx.logical_divide(fx.slice(C_buf, (fx.Int32(lane_i32 * 4 + r), None)), fx.make_layout(1, 1)), (None, 0)))

    @flyc.jit
    def launch(C: fx.Tensor, stream: fx.Stream):
        kern(C).launch(grid=(1, 1, 1), block=(WARP_SIZE, 1, 1), stream=stream)
    return launch


def build_n():
    @flyc.kernel(known_block_size=[WARP_SIZE, 1, 1])
    def kern(C: fx.Tensor):
        tid = fx.thread_idx.x
        C_buf = fx.rocdl.make_buffer_tensor(C)
        bf16 = fx.BFloat16
        f32  = fx.Float32
        v4f32  = Vec.make_type(4, f32)
        lane = fx.Index(tid % WARP_SIZE)
        lane_mod_16 = fx.Index(lane % 16)
        b_val = Vec.from_elements([lane_mod_16], fx.Int32).to(bf16)[0]
        one   = Vec.from_elements([fx.Int32(1)], fx.Int32).to(bf16)[0]
        a_pack = Vec.from_elements([one] * 8, bf16).ir_value()
        b_pack = Vec.from_elements([b_val] * 8, bf16).ir_value()
        zero_c = Vec.from_elements([fx.Float32(0.0)] * 4, f32).ir_value()
        c_out  = rocdl.mfma_f32_16x16x32_bf16(v4f32, [a_pack, b_pack, zero_c])
        catom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), f32)
        lane_i32 = fx.Int32(lane)
        for r in range_constexpr(4):
            rr = fx.make_rmem_tensor(1, f32)
            fx.memref_store(Vec(c_out)[r], rr, 0)
            fx.copy_atom_call(catom, rr, fx.slice(fx.logical_divide(fx.slice(C_buf, (fx.Int32(lane_i32 * 4 + r), None)), fx.make_layout(1, 1)), (None, 0)))

    @flyc.jit
    def launch(C: fx.Tensor, stream: fx.Stream):
        kern(C).launch(grid=(1, 1, 1), block=(WARP_SIZE, 1, 1), stream=stream)
    return launch


if __name__ == "__main__":
    dev = "cuda"
    lm = build_m()
    ln = build_n()
    Cm = torch.zeros(WARP_SIZE * 4, 1, dtype=torch.float32, device=dev)
    Cn = torch.zeros(WARP_SIZE * 4, 1, dtype=torch.float32, device=dev)
    cm = flyc.compile(lm, Cm, torch.cuda.current_stream())
    cn = flyc.compile(ln, Cn, torch.cuda.current_stream())
    cm(Cm, torch.cuda.current_stream())
    cn(Cn, torch.cuda.current_stream())
    torch.cuda.synchronize()
    Cvm = Cm.view(WARP_SIZE, 4)
    Cvn = Cn.view(WARP_SIZE, 4)
    print(f"mfma_f32_16x16x32_bf16 layout (K_total={K_TOTAL})")
    print("lane | m[r0..r3] | n[r0..r3]")
    for lane in range(WARP_SIZE):
        m = [round(Cvm[lane, r].item() / K_TOTAL) for r in range(4)]
        n = [round(Cvn[lane, r].item() / K_TOTAL) for r in range(4)]
        print(f"  {lane:2d} | {m[0]:2d} {m[1]:2d} {m[2]:2d} {m[3]:2d} | {n[0]:2d} {n[1]:2d} {n[2]:2d} {n[3]:2d}")
        if lane in [15, 31, 47]:
            print("  ---")
