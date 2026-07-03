"""Empirically determine mfma_f32_32x32x16_bf16 register layout on gfx950.

Two probes, one wave (64 lanes):
  Probe M: A[m,k]=m (row index), B[k,n]=1  -> C[m,n] = sum_k m*1 = 16*m
           => output value / 16 reveals M(lane,reg)
  Probe N: A[m,k]=1, B[k,n]=n              -> C[m,n] = sum_k 1*n = 16*n
           => output value / 16 reveals N(lane,reg)

We dump C for lanes 0,1,32,33 and all 16 regs, then print the decoded M,N.
"""
import sys
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr, buffer_ops
from flydsl.expr.typing import Vector as Vec

MFMA_K = 16
LK = 8


def build(mode):
    @flyc.kernel(known_block_size=[64, 1, 1])
    def kern(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        tid = fx.thread_idx.x
        lane = fx.Index(tid % 64)
        lmod = fx.Index(lane % 32)
        ldiv = fx.Index(lane // 32)

        A_rsrc = buffer_ops.create_buffer_resource(A)
        B_rsrc = buffer_ops.create_buffer_resource(B)
        C_buf = fx.rocdl.make_buffer_tensor(C)

        bf16 = fx.BFloat16
        v16f = Vec.make_type(16, fx.Float32)

        # A[m,k]: free=lmod (m), k=ldiv*8+e. A row-major [32,16].
        a_r = fx.make_rmem_tensor(LK, bf16)
        b_r = fx.make_rmem_tensor(LK, bf16)
        for e in range_constexpr(LK):
            m = lmod
            k = ldiv * LK + e
            a_idx = m * MFMA_K + k
            b_idx = k * 32 + lmod  # B[k,n]: free=lmod (n), k=ldiv*8+e -> B row-major [16,32]
            a_sc = buffer_ops.buffer_load(A_rsrc, a_idx, vec_width=1, dtype=bf16)
            b_sc = buffer_ops.buffer_load(B_rsrc, b_idx, vec_width=1, dtype=bf16)
            fx.memref_store(a_sc, a_r, e)
            fx.memref_store(b_sc, b_r, e)

        c0 = Vec.filled(16, 0.0, fx.Float32)
        acc = fx.rocdl.mfma_f32_32x32x16_bf16(
            v16f, [fx.memref_load_vec(a_r), fx.memref_load_vec(b_r), c0]
        )
        lane_i32 = fx.Int32(tid % 64)
        for r in range_constexpr(16):
            val = Vec(acc)[r]
            out_idx = lane_i32 * 16 + r
            r_out = fx.make_rmem_tensor(1, fx.Float32)
            fx.memref_store(val, r_out, 0)
            row_sl = fx.slice(C_buf, (out_idx, None))
            div_1 = fx.logical_divide(row_sl, fx.make_layout(1, 1))
            catom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
            fx.copy_atom_call(catom, r_out, fx.slice(div_1, (None, 0)))

    @flyc.jit
    def launch(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream):
        kern(A, B, C).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    return launch


def run(mode):
    dev = "cuda"
    A = torch.zeros(32, 16, dtype=torch.bfloat16, device=dev)
    B = torch.zeros(16, 32, dtype=torch.bfloat16, device=dev)
    if mode == "M":
        for m in range(32):
            A[m, :] = m
        B[:, :] = 1.0
    else:
        A[:, :] = 1.0
        for n in range(32):
            B[:, n] = n
    C = torch.zeros(64 * 16, 1, dtype=torch.float32, device=dev)
    A16 = A.view(torch.int16)
    B16 = B.view(torch.int16)
    ln = build(mode)
    comp = flyc.compile(ln, A16, B16, C, torch.cuda.current_stream())
    comp(A16, B16, C, torch.cuda.current_stream())
    torch.cuda.synchronize()
    Cv = C.view(64, 16).float() / 16.0
    print(f"=== Probe {mode} (value/16 = {mode}-coord) ===")
    for lane in [0, 1, 2, 32, 33]:
        vals = [int(round(Cv[lane, r].item())) for r in range(16)]
        print(f"  lane {lane:2d}: {vals}")


if __name__ == "__main__":
    run("M")
    run("N")
