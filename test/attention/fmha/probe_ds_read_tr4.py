"""Probe4: Systematic study of ds_read_tr16_b64 with all-same-row addresses.

All 64 lanes read from the SAME row (m=0) but DIFFERENT d-columns based on lane_mod_32.
This isolates the d-column mapping.

Address for lane L: lds[0 * D + lane_mod_32]  (each lane reads at its own d-position)
"""
import sys
sys.setrecursionlimit(20000)
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.compiler.kernel_function import CompilationContext
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as std_arith
from flydsl._mlir.dialects import llvm
from flydsl._mlir.extras import types as T


def _local_local_create_llvm_ptr(value, address_space: int = 3):
    raw = value.ir_value() if hasattr(value, "ir_value") else value
    if isinstance(raw.type, ir.IndexType):
        raw = std_arith.IndexCastOp(T.i64(), raw).result
    ptr_type = ir.Type.parse(f"!llvm.ptr<{address_space}>")
    return llvm.IntToPtrOp(ptr_type, raw).result

D = 64
NROWS  = 32
LDS_N  = NROWS * D


def build():
    allocator = SmemAllocator(None, arch="gfx950", global_sym_name="probe_tr4_smem")
    lds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_off + LDS_N * 2

    @flyc.kernel(known_block_size=[64, 1, 1])
    def kern(C: fx.Tensor):
        tid = fx.thread_idx.x
        C_buf = fx.rocdl.make_buffer_tensor(C)
        bf16 = fx.BFloat16
        v4b = Vec.make_type(4, bf16)

        base_ptr = allocator.get_base()
        lds = SmemPtr(base_ptr, lds_off, bf16.ir_type, shape=(LDS_N,)).get()

        # Cooperative init: lds[m*D + d] = m*100 + d
        tid_idx = fx.Index(tid)
        for i in range_constexpr(LDS_N // 64):
            g_idx = tid_idx + fx.Index(i * 64)
            m_i = fx.Index(g_idx // D)
            d_i = fx.Index(g_idx % D)
            val_i32 = fx.Int32(m_i * 100 + d_i)
            val_bf16 = Vec.from_elements([val_i32], fx.Int32).to(bf16)
            val_bf16.store(lds, [g_idx])
        fx.gpu.barrier()

        lane = fx.Index(tid % 64)
        lane_mod_32 = fx.Index(lane % 32)

        # Address: each lane reads from lds[0*D + lane_mod_32]
        # This tests which d-column each lane's position maps to after transpose.
        lds_idx = fx.Index(0) * D + lane_mod_32
        byte_off = fx.Int64(lds_idx * 2 + lds_off)
        ptr = _local_create_llvm_ptr(byte_off, address_space=3)
        res = rocdl.ds_read_tr16_b64(v4b, ptr).result

        lane_i32 = fx.Int32(lane)
        catom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        for e in range_constexpr(4):
            val = Vec(res)[e].to(fx.Float32)
            out_idx = fx.Int32(lane_i32 * 4 + e)
            r_out = fx.make_rmem_tensor(1, fx.Float32)
            fx.memref_store(val, r_out, 0)
            row_sl = fx.slice(C_buf, (out_idx, None))
            div_1 = fx.logical_divide(row_sl, fx.make_layout(1, 1))
            fx.copy_atom_call(catom, r_out, fx.slice(div_1, (None, 0)))

    @flyc.jit
    def launch(C: fx.Tensor, stream: fx.Stream):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        kern(C).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    return launch


if __name__ == "__main__":
    dev = "cuda"
    C = torch.zeros(64 * 4, 1, dtype=torch.float32, device=dev)
    launch = build()
    comp = flyc.compile(launch, C, torch.cuda.current_stream())
    comp(C, torch.cuda.current_stream())
    torch.cuda.synchronize()
    Cv = C.view(64, 4)
    print("Probe4: addr = lds[0*D + lane_mod_32], shows d-column mapping")
    print("Lane | [e0,e1,e2,e3] decoded as (m,d)")
    for lane in range(32):
        lane_div = lane // 32
        lane_mod = lane % 32
        tr_k = (lane % 16) // 4
        tr_col = lane % 4
        tr_ch = (lane % 32) // 16
        vals = [int(round(Cv[lane, e].item())) for e in range(4)]
        decoded = [(v // 100, v % 100) for v in vals]
        dec_str = " ".join(f"{m},{d}" for m, d in decoded)
        print(f"  {lane:2d}(mod={lane_mod:2d}, tr_k={tr_k}, tr_col={tr_col}, tr_ch={tr_ch}): {dec_str}")
