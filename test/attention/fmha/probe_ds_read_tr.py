"""Probe ds_read_tr16_b64 (HW-transpose LDS read) on gfx950.

Store a known ramp into LDS: lds[row*STRIDE + col] = row*100 + col (as f16-ish via bf16).
Actually use bf16 storing integer values 0..N so we can read them back exactly.
Each lane calls ds_read_tr16_b64 at base = row0*STRIDE + col0(lane), reads v4 bf16.
Record result[lane][e]; decode source (row,col) = value//100, value%100.

We probe ONE 16-lane block (lanes 0..15) and one 64-lane wave to see the 4x4 transpose.
"""
import sys
sys.setrecursionlimit(20000)
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as _raw
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.compiler.kernel_function import CompilationContext
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, vector


def _ds_read_tr16_b64_imm(result_type, addr_i32, imm_offset=0):
    """gfx950 ds_read_b64_tr_b16 with immediate byte offset.

    Mirrors FlyDSL production kernels/flash_attn_gfx950.py. Avoids the wrapped
    rocdl.ds_read_tr16_b64 + buffer_ops.create_llvm_ptr helper pair that trips
    standalone JIT dependency scanning.
    """
    imm = int(imm_offset)
    raw_type = ir.VectorType.get([2], ir.IntegerType.get_signless(32))
    raw = llvm.inline_asm(
        raw_type,
        [_raw(addr_i32)],
        f"ds_read_b64_tr_b16 $0, $1 offset:{imm}\n",
        "=v,v,~{memory}",
        has_side_effects=True,
    )
    return vector.BitCastOp(result_type, raw).result

STRIDE = 16   # LDS row stride (elements)
NROWS  = 16
LDS_N  = NROWS * STRIDE


def build():
    allocator = SmemAllocator(None, arch="gfx950", global_sym_name="probe_tr_smem")
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

        # Cooperative store: lds[i] = i. Decode row=i//16, col=i%16 (STRIDE=16).
        # bf16 exactly represents integers up to 256 -> LDS_N=256 boundary OK (0..255).
        tid_idx = fx.Index(tid)
        if tid_idx < fx.Index(LDS_N):
            vv = Vec.from_elements([fx.Int32(tid_idx)], fx.Int32).to(bf16)
            vv.store(lds, [tid_idx])
        fx.gpu.barrier()

        # HW-transpose read; all lanes same base addr (elem 0).
        b_i32 = fx.Int32(0 * 2 + lds_off)
        res = _ds_read_tr16_b64_imm(v4b, b_i32, 0)

        lane_i32 = fx.Int32(tid % 64)
        for e in range_constexpr(4):
            val = Vec(res)[e].to(fx.Float32)
            out_idx = lane_i32 * 4 + e
            r_out = fx.make_rmem_tensor(1, fx.Float32)
            fx.memref_store(val, r_out, 0)
            row_sl = fx.slice(C_buf, (out_idx, None))
            div_1 = fx.logical_divide(row_sl, fx.make_layout(1, 1))
            catom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
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
    print("ds_read_tr16_b64: value = row*16+col stored at lds[row*16+col]")
    print("lane: [e0(row,col) e1 e2 e3]")
    for lane in range(64):
        vals = [int(round(Cv[lane, e].item())) for e in range(4)]
        dec = " ".join(f"{v//16},{v%16}" for v in vals)
        print(f"  {lane:2d}: {dec}")
        if lane == 15:
            print("  --- (16-lane block boundary) ---")
        if lane == 31:
            print("  === (32-lane boundary) ===")
