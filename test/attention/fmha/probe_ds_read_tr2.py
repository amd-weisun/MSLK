"""Enhanced probe: test ds_read_tr16_b64 with per-lane distinct addresses.

Stores lds[m * D + d] = m * 100 + d (values 0..1599, fits in bf16 exactly).
Each lane computes two addresses matching our GEMM2 B-operand layout:
  m_row_a = m_base + lane_div_32 * 8 + tr_k_group
  m_row_b = m_base + lane_div_32 * 8 + tr_k_group + 4
  d_col   = tr_col_half * 16 + tr_col_sub * 4  (wave_d_sub=0 for simplicity)
Then calls ds_read_tr16_b64 at lds_a and lds_b, reads v4 each, concatenates.
Prints what each lane gets; we expect e-th element = dO[m_base + e, d_local(lane)]
where d_local = lane_mod_32 (after the transpose collapses the d-sub grouping).
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
M_BASE = 0   # which m-row group to start from
NROWS  = 16  # enough m-rows for the probe (m_base + lane_div_32*8 + tr_k_group + 4 <= 16 for lane_div_32=1)
LDS_N  = NROWS * D


def build():
    allocator = SmemAllocator(None, arch="gfx950", global_sym_name="probe_tr2_smem")
    lds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_off + LDS_N * 2   # bf16 = 2 bytes each

    @flyc.kernel(known_block_size=[64, 1, 1])
    def kern(C: fx.Tensor):
        tid = fx.thread_idx.x
        C_buf = fx.rocdl.make_buffer_tensor(C)
        bf16 = fx.BFloat16
        v4b = Vec.make_type(4, bf16)
        v8b = Vec.make_type(8, bf16)

        base_ptr = allocator.get_base()
        lds = SmemPtr(base_ptr, lds_off, bf16.ir_type, shape=(LDS_N,)).get()

        # --- Cooperative init: lds[m*D + d] = m*100 + d ---
        # 64 threads, each initializes 1 element (for m=0..0 only, covers NROWS*D/64 iters)
        tid_idx = fx.Index(tid)
        total = fx.Index(LDS_N)
        for i in range_constexpr(LDS_N // 64):
            g_idx = tid_idx + fx.Index(i * 64)
            m_i = fx.Index(g_idx // D)
            d_i = fx.Index(g_idx % D)
            val_i32 = fx.Int32(m_i * 100 + d_i)
            val_bf16 = Vec.from_elements([val_i32], fx.Int32).to(bf16)
            val_bf16.store(lds, [g_idx])
        fx.gpu.barrier()

        # --- Lane indices ---
        lane = fx.Index(tid % 64)
        lane_div_32 = fx.Index(lane // 32)
        lane_mod_32 = fx.Index(lane % 32)
        tr_k_group   = fx.Index((lane % 16) // 4)   # 0..3
        tr_col_sub   = fx.Index(lane % 4)            # 0..3
        tr_col_half  = fx.Index((lane % 32) // 16)   # 0 or 1

        # Address for lo call: m_row = M_BASE + lane_div_32 * 8 + tr_k_group
        m_row_a = fx.Index(M_BASE) + lane_div_32 * 8 + tr_k_group
        # Address for hi call: m_row_b = m_row_a + 4
        m_row_b = m_row_a + fx.Index(4)
        # d_col: the sub-group of 4 d-columns this lane addresses
        d_col = tr_col_half * 16 + tr_col_sub * 4

        lds_idx_a = m_row_a * D + d_col
        lds_idx_b = m_row_b * D + d_col

        byte_a = fx.Int64(lds_idx_a * 2 + lds_off)
        byte_b = fx.Int64(lds_idx_b * 2 + lds_off)
        ptr_a = _local_create_llvm_ptr(byte_a, address_space=3)
        ptr_b = _local_create_llvm_ptr(byte_b, address_space=3)

        res_a = rocdl.ds_read_tr16_b64(v4b, ptr_a).result
        res_b = rocdl.ds_read_tr16_b64(v4b, ptr_b).result

        # Combine into v8
        res_v8 = Vec(res_a).shuffle(Vec(res_b), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()

        lane_i32 = fx.Int32(lane)
        catom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        for e in range_constexpr(8):
            val = Vec(res_v8)[e].to(fx.Float32)
            out_idx = fx.Int32(lane_i32 * 8 + e)
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
    C = torch.zeros(64 * 8, 1, dtype=torch.float32, device=dev)
    launch = build()
    comp = flyc.compile(launch, C, torch.cuda.current_stream())
    comp(C, torch.cuda.current_stream())
    torch.cuda.synchronize()
    Cv = C.view(64, 8)
    print(f"ds_read_tr16_b64 probe: D={D}, M_BASE={M_BASE}")
    print("lds[m*D+d] = m*100 + d")
    print("Each lane points to: m_row_a = M_BASE + lane_div_32*8 + tr_k_group, d_col = tr_col_half*16 + tr_col_sub*4")
    print("Expected MFMA B-operand: element e = dO[M_BASE + e, lane_mod_32]")
    print("lane | lane_div | lane_mod | tr_k | tr_col | tr_ch | [e0..e7 decoded as m,d]")
    ok = 0
    fail = 0
    for lane in range(64):
        lane_div = lane // 32
        lane_mod = lane % 32
        tr_k = (lane % 16) // 4
        tr_col = lane % 4
        tr_ch = (lane % 32) // 16
        vals = [int(round(Cv[lane, e].item())) for e in range(8)]
        decoded = [(v // 100, v % 100) for v in vals]
        # Expected: e-th element = dO[M_BASE + e, lane_mod_32]
        expected_m = M_BASE
        expected_d = lane_mod
        label = ""
        for e, (m, d) in enumerate(decoded):
            exp_m = expected_m + e
            if m == exp_m and d == expected_d:
                pass
            else:
                label = f"FAIL e{e}: got ({m},{d}) want ({exp_m},{expected_d})"
                fail += 1
                break
        else:
            ok += 1
            label = "OK"
        if lane < 4 or label.startswith("FAIL") or lane in [16, 32, 48]:
            dec_str = " ".join(f"{m},{d}" for m, d in decoded)
            print(f"  {lane:2d} | div={lane_div} mod={lane_mod:2d} | tr_k={tr_k} tr_col={tr_col} tr_ch={tr_ch} | {dec_str} | {label}")
    print(f"\nSummary: {ok} lanes OK, {fail} lanes FAIL")
    if fail:
        print("Full output (first 8 lanes per group):")
        for lane in range(64):
            vals = [int(round(Cv[lane, e].item())) for e in range(8)]
            decoded = [(v // 100, v % 100) for v in vals]
            dec_str = " ".join(f"{m},{d}" for m, d in decoded)
            print(f"  {lane:2d} | {dec_str}")
