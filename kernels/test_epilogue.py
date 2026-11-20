import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import torch
import time
import triton
import kernels.factored_gemm as fg

x, V, U, v2, v1, u2, u1 = fg.make_inputs(4096, 12288, 824)
s = v1 * u2
h = torch.empty(824, dtype=torch.bfloat16, device="cuda")
y = torch.empty(12288, dtype=torch.bfloat16, device="cuda")
y2 = torch.empty_like(y)

fg._scale_v_kernel[(triton.cdiv(824, 32),)](x, V, v2, s, h, 4096, 824,
                                            BLOCK_IN=512, BLOCK_OUT=32, num_warps=4)
fg._scale_u_kernel[(triton.cdiv(12288, 64),)](h, U, u1, y, 824, 12288,
                                              BLOCK_R=1024, BLOCK_OUT=64, num_warps=4)
torch.cuda.synchronize()
fg._scale_v_kernel[(triton.cdiv(824, 32),)](x, V, v2, s, h, 4096, 824,
                                            BLOCK_IN=512, BLOCK_OUT=32, num_warps=4)
fg._scale_u_res_kernel[(triton.cdiv(12288, 64),)](h, U, u1, y2, y, 824, 12288,
                                                  BLOCK_R=1024, BLOCK_OUT=64, num_warps=4)
torch.cuda.synchronize()
err = ((y2.float() - 2 * y.float()).abs().max() / y.float().abs().max()).item()
print("epilogue(res=y) vs y+y rel err:", err)


def timeit(fn, rep=300):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(rep):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / rep * 1e6


t_a = timeit(lambda: fg._scale_u_kernel[(triton.cdiv(12288, 64),)](
    h, U, u1, y, 824, 12288, BLOCK_R=1024, BLOCK_OUT=64, num_warps=4))
t_b = timeit(lambda: fg._scale_u_res_kernel[(triton.cdiv(12288, 64),)](
    h, U, u1, y2, y, 824, 12288, BLOCK_R=1024, BLOCK_OUT=64, num_warps=4))
print(f"plain: {t_a:.1f} us | epilogue: {t_b:.1f} us")
