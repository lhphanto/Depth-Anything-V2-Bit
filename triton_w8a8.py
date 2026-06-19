"""
Triton W8A8 GEMM for the 1.58-bit student: int8 activations x int8 (ternary {-1,0,1}) weights,
int32 tensor-core accumulation, dequantized to fp16.

Why W8A8 and not "W2A8": in every low-bit kernel (BitNet's hand-written one included) the
tensor cores multiply int8 x int8 -- the 2-bit weights are just a storage format that is
*decoded to int8* before the MAC. So the compute speedup comes from int8 tensor cores (A100:
~624 TOPS int8 vs ~312 TFLOPS bf16), which this kernel uses directly. The extra weight-bandwidth
saving from 2-bit packing is small at K=384 and is already captured by the storage packer in
bitnet.py. Adding in-kernel int2 unpacking is a later refinement, not where the speed is.

This file is a prototype + harness: it checks the kernel against a fp reference and times it
vs cuBLAS bf16 for the real ViT-S encoder shapes. Run on the A100 (needs triton + CUDA).

    python triton_w8a8.py
"""

import argparse

import torch

import triton
import triton.language as tl


# ViT-S / DINOv2-small (embed_dim=384) encoder linear shapes as (name, N_out, K_in).
VITS_SHAPES = [
    ('attn.qkv',  1152, 384),
    ('attn.proj',  384, 384),
    ('mlp.fc1',   1536, 384),
    ('mlp.fc2',    384, 1536),
]


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': bm, 'BLOCK_N': bn, 'BLOCK_K': bk, 'GROUP_M': 8},
                      num_warps=nw, num_stages=ns)
        for bm in (64, 128) for bn in (64, 128, 256) for bk in (64, 128)
        for nw, ns in ((4, 3), (8, 4))
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _w8a8_kernel(
    x_ptr, w_ptr, y_ptr, act_scale_ptr, w_scale,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
):
    # C[M,N] = (A_int8[M,K] @ W_int8[N,K]^T) * w_scale / act_scale[M]
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    # Group program ordering so neighboring blocks reuse the same rows in L2.
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)  # W^T tile [BK,BN]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(x_ptrs, mask=offs_k[None, :] < k_remaining, other=0)   # int8 [BM,BK]
        b = tl.load(w_ptrs, mask=offs_k[:, None] < k_remaining, other=0)   # int8 [BK,BN]
        acc += tl.dot(a, b, out_dtype=tl.int32)                            # int8 tensor cores
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # Dequantize: per-token activation scale (a row vector) and a scalar weight scale.
    offs_m2 = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n2 = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    a_scale = tl.load(act_scale_ptr + offs_m2, mask=offs_m2 < M, other=1.0)   # s per row
    y = acc.to(tl.float32) * w_scale / a_scale[:, None]
    y = y.to(tl.float16)

    y_ptrs = y_ptr + offs_m2[:, None] * stride_ym + offs_n2[None, :] * stride_yn
    tl.store(y_ptrs, y, mask=(offs_m2[:, None] < M) & (offs_n2[None, :] < N))


def w8a8_matmul(x_int8, w_int8, act_scale, w_scale):
    """C[M,N] = (x_int8 @ w_int8^T) * w_scale / act_scale, returned fp16.

    x_int8: [M,K] int8, w_int8: [N,K] int8 (ternary), act_scale: [M] fp32, w_scale: python float.
    """
    M, K = x_int8.shape
    N, _ = w_int8.shape
    y = torch.empty((M, N), device=x_int8.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    _w8a8_kernel[grid](
        x_int8, w_int8, y, act_scale, float(w_scale),
        M, N, K,
        x_int8.stride(0), x_int8.stride(1),
        w_int8.stride(0), w_int8.stride(1),
        y.stride(0), y.stride(1),
    )
    return y


def quant_act(x):
    """Per-token int8 activation quant. Returns (x_int8 [M,K], s [M])."""
    s = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5)   # [M,1]
    x_int8 = (x * s).round().clamp_(-128, 127).to(torch.int8)
    return x_int8, s.squeeze(-1)                                     # s: [M]


def w8a8_linear(x, w_int8, w_scale):
    """End-to-end: quantize fp16 activations, run the kernel. (No SubLN here -- prototype.)"""
    x_int8, s = quant_act(x)
    return w8a8_matmul(x_int8, w_int8, s, w_scale)


def make_ternary_weight(N, K, device):
    """Ternary {-1,0,1} weight as int8 + scalar absmean scale mean(|w|)."""
    w_fp = torch.randn(N, K, device=device)
    scale = w_fp.abs().mean().clamp_min(1e-5)                        # = mean(|w|)
    w_int8 = (w_fp / scale).round().clamp_(-1, 1).to(torch.int8)
    return w_int8, scale.item()


def bench(fn, iters=100, warmup=25):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1e3   # us


def run_shape(name, N, K, M, args):
    device = 'cuda'
    print(f'\n=== {name}  (M={M}, N={N}, K={K}) ===')
    x = torch.randn(M, K, device=device, dtype=torch.float16)
    w_int8, w_scale = make_ternary_weight(N, K, device)

    # correctness vs exact-integer fp reference
    x_int8, s = quant_act(x)
    y_tri = w8a8_matmul(x_int8, w_int8, s, w_scale)
    y_ref = (x_int8.float() @ w_int8.float().t()) * w_scale / s[:, None]
    max_err = (y_tri.float() - y_ref).abs().max().item()
    rel = max_err / (y_ref.abs().max().item() + 1e-9)
    print(f'  numeric: max_err={max_err:.4g} (rel={rel:.2e}) '
          f'{"OK" if rel < 1e-2 else "MISMATCH"}')

    # timing: cuBLAS bf16 baseline vs the Triton W8A8 (kernel-only and end-to-end)
    w_bf16 = w_int8.to(torch.bfloat16).t().contiguous()
    x_bf16 = x.to(torch.bfloat16)
    t_bf16 = bench(lambda: torch.matmul(x_bf16, w_bf16))
    t_kernel = bench(lambda: w8a8_matmul(x_int8, w_int8, s, w_scale))
    t_e2e = bench(lambda: w8a8_linear(x, w_int8, w_scale))
    print(f'  bf16 matmul     : {t_bf16:8.2f} us')
    print(f'  W8A8 kernel-only: {t_kernel:8.2f} us   ({t_bf16 / t_kernel:.2f}x vs bf16)')
    print(f'  W8A8 end-to-end : {t_e2e:8.2f} us   ({t_bf16 / t_e2e:.2f}x vs bf16, incl act-quant)')
    best = _w8a8_kernel.best_config
    print(f'  best config: {best}')


def main():
    p = argparse.ArgumentParser(description='Triton W8A8 GEMM prototype/benchmark')
    p.add_argument('--M', type=int, default=1370)
    p.add_argument('--N', type=int, default=None)
    p.add_argument('--K', type=int, default=None)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('Needs a CUDA GPU (run on the A100).')
    print(f'torch {torch.__version__} | triton {triton.__version__} | '
          f'GPU {torch.cuda.get_device_name(0)}')

    shapes = ([('custom', args.N, args.K)] if args.N and args.K else VITS_SHAPES)
    for name, N, K in shapes:
        run_shape(name, N, K, args.M, args)

    print('\nDone. kernel-only >1x means the int8 tensor-core GEMM beats cuBLAS bf16 at this '
          'shape. end-to-end includes per-token activation quant (paid every layer in the model).')


if __name__ == '__main__':
    main()
