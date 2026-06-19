"""
Microbenchmark: BitBLAS W2A8 (int2 weight x int8 activation) vs cuBLAS bf16, for the real
linear-layer shapes of a ViT-S (DINOv2-small) encoder.

This is "step 1" -- decide whether a maintained ternary GEMM is worth wiring into the
distilled student before investing in the full integration. For each shape it:

  1. builds a BitBLAS Matmul op for (M, N, K),
  2. ternarizes a random weight to {-1,0,1} (absmean scale) and packs it,
  3. checks the W2A8 output matches a dense fp reference (so we know the kernel + scales
     are wired correctly), and
  4. times W2A8 (kernel-only and end-to-end incl. activation quant) vs torch bf16 matmul,
     and reports the weight-memory ratio.

M defaults to 1370 = the ViT token count at 518x518 ((518/14)^2 + 1 cls token).

IMPORTANT: BitBLAS's Python API drifts across releases. The calls marked `BITBLAS API`
below (MatmulConfig fields, Matmul(...), transform_weight, the __call__ scale argument) may
need small tweaks for your installed version -- the script prints bitblas.__version__ at
startup so you can cross-check against its examples. Run this ON THE A100.

Examples
--------
    python bench_bitblas.py                       # all 4 ViT-S encoder shapes, M=1370
    python bench_bitblas.py --M 2740              # e.g. batch of 2 images
    python bench_bitblas.py --N 1536 --K 384      # one custom shape
"""

import argparse

import torch

# ViT-S / DINOv2-small (embed_dim = 384) encoder linear shapes as (name, N_out, K_in).
VITS_SHAPES = [
    ('attn.qkv',  1152, 384),
    ('attn.proj',  384, 384),
    ('mlp.fc1',   1536, 384),
    ('mlp.fc2',    384, 1536),
]


def parse_args():
    p = argparse.ArgumentParser(description='BitBLAS W2A8 vs bf16 microbenchmark')
    p.add_argument('--M', type=int, default=1370, help='token count (rows of the activation)')
    p.add_argument('--N', type=int, default=None, help='out features (override the ViT-S preset)')
    p.add_argument('--K', type=int, default=None, help='in features (override the ViT-S preset)')
    p.add_argument('--iters', type=int, default=100)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--atol', type=float, default=1e-1,
                   help='max abs error tolerated in the numeric check (int8 GEMM is exact, '
                        'but fp16 scale application introduces small rounding)')
    return p.parse_args()


def build_op(M, N, K):
    """Build a BitBLAS W2A8 GEMM operator. <<< BITBLAS API >>>"""
    import bitblas
    config = bitblas.MatmulConfig(
        M=M, N=N, K=K,
        A_dtype='int8',          # activations: int8
        W_dtype='int2',          # weights: signed 2-bit (covers ternary {-1,0,1})
        accum_dtype='int32',     # int8 x int2 accumulates in int32
        out_dtype='float16',     # dequantized output
        layout='nt',             # activation row-major, weight transposed (N,K)
        with_bias=False,
        with_scaling=True,       # apply a per-output-channel weight scale inside the kernel
        with_zeros=False,        # symmetric (no zero point) -- ternary is symmetric
        group_size=-1,           # one scale per output channel
    )
    # enable_tuning=True tunes+caches a kernel for this exact shape (slow first run only).
    return bitblas.Matmul(config=config)


def call_op(op, x_int, packed, scale):
    """Invoke the op, tolerating the two common __call__ signatures. <<< BITBLAS API >>>"""
    try:
        return op(x_int, packed, scale=scale)
    except TypeError:
        return op(x_int, packed, scale)


def bench(fn, iters, warmup):
    """Return mean latency in microseconds using CUDA events."""
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
    return start.elapsed_time(end) / iters * 1e3   # ms -> us


def run_shape(name, N, K, M, args):
    device = 'cuda'
    print(f'\n=== {name}  (M={M}, N={N}, K={K}) ===')

    # --- ternary weight {-1,0,1} via absmean (BitNet b1.58), one global scalar scale ---
    w_fp = torch.randn(N, K, device=device)
    s = 1.0 / w_fp.abs().mean().clamp_(min=1e-5)
    w_int = (w_fp * s).round().clamp_(-1, 1).to(torch.int8)          # [N, K] in {-1,0,1}
    w_scale = (1.0 / s).to(torch.float16).expand(N).contiguous()    # per-channel (all equal)

    # --- activations, per-token int8 quant ---
    x = torch.randn(M, K, device=device, dtype=torch.float16)
    act_s = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)   # [M, 1]
    x_int = (x * act_s).round().clamp_(-128, 127).to(torch.int8)     # [M, K]

    # --- build op + pack weight ---
    op = build_op(M, N, K)
    packed = op.transform_weight(w_int)                              # <<< BITBLAS API >>>

    # --- numeric check vs dense fp reference (same math the fake-quant path computes) ---
    y_bb = call_op(op, x_int, packed, w_scale).float() / act_s
    y_ref = (x_int.float() @ w_int.float().t()) * (1.0 / s) / act_s
    max_err = (y_bb - y_ref).abs().max().item()
    rel = max_err / (y_ref.abs().max().item() + 1e-9)
    ok = max_err < args.atol
    print(f'  numeric check: max_abs_err={max_err:.4g} (rel={rel:.2e})  '
          f'{"OK" if ok else "MISMATCH -- check scale wiring / encoding"}')

    # --- timing ---
    w_bf16 = w_int.to(torch.bfloat16).t().contiguous()              # [K, N], cuBLAS bf16
    x_bf16 = x.to(torch.bfloat16)

    t_bf16 = bench(lambda: torch.matmul(x_bf16, w_bf16), args.iters, args.warmup)
    t_kernel = bench(lambda: call_op(op, x_int, packed, w_scale), args.iters, args.warmup)

    def e2e():                                                      # incl. activation quant
        a_s = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        xi = (x * a_s).round().clamp_(-128, 127).to(torch.int8)
        return call_op(op, xi, packed, w_scale).float() / a_s
    t_e2e = bench(e2e, args.iters, args.warmup)

    mem_bf16 = w_int.numel() * 2                                    # bf16 weight bytes
    mem_packed = packed.numel() * packed.element_size()
    print(f'  bf16 matmul     : {t_bf16:8.2f} us')
    print(f'  W2A8 kernel-only: {t_kernel:8.2f} us   ({t_bf16 / t_kernel:.2f}x vs bf16)')
    print(f'  W2A8 end-to-end : {t_e2e:8.2f} us   ({t_bf16 / t_e2e:.2f}x vs bf16, incl act-quant)')
    print(f'  weight memory   : bf16 {mem_bf16/1024:.1f} KiB -> packed {mem_packed/1024:.1f} KiB '
          f'({mem_bf16 / max(mem_packed, 1):.2f}x smaller)')


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('This benchmark needs a CUDA GPU (run it on the A100).')

    try:
        import bitblas
        print(f'bitblas {getattr(bitblas, "__version__", "?")} | '
              f'torch {torch.__version__} | GPU {torch.cuda.get_device_name(0)}')
    except ImportError:
        raise SystemExit('bitblas not installed. pip install bitblas (pulls a TVM build).')

    shapes = ([('custom', args.N, args.K)] if args.N and args.K else VITS_SHAPES)
    for name, N, K in shapes:
        try:
            run_shape(name, N, K, args.M, args)
        except Exception as e:                     # one bad shape shouldn't kill the sweep
            print(f'  [error] {name} ({N}x{K}): {type(e).__name__}: {e}')

    print('\nDone. If kernel-only is <1x (slower) and you do not need the memory saving, '
          'the W2A8 path is not worth wiring into ViT-S.')


if __name__ == '__main__':
    main()
