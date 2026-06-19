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
    p.add_argument('--topk', type=int, default=20,
                   help='# of schedule candidates BitBLAS tunes over (higher = slower tune, '
                        'possibly faster kernel)')
    p.add_argument('--atol', type=float, default=1e-1,
                   help='max abs error tolerated in the numeric check (int8 GEMM is exact, '
                        'but fp16 scale application introduces small rounding)')
    return p.parse_args()


def build_op(M, N, K, topk):
    """Build + tune a BitBLAS int2xint8 -> int32 GEMM operator. <<< BITBLAS API >>>

    We deliberately do NOT use in-kernel scaling (with_scaling=False): that path needs the
    'i2_to_i8_scale_offset' decode template, which has no schedule in this BitBLAS build.
    Instead the kernel returns the raw int32 accumulation and we apply the weight + activation
    scales in PyTorch afterward (same as the hand-written BitNet kernel's `/s*ws`).
    """
    import bitblas
    config = bitblas.MatmulConfig(
        M=M, N=N, K=K,
        A_dtype='int8',          # activations: int8
        W_dtype='int2',          # weights: signed 2-bit (covers ternary {-1,0,1})
        accum_dtype='int32',     # int8 x int2 accumulates in int32
        out_dtype='int32',       # return the raw accumulation; we dequantize outside
        layout='nt',             # activation row-major, weight transposed (N,K)
        with_bias=False,
        with_scaling=False,      # no fused decode+scale template -> avoids the failing path
        with_zeros=False,        # symmetric (ternary has no zero point)
    )
    # enable_tuning=True makes the constructor hardware-aware tune (compile a kernel for this
    # exact (M,N,K) on this GPU) instead of giving up when no default schedule exists.
    # Slow the first time (minutes), then cached to disk.
    print(f'  building + tuning (topk={topk}); first run is slow, result is cached...')
    try:
        op = bitblas.Matmul(config=config, enable_tuning=True, topk=topk)
    except TypeError:                      # older/newer signature without these kwargs
        op = bitblas.Matmul(config=config)
        op.hardware_aware_finetune(topk=topk)
    return op


def call_op(op, x_int, packed):
    """Invoke the int2xint8->int32 op. <<< BITBLAS API >>>"""
    return op(x_int, packed)


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
    w_scale = 1.0 / s                                               # = mean(|w|), scalar

    # --- activations, per-token int8 quant ---
    x = torch.randn(M, K, device=device, dtype=torch.float16)
    act_s = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)   # [M, 1]
    x_int = (x * act_s).round().clamp_(-128, 127).to(torch.int8)     # [M, K]

    # --- build op + pack weight ---
    op = build_op(M, N, K, args.topk)
    packed = op.transform_weight(w_int)                              # <<< BITBLAS API >>>

    # --- numeric check vs dense fp reference (same math the fake-quant path computes) ---
    # The kernel returns raw int32 accumulation; dequant = * w_scale / act_scale in torch.
    acc = call_op(op, x_int, packed).float()                        # [M, N] int32 accumulation
    y_bb = acc * w_scale / act_s
    y_ref = (x_int.float() @ w_int.float().t()) * w_scale / act_s
    max_err = (y_bb - y_ref).abs().max().item()
    rel = max_err / (y_ref.abs().max().item() + 1e-9)
    ok = max_err < args.atol
    print(f'  numeric check: max_abs_err={max_err:.4g} (rel={rel:.2e})  '
          f'{"OK" if ok else "MISMATCH -- check scale wiring / encoding"}')

    # --- timing ---
    w_bf16 = w_int.to(torch.bfloat16).t().contiguous()              # [K, N], cuBLAS bf16
    x_bf16 = x.to(torch.bfloat16)

    t_bf16 = bench(lambda: torch.matmul(x_bf16, w_bf16), args.iters, args.warmup)
    t_kernel = bench(lambda: call_op(op, x_int, packed), args.iters, args.warmup)

    def e2e():                                                      # incl. activation quant + dequant
        a_s = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        xi = (x * a_s).round().clamp_(-128, 127).to(torch.int8)
        return call_op(op, xi, packed).float() * w_scale / a_s
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
