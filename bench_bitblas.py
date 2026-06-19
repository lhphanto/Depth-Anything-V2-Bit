"""
Microbenchmark / config-prober: find a BitBLAS low-bit GEMM config that actually tunes on
this box for the real ViT-S (DINOv2-small) encoder linear shapes, then time it vs cuBLAS bf16.

Background: BitBLAS's int8-activation x int2-weight path (true "W2A8") is lightly supported
and may fail to tune ("No tuning config found"). Its flagship path is weight-only quant
(int2 weight, fp16 activations = "W2A16"). So for each shape we try a LADDER of variants and
use the first that builds, tunes, and matches a dense reference:

    1. w2a8-i32 : A=int8,  W=int2 -> int32 accumulation, dequant in torch   (true W2A8)
    2. w2a8-f16 : A=int8,  W=int2 -> float16                                  (true W2A8)
    3. w2a16    : A=fp16,  W=int2, in-kernel scaling                          (weight-only)

M defaults to 1370 = ViT token count at 518x518 ((518/14)^2 + 1 cls token).

The BitBLAS calls are marked `<<< BITBLAS API >>>`; the API drifts across releases. Run on A100.

Examples
--------
    python bench_bitblas.py                  # all 4 ViT-S shapes, M=1370, try all variants
    python bench_bitblas.py --N 1536 --K 384 # one custom shape
    python bench_bitblas.py --only w2a16     # only probe the weight-only variant
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

# Variant ladder. `act` = how activations are fed; `scaling` = scale applied inside the kernel.
VARIANTS = [
    dict(name='w2a8-i32', act='int8', scaling=False,
         cfg=dict(A_dtype='int8', W_dtype='int2', accum_dtype='int32', out_dtype='int32')),
    dict(name='w2a8-f16', act='int8', scaling=False,
         cfg=dict(A_dtype='int8', W_dtype='int2', accum_dtype='int32', out_dtype='float16')),
    dict(name='w2a16', act='fp16', scaling=True,
         cfg=dict(A_dtype='float16', W_dtype='int2', accum_dtype='float16', out_dtype='float16',
                  group_size=-1)),
]


def parse_args():
    p = argparse.ArgumentParser(description='BitBLAS low-bit GEMM prober/benchmark')
    p.add_argument('--M', type=int, default=1370, help='token count (rows of the activation)')
    p.add_argument('--N', type=int, default=None, help='out features (override the ViT-S preset)')
    p.add_argument('--K', type=int, default=None, help='in features (override the ViT-S preset)')
    p.add_argument('--only', type=str, default=None, help='probe only this variant name')
    p.add_argument('--iters', type=int, default=100)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--topk', type=int, default=20, help='# schedule candidates to tune over')
    p.add_argument('--atol', type=float, default=1e-1, help='max abs error in the numeric check')
    return p.parse_args()


def build_op(cfg_kwargs, M, N, K, topk):
    """Build + tune a BitBLAS Matmul op. Raises if it cannot tune. <<< BITBLAS API >>>"""
    import bitblas
    config = bitblas.MatmulConfig(
        M=M, N=N, K=K, layout='nt', with_bias=False, with_zeros=False, **cfg_kwargs)
    try:
        op = bitblas.Matmul(config=config, enable_tuning=True)   # tunes during construction
    except TypeError:
        op = bitblas.Matmul(config=config)
        op.hardware_aware_finetune(topk=topk)
    return op


def bench(fn, iters, warmup):
    """Mean latency in microseconds via CUDA events."""
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


def make_weight(N, K, device):
    """Ternary {-1,0,1} weight (int8) + scalar absmean scale (BitNet b1.58)."""
    w_fp = torch.randn(N, K, device=device)
    s = 1.0 / w_fp.abs().mean().clamp_(min=1e-5)
    w_int = (w_fp * s).round().clamp_(-1, 1).to(torch.int8)
    return w_int, 1.0 / s                                   # w_int [N,K], w_scale scalar


def try_variant(v, N, K, M, x, w_int, w_scale, args):
    """Build+tune the op for variant `v`, check numerics, time it. Returns a result dict."""
    device = x.device
    op = build_op(v['cfg'], M, N, K, args.topk)
    packed = op.transform_weight(w_int)                    # <<< BITBLAS API >>>
    y_ref_dense = (x.float() @ w_int.float().t()) * w_scale   # [M,N], pre activation-scale

    if v['act'] == 'int8':
        act_s = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        x_int = (x * act_s).round().clamp_(-128, 127).to(torch.int8)
        out = op(x_int, packed)                            # int32 or fp16 accumulation
        y_bb = out.float() * w_scale / act_s
        y_ref = y_ref_dense / act_s
        call = lambda: op(x_int, packed)
        def e2e():
            a_s = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
            xi = (x * a_s).round().clamp_(-128, 127).to(torch.int8)
            return op(xi, packed).float() * w_scale / a_s
    else:                                                  # weight-only: fp16 activations
        scale = torch.full((N, 1), float(w_scale), dtype=torch.float16, device=device)
        try:
            out = op(x, packed, scale)
            call = lambda: op(x, packed, scale)
        except TypeError:
            out = op(x, packed, scale=scale)
            call = lambda: op(x, packed, scale=scale)
        y_bb = out.float()
        y_ref = y_ref_dense
        e2e = call                                         # no separate activation quant

    max_err = (y_bb - y_ref).abs().max().item()
    rel = max_err / (y_ref.abs().max().item() + 1e-9)
    t_kernel = bench(call, args.iters, args.warmup)
    t_e2e = bench(e2e, args.iters, args.warmup)
    mem_packed = packed.numel() * packed.element_size()
    return dict(max_err=max_err, rel=rel, t_kernel=t_kernel, t_e2e=t_e2e, mem_packed=mem_packed)


def run_shape(name, N, K, M, args, variants):
    device = 'cuda'
    print(f'\n=== {name}  (M={M}, N={N}, K={K}) ===')
    x = torch.randn(M, K, device=device, dtype=torch.float16)
    w_int, w_scale = make_weight(N, K, device)

    # bf16 baseline (what we run today)
    w_bf16 = w_int.to(torch.bfloat16).t().contiguous()
    t_bf16 = bench(lambda: torch.matmul(x.to(torch.bfloat16), w_bf16), args.iters, args.warmup)
    mem_bf16 = w_int.numel() * 2
    print(f'  bf16 matmul (baseline): {t_bf16:8.2f} us   weight {mem_bf16/1024:.1f} KiB')

    for v in variants:
        print(f'  -- variant {v["name"]} (tuning, topk={args.topk}; first run slow)...')
        try:
            r = try_variant(v, N, K, M, x, w_int, w_scale, args)
        except Exception as e:
            print(f'     FAILED: {type(e).__name__}: {str(e)[:160]}')
            continue
        ok = r['max_err'] < args.atol
        print(f'     numeric: max_err={r["max_err"]:.4g} (rel={r["rel"]:.2e}) '
              f'{"OK" if ok else "MISMATCH (encoding/scale)"}')
        print(f'     kernel-only: {r["t_kernel"]:8.2f} us  ({t_bf16/r["t_kernel"]:.2f}x)  '
              f'end-to-end: {r["t_e2e"]:8.2f} us  ({t_bf16/r["t_e2e"]:.2f}x)  '
              f'weight {r["mem_packed"]/1024:.1f} KiB ({mem_bf16/max(r["mem_packed"],1):.2f}x smaller)')


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

    variants = [v for v in VARIANTS if not args.only or v['name'] == args.only]
    if not variants:
        raise SystemExit(f'--only {args.only!r} matched no variant; choose from '
                         f'{[v["name"] for v in VARIANTS]}')

    shapes = ([('custom', args.N, args.K)] if args.N and args.K else VITS_SHAPES)
    for name, N, K in shapes:
        run_shape(name, N, K, args.M, args, variants)

    print('\nDone. Use the first variant that prints OK. If only w2a16 works, BitBLAS '
          'gives weight compression + weight-only speedup; true W2A8 is unsupported here.')


if __name__ == '__main__':
    main()
