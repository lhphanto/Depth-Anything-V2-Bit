"""
Measure ternary (1.58-bit) CPU matmul vs a tuned BLAS baseline, across regimes.

The CPU win from ternary weights is a *memory-bandwidth* win: ternary weights are tiny, so
when you are bandwidth-bound -- big weights that exceed cache AND low batch (few rows reusing
each weight, e.g. single-token decode) -- streaming 8-16x fewer weight bytes is faster. In the
compute-bound regime (large M, heavy weight reuse), a tuned BLAS (numpy @) wins, same as on GPU.

So this benchmarks several (shape x M) points so you can SEE the crossover:
  - ViT-S encoder shapes (your model): weights are small (fit in cache) -> ternary won't help.
  - An LLM-scale shape (big weights > cache): at M=1 (memory-bound) ternary can win.

Baseline = numpy fp32 matmul (calls your BLAS). Ternary kernels are numba-JIT'd:
  - sign-based: s += a where w=+1, s -= a where w=-1   (no multiply)
  - packed:     same, reading 2-bit-packed weights (the real bandwidth saving)

Single-threaded by default (env vars set below) for an apples-to-apples algorithm comparison;
export OMP_NUM_THREADS etc. yourself to test multi-thread.

    pip install numba
    python bench_cpu_ternary.py
"""

import os
# Pin to 1 thread BEFORE importing numpy, so BLAS doesn't mask the algorithmic difference with
# multi-core. Override by exporting these yourself before running.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import time

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False


SHAPES = [
    # name,           N,    K,    note
    ('vits.qkv',     1152,  384, 'ViT-S (small weights, fit in cache)'),
    ('vits.fc1',     1536,  384, 'ViT-S'),
    ('llm.ffn',     11008, 4096, 'LLM-scale (big weights, exceed cache)'),
]
M_VALUES = [1, 1370]   # 1 = single token (memory-bound), 1370 = ViT batch (compute-bound)


if HAVE_NUMBA:
    @njit(cache=True, fastmath=True)
    def ternary_matmul(A, W, C):
        """C[M,N] = sum_k A[m,k]*W[n,k], W in {-1,0,1}. Add/subtract only, no multiply."""
        M, K = A.shape
        N = W.shape[0]
        for n in range(N):
            for m in range(M):
                s = 0.0
                for k in range(K):
                    w = W[n, k]
                    if w > 0:
                        s += A[m, k]
                    elif w < 0:
                        s -= A[m, k]
                C[m, n] = s

    @njit(cache=True, fastmath=True)
    def ternary_matvec_lut(x, codes, K, y):
        """TL1-style LUT matvec, K blocked into quads (g=4).

        For each K-block of 4 activations, build a 81-entry table of partial sums for every
        possible quad of ternary weights (3^4=81). Then every output row is just a gather +
        add into that table -- no multiply, no branch in the hot loop. The table is built ONCE
        per block and reused across all N rows (that amortization is the whole point).
        `codes`: uint8 [num_blocks, N], each a base-3 index of 4 ternary weights (built offline).
        """
        nb, N = codes.shape
        for n in range(N):
            y[n] = 0.0
        T = np.empty(81, dtype=np.float32)
        for b in range(nb):
            base = b * 4
            # Build the table for this block (small; amortized across all N rows below).
            for c in range(81):
                cc = c
                s = 0.0
                for j in range(4):
                    d = cc % 3 - 1            # {-1,0,1}
                    cc //= 3
                    kk = base + j
                    if kk < K:
                        s += d * x[kk]        # multiply only here, in the cheap build step
                T[c] = s
            cb = codes[b]                     # [N] contiguous
            for n in range(N):                # hot loop: pure gather + add
                y[n] += T[cb[n]]

    @njit(cache=True, fastmath=True)
    def ternary_matmul_lut(A, codes, K, C):
        """M>1 LUT GEMM: same TL1 scheme, but the table is rebuilt for each activation row.

        The per-block table depends on the activation, so for M rows it is built M times -- the
        across-N amortization survives, but the across-M one does not. That is exactly why LUT
        favors small M (single-token decode); watch it lose to BLAS as M grows.
        """
        M = A.shape[0]
        nb, N = codes.shape
        T = np.empty(81, dtype=np.float32)
        for m in range(M):
            x = A[m]
            for n in range(N):
                C[m, n] = 0.0
            for b in range(nb):
                base = b * 4
                for c in range(81):
                    cc = c
                    s = 0.0
                    for j in range(4):
                        d = cc % 3 - 1
                        cc //= 3
                        kk = base + j
                        if kk < K:
                            s += d * x[kk]
                    T[c] = s
                cb = codes[b]
                for n in range(N):
                    C[m, n] += T[cb[n]]

    @njit(cache=True, fastmath=True)
    def ternary_matvec_packed(x, Wp, K, y):
        """y[N] = W @ x, with W stored 2-bit-packed (4 weights/byte, code 0/1/2 -> -1/0/1).

        This reads ~16x fewer weight bytes than fp32 -- the bandwidth saving. Unpack + add/sub
        inline; no multiply.
        """
        N, KB = Wp.shape
        for n in range(N):
            s = 0.0
            for b in range(KB):
                byte = Wp[n, b]
                base = b * 4
                for j in range(4):
                    k = base + j
                    if k < K:
                        code = (byte >> (2 * j)) & 3      # 0->-1, 1->0, 2->+1
                        if code == 2:
                            s += x[k]
                        elif code == 0:
                            s -= x[k]
            y[n] = s


def pack_ternary_np(w_int):
    """[N,K] in {-1,0,1} -> uint8 [N, ceil(K/4)], codes {0,1,2}, 4/byte (matches bitnet.py)."""
    N, K = w_int.shape
    u = (w_int + 1).astype(np.uint8)
    pad = (-K) % 4
    if pad:
        u = np.pad(u, ((0, 0), (0, pad)))
    u = u.reshape(N, (K + pad) // 4, 4)
    return (u[..., 0] | (u[..., 1] << 2) | (u[..., 2] << 4) | (u[..., 3] << 6)).copy()


def encode_lut_codes(w_int):
    """[N,K] in {-1,0,1} -> uint8 [num_blocks, N] base-3 quad codes for ternary_matvec_lut.

    Each entry packs 4 ternary weights as w0' + 3*w1' + 9*w2' + 27*w3' (w'=w+1), range 0..80.
    Transposed to [num_blocks, N] so the LUT hot loop reads each block's codes contiguously.
    """
    N, K = w_int.shape
    nb = (K + 3) // 4
    codes = np.zeros((nb, N), dtype=np.uint8)
    pow3 = (1, 3, 9, 27)
    for b in range(nb):
        for j in range(4):
            k = b * 4 + j
            if k < K:
                codes[b] += (w_int[:, k].astype(np.uint8) + 1) * pow3[j]
    return codes


def timeit(fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1e6   # us


def run(name, N, K, M, note, iters):
    A = np.random.randn(M, K).astype(np.float32)
    w_int = np.random.randint(-1, 2, size=(N, K)).astype(np.int8)
    Wf = w_int.astype(np.float32)
    weight_mb = N * K * 4 / 1e6
    print(f'\n=== {name}  M={M} N={N} K={K}  ({note}; fp32 weight {weight_mb:.1f} MB) ===')

    t_blas = timeit(lambda: A @ Wf.T, iters)
    print(f'  numpy fp32 (BLAS)   : {t_blas:9.1f} us')

    if not HAVE_NUMBA:
        print('  (numba not installed -> skipping ternary kernels; pip install numba)')
        return

    C = np.empty((M, N), dtype=np.float32)
    ref = A @ Wf.T
    ternary_matmul(A, w_int, C)                       # also triggers JIT compile
    err = np.abs(C - ref).max()
    t_tern = timeit(lambda: ternary_matmul(A, w_int, C), iters)
    print(f'  ternary add/sub     : {t_tern:9.1f} us   ({t_blas/t_tern:.2f}x vs BLAS)  '
          f'max_err={err:.2e}')

    if M == 1:                                         # matvec: the memory-bound regime
        Wp = pack_ternary_np(w_int)
        y = np.empty(N, dtype=np.float32)
        ternary_matvec_packed(A[0], Wp, K, y)
        err2 = np.abs(y - ref[0]).max()
        t_pk = timeit(lambda: ternary_matvec_packed(A[0], Wp, K, y), iters)
        print(f'  ternary packed 2bit : {t_pk:9.1f} us   ({t_blas/t_pk:.2f}x vs BLAS)  '
              f'weight {Wp.nbytes/1e6:.2f} MB ({Wf.nbytes/Wp.nbytes:.0f}x smaller)  '
              f'max_err={err2:.2e}')

        codes = encode_lut_codes(w_int)
        yl = np.empty(N, dtype=np.float32)
        ternary_matvec_lut(A[0], codes, K, yl)         # JIT compile
        err3 = np.abs(yl - ref[0]).max()
        t_lut = timeit(lambda: ternary_matvec_lut(A[0], codes, K, yl), iters)
        print(f'  ternary LUT (TL1)   : {t_lut:9.1f} us   ({t_blas/t_lut:.2f}x vs BLAS, '
              f'{t_pk/t_lut:.2f}x vs add/sub)  weight {codes.nbytes/1e6:.2f} MB  '
              f'max_err={err3:.2e}')


def sweep(N, K, m_values, iters):
    """Sweep M on a big-weight shape: watch ternary LUT/BLAS cross 1x as M grows."""
    if not HAVE_NUMBA:
        print('numba not installed -> skipping sweep')
        return
    print(f'\n=== M sweep,  N={N} K={K}  (fp32 weight {N*K*4/1e6:.0f} MB, exceeds cache) ===')
    print(f'{"M":>6} {"BLAS us":>11} {"add/sub us":>11} {"LUT us":>11} '
          f'{"LUT/BLAS":>9} {"LUT/addsub":>11}')
    w_int = np.random.randint(-1, 2, size=(N, K)).astype(np.int8)
    Wf = w_int.astype(np.float32)
    codes = encode_lut_codes(w_int)
    for M in m_values:
        A = np.random.randn(M, K).astype(np.float32)
        C = np.empty((M, N), dtype=np.float32)
        it = max(2, iters // max(1, M // 8))           # fewer iters as M (and cost) grows
        t_blas = timeit(lambda: A @ Wf.T, it)
        ternary_matmul(A, w_int, C)
        t_add = timeit(lambda: ternary_matmul(A, w_int, C), it)
        ternary_matmul_lut(A, codes, K, C)
        t_lut = timeit(lambda: ternary_matmul_lut(A, codes, K, C), it)
        print(f'{M:>6} {t_blas:>11.1f} {t_add:>11.1f} {t_lut:>11.1f} '
              f'{t_blas/t_lut:>8.2f}x {t_add/t_lut:>10.2f}x')


def main():
    p = argparse.ArgumentParser(description='CPU ternary vs BLAS benchmark')
    p.add_argument('--iters', type=int, default=20)
    p.add_argument('--sweep', action='store_true',
                   help='sweep M on a big-weight shape instead of the per-shape table')
    p.add_argument('--sweep-N', type=int, default=4096)
    p.add_argument('--sweep-K', type=int, default=4096)
    args = p.parse_args()
    print(f'numpy {np.__version__} | threads pinned to '
          f'{os.environ.get("OMP_NUM_THREADS")} | numba={HAVE_NUMBA}')

    if args.sweep:
        sweep(args.sweep_N, args.sweep_K, [1, 4, 16, 64, 256], args.iters)
        print('\nExpect LUT/BLAS to fall through 1x as M grows: ternary wins only while '
              'memory-bound (small M); BLAS wins once compute-bound (large M).')
        return

    for name, N, K, note in SHAPES:
        for M in M_VALUES:
            run(name, N, K, M, note, args.iters)
    print('\nWatch for: ternary wins (>1x) only at M=1 on the LLM shape (memory-bound, big '
          'weights). On ViT-S it loses -- weights fit in cache, so there is no bandwidth to save.')


if __name__ == '__main__':
    main()
