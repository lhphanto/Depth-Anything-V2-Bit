# 1.58-bit Depth Anything V2 — Distillation & Efficiency Report

A quantized (BitNet b1.58, ternary `{-1,0,1}`) DAV2-Small student is distilled from a frozen
full-precision DAV2-Large teacher on unlabeled images. This report summarizes the **accuracy**
of the quantized student and the **latency** experiments, and explains why the 1.58-bit model
does **not** run faster than the bf16 baseline at this model scale.

- **Teacher:** DAV2-Large (frozen), produces pseudo depth labels.
- **Student:** DAV2-Small, DINOv2 encoder `nn.Linear` layers replaced by ternary `BitLinear`.
- **Quantization scheme:** absmean weight ternarization + per-token int8 activations, STE in
  training (`bitnet.py`). Numerically exact fake-quant / fold / packed inference paths.

---

## 1. Accuracy on DA-2K

DA-2K: 2068 pairwise relative-depth annotations over 1033 images; metric = pairwise accuracy
(fraction of pairs where the predicted depth orders the closer/farther points correctly).
Evaluated with `eval_da2k.py`. (Numbers to be filled in.)

| Model | Encoder | Quantization | Overall acc | Notes |
|---|---|---|---|---|
| Student (full precision) | ViT-S | none (fp) | **0.6489** | `--quantize` off |
| Student (1.58-bit, fake-quant) | ViT-S | ternary W / int8 A | **0.6407** | `--quantize` |
| Student (1.58-bit, fold-inference) | ViT-S | ternary int path | 0.6407 | `--quantize --fold-inference` (= fake-quant) |
| Student (1.58-bit, packed) | ViT-S | ternary, bit-packed | 0.6407 | `--packed` (= fake-quant) |

Per-scene breakdown (8 scenes: indoor, outdoor, non_real, transparent_reflective,
adverse_style, aerial, underwater, object) — to be filled in:

| Scene | FP student | 1.58-bit student | Δ (1.58 − fp) |
|---|---|---|---|
| indoor | 0.6333 | 0.6119 | −0.0214 |
| outdoor | 0.6599 | 0.6453 | −0.0146 |
| non_real | 0.7129 | 0.7030 | −0.0099 |
| transparent_reflective | 0.6028 | 0.5981 | −0.0047 |
| adverse_style | 0.6128 | 0.6067 | −0.0061 |
| aerial | 0.6082 | 0.6289 | +0.0207 |
| underwater | 0.6667 | 0.6752 | +0.0085 |
| object | 0.7230 | 0.7095 | −0.0135 |
| **OVERALL** | **0.6489** | **0.6407** | **−0.0082** |

**Takeaway:** ternary quantization costs only **−0.0082 overall accuracy** (0.6489 → 0.6407,
−1.3% relative) — the 1.58-bit student nearly matches the full-precision student. The gap is small
and uneven across scenes (5 down, 2 up, within noise on several), so the encoder tolerates ternary
weights well. Both students sit far below the published DAV2-S DA-2K numbers because this is an
early distillation run on unlabeled data, not a fully trained model — so the **relative** fp↔1.58
gap is the meaningful quantity here, and it is small.

**Note:** the three 1.58-bit rows (fake-quant / fold-inference / packed) are numerically
equivalent and report the **same** accuracy — running more than one is a correctness check of the
fold + pack/unpack round-trip, not an accuracy comparison.

**Weight memory:** packing the encoder's ternary weights to 2-bit (4 per byte) yields ~8× smaller
encoder weights vs fp16 (16× vs fp32). The DPT head and norms stay full precision, so whole-model
size reduction is smaller than 8×. This is the one **realized** efficiency win.

---

## 2. Latency Benchmarks

All GPU numbers on a single **NVIDIA A100-SXM4-40GB**, `torch 2.12.1+cu130`, at the ViT-S encoder
linear-layer shapes, `M = 1370` tokens (= `(518/14)² + 1` at 518×518 input).

### 2a. GPU: Triton W8A8 (int8×int8 tensor-core GEMM) vs cuBLAS bf16

`triton_w8a8.py`. The int8 path is numerically correct (max rel err ~2e-4) but **slower**:

| Layer | Shape (N×K) | bf16 (µs) | W8A8 kernel-only (µs) | vs bf16 | W8A8 end-to-end (µs) | vs bf16 |
|---|---|---|---|---|---|---|
| attn.qkv | 1152×384 | 12.84 | 59.08 | 0.22× | 194.31 | 0.07× |
| attn.proj | 384×384 | 8.34 | 56.62 | 0.15× | 216.87 | 0.04× |
| mlp.fc1 | 1536×384 | 16.31 | 75.03 | 0.22× | 219.05 | 0.07× |
| mlp.fc2 | 384×1536 | 15.98 | 76.27 | 0.21× | 220.34 | 0.07× |

- **kernel-only** = the GEMM alone. ~5× slower than cuBLAS bf16 at every shape.
- **end-to-end** = incl. per-token int8 activation quantization. ~15–25× slower — the unfused
  activation quant alone costs ~10× the matmul.

### 2b. GPU: BitBLAS (maintained W2A8 / low-bit GEMM) — blocked

`bench_bitblas.py`. `bitblas 0.1.0.post1` (latest on PyPI) **cannot tune even a plain fp16×fp16
matmul** on this box: every build fails with `AssertionError: No tuning config found for this
operator`. Root cause: BitBLAS 0.1.x's bundled TVM is incompatible with the CUDA 13 / torch 2.12
stack (BitBLAS 0.1.x targets CUDA 12.x). No usable low-bit kernel was obtained from BitBLAS.

The hand-written kernel in `~/Documents/BitNet/gpu/bitnet_kernels` is also not applicable: its
`ladder_int8xint2_kernel` is hardcoded to `M == 1` (LLM single-token decode GEMV) and a fixed set
of LLM `(N,K)` shapes — neither matches the ViT-S (`M=1370`, dims 384/1152/1536).

### 2c. CPU: ternary add/sub and LUT (TL1-style) vs BLAS

`bench_cpu_ternary.py` (single-thread; numba kernels: naive add/sub, 2-bit-packed, and TL1
lookup-table; baseline = numpy fp32 BLAS). All kernels numerically correct (max err ≤ 3e-4).

| Shape | M | BLAS (µs) | add/sub | LUT (TL1) | LUT vs BLAS | LUT vs add/sub |
|---|---|---|---|---|---|---|
| vits.qkv (1152×384) | 1 | 43.6 | 198.1 | 93.3 | 0.47× | 22.9× |
| vits.qkv (1152×384) | 1370 | 12 666 | 286 867 | — | — | — |
| vits.fc1 (1536×384) | 1 | 51.2 | 267.3 | 108.8 | 0.47× | 26.0× |
| vits.fc1 (1536×384) | 1370 | 16 959 | 375 447 | — | — | — |
| **llm.ffn (11008×4096)** | **1** | **4 727** | 19 986 | **6 333** | **0.75×** | **34.0×** |
| llm.ffn (11008×4096) | 1370 | 1 274 189 | 27 534 680 | — | — | — |

(2-bit-packed kernel omitted from the table — correct but ~50× slower than BLAS everywhere; the
scalar bit-unpacking dominates. The LUT supersedes it.)

Two things are confirmed and one limitation is exposed:

- **The LUT method is real:** 23–34× faster than the naive add/sub loop, exactly as the
  table-reuse argument predicts.
- **Regime direction is correct:** LUT is closest to BLAS in the **LLM-scale + M=1** corner
  (0.75×, memory-bound, big weights) and falls away for cache-resident ViT-S (0.47×) and for
  large M (the M=1370 add/sub rows are 20–25× slower than BLAS — compute-bound, BLAS wins).
- **Limitation — our LUT never beats BLAS, even where it should.** numpy's single-thread BLAS
  uses SIMD FMA; our numba LUT does **scalar** gathers. The real `bitnet.cpp` TL1/TL2 kernels do
  the table lookups with **AVX-512 byte-shuffle (`pshufb`)**, 16–64 lookups per instruction —
  that vectorization is what pushes the LLM+M=1 case above 1×. A pure-Python/numba sketch cannot
  reach it. So this experiment validates the *algorithm and the regime*, not an absolute CPU win.

---

## 3. Why we don't see a speedup from the 1.58-bit model

**The 1.58-bit advantage is energy and memory, not compute latency on existing hardware — and at
ViT-S scale neither translates into a wall-clock GPU speedup.** Three layered reasons:

### (i) The compute speedup never came from "fewer multiplies"
Ternary weights make the matmul integer-addition-only (no multiplies), which the BitNet paper
correctly notes **saves orders of magnitude of arithmetic *energy*** (an adder is far cheaper than
a multiplier in silicon). But on a GPU/CPU a multiply-accumulate is **not slower** than an add —
FMA units and tensor cores execute MACs at full throughput, and the multiplier hardware already
exists. So eliminating multiplies cuts joules, not seconds, on this hardware. The paper's "faster
computation" claim is explicitly conditional on *power-limited or custom (ASIC/FPGA)* hardware.

### (ii) Real low-bit kernels run on int8 tensor cores — and these matmuls are too small to benefit
Every practical low-bit kernel (BitNet's included) decodes ternary → int8 and multiplies on int8
tensor cores. int8's only edge is a ~2× higher *compute ceiling* than bf16. But that helps only if
you are compute-bound. At ViT-S shapes (`K=384`, `N≤1536`), cuBLAS bf16 already runs at only ~30%
of its peak — these GEMMs are **memory-/overhead-bound**, not compute-bound. A higher ceiling buys
nothing, and a generic int8 kernel loses to cuBLAS's hand-tuned bf16 (measured: 0.15–0.22×). The
fp16 output dominates memory traffic and is unchanged by int8, and the per-token activation quant
adds large unfused overhead.

### (iii) The memory-bandwidth win requires LLM scale + low batch — the opposite of our workload
The reliable 1.58-bit speed win is bandwidth: ternary weights are 8–16× smaller, so when you are
**memory-bandwidth-bound** you stream far fewer weight bytes. That regime requires (a) weights too
large to fit in cache, and (b) low arithmetic intensity, i.e. small M (little weight reuse — each
weight feeds ~M MACs, so intensity ∝ M). It is the **single-token LLM decode** case. Our student
has neither property: ViT-S encoder weights are ~2 MB (cache-resident, nothing to stream) and
each image is `M=1370` tokens (heavy reuse → compute-bound). The CPU experiments (§2c) show the
regime directly: the LUT kernel is closest to BLAS only in the LLM-scale + M=1 corner (0.75×) and
worse for cache-resident ViT-S (0.47×) and for large M; and even there it cannot beat BLAS without
SIMD-shuffle vectorization (`bitnet.cpp`'s AVX-512 path). At ViT-S scale there is no bandwidth
bottleneck for ternary weights to relieve.

### Conclusion
For the 1.58-bit DAV2-Small student, the **defensible wins are accuracy parity (within the
quantization gap) and ~8× smaller encoder weights**. **Inference latency is not improvable** via
1.58-bit at this scale: low-bit GEMM speedups are an LLM-scale, low-batch, compute-bound phenomenon,
and a ViT-S at `M=1370` is small-matrix and memory-bound, where cuBLAS bf16 (+ xFormers + AMP) is
already near-optimal. The 1.58-bit benefit here is best framed as **energy/storage**, and as a path
to speed **only on power-limited or custom (ASIC/FPGA) hardware**, not on a stock A100.

---

## Appendix: artifacts

| File | Purpose |
|---|---|
| `train_distill.py` | distillation training; `--quantize` swaps encoder linears to BitLinear; saves `student_final.pth` + packed `student_packed.pth` |
| `bitnet.py` | ternary `BitLinear` (train / fold-inference paths), 2-bit storage packing, self-test (`python bitnet.py`) |
| `eval_da2k.py` | DA-2K accuracy; `--quantize`, `--fold-inference`, `--packed` |
| `triton_w8a8.py` | Triton int8 GEMM prototype + bf16 latency benchmark (§2a) |
| `bench_bitblas.py` | BitBLAS low-bit GEMM probe (§2b); `--smoke`, `--debug` |
| `bench_cpu_ternary.py` | CPU ternary add/sub + LUT vs BLAS, with `--sweep` over M (§2c) |
