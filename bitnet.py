"""
1.58-bit (ternary) BitLinear with separate training and inference paths, plus helpers
to swap nn.Linear layers and to fold a trained module to its inference form.

Adapted from the BitNet b1.58 training tips / FAQ and the HF write-up:
https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf
https://huggingface.co/blog/1_58_llm_extreme_quantization

Training (default): weights/activations are *fake-quantized* -- the forward sees
quantized values but gradients flow to the fp master weights via the straight-through
estimator (`x + (quant(x) - x).detach()`).

Inference (after `quantize_for_inference()`): the fp weight is folded once into ternary
integers `{-1,0,1}` plus a scalar `w_scale`; activations are quantized to int8 per token,
the matmul runs on the integer grids, and the result is rescaled by `1/w_scale/x_scale`.
The two paths are numerically equivalent; the inference one is what a real packed
1.58-bit kernel would compute (here we use F.linear on the integer-valued tensors as a
stand-in for that kernel).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def activation_quant(x):
    """Per-token absmax quantization of activations to 8-bit, then dequantized.

    `x`: [..., hidden]. The scale is computed over the last dim so each token gets its
    own scale. Returns a tensor of the same shape/dtype with values on the int8 grid.
    """
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y


def activation_quant_inference(x):
    """Inference activation quant: return the int8-grid tensor AND its per-token scale.

    Unlike `activation_quant`, this keeps the value on the integer grid (no divide-back)
    and returns `x_scale` separately, so the caller can do the rescale once after the
    integer matmul: `y = kernel(x_int, w_int) / w_scale / x_scale`.
    """
    x = F.layer_norm(x, (x.shape[-1],))
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127)
    return y, scale


def weight_quant(w):
    """Absmean ternarization of weights to {-1, 0, 1}, then dequantized to fp.

    A single scalar scale (1 / mean(|w|)) is applied to the whole weight matrix, the
    BitNet b1.58 scheme. Returns a tensor of the same shape with ternary * scale values.
    """
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    u = (w * scale).round().clamp_(-1, 1) / scale
    return u


# --------------------------------------------------------------------------------------
# Ternary weight packing (storage only)
# --------------------------------------------------------------------------------------
# Pack 4 ternary values {-1,0,1} into one byte (2 bits each) for ~8x smaller weights vs
# fp16 (16x vs fp32). This is a *portable* bit layout -- no tensor-core permutation -- so we
# can unpack it back to ternary fp in PyTorch and run the existing _forward_inference. It
# captures the 1.58-bit memory win without needing a specialized W2A8 GEMM kernel.

def pack_ternary(w_int):
    """Pack a ternary weight matrix to uint8, 4 weights per byte.

    `w_int`: [N, K] with values in {-1, 0, 1} (any int/float dtype). Returns a uint8
    tensor of shape [N, ceil(K/4)]. K is padded to a multiple of 4 with zeros (those
    columns are discarded on unpack, which is given the true K).
    """
    N, K = w_int.shape
    u = (w_int.round().to(torch.int16) + 1).clamp_(0, 2).to(torch.uint8)   # {-1,0,1} -> {0,1,2}
    pad = (-K) % 4
    if pad:
        u = F.pad(u, (0, pad))                                            # pad K to multiple of 4
    u = u.view(N, (K + pad) // 4, 4)
    packed = u[..., 0] | (u[..., 1] << 2) | (u[..., 2] << 4) | (u[..., 3] << 6)
    return packed.contiguous()                                            # [N, ceil(K/4)] uint8


def unpack_ternary(packed, K):
    """Inverse of `pack_ternary`. Returns a float32 [N, K] tensor in {-1, 0, 1}."""
    p = packed.to(torch.uint8)
    fields = torch.stack([p & 3, (p >> 2) & 3, (p >> 4) & 3, (p >> 6) & 3], dim=-1)
    u = fields.view(packed.shape[0], -1)[:, :K]                           # drop K-padding
    return u.to(torch.float32) - 1.0                                      # {0,1,2} -> {-1,0,1}


class BitLinear(nn.Linear):
    """nn.Linear with 1.58-bit weights and 8-bit activations.

    Defaults to the training (fake-quant) forward. Call `quantize_for_inference()` to
    fold the weights to ternary integers and switch to the integer inference forward.
    A parameter-free LayerNorm (SubLN) is applied to the input before activation
    quantization, as recommended by the BitNet b1.58 recipe.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inference_mode = False
        # Scalar dequant scale for the folded ternary weights; set by
        # quantize_for_inference(). Registered so it rides along in state_dict / .to().
        self.register_buffer('w_scale', None)
        # Packed ternary weight (uint8, 4 weights/byte); set by pack_for_storage().
        self.register_buffer('packed_weight', None)

    def forward(self, x):
        if self.inference_mode:
            return self._forward_inference(x)
        return self._forward_train(x)

    def _forward_train(self, x):
        w = self.weight
        # SubLN: normalize over the input-feature dim with no learnable affine.
        x_norm = F.layer_norm(x, (self.in_features,))

        # Straight-Through-Estimator: forward uses the quantized value, backward sees
        # an identity (the (quant - x) term is detached so it has zero gradient).
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (weight_quant(w) - w).detach()

        return F.linear(x_quant, w_quant, self.bias)

    def _forward_inference(self, x):
        # self.weight already holds the ternary {-1,0,1} integers; w_scale = 1/mean(|w|).
        x_quant, x_scale = activation_quant_inference(x)
        y = F.linear(x_quant, self.weight) / self.w_scale / x_scale
        if self.bias is not None:
            y = y + self.bias
        return y

    @torch.no_grad()
    def quantize_for_inference(self):
        """Fold the fp master weight into ternary integers + a scalar scale (in place).

        After this, `forward` uses the integer inference path. Equivalent in value to the
        training fake-quant, but in the form a packed 1.58-bit kernel would consume.
        """
        scale = 1.0 / self.weight.abs().mean().clamp_(min=1e-5)   # = 1 / mean(|w|)
        w_int = (self.weight * scale).round().clamp_(-1, 1)       # ternary {-1,0,1}
        self.weight.data = w_int.to(self.weight.dtype)
        self.w_scale = scale.to(self.weight.dtype)
        self.inference_mode = True
        return self

    @torch.no_grad()
    def pack_for_storage(self):
        """Ternarize (if needed) and bit-pack the weight to uint8, freeing the fp weight.

        Leaves the module in inference_mode with `packed_weight` + `w_scale` set and
        `weight = None`. Call unpack_from_storage() to make it runnable again.
        """
        if self.inference_mode and self.w_scale is not None:
            w_int = self.weight                                  # already ternary {-1,0,1}
        else:
            scale = 1.0 / self.weight.abs().mean().clamp_(min=1e-5)
            w_int = (self.weight * scale).round().clamp_(-1, 1)
            self.w_scale = scale.to(self.weight.dtype)
        self.packed_weight = pack_ternary(w_int)
        self.weight = None                                       # drop the big fp weight
        self.inference_mode = True
        return self

    @torch.no_grad()
    def unpack_from_storage(self):
        """Inverse of pack_for_storage(): rebuild the ternary weight so forward() runs."""
        w = unpack_ternary(self.packed_weight, self.in_features)
        dtype = self.w_scale.dtype if self.w_scale is not None else torch.float32
        self.weight = nn.Parameter(w.to(self.packed_weight.device, dtype),
                                   requires_grad=False)
        self.inference_mode = True
        return self


def convert_linear_to_bitlinear(module):
    """Recursively swap every nn.Linear in `module` for a BitLinear, reusing weights.

    Pass `student.pretrained` to quantize only the DINOv2 encoder linear layers.
    Returns the number of layers replaced.
    """
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and not isinstance(child, BitLinear):
            new = BitLinear(child.in_features, child.out_features, bias=child.bias is not None)
            new.weight = child.weight
            if child.bias is not None:
                new.bias = child.bias
            setattr(module, name, new)
            n += 1
        else:
            n += convert_linear_to_bitlinear(child)
    return n


def quantize_for_inference(module):
    """Fold every BitLinear under `module` to its ternary-integer inference form.

    Call once on a trained model before evaluation/deployment. Returns the count folded.
    """
    n = 0
    for child in module.modules():
        if isinstance(child, BitLinear) and not child.inference_mode:
            child.quantize_for_inference()
            n += 1
    return n


def pack_model_for_storage(module):
    """Bit-pack every BitLinear under `module` for a small (1.58-bit) checkpoint.

    After this, `module.state_dict()` carries `packed_weight` (uint8) + `w_scale` instead of
    the fp `weight`. Returns the count packed. See unpack_model_from_storage / prepare_packed_load.
    """
    n = 0
    for child in module.modules():
        if isinstance(child, BitLinear) and child.packed_weight is None:
            child.pack_for_storage()
            n += 1
    return n


def unpack_model_from_storage(module):
    """Rebuild runnable ternary weights for every packed BitLinear under `module`."""
    n = 0
    for child in module.modules():
        if isinstance(child, BitLinear) and child.packed_weight is not None:
            child.unpack_from_storage()
            n += 1
    return n


def prepare_packed_load(module):
    """Pre-allocate the packed buffers so a packed checkpoint can be load_state_dict'd.

    load_state_dict copies into existing tensors, so before loading a packed checkpoint the
    BitLinears must already have `packed_weight` / `w_scale` buffers of the right shape and
    `weight = None`. Run this after convert_linear_to_bitlinear(), then load, then
    unpack_model_from_storage().
    """
    for child in module.modules():
        if isinstance(child, BitLinear):
            k4 = (child.in_features + 3) // 4
            child.packed_weight = torch.zeros(child.out_features, k4, dtype=torch.uint8)
            child.w_scale = torch.zeros((), dtype=torch.float32)
            child.weight = None
            child.inference_mode = True


def _self_test():
    """CPU self-test: pack/unpack round-trips, and fold vs packed forward agree. `python bitnet.py`."""
    torch.manual_seed(0)

    # 1. pack/unpack is lossless for ternary values, including a non-multiple-of-4 K.
    for K in (384, 1536, 13, 1):
        w = torch.randint(-1, 2, (7, K)).float()
        assert torch.equal(unpack_ternary(pack_ternary(w), K), w), f'round-trip failed K={K}'
    print('pack/unpack round-trip: OK')

    # 2. train fake-quant vs fold-inference vs pack->unpack inference all agree.
    bl = BitLinear(384, 256)
    x = torch.randn(10, 384)
    y_train = bl(x)                                              # fake-quant (training path)

    import copy
    bl_fold = copy.deepcopy(bl); bl_fold.quantize_for_inference()
    y_fold = bl_fold(x)

    bl_pack = copy.deepcopy(bl); bl_pack.pack_for_storage(); bl_pack.unpack_from_storage()
    y_pack = bl_pack(x)

    e_tf = (y_train - y_fold).abs().max().item()
    e_fp = (y_fold - y_pack).abs().max().item()
    print(f'train vs fold:   max_err {e_tf:.3e} (fp rounding)')
    print(f'fold vs packed:  max_err {e_fp:.3e} (should be ~0)')
    assert e_tf < 1e-3 and e_fp < 1e-6, 'forward paths disagree'

    # 3. size report.
    fp16_bytes = bl.weight.numel() * 2
    packed_bytes = bl_pack.packed_weight.numel() * 1
    print(f'weight memory: fp16 {fp16_bytes} B -> packed {packed_bytes} B '
          f'({fp16_bytes / packed_bytes:.1f}x smaller)')
    print('ALL OK')


if __name__ == '__main__':
    _self_test()
