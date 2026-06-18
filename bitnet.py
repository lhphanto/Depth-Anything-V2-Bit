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
