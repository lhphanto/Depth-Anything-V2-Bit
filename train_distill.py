"""
Distillation training for Depth Anything V2.

Teacher : full-precision DAV2-Large (frozen) -> pseudo depth labels on unlabeled images.
Student : DAV2-Small, whose DINOv2 linear layers can be swapped for 1.58-bit BitLinear.
Data    : a directory of unlabeled images (e.g. Google OpenImages). No ground truth.

Loss    : scale-shift-invariant (affine-invariant) L1 + multi-scale gradient matching,
          the MiDaS / Depth-Anything objective appropriate for *relative* depth. Because
          the relative model is only defined up to an unknown scale and shift, we align
          the student to the teacher before comparing.

Example
-------
    python train_distill.py \
        --data-dir /path/to/openimages \
        --teacher-ckpt checkpoints/depth_anything_v2_vitl.pth \
        --student-ckpt checkpoints/depth_anything_v2_vits.pth \
        --quantize \
        --bs 4 --lr 5e-6 --epochs 1 --save-path exp/distill_vits_1p58bit
"""

import argparse
import glob
import os

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose

from depth_anything_v2.dpt import DepthAnythingV2
from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet


MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
}

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')


# --------------------------------------------------------------------------------------
# 1.58-bit quantization placeholder
# --------------------------------------------------------------------------------------
class BitLinear(nn.Linear):
    """Drop-in replacement for nn.Linear, ready for 1.58-bit (ternary) quantization.

    For now this is a pass-through identical to nn.Linear, so the student trains in full
    precision as a placeholder. Fill in the two hooks below with your BitNet b1.58
    implementation later; the rest of the pipeline does not need to change.
    """

    def _quantize_weight(self, w):
        # TODO(1.58-bit): ternarize weights to {-1, 0, 1} with absmean scaling, e.g.
        #     scale = w.abs().mean().clamp_min(1e-5)
        #     w_q   = (w / scale).round().clamp(-1, 1) * scale
        #     return w + (w_q - w).detach()        # straight-through estimator
        return w

    def _quantize_activation(self, x):
        # TODO(1.58-bit): per-token absmax int8 quantization of the activations.
        return x

    def forward(self, x):
        x = self._quantize_activation(x)
        w = self._quantize_weight(self.weight)
        return F.linear(x, w, self.bias)


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


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
class UnlabeledImageDataset(Dataset):
    """Loads images from a directory tree and applies DAV2 preprocessing.

    Uses a fixed square resize (default 518 = 37 x 14) so samples can be batched.
    """

    def __init__(self, data_dir, input_size=518):
        self.files = sorted(
            f for f in glob.glob(os.path.join(data_dir, '**', '*'), recursive=True)
            if f.lower().endswith(IMG_EXTS)
        )
        if not self.files:
            raise FileNotFoundError(f'No images found under {data_dir}')

        self.transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=False,        # force square so we can stack into a batch
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = cv2.imread(self.files[i])
        if img is None:                          # skip unreadable file
            return self.__getitem__((i + 1) % len(self.files))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        img = self.transform({'image': img})['image']
        return torch.from_numpy(img)


# --------------------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------------------
class AffineInvariantDistillLoss(nn.Module):
    """Scale-shift-invariant L1 + multi-scale gradient matching (MiDaS / DAV2).

    Both predictions are first normalized per-sample to zero median and unit mean
    absolute deviation, which removes the unknown affine ambiguity of relative depth,
    then compared.
    """

    def __init__(self, grad_weight=0.5, grad_scales=4):
        super().__init__()
        self.grad_weight = grad_weight
        self.grad_scales = grad_scales

    @staticmethod
    def _normalize(d, eps=1e-6):
        b, h, w = d.shape
        flat = d.reshape(b, -1)
        t = flat.median(dim=1, keepdim=True).values
        s = (flat - t).abs().mean(dim=1, keepdim=True) + eps
        return ((flat - t) / s).reshape(b, h, w)

    def _grad_loss(self, diff):
        loss = diff.new_zeros(())
        for scale in range(self.grad_scales):
            step = 2 ** scale
            d = diff[:, ::step, ::step]
            gx = (d[:, :, 1:] - d[:, :, :-1]).abs().mean()
            gy = (d[:, 1:, :] - d[:, :-1, :]).abs().mean()
            loss = loss + gx + gy
        return loss

    def forward(self, pred, target):
        diff = self._normalize(pred) - self._normalize(target)
        ssi = diff.abs().mean()
        if self.grad_weight > 0:
            return ssi + self.grad_weight * self._grad_loss(diff)
        return ssi


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------
def build_model(encoder, ckpt, device):
    model = DepthAnythingV2(**MODEL_CONFIGS[encoder])
    if ckpt:
        model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    return model.to(device)


def parse_args():
    p = argparse.ArgumentParser(description='Depth Anything V2 distillation')
    p.add_argument('--data-dir', type=str, required=True, help='directory of unlabeled images')
    p.add_argument('--teacher-encoder', type=str, default='vitl', choices=list(MODEL_CONFIGS))
    p.add_argument('--student-encoder', type=str, default='vits', choices=list(MODEL_CONFIGS))
    p.add_argument('--teacher-ckpt', type=str, required=True)
    p.add_argument('--student-ckpt', type=str, default=None, help='optional warm-start for the student')
    p.add_argument('--quantize', action='store_true', help='swap student DINOv2 linears for BitLinear')
    p.add_argument('--input-size', type=int, default=518)
    p.add_argument('--bs', type=int, default=4)
    p.add_argument('--lr', type=float, default=5e-6)
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--max-steps', type=int, default=0, help='stop early after N optimizer steps (0 = no limit)')
    p.add_argument('--grad-weight', type=float, default=0.5)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--save-path', type=str, default='exp/distill')
    p.add_argument('--amp', action='store_true', help='mixed precision (CUDA only)')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else \
             'mps' if torch.backends.mps.is_available() else 'cpu'
    use_amp = args.amp and device == 'cuda'
    print(f'Device: {device} | AMP: {use_amp}')

    # Teacher: frozen, eval.
    teacher = build_model(args.teacher_encoder, args.teacher_ckpt, device).eval()
    for prm in teacher.parameters():
        prm.requires_grad_(False)

    # Student: optionally quantized, trainable.
    student = build_model(args.student_encoder, args.student_ckpt, device)
    if args.quantize:
        n = convert_linear_to_bitlinear(student.pretrained)
        print(f'Quantized {n} linear layers in the student DINOv2 encoder -> BitLinear')
    student.train()

    loader = DataLoader(
        UnlabeledImageDataset(args.data_dir, args.input_size),
        batch_size=args.bs, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )
    print(f'Training images: {len(loader.dataset)} | steps/epoch: {len(loader)}')

    criterion = AffineInvariantDistillLoss(grad_weight=args.grad_weight)
    optimizer = torch.optim.AdamW(
        (p for p in student.parameters() if p.requires_grad),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    total_steps = args.max_steps or args.epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    step = 0
    for epoch in range(args.epochs):
        for batch in loader:
            x = batch.to(device, non_blocking=True)

            with torch.no_grad(), torch.autocast(device_type='cuda', enabled=use_amp):
                target = teacher(x)

            with torch.autocast(device_type='cuda', enabled=use_amp):
                pred = student(x)
                loss = criterion(pred, target)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1

            if step % args.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                print(f'epoch {epoch} step {step}/{total_steps} '
                      f'loss {loss.item():.4f} lr {lr:.2e}')

            if args.max_steps and step >= args.max_steps:
                break

        ckpt_path = os.path.join(args.save_path, f'student_epoch{epoch}.pth')
        torch.save(student.state_dict(), ckpt_path)
        print(f'Saved {ckpt_path}')

        if args.max_steps and step >= args.max_steps:
            break

    final = os.path.join(args.save_path, 'student_final.pth')
    torch.save(student.state_dict(), final)
    print(f'Done. Final student: {final}')


if __name__ == '__main__':
    main()
