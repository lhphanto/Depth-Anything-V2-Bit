"""
Precompute and cache the teacher (DAV2-Large) depth targets for distillation.

The teacher is frozen and the preprocessing is deterministic, so its output for each
image is fixed. Running it once and caching to disk removes the expensive ViT-L forward
from every training step / epoch -- the main GPU cost in `train_distill.py`.

Each target is the teacher's `forward()` output at the SAME square resolution the student
trains at, saved as an fp16 `.npy` (~0.5 MB per 518x518 map). Existing files are skipped,
so the job is resumable. Train later with the matching `--teacher-cache` and `--input-size`.

Example
-------
    python precompute_teacher.py \
        --data-dir /home/ubuntu/disk1/datasets/open_images/train \
        --teacher-ckpt checkpoints/depth_anything_v2_vitl.pth \
        --teacher-cache /home/ubuntu/disk1/cache/openimages_vitl_518 \
        --input-size 518 --bs 16 --num-workers 16 --amp
"""

import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from train_distill import (
    MODEL_CONFIGS, UnlabeledImageDataset, build_model, teacher_cache_path,
)


class _IndexedDataset(Dataset):
    """Wraps UnlabeledImageDataset to also yield each sample's index (for naming)."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        return i, self.base[i]


def parse_args():
    p = argparse.ArgumentParser(description='Precompute DAV2 teacher targets')
    p.add_argument('--data-dir', type=str, required=True)
    p.add_argument('--teacher-ckpt', type=str, required=True)
    p.add_argument('--teacher-cache', type=str, required=True, help='output dir for .npy targets')
    p.add_argument('--teacher-encoder', type=str, default='vitl', choices=list(MODEL_CONFIGS))
    p.add_argument('--input-size', type=int, default=518)
    p.add_argument('--bs', type=int, default=16)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--prefetch-factor', type=int, default=4)
    p.add_argument('--filelist-cache', type=str, default=None)
    p.add_argument('--rebuild-filelist', action='store_true')
    p.add_argument('--overwrite', action='store_true', help='recompute even if a cache file exists')
    p.add_argument('--amp', action='store_true')
    p.add_argument('--amp-dtype', type=str, default='bf16', choices=['bf16', 'fp16'])
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(args.teacher_cache, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else \
             'mps' if torch.backends.mps.is_available() else 'cpu'
    use_amp = args.amp and device == 'cuda'
    amp_dtype = torch.bfloat16 if args.amp_dtype == 'bf16' else torch.float16
    print(f'Device: {device} | AMP: {use_amp} ({args.amp_dtype if use_amp else "off"})')

    teacher = build_model(args.teacher_encoder, args.teacher_ckpt, device).eval()

    base = UnlabeledImageDataset(
        args.data_dir, args.input_size,
        cache_path=args.filelist_cache, rebuild_cache=args.rebuild_filelist,
    )
    loader = DataLoader(
        _IndexedDataset(base), batch_size=args.bs, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    n_total = len(base)
    n_done = n_saved = 0
    t_log = time.perf_counter()
    for indices, imgs in loader:
        # Skip the batch entirely if every target already exists (fast resume).
        paths = [teacher_cache_path(args.teacher_cache, base.files[i], args.input_size)
                 for i in indices.tolist()]
        if not args.overwrite and all(os.path.exists(p) for p in paths):
            n_done += len(paths)
            continue

        x = imgs.to(device, non_blocking=True)
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
            depth = teacher(x)                     # [B, H, W]
        depth = depth.float().cpu().numpy().astype(np.float16)

        for path, d in zip(paths, depth):
            if args.overwrite or not os.path.exists(path):
                np.save(path, d)
                n_saved += 1
        n_done += len(paths)

        if n_done % (args.bs * 20) < args.bs:
            now = time.perf_counter()
            ips = (args.bs * 20) / (now - t_log)
            t_log = now
            print(f'{n_done}/{n_total} images | saved {n_saved} | {ips:.1f} img/s')

    print(f'Done. {n_saved} targets written to {args.teacher_cache} '
          f'({n_done}/{n_total} images processed).')


if __name__ == '__main__':
    main()
