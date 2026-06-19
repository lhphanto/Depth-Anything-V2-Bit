"""
Evaluate a (distilled) Depth Anything V2 model on the DA-2K benchmark.

DA-2K provides 2068 point pairs over 1033 images; for each pair `point1` is the
ground-truth closer point. The relative model outputs disparity (higher = closer),
so a pair is scored correct when  depth[point1] > depth[point2].  The reported number
is pairwise accuracy, overall and per scene (the Table 3 / Table 14 metric in the paper).

The data ships as a single `DA-2K.zip`; this script can read images straight from the
zip (no extraction needed) or from an already-extracted `DA-2K/` directory.

Example
-------
    python eval_da2k.py \
        --checkpoint exp/distill_vits_1p58bit/student_final.pth \
        --encoder vits \
        --data-root /Users/huiliu/.cache/huggingface/hub/datasets--depth-anything--DA-2K/snapshots/528f83c26f02a6a7a11c5a036dfc9ee19907ebd9/DA-2K.zip
"""

import argparse
import json
import os
import zipfile
from collections import defaultdict

import cv2
import numpy as np
import torch

from bitnet import convert_linear_to_bitlinear, quantize_for_inference
from depth_anything_v2.dpt import DepthAnythingV2
from train_distill import MODEL_CONFIGS


def resolve_data_root(data_root):
    """Turn `data_root` into a usable local path.

    A local path (a DA-2K.zip or an extracted dir) is returned as-is. Otherwise it is
    treated as a Hugging Face dataset repo id (e.g. 'depth-anything/DA-2K') and the
    DA-2K.zip is resolved from / downloaded to the HF cache. This follows the
    refs/main -> snapshots/<hash> symlink, so no commit hash is hardcoded.
    """
    if os.path.exists(data_root):
        return data_root
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise SystemExit(
            f'--data-root {data_root!r} is not a local path and huggingface_hub is not '
            f'installed to resolve it as a repo id ({e}). pip install huggingface_hub.')
    print(f'{data_root!r} is not a local path; resolving as a HF dataset repo id...')
    return hf_hub_download(data_root, 'DA-2K.zip', repo_type='dataset')


class DA2KSource:
    """Reads annotations.json and images from either a DA-2K.zip or an extracted dir."""

    def __init__(self, data_root):
        data_root = resolve_data_root(data_root)
        self.zip = None
        if data_root.endswith('.zip'):
            self.zip = zipfile.ZipFile(data_root, 'r')
            self.prefix = self._find_prefix(self.zip.namelist())
            ann = self.zip.read(self.prefix + 'annotations.json')
            self.annotations = json.loads(ann)
        else:
            # Accept either <root> or <root>/DA-2K.
            root = data_root
            if not os.path.exists(os.path.join(root, 'annotations.json')):
                root = os.path.join(root, 'DA-2K')
            self.dir = root
            with open(os.path.join(root, 'annotations.json')) as f:
                self.annotations = json.load(f)

    @staticmethod
    def _find_prefix(names):
        for n in names:
            if n.endswith('annotations.json') and not os.path.basename(n).startswith('._'):
                return n[: -len('annotations.json')]
        raise FileNotFoundError('annotations.json not found in zip')

    def read_image(self, rel_path):
        """Return a BGR uint8 image for an annotation key like 'images/indoor/x.jpg'."""
        if self.zip is not None:
            buf = self.zip.read(self.prefix + rel_path)
            return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        return cv2.imread(os.path.join(self.dir, rel_path))


def parse_args():
    p = argparse.ArgumentParser(description='DA-2K evaluation')
    p.add_argument('--checkpoint', type=str, required=True, help='student .pth to evaluate')
    p.add_argument('--encoder', type=str, default='vits', choices=list(MODEL_CONFIGS))
    p.add_argument('--data-root', type=str, required=True,
                   help='DA-2K.zip, an extracted DA-2K dir, or a HF dataset repo id '
                        '(e.g. depth-anything/DA-2K) to resolve from the HF cache')
    p.add_argument('--input-size', type=int, default=518)
    p.add_argument('--quantize', action='store_true',
                   help='swap encoder linears for BitLinear before loading (match a quantized checkpoint)')
    p.add_argument('--fold-inference', action='store_true',
                   help='fold BitLinear weights to ternary ints and use the integer '
                        'inference path (requires --quantize)')
    p.add_argument('--scene-type', type=str, default='',
                   help='restrict to one scene (indoor, outdoor, ...); empty = all')
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else \
             'mps' if torch.backends.mps.is_available() else 'cpu'

    if args.fold_inference and not args.quantize:
        raise SystemExit('--fold-inference requires --quantize')

    model = DepthAnythingV2(**MODEL_CONFIGS[args.encoder])
    if args.quantize:
        n = convert_linear_to_bitlinear(model.pretrained)
        print(f'Quantized {n} encoder linear layers -> BitLinear')
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))

    # Fold to ternary integers AFTER loading the trained fp weights. The integer
    # inference path is numerically equivalent to the fake-quant training path; this
    # checks the deployment form gives the same DA-2K accuracy.
    if args.fold_inference:
        n = quantize_for_inference(model.pretrained)
        print(f'Folded {n} BitLinear layers to ternary-int inference form')

    model = model.to(device).eval()

    source = DA2KSource(args.data_root)
    items = [(k, v) for k, v in source.annotations.items()
             if not args.scene_type or f'/{args.scene_type}/' in k]
    print(f'Evaluating {len(items)} images on {device} '
          f'(scene: {args.scene_type or "all"})')

    correct = defaultdict(int)
    total = defaultdict(int)

    for i, (img_path, pairs) in enumerate(items):
        image = source.read_image(img_path)
        if image is None:
            print(f'  [warn] could not read {img_path}, skipping')
            continue

        depth = model.infer_image(image, args.input_size)   # HxW, higher = closer
        h, w = depth.shape
        scene = img_path.split('/')[1]

        for pair in pairs:
            closer = pair['closer_point']                 # 'point1' or 'point2'
            farther = 'point2' if closer == 'point1' else 'point1'
            hc, wc = pair[closer]
            hf, wf = pair[farther]
            # Coordinates are in original-image pixels; infer_image returns that size.
            d_closer = depth[min(hc, h - 1), min(wc, w - 1)]
            d_farther = depth[min(hf, h - 1), min(wf, w - 1)]
            # The closer point should have the larger disparity (higher = closer).
            ok = d_closer > d_farther
            correct[scene] += int(ok)
            total[scene] += 1
            correct['all'] += int(ok)
            total['all'] += 1

        if (i + 1) % 100 == 0:
            print(f'  {i + 1}/{len(items)} images | running acc '
                  f'{correct["all"] / total["all"]:.4f}')

    print('\n=== DA-2K accuracy ===')
    for scene in sorted(k for k in total if k != 'all'):
        print(f'  {scene:<24} {correct[scene] / total[scene]:.4f} '
              f'({correct[scene]}/{total[scene]})')
    print(f'  {"OVERALL":<24} {correct["all"] / total["all"]:.4f} '
          f'({correct["all"]}/{total["all"]})')


if __name__ == '__main__':
    main()
