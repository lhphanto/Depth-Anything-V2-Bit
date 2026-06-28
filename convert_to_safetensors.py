"""
Convert DepthAnythingV2 student checkpoints (.pth pickle) to .safetensors for upload.

safetensors is the format HF recommends: it stores tensors only (no executable pickle), loads
faster, and is the community default. This script loads each .pth state_dict and re-saves it as
.safetensors next to the original (or to --out-dir).

Note on the 1.58-bit checkpoint: its weights are ternary {-1,0,1} but stored as fp32 *masters*
(the BitLinear training weights), so the file is the same size as the fp baseline. That is the
form eval_da2k.py --quantize expects. Bit-packing for 8x-smaller storage is a separate step (see
bitnet.py pack_model_for_storage); this script does NOT pack -- it only changes the container.

    pip install safetensors torch
    python convert_to_safetensors.py archive/distill_vits_0617a/student_final.pth \
                                     archive/distill_vits_0617bit/student_final.pth
"""

import argparse
import os

import torch
from safetensors.torch import save_file


def convert(pth_path, out_dir=None):
    sd = torch.load(pth_path, map_location='cpu', weights_only=True)
    if not isinstance(sd, dict):
        raise SystemExit(f'{pth_path}: expected a state_dict, got {type(sd)}')

    # safetensors requires plain, contiguous tensors and no shared storage. Clone to be safe
    # and drop any non-tensor entries (rare in these checkpoints, but be defensive).
    tensors = {}
    skipped = []
    for k, v in sd.items():
        if torch.is_tensor(v):
            tensors[k] = v.contiguous().clone()
        else:
            skipped.append(k)
    if skipped:
        print(f'  [warn] skipped {len(skipped)} non-tensor keys: {skipped}')

    out_dir = out_dir or os.path.dirname(pth_path)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(pth_path))[0] + '.safetensors'
    out_path = os.path.join(out_dir, base)
    save_file(tensors, out_path, metadata={'format': 'pt'})

    src_mb = os.path.getsize(pth_path) / 1e6
    dst_mb = os.path.getsize(out_path) / 1e6
    print(f'  {pth_path} ({src_mb:.0f} MB) -> {out_path} ({dst_mb:.0f} MB, {len(tensors)} tensors)')
    return out_path


def main():
    p = argparse.ArgumentParser(description='Convert .pth checkpoints to .safetensors')
    p.add_argument('checkpoints', nargs='+', help='one or more .pth files')
    p.add_argument('--out-dir', default=None, help='output dir (default: alongside each input)')
    args = p.parse_args()
    for ckpt in args.checkpoints:
        convert(ckpt, args.out_dir)
    print('Done.')


if __name__ == '__main__':
    main()
