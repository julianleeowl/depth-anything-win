import argparse
import copy
import cv2
import glob
import numpy as np
import os
import re
import torch
import torch.onnx
from pathlib import Path

from depth_anything_v2.dpt import DepthAnythingV2

ENCODERS = ['vits', 'vitb', 'vitl', 'vitg']

MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
}


def infer_encoder(checkpoint_path):
    """Infer encoder type from the checkpoint filename/path.

    Searches for 'vits', 'vitb', 'vitl', 'vitg' in the path string.
    Raises ValueError if none or multiple encoders match.
    """
    path_lower = checkpoint_path.lower()
    matches = [enc for enc in ENCODERS if enc in path_lower]
    if len(matches) == 0:
        raise ValueError(
            f"Cannot infer encoder from '{checkpoint_path}'. "
            f"Expected one of {ENCODERS} in the filename/path."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous encoder in '{checkpoint_path}': matched {matches}. "
            f"Rename the file so only one encoder name appears."
        )
    return matches[0]


def load_checkpoint(model, checkpoint_path):
    """Load checkpoint supporting both .pth and .safetensors formats."""
    if checkpoint_path.endswith('.safetensors'):
        from safetensors.torch import load_file
        state_dict = load_file(checkpoint_path, device='cpu')
    else:  # .pth or other PyTorch formats
        state_dict = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(state_dict)
    return model


def main():
    parser = argparse.ArgumentParser(description='Depth Anything V2')

    parser.add_argument('--checkpoint', nargs='+', required=True,
                        help='Path(s) to model checkpoint(s) (.pth or .safetensors)')
    parser.add_argument('--precision', nargs='+', default=['fp32'],
                        choices=['fp32', 'fp16'],
                        help='Model precision(s): fp32 and/or fp16 (default: fp32)')
    parser.add_argument('--opset-version', nargs='+', type=int, default=[11],
                        help='ONNX opset version(s) (default: 11)')
    parser.add_argument('--input-size', nargs='+', type=int, default=[518],
                        help='Input size(s) (default: 518)')

    args = parser.parse_args()

    device = 'cpu'

    for checkpoint_path in args.checkpoint:
        encoder = infer_encoder(checkpoint_path)
        print(f"Checkpoint: {checkpoint_path}  (encoder: {encoder})")

        depth_anything = DepthAnythingV2(**MODEL_CONFIGS[encoder])
        depth_anything = load_checkpoint(depth_anything, checkpoint_path)

        checkpoint_p = Path(checkpoint_path)
        stem = checkpoint_p.stem

        for precision in args.precision:
            model = copy.deepcopy(depth_anything).to(device)
            if precision == 'fp16':
                model = model.half()
                dtype = torch.float16
            else:
                dtype = torch.float32
            model.eval()

            for input_size in args.input_size:
                dummy_input = torch.ones((1, 3, input_size, input_size),
                                        dtype=dtype).to(device)

                for opset in args.opset_version:
                    output_dir = checkpoint_p.parent / f"{input_size}"
                    output_dir.mkdir(parents=True, exist_ok=True)

                    onnx_filename = f"{stem}-{input_size}-{precision}-op{opset}.onnx"
                    onnx_path = output_dir / onnx_filename

                    torch.onnx.export(
                        model,
                        dummy_input,
                        str(onnx_path),
                        opset_version=opset,
                        input_names=["input"],
                        output_names=["output"],
                        verbose=True,
                        export_params=True,
                        do_constant_folding=True,
                    )

                    print(f"Exported: {onnx_path}")


if __name__ == "__main__":
    main()
