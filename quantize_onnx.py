#!/usr/bin/env python3
"""
Quantize Depth Anything V2 ONNX model to INT8 using modelopt ONNX-level quantization.

This bypasses PyTorch export issues by operating directly on the ONNX graph:
  1. Load a clean FP32 ONNX model
  2. Calibrate and insert Q/DQ nodes using modelopt.onnx.quantization
  3. Output a quantized ONNX ready for TensorRT --int8 --fp16

Usage:
    python quantize_onnx.py \
        --onnx depth_anything_v2_vits.onnx \
        --image /home/julian/test.png \
        --output depth_anything_v2_vits_int8.onnx

    # Exclude sensitive layers (from sensitivity analysis):
    python quantize_onnx.py \
        --onnx depth_anything_v2_vits.onnx \
        --image /home/julian/test.png \
        --output depth_anything_v2_vits_int8.onnx \
        --nodes-to-exclude ".*patch_embed.*" ".*blocks\\.0\\..*"

    # Then build TensorRT engine:
    trtexec --onnx=depth_anything_v2_vits_int8.onnx --saveEngine=model.engine --int8 --fp16
"""

import argparse

import cv2
import numpy as np

from modelopt.onnx.quantization import quantize


def load_calibration_image(image_path: str, input_size: int = 518) -> np.ndarray:
    """Load and preprocess a single image, matching Depth Anything V2 preprocessing."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    target_size = (input_size // 14) * 14
    img_resized = cv2.resize(img_rgb, (target_size, target_size))

    img_norm = img_resized / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_norm = (img_norm - mean) / std

    # HWC -> CHW, add batch dim -> (1, 3, H, W)
    return img_norm.transpose(2, 0, 1).astype(np.float32)[np.newaxis]


def main():
    parser = argparse.ArgumentParser(
        description="Quantize Depth Anything V2 ONNX to INT8 with modelopt"
    )
    parser.add_argument("--onnx", type=str, required=True, help="Input FP32 ONNX model")
    parser.add_argument("--image", type=str, required=True, help="Calibration image")
    parser.add_argument("--output", type=str, default=None, help="Output quantized ONNX path")
    parser.add_argument(
        "--calibration-method", type=str, default="entropy",
        choices=["entropy", "max"],
        help="Calibration method (default: entropy)",
    )
    parser.add_argument(
        "--nodes-to-exclude", type=str, nargs="*", default=None,
        help="Regex patterns for nodes to exclude from quantization",
    )
    args = parser.parse_args()

    # Load calibration image
    print(f"Loading calibration image: {args.image}")
    calib_data = load_calibration_image(args.image)
    print(f"  Calibration tensor shape: {calib_data.shape}")

    output_path = args.output or args.onnx.replace(".onnx", "_int8.onnx")

    print(f"\nQuantizing: {args.onnx}")
    print(f"  Method:          {args.calibration_method}")
    print(f"  Nodes excluded:  {args.nodes_to_exclude or 'none'}")
    print(f"  Output:          {output_path}")

    quantize(
        onnx_path=args.onnx,
        quantize_mode="int8",
        calibration_data={"input": calib_data},
        calibration_method=args.calibration_method,
        output_path=output_path,
        high_precision_dtype="fp16",
        nodes_to_exclude=args.nodes_to_exclude,
    )

    print(f"\nDone. Build TensorRT engine with:")
    print(f"  trtexec --onnx={output_path} --saveEngine=model.engine --int8 --fp16")


if __name__ == "__main__":
    main()
