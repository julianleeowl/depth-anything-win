#!/usr/bin/env python3
"""
Simple Sensitivity Analysis Demo

This script demonstrates the core concept of sensitivity analysis:
1. Run FP16 inference to get baseline output
2. Quantize ONE layer to INT8
3. Compare quantized output vs FP16 baseline
4. Visualize the difference

Usage:
    python sensitivity_demo.py \
        --checkpoint checkpoints/depth_anything_v2_vits.pth \
        --image test.jpg \
        --layer pretrained.blocks.0.attn.qkv
"""

import argparse
import os
from copy import deepcopy

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

# NVIDIA Model Optimizer
try:
    import modelopt.torch.quantization as mtq
except ImportError:
    raise ImportError("Install with: pip install nvidia-modelopt")

from depth_anything_v2.dpt import DepthAnythingV2


# =============================================================================
# CONFIGURATION
# =============================================================================

# Using ViT-Small for faster demo (change to vitl for production)
MODEL_CONFIG = {
    'encoder': 'vits',
    'features': 64,
    'out_channels': [48, 96, 192, 384]
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def preprocess_image(image_path: str, input_size: int = 518) -> torch.Tensor:
    """Load and preprocess image for Depth Anything V2."""
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    h, w = img.shape[:2]

    # Convert BGR -> RGB, normalize to [0, 1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    # Resize (must be multiple of 14 for ViT)
    scale = input_size / min(h, w)
    new_h = (int(h * scale) // 14) * 14
    new_w = (int(w * scale) // 14) * 14
    img = cv2.resize(img, (new_w, new_h))

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std

    # HWC -> CHW, add batch dim
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)

    return tensor


def list_quantizable_layers(model: torch.nn.Module) -> list:
    """List all Linear and Conv2d layers in the model."""
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            layers.append(name)
    return layers


def create_single_layer_quant_config(layer_name: str) -> dict:
    """
    Create a config that quantizes ONLY the specified layer.
    All other layers remain in FP16.
    """
    cfg = deepcopy(mtq.INT8_DEFAULT_CFG)

    # Disable all quantization first
    cfg["quant_cfg"]["*"] = {"enable": False}

    # Enable only for target layer (escape dots for regex)
    escaped = layer_name.replace(".", r"\.")
    cfg["quant_cfg"][f"*{escaped}*weight_quantizer"] = {"num_bits": 8, "axis": 0}
    cfg["quant_cfg"][f"*{escaped}*input_quantizer"] = {"num_bits": 8, "axis": None}

    return cfg


def compute_error(pred: torch.Tensor, ref: torch.Tensor) -> dict:
    """Compute error metrics between quantized and reference outputs."""
    # Normalize both to [0, 1] for fair comparison
    pred_norm = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    ref_norm = (ref - ref.min()) / (ref.max() - ref.min() + 1e-8)

    diff = torch.abs(pred_norm - ref_norm)

    return {
        "mae": diff.mean().item(),
        "rmse": torch.sqrt((diff ** 2).mean()).item(),
        "max_error": diff.max().item(),
    }


# =============================================================================
# MAIN DEMO
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Sensitivity Analysis Demo")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to test image")
    parser.add_argument("--layer", type=str, default=None,
                        help="Layer to quantize (default: first attention layer)")
    parser.add_argument("--list-layers", action="store_true",
                        help="List all quantizable layers and exit")
    parser.add_argument("--all-layers", action="store_true",
                        help="Quantize ALL layers instead of just one")
    parser.add_argument("--output-onnx", type=str, default="depth_anything_int8.onnx",
                        help="Output ONNX path (only used with --all-layers)")
    parser.add_argument("--encoder", type=str, default="vits",
                        choices=["vits", "vitb", "vitl"],
                        help="Model encoder size")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # =========================================================================
    # Step 1: Load Model
    # =========================================================================
    print("\n[1] Loading model...")

    configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    model = DepthAnythingV2(**configs[args.encoder])
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model = model.to(device).eval()
    print(f"    Loaded {args.encoder} model from {args.checkpoint}")

    # List layers if requested
    layers = list_quantizable_layers(model)
    if args.list_layers:
        print(f"\nQuantizable layers ({len(layers)} total):")
        for i, name in enumerate(layers):
            print(f"  {i:3d}. {name}")
        return

    # Select layer to quantize
    if args.layer:
        target_layer = args.layer
    else:
        # Default: first attention QKV projection (typically sensitive)
        target_layer = next((l for l in layers if "attn.qkv" in l), layers[0])

    print(f"    Target layer: {target_layer}")

    # =========================================================================
    # Step 2: Load and Preprocess Image
    # =========================================================================
    print("\n[2] Loading image...")
    image = preprocess_image(args.image)
    image = image.to(device)
    print(f"    Image shape: {image.shape}")

    # =========================================================================
    # Step 3: Run FP16 Baseline Inference
    # =========================================================================
    print("\n[3] Running FP16 baseline inference...")
    with torch.no_grad():
        depth_fp16 = model(image)
    print(f"    Output shape: {depth_fp16.shape}")
    print(f"    Depth range: [{depth_fp16.min().item():.3f}, {depth_fp16.max().item():.3f}]")

    # =========================================================================
    # Step 4: Quantize Layer(s)
    # =========================================================================
    if args.all_layers:
        print("\n[4] Quantizing ALL layers (INT8 SmoothQuant)...")
    else:
        print(f"\n[4] Quantizing single layer: {target_layer}")

    # Create a fresh copy of the model
    model_quant = deepcopy(model)
    model_quant = model_quant.to(device).eval()

    # Create quantization config
    if args.all_layers:
        # Quantize all layers using SmoothQuant (good for transformers)
        # quant_config = mtq.INT8_SMOOTHQUANT_CFG
        quant_config = mtq.INT8_DEFAULT_CFG
    else:
        # Quantize only the target layer
        quant_config = create_single_layer_quant_config(target_layer)

    # Calibration forward pass (quantizer needs to see data to set scales)
    def calibration_forward(m):
        with torch.no_grad():
            m(image)

    # Apply quantization
    mtq.quantize(model_quant, quant_config, calibration_forward)

    if args.all_layers:
        print("    Quantization applied to ALL Linear/Conv2d layers")
    else:
        print("    Quantization applied (INT8 weights + activations for this layer)")

    # =========================================================================
    # Step 4b: Export to ONNX (only when --all-layers is used)
    # =========================================================================
    if args.all_layers:
        print(f"\n[4b] Exporting quantized model to ONNX...")
        onnx_path = args.output_onnx

        try:
            # Move model to CPU for export (more stable, avoids GPU memory issues)
            model_export = deepcopy(model_quant).cpu().eval()
            dummy_input = torch.randn(1, 3, 518, 518).cpu()

            # Use torch.onnx.export with fixed shape (most compatible)
            torch.onnx.export(
                model_export,
                dummy_input,
                onnx_path,
                opset_version=14,  # Use opset 14 for better compatibility
                input_names=["input"],
                output_names=["depth"],
                do_constant_folding=True,
                dynamo=False,  # Use legacy TorchScript exporter
                verbose=False,
            )
            print(f"    Exported quantized ONNX to: {onnx_path}")

            # Print file size
            if os.path.exists(onnx_path):
                size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
                print(f"    ONNX file size: {size_mb:.1f} MB")

            # Cleanup
            del model_export
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"    ONNX export failed: {e}")
            print(f"    Continuing with evaluation anyway...")

    # =========================================================================
    # Step 5: Run Quantized Inference
    # =========================================================================
    print("\n[5] Running quantized inference...")
    with torch.no_grad():
        depth_quant = model_quant(image)
    print(f"    Depth range: [{depth_quant.min().item():.3f}, {depth_quant.max().item():.3f}]")

    # =========================================================================
    # Step 6: Compare Results
    # =========================================================================
    print("\n[6] Comparing FP16 vs Quantized outputs...")

    errors = compute_error(depth_quant, depth_fp16)

    print(f"\n    {'='*50}")
    if args.all_layers:
        print(f"    QUANTIZATION RESULTS: ALL LAYERS (INT8)")
    else:
        print(f"    SENSITIVITY RESULTS for: {target_layer}")
    print(f"    {'='*50}")
    print(f"    MAE  (Mean Absolute Error):  {errors['mae']:.6f}")
    print(f"    RMSE (Root Mean Sq Error):   {errors['rmse']:.6f}")
    print(f"    Max Error:                   {errors['max_error']:.6f}")
    print(f"    {'='*50}")

    # Interpret results
    if errors['mae'] < 0.001:
        sensitivity = "LOW (safe to quantize)"
    elif errors['mae'] < 0.01:
        sensitivity = "MEDIUM (test carefully)"
    else:
        sensitivity = "HIGH (consider keeping in FP16)"

    print(f"\n    Sensitivity: {sensitivity}")

    # =========================================================================
    # Step 7: Visualize
    # =========================================================================
    print("\n[7] Generating visualization...")

    # Convert to numpy for visualization
    fp16_np = depth_fp16[0, 0].cpu().numpy()
    quant_np = depth_quant[0, 0].cpu().numpy()

    # Normalize for display
    fp16_display = (fp16_np - fp16_np.min()) / (fp16_np.max() - fp16_np.min())
    quant_display = (quant_np - quant_np.min()) / (quant_np.max() - quant_np.min())
    diff_display = np.abs(fp16_display - quant_display)

    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Original image
    orig_img = cv2.imread(args.image)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    axes[0].imshow(orig_img)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    # FP16 depth
    im1 = axes[1].imshow(fp16_display, cmap="magma")
    axes[1].set_title("FP16 Depth (Baseline)")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # Quantized depth
    im2 = axes[2].imshow(quant_display, cmap="magma")
    if args.all_layers:
        axes[2].set_title("INT8 Quantized\n(ALL layers)")
    else:
        axes[2].set_title(f"INT8 Quantized\n(layer: {target_layer.split('.')[-1]})")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    # Difference map
    im3 = axes[3].imshow(diff_display, cmap="hot")
    axes[3].set_title(f"Absolute Difference\nMAE={errors['mae']:.4f}")
    axes[3].axis("off")
    plt.colorbar(im3, ax=axes[3], fraction=0.046)

    if args.all_layers:
        plt.suptitle("Full Model INT8 Quantization vs FP16", fontsize=12)
    else:
        plt.suptitle(f"Sensitivity Analysis: {target_layer}", fontsize=12)
    plt.tight_layout()

    output_path = "sensitivity_demo_output.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"    Saved visualization to: {output_path}")

    plt.show()

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if args.all_layers:
        print(f"""
This demo quantized ALL layers to INT8 and exported to ONNX.

Key metrics:
- MAE = {errors['mae']:.6f} (lower is better, <0.01 is usually acceptable)
- Sensitivity: {sensitivity}

Exported ONNX: {args.output_onnx}

Next steps - build TensorRT engine:
    trtexec --onnx={args.output_onnx} --saveEngine=depth_anything_int8.engine --int8 --fp16

If accuracy is too low, run per-layer sensitivity analysis to find
which layers to skip:
    python sensitivity_analysis.py --checkpoint {args.checkpoint} --calib-dir ./images

To test a single layer:
    python sensitivity_demo.py --checkpoint {args.checkpoint} --image {args.image} --layer <layer_name>
""")
    else:
        print(f"""
This demo quantized a single layer ({target_layer}) to INT8
and measured the output degradation compared to FP16.

Key metrics:
- MAE = {errors['mae']:.6f} (lower is better, <0.01 is usually acceptable)
- Sensitivity: {sensitivity}

To quantize ALL layers at once:
    python sensitivity_demo.py --checkpoint {args.checkpoint} --image {args.image} --all-layers

To run full sensitivity analysis on all layers:
    python sensitivity_analysis.py --checkpoint {args.checkpoint} --calib-dir ./images

To test a different layer:
    python sensitivity_demo.py --checkpoint {args.checkpoint} --image {args.image} --layer <layer_name>

To list all layers:
    python sensitivity_demo.py --checkpoint {args.checkpoint} --image {args.image} --list-layers
""")


if __name__ == "__main__":
    main()
