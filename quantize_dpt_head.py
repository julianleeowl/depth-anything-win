#!/usr/bin/env python3
"""
Quantize only the DPT head (decoder) of Depth Anything V2 to INT8,
keeping the ViT backbone in FP16. Compares against FP16 baseline
and visualizes results.

The DPT head contains Conv2d and ConvTranspose2d layers used for
feature projection, upsampling, and fusion. These are typically
less sensitive to quantization than the ViT attention layers.

Usage:
    python quantize_dpt_head.py \
        --checkpoint checkpoints/depth_anything_v2_vits.pth \
        --image test.jpg

    # Use ViT-Large encoder
    python quantize_dpt_head.py \
        --checkpoint checkpoints/depth_anything_v2_vitl.pth \
        --image test.jpg \
        --encoder vitl

    # Export quantized model to ONNX
    python quantize_dpt_head.py \
        --checkpoint checkpoints/depth_anything_v2_vits.pth \
        --image test.jpg \
        --export-onnx depth_anything_head_int8.onnx
"""

import argparse
import os
from copy import deepcopy

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
try:
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.utils import export_torch_mode
except ImportError:
    raise ImportError("Install with: pip install nvidia-modelopt")

from depth_anything_v2.dpt import DepthAnythingV2


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}


# ============================================================================
# IMAGE LOADING
# ============================================================================

def load_image(image_path: str, input_size: int = 518):
    """Load and preprocess image for Depth Anything V2."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    target_size = (input_size // 14) * 14
    img_resized = cv2.resize(img_rgb, (target_size, target_size))
    original_display = img_resized.copy()

    img_norm = img_resized / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_norm = (img_norm - mean) / std

    tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor, original_display


# ============================================================================
# DPT-HEAD-ONLY QUANTIZATION CONFIG
# ============================================================================

def create_dpt_head_quant_config():
    """
    Create a quantization config that only quantizes the DPT head layers.

    The model structure is:
        model.pretrained.*     -> ViT backbone (keep FP16)
        model.depth_head.*     -> DPT decoder (quantize to INT8)

    This uses modelopt's pattern matching to selectively enable quantization.
    """
    cfg = deepcopy(mtq.INT8_DEFAULT_CFG)

    # Disable all quantization by default
    cfg["quant_cfg"]["*"] = {"enable": False}

    # Enable INT8 only for depth_head layers
    # weight_quantizer: per-channel (axis=0) for Conv2d/Linear weights
    # input_quantizer: per-tensor (axis=None) for activations
    cfg["quant_cfg"]["*depth_head*weight_quantizer"] = {"num_bits": 8, "axis": 0}
    cfg["quant_cfg"]["*depth_head*input_quantizer"] = {"num_bits": 8, "axis": None}

    return cfg


def list_quantized_layers(model):
    """List all layers in the DPT head that will be quantized."""
    layers = []
    for name, module in model.named_modules():
        if "depth_head" in name and isinstance(
            module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear)
        ):
            layers.append((name, type(module).__name__))
    return layers


# ============================================================================
# ERROR METRICS
# ============================================================================

def compute_error(pred: np.ndarray, ref: np.ndarray) -> dict:
    """Compute error between quantized and FP16 reference."""
    pred_n = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    ref_n = (ref - ref.min()) / (ref.max() - ref.min() + 1e-8)
    diff = np.abs(pred_n - ref_n)
    return {
        "mae": diff.mean(),
        "rmse": np.sqrt((diff ** 2).mean()),
        "max_error": diff.max(),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Quantize DPT Head Only (INT8) — ViT backbone stays FP16"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--encoder", type=str, default="vits",
                        choices=["vits", "vitb", "vitl"])
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--export-onnx", type=str, default=None,
                        help="Export quantized model to ONNX")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load image
    print(f"Loading image: {args.image}")
    input_tensor, original_image = load_image(args.image, args.input_size)
    input_tensor = input_tensor.to(device)

    print("=" * 60)
    print("DPT HEAD INT8 QUANTIZATION")
    print("=" * 60)
    print(f"Encoder:    {args.encoder}")
    print(f"Input:      {input_tensor.shape}")
    print(f"Device:     {device}")
    print("=" * 60)

    # ========================================================================
    # Step 1: Load model and list DPT head layers
    # ========================================================================
    print("\n[1/5] Loading model...")
    model = DepthAnythingV2(**MODEL_CONFIGS[args.encoder])
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model = model.to(device).eval()

    layers = list_quantized_layers(model)
    print(f"  DPT head quantizable layers ({len(layers)}):")
    for name, ltype in layers:
        print(f"    {name} ({ltype})")

    # ========================================================================
    # Step 2: FP16 baseline inference
    # ========================================================================
    print("\n[2/5] Running FP16 baseline inference...")
    with torch.no_grad():
        depth_fp16 = model(input_tensor)
    depth_fp16_np = depth_fp16[0, 0].cpu().numpy()
    print(f"  Depth range: [{depth_fp16_np.min():.3f}, {depth_fp16_np.max():.3f}]")

    # ========================================================================
    # Step 3: Quantize DPT head only
    # ========================================================================
    print("\n[3/5] Quantizing DPT head to INT8 (backbone stays FP16)...")
    model_quant = deepcopy(model).to(device).eval()

    quant_config = create_dpt_head_quant_config()

    def calibration_forward(m):
        with torch.no_grad():
            m(input_tensor)

    mtq.quantize(model_quant, quant_config, calibration_forward)
    print("  Quantization applied to DPT head layers")

    # ========================================================================
    # Step 4: Quantized inference
    # ========================================================================
    print("\n[4/5] Running quantized inference...")
    with torch.no_grad():
        depth_quant = model_quant(input_tensor)
    depth_quant_np = depth_quant[0, 0].cpu().numpy()
    print(f"  Depth range: [{depth_quant_np.min():.3f}, {depth_quant_np.max():.3f}]")

    # Error metrics
    errors = compute_error(depth_quant_np, depth_fp16_np)
    print(f"\n  {'='*50}")
    print(f"  DPT HEAD QUANTIZATION RESULTS")
    print(f"  {'='*50}")
    print(f"  MAE:       {errors['mae']:.6f}")
    print(f"  RMSE:      {errors['rmse']:.6f}")
    print(f"  Max Error: {errors['max_error']:.6f}")
    print(f"  {'='*50}")

    if errors['mae'] < 0.001:
        quality = "EXCELLENT (negligible degradation)"
    elif errors['mae'] < 0.01:
        quality = "GOOD (minor degradation)"
    elif errors['mae'] < 0.05:
        quality = "ACCEPTABLE (noticeable degradation)"
    else:
        quality = "POOR (significant degradation)"
    print(f"  Quality:   {quality}")

    # ========================================================================
    # Step 4b: ONNX export (optional)
    # ========================================================================
    if args.export_onnx:
        print(f"\n[4b] Exporting quantized model to ONNX: {args.export_onnx}")
        try:
            model_export = deepcopy(model_quant).cpu().eval()
            dummy_input = torch.randn(1, 3, 518, 518).cpu()

            torch.onnx.export(
                model_export,
                dummy_input,
                args.export_onnx,
                opset_version=14,
                input_names=["input"],
                output_names=["depth"],
                do_constant_folding=True,
                dynamo=False,
            )

            with export_torch_mode():
                torch.onnx.export(
                    model_export,
                    dummy_input,
                    args.export_onnx,
                    opset_version=14,
                    input_names=["input"],
                    output_names=["depth"],
                    do_constant_folding=True,
                    dynamo=False,
                )          

            if os.path.exists(args.export_onnx):
                size_mb = os.path.getsize(args.export_onnx) / (1024 * 1024)
                print(f"  Exported: {args.export_onnx} ({size_mb:.1f} MB)")

            del model_export
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ONNX export failed: {e}")

    # ========================================================================
    # Step 5: Visualization
    # ========================================================================
    print("\n[5/5] Generating comparison plot...")

    # Normalize depth maps for display
    fp16_disp = (depth_fp16_np - depth_fp16_np.min()) / (depth_fp16_np.max() - depth_fp16_np.min() + 1e-8)
    quant_disp = (depth_quant_np - depth_quant_np.min()) / (depth_quant_np.max() - depth_quant_np.min() + 1e-8)
    diff_disp = np.abs(fp16_disp - quant_disp)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image", fontsize=10)
    axes[0].axis("off")

    # FP16 baseline
    axes[1].imshow(fp16_disp, cmap="magma")
    axes[1].set_title("FP16 Baseline\n(full model)", fontsize=9)
    axes[1].axis("off")

    # Quantized (DPT head INT8)
    axes[2].imshow(quant_disp, cmap="magma")
    axes[2].set_title("DPT Head INT8\n(backbone FP16)", fontsize=9)
    axes[2].axis("off")

    # Difference map
    im = axes[3].imshow(diff_disp, cmap="hot")
    axes[3].set_title(f"Absolute Difference\nMAE={errors['mae']:.4f}", fontsize=9)
    axes[3].axis("off")
    plt.colorbar(im, ax=axes[3], fraction=0.046)

    plt.suptitle(
        f"Depth Anything V2 ({args.encoder}): DPT Head INT8 Quantization",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()

    output_path = "dpt_head_int8_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved plot to: {output_path}")
    plt.show()

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"""
Quantized: DPT head only ({len(layers)} layers) -> INT8
Kept FP16: ViT backbone (pretrained.*)
MAE: {errors['mae']:.6f}
Quality: {quality}

This approach is useful for TensorRT mixed-precision deployment:
  - ViT backbone runs in FP16 (avoids Q/DQ reformat overhead)
  - DPT head runs in INT8 (Conv2d layers benefit from INT8 tensor cores)

To build TensorRT engine:
  python quantize_dpt_head.py --checkpoint {args.checkpoint} --image {args.image} --export-onnx dpt_head_int8.onnx
  trtexec --onnx=dpt_head_int8.onnx --saveEngine=model.engine --int8 --fp16
""")


if __name__ == "__main__":
    main()
