#!/usr/bin/env python3
"""
Sensitivity Analysis for Depth Anything V2 using NVIDIA Model Optimizer.

=============================================================================
OVERVIEW
=============================================================================
This script performs layer-by-layer sensitivity analysis to identify which
layers in the Depth Anything V2 model cause the most accuracy degradation
when quantized to INT8. This information is crucial for creating an optimal
mixed-precision quantization configuration.

The key insight is that not all layers are equally tolerant to quantization:
- Some layers (like patch embeddings, early transformer blocks) are very
  sensitive and should remain in FP16/FP32
- Other layers (like middle MLP blocks) tolerate INT8 well and provide
  significant speedup when quantized

=============================================================================
HOW IT WORKS
=============================================================================
1. Load the model and calibration images
2. Compute FP16 baseline outputs (ground truth for comparison)
3. For each quantizable layer (Linear, Conv2d):
   a. Create a fresh model copy
   b. Quantize ONLY that single layer to INT8
   c. Run inference and measure error vs FP16 baseline
   d. Record the degradation metrics (MAE, RMSE)
4. Rank layers by sensitivity (higher error = more sensitive)
5. Group layers by type and compute average sensitivity per group
6. Generate recommendations for which layers to skip during quantization

=============================================================================
USAGE
=============================================================================
    python sensitivity_analysis.py \\
        --encoder vitl \\
        --checkpoint checkpoints/depth_anything_v2_vitl.pth \\
        --calib-dir ./calib_images \\
        --output sensitivity_results.json \\
        --num-calib 64 \\
        --num-eval 32

=============================================================================
OUTPUT
=============================================================================
- Console: Summary of most sensitive layers and recommended skip patterns
- JSON file: Detailed results for all layers, suitable for further analysis

=============================================================================
REFERENCES
=============================================================================
- NVIDIA Model Optimizer: https://github.com/NVIDIA/TensorRT-Model-Optimizer
- SmoothQuant paper: https://arxiv.org/abs/2211.10438
- Depth Anything V2: https://github.com/DepthAnything/Depth-Anything-V2
"""

import argparse
import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# =============================================================================
# NVIDIA Model Optimizer Import
# =============================================================================
# modelopt.torch.quantization (mtq) provides:
# - Quantization configs (INT8_DEFAULT_CFG, INT8_SMOOTHQUANT_CFG, etc.)
# - mtq.quantize() function to insert fake quantizers into model
# - Support for per-layer quantization control via config patterns
try:
    import modelopt.torch.quantization as mtq
except ImportError:
    raise ImportError(
        "nvidia-modelopt not installed. Install with: pip install nvidia-modelopt"
    )

from depth_anything_v2.dpt import DepthAnythingV2


# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================
# Depth Anything V2 comes in 4 sizes, each with different encoder dimensions
# and DPT head channel configurations. These must match the checkpoint.
MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
}


# =============================================================================
# IMAGE LOADING UTILITIES
# =============================================================================

def list_images(image_dir: Path, max_images: Optional[int] = None) -> List[Path]:
    """
    Recursively find all image files in a directory.

    Args:
        image_dir: Root directory to search
        max_images: Maximum number of images to return (None = no limit)

    Returns:
        Sorted list of image file paths
    """
    # Supported image extensions (common formats)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    files = []
    for p in image_dir.rglob("*"):  # rglob = recursive glob
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)

    # Sort for reproducibility (same images used across runs)
    files.sort()

    if max_images:
        files = files[:max_images]

    return files


def preprocess_image(
    img_path: Path,
    input_size: int = 518,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Preprocess a single image for Depth Anything V2 inference.

    This preprocessing MUST match what the model expects:
    1. Load image (BGR format from OpenCV)
    2. Convert BGR -> RGB
    3. Scale pixel values to [0, 1]
    4. Resize while keeping aspect ratio, ensuring dimensions are multiples of 14
       (required by ViT patch size of 14x14)
    5. Normalize with ImageNet mean/std (model was pretrained on ImageNet)
    6. Convert HWC -> CHW format (PyTorch convention)
    7. Add batch dimension

    Args:
        img_path: Path to image file
        input_size: Target size for the shorter edge (default 518 for Depth Anything)

    Returns:
        Tuple of (preprocessed tensor [1, 3, H, W], original image size (h, w))
    """
    # Load image using OpenCV (returns BGR format)
    raw_image = cv2.imread(str(img_path))
    if raw_image is None:
        raise ValueError(f"Could not load image: {img_path}")

    h, w = raw_image.shape[:2]

    # Convert BGR -> RGB and normalize to [0, 1] range
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

    # Resize keeping aspect ratio
    # The shorter edge becomes input_size, longer edge scales proportionally
    scale = input_size / min(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)

    # IMPORTANT: Dimensions must be multiples of 14 for ViT patch embedding
    # ViT divides image into 14x14 patches, so H and W must be divisible by 14
    new_h = (new_h // 14) * 14
    new_w = (new_w // 14) * 14

    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # ImageNet normalization (model was pretrained with these stats)
    # This centers the data and scales it to have unit variance
    mean = np.array([0.485, 0.456, 0.406])  # ImageNet RGB means
    std = np.array([0.229, 0.224, 0.225])   # ImageNet RGB stds
    image = (image - mean) / std

    # Convert from HWC (Height, Width, Channels) to CHW (Channels, Height, Width)
    # PyTorch expects CHW format, NumPy/OpenCV use HWC
    # Also add batch dimension: (H,W,C) -> (C,H,W) -> (1,C,H,W)
    image = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)

    return image, (h, w)


def load_calibration_data(
    calib_dir: Path,
    num_images: int,
    input_size: int = 518,
) -> List[torch.Tensor]:
    """
    Load and preprocess multiple images for calibration/evaluation.

    Calibration images are used by the quantizer to determine optimal
    scale factors for INT8 conversion. More diverse images = better calibration.

    Args:
        calib_dir: Directory containing calibration images
        num_images: Number of images to load
        input_size: Target input size for preprocessing

    Returns:
        List of preprocessed image tensors
    """
    image_paths = list_images(calib_dir, num_images)
    if not image_paths:
        raise RuntimeError(f"No images found in {calib_dir}")

    print(f"Loading {len(image_paths)} calibration images...")
    images = []

    for path in tqdm(image_paths, desc="Loading images"):
        try:
            img, _ = preprocess_image(path, input_size)
            images.append(img)
        except Exception as e:
            # Skip problematic images but continue with others
            print(f"Warning: Failed to load {path}: {e}")

    return images


# =============================================================================
# MODEL ANALYSIS UTILITIES
# =============================================================================

def get_quantizable_layers(model: nn.Module) -> List[str]:
    """
    Find all layers in the model that can be quantized.

    Quantization primarily targets:
    - nn.Linear: Dense/fully-connected layers (dominant in transformers)
    - nn.Conv2d: Convolutional layers (used in patch embedding and DPT head)

    Other layers like LayerNorm, Dropout, and activations are typically
    kept in FP16/FP32 as they have negligible compute cost.

    Args:
        model: PyTorch model to analyze

    Returns:
        List of layer names (dot-separated paths like "pretrained.blocks.0.attn.qkv")
    """
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            layers.append(name)
    return layers


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def compute_depth_error(
    depth_pred: torch.Tensor,
    depth_ref: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute error metrics between quantized and reference (FP16) depth outputs.

    Since we don't have ground truth depth, we use the FP16 model output as
    our reference and measure how much quantization degrades the output.

    We normalize both outputs to [0, 1] before comparison to ensure fair
    comparison regardless of the absolute depth scale.

    Metrics computed:
    - MAE (Mean Absolute Error): Average pixel-wise error, easy to interpret
    - RMSE (Root Mean Squared Error): Penalizes large errors more heavily
    - Max Error: Worst-case error, useful for detecting outliers

    Args:
        depth_pred: Predicted depth from quantized model
        depth_ref: Reference depth from FP16 model

    Returns:
        Dictionary with 'mae', 'rmse', and 'max_error' keys
    """
    # Flatten tensors for element-wise comparison
    pred = depth_pred.flatten()
    ref = depth_ref.flatten()

    # Normalize both to [0, 1] range for fair comparison
    # This removes scale differences and focuses on relative depth ordering
    # Adding small epsilon (1e-8) prevents division by zero
    pred_norm = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    ref_norm = (ref - ref.min()) / (ref.max() - ref.min() + 1e-8)

    # MAE: Average absolute difference
    # Lower is better; 0.01 means 1% average error
    mae = torch.mean(torch.abs(pred_norm - ref_norm)).item()

    # RMSE: Root mean squared error
    # More sensitive to large errors than MAE
    rmse = torch.sqrt(torch.mean((pred_norm - ref_norm) ** 2)).item()

    # Max error: Largest single-pixel error
    # Useful for identifying catastrophic quantization failures
    max_err = torch.max(torch.abs(pred_norm - ref_norm)).item()

    return {
        "mae": mae,
        "rmse": rmse,
        "max_error": max_err,
    }


def evaluate_model(
    model: nn.Module,
    eval_images: List[torch.Tensor],
    reference_outputs: Optional[List[torch.Tensor]] = None,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Evaluate a model on a set of images and compute average error metrics.

    This function runs inference on all evaluation images and computes
    error metrics against the reference (FP16) outputs.

    Args:
        model: Model to evaluate (may be quantized)
        eval_images: List of preprocessed input tensors
        reference_outputs: FP16 baseline outputs to compare against
        device: Device to run inference on

    Returns:
        Dictionary with averaged 'mae', 'rmse', and 'max_error'
    """
    model.eval()

    total_mae = 0.0
    total_rmse = 0.0
    total_max = 0.0

    with torch.no_grad():  # Disable gradient computation for inference
        for i, img in enumerate(eval_images):
            img = img.to(device)
            output = model(img)

            if reference_outputs is not None:
                ref = reference_outputs[i]
                metrics = compute_depth_error(output.cpu(), ref)
                total_mae += metrics["mae"]
                total_rmse += metrics["rmse"]
                total_max = max(total_max, metrics["max_error"])

    # Compute averages
    n = len(eval_images)
    return {
        "mae": total_mae / n if n > 0 else 0.0,
        "rmse": total_rmse / n if n > 0 else 0.0,
        "max_error": total_max,
    }


def get_reference_outputs(
    model: nn.Module,
    images: List[torch.Tensor],
    device: str = "cuda",
) -> List[torch.Tensor]:
    """
    Compute FP16/FP32 reference outputs for all evaluation images.

    These outputs serve as the "ground truth" for measuring quantization
    degradation. We compare quantized model outputs against these.

    Args:
        model: Original (non-quantized) model
        images: List of preprocessed input tensors
        device: Device to run inference on

    Returns:
        List of output tensors (stored on CPU to save GPU memory)
    """
    model.eval()
    outputs = []

    with torch.no_grad():
        for img in tqdm(images, desc="Computing reference outputs"):
            img = img.to(device)
            output = model(img)
            # Store on CPU to save GPU memory during sensitivity analysis
            outputs.append(output.cpu())

    return outputs


# =============================================================================
# QUANTIZATION CONFIGURATION
# =============================================================================

def create_single_layer_config(
    layer_name: str,
    base_config: Dict = None,
) -> Dict:
    """
    Create a quantization config that quantizes ONLY a single layer.

    This is the core of sensitivity analysis: by quantizing one layer at a time,
    we can isolate each layer's contribution to overall accuracy degradation.

    The config uses pattern matching to enable/disable quantization:
    - "*" matches all layers
    - "*layer_name*" matches specific layer
    - "weight_quantizer" controls weight quantization
    - "input_quantizer" controls activation quantization

    Args:
        layer_name: Full name of the layer to quantize (e.g., "pretrained.blocks.0.attn.qkv")
        base_config: Base config to start from (default: INT8_DEFAULT_CFG)

    Returns:
        Quantization config dictionary for use with mtq.quantize()
    """
    if base_config is None:
        base_config = mtq.INT8_DEFAULT_CFG

    cfg = deepcopy(base_config)

    # First, disable ALL quantization by setting "*" pattern to disabled
    cfg["quant_cfg"]["*"] = {"enable": False}

    # Escape dots in layer name for regex pattern matching
    # "pretrained.blocks.0" -> "pretrained\.blocks\.0"
    escaped_name = layer_name.replace(".", r"\.")

    # Enable quantization only for this specific layer
    # Weight quantizer: quantize weights with per-channel scaling (axis=0)
    cfg["quant_cfg"][f"*{escaped_name}*weight_quantizer"] = {"num_bits": 8, "axis": 0}
    # Input quantizer: quantize activations with per-tensor scaling (axis=None)
    cfg["quant_cfg"][f"*{escaped_name}*input_quantizer"] = {"num_bits": 8, "axis": None}

    return cfg


# =============================================================================
# CORE SENSITIVITY ANALYSIS
# =============================================================================

def run_sensitivity_analysis(
    model: nn.Module,
    calibration_images: List[torch.Tensor],
    eval_images: List[torch.Tensor],
    reference_outputs: List[torch.Tensor],
    device: str = "cuda",
    layer_filter: Optional[str] = None,
) -> Dict[str, Dict]:
    """
    Run layer-by-layer sensitivity analysis.

    This is the main analysis loop. For each quantizable layer:
    1. Create a fresh copy of the model (to avoid accumulated quantization)
    2. Quantize only that single layer to INT8
    3. Run calibration (required to determine quantization scales)
    4. Evaluate accuracy vs FP16 baseline
    5. Record the degradation metrics

    The key insight: layers with HIGH degradation are "sensitive" and should
    be kept in FP16. Layers with LOW degradation can be safely quantized to INT8.

    Args:
        model: Original model (will not be modified)
        calibration_images: Images for quantizer calibration
        eval_images: Images for accuracy evaluation
        reference_outputs: FP16 baseline outputs for comparison
        device: Device to run on
        layer_filter: Optional substring to filter which layers to analyze
                      (e.g., "pretrained" to analyze only backbone)

    Returns:
        Dictionary mapping layer names to their sensitivity metrics
    """
    # Get list of all quantizable layers
    all_layers = get_quantizable_layers(model)

    # Optionally filter to specific layers (useful for faster testing)
    if layer_filter:
        all_layers = [l for l in all_layers if layer_filter in l]

    print(f"\nFound {len(all_layers)} quantizable layers")
    print("=" * 60)

    results = {}

    for layer_name in tqdm(all_layers, desc="Analyzing layers"):
        try:
            # IMPORTANT: Create a fresh copy for each layer
            # We cannot reuse a quantized model because quantization modifies
            # the model in-place by inserting fake quantizer modules
            model_copy = deepcopy(model)
            model_copy = model_copy.to(device).eval()

            # Create config that only quantizes this one layer
            cfg = create_single_layer_config(layer_name)

            # Define calibration forward loop
            # The quantizer needs to see representative data to determine
            # optimal scale factors for converting FP32 -> INT8
            def forward_loop(m):
                for img in calibration_images[:32]:  # Use subset for speed
                    with torch.no_grad():
                        m(img.to(device))

            # Apply quantization with calibration
            # This inserts "fake quantizers" that simulate INT8 behavior
            # while keeping computations in FP32 (for accurate error measurement)
            mtq.quantize(model_copy, cfg, forward_loop)

            # Evaluate the quantized model
            metrics = evaluate_model(
                model_copy,
                eval_images,
                reference_outputs,
                device,
            )

            results[layer_name] = {
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "max_error": metrics["max_error"],
            }

            # Print progress (using tqdm.write to avoid messing up progress bar)
            tqdm.write(f"  {layer_name}: MAE={metrics['mae']:.6f}, RMSE={metrics['rmse']:.6f}")

            # Cleanup to free GPU memory
            del model_copy
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  Warning: Failed to analyze {layer_name}: {e}")
            results[layer_name] = {"error": str(e)}

    return results


# =============================================================================
# RESULTS ANALYSIS AND RECOMMENDATIONS
# =============================================================================

def analyze_results(results: Dict[str, Dict]) -> Dict:
    """
    Analyze sensitivity results and generate recommendations.

    This function:
    1. Ranks layers by sensitivity (MAE and RMSE)
    2. Groups layers by type (patch_embed, attention, mlp, etc.)
    3. Computes average sensitivity per group
    4. Generates skip patterns for layers that should stay in FP16

    Args:
        results: Dictionary of layer_name -> metrics from run_sensitivity_analysis()

    Returns:
        Dictionary with ranked layers, group averages, and recommendations
    """
    # Filter out layers that failed analysis
    valid_results = {k: v for k, v in results.items() if "error" not in v}

    # Rank by MAE (higher MAE = more sensitive = quantization hurts more)
    ranked_by_mae = sorted(
        valid_results.items(),
        key=lambda x: x[1]["mae"],
        reverse=True,  # Descending: most sensitive first
    )

    # Rank by RMSE (alternative metric, penalizes large errors more)
    ranked_by_rmse = sorted(
        valid_results.items(),
        key=lambda x: x[1]["rmse"],
        reverse=True,
    )

    # Group layers by their role in the model architecture
    # This helps identify patterns (e.g., "all attention layers are sensitive")
    groups = {
        "patch_embed": [],      # Initial patch embedding layer
        "blocks_early": [],     # Transformer blocks 0-3 (often sensitive)
        "blocks_middle": [],    # Transformer blocks 4-7 (often tolerant)
        "blocks_late": [],      # Transformer blocks 8+ (mixed sensitivity)
        "attention": [],        # Attention layers (qkv, proj)
        "mlp": [],              # MLP/FFN layers (often most tolerant)
        "head": [],             # DPT decoder head
        "other": [],            # Anything else
    }

    for name, metrics in valid_results.items():
        # Classify each layer into a group based on its name
        if "patch_embed" in name:
            groups["patch_embed"].append((name, metrics))
        elif "blocks." in name:
            # Extract block number using regex
            match = re.search(r"blocks\.(\d+)", name)
            if match:
                block_num = int(match.group(1))
                # Classify by block position (early blocks are often more sensitive)
                if block_num <= 3:
                    groups["blocks_early"].append((name, metrics))
                elif block_num <= 7:
                    groups["blocks_middle"].append((name, metrics))
                else:
                    groups["blocks_late"].append((name, metrics))
        elif "attn" in name.lower():
            groups["attention"].append((name, metrics))
        elif "mlp" in name.lower():
            groups["mlp"].append((name, metrics))
        elif "head" in name.lower() or "depth_head" in name:
            groups["head"].append((name, metrics))
        else:
            groups["other"].append((name, metrics))

    # Compute average sensitivity for each group
    # This helps make decisions like "skip all early blocks" vs individual layers
    group_avg = {}
    for group_name, layers in groups.items():
        if layers:
            avg_mae = np.mean([l[1]["mae"] for l in layers])
            avg_rmse = np.mean([l[1]["rmse"] for l in layers])
            group_avg[group_name] = {
                "avg_mae": float(avg_mae),
                "avg_rmse": float(avg_rmse),
                "num_layers": len(layers),
            }

    return {
        "ranked_by_mae": [(name, metrics) for name, metrics in ranked_by_mae[:20]],
        "ranked_by_rmse": [(name, metrics) for name, metrics in ranked_by_rmse[:20]],
        "group_averages": group_avg,
        "recommendations": generate_recommendations(group_avg, ranked_by_mae[:10]),
    }


def generate_recommendations(
    group_avg: Dict,
    top_sensitive: List[Tuple[str, Dict]],
) -> Dict:
    """
    Generate quantization skip patterns based on sensitivity analysis.

    Uses a threshold-based approach:
    - Groups/layers with MAE > threshold should be skipped (kept in FP16)
    - Groups/layers with MAE < threshold can be quantized to INT8

    The threshold (default 1% MAE) can be tuned based on accuracy requirements.

    Args:
        group_avg: Average sensitivity per layer group
        top_sensitive: Top 10 most sensitive individual layers

    Returns:
        Dictionary with skip_patterns, quantize_patterns, and suggested_config
    """
    skip_patterns = []
    quantize_patterns = []

    # Sensitivity threshold: layers above this MAE should be skipped
    # 0.01 = 1% mean absolute error is a reasonable starting point
    # Decrease for stricter accuracy, increase for more aggressive quantization
    threshold = 0.01

    # Analyze each group and decide whether to skip or quantize
    for group_name, stats in group_avg.items():
        if stats["avg_mae"] > threshold:
            # This group is sensitive - add skip patterns
            if group_name == "patch_embed":
                skip_patterns.append("*patch_embed*")
            elif group_name == "blocks_early":
                # Skip first 4 transformer blocks
                skip_patterns.extend([
                    "*blocks.0.*", "*blocks.1.*",
                    "*blocks.2.*", "*blocks.3.*"
                ])
            elif group_name == "attention":
                skip_patterns.append("*attn*")
            elif group_name == "head":
                skip_patterns.append("*head*")
        else:
            # This group tolerates quantization - add to quantize list
            if group_name == "mlp":
                quantize_patterns.append("*mlp*")
            elif group_name == "blocks_middle":
                quantize_patterns.append("*blocks.[4-7].*")

    # Also check top sensitive individual layers
    # Use higher threshold (2x) since these are outliers
    for name, metrics in top_sensitive:
        if metrics["mae"] > threshold * 2:
            # Extract a pattern from the layer name
            pattern = f"*{name.split('.')[-2]}*" if '.' in name else f"*{name}*"
            if pattern not in skip_patterns:
                skip_patterns.append(pattern)

    # Always skip normalization layers (they have negligible compute cost anyway)
    skip_patterns.append("*norm*")

    return {
        "skip_patterns": list(set(skip_patterns)),      # Remove duplicates
        "quantize_patterns": list(set(quantize_patterns)),
        "suggested_config": generate_config_code(skip_patterns),
    }


def generate_config_code(skip_patterns: List[str]) -> str:
    """
    Generate ready-to-use Python code for the recommended quantization config.

    This makes it easy to copy-paste the results into your quantization script.

    Args:
        skip_patterns: List of layer patterns to skip

    Returns:
        Python code string that can be executed to create the config
    """
    code = '''from copy import deepcopy
import modelopt.torch.quantization as mtq

# Depth Anything V2 optimized INT8 config based on sensitivity analysis
# Using SmoothQuant which works well for transformer architectures
DEPTH_ANYTHING_INT8_CFG = deepcopy(mtq.INT8_SMOOTHQUANT_CFG)

# Skip sensitive layers identified by sensitivity analysis
# These layers cause significant accuracy degradation when quantized
'''
    for pattern in skip_patterns:
        code += f'DEPTH_ANYTHING_INT8_CFG["quant_cfg"]["{pattern}"] = {{"enable": False}}\n'

    return code


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """
    Main function: parse arguments, run analysis, and save results.
    """
    parser = argparse.ArgumentParser(
        description="Sensitivity Analysis for Depth Anything V2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze ViT-Large model
  python sensitivity_analysis.py --encoder vitl --checkpoint checkpoints/depth_anything_v2_vitl.pth --calib-dir ./images

  # Analyze only backbone layers (faster)
  python sensitivity_analysis.py --encoder vitl --checkpoint ... --calib-dir ./images --layer-filter pretrained

  # Analyze only decoder head
  python sensitivity_analysis.py --encoder vitl --checkpoint ... --calib-dir ./images --layer-filter depth_head
        """
    )

    # Model configuration
    parser.add_argument(
        "--encoder",
        type=str,
        default="vitl",
        choices=["vits", "vitb", "vitl", "vitg"],
        help="Model encoder size: vits (small), vitb (base), vitl (large), vitg (giant)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth file)",
    )

    # Data configuration
    parser.add_argument(
        "--calib-dir",
        type=str,
        required=True,
        help="Directory containing calibration/evaluation images",
    )
    parser.add_argument(
        "--num-calib",
        type=int,
        default=64,
        help="Number of images for calibration (more = better but slower)",
    )
    parser.add_argument(
        "--num-eval",
        type=int,
        default=32,
        help="Number of images for evaluation (separate from calibration)",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=518,
        help="Input image size (default 518 for Depth Anything V2)",
    )

    # Output configuration
    parser.add_argument(
        "--output",
        type=str,
        default="sensitivity_results.json",
        help="Output JSON file for detailed results",
    )

    # Analysis options
    parser.add_argument(
        "--layer-filter",
        type=str,
        default=None,
        help="Only analyze layers containing this substring (e.g., 'pretrained' for backbone only)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)",
    )

    args = parser.parse_args()

    # =========================================================================
    # Setup
    # =========================================================================

    # Select device (fall back to CPU if CUDA unavailable)
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # =========================================================================
    # Load Model
    # =========================================================================

    print(f"\nLoading model: {args.encoder}")
    model = DepthAnythingV2(**MODEL_CONFIGS[args.encoder])
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model = model.to(device).eval()

    # =========================================================================
    # Load Images
    # =========================================================================

    calib_dir = Path(args.calib_dir)
    total_images = args.num_calib + args.num_eval

    all_images = load_calibration_data(calib_dir, total_images, args.input_size)

    # Split images into calibration and evaluation sets
    # Using separate sets prevents overfitting the quantization to eval data
    if len(all_images) < total_images:
        print(f"Warning: Only {len(all_images)} images available, adjusting split...")
        split = len(all_images) // 2
        calib_images = all_images[:split]
        eval_images = all_images[split:]
    else:
        calib_images = all_images[:args.num_calib]
        eval_images = all_images[args.num_calib:args.num_calib + args.num_eval]

    print(f"Calibration images: {len(calib_images)}")
    print(f"Evaluation images: {len(eval_images)}")

    # =========================================================================
    # Compute FP16 Baseline
    # =========================================================================

    print("\nComputing FP16 baseline outputs...")
    reference_outputs = get_reference_outputs(model, eval_images, device)

    # =========================================================================
    # Run Sensitivity Analysis
    # =========================================================================

    print("\nRunning sensitivity analysis...")
    print("This may take a while depending on model size and number of layers...")
    results = run_sensitivity_analysis(
        model,
        calib_images,
        eval_images,
        reference_outputs,
        device,
        args.layer_filter,
    )

    # =========================================================================
    # Analyze and Report Results
    # =========================================================================

    print("\nAnalyzing results...")
    analysis = analyze_results(results)

    # Print human-readable summary
    print("\n" + "=" * 60)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("=" * 60)

    print("\nTop 10 Most Sensitive Layers (by MAE):")
    print("(Higher MAE = more sensitive = should keep in FP16)")
    for i, (name, metrics) in enumerate(analysis["ranked_by_mae"][:10], 1):
        print(f"  {i:2d}. {name}: MAE={metrics['mae']:.6f}")

    print("\nGroup Average Sensitivities:")
    print("(Groups with high MAE should be skipped during quantization)")
    for group, stats in sorted(
        analysis["group_averages"].items(),
        key=lambda x: x[1]["avg_mae"],
        reverse=True,
    ):
        print(f"  {group:20s}: MAE={stats['avg_mae']:.6f} ({stats['num_layers']} layers)")

    print("\nRecommended Skip Patterns:")
    print("(Add these to your quantization config to skip sensitive layers)")
    for pattern in analysis["recommendations"]["skip_patterns"]:
        print(f"  - {pattern}")

    print("\n" + "-" * 60)
    print("Suggested Quantization Config:")
    print("-" * 60)
    print(analysis["recommendations"]["suggested_config"])

    # =========================================================================
    # Save Results to JSON
    # =========================================================================

    output_data = {
        "encoder": args.encoder,
        "checkpoint": args.checkpoint,
        "num_calib": len(calib_images),
        "num_eval": len(eval_images),
        "layer_results": results,
        "analysis": {
            "ranked_by_mae": analysis["ranked_by_mae"],
            "ranked_by_rmse": analysis["ranked_by_rmse"],
            "group_averages": analysis["group_averages"],
            "recommendations": {
                "skip_patterns": analysis["recommendations"]["skip_patterns"],
                "quantize_patterns": analysis["recommendations"]["quantize_patterns"],
            },
        },
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {args.output}")
    print("\nNext steps:")
    print("  1. Review the recommended skip patterns above")
    print("  2. Copy the suggested config into your quantization script")
    print("  3. Run full PTQ with the optimized config")
    print("  4. If accuracy is still too low, consider QAT (quantization-aware training)")


if __name__ == "__main__":
    main()
