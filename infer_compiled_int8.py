#!/usr/bin/env python3
"""
PyTorch Compiled + Flash Attention + INT8 Quantization Inference

Combines three optimizations for maximum PyTorch-native inference speed:
1. TorchAO INT8 quantization (weight-only or dynamic)
2. Flash Attention (or xFormers fallback)
3. torch.compile with max-autotune

Benchmarks the combined pipeline against an FP16 eager baseline and
visualizes depth outputs side by side.

Usage:
    # Full pipeline: INT8 weight-only + Flash Attention + torch.compile
    python infer_compiled_int8.py \
        --checkpoint checkpoints/depth_anything_v2_vits.pth \
        --image test.jpg

    # INT8 dynamic quantization (more aggressive)
    python infer_compiled_int8.py \
        --checkpoint checkpoints/depth_anything_v2_vits.pth \
        --image test.jpg \
        --quant-mode int8dq

    # Disable individual optimizations to isolate their effect
    python infer_compiled_int8.py \
        --checkpoint checkpoints/depth_anything_v2_vits.pth \
        --image test.jpg \
        --no-compile          # skip torch.compile
        --no-flash-attn       # skip flash attention
        --quant-mode none     # skip quantization (FP16 only)

Requirements:
    pip install torchao flash-attn --no-build-isolation
"""

import argparse
import time
from copy import deepcopy

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from depth_anything_v2.dpt import DepthAnythingV2

# ============================================================================
# DEPENDENCY CHECKS
# ============================================================================

TORCHAO_AVAILABLE = False
try:
    from torchao.quantization import quantize_, int8_weight_only
    TORCHAO_AVAILABLE = True
    print("TorchAO: Available")
except ImportError:
    print("TorchAO: Not available (pip install torchao)")

TORCHAO_DYNAMIC_AVAILABLE = False
try:
    from torchao.quantization import int8_dynamic_activation_int8_weight
    TORCHAO_DYNAMIC_AVAILABLE = True
except ImportError:
    pass

FLASH_ATTN_AVAILABLE = False
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
    print("Flash Attention: Available")
except ImportError:
    print("Flash Attention: Not available (pip install flash-attn)")

XFORMERS_AVAILABLE = False
try:
    import xformers.ops as xops
    XFORMERS_AVAILABLE = True
    print("xFormers: Available")
except ImportError:
    print("xFormers: Not available (pip install xformers)")


# ============================================================================
# MODEL CONFIGS
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
    """Load and preprocess image for Depth Anything V2.
    Returns (tensor, original_image_for_display).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to fixed size (must be multiple of 14 for ViT)
    target_size = (input_size // 14) * 14
    img_resized = cv2.resize(img_rgb, (target_size, target_size))
    original_display = img_resized.copy()

    # Normalize to [0, 1], then ImageNet normalize
    img_norm = img_resized / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_norm = (img_norm - mean) / std

    # HWC -> CHW, add batch dim
    tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor, original_display


# ============================================================================
# FLASH ATTENTION MONKEY PATCH
# ============================================================================

def patch_attention_with_flash(model):
    """Replace standard attention with Flash Attention in DINOv2 backbone."""
    if not FLASH_ATTN_AVAILABLE and not XFORMERS_AVAILABLE:
        print("  Warning: Neither Flash Attention nor xFormers available, skipping")
        return model

    from depth_anything_v2.dinov2_layers.attention import Attention

    def flash_attention_forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 1, 3, 4)  # (3, B, N, num_heads, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if FLASH_ATTN_AVAILABLE:
            out = flash_attn_func(q, k, v, causal=False)
        elif XFORMERS_AVAILABLE:
            out = xops.memory_efficient_attention(q, k, v)

        out = out.reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    patched = 0
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            module.forward = lambda x, m=module: flash_attention_forward(m, x)
            patched += 1

    print(f"  Patched {patched} attention modules with Flash Attention")
    return model


# ============================================================================
# INT8 QUANTIZATION
# ============================================================================

def apply_int8_quantization(model, quant_mode: str):
    """Apply TorchAO INT8 quantization to the model.

    Args:
        model: Model in FP32 on CPU (before .half() or .cuda())
        quant_mode: "int8wo" for weight-only, "int8dq" for dynamic activation+weight

    Returns:
        Quantized model (still on CPU, call .cuda().half() after)
    """
    if quant_mode == "none":
        print("  Quantization: disabled (FP16 only)")
        return model

    if not TORCHAO_AVAILABLE:
        raise RuntimeError("TorchAO not installed. Install with: pip install torchao")

    if quant_mode == "int8wo":
        print("  Applying INT8 weight-only quantization (TorchAO)...")
        print("    - Weights stored as INT8, dequantized to FP16 during compute")
        print("    - Reduces memory bandwidth, no calibration needed")
        quantize_(model, int8_weight_only())

    elif quant_mode == "int8dq":
        if not TORCHAO_DYNAMIC_AVAILABLE:
            raise RuntimeError(
                "int8_dynamic_activation_int8_weight not available in your torchao version. "
                "Update with: pip install -U torchao"
            )
        print("  Applying INT8 dynamic activation + weight quantization (TorchAO)...")
        print("    - Both weights and activations quantized to INT8")
        print("    - More aggressive, uses INT8 tensor cores")
        quantize_(model, int8_dynamic_activation_int8_weight())

    else:
        raise ValueError(f"Unknown quant_mode: {quant_mode}. Use: int8wo, int8dq, none")

    # Count quantized layers
    n_linear = sum(1 for _, m in model.named_modules() if isinstance(m, torch.nn.Linear))
    print(f"    Quantized {n_linear} Linear layers")

    return model


# ============================================================================
# BENCHMARK
# ============================================================================

def benchmark_model(model, input_tensor, warmup=10, iterations=100, name="Model"):
    """Benchmark with CUDA event timing. Returns dict with latency stats and depth output."""
    print(f"\nBenchmarking: {name}")
    print(f"  Inference input size: {input_tensor.shape[3]}x{input_tensor.shape[2]} (WxH)")

    # Warmup
    print(f"  Warming up ({warmup} iterations)...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
    torch.cuda.synchronize()

    # Timed runs with CUDA events
    print(f"  Running benchmark ({iterations} iterations)...")
    latencies = []
    with torch.no_grad():
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = model(input_tensor)
            end.record()
            torch.cuda.synchronize()
            latencies.append(start.elapsed_time(end))

    latencies = torch.tensor(latencies)
    mean_ms = latencies.mean().item()
    std_ms = latencies.std().item()

    print(f"  Results: {mean_ms:.2f} +/- {std_ms:.2f} ms ({1000/mean_ms:.1f} FPS)")

    # Capture output
    with torch.no_grad():
        output = model(input_tensor)
        depth = output.cpu().numpy()

    return {
        "name": name,
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "fps": 1000 / mean_ms,
        "depth": depth,
    }


# ============================================================================
# ERROR METRICS
# ============================================================================

def compute_error(pred: np.ndarray, ref: np.ndarray) -> dict:
    """Compute error between quantized and FP16 reference depth outputs."""
    # Normalize both to [0, 1]
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
        description="PyTorch Compiled + Flash Attention + INT8 Inference"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--encoder", type=str, default="vits",
                        choices=["vits", "vitb", "vitl"])
    parser.add_argument("--quant-mode", type=str, default="int8wo",
                        choices=["int8wo", "int8dq", "none"],
                        help="INT8 quantization mode (default: int8wo)")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--no-flash-attn", action="store_true",
                        help="Disable Flash Attention")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile")
    args = parser.parse_args()

    device = "cuda"

    # Load image
    print(f"Loading image: {args.image}")
    input_tensor, original_image = load_image(args.image, args.input_size)
    input_tensor = input_tensor.to(device).half()

    print("=" * 60)
    print("COMPILED INT8 INFERENCE BENCHMARK")
    print("=" * 60)
    print(f"Encoder:        {args.encoder}")
    print(f"Quantization:   {args.quant_mode}")
    print(f"Flash Attention: {'OFF' if args.no_flash_attn else 'ON'}")
    print(f"torch.compile:  {'OFF' if args.no_compile else 'ON'}")
    print(f"Input:          {input_tensor.shape}")
    print("=" * 60)

    results = []

    # ========================================================================
    # 1. FP16 Eager Baseline
    # ========================================================================
    print("\n[1/2] FP16 Eager Baseline")
    model_baseline = DepthAnythingV2(**MODEL_CONFIGS[args.encoder])
    model_baseline.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model_baseline = model_baseline.to(device).half().eval()

    results.append(benchmark_model(
        model_baseline, input_tensor,
        args.warmup, args.iterations,
        "Eager FP16 (baseline)"
    ))

    # Free baseline model memory
    del model_baseline
    torch.cuda.empty_cache()

    # ========================================================================
    # 2. INT8 + Flash Attention + torch.compile
    # ========================================================================
    # Build the method name dynamically
    parts = []
    if args.quant_mode != "none":
        parts.append(f"INT8 ({args.quant_mode})")
    if not args.no_flash_attn and (FLASH_ATTN_AVAILABLE or XFORMERS_AVAILABLE):
        parts.append("FlashAttn")
    if not args.no_compile:
        parts.append("compiled")
    method_name = " + ".join(parts) if parts else "Eager FP16"

    print(f"\n[2/2] {method_name}")

    # Step A: Load fresh model (FP32 on CPU)
    print("  Loading model...")
    model = DepthAnythingV2(**MODEL_CONFIGS[args.encoder])
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    # Step B: Apply INT8 quantization (on CPU, FP32)
    if args.quant_mode != "none":
        model = apply_int8_quantization(model, args.quant_mode)

    # Step C: Move to GPU and convert to FP16
    model = model.to(device).half()

    # Step D: Patch Flash Attention
    if not args.no_flash_attn:
        model = patch_attention_with_flash(model)

    # Step E: Apply torch.compile
    if not args.no_compile:
        print("  Compiling with torch.compile (max-autotune)...")
        model = torch.compile(model, mode="max-autotune", fullgraph=False)

    results.append(benchmark_model(
        model, input_tensor,
        args.warmup, args.iterations,
        method_name
    ))

    # ========================================================================
    # Accuracy Comparison
    # ========================================================================
    print("\n" + "=" * 60)
    print("ACCURACY COMPARISON (vs FP16 baseline)")
    print("=" * 60)

    ref_depth = results[0]["depth"]
    opt_depth = results[1]["depth"]
    errors = compute_error(opt_depth, ref_depth)

    print(f"  MAE:       {errors['mae']:.6f}")
    print(f"  RMSE:      {errors['rmse']:.6f}")
    print(f"  Max Error: {errors['max_error']:.6f}")

    if errors['mae'] < 0.001:
        print(f"  Quality:   EXCELLENT (negligible degradation)")
    elif errors['mae'] < 0.01:
        print(f"  Quality:   GOOD (minor degradation)")
    elif errors['mae'] < 0.05:
        print(f"  Quality:   ACCEPTABLE (noticeable degradation)")
    else:
        print(f"  Quality:   POOR (significant degradation)")

    # ========================================================================
    # Performance Summary
    # ========================================================================
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"{'Method':<35} {'Latency':<12} {'FPS':<10} {'Speedup':<10}")
    print("-" * 65)

    baseline_ms = results[0]["mean_ms"]
    for r in results:
        speedup = baseline_ms / r["mean_ms"]
        print(f"{r['name']:<35} {r['mean_ms']:.2f} ms    {r['fps']:.1f}      {speedup:.2f}x")

    print("-" * 65)

    # ========================================================================
    # Visualization
    # ========================================================================
    print("\nGenerating comparison plot...")

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image", fontsize=10)
    axes[0].axis("off")

    # FP16 baseline depth
    d_base = ref_depth[0, 0] if ref_depth.ndim == 4 else ref_depth[0]
    d_base_n = (d_base - d_base.min()) / (d_base.max() - d_base.min() + 1e-8)
    axes[1].imshow(d_base_n, cmap="magma")
    axes[1].set_title(f"FP16 Baseline\n{results[0]['mean_ms']:.2f}ms | {results[0]['fps']:.0f}FPS",
                      fontsize=9)
    axes[1].axis("off")

    # Optimized depth
    d_opt = opt_depth[0, 0] if opt_depth.ndim == 4 else opt_depth[0]
    d_opt_n = (d_opt - d_opt.min()) / (d_opt.max() - d_opt.min() + 1e-8)
    axes[2].imshow(d_opt_n, cmap="magma")
    axes[2].set_title(f"{method_name}\n{results[1]['mean_ms']:.2f}ms | {results[1]['fps']:.0f}FPS",
                      fontsize=9)
    axes[2].axis("off")

    # Difference map
    diff = np.abs(d_base_n - d_opt_n)
    im = axes[3].imshow(diff, cmap="hot")
    axes[3].set_title(f"Absolute Difference\nMAE={errors['mae']:.4f}", fontsize=9)
    axes[3].axis("off")
    plt.colorbar(im, ax=axes[3], fraction=0.046)

    speedup = baseline_ms / results[1]["mean_ms"]
    plt.suptitle(
        f"Depth Anything V2 ({args.encoder}): FP16 vs {method_name} ({speedup:.2f}x speedup)",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()

    output_path = "compiled_int8_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
