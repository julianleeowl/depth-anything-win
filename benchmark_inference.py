#!/usr/bin/env python3
"""
Benchmark: torch.compile + Flash Attention + INT8 vs TensorRT FP16 vs Eager

This script compares different inference backends for Depth Anything V2:
1. Eager FP16 (baseline)
2. torch.compile FP16
3. torch.compile + Flash Attention (if available)
4. torch.compile + Flash Attention + INT8 (if torchao available)
5. TensorRT FP16 (if engine exists)

Usage:
    python benchmark_inference.py \
        --checkpoint checkpoints/depth_anything_v2_vits.pth \
        --encoder vits \
        --warmup 10 \
        --iterations 100

    # With INT8 dynamic quantization instead of weight-only:
    python benchmark_inference.py \
        --checkpoint checkpoints/depth_anything_v2_vits.pth \
        --image test.jpg \
        --quant-mode int8dq
"""

import argparse
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

# Check for Flash Attention availability
FLASH_ATTN_AVAILABLE = False
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
    print("Flash Attention: Available")
except ImportError:
    print("Flash Attention: Not available (pip install flash-attn)")

# Check for xFormers availability
XFORMERS_AVAILABLE = False
try:
    import xformers.ops as xops
    XFORMERS_AVAILABLE = True
    print("xFormers: Available")
except ImportError:
    print("xFormers: Not available (pip install xformers)")

# Check for TorchAO availability
TORCHAO_AVAILABLE = False
TORCHAO_DYNAMIC_AVAILABLE = False
try:
    from torchao.quantization import quantize_, int8_weight_only
    TORCHAO_AVAILABLE = True
    print("TorchAO: Available")
    try:
        from torchao.quantization import int8_dynamic_activation_int8_weight
        TORCHAO_DYNAMIC_AVAILABLE = True
    except ImportError:
        pass
except ImportError:
    print("TorchAO: Not available (pip install torchao)")

# Check for TensorRT availability
TENSORRT_AVAILABLE = False
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
    print("TensorRT: Available")
except ImportError:
    print("TensorRT: Not available")

from depth_anything_v2.dpt import DepthAnythingV2


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}


# =============================================================================
# IMAGE LOADING
# =============================================================================

def load_image(image_path: str, input_size: int = 518):
    """Load and preprocess image for Depth Anything V2.
    Returns (tensor, original_image_for_display).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    # Convert BGR -> RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to fixed size (must be multiple of 14 for ViT)
    target_size = (input_size // 14) * 14
    img_resized = cv2.resize(img_rgb, (target_size, target_size))

    # Keep original for display
    original_display = img_resized.copy()

    # Normalize to [0, 1]
    img_norm = img_resized / 255.0

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_norm = (img_norm - mean) / std

    # HWC -> CHW, add batch dim
    tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0)

    return tensor, original_display


# =============================================================================
# TIMING UTILITIES
# =============================================================================

def benchmark_model(model, input_tensor, warmup=10, iterations=100, name="Model"):
    """
    Benchmark a model with warmup and multiple iterations.
    Returns mean and std latency in milliseconds.
    """
    print(f"\nBenchmarking: {name}")
    print(f"  Input shape: {input_tensor.shape}")
    print(f"  Inference input size: {input_tensor.shape[3]}x{input_tensor.shape[2]} (WxH)")

    # Check if model is TensorRT (returns numpy) or PyTorch (returns tensor)
    is_trt = isinstance(model, TensorRTInference)

    # Warmup
    print(f"  Warming up ({warmup} iterations)...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
    if not is_trt:
        torch.cuda.synchronize()

    # Benchmark - use time.perf_counter for TensorRT (same as infer_engine.py)
    print(f"  Running benchmark ({iterations} iterations)...")
    latencies = []

    if is_trt:
        # TensorRT: batch timing with perf_counter (identical to infer_engine.py)
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(input_tensor)
        t1 = time.perf_counter()
        avg_latency = (t1 - t0) * 1000.0 / iterations
        latencies = [avg_latency] * iterations  # uniform for stats
    else:
        # PyTorch: per-iteration CUDA events (more accurate for GPU timing)
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
    mean_latency = latencies.mean().item()
    std_latency = latencies.std().item()
    min_latency = latencies.min().item()
    max_latency = latencies.max().item()

    print(f"  Results:")
    print(f"    Mean: {mean_latency:.2f} ms")
    print(f"    Std:  {std_latency:.2f} ms")
    print(f"    Min:  {min_latency:.2f} ms")
    print(f"    Max:  {max_latency:.2f} ms")
    print(f"    FPS:  {1000/mean_latency:.1f}")

    # Get output for visualization
    with torch.no_grad():
        output = model(input_tensor)
        if isinstance(output, torch.Tensor):
            depth_output = output.cpu().numpy()
        elif isinstance(output, np.ndarray):
            depth_output = output
        else:
            depth_output = np.array(output)

    return {
        "name": name,
        "mean_ms": mean_latency,
        "std_ms": std_latency,
        "min_ms": min_latency,
        "max_ms": max_latency,
        "fps": 1000 / mean_latency,
        "depth": depth_output,
    }


# =============================================================================
# FLASH ATTENTION MONKEY PATCH
# =============================================================================

def patch_attention_with_flash(model):
    """
    Replace standard attention with Flash Attention in the model.
    This patches the DinoV2 attention modules.
    """
    if not FLASH_ATTN_AVAILABLE and not XFORMERS_AVAILABLE:
        print("  Warning: Neither Flash Attention nor xFormers available, skipping patch")
        return model

    from depth_anything_v2.dinov2_layers.attention import Attention

    # Store original forward
    original_forward = Attention.forward

    def flash_attention_forward(self, x):
        B, N, C = x.shape

        # QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 1, 3, 4)  # (3, B, N, num_heads, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, N, num_heads, head_dim)

        if FLASH_ATTN_AVAILABLE:
            # Flash Attention expects (B, N, num_heads, head_dim)
            # flash_attn_func handles the attention computation efficiently
            out = flash_attn_func(q, k, v, causal=False)
        elif XFORMERS_AVAILABLE:
            # xFormers memory efficient attention
            # Expects (B, N, num_heads, head_dim)
            out = xops.memory_efficient_attention(q, k, v)

        # Reshape back
        out = out.reshape(B, N, C)

        # Output projection
        out = self.proj(out)
        out = self.proj_drop(out)

        return out

    # Patch all attention modules
    patched_count = 0
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            module.forward = lambda x, m=module: flash_attention_forward(m, x)
            patched_count += 1

    print(f"  Patched {patched_count} attention modules with Flash Attention")
    return model


# =============================================================================
# TENSORRT INFERENCE (TRT 10+ API, identical to infer_engine.py)
# =============================================================================

class TensorRTInference:
    """TensorRT inference wrapper using TRT 10+ API with async execution."""

    def __init__(self, engine_path):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401

        self.trt = trt
        self.cuda = cuda
        self.logger = trt.Logger(trt.Logger.WARNING)

        # Load engine
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to load engine: {engine_path}")

        self.context = self.engine.create_execution_context()

        # Get input/output tensor info
        self.in_name = None
        self.in_dtype = None
        self.out_name = None
        self.out_dtype = None

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.in_name = name
                self.in_dtype = dtype
            else:
                self.out_name = name
                self.out_dtype = dtype

        # Create CUDA stream
        self.stream = cuda.Stream()

        # Buffers will be allocated on first inference
        self.host_in = None
        self.dev_in = None
        self.host_out = None
        self.dev_out = None
        self.current_shape = None

    def _allocate_buffers(self, input_shape):
        """Allocate host/device buffers for given input shape."""
        # Set input shape (for dynamic engines)
        self.context.set_input_shape(self.in_name, tuple(input_shape))

        # Get actual shapes after setting input
        in_shape = tuple(self.context.get_tensor_shape(self.in_name))
        out_shape = tuple(self.context.get_tensor_shape(self.out_name))

        in_vol = int(np.prod(in_shape))
        out_vol = int(np.prod(out_shape))

        # Allocate page-locked host memory and device memory
        self.host_in = self.cuda.pagelocked_empty(in_vol, dtype=self.in_dtype)
        self.dev_in = self.cuda.mem_alloc(self.host_in.nbytes)

        self.host_out = self.cuda.pagelocked_empty(out_vol, dtype=self.out_dtype)
        self.dev_out = self.cuda.mem_alloc(self.host_out.nbytes)

        # Bind tensor addresses
        self.context.set_tensor_address(self.in_name, int(self.dev_in))
        self.context.set_tensor_address(self.out_name, int(self.dev_out))

        self.current_shape = input_shape
        self.out_shape = out_shape

    def __call__(self, input_tensor):
        input_shape = tuple(input_tensor.shape)

        # Reallocate buffers if shape changed
        if self.current_shape != input_shape:
            self._allocate_buffers(input_shape)

        # Prepare input (contiguous, correct dtype)
        input_np = np.ascontiguousarray(input_tensor.cpu().numpy().astype(self.in_dtype))

        # H2D async
        np.copyto(self.host_in, input_np.ravel())
        self.cuda.memcpy_htod_async(self.dev_in, self.host_in, self.stream)

        # Execute async
        ok = self.context.execute_async_v3(stream_handle=self.stream.handle)
        if not ok:
            raise RuntimeError("TensorRT execution failed")

        # D2H async
        self.cuda.memcpy_dtoh_async(self.host_out, self.dev_out, self.stream)

        # Synchronize
        self.stream.synchronize()

        # Reshape and return as numpy (same as infer_engine.py)
        # No extra GPU copy - return numpy array
        return self.host_out.reshape(self.out_shape).copy()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark Inference Methods")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--encoder", type=str, default="vits", choices=["vits", "vitb", "vitl"])
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--trt-engine", type=str, default=None, help="Path to TensorRT engine")
    parser.add_argument("--quant-mode", type=str, default="int8wo",
                        choices=["int8wo", "int8dq"],
                        help="INT8 quantization mode: int8wo (weight-only) or int8dq (dynamic)")
    args = parser.parse_args()

    device = "cuda"

    # Create input tensor from image or random
    original_image = None
    if args.image:
        print(f"Loading image: {args.image}")
        input_tensor, original_image = load_image(args.image, args.input_size)
        input_tensor = input_tensor.to(device).half()
    else:
        print("Using random input tensor")
        input_size = (args.input_size // 14) * 14  # Must be multiple of 14
        input_tensor = torch.randn(args.batch_size, 3, input_size, input_size).to(device).half()

    print("=" * 60)
    print("INFERENCE BENCHMARK")
    print("=" * 60)
    print(f"Encoder: {args.encoder}")
    print(f"Input: {input_tensor.shape}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iterations}")
    print("=" * 60)

    results = []

    # =========================================================================
    # 1. Eager FP16 (Baseline)
    # =========================================================================
    print("\n[1/5] Loading model for Eager FP16...")
    model_eager = DepthAnythingV2(**MODEL_CONFIGS[args.encoder])
    model_eager.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model_eager = model_eager.to(device).half().eval()

    results.append(benchmark_model(
        model_eager, input_tensor,
        args.warmup, args.iterations,
        "Eager FP16"
    ))

    # =========================================================================
    # 2. torch.compile FP16
    # =========================================================================
    print("\n[2/5] Compiling model with torch.compile...")
    try:
        model_compiled = torch.compile(
            model_eager,
            mode="max-autotune",
            fullgraph=False,  # Allow graph breaks for compatibility
        )

        results.append(benchmark_model(
            model_compiled, input_tensor,
            args.warmup, args.iterations,
            "torch.compile FP16"
        ))
    except Exception as e:
        print(f"  torch.compile failed: {e}")

    # =========================================================================
    # 3. torch.compile + Flash Attention
    # =========================================================================
    if FLASH_ATTN_AVAILABLE or XFORMERS_AVAILABLE:
        print("\n[3/5] Loading model with Flash Attention...")
        model_flash = DepthAnythingV2(**MODEL_CONFIGS[args.encoder])
        model_flash.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        model_flash = model_flash.to(device).half().eval()

        # Patch attention
        model_flash = patch_attention_with_flash(model_flash)

        # Compile
        try:
            model_flash_compiled = torch.compile(
                model_flash,
                mode="max-autotune",
                fullgraph=False,
            )

            results.append(benchmark_model(
                model_flash_compiled, input_tensor,
                args.warmup, args.iterations,
                "torch.compile + FlashAttn"
            ))
        except Exception as e:
            print(f"  torch.compile + FlashAttn failed: {e}")
    else:
        print("\n[3/5] Skipping Flash Attention (not available)")

    # =========================================================================
    # 4. torch.compile + Flash Attention + INT8
    # =========================================================================
    if TORCHAO_AVAILABLE and (FLASH_ATTN_AVAILABLE or XFORMERS_AVAILABLE):
        quant_label = "int8wo" if args.quant_mode == "int8wo" else "int8dq"
        print(f"\n[4/5] Loading model with INT8 ({quant_label}) + FlashAttn + compile...")
        try:
            # Load fresh model in FP32 on CPU (torchao quantizes FP32 weights)
            model_int8 = DepthAnythingV2(**MODEL_CONFIGS[args.encoder])
            model_int8.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
            model_int8.eval()

            # Apply TorchAO INT8 quantization on CPU
            if args.quant_mode == "int8dq" and TORCHAO_DYNAMIC_AVAILABLE:
                print("  Applying INT8 dynamic activation + weight quantization...")
                quantize_(model_int8, int8_dynamic_activation_int8_weight())
            else:
                print("  Applying INT8 weight-only quantization...")
                quantize_(model_int8, int8_weight_only())

            # Move to GPU and convert to FP16
            model_int8 = model_int8.to(device).half()

            # Patch flash attention
            model_int8 = patch_attention_with_flash(model_int8)

            # Compile
            model_int8_compiled = torch.compile(
                model_int8,
                mode="max-autotune",
                fullgraph=False,
            )

            results.append(benchmark_model(
                model_int8_compiled, input_tensor,
                args.warmup, args.iterations,
                f"compile+FlashAttn+INT8({quant_label})"
            ))

            # Free memory
            del model_int8, model_int8_compiled
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  INT8 + FlashAttn + compile failed: {e}")
    elif not TORCHAO_AVAILABLE:
        print("\n[4/5] Skipping INT8 (TorchAO not available: pip install torchao)")
    else:
        print("\n[4/5] Skipping INT8+FlashAttn (Flash Attention not available)")

    # =========================================================================
    # 5. TensorRT FP16
    # =========================================================================
    if args.trt_engine and TENSORRT_AVAILABLE:
        print("\n[5/5] Loading TensorRT engine...")
        try:
            trt_model = TensorRTInference(args.trt_engine)

            results.append(benchmark_model(
                trt_model, input_tensor,
                args.warmup, args.iterations,
                "TensorRT FP16"
            ))
        except Exception as e:
            print(f"  TensorRT failed: {e}")
    else:
        print("\n[5/5] Skipping TensorRT (no engine provided or TRT not available)")
        print("  To test TensorRT, first export and build engine:")
        print("    python sensitivity_demo.py --checkpoint ... --image ... --all-layers")
        print("    trtexec --onnx=depth_anything_int8.onnx --saveEngine=model.engine --fp16")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"{'Method':<30} {'Mean (ms)':<12} {'FPS':<10} {'Speedup':<10}")
    print("-" * 60)

    baseline_ms = results[0]["mean_ms"] if results else 1.0

    for r in results:
        speedup = baseline_ms / r["mean_ms"]
        print(f"{r['name']:<30} {r['mean_ms']:<12.2f} {r['fps']:<10.1f} {speedup:<10.2f}x")

    print("-" * 60)

    # Find winner
    if results:
        winner = min(results, key=lambda x: x["mean_ms"])
        print(f"\nFastest: {winner['name']} ({winner['mean_ms']:.2f} ms, {winner['fps']:.1f} FPS)")

    print("\nNotes:")
    print("  - First torch.compile run includes compilation time")
    print("  - Flash Attention requires: pip install flash-attn --no-build-isolation")
    print("  - xFormers alternative: pip install xformers")
    print("  - INT8 quantization requires: pip install torchao")

    # =========================================================================
    # Plot Results
    # =========================================================================
    if results and original_image is not None:
        print("\nGenerating comparison plot...")

        n_plots = len(results) + 1  # +1 for original image
        fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))

        # Plot original image
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image", fontsize=10)
        axes[0].axis("off")

        # Plot each inference result
        for i, r in enumerate(results):
            depth = r["depth"]
            # Handle different output shapes
            if depth.ndim == 4:
                depth = depth[0, 0]
            elif depth.ndim == 3:
                depth = depth[0]

            # Normalize for display
            depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

            axes[i + 1].imshow(depth_norm, cmap="magma")
            axes[i + 1].set_title(f"{r['name']}\n{r['mean_ms']:.2f}ms | {r['fps']:.0f}FPS", fontsize=9)
            axes[i + 1].axis("off")

        plt.suptitle("Depth Estimation: Inference Method Comparison", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig("benchmark_results.png", dpi=150, bbox_inches="tight")
        print("Saved plot to: benchmark_results.png")
        plt.show()


if __name__ == "__main__":
    main()
