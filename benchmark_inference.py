#!/usr/bin/env python3
"""
Benchmark: torch.compile + Flash Attention vs TensorRT FP16 vs Eager

This script compares different inference backends for Depth Anything V2:
1. Eager FP16 (baseline)
2. torch.compile FP16
3. torch.compile + Flash Attention (if available)
4. TensorRT FP16 (if engine exists)

Usage:
    python benchmark_inference.py \
        --checkpoint checkpoints/depth_anything_v2_vits.pth \
        --encoder vits \
        --warmup 10 \
        --iterations 100
"""

import argparse
import time
from contextlib import contextmanager

import torch
import torch.nn as nn

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
# TIMING UTILITIES
# =============================================================================

@contextmanager
def cuda_timer():
    """Context manager for accurate CUDA timing."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start.record()

    yield

    end.record()
    torch.cuda.synchronize()

    # Return time in milliseconds
    elapsed = start.elapsed_time(end)
    yield elapsed


def benchmark_model(model, input_tensor, warmup=10, iterations=100, name="Model"):
    """
    Benchmark a model with warmup and multiple iterations.
    Returns mean and std latency in milliseconds.
    """
    print(f"\nBenchmarking: {name}")
    print(f"  Input shape: {input_tensor.shape}")

    # Warmup
    print(f"  Warming up ({warmup} iterations)...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
    torch.cuda.synchronize()

    # Benchmark
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

    return {
        "name": name,
        "mean_ms": mean_latency,
        "std_ms": std_latency,
        "min_ms": min_latency,
        "max_ms": max_latency,
        "fps": 1000 / mean_latency,
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
# TENSORRT INFERENCE
# =============================================================================

class TensorRTInference:
    """Simple TensorRT inference wrapper."""

    def __init__(self, engine_path):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit

        self.logger = trt.Logger(trt.Logger.WARNING)

        # Load engine
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Allocate buffers
        self.inputs = []
        self.outputs = []
        self.bindings = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.engine.get_tensor_shape(name)
            size = trt.volume(shape)

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append({"host": host_mem, "device": device_mem, "shape": shape})
            else:
                self.outputs.append({"host": host_mem, "device": device_mem, "shape": shape})

    def __call__(self, input_tensor):
        import pycuda.driver as cuda

        # Copy input to device
        input_np = input_tensor.cpu().numpy().ravel()
        self.inputs[0]["host"][:len(input_np)] = input_np
        cuda.memcpy_htod(self.inputs[0]["device"], self.inputs[0]["host"])

        # Run inference
        self.context.execute_v2(self.bindings)

        # Copy output to host
        cuda.memcpy_dtoh(self.outputs[0]["host"], self.outputs[0]["device"])

        # Convert to tensor
        output = torch.from_numpy(self.outputs[0]["host"].reshape(self.outputs[0]["shape"]))
        return output.cuda()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark Inference Methods")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--encoder", type=str, default="vits", choices=["vits", "vitb", "vitl"])
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--trt-engine", type=str, default=None, help="Path to TensorRT engine")
    args = parser.parse_args()

    device = "cuda"

    # Create input tensor
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
    print("\n[1/4] Loading model for Eager FP16...")
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
    print("\n[2/4] Compiling model with torch.compile...")
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
        print("\n[3/4] Loading model with Flash Attention...")
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
        print("\n[3/4] Skipping Flash Attention (not available)")

    # =========================================================================
    # 4. TensorRT FP16
    # =========================================================================
    if args.trt_engine and TENSORRT_AVAILABLE:
        print("\n[4/4] Loading TensorRT engine...")
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
        print("\n[4/4] Skipping TensorRT (no engine provided or TRT not available)")
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


if __name__ == "__main__":
    main()
