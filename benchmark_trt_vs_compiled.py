#!/usr/bin/env python3
"""
Fair Benchmark: TensorRT Engine vs PyTorch Compiled + Flash Attention

Designed for production deployment comparison with consistent methodology:
- Same wall-clock timing for both backends (what your app actually experiences)
- Percentile latencies (p50, p95, p99) for SLA planning
- GPU memory tracking
- Output quality comparison
- Optional end-to-end pipeline timing (preprocessing + inference)

Key fairness principles:
- Both backends timed with time.perf_counter() + torch.cuda.synchronize()
  (or pycuda stream.synchronize() for TRT) to measure real wall-clock latency
- Same input, same warmup strategy, same iteration count
- torch.compile gets extra warmup to finish JIT compilation before measurement
- TensorRT H2D/D2H is part of its measured latency (as it would be in production)
- PyTorch tensor is pre-loaded on GPU (as it would be in a real serving pipeline)

Usage:
    python benchmark_trt_vs_compiled.py \
        --checkpoint checkpoints/depth_anything_v2_vits.pth \
        --trt-engine engines/depth_anything_v2_vits_fp16.engine \
        --image assets/example.jpg

    # With INT8 quantization on PyTorch side:
    python benchmark_trt_vs_compiled.py \
        --checkpoint checkpoints/depth_anything_v2_vits.pth \
        --trt-engine engines/depth_anything_v2_vits_fp16.engine \
        --image assets/example.jpg \
        --quant-mode int8wo

    # Include preprocessing in timing (end-to-end):
    python benchmark_trt_vs_compiled.py \
        --checkpoint checkpoints/depth_anything_v2_vits.pth \
        --trt-engine engines/depth_anything_v2_vits_fp16.engine \
        --image assets/example.jpg \
        --include-preprocess
"""

import argparse
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

# ============================================================================
# Dependency checks
# ============================================================================

FLASH_ATTN_AVAILABLE = False
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    pass

XFORMERS_AVAILABLE = False
try:
    import xformers.ops as xops
    XFORMERS_AVAILABLE = True
except ImportError:
    pass

TORCHAO_AVAILABLE = False
TORCHAO_DYNAMIC_AVAILABLE = False
try:
    from torchao.quantization import quantize_, int8_weight_only
    TORCHAO_AVAILABLE = True
    try:
        from torchao.quantization import int8_dynamic_activation_int8_weight
        TORCHAO_DYNAMIC_AVAILABLE = True
    except ImportError:
        pass
except ImportError:
    pass

TENSORRT_AVAILABLE = False
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
    TENSORRT_AVAILABLE = True
except ImportError:
    pass

from depth_anything_v2.dpt import DepthAnythingV2

# ============================================================================
# Configuration
# ============================================================================

MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ============================================================================
# Preprocessing
# ============================================================================

def preprocess_numpy(bgr: np.ndarray, input_size: int) -> np.ndarray:
    """Preprocess BGR image to NCHW float16 numpy array (for TensorRT)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    target = (input_size // 14) * 14
    rgb = cv2.resize(rgb, (target, target), interpolation=cv2.INTER_CUBIC)
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    nchw = np.expand_dims(rgb.transpose(2, 0, 1), axis=0)
    return np.ascontiguousarray(nchw)


def preprocess_torch(bgr: np.ndarray, input_size: int, device: str) -> torch.Tensor:
    """Preprocess BGR image to NCHW float16 torch tensor on GPU (for PyTorch)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    target = (input_size // 14) * 14
    rgb = cv2.resize(rgb, (target, target), interpolation=cv2.INTER_CUBIC)
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
    return tensor.to(device).half()


# ============================================================================
# TensorRT inference wrapper
# ============================================================================

class TensorRTModel:
    """TensorRT inference using TRT 10+ async API with pycuda."""

    def __init__(self, engine_path: str):
        if not TENSORRT_AVAILABLE:
            raise RuntimeError("TensorRT or pycuda not installed")

        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to load engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # Discover I/O tensors
        self.in_name = self.out_name = None
        self.in_dtype = self.out_dtype = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.in_name, self.in_dtype = name, dtype
            else:
                self.out_name, self.out_dtype = name, dtype

        self._allocated_shape = None
        self.out_shape = None
        self.host_in = self.dev_in = None
        self.host_out = self.dev_out = None

    def _ensure_buffers(self, input_shape):
        if self._allocated_shape == input_shape:
            return
        self.context.set_input_shape(self.in_name, input_shape)
        in_shape = tuple(self.context.get_tensor_shape(self.in_name))
        self.out_shape = tuple(self.context.get_tensor_shape(self.out_name))

        self.host_in = cuda.pagelocked_empty(int(np.prod(in_shape)), dtype=self.in_dtype)
        self.dev_in = cuda.mem_alloc(self.host_in.nbytes)
        self.host_out = cuda.pagelocked_empty(int(np.prod(self.out_shape)), dtype=self.out_dtype)
        self.dev_out = cuda.mem_alloc(self.host_out.nbytes)

        self.context.set_tensor_address(self.in_name, int(self.dev_in))
        self.context.set_tensor_address(self.out_name, int(self.dev_out))
        self._allocated_shape = input_shape

    def infer(self, input_np: np.ndarray) -> np.ndarray:
        """Run inference. Input: NCHW numpy. Output: numpy with engine output shape."""
        self._ensure_buffers(tuple(input_np.shape))
        inp = np.ascontiguousarray(input_np.astype(self.in_dtype))
        np.copyto(self.host_in, inp.ravel())

        cuda.memcpy_htod_async(self.dev_in, self.host_in, self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.host_out, self.dev_out, self.stream)
        self.stream.synchronize()

        return self.host_out.reshape(self.out_shape).copy()

    def infer_sync_only(self):
        """Run inference without re-copying input (for sustained throughput test).
        Assumes input was already copied in a prior infer() call."""
        cuda.memcpy_htod_async(self.dev_in, self.host_in, self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.host_out, self.dev_out, self.stream)
        self.stream.synchronize()


# ============================================================================
# Flash Attention monkey-patch
# ============================================================================

def patch_flash_attention(model):
    """Replace DINOv2 attention with Flash Attention / xFormers."""
    if not FLASH_ATTN_AVAILABLE and not XFORMERS_AVAILABLE:
        print("  [WARN] Neither flash-attn nor xFormers available, skipping patch")
        return model

    from depth_anything_v2.dinov2_layers.attention import Attention

    def flash_forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if FLASH_ATTN_AVAILABLE:
            out = flash_attn_func(q, k, v, causal=False)
        else:
            out = xops.memory_efficient_attention(q, k, v)

        out = out.reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    count = 0
    for _, module in model.named_modules():
        if isinstance(module, Attention):
            module.forward = lambda x, m=module: flash_forward(m, x)
            count += 1

    backend = "flash-attn" if FLASH_ATTN_AVAILABLE else "xFormers"
    print(f"  Patched {count} attention layers with {backend}")
    return model


# ============================================================================
# Latency statistics
# ============================================================================

def compute_latency_stats(latencies_ms: list) -> dict:
    """Compute production-relevant latency statistics."""
    arr = np.array(latencies_ms)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "fps": float(1000.0 / np.mean(arr)),
    }


# ============================================================================
# Benchmark runners
# ============================================================================

def benchmark_trt(trt_model: TensorRTModel, input_np: np.ndarray,
                  warmup: int, iterations: int) -> dict:
    """Benchmark TensorRT with wall-clock timing per iteration.

    Each iteration includes H2D copy, execution, D2H copy, and stream sync
    -- identical to what a production serving loop does.
    """
    # Warmup
    for _ in range(warmup):
        trt_model.infer(input_np)

    # Timed runs -- per-iteration wall-clock
    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        trt_model.infer_sync_only()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    output = trt_model.infer(input_np)
    return {"latencies": latencies, "output": output}


def benchmark_pytorch(model, input_tensor: torch.Tensor,
                      warmup: int, iterations: int) -> dict:
    """Benchmark PyTorch with wall-clock timing per iteration.

    Uses torch.cuda.synchronize() to ensure GPU work is complete before
    stopping the timer -- same wall-clock semantics as TensorRT benchmark.
    """
    # Warmup (extra for torch.compile JIT)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
        torch.cuda.synchronize()

    # Timed runs -- per-iteration wall-clock
    latencies = []
    with torch.no_grad():
        for _ in range(iterations):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(input_tensor)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    with torch.no_grad():
        output = model(input_tensor).cpu().numpy()

    return {"latencies": latencies, "output": output}


def benchmark_e2e_trt(trt_model: TensorRTModel, bgr: np.ndarray,
                      input_size: int, warmup: int, iterations: int) -> list:
    """End-to-end: preprocess (CPU) + TRT inference per iteration."""
    for _ in range(warmup):
        inp = preprocess_numpy(bgr, input_size)
        trt_model.infer(inp)

    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        inp = preprocess_numpy(bgr, input_size)
        trt_model.infer(inp)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
    return latencies


def benchmark_e2e_pytorch(model, bgr: np.ndarray, input_size: int,
                          device: str, warmup: int, iterations: int) -> list:
    """End-to-end: preprocess (CPU) + H2D + PyTorch inference per iteration."""
    with torch.no_grad():
        for _ in range(warmup):
            t = preprocess_torch(bgr, input_size, device)
            _ = model(t)
        torch.cuda.synchronize()

    latencies = []
    with torch.no_grad():
        for _ in range(iterations):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            t = preprocess_torch(bgr, input_size, device)
            _ = model(t)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)
    return latencies


# ============================================================================
# Output quality comparison
# ============================================================================

def normalize_depth(d: np.ndarray) -> np.ndarray:
    """Squeeze to 2D and normalize to [0,1]."""
    if d.ndim == 4:
        d = d[0, 0] if d.shape[1] == 1 else d[0]
    elif d.ndim == 3:
        d = d[0]
    d = d.astype(np.float32)
    return (d - d.min()) / (d.max() - d.min() + 1e-8)


def compare_outputs(depth_a: np.ndarray, depth_b: np.ndarray) -> dict:
    """Compare two depth maps after normalization."""
    a = normalize_depth(depth_a)
    b = normalize_depth(depth_b)
    # Resize to match if needed
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
    diff = np.abs(a - b)
    return {
        "mae": float(diff.mean()),
        "rmse": float(np.sqrt((diff ** 2).mean())),
        "max_error": float(diff.max()),
    }


# ============================================================================
# GPU memory measurement
# ============================================================================

def get_gpu_memory_mb() -> float:
    return torch.cuda.memory_allocated() / (1024 ** 2)


def get_gpu_memory_reserved_mb() -> float:
    return torch.cuda.memory_reserved() / (1024 ** 2)


# ============================================================================
# Pretty printing
# ============================================================================

def print_header(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_latency_table(results: dict):
    """Print a comparison table of latency results."""
    names = list(results.keys())
    print(f"\n{'Metric':<12}", end="")
    for name in names:
        print(f" {name:>22}", end="")
    print()
    print("-" * (12 + 23 * len(names)))

    for metric in ["mean", "median", "p95", "p99", "min", "max", "std"]:
        label = f"{metric} (ms)" if metric != "std" else "std (ms)"
        print(f"{label:<12}", end="")
        for name in names:
            val = results[name]["stats"][metric]
            print(f" {val:>22.2f}", end="")
        print()

    print(f"{'FPS':<12}", end="")
    for name in names:
        print(f" {results[name]['stats']['fps']:>22.1f}", end="")
    print()

    if len(names) == 2:
        a, b = names
        speedup = results[a]["stats"]["mean"] / results[b]["stats"]["mean"]
        faster = b if speedup > 1 else a
        ratio = max(speedup, 1.0 / speedup)
        print(f"\n  {faster} is {ratio:.2f}x faster (by mean latency)")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fair Benchmark: TensorRT vs PyTorch Compiled + Flash Attention",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to Depth Anything V2 .pth checkpoint")
    parser.add_argument("--trt-engine", type=str, required=True,
                        help="Path to TensorRT .engine file")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--encoder", type=str, default="vits",
                        choices=["vits", "vitb", "vitl"])
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--warmup", type=int, default=50,
                        help="Warmup iterations (default: 50, enough for torch.compile)")
    parser.add_argument("--iterations", type=int, default=300,
                        help="Timed iterations (default: 300)")
    parser.add_argument("--quant-mode", type=str, default="none",
                        choices=["int8wo", "int8dq", "none"],
                        help="INT8 quantization for PyTorch side (default: none)")
    parser.add_argument("--no-flash-attn", action="store_true",
                        help="Disable Flash Attention on PyTorch side")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile on PyTorch side")
    parser.add_argument("--include-preprocess", action="store_true",
                        help="Include CPU preprocessing in timing (end-to-end)")
    parser.add_argument("--output", type=str, default="benchmark_trt_vs_compiled.png",
                        help="Output plot path")
    args = parser.parse_args()

    device = "cuda"

    # Print environment info
    print_header("ENVIRONMENT")
    print(f"  PyTorch:        {torch.__version__}")
    print(f"  CUDA:           {torch.version.cuda}")
    print(f"  GPU:            {torch.cuda.get_device_name(0)}")
    if TENSORRT_AVAILABLE:
        print(f"  TensorRT:       {trt.__version__}")
    print(f"  Flash Attention: {'yes' if FLASH_ATTN_AVAILABLE else 'no'}")
    print(f"  xFormers:       {'yes' if XFORMERS_AVAILABLE else 'no'}")
    print(f"  TorchAO:        {'yes' if TORCHAO_AVAILABLE else 'no'}")

    # Validate inputs
    if not os.path.isfile(args.trt_engine):
        raise FileNotFoundError(f"TensorRT engine not found: {args.trt_engine}")
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # Load image
    bgr = cv2.imread(args.image)
    if bgr is None:
        raise ValueError(f"Cannot load image: {args.image}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Prepare inputs
    input_np = preprocess_numpy(bgr, args.input_size)
    input_tensor = preprocess_torch(bgr, args.input_size, device)
    input_shape_str = f"{input_tensor.shape[3]}x{input_tensor.shape[2]}"

    # Build PyTorch method label
    pt_parts = []
    if args.quant_mode != "none":
        pt_parts.append(f"INT8({args.quant_mode})")
    if not args.no_flash_attn and (FLASH_ATTN_AVAILABLE or XFORMERS_AVAILABLE):
        pt_parts.append("FlashAttn")
    if not args.no_compile:
        pt_parts.append("compiled")
    pt_label = "PyTorch " + "+".join(pt_parts) if pt_parts else "PyTorch Eager FP16"

    trt_label = f"TensorRT ({os.path.basename(args.trt_engine)})"

    print_header("BENCHMARK CONFIGURATION")
    print(f"  Image:          {args.image} ({bgr.shape[1]}x{bgr.shape[0]})")
    print(f"  Inference size: {input_shape_str}")
    print(f"  Encoder:        {args.encoder}")
    print(f"  Warmup:         {args.warmup} iterations")
    print(f"  Timed runs:     {args.iterations} iterations")
    print(f"  PyTorch mode:   {pt_label}")
    print(f"  TRT engine:     {args.trt_engine}")
    print(f"  Preprocess:     {'included in timing' if args.include_preprocess else 'excluded (inference only)'}")

    results = {}

    # ==================================================================
    # 1. TensorRT
    # ==================================================================
    print_header(f"BENCHMARKING: {trt_label}")

    torch.cuda.reset_peak_memory_stats()
    mem_before_trt = get_gpu_memory_mb()

    trt_model = TensorRTModel(args.trt_engine)
    trt_input_dtype = trt_model.in_dtype
    trt_input = np.ascontiguousarray(input_np.astype(trt_input_dtype))

    print(f"  Engine input dtype: {trt_input_dtype}")
    print(f"  Running warmup ({args.warmup} iters)...")

    trt_bench = benchmark_trt(trt_model, trt_input, args.warmup, args.iterations)

    # TRT memory is managed by pycuda, not torch -- report what we can
    mem_after_trt = get_gpu_memory_mb()

    trt_stats = compute_latency_stats(trt_bench["latencies"])
    results[trt_label] = {
        "stats": trt_stats,
        "output": trt_bench["output"],
        "gpu_mem_delta_mb": mem_after_trt - mem_before_trt,
    }

    print(f"  Mean:   {trt_stats['mean']:.2f} ms")
    print(f"  Median: {trt_stats['median']:.2f} ms")
    print(f"  p95:    {trt_stats['p95']:.2f} ms")
    print(f"  p99:    {trt_stats['p99']:.2f} ms")
    print(f"  FPS:    {trt_stats['fps']:.1f}")

    # ==================================================================
    # 2. PyTorch Compiled + Flash Attention
    # ==================================================================
    print_header(f"BENCHMARKING: {pt_label}")

    # Clear GPU memory from TRT
    del trt_model
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mem_before_pt = get_gpu_memory_mb()

    # Step A: Load model (FP32 on CPU)
    print("  Loading model...")
    model = DepthAnythingV2(**MODEL_CONFIGS[args.encoder])
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    # Step B: INT8 quantization (on CPU, FP32)
    if args.quant_mode != "none":
        if not TORCHAO_AVAILABLE:
            raise RuntimeError("torchao not installed (pip install torchao)")
        if args.quant_mode == "int8dq":
            if not TORCHAO_DYNAMIC_AVAILABLE:
                raise RuntimeError("int8_dynamic_activation_int8_weight not available, update torchao")
            print("  Applying INT8 dynamic quantization...")
            quantize_(model, int8_dynamic_activation_int8_weight())
        else:
            print("  Applying INT8 weight-only quantization...")
            quantize_(model, int8_weight_only())

    # Step C: Move to GPU + FP16
    model = model.to(device).half()

    # Step D: Flash Attention
    if not args.no_flash_attn:
        model = patch_flash_attention(model)

    # Step E: torch.compile
    if not args.no_compile:
        print("  Compiling with torch.compile (max-autotune)...")
        model = torch.compile(model, mode="max-autotune", fullgraph=False)

    print(f"  Running warmup ({args.warmup} iters, includes compile JIT)...")
    pt_bench = benchmark_pytorch(model, input_tensor, args.warmup, args.iterations)

    mem_after_pt = get_gpu_memory_mb()
    peak_pt = torch.cuda.max_memory_allocated() / (1024 ** 2)

    pt_stats = compute_latency_stats(pt_bench["latencies"])
    results[pt_label] = {
        "stats": pt_stats,
        "output": pt_bench["output"],
        "gpu_mem_delta_mb": mem_after_pt - mem_before_pt,
        "gpu_peak_mb": peak_pt,
    }

    print(f"  Mean:   {pt_stats['mean']:.2f} ms")
    print(f"  Median: {pt_stats['median']:.2f} ms")
    print(f"  p95:    {pt_stats['p95']:.2f} ms")
    print(f"  p99:    {pt_stats['p99']:.2f} ms")
    print(f"  FPS:    {pt_stats['fps']:.1f}")
    print(f"  GPU peak memory: {peak_pt:.1f} MB")

    # ==================================================================
    # 3. End-to-end (optional)
    # ==================================================================
    e2e_results = {}
    if args.include_preprocess:
        print_header("END-TO-END BENCHMARK (preprocess + inference)")

        # Reload TRT model
        trt_model = TensorRTModel(args.trt_engine)
        print(f"  TensorRT E2E ({args.iterations} iters)...")
        e2e_trt = benchmark_e2e_trt(
            trt_model, bgr, args.input_size, args.warmup, args.iterations
        )
        e2e_results[f"{trt_label} (E2E)"] = compute_latency_stats(e2e_trt)
        del trt_model

        print(f"  PyTorch E2E ({args.iterations} iters)...")
        e2e_pt = benchmark_e2e_pytorch(
            model, bgr, args.input_size, device, args.warmup, args.iterations
        )
        e2e_results[f"{pt_label} (E2E)"] = compute_latency_stats(e2e_pt)

    # ==================================================================
    # Output quality comparison
    # ==================================================================
    print_header("OUTPUT QUALITY COMPARISON")

    trt_out = results[trt_label]["output"]
    pt_out = results[pt_label]["output"]
    quality = compare_outputs(trt_out, pt_out)

    print(f"  MAE:       {quality['mae']:.6f}")
    print(f"  RMSE:      {quality['rmse']:.6f}")
    print(f"  Max Error: {quality['max_error']:.6f}")

    if quality['mae'] < 0.001:
        grade = "EXCELLENT -- outputs are nearly identical"
    elif quality['mae'] < 0.01:
        grade = "GOOD -- minor differences, acceptable for most applications"
    elif quality['mae'] < 0.05:
        grade = "ACCEPTABLE -- noticeable differences in edge cases"
    else:
        grade = "POOR -- significant divergence, check quantization settings"
    print(f"  Quality:   {grade}")

    # ==================================================================
    # Summary table
    # ==================================================================
    print_header("INFERENCE-ONLY LATENCY COMPARISON")
    print_latency_table(results)

    if e2e_results:
        print_header("END-TO-END LATENCY COMPARISON (preprocess + inference)")
        e2e_table = {}
        for name, stats in e2e_results.items():
            e2e_table[name] = {"stats": stats}
        print_latency_table(e2e_table)

    # ==================================================================
    # Production recommendation
    # ==================================================================
    print_header("PRODUCTION NOTES")
    trt_mean = results[trt_label]["stats"]["mean"]
    pt_mean = results[pt_label]["stats"]["mean"]
    trt_p99 = results[trt_label]["stats"]["p99"]
    pt_p99 = results[pt_label]["stats"]["p99"]
    trt_std = results[trt_label]["stats"]["std"]
    pt_std = results[pt_label]["stats"]["std"]

    print(f"  Latency consistency (lower std = more predictable):")
    print(f"    {trt_label}: std = {trt_std:.2f} ms")
    print(f"    {pt_label}: std = {pt_std:.2f} ms")

    print(f"\n  p99 tail latency (matters for SLAs):")
    print(f"    {trt_label}: p99 = {trt_p99:.2f} ms")
    print(f"    {pt_label}: p99 = {pt_p99:.2f} ms")

    if trt_mean < pt_mean:
        ratio = pt_mean / trt_mean
        print(f"\n  TensorRT is {ratio:.2f}x faster on average.")
    else:
        ratio = trt_mean / pt_mean
        print(f"\n  PyTorch compiled is {ratio:.2f}x faster on average.")

    print(f"  Output quality (MAE): {quality['mae']:.6f}")

    # ==================================================================
    # Visualization
    # ==================================================================
    print_header("GENERATING PLOTS")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1: Depth maps
    target = (args.input_size // 14) * 14
    rgb_resized = cv2.resize(rgb, (target, target))
    axes[0, 0].imshow(rgb_resized)
    axes[0, 0].set_title("Input Image", fontsize=11)
    axes[0, 0].axis("off")

    d_trt = normalize_depth(trt_out)
    axes[0, 1].imshow(d_trt, cmap="magma")
    axes[0, 1].set_title(
        f"TensorRT\n{trt_stats['mean']:.2f}ms mean | {trt_stats['fps']:.0f} FPS",
        fontsize=10,
    )
    axes[0, 1].axis("off")

    d_pt = normalize_depth(pt_out)
    axes[0, 2].imshow(d_pt, cmap="magma")
    axes[0, 2].set_title(
        f"{pt_label}\n{pt_stats['mean']:.2f}ms mean | {pt_stats['fps']:.0f} FPS",
        fontsize=10,
    )
    axes[0, 2].axis("off")

    # Row 2: Difference map + latency distribution
    diff_map = np.abs(d_trt - cv2.resize(d_pt, (d_trt.shape[1], d_trt.shape[0])))
    im = axes[1, 0].imshow(diff_map, cmap="hot")
    axes[1, 0].set_title(f"Absolute Difference\nMAE={quality['mae']:.4f}", fontsize=10)
    axes[1, 0].axis("off")
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

    # Latency distributions
    ax_hist = axes[1, 1]
    trt_lats = trt_bench["latencies"]
    pt_lats = pt_bench["latencies"]
    ax_hist.hist(trt_lats, bins=40, alpha=0.6, label=f"TensorRT (mean={trt_stats['mean']:.2f}ms)", color="tab:blue")
    ax_hist.hist(pt_lats, bins=40, alpha=0.6, label=f"PyTorch (mean={pt_stats['mean']:.2f}ms)", color="tab:orange")
    ax_hist.set_xlabel("Latency (ms)")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Latency Distribution", fontsize=10)
    ax_hist.legend(fontsize=8)

    # Latency over time (stability)
    ax_time = axes[1, 2]
    ax_time.plot(trt_lats, alpha=0.7, linewidth=0.8, label="TensorRT", color="tab:blue")
    ax_time.plot(pt_lats, alpha=0.7, linewidth=0.8, label="PyTorch", color="tab:orange")
    ax_time.set_xlabel("Iteration")
    ax_time.set_ylabel("Latency (ms)")
    ax_time.set_title("Latency Stability Over Time", fontsize=10)
    ax_time.legend(fontsize=8)

    speedup = max(trt_mean, pt_mean) / min(trt_mean, pt_mean)
    faster = "TensorRT" if trt_mean < pt_mean else "PyTorch"
    plt.suptitle(
        f"Depth Anything V2 ({args.encoder}) -- {faster} {speedup:.2f}x faster  |  Input: {input_shape_str}",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"  Saved to: {args.output}")


if __name__ == "__main__":
    main()
