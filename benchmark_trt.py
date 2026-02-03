#!/usr/bin/env python3
"""
TensorRT Depth Estimation Benchmark

Benchmarks one or more TensorRT engines for depth estimation:
  1. Depth metric calculation (cross-engine comparison + optional GT evaluation)
  2. Inference speed profiling (production-representative: pre-allocated buffers,
     per-iteration wall-clock timing including H2D/D2H/sync)
  3. GPU utilization monitoring (background pynvml sampling in a separate pass)
  4. Depth prediction visualization (original image + per-engine depth maps)

Outputs 3 separate PNG files + a stdout summary table.
"""

import argparse
import os
import time
from threading import Thread, Condition

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pynvml
import tensorrt as trt

import pycuda.driver as cuda
import pycuda.autoinit

from bench import IMAGENET_MEAN, IMAGENET_STD, preprocess_bgr_to_nchw, normalize_depth, compute_gt_metrics, inverse_depth_to_depth


# =============================================================================
# Preprocessing
# =============================================================================


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


def get_image_paths(directory: str) -> list[str]:
    """Return sorted list of image file paths in a directory."""
    paths = []
    for fname in sorted(os.listdir(directory)):
        if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS:
            paths.append(os.path.join(directory, fname))
    return paths


# =============================================================================
# TensorRT helpers  (adapted from infer_engine.py)
# =============================================================================

def load_engine(engine_path: str, logger: trt.ILogger) -> trt.ICudaEngine:
    """Load a serialized TensorRT engine from disk."""
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(
            f"Failed to deserialize engine '{engine_path}'. "
            "Common causes: TRT/CUDA version mismatch, GPU mismatch, "
            "or corrupted engine file."
        )
    return engine


def get_io_tensors(engine: trt.ICudaEngine):
    """Return (inputs, outputs) lists of (name, numpy_dtype) tuples."""
    inputs, outputs = [], []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        (inputs if mode == trt.TensorIOMode.INPUT else outputs).append(
            (name, dtype))
    if len(inputs) != 1:
        raise ValueError(f"Expected 1 input, found {len(inputs)}: {inputs}")
    if len(outputs) < 1:
        raise ValueError("Expected at least 1 output")
    return inputs, outputs


def _is_dynamic_shape(shape):
    return any(int(d) == -1 for d in shape)


def allocate_io(engine: trt.ICudaEngine, context: trt.IExecutionContext,
                input_shape):
    """Allocate pinned host + device buffers and bind to tensor names.

    Returns ``io_buffers`` tuple that can be reused across iterations.
    """
    inputs, outputs = get_io_tensors(engine)
    in_name, in_dtype = inputs[0]

    cur_in_shape = tuple(context.get_tensor_shape(in_name))
    if cur_in_shape != tuple(input_shape):
        context.set_input_shape(in_name, tuple(input_shape))

    in_shape = tuple(context.get_tensor_shape(in_name))
    if _is_dynamic_shape(in_shape):
        raise RuntimeError(
            f"Input '{in_name}' is still dynamic after set_input_shape: "
            f"{in_shape}."
        )

    stream = cuda.Stream()

    # Input buffers
    in_vol = int(np.prod(in_shape))
    host_in = cuda.pagelocked_empty(in_vol, dtype=in_dtype)
    dev_in = cuda.mem_alloc(host_in.nbytes)

    # Output buffers
    host_outs, dev_outs, out_meta = [], [], []
    for out_name, out_dtype in outputs:
        out_shape = tuple(context.get_tensor_shape(out_name))
        if _is_dynamic_shape(out_shape):
            raise RuntimeError(
                f"Output '{out_name}' is dynamic: {out_shape}."
            )
        out_vol = int(np.prod(out_shape))
        host_out = cuda.pagelocked_empty(out_vol, dtype=out_dtype)
        dev_out = cuda.mem_alloc(host_out.nbytes)
        host_outs.append(host_out)
        dev_outs.append(dev_out)
        out_meta.append((out_name, out_dtype, out_shape))

    # Bind device pointers
    context.set_tensor_address(in_name, int(dev_in))
    for dev_out, (out_name, _, _) in zip(dev_outs, out_meta):
        context.set_tensor_address(out_name, int(dev_out))

    return (in_name, host_in, dev_in, in_shape), \
           (host_outs, dev_outs, out_meta), stream


def infer_one(context: trt.IExecutionContext, inp_nchw: np.ndarray,
              io_buffers):
    """Run one synchronous inference (H2D -> exec -> D2H -> sync).

    ``io_buffers`` must be pre-allocated via :func:`allocate_io`.
    Returns list of ``(output_name, output_array_reshaped)``.
    """
    (in_name, host_in, dev_in, in_shape), \
        (host_outs, dev_outs, out_meta), stream = io_buffers

    # H2D
    np.copyto(host_in, inp_nchw.ravel())
    cuda.memcpy_htod_async(dev_in, host_in, stream)

    # Execute (TRT 10+)
    ok = context.execute_async_v3(stream_handle=stream.handle)
    if not ok:
        raise RuntimeError("TensorRT execution failed (execute_async_v3).")

    # D2H
    for host_out, dev_out in zip(host_outs, dev_outs):
        cuda.memcpy_dtoh_async(host_out, dev_out, stream)
    stream.synchronize()

    # Reshape
    outputs = []
    for host_out, (name, _, shape) in zip(host_outs, out_meta):
        outputs.append((name, host_out.reshape(shape)))
    return outputs


def squeeze_depth_to_hw(out: np.ndarray, name: str) -> np.ndarray:
    """Convert depth tensor to a plain HxW float32 array."""
    depth = out
    if depth.ndim == 4 and depth.shape[1] == 1:   # [N,1,H,W]
        depth = depth[:, 0, :, :]
    if depth.ndim == 3:                            # [N,H,W]
        depth2d = depth[0]
    elif depth.ndim == 2:                          # [H,W]
        depth2d = depth
    else:
        raise ValueError(
            f"Unexpected output shape for {name}: {out.shape}")
    return depth2d.astype(np.float32)


# =============================================================================
# GPU monitoring via pynvml
# =============================================================================

def _get_sm_clock(handle) -> int:
    """Read current SM clock (MHz) from an NVML handle."""
    return pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)


class GPUUtilSampler:
    """Background thread that samples GPU metrics via pynvml.

    Produces time-series arrays of:
      - gpu_util_pct    (SM utilization %)
      - mem_util_pct    (memory-controller utilization %)
      - vram_used_mb    (allocated framebuffer in MB)
      - sm_clock_mhz    (current SM clock speed)
      - mem_clock_mhz   (current memory clock speed)
    """

    def __init__(self, gpu_id: int = 0, interval: float = 0.1):
        self.gpu_id = gpu_id
        self.interval = interval
        self._cond = Condition()
        self._done = False
        self._thread = Thread(target=self._run, daemon=True)
        # Collected samples
        self.timestamps: list[float] = []
        self.gpu_util_pct: list[int] = []
        self.mem_util_pct: list[int] = []
        self.vram_used_mb: list[float] = []
        self.sm_clock_mhz: list[int] = []
        self.mem_clock_mhz: list[int] = []

    # -- context manager -------------------------------------------------
    def start(self):
        self._done = False
        self._thread.start()

    def stop(self):
        with self._cond:
            self._done = True
            self._cond.notify()
        self._thread.join()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

    # -- sampling loop ----------------------------------------------------
    def _run(self):
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
        t0 = time.monotonic()
        with self._cond:
            while not self._done:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    sm_clk = pynvml.nvmlDeviceGetClockInfo(
                        handle, pynvml.NVML_CLOCK_SM)
                    mem_clk = pynvml.nvmlDeviceGetClockInfo(
                        handle, pynvml.NVML_CLOCK_MEM)

                    self.timestamps.append(time.monotonic() - t0)
                    self.gpu_util_pct.append(util.gpu)
                    self.mem_util_pct.append(util.memory)
                    self.vram_used_mb.append(mem_info.used / (1024 ** 2))
                    self.sm_clock_mhz.append(sm_clk)
                    self.mem_clock_mhz.append(mem_clk)
                except pynvml.NVMLError:
                    pass  # skip failed samples silently
                self._cond.wait(self.interval)
        pynvml.nvmlShutdown()


def query_sm_clock(gpu_id: int = 0) -> int:
    """One-shot SM clock query (MHz)."""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    clk = _get_sm_clock(handle)
    pynvml.nvmlShutdown()
    return clk


# =============================================================================
# Latency statistics
# =============================================================================

def compute_latency_stats(latencies_ms: list) -> dict:
    """Production-relevant latency statistics."""
    arr = np.array(latencies_ms)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "fps": float(1000.0 / np.mean(arr)),
    }


# =============================================================================
# Metric calculation
# =============================================================================

def compute_cross_engine_metrics(pred: np.ndarray,
                                  ref: np.ndarray) -> dict:
    """Compare two depth maps after normalization to [0,1].

    Returns MAE, RMSE, max_error.
    """
    a = normalize_depth(pred)
    b = normalize_depth(ref)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]),
                       interpolation=cv2.INTER_LINEAR)
    diff = np.abs(a - b)
    return {
        "mae": float(diff.mean()),
        "rmse": float(np.sqrt((diff ** 2).mean())),
        "max_error": float(diff.max()),
    }


# =============================================================================
# Visualization helpers
# =============================================================================

def plot_depth_comparison(rgb: np.ndarray, engine_results: list,
                          gt_depth_norm: np.ndarray | None,
                          output_path: str):
    """Save ``depth_comparison.png``: [RGB] [GT?] [Engine1] [Engine2] ..."""
    n_engines = len(engine_results)
    n_cols = 1 + (1 if gt_depth_norm is not None else 0) + n_engines
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    idx = 0

    # Original image
    axes[idx].imshow(rgb)
    axes[idx].set_title("Original Input", fontsize=10)
    axes[idx].axis("off")
    idx += 1

    # GT (optional)
    if gt_depth_norm is not None:
        axes[idx].imshow(gt_depth_norm, cmap="inferno")
        axes[idx].set_title("Ground Truth", fontsize=10, fontweight="bold")
        axes[idx].axis("off")
        idx += 1

    # Per-engine depth maps
    for res in engine_results:
        title_lines = [res["name"]]
        stats = res.get("speed_stats")
        if stats:
            title_lines.append(
                f"{stats['mean']:.2f} ms  ({stats['fps']:.1f} FPS)")
        gt_m = res.get("gt_metrics")
        if gt_m and not np.isnan(gt_m.get("abs_rel", float("nan"))):
            n_imgs = res.get("gt_metrics_n_images")
            avg_tag = f" (avg/{n_imgs})" if n_imgs else ""
            title_lines.append(
                f"AbsRel={gt_m['abs_rel']:.4f}  "
                f"\u03b4<1.25={gt_m['d1']:.1f}%{avg_tag}")
        cross_m = res.get("cross_metrics")
        if cross_m:
            title_lines.append(f"MAE vs ref={cross_m['mae']:.5f}")

        axes[idx].imshow(res["depth_norm"], cmap="inferno")
        axes[idx].set_title("\n".join(title_lines), fontsize=8)
        axes[idx].axis("off")
        idx += 1

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_speed_benchmark(engine_results: list, output_path: str):
    """Save ``speed_benchmark.png``: box plot + stats table."""
    names = [r["name"] for r in engine_results]
    latency_lists = [r["latencies"] for r in engine_results]

    fig, (ax_box, ax_table) = plt.subplots(
        2, 1, figsize=(max(8, 3 * len(names)), 8),
        gridspec_kw={"height_ratios": [3, 2]})

    # ---- Box plot ----
    bp = ax_box.boxplot(latency_lists, labels=names, patch_artist=True,
                        showfliers=True, whis=1.5)
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i % len(colors)])
    ax_box.set_ylabel("Latency (ms)")
    ax_box.set_title("Inference Latency Distribution")
    ax_box.grid(axis="y", alpha=0.3)

    # ---- Stats table ----
    ax_table.axis("off")
    stats_list = [r["speed_stats"] for r in engine_results]
    col_labels = names
    row_labels = ["Mean (ms)", "Std (ms)", "Median (ms)",
                  "P50 (ms)", "P95 (ms)", "P99 (ms)",
                  "Min (ms)", "Max (ms)", "FPS"]
    cell_text = []
    for key in ["mean", "std", "median", "p50", "p95", "p99",
                "min", "max", "fps"]:
        row = []
        for s in stats_list:
            val = s[key]
            row.append(f"{val:.2f}" if key != "fps" else f"{val:.1f}")
        cell_text.append(row)

    table = ax_table.table(cellText=cell_text,
                           rowLabels=row_labels,
                           colLabels=col_labels,
                           loc="center",
                           cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_gpu_utilization(engine_results: list, output_path: str):
    """Save ``gpu_utilization.png``: 4 stacked time-series subplots."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    metrics = [
        ("gpu_util_pct", "GPU Compute Utilization (%)", "%"),
        ("mem_util_pct", "GPU Memory Utilization (%)", "%"),
        ("vram_used_mb", "VRAM Usage (MB)", "MB"),
        ("sm_clock_mhz", "SM Clock Speed (MHz)", "MHz"),
    ]
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

    for ax, (key, title, unit) in zip(axes, metrics):
        for i, res in enumerate(engine_results):
            gpu_data = res.get("gpu_samples")
            if gpu_data is None:
                continue
            ts = gpu_data["timestamps"]
            vals = gpu_data[key]
            if len(ts) == 0:
                continue
            ax.plot(ts, vals, label=res["name"],
                    color=colors[i % len(colors)], alpha=0.85)
        ax.set_ylabel(f"{title}")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_metrics_markdown(engine_results: list, output_path: str):
    """Save all benchmark metrics as a markdown file with tables."""
    lines: list[str] = []
    lines.append("# TensorRT Depth Estimation Benchmark Results\n")

    # -- Speed table --
    lines.append("## Speed Benchmark\n")
    lines.append("| Engine | Mean (ms) | Std (ms) | Median (ms) | P50 (ms) "
                 "| P95 (ms) | P99 (ms) | Min (ms) | Max (ms) | FPS |")
    lines.append("|--------|-----------|----------|-------------|----------"
                 "|----------|----------|----------|----------|-----|")
    for r in engine_results:
        s = r["speed_stats"]
        lines.append(
            f"| {r['name']} "
            f"| {s['mean']:.2f} | {s['std']:.2f} | {s['median']:.2f} "
            f"| {s['p50']:.2f} | {s['p95']:.2f} | {s['p99']:.2f} "
            f"| {s['min']:.2f} | {s['max']:.2f} | {s['fps']:.1f} |"
        )
    lines.append("")

    # -- GT metrics table --
    has_gt = any(r.get("gt_metrics") for r in engine_results)
    if has_gt:
        n_imgs = engine_results[0].get("gt_metrics_n_images")
        subtitle = (f" (averaged over {n_imgs} images)" if n_imgs
                    else " (single image)")
        lines.append(f"## Ground Truth Depth Metrics{subtitle}\n")
        lines.append("| Engine | AbsRel | SqRel | RMSE | d1 (%) "
                     "| d2 (%) | d3 (%) |")
        lines.append("|--------|--------|-------|------|--------"
                     "|--------|--------|")
        for r in engine_results:
            m = r.get("gt_metrics")
            if m:
                lines.append(
                    f"| {r['name']} "
                    f"| {m['abs_rel']:.4f} | {m['sq_rel']:.4f} "
                    f"| {m['rmse']:.4f} "
                    f"| {m['d1']:.1f} | {m['d2']:.1f} | {m['d3']:.1f} |"
                )
        lines.append("")

    # -- Cross-engine metrics table --
    has_cross = any(r.get("cross_metrics") for r in engine_results)
    if has_cross:
        lines.append("## Cross-Engine Comparison\n")
        lines.append("| Engine | MAE | RMSE | Max Error |")
        lines.append("|--------|-----|------|-----------|")
        for r in engine_results:
            m = r.get("cross_metrics")
            if m:
                lines.append(
                    f"| {r['name']} "
                    f"| {m['mae']:.6f} | {m['rmse']:.6f} "
                    f"| {m['max_error']:.6f} |"
                )
            else:
                lines.append(f"| {r['name']} | *(reference)* | - | - |")
        lines.append("")

    # -- GPU utilization summary table --
    has_gpu = any(r.get("gpu_samples") for r in engine_results)
    if has_gpu:
        lines.append("## GPU Utilization Summary\n")
        lines.append("| Engine | Avg GPU Util (%) | Avg Mem Util (%) "
                     "| Avg VRAM (MB) | Avg SM Clock (MHz) |")
        lines.append("|--------|------------------|------------------"
                     "|---------------|---------------------|")
        for r in engine_results:
            g = r.get("gpu_samples")
            if g and len(g["timestamps"]) > 0:
                avg_gpu = np.mean(g["gpu_util_pct"])
                avg_mem = np.mean(g["mem_util_pct"])
                avg_vram = np.mean(g["vram_used_mb"])
                avg_sm = np.mean(g["sm_clock_mhz"])
                lines.append(
                    f"| {r['name']} "
                    f"| {avg_gpu:.1f} | {avg_mem:.1f} "
                    f"| {avg_vram:.0f} | {avg_sm:.0f} |"
                )
        lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def print_summary(engine_results: list):
    """Print a combined summary table to stdout."""
    has_gt = any(r.get("gt_metrics") for r in engine_results)
    has_cross = any(r.get("cross_metrics") for r in engine_results)

    print("\n" + "=" * 90)
    print("  BENCHMARK SUMMARY")
    print("=" * 90)

    # Speed table
    print(f"\n{'Engine':<40} {'Mean':>8} {'P50':>8} {'P95':>8} "
          f"{'P99':>8} {'FPS':>8}")
    print("-" * 90)
    for r in engine_results:
        s = r["speed_stats"]
        print(f"{r['name']:<40} {s['mean']:>7.2f}ms {s['p50']:>7.2f}ms "
              f"{s['p95']:>7.2f}ms {s['p99']:>7.2f}ms {s['fps']:>7.1f}")

    # GT metrics
    if has_gt:
        n_imgs = engine_results[0].get("gt_metrics_n_images")
        gt_label = (f"GT Depth Metrics (averaged over {n_imgs} images)"
                    if n_imgs else "GT Depth Metrics (single image)")
        print(f"\n  {gt_label}")
        print(f"  {'Engine':<38} {'AbsRel':>8} {'SqRel':>8} {'RMSE':>8} "
              f"{'d1':>8} {'d2':>8} {'d3':>8}")
        print("-" * 90)
        for r in engine_results:
            m = r.get("gt_metrics")
            if m:
                print(f"  {r['name']:<38} {m['abs_rel']:>8.4f} "
                      f"{m['sq_rel']:>8.4f} {m['rmse']:>8.4f} "
                      f"{m['d1']:>7.1f}% {m['d2']:>7.1f}% "
                      f"{m['d3']:>7.1f}%")

    # Cross-engine metrics
    if has_cross:
        print(f"\n{'Engine vs reference':<40} {'MAE':>10} {'RMSE':>10} "
              f"{'MaxErr':>10}")
        print("-" * 90)
        for r in engine_results:
            m = r.get("cross_metrics")
            if m:
                print(f"{r['name']:<40} {m['mae']:>10.6f} "
                      f"{m['rmse']:>10.6f} {m['max_error']:>10.6f}")
            else:
                print(f"{r['name']:<40} {'(reference)':>10}")

    print("=" * 90)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark TensorRT engines for depth estimation")
    parser.add_argument("--engines", type=str, nargs="+", required=True,
                        help="Paths to TensorRT engine files")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image (used for speed/GPU "
                             "benchmarking and depth visualization)")
    parser.add_argument("--gt", type=str, default=None,
                        help="Path to ground-truth depth (.npy) for the "
                             "visualization image")
    parser.add_argument("--metric-images-dir", type=str, default=None,
                        help="Directory of images for multi-image metric "
                             "evaluation. Each image should have a matching "
                             ".npy GT file (same basename) in --metric-gt-dir")
    parser.add_argument("--metric-gt-dir", type=str, default=None,
                        help="Directory of ground-truth .npy files matching "
                             "images in --metric-images-dir (by basename)")
    parser.add_argument("--input-size", type=int, default=518,
                        help="Model input size (default: 518)")
    parser.add_argument("--warmup", type=int, default=50,
                        help="Warmup iterations (default: 50)")
    parser.add_argument("--iterations", type=int, default=1000,
                        help="Timed iterations (default: 1000)")
    parser.add_argument("--gpu-sample-interval", type=float, default=0.1,
                        help="GPU monitoring sample interval in seconds "
                             "(default: 0.1)")
    parser.add_argument("--gpu-id", type=int, default=0,
                        help="GPU device index for pynvml (default: 0)")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                        help="Directory for output files (default: "
                             "benchmark_results/)")
    args = parser.parse_args()

    # Validate inputs
    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")
    for ep in args.engines:
        if not os.path.isfile(ep):
            raise FileNotFoundError(f"Engine not found: {ep}")
    if args.metric_images_dir and not os.path.isdir(args.metric_images_dir):
        raise FileNotFoundError(
            f"Metric images dir not found: {args.metric_images_dir}")
    if args.metric_gt_dir and not os.path.isdir(args.metric_gt_dir):
        raise FileNotFoundError(
            f"Metric GT dir not found: {args.metric_gt_dir}")
    if args.metric_images_dir and not args.metric_gt_dir:
        raise ValueError("--metric-gt-dir is required when "
                         "--metric-images-dir is provided")
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Stage 1 - Setup
    # ------------------------------------------------------------------
    print("=" * 70)
    print("  Stage 1: Setup")
    print("=" * 70)

    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {args.image}")
    h0, w0 = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    print(f"Image: {args.image}  ({w0}x{h0})")

    # Load GT if provided
    gt_raw = None
    gt_depth_norm = None
    if args.gt:
        if not os.path.isfile(args.gt):
            raise FileNotFoundError(f"GT file not found: {args.gt}")
        gt_raw = np.load(args.gt).astype(np.float32)
        if gt_raw.ndim == 3:
            gt_raw = gt_raw.squeeze()
        gt_raw = cv2.resize(gt_raw, (w0, h0), interpolation=cv2.INTER_CUBIC)
        gt_depth_norm = normalize_depth(gt_raw)
        print(f"Ground truth: {args.gt}  shape={gt_raw.shape}")

    trt_logger = trt.Logger(trt.Logger.WARNING)

    # Load engines and pre-allocate I/O buffers
    engine_data = []  # list of dicts per engine
    for ep in args.engines:
        name = os.path.basename(ep)
        print(f"Loading engine: {name} ...")
        engine = load_engine(ep, trt_logger)
        context = engine.create_execution_context()

        # Detect input dtype
        inputs, _ = get_io_tensors(engine)
        _, in_dtype = inputs[0]
        inp = preprocess_bgr_to_nchw(bgr, args.input_size, dtype=in_dtype)

        # Pre-allocate buffers once
        io_buffers = allocate_io(engine, context, tuple(inp.shape))
        print(f"  input shape={inp.shape}  dtype={in_dtype}")

        engine_data.append({
            "name": name,
            "path": ep,
            "engine": engine,
            "context": context,
            "io_buffers": io_buffers,
            "inp": inp,
        })

    print(f"\nLoaded {len(engine_data)} engine(s).\n")

    # ------------------------------------------------------------------
    # Stage 2 - Speed Benchmark  (inference-only, tight timing loop)
    # ------------------------------------------------------------------
    print("=" * 70)
    print("  Stage 2: Speed Benchmark")
    print("=" * 70)

    engine_results = []  # final results per engine

    for ed in engine_data:
        name = ed["name"]
        context = ed["context"]
        io_buffers = ed["io_buffers"]
        inp = ed["inp"]

        print(f"\n[{name}] Warmup ({args.warmup} iters) ...", end=" ",
              flush=True)
        for _ in range(args.warmup):
            infer_one(context, inp, io_buffers)
        print("done.")

        # Check SM clock before timed loop
        clk_before = query_sm_clock(args.gpu_id)

        print(f"[{name}] Benchmarking ({args.iterations} iters) ...",
              end=" ", flush=True)
        latencies = []
        last_out = None
        for _ in range(args.iterations):
            t0 = time.perf_counter()
            last_out = infer_one(context, inp, io_buffers)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)
        print("done.")

        # Check SM clock after timed loop
        clk_after = query_sm_clock(args.gpu_id)
        if clk_before > 0 and clk_after < clk_before * 0.95:
            print(f"[WARN] SM clock dropped from {clk_before} MHz to "
                  f"{clk_after} MHz during benchmark for {name}. "
                  "Results may be affected by thermal throttling.")

        stats = compute_latency_stats(latencies)
        print(f"[{name}] Mean={stats['mean']:.2f} ms  "
              f"P95={stats['p95']:.2f} ms  FPS={stats['fps']:.1f}")

        # Extract depth output
        out_name, out_arr = last_out[0]
        depth_hw = squeeze_depth_to_hw(out_arr, out_name)
        depth_resized = cv2.resize(depth_hw, (w0, h0),
                                   interpolation=cv2.INTER_CUBIC)
        depth_norm = normalize_depth(depth_resized)

        engine_results.append({
            "name": name,
            "latencies": latencies,
            "speed_stats": stats,
            "depth_raw": depth_resized,
            "depth_norm": depth_norm,
        })

    # ------------------------------------------------------------------
    # Stage 3 - GPU Utilization Monitoring  (separate pass)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Stage 3: GPU Utilization Monitoring")
    print("=" * 70)

    for ed, res in zip(engine_data, engine_results):
        name = ed["name"]
        context = ed["context"]
        io_buffers = ed["io_buffers"]
        inp = ed["inp"]

        print(f"\n[{name}] GPU monitoring pass ({args.iterations} iters, "
              f"sampling every {args.gpu_sample_interval}s) ...",
              end=" ", flush=True)

        sampler = GPUUtilSampler(gpu_id=args.gpu_id,
                                 interval=args.gpu_sample_interval)
        with sampler:
            for _ in range(args.iterations):
                infer_one(context, inp, io_buffers)
        print("done.")

        n_samples = len(sampler.timestamps)
        print(f"[{name}] Collected {n_samples} GPU samples.")

        res["gpu_samples"] = {
            "timestamps": sampler.timestamps,
            "gpu_util_pct": sampler.gpu_util_pct,
            "mem_util_pct": sampler.mem_util_pct,
            "vram_used_mb": sampler.vram_used_mb,
            "sm_clock_mhz": sampler.sm_clock_mhz,
            "mem_clock_mhz": sampler.mem_clock_mhz,
        }

        if n_samples > 0:
            avg_gpu = np.mean(sampler.gpu_util_pct)
            avg_mem = np.mean(sampler.mem_util_pct)
            avg_vram = np.mean(sampler.vram_used_mb)
            print(f"[{name}] Avg GPU util={avg_gpu:.1f}%  "
                  f"Mem util={avg_mem:.1f}%  VRAM={avg_vram:.0f} MB")

    # ------------------------------------------------------------------
    # Stage 4 - Metric Calculation
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Stage 4: Metric Calculation")
    print("=" * 70)

    # -- Cross-engine comparison on the visualization image ---------------
    if len(engine_results) > 1:
        ref_depth = engine_results[0]["depth_raw"]
        engine_results[0]["cross_metrics"] = None  # reference
        for res in engine_results[1:]:
            cross = compute_cross_engine_metrics(res["depth_raw"], ref_depth)
            res["cross_metrics"] = cross
            print(f"[{res['name']}] vs {engine_results[0]['name']}: "
                  f"MAE={cross['mae']:.6f}  RMSE={cross['rmse']:.6f}  "
                  f"MaxErr={cross['max_error']:.6f}")

    # -- GT metrics (single image or multi-image) -------------------------
    if args.metric_images_dir:
        # Multi-image metric evaluation
        metric_image_paths = get_image_paths(args.metric_images_dir)
        if not metric_image_paths:
            print("[WARN] No images found in --metric-images-dir: "
                  f"{args.metric_images_dir}")
        else:
            print(f"\nMulti-image metric evaluation: {len(metric_image_paths)}"
                  f" images from {args.metric_images_dir}")
            print(f"GT directory: {args.metric_gt_dir}")

            # Accumulators: engine_name -> metric_key -> list of values
            GT_METRIC_KEYS = ("abs_rel", "sq_rel", "rmse", "d1", "d2", "d3")
            accum = {res["name"]: {k: [] for k in GT_METRIC_KEYS}
                     for res in engine_results}
            n_evaluated = 0

            for img_path in metric_image_paths:
                basename = os.path.splitext(os.path.basename(img_path))[0]
                gt_path = os.path.join(args.metric_gt_dir, basename + ".npy")
                if not os.path.isfile(gt_path):
                    continue

                # Load image and GT
                img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img_bgr is None:
                    continue
                ih, iw = img_bgr.shape[:2]

                img_gt = np.load(gt_path).astype(np.float32)
                if img_gt.ndim == 3:
                    img_gt = img_gt.squeeze()
                img_gt = cv2.resize(img_gt, (iw, ih),
                                    interpolation=cv2.INTER_CUBIC)

                # Run inference on each engine and compute metrics
                for ed, res in zip(engine_data, engine_results):
                    inp_m = preprocess_bgr_to_nchw(
                        img_bgr, args.input_size, dtype=ed["inp"].dtype)
                    out = infer_one(ed["context"], inp_m, ed["io_buffers"])
                    out_name, out_arr = out[0]
                    inv_depth_hw = squeeze_depth_to_hw(out_arr, out_name)
                    inv_depth_resized = cv2.resize(inv_depth_hw, (iw, ih),
                                               interpolation=cv2.INTER_CUBIC)
                    depth_resized = inverse_depth_to_depth(inv_depth_resized)
                    pred_norm = normalize_depth(depth_resized)

                    gt_m = compute_gt_metrics(depth_resized, img_gt)
                    for k in GT_METRIC_KEYS:
                        if not np.isnan(gt_m[k]):
                            accum[res["name"]][k].append(gt_m[k])

                n_evaluated += 1

            # Average accumulated metrics
            print(f"Evaluated {n_evaluated} images with matching GT.\n")
            for res in engine_results:
                avg = {}
                for k in GT_METRIC_KEYS:
                    vals = accum[res["name"]][k]
                    avg[k] = float(np.mean(vals)) if vals else float("nan")
                res["gt_metrics"] = avg
                res["gt_metrics_n_images"] = n_evaluated
                print(f"[{res['name']}] GT (avg over {n_evaluated} imgs): "
                      f"AbsRel={avg['abs_rel']:.4f}  "
                      f"RMSE={avg['rmse']:.4f}  "
                      f"d1={avg['d1']:.1f}%  d2={avg['d2']:.1f}%  "
                      f"d3={avg['d3']:.1f}%")

    elif gt_raw is not None:
        # Single-image GT metric evaluation (original behavior)
        for res in engine_results:
            gt_m = compute_gt_metrics(res["depth_raw"], gt_raw)
            res["gt_metrics"] = gt_m
            print(f"[{res['name']}] GT: AbsRel={gt_m['abs_rel']:.4f}  "
                  f"RMSE={gt_m['rmse']:.4f}  "
                  f"d1={gt_m['d1']:.1f}%  d2={gt_m['d2']:.1f}%  "
                  f"d3={gt_m['d3']:.1f}%")

    # ------------------------------------------------------------------
    # Stage 5 - Visualization
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Stage 5: Visualization")
    print("=" * 70)

    # 1. Depth comparison
    depth_path = os.path.join(args.output_dir, "depth_comparison.png")
    plot_depth_comparison(rgb, engine_results, gt_depth_norm, depth_path)
    print(f"Saved: {depth_path}")

    # 2. Speed benchmark
    speed_path = os.path.join(args.output_dir, "speed_benchmark.png")
    plot_speed_benchmark(engine_results, speed_path)
    print(f"Saved: {speed_path}")

    # 3. GPU utilization
    gpu_path = os.path.join(args.output_dir, "gpu_utilization.png")
    plot_gpu_utilization(engine_results, gpu_path)
    print(f"Saved: {gpu_path}")

    # 4. Metrics markdown table
    md_path = os.path.join(args.output_dir, "benchmark_metrics.md")
    save_metrics_markdown(engine_results, md_path)
    print(f"Saved: {md_path}")

    # Print combined summary
    print_summary(engine_results)

    print(f"\nAll outputs saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
