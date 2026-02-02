# benchmark_trt.py

Comprehensive TensorRT benchmark for depth estimation models. Compares one or more TensorRT engines across four dimensions: depth quality metrics, inference speed, GPU utilization, and visual output.

## Context

The project has several existing scripts that each cover part of the benchmarking picture (`infer_trt.py` for single-engine inference, `infer_engine.py` for multi-engine comparison, `benchmark_trt_vs_compiled.py` for TRT vs PyTorch). This script consolidates all four benchmarking needs into a single tool with production-representative methodology, producing three separate output figures plus a stdout summary.

## Usage

```bash
# Basic: speed + GPU + visualization (single image)
python benchmark_trt.py \
    --engines fp16.engine int8.engine mixed.engine \
    --image path/to/image.jpg \
    --output-dir benchmark_results/

# With single-image GT metrics
python benchmark_trt.py \
    --engines fp16.engine int8.engine \
    --image path/to/image.jpg \
    --gt path/to/gt_depth.npy \
    --output-dir benchmark_results/

# With multi-image GT metrics (averaged over a dataset)
python benchmark_trt.py \
    --engines fp16.engine int8.engine \
    --image path/to/viz_image.jpg \
    --metric-images-dir path/to/images/ \
    --metric-gt-dir path/to/gt_npy/ \
    --output-dir benchmark_results/
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--engines` | (required) | One or more `.engine` file paths |
| `--image` | (required) | Input image for speed/GPU benchmarking and depth visualization |
| `--gt` | None | Ground-truth depth `.npy` for the visualization image (single-image GT mode) |
| `--metric-images-dir` | None | Directory of images for multi-image metric evaluation |
| `--metric-gt-dir` | None | Directory of GT `.npy` files matching images by basename (required with `--metric-images-dir`) |
| `--input-size` | 518 | Model input resolution (must match engine) |
| `--warmup` | 50 | Warmup iterations before timed loop |
| `--iterations` | 1000 | Timed iterations (1000 gives reliable p99 with ~10 tail samples) |
| `--gpu-sample-interval` | 0.1 | pynvml polling interval in seconds during GPU monitoring pass |
| `--gpu-id` | 0 | GPU device index for pynvml queries |
| `--output-dir` | `benchmark_results/` | Directory for output files |

### Output Files

| File | Contents |
|------|----------|
| `depth_comparison.png` | Horizontal grid: original image, GT (if provided), depth prediction per engine. Titled with engine name, latency, FPS, and metrics. |
| `speed_benchmark.png` | Top: box plot of per-engine latency distributions. Bottom: stats table (mean, std, median, p50, p95, p99, min, max, FPS). |
| `gpu_utilization.png` | Four stacked time-series plots (shared x-axis): GPU compute util %, memory util %, VRAM usage (MB), SM clock (MHz). One line per engine. |

Plus a combined summary table printed to stdout.

## Design Decisions

### Production-representative speed benchmarking

The speed benchmark is designed to mimic a real serving loop:

- **Pre-allocated buffers**: `allocate_io()` is called once per engine. The same pinned host buffers and device buffers are reused across all iterations, matching production behavior where you allocate once at startup.
- **Full round-trip timing**: Each timed iteration measures the complete H2D copy -> `execute_async_v3` -> D2H copy -> `stream.synchronize()` pipeline via `time.perf_counter()`. This wall-clock approach captures exactly what a serving loop pays, unlike CUDA event timing which only measures GPU-side kernel time.
- **Per-iteration synchronization**: `stream.synchronize()` inside `infer_one()` ensures each measurement reflects a fully completed inference, with no overlap from async execution leaking across iterations.
- **Inference-only timing**: Preprocessing (resize + normalize) happens once before the timed loop, not inside it. This isolates engine performance from CPU preprocessing cost.
- **1000 iterations by default**: Provides enough samples for stable tail-latency percentiles -- p95 has ~50 samples, p99 has ~10 samples.

### Thermal throttle detection

SM clock speed is queried via `pynvml` before and after the timed loop. If clock drops >5%, a warning is printed. This alerts the user that results may not represent sustained performance. The script does not lock clocks automatically (that requires root), but the existing `utils/config_gpu.py` `GPUConfigurator` can be used externally if needed.

### Separate passes for speed vs GPU monitoring

The speed benchmark runs in a tight loop with no background threads to avoid any interference with latency measurements. GPU utilization monitoring runs in a second pass with a background `pynvml` polling thread. The pynvml queries are lightweight driver calls (microseconds each) and run in a separate thread, but we still isolate the passes to ensure the speed numbers are as clean as possible.

### Metric evaluation modes

Three metric modes, selected by which arguments are provided:

- **Cross-engine comparison** (always, if >1 engine): First engine is treated as the reference. Both depth maps are normalized to [0,1], then MAE, RMSE, and max_error are computed. Useful for comparing FP16 vs INT8 output quality without needing ground truth.
- **Single-image GT** (when `--gt` is provided): Evaluates against one GT depth map for the visualization image. Uses median scaling to align prediction to GT scale. Computes: abs_rel, sq_rel, rmse, d1, d2, d3.
- **Multi-image GT** (when `--metric-images-dir` and `--metric-gt-dir` are provided): Evaluates each engine across all images in the directory. GT files are matched by basename (e.g., `img001.jpg` pairs with `img001.npy`). Per-image metrics are computed then averaged. The visualization still uses `--image` for the depth plot, with averaged metrics shown in titles. This mode takes precedence over `--gt` for metric computation.

### GPU utilization metrics

The `GPUUtilSampler` background thread captures:

| Metric | What it tells you |
|--------|-------------------|
| GPU compute util % | Is the GPU busy or idle-waiting? Low values indicate CPU/memory bottleneck. |
| Memory util % | Memory bandwidth usage. ViT-based depth models tend to be memory-bound. |
| VRAM usage (MB) | How much GPU memory the engine occupies. Critical for deployment planning. |
| SM clock (MHz) | Current operating frequency. Drops indicate thermal throttling. |

Power draw and temperature were intentionally excluded -- the pynvml overhead is the same either way, but they add chart noise without being essential for most benchmarking needs.

## Execution Stages

### Stage 1 - Setup
- Load image (BGR via OpenCV), convert to RGB for visualization
- Load optional GT depth (`.npy` format, resized to image dimensions)
- For each engine: deserialize with `trt.Runtime`, create execution context, detect input dtype, preprocess image, pre-allocate I/O buffers

### Stage 2 - Speed Benchmark
- For each engine:
  1. Run warmup iterations (not timed) to stabilize GPU clocks and warm caches
  2. Query SM clock (before)
  3. Run timed iterations in a tight loop, recording per-iteration wall-clock latency
  4. Query SM clock (after), warn if >5% drop
  5. Compute latency statistics and capture depth output from last iteration

### Stage 3 - GPU Utilization Monitoring
- For each engine:
  1. Start `GPUUtilSampler` background thread (polls pynvml at `--gpu-sample-interval`)
  2. Run inference iterations (same count as speed benchmark)
  3. Stop sampler, store time-series arrays

### Stage 4 - Metric Calculation
- Cross-engine: normalize depth maps to [0,1], compute MAE/RMSE/max_error vs first engine (uses the `--image` output)
- GT metrics (two modes):
  - **Single-image**: if `--gt` is provided, compute metrics on the visualization image
  - **Multi-image**: if `--metric-images-dir` is provided, loop over all images, run inference per engine, compute per-image GT metrics, average across the dataset. Images without a matching `.npy` in `--metric-gt-dir` are skipped.

### Stage 5 - Visualization
- Generate three separate PNG files using the `--image` for depth plots
- Depth comparison subplot titles show averaged metrics (with image count) when multi-image mode is used
- Summary table to stdout indicates whether GT metrics are single-image or averaged

## Dependencies

All already present in the project -- no new dependencies required:

- `tensorrt` -- engine loading and inference
- `pycuda` -- CUDA memory management and H2D/D2H transfers
- `pynvml` -- GPU utilization monitoring and clock queries
- `numpy` -- array operations and metric computation
- `opencv-python` -- image I/O and preprocessing
- `matplotlib` -- all visualization (depth maps, box plots, time-series, tables)

## Code Lineage

Patterns adapted from existing project files:

- **TRT I/O and inference**: `infer_engine.py` -- `load_engine()`, `get_io_tensors()`, `allocate_io()`, `infer_one()`, `squeeze_depth_to_hw()`
- **Latency statistics**: `benchmark_trt_vs_compiled.py` -- `compute_latency_stats()` with percentile metrics
- **GPU monitoring**: `utils/config_gpu.py` -- `GPUMonitor` class pattern (background thread with `Condition`-based sleep and pynvml sampling)
- **Depth metrics**: `metric_depth/util/metric.py` (GT evaluation) and `infer_engine.py` (cross-engine comparison with median scaling)
