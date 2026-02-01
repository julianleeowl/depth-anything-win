#!/usr/bin/env python3
"""
TensorRT Engine Comparison Tool

Runs inference on multiple TensorRT engines, profiles inference speed,
and plots all results in a single comparison grid.
"""
import argparse
import math
import os
import time

import cv2
import numpy as np
import tensorrt as trt
import matplotlib.pyplot as plt
from tqdm import tqdm

import pycuda.driver as cuda
import pycuda.autoinit  # initializes CUDA context


# DepthAnythingV2 preprocessing:
# - BGR -> RGB
# - scale to [0, 1]
# - normalize with ImageNet mean/std
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_bgr_to_nchw(bgr: np.ndarray, input_size: int, dtype=np.float16) -> np.ndarray:
    """Preprocess BGR image to NCHW format for TensorRT inference."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = cv2.resize(rgb, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    chw = np.transpose(rgb, (2, 0, 1))  # HWC -> CHW
    nchw = np.expand_dims(chw, axis=0)  # 1x3xHxW
    return np.ascontiguousarray(nchw.astype(dtype))


def normalize_depth(depth_map: np.ndarray) -> np.ndarray:
    """Normalizes depth map to 0.0 - 1.0 for visualization."""
    d_min = depth_map.min()
    d_max = depth_map.max()
    return (depth_map - d_min) / (d_max - d_min + 1e-8)


def get_image_paths(path: str) -> list:
    """
    Get list of image paths from a file or directory.

    Args:
        path: Path to an image file or directory containing images

    Returns:
        List of image file paths
    """
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    if os.path.isfile(path):
        return [path]
    elif os.path.isdir(path):
        image_paths = []
        for fname in sorted(os.listdir(path)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                image_paths.append(os.path.join(path, fname))
        return image_paths
    else:
        raise FileNotFoundError(f"Path not found: {path}")


def compute_depth_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """
    Compute depth estimation metrics between prediction and ground truth.

    Uses median scaling to align prediction to ground truth scale.

    Args:
        pred: Predicted depth map (HxW)
        gt: Ground truth depth map (HxW)

    Returns:
        dict with:
            - absrel: Absolute Relative Error
            - delta1: δ < 1.25 accuracy (percentage)
    """
    # Create valid mask (gt > 0 to avoid division by zero)
    valid_mask = gt > 1e-8

    pred_valid = pred[valid_mask]
    gt_valid = gt[valid_mask]

    if len(pred_valid) == 0:
        return {"absrel": float('nan'), "delta1": float('nan')}

    # Median scaling to align prediction to ground truth
    scale = np.median(gt_valid) / (np.median(pred_valid) + 1e-8)
    pred_scaled = pred_valid * scale

    # AbsRel: mean(|pred - gt| / gt)
    absrel = np.mean(np.abs(pred_scaled - gt_valid) / gt_valid)

    # δ < 1.25: percentage of pixels where max(pred/gt, gt/pred) < 1.25
    ratio = np.maximum(pred_scaled / gt_valid, gt_valid / pred_scaled)
    delta1 = np.mean(ratio < 1.25) * 100.0  # as percentage

    return {"absrel": absrel, "delta1": delta1}


def load_engine(engine_path: str, logger: trt.ILogger) -> trt.ICudaEngine:
    """Load a TensorRT engine from file."""
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(
            f"Failed to deserialize engine '{engine_path}'. Common causes: "
            "TensorRT/CUDA version mismatch, GPU mismatch, or corrupted engine file."
        )
    return engine


def get_io_tensors(engine: trt.ICudaEngine):
    """Get input/output tensor info (TensorRT 10+ API)."""
    inputs, outputs = [], []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        (inputs if mode == trt.TensorIOMode.INPUT else outputs).append((name, dtype))

    if len(inputs) != 1:
        raise ValueError(f"Expected exactly 1 input, found {len(inputs)}: {inputs}")
    if len(outputs) < 1:
        raise ValueError("Expected at least 1 output")

    return inputs, outputs


def _is_dynamic_shape(shape):
    return any(int(d) == -1 for d in shape)


def allocate_io(engine: trt.ICudaEngine, context: trt.IExecutionContext, input_shape):
    """Allocate host/device buffers and bind to tensor names."""
    inputs, outputs = get_io_tensors(engine)
    in_name, in_dtype = inputs[0]

    # Set input shape for dynamic engines
    cur_in_shape = tuple(context.get_tensor_shape(in_name))
    if cur_in_shape != tuple(input_shape):
        context.set_input_shape(in_name, tuple(input_shape))

    in_shape = tuple(context.get_tensor_shape(in_name))
    if _is_dynamic_shape(in_shape):
        raise RuntimeError(
            f"Input shape for '{in_name}' is still dynamic after set_input_shape: {in_shape}."
        )

    stream = cuda.Stream()

    # Allocate input buffers
    in_vol = int(np.prod(in_shape))
    host_in = cuda.pagelocked_empty(in_vol, dtype=in_dtype)
    dev_in = cuda.mem_alloc(host_in.nbytes)

    # Allocate outputs
    host_outs, dev_outs, out_meta = [], [], []
    for out_name, out_dtype in outputs:
        out_shape = tuple(context.get_tensor_shape(out_name))
        if _is_dynamic_shape(out_shape):
            raise RuntimeError(
                f"Output shape for '{out_name}' is dynamic: {out_shape}."
            )
        out_vol = int(np.prod(out_shape))
        host_out = cuda.pagelocked_empty(out_vol, dtype=out_dtype)
        dev_out = cuda.mem_alloc(host_out.nbytes)

        host_outs.append(host_out)
        dev_outs.append(dev_out)
        out_meta.append((out_name, out_dtype, out_shape))

    # Bind device pointers to tensor names
    context.set_tensor_address(in_name, int(dev_in))
    for dev_out, (out_name, _, _) in zip(dev_outs, out_meta):
        context.set_tensor_address(out_name, int(dev_out))

    return (in_name, host_in, dev_in, in_shape), (host_outs, dev_outs, out_meta), stream


def infer_one(engine: trt.ICudaEngine, context: trt.IExecutionContext,
              inp_nchw: np.ndarray, io_buffers=None):
    """
    Run one inference using TRT10 execute_async_v3.
    Returns list of (output_name, output_array_reshaped).
    """
    input_shape = tuple(inp_nchw.shape)

    if io_buffers is None:
        io_buffers = allocate_io(engine, context, input_shape)

    (in_name, host_in, dev_in, in_shape), (host_outs, dev_outs, out_meta), stream = io_buffers

    if int(np.prod(in_shape)) != inp_nchw.size:
        raise ValueError(f"Input size mismatch. Engine expects {in_shape}, got {inp_nchw.shape}.")

    # H2D
    np.copyto(host_in, inp_nchw.ravel())
    cuda.memcpy_htod_async(dev_in, host_in, stream)

    # Execute
    ok = context.execute_async_v3(stream_handle=stream.handle)
    if not ok:
        raise RuntimeError("TensorRT execution failed.")

    # D2H
    for host_out, dev_out in zip(host_outs, dev_outs):
        cuda.memcpy_dtoh_async(host_out, dev_out, stream)
    stream.synchronize()

    # Reshape outputs
    outputs = []
    for host_out, (name, _, shape) in zip(host_outs, out_meta):
        outputs.append((name, host_out.reshape(shape)))
    return outputs, io_buffers


def squeeze_depth_to_hw(out: np.ndarray, name: str) -> np.ndarray:
    """Convert depth output to HxW float32."""
    depth = out
    if depth.ndim == 4 and depth.shape[1] == 1:  # [N,1,H,W]
        depth = depth[:, 0, :, :]
    if depth.ndim == 3:  # [N,H,W]
        depth2d = depth[0]
    elif depth.ndim == 2:  # [H,W]
        depth2d = depth
    else:
        raise ValueError(f"Unexpected output shape for {name}: {out.shape}")
    return depth2d.astype(np.float32)


def run_engine_inference(engine_path: str, bgr: np.ndarray, input_size: int,
                         warmup: int = 5, runs: int = 20,
                         logger: trt.ILogger = None, silent: bool = False) -> dict:
    """
    Run inference on a TensorRT engine and profile speed.

    Returns dict with:
        - name: engine filename
        - depth: normalized depth map (resized to original image size)
        - depth_raw: raw depth for metric computation
        - latency_ms: average latency in milliseconds
        - fps: frames per second
    """
    engine_name = os.path.basename(engine_path)
    if not silent:
        print(f"[INFO] Loading engine: {engine_name}...")

    try:
        if logger is None:
            logger = trt.Logger(trt.Logger.WARNING)

        engine = load_engine(engine_path, logger)
        context = engine.create_execution_context()

        h0, w0 = bgr.shape[:2]
        # print(f"[INFO] Input image size: {w0}x{h0}")

        # Get input dtype from engine
        inputs, _ = get_io_tensors(engine)
        _, in_dtype = inputs[0]
        inp = preprocess_bgr_to_nchw(bgr, input_size, dtype=in_dtype)

        print(f"[INFO] Inference input size: {inp.shape[3]}x{inp.shape[2]} (WxH)")

        # Warmup (also allocates buffers)
        io_buffers = None
        for _ in range(warmup):
            _, io_buffers = infer_one(engine, context, inp, io_buffers)

        # Timed runs
        if not silent:
            print(f"[INFO] Running {runs} inference iterations on {engine_name}...")
        t0 = time.perf_counter()
        last_out = None
        for _ in range(runs):
            last_out, io_buffers = infer_one(engine, context, inp, io_buffers)
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000.0 / runs
        fps = 1000.0 / latency_ms

        # Process depth output
        name, out = last_out[0]
        depth2d = squeeze_depth_to_hw(out, name)
        depth_resized = cv2.resize(depth2d, (w0, h0), interpolation=cv2.INTER_CUBIC)
        depth_norm = normalize_depth(depth_resized)

        if not silent:
            print(f"[INFO] {engine_name}: {latency_ms:.3f} ms ({fps:.1f} FPS)")

        return {
            "name": engine_name,
            "depth": depth_norm,
            "depth_raw": depth_resized,  # raw depth for metric computation
            "latency_ms": latency_ms,
            "fps": fps,
        }

    except Exception as e:
        if not silent:
            print(f"[ERR] Failed to run {engine_name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Compare Multiple TensorRT Engines for Depth Estimation"
    )
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image or directory containing images")
    parser.add_argument("--output", type=str, default="engine_comparison.png",
                        help="Path to save output comparison image")
    parser.add_argument("--input-size", type=int, default=518,
                        help="Input size for preprocessing (default: 518)")
    parser.add_argument("--warmup", type=int, default=20,
                        help="Number of warmup iterations (default: 20)")
    parser.add_argument("--runs", type=int, default=100,
                        help="Number of timed runs (default: 100)")
    parser.add_argument("--engines", type=str, nargs="+",
                        help="List of engine paths (overrides built-in list)")
    args = parser.parse_args()

    # Default engine list - modify as needed
    default_engines = [
        "/home/hoiliu/model-weights/depth-model/depth_anything_v2_vits_518_fp16.engine",
        "/home/hoiliu/model-weights/depth-model/depth_anything_v2_vits_518_int8.engine",
        "/home/hoiliu/model-weights/depth-model/distill_any_depth_multi_teacher_small_vits_518_fp16.engine",
        "/home/hoiliu/model-weights/depth-model/distill_any_depth_multi_teacher_small_vits_518_int8.engine",
    ]

    engine_list = args.engines if args.engines else default_engines

    # Get image paths
    image_paths = get_image_paths(args.image)
    if not image_paths:
        raise ValueError(f"No images found in: {args.image}")

    print(f"Found {len(image_paths)} image(s)")
    print(f"Starting comparison on {len(engine_list)} engines...")
    print("=" * 80)

    logger = trt.Logger(trt.Logger.WARNING)

    # Filter valid engines
    valid_engines = [e for e in engine_list if os.path.exists(e)]
    for e in engine_list:
        if not os.path.exists(e):
            print(f"[SKIP] Engine file not found: {e}")

    if not valid_engines:
        print("[ERR] No valid engines found.")
        return

    # Initialize accumulators for metrics (per engine)
    engine_metrics = {os.path.basename(e): {"absrel": [], "delta1": [], "latency_ms": [], "fps": []}
                      for e in valid_engines}

    # Store results from last image for visualization
    last_results = None
    last_rgb = None
    last_gt_norm = None
    last_image_path = None

    # Process each image
    pbar = tqdm(image_paths, desc="Processing images", unit="img")
    for image_path in pbar:
        pbar.set_postfix_str(os.path.basename(image_path)[:30])

        bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if bgr is None:
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h0, w0 = bgr.shape[:2]

        # Run inference on all engines for this image
        results = []
        for engine_path in valid_engines:
            result = run_engine_inference(
                engine_path, bgr, args.input_size,
                warmup=args.warmup, runs=args.runs, logger=logger,
                silent=True
            )
            if result is not None:
                results.append(result)
                # Accumulate timing
                engine_metrics[result["name"]]["latency_ms"].append(result["latency_ms"])
                engine_metrics[result["name"]]["fps"].append(result["fps"])

        if not results:
            continue

        # Check for ground truth (.npy)
        gt_data_raw = None
        gt_data_norm = None
        base_name = os.path.splitext(image_path)[0]
        gt_path = base_name + ".npy"

        if os.path.exists(gt_path):
            try:
                gt_arr = np.load(gt_path).astype(np.float32)
                if gt_arr.ndim == 3:
                    gt_arr = gt_arr.squeeze()
                gt_arr = cv2.resize(gt_arr, (w0, h0), interpolation=cv2.INTER_CUBIC)
                gt_arr = 1.0 / gt_arr  # Convert to inverse depth
                gt_data_raw = gt_arr
                gt_data_norm = normalize_depth(gt_arr)
            except Exception:
                pass

        # Compute metrics if GT is available
        if gt_data_raw is not None:
            for res in results:
                metrics = compute_depth_metrics(res["depth_raw"], gt_data_raw)
                res["absrel"] = metrics["absrel"]
                res["delta1"] = metrics["delta1"]
                # Accumulate metrics
                engine_metrics[res["name"]]["absrel"].append(metrics["absrel"])
                engine_metrics[res["name"]]["delta1"].append(metrics["delta1"])

        # Store last results for visualization
        last_results = results
        last_rgb = rgb
        last_gt_norm = gt_data_norm
        last_image_path = image_path

    print("\n" + "=" * 80)

    if last_results is None:
        print("[ERR] No valid results to plot.")
        return

    # Compute averaged metrics
    results = []
    for engine_path in valid_engines:
        name = os.path.basename(engine_path)
        m = engine_metrics[name]

        res = {
            "name": name,
            "latency_ms": np.mean(m["latency_ms"]) if m["latency_ms"] else 0,
            "fps": np.mean(m["fps"]) if m["fps"] else 0,
        }

        if m["absrel"]:
            res["absrel"] = np.mean(m["absrel"])
            res["delta1"] = np.mean(m["delta1"])

        results.append(res)

    # Get depth visualizations from last image
    for res, last_res in zip(results, last_results):
        res["depth"] = last_res["depth"]

    rgb = last_rgb
    gt_data_norm = last_gt_norm

    num_images = len(image_paths)
    has_metrics = any("absrel" in res for res in results)
    if has_metrics:
        print(f"[INFO] Metrics averaged over {len(engine_metrics[results[0]['name']]['absrel'])} images with GT")

    # Grid Plotting
    # Layout: Original | GT (or blank) | Engine results...
    total_slots = 2 + len(results)
    cols = 2
    rows = math.ceil(total_slots / cols)

    print(f"[INFO] Plotting grid: {rows} rows x {cols} cols")
    fig = plt.figure(figsize=(10, 5 * rows))

    # Plot 1: Original Image
    plt.subplot(rows, cols, 1)
    sample_title = f"Original Input\n({os.path.basename(last_image_path)})"
    plt.title(sample_title, fontsize=8)
    plt.imshow(rgb)
    plt.axis("off")

    # Plot 2: Ground Truth or Blank
    plt.subplot(rows, cols, 2)
    if gt_data_norm is not None:
        plt.title("Ground Truth", fontweight='bold')
        plt.imshow(gt_data_norm, cmap='inferno')
        plt.axis("off")
    else:
        plt.axis("off")

    # Plot 3..N: Engine Results with timing and metrics info
    avg_label = " (avg)" if num_images > 1 else ""
    for idx, res in enumerate(results):
        plt.subplot(rows, cols, 3 + idx)
        title = f"{res['name']}\n{res['latency_ms']:.2f}ms ({res['fps']:.1f} FPS){avg_label}"
        if "absrel" in res:
            title += f"\nAbsRel={res['absrel']:.4f}, δ<1.25={res['delta1']:.1f}%{avg_label}"
        plt.title(title, fontsize=8)
        plt.imshow(res["depth"], cmap='inferno')
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    plt.close()

    print(f"[SUCCESS] Comparison grid saved to: {args.output}")

    # Print summary table
    print("\n" + "=" * 80)
    if num_images > 1:
        print(f"PERFORMANCE SUMMARY (averaged over {num_images} images)")
    else:
        print("PERFORMANCE SUMMARY")
    print("=" * 80)

    if has_metrics:
        print(f"{'Engine':<45} {'Latency':>10} {'FPS':>8} {'AbsRel':>10} {'δ<1.25':>10}")
        print("-" * 80)
        for res in results:
            absrel_str = f"{res['absrel']:.4f}" if "absrel" in res else "N/A"
            delta1_str = f"{res['delta1']:.2f}%" if "delta1" in res else "N/A"
            print(f"{res['name']:<45} {res['latency_ms']:>7.2f}ms {res['fps']:>7.1f} {absrel_str:>10} {delta1_str:>10}")
    else:
        print(f"{'Engine':<55} {'Latency':>10} {'FPS':>8}")
        print("-" * 80)
        for res in results:
            print(f"{res['name']:<55} {res['latency_ms']:>7.2f}ms {res['fps']:>7.1f}")

    print("=" * 80)


if __name__ == "__main__":
    main()
