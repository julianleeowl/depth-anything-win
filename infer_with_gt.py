#!/usr/bin/env python3
"""
Inference with Ground Truth Comparison

Loads a TensorRT engine, runs inference on an input image, and compares
the depth prediction against ground truth with metrics calculation and
visualization.
"""

import argparse
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorrt as trt

import pycuda.driver as cuda
import pycuda.autoinit


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_bgr_to_nchw(bgr: np.ndarray, input_size: int,
                            dtype=np.float16) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = cv2.resize(rgb, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    chw = np.transpose(rgb, (2, 0, 1))
    nchw = np.expand_dims(chw, axis=0)
    return np.ascontiguousarray(nchw.astype(dtype))


def normalize_depth(depth_map: np.ndarray) -> np.ndarray:
    d_min = depth_map.min()
    d_max = depth_map.max()
    return (depth_map - d_min) / (d_max - d_min + 1e-8)


def load_engine(engine_path: str, logger: trt.ILogger) -> trt.ICudaEngine:
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize engine '{engine_path}'")
    return engine


def get_io_tensors(engine: trt.ICudaEngine):
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


def allocate_io(engine: trt.ICudaEngine, context: trt.IExecutionContext,
                input_shape):
    inputs, outputs = get_io_tensors(engine)
    in_name, in_dtype = inputs[0]

    cur_in_shape = tuple(context.get_tensor_shape(in_name))
    if cur_in_shape != tuple(input_shape):
        context.set_input_shape(in_name, tuple(input_shape))

    in_shape = tuple(context.get_tensor_shape(in_name))

    stream = cuda.Stream()

    in_vol = int(np.prod(in_shape))
    host_in = cuda.pagelocked_empty(in_vol, dtype=in_dtype)
    dev_in = cuda.mem_alloc(host_in.nbytes)

    host_outs, dev_outs, out_meta = [], [], []
    for out_name, out_dtype in outputs:
        out_shape = tuple(context.get_tensor_shape(out_name))
        out_vol = int(np.prod(out_shape))
        host_out = cuda.pagelocked_empty(out_vol, dtype=out_dtype)
        dev_out = cuda.mem_alloc(host_out.nbytes)
        host_outs.append(host_out)
        dev_outs.append(dev_out)
        out_meta.append((out_name, out_dtype, out_shape))

    context.set_tensor_address(in_name, int(dev_in))
    for dev_out, (out_name, _, _) in zip(dev_outs, out_meta):
        context.set_tensor_address(out_name, int(dev_out))

    return (in_name, host_in, dev_in, in_shape), \
           (host_outs, dev_outs, out_meta), stream


def infer_one(context: trt.IExecutionContext, inp_nchw: np.ndarray,
              io_buffers):
    (in_name, host_in, dev_in, in_shape), \
        (host_outs, dev_outs, out_meta), stream = io_buffers

    np.copyto(host_in, inp_nchw.ravel())
    cuda.memcpy_htod_async(dev_in, host_in, stream)

    ok = context.execute_async_v3(stream_handle=stream.handle)
    if not ok:
        raise RuntimeError("TensorRT execution failed")

    for host_out, dev_out in zip(host_outs, dev_outs):
        cuda.memcpy_dtoh_async(host_out, dev_out, stream)
    stream.synchronize()

    outputs = []
    for host_out, (name, _, shape) in zip(host_outs, out_meta):
        outputs.append((name, host_out.reshape(shape)))
    return outputs


def squeeze_depth_to_hw(out: np.ndarray, name: str) -> np.ndarray:
    depth = out
    if depth.ndim == 4 and depth.shape[1] == 1:
        depth = depth[:, 0, :, :]
    if depth.ndim == 3:
        depth2d = depth[0]
    elif depth.ndim == 2:
        depth2d = depth
    else:
        raise ValueError(f"Unexpected output shape for {name}: {out.shape}")
    return depth2d.astype(np.float32)


def inverse_depth_to_depth(inv_depth: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    depth = 1.0 / (inv_depth + eps)
    return depth


def compute_gt_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    valid = gt > 1e-8
    pred_v = pred[valid].astype(np.float64)
    gt_v = gt[valid].astype(np.float64)

    if len(pred_v) == 0:
        return {k: float("nan")
                for k in ("abs_rel", "sq_rel", "rmse", "d1", "d2", "d3")}

    scale = np.median(gt_v) / (np.median(pred_v) + 1e-8)
    pred_v = pred_v * scale

    pred_v = np.clip(pred_v, 1e-8, None)
    gt_v = np.clip(gt_v, 1e-8, None)

    thresh = np.maximum(pred_v / gt_v, gt_v / pred_v)
    d1 = float(np.mean(thresh < 1.25) * 100.0)
    d2 = float(np.mean(thresh < 1.25 ** 2) * 100.0)
    d3 = float(np.mean(thresh < 1.25 ** 3) * 100.0)

    diff = pred_v - gt_v
    abs_rel = float(np.mean(np.abs(diff) / gt_v))
    sq_rel = float(np.mean(diff ** 2 / gt_v))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    return {
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
        "d1": d1,
        "d2": d2,
        "d3": d3,
    }


def plot_comparison(rgb: np.ndarray, pred_norm: np.ndarray, gt_norm: np.ndarray,
                   metrics: dict, output_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(rgb)
    axes[0].set_title("Original Input", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(pred_norm, cmap="inferno")
    title_pred = "Prediction (Depth)"
    stats_text = (f"AbsRel={metrics['abs_rel']:.4f}\n"
                  f"RMSE={metrics['rmse']:.4f}\n"
                  f"d1={metrics['d1']:.1f}%\n"
                  f"d2={metrics['d2']:.1f}%\n"
                  f"d3={metrics['d3']:.1f}%")
    axes[1].set_title(f"{title_pred}\n{stats_text}", fontsize=9)
    axes[1].axis("off")

    axes[2].imshow(gt_norm, cmap="inferno")
    axes[2].set_title("Ground Truth", fontsize=10, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Run inference and compare with ground truth")
    parser.add_argument("--engine", type=str, required=True,
                        help="Path to TensorRT engine file")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--gt", type=str, required=True,
                        help="Path to ground-truth depth (.npy)")
    parser.add_argument("--input-size", type=int, default=518,
                        help="Model input size (default: 518)")
    parser.add_argument("--output", type=str, default="inference_result.png",
                        help="Output plot path (default: inference_result.png)")
    args = parser.parse_args()

    if not os.path.isfile(args.engine):
        raise FileNotFoundError(f"Engine not found: {args.engine}")
    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not os.path.isfile(args.gt):
        raise FileNotFoundError(f"GT not found: {args.gt}")

    print("=" * 70)
    print("  Inference with Ground Truth Comparison")
    print("=" * 70)

    print(f"Engine: {args.engine}")
    print(f"Image: {args.image}")
    print(f"GT: {args.gt}")

    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {args.image}")
    h0, w0 = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    print(f"Image size: {w0}x{h0}")

    gt_raw = np.load(args.gt).astype(np.float32)
    if gt_raw.ndim == 3:
        gt_raw = gt_raw.squeeze()
    gt_raw = cv2.resize(gt_raw, (w0, h0), interpolation=cv2.INTER_CUBIC)
    gt_norm = normalize_depth(gt_raw)
    print(f"GT shape: {gt_raw.shape}  range=[{gt_raw.min():.3f}, {gt_raw.max():.3f}]")

    trt_logger = trt.Logger(trt.Logger.WARNING)
    print(f"Loading engine: {args.engine} ...")
    engine = load_engine(args.engine, trt_logger)
    context = engine.create_execution_context()

    inputs, _ = get_io_tensors(engine)
    _, in_dtype = inputs[0]
    inp = preprocess_bgr_to_nchw(bgr, args.input_size, dtype=in_dtype)
    print(f"Input shape: {inp.shape}  dtype={in_dtype}")

    io_buffers = allocate_io(engine, context, tuple(inp.shape))
    print("Running inference ...")
    out = infer_one(context, inp, io_buffers)
    out_name, out_arr = out[0]
    print(f"Output shape: {out_arr.shape}  dtype={out_arr.dtype}")

    inv_depth_hw = squeeze_depth_to_hw(out_arr, out_name)
    inv_depth_resized = cv2.resize(inv_depth_hw, (w0, h0), interpolation=cv2.INTER_CUBIC)
    depth_resized = inverse_depth_to_depth(inv_depth_resized)
    pred_norm = normalize_depth(depth_resized)
    print(f"Inverse depth range: [{inv_depth_resized.min():.6f}, {inv_depth_resized.max():.6f}]")
    print(f"Depth range: [{depth_resized.min():.3f}, {depth_resized.max():.3f}]")

    metrics = compute_gt_metrics(depth_resized, gt_raw)

    print("\nMetrics:")
    print(f"  AbsRel:  {metrics['abs_rel']:.6f}")
    print(f"  SqRel:   {metrics['sq_rel']:.6f}")
    print(f"  RMSE:    {metrics['rmse']:.6f}")
    print(f"  d1:      {metrics['d1']:.2f}%")
    print(f"  d2:      {metrics['d2']:.2f}%")
    print(f"  d3:      {metrics['d3']:.2f}%")

    plot_comparison(rgb, pred_norm, gt_norm, metrics, args.output)
    print(f"\nSaved comparison plot: {args.output}")


if __name__ == "__main__":
    main()
