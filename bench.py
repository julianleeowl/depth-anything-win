#!/usr/bin/env python3
"""
Common benchmark and inference utilities for depth estimation.

This module contains shared functions used by both benchmark_trt.py and infer_with_gt.py.
"""

import cv2
import numpy as np
import torch


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_bgr_to_nchw(bgr: np.ndarray, input_size: int,
                            dtype=np.float16) -> np.ndarray:
    """Preprocess BGR image to NCHW tensor for TensorRT inference."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = cv2.resize(rgb, (input_size, input_size),
                     interpolation=cv2.INTER_CUBIC)
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    chw = np.transpose(rgb, (2, 0, 1))   # HWC -> CHW
    nchw = np.expand_dims(chw, axis=0)    # 1x3xHxW
    return np.ascontiguousarray(nchw.astype(dtype))


def normalize_depth(depth_map: np.ndarray) -> np.ndarray:
    """Normalize depth map to [0, 1] for visualization."""
    d_min = depth_map.min()
    d_max = depth_map.max()
    return (depth_map - d_min) / (d_max - d_min + 1e-8)


# def inverse_depth_to_depth(inv_depth: np.ndarray, eps: float = 1e-8, 
#                           max_depth: float = 1e3) -> np.ndarray:
#     """Convert inverse depth to depth with clamping.
    
#     Args:
#         inv_depth: Inverse depth array
#         eps: Small value to prevent division by zero
#         max_depth: Maximum reasonable depth value (default: 1000m)
    
#     Returns:
#         Depth array clamped to [eps, max_depth]
#     """
#     depth = 1.0 / (inv_depth + eps)
#     # Clamp to reasonable depth range to prevent explosions
#     depth = np.clip(depth, eps, max_depth)
#     return depth


def compute_gt_metrics(pred: np.ndarray, gt: np.ndarray, 
                      min_depth: float = 1e-4, max_depth: float = 1e3) -> dict:
    """Compute depth estimation metrics against ground truth.

    Uses median scaling to align prediction to GT scale with robust filtering
    to prevent exploding metrics from outliers or extreme values.

    Args:
        pred: Predicted depth map
        gt: Ground truth depth map
        min_depth: Minimum valid depth threshold (default: 0.0001m = 0.1mm)
        max_depth: Maximum valid depth threshold (default: 1000m)

    Returns:
        Dictionary with abs_rel, sq_rel, rmse, d1, d2, d3 metrics.
        Returns NaN for all metrics if insufficient valid pixels.
    """
    # Create robust valid mask: filter both GT and pred for finite values and depth range
    valid = (gt > min_depth) & (gt < max_depth) & np.isfinite(gt)
    valid = valid & np.isfinite(pred) & (pred > 0)
    
    pred_v = pred[valid].astype(np.float64)
    gt_v = gt[valid].astype(np.float64)

    # Need sufficient valid pixels for reliable metrics
    if len(pred_v) < 100:
        return {k: float("nan")
                for k in ("abs_rel", "sq_rel", "rmse", "d1", "d2", "d3")}

    # Compute medians for scaling
    pred_median = np.median(pred_v)
    gt_median = np.median(gt_v)
    
    # Check for degenerate medians
    if pred_median < 1e-6 or gt_median < 1e-6:
        return {k: float("nan")
                for k in ("abs_rel", "sq_rel", "rmse", "d1", "d2", "d3")}
    
    # Median scaling alignment
    scale = gt_median / pred_median
    
    # Sanity check: reject extreme scale factors (indicates data mismatch)
    if scale < 0.01 or scale > 100.0:
        # Scale factor is unreasonable - likely incompatible data
        return {k: float("nan")
                for k in ("abs_rel", "sq_rel", "rmse", "d1", "d2", "d3")}
    
    pred_v = pred_v * scale

    # Clamp scaled predictions to valid range
    pred_v = np.clip(pred_v, min_depth, max_depth)
    gt_v = np.clip(gt_v, min_depth, max_depth)

    # Threshold metrics (d1, d2, d3)
    thresh = np.maximum(pred_v / gt_v, gt_v / pred_v)
    d1 = float(np.mean(thresh < 1.25) * 100.0)
    d2 = float(np.mean(thresh < 1.25 ** 2) * 100.0)
    d3 = float(np.mean(thresh < 1.25 ** 3) * 100.0)

    # Error metrics with safe division
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

    # valid_mask = (target >= min_depth) & (target <= max_depth)

def align_scale_shift(pred_valid, target_valid):
    """
    Standard Least Squares Alignment (Affine Alignment)
    Source: MiDaS / Depth Anything Standard Evaluation Protocol
    
    Solves for s (scale) and t (shift) such that:
    s * pred + t = target
    """
    
    # # 1. Select only valid pixels (usually masked by ground truth validity)
    # pred_valid = pred[mask]
    # target_valid = target[mask]
    
    # 2. Stack prediction values with 1s to form matrix A: [pred, 1]
    #    This allows solving y = mx + c (or s*pred + t)
    A = np.vstack([pred_valid, np.ones_like(pred_valid)]).T
    b = target_valid
    
    # 3. Solve the linear system using Least Squares
    #    Result x = [scale, shift]
    (s, t), _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    # 4. Apply the scale and shift to the entire prediction
    pred_aligned = pred_valid * s + t
    
    return pred_aligned, s, t


def eval_depth_metrics(pred, target, min_depth=1e-4, max_depth=1e3):
    assert pred.shape == target.shape

    thresh = torch.max((target / pred), (pred / target))
    print(f"{thresh=}")

    d1 = torch.sum(thresh < 1.25).float() / len(thresh)
    d2 = torch.sum(thresh < 1.25 ** 2).float() / len(thresh)
    d3 = torch.sum(thresh < 1.25 ** 3).float() / len(thresh)

    diff = pred - target
    diff_log = torch.log(pred) - torch.log(target)

    abs_rel = torch.mean(torch.abs(diff) / target)
    sq_rel = torch.mean(torch.pow(diff, 2) / target)

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log , 2)))

    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
    silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))

    return {'d1': d1.item(), 'd2': d2.item(), 'd3': d3.item(), 'abs_rel': abs_rel.item(), 'sq_rel': sq_rel.item(), 
            'rmse': rmse.item(), 'rmse_log': rmse_log.item(), 'log10':log10.item(), 'silog':silog.item()}
    

def depth_metric(inv_depth_hw, gt_raw, min_depth=1e-5, max_depth=1e5):
    depth_hw = 1.0 / inv_depth_hw
    depth_resized = cv2.resize(depth_hw, (gt_raw.shape[1], gt_raw.shape[0]), interpolation=cv2.INTER_LINEAR)

    gt_mask = (gt_raw > min_depth) & (gt_raw < max_depth)
    gt_valid = gt_raw[gt_mask]
    depth_valid = depth_resized[gt_mask]

    align_depth_resized, s, t = align_scale_shift(depth_valid, gt_valid)
    metrics = eval_depth_metrics(torch.tensor(align_depth_resized), torch.tensor(gt_valid))

    gt_norm = normalize_depth(gt_raw)
    pred_norm = normalize_depth(depth_resized * s + t)
    return metrics, pred_norm, gt_norm