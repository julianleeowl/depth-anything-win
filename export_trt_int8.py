#!/usr/bin/env python3
"""
TensorRT PTQ (INT8) build from ONNX with calibration images.

Input shape is read directly from the ONNX model (no --input-shape needed).

Example:
  python ptq_trt.py \
    --onnx depth_anything_v2.onnx \
    --calib-dir ./calib_images \
    --batch 8 \
    --calib-batches 50 \
    --calibrator entropy2

  Engine path is auto-generated from build parameters when --engine is omitted,
  e.g. depth_anything_v2-4090-int8-entropy2-lv3.engine
"""

import os
import re
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

import tensorrt as trt

# PyCUDA is commonly used for TRT Python calibration
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401


def get_gpu_id() -> str:
    """Return digits-only GPU identifier, e.g. '4090'."""
    name = cuda.Device(0).name()
    return re.sub(r"\D", "", name)


def run_trtexec(cmd: list, log_path: str, verbose: bool = False) -> bool:
    """Run a trtexec command, saving stdout to log_path. Returns success."""
    if verbose:
        print(f"  $ {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        with open(log_path, "w") as f:
            f.write(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        with open(log_path, "w") as f:
            f.write(e.output or "")
        print(f"  trtexec failed (exit code {e.returncode}). See {log_path}")
        return False
    except FileNotFoundError:
        print("  Error: trtexec not found. Make sure TensorRT is installed and trtexec is on PATH.")
        return False


def profile_engine(engine_path: str, graph_json_path: str,
                   profile_json_path: str, timing_json_path: str,
                   profile_log_path: str, verbose: bool = False) -> bool:
    """Profile engine via trtexec."""
    cmd = [
        "trtexec",
        "--verbose",
        "--noDataTransfers",
        "--useCudaGraph",
        "--separateProfileRun",
        "--useSpinWait",
        f"--loadEngine={engine_path}",
        f"--exportProfile={profile_json_path}",
        f"--exportTimes={timing_json_path}",
        f"--exportLayerInfo={graph_json_path}",
        "--profilingVerbosity=detailed",
    ]
    print("\n[Profile] Profiling engine...")
    return run_trtexec(cmd, profile_log_path, verbose=verbose)


def generate_svg(graph_json_path: str, profile_json_path: str):
    """Generate SVG from graph JSON (best-effort)."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))
        from draw_engine import draw_engine
        print("\n[SVG] Generating engine graph SVG...")
        draw_engine(graph_json_path, profiling_json_fname=profile_json_path)
    except Exception as e:
        print(f"\n[SVG] Could not generate SVG: {e}")
        print("  To generate manually:")
        print(f"    python utils/draw_engine.py {graph_json_path} -pj={profile_json_path}")


def list_images(calib_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    files = []
    for p in calib_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    files.sort()
    return files


def preprocess_image(
    img_path: Path,
    hw: Tuple[int, int],
    to_rgb: bool = True,
) -> np.ndarray:
    """
    IMPORTANT: You MUST match your model's training/inference preprocessing.

    This default does:
      - load image
      - resize to (W,H)
      - RGB
      - scale to [0,1]
      - CHW float32
      - ImageNet mean/std normalize (common, but may differ for your model)
    """
    H, W = hw
    img = Image.open(img_path)
    if to_rgb:
        img = img.convert("RGB")
    img = img.resize((W, H), Image.BILINEAR)

    x = np.asarray(img, dtype=np.float32) / 255.0  # HWC, [0,1]
    # HWC -> CHW
    x = np.transpose(x, (2, 0, 1))

    # ImageNet normalization (adjust if your model expects something else)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    x = (x - mean) / std

    return np.ascontiguousarray(x, dtype=np.float32)


CALIBRATOR_MAP = {
    "entropy2": trt.IInt8EntropyCalibrator2,
    "entropy": trt.IInt8EntropyCalibrator,
    "minmax": trt.IInt8MinMaxCalibrator,
}


def make_calibrator_class(calibrator_type: str):
    base = CALIBRATOR_MAP[calibrator_type]
    return type(f"ImageFolderCalibrator_{calibrator_type}", (_ImageFolderCalibratorMixin, base), {})


class _ImageFolderCalibratorMixin:
    def __init__(
        self,
        calib_dir: Path,
        input_name: str,
        input_shape: Tuple[int, int, int, int],  # NCHW (N is calibration batch)
        batch_size: int,
        max_batches: int,
        cache_file: Optional[Path] = None,
    ):
        super().__init__()

        self.calib_dir = calib_dir
        self.input_name = input_name
        self.n, self.c, self.h, self.w = input_shape
        self.batch_size = batch_size
        self.max_batches = max_batches
        self.cache_file = cache_file

        self.image_paths = list_images(calib_dir)
        if not self.image_paths:
            raise RuntimeError(f"No images found in {calib_dir}")

        # Auto-compute max_batches from available images when not specified
        if max_batches <= 0:
            self.max_batches = len(self.image_paths) // batch_size
            if self.max_batches == 0:
                raise RuntimeError(
                    f"Not enough calibration images ({len(self.image_paths)}) "
                    f"for even 1 batch of {batch_size}."
                )
        total_images = self.max_batches * batch_size
        print(f"[INFO] Calibrating with {total_images} images in "
              f"{self.max_batches} batches of {batch_size}")

        # Allocate one device buffer for a batch
        self.device_input = cuda.mem_alloc(batch_size * self.c * self.h * self.w * np.float32().nbytes)

        self.batch_idx = 0

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names: List[str]) -> Optional[List[int]]:
        if self.batch_idx >= self.max_batches:
            return None

        # Create a batch
        batch = np.empty((self.batch_size, self.c, self.h, self.w), dtype=np.float32)

        start = self.batch_idx * self.batch_size
        for i in range(self.batch_size):
            p = self.image_paths[start + i]
            batch[i] = preprocess_image(p, (self.h, self.w))

        # Copy to device
        cuda.memcpy_htod(self.device_input, batch)

        self.batch_idx += 1
        return [int(self.device_input)]

    def read_calibration_cache(self) -> Optional[bytes]:
        if self.cache_file and self.cache_file.exists():
            print(f"[INFO] Using calibration cache: {self.cache_file}")
            return self.cache_file.read_bytes()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        if self.cache_file:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            self.cache_file.write_bytes(cache)
            print(f"[INFO] Wrote calibration cache: {self.cache_file}")


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    batch_size: int,
    calib_batches: int,
    workspace_gb: float,
    calib_cache: Optional[Path],
    calibrator_type: str,
    opt_level: int,
    calib_dir: Path,
    build_log_path: Optional[Path] = None,
    verbose: bool = False,
):
    log_lines = []

    class _Logger(trt.ILogger):
        def log(self, severity, msg):
            tag = {
                trt.ILogger.INTERNAL_ERROR: "INTERNAL_ERROR",
                trt.ILogger.ERROR: "ERROR",
                trt.ILogger.WARNING: "WARNING",
                trt.ILogger.INFO: "INFO",
                trt.ILogger.VERBOSE: "VERBOSE",
            }.get(severity, "UNKNOWN")
            line = f"[TRT] [{tag}] {msg}"
            log_lines.append(line)
            if verbose or severity in (
                trt.ILogger.INTERNAL_ERROR,
                trt.ILogger.ERROR,
                trt.ILogger.WARNING,
            ):
                print(line)

    logger = _Logger()
    trt.init_libnvinfer_plugins(logger, "")

    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    builder = trt.Builder(logger)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, logger)

    onnx_bytes = onnx_path.read_bytes()
    if not parser.parse(onnx_bytes):
        print("[ERROR] ONNX parse failed:")
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise SystemExit(1)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1024**3)))
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.builder_optimization_level = opt_level

    config.set_flag(trt.BuilderFlag.FP16)

    if not builder.platform_has_fast_int8:
        print("[WARN] Platform does not report fast INT8. Will still try INT8 build.")
    config.set_flag(trt.BuilderFlag.INT8)

    # Resolve input tensor and shape from ONNX
    inp = network.get_input(0)
    input_name_resolved = inp.name
    n, c, h, w = inp.shape
    print(f"[INFO] Using input: {input_name_resolved}, shape: ({n},{c},{h},{w})")

    # Calibrator uses *batch_size* as calibration batch
    CalibratorClass = make_calibrator_class(calibrator_type)
    calibrator = CalibratorClass(
        calib_dir=calib_dir,
        input_name=input_name_resolved,
        input_shape=(batch_size, c, h, w),
        batch_size=batch_size,
        max_batches=calib_batches,
        cache_file=calib_cache,
    )
    config.int8_calibrator = calibrator

    print("[INFO] Building engine... (this can take a while)")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build serialized engine (got None).")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized_engine)
    print(f"[OK] Wrote TensorRT engine: {engine_path}")

    if build_log_path:
        build_log_path.parent.mkdir(parents=True, exist_ok=True)
        build_log_path.write_text("\n".join(log_lines) + "\n")
        print(f"[OK] Wrote build log: {build_log_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=str, required=True)
    ap.add_argument("--calib-dir", type=str, required=True)
    ap.add_argument("--engine", type=str, default="",
                    help="Output engine path (default: auto-generated in output folder)")

    ap.add_argument("--batch", type=int, default=8, help="Calibration batch size & TRT opt/max batch")
    ap.add_argument("--calib-batches", type=int, default=0, help="Number of calibration batches (default: 0 = use all images, no repeats)")
    ap.add_argument("--calibrator", type=str, default="entropy2",
                    choices=["entropy2", "entropy", "minmax"],
                    help="INT8 calibration algorithm (default: entropy2)")
    ap.add_argument("--opt-level", type=int, default=3, choices=range(6),
                    help="Builder optimization level 0-5 (default: 3)")
    ap.add_argument("--workspace-gb", type=float, default=8.0)
    ap.add_argument("--calib-cache", type=str, default="", help="Path to calibration cache file")
    ap.add_argument("--verbose", action="store_true",
                    help="Print all TRT builder logs to console (default: quiet, logs still saved to file)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip build if engine file already exists")

    args = ap.parse_args()

    onnx_path = Path(args.onnx)
    gpu_id = get_gpu_id()
    onnx_basename = onnx_path.stem.replace("_", "-")

    # Artifact prefix
    prefix = f"{onnx_basename}-{gpu_id}-int8-{args.calibrator}-lv{args.opt_level}"

    # Create output folder next to the ONNX file, named after the engine basename
    out_dir = onnx_path.parent / prefix
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.engine:
        engine_path = Path(args.engine)
    else:
        engine_path = out_dir / f"{prefix}.engine"

    if args.skip_existing and engine_path.exists():
        print(f"[SKIP] Engine already exists: {engine_path}")
        sys.exit(0)

    graph_json_path = str(out_dir / f"{prefix}.engine.graph.json")
    profile_json_path = str(out_dir / f"{prefix}.engine.profile.json")
    timing_json_path = str(out_dir / f"{prefix}.engine.timing.json")
    build_log_path = out_dir / f"{prefix}.engine.build.log"
    profile_log_path = str(out_dir / f"{prefix}.engine.profile.log")

    # Calibration cache goes into the output folder when not explicitly set
    if args.calib_cache:
        cache_path = Path(args.calib_cache)
    else:
        cache_path = out_dir / f"{prefix}.calib.cache"

    print(f"\n{'='*60}")
    print(f"ONNX:   {onnx_path}")
    print(f"GPU:    {gpu_id}")
    print(f"Output: {out_dir}/")
    print(f"Level:  {args.opt_level}")
    print(f"{'='*60}")

    # Build
    build_engine(
        onnx_path=onnx_path,
        engine_path=engine_path,
        batch_size=args.batch,
        calib_batches=args.calib_batches,
        workspace_gb=args.workspace_gb,
        calib_cache=cache_path,
        calibrator_type=args.calibrator,
        opt_level=args.opt_level,
        calib_dir=Path(args.calib_dir),
        build_log_path=build_log_path,
        verbose=args.verbose,
    )

    # Profile
    if not profile_engine(str(engine_path), graph_json_path,
                          profile_json_path, timing_json_path,
                          profile_log_path, verbose=args.verbose):
        print(f"\nProfiling failed. Skipping SVG generation.")
        sys.exit(1)

    # SVG (best-effort)
    generate_svg(graph_json_path, profile_json_path)

    print(f"\nDone. Artifacts in: {out_dir}/")
