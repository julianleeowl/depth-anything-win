#!/usr/bin/env python3
"""
Build FP16 TensorRT engines from ONNX files using trtexec.

Converts one or more ONNX files to FP16 TensorRT engines, runs profiling,
and saves all artifacts (engine, graph JSON, profile JSON, timing JSON, logs,
and optionally SVG) into an auto-named output folder.

Usage:
    # Single ONNX file
    python export_trt_fp16.py model.onnx

    # Multiple ONNX files
    python export_trt_fp16.py a.onnx b.onnx

    # Custom optimization level and workspace
    python export_trt_fp16.py model.onnx --optimization-level 5 --workspace 12
"""

import argparse
import os
import re
import subprocess
import sys


def get_gpu_name() -> str:
    """Get GPU model number (digits only) via nvidia-smi, e.g. '5090'."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        name = result.stdout.strip().splitlines()[0].strip()
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
        print("Warning: nvidia-smi failed, using '0000' as GPU name")
        name = "0000"
    # Extract the digits portion (e.g. "NVIDIA RTX 5090" -> "5090")
    digits = re.findall(r"\d+", name)
    return digits[-1] if digits else "0000"


def run_trtexec(cmd: list, log_path: str) -> bool:
    """Run a trtexec command, saving stdout to log_path. Returns success."""
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


def build_engine(onnx_path: str, engine_path: str, graph_json_path: str,
                 build_log_path: str, workspace_gb: float,
                 optimization_level: int) -> bool:
    """Build engine from ONNX via trtexec."""
    cmd = [
        "trtexec",
        "--verbose",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16",
        f"--builderOptimizationLevel={optimization_level}",
        f"--memPoolSize=workspace:{workspace_gb}G",
        "--profilingVerbosity=detailed",
        f"--exportLayerInfo={graph_json_path}",
    ]
    print("\n[Build] Building FP16 engine...")
    return run_trtexec(cmd, build_log_path)


def profile_engine(engine_path: str, graph_json_path: str,
                   profile_json_path: str, timing_json_path: str,
                   profile_log_path: str) -> bool:
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
    return run_trtexec(cmd, profile_log_path)


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


def process_onnx(onnx_path: str, gpu_name: str, workspace_gb: float,
                 optimization_level: int):
    """Process a single ONNX file: build, profile, draw."""
    onnx_basename = os.path.splitext(os.path.basename(onnx_path))[0].replace("_", "-")
    onnx_dir = os.path.dirname(os.path.abspath(onnx_path))
    prefix = f"{onnx_basename}-{gpu_name}-fp16-lv{optimization_level}"
    out_dir = os.path.join(onnx_dir, prefix)
    os.makedirs(out_dir, exist_ok=True)

    prefix = os.path.join(out_dir, f"{onnx_basename}-{gpu_name}-fp16-lv{optimization_level}")
    engine_path = f"{prefix}.engine"
    graph_json_path = f"{prefix}.engine.graph.json"
    profile_json_path = f"{prefix}.engine.profile.json"
    timing_json_path = f"{prefix}.engine.timing.json"
    build_log_path = f"{prefix}.engine.build.log"
    profile_log_path = f"{prefix}.engine.profile.log"

    print(f"\n{'='*60}")
    print(f"ONNX:   {onnx_path}")
    print(f"GPU:    {gpu_name}")
    print(f"Output: {out_dir}/")
    print(f"Level:  {optimization_level}")
    print(f"{'='*60}")

    # Build
    if not build_engine(onnx_path, engine_path, graph_json_path,
                        build_log_path, workspace_gb, optimization_level):
        print(f"\nBuild failed for {onnx_path}. Skipping profiling.")
        return False

    # Profile
    if not profile_engine(engine_path, graph_json_path, profile_json_path,
                          timing_json_path, profile_log_path):
        print(f"\nProfiling failed for {onnx_path}. Skipping SVG generation.")
        return False

    # SVG (best-effort)
    generate_svg(graph_json_path, profile_json_path)

    print(f"\nDone. Artifacts in: {out_dir}/")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert ONNX models to FP16 TensorRT engines with profiling artifacts.",
    )
    parser.add_argument("onnx", nargs="+", help="One or more ONNX file paths")
    parser.add_argument("--workspace", type=float, default=8.0,
                        help="Workspace size in GB (default: 8.0)")
    parser.add_argument("--optimization-level", type=int, default=3,
                        help="Builder optimization level 0-5 (default: 3)")
    args = parser.parse_args()

    gpu_name = get_gpu_name()
    print(f"Detected GPU: {gpu_name}")

    results = {}
    for onnx_path in args.onnx:
        if not os.path.isfile(onnx_path):
            print(f"\nError: ONNX file not found: {onnx_path}")
            results[onnx_path] = False
            continue
        results[onnx_path] = process_onnx(
            onnx_path, gpu_name, args.workspace, args.optimization_level)

    # Summary for multiple files
    if len(args.onnx) > 1:
        print(f"\n{'='*60}")
        print("Summary:")
        for path, ok in results.items():
            status = "OK" if ok else "FAILED"
            print(f"  [{status}] {path}")
        print(f"{'='*60}")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
