#!/usr/bin/env python3
"""
Batch runner for export_trt_fp16.py and export_trt_int8.py.

Finds all .onnx files under an input folder and runs both export scripts
with all combinations of optimization levels (and calibrator methods for INT8).

Examples:
    # Defaults: opt levels [3, 5], all three calibrators
    python run_all_exports.py ./model-weights

    # Custom opt levels and calibrators
    python run_all_exports.py ./model-weights --opt-levels 3 --calibrators entropy2 minmax

    # INT8 args forwarded to export_trt_int8.py
    python run_all_exports.py ./model-weights --calib-dir ./calib_images --batch 16
"""

import argparse
import subprocess
import sys
from itertools import product
from pathlib import Path
from tqdm import tqdm

ALL_CALIBRATORS = ["entropy2", "entropy", "minmax"]


def find_onnx_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.rglob("*.onnx"))
    return files


def run_cmd(cmd: list[str], dry_run: bool = False) -> bool:
    print(f"\n{'─'*70}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'─'*70}")
    if dry_run:
        print("  [dry-run] skipped")
        return True
    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Run export_trt_fp16.py and export_trt_int8.py for all ONNX files with all arg combinations.",
    )
    parser.add_argument("input_dir", type=str, help="Directory to search for .onnx files")
    parser.add_argument("--opt-levels", type=int, nargs="+", default=[3, 5],
                        help="Optimization levels to try (default: 3 5)")
    parser.add_argument("--calibrators", type=str, nargs="+", default=ALL_CALIBRATORS,
                        choices=ALL_CALIBRATORS,
                        help="INT8 calibrator methods (default: all three)")

    # Forwarded to both scripts
    parser.add_argument("--workspace", type=float, default=8.0,
                        help="Workspace size in GB (default: 8.0)")

    # Forwarded to INT8 only
    parser.add_argument("--calib-dir", type=str, required=True,
                        help="Calibration images directory (required for INT8)")
    parser.add_argument("--batch", type=int, default=8,
                        help="Calibration batch size (default: 8)")
    parser.add_argument("--calib-batches", type=int, default=0,
                        help="Number of calibration batches (default: 0 = all images)")

    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")

    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    fp16_script = script_dir / "export_trt_fp16.py"
    int8_script = script_dir / "export_trt_int8.py"

    onnx_files = find_onnx_files(Path(args.input_dir))
    if not onnx_files:
        print(f"No .onnx files found under {args.input_dir}")
        sys.exit(1)

    print(f"Found {len(onnx_files)} ONNX file(s):")
    for f in onnx_files:
        print(f"  {f}")
    print(f"Optimization levels: {args.opt_levels}")
    print(f"Calibrators (INT8):  {args.calibrators}")

    total_fp16 = len(onnx_files) * len(args.opt_levels)
    total_int8 = len(onnx_files) * len(args.opt_levels) * len(args.calibrators)
    total_runs = total_fp16 + total_int8
    print(f"Total runs: {total_fp16} FP16 + {total_int8} INT8 = {total_runs}")

    results = []

    # --- FP16 ---
    with tqdm(total=total_runs, desc="Exporting engines", unit="engine") as pbar:
        for onnx_path, opt_level in product(onnx_files, args.opt_levels):
            label = f"FP16 | {onnx_path.name} | lv{opt_level}"
            pbar.set_description(f"Exporting {label}")
            cmd = [
                sys.executable, str(fp16_script),
                str(onnx_path),
                "--optimization-level", str(opt_level),
                "--workspace", str(args.workspace),
            ]
            ok = run_cmd(cmd, dry_run=args.dry_run)
            results.append((label, ok))
            pbar.update(1)

        # --- INT8 ---
        for onnx_path, opt_level, calibrator in product(onnx_files, args.opt_levels, args.calibrators):
            label = f"INT8 | {onnx_path.name} | lv{opt_level} | {calibrator}"
            pbar.set_description(f"Exporting {label}")
            cmd = [
                sys.executable, str(int8_script),
                "--onnx", str(onnx_path),
                "--opt-level", str(opt_level),
                "--workspace-gb", str(args.workspace),
                "--calib-dir", args.calib_dir,
                "--batch", str(args.batch),
                "--calib-batches", str(args.calib_batches),
                "--calibrator", calibrator,
            ]
            ok = run_cmd(cmd, dry_run=args.dry_run)
            results.append((label, ok))
            pbar.update(1)

    # --- Summary ---
    print(f"\n{'='*70}")
    print("Summary:")
    print(f"{'='*70}")
    for label, ok in results:
        status = "OK" if ok else "FAILED"
        print(f"  [{status:>6}] {label}")

    failed = sum(1 for _, ok in results if not ok)
    print(f"\n{len(results) - failed}/{len(results)} succeeded.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
