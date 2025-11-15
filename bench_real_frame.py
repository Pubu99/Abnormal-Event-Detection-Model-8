#!/usr/bin/env python3
"""
Real-frame YOLO benchmark helper for this project.

Usage:
    python3 bench_real_frame.py --frame test_frame.jpg --runs 50 --mode detect --fast

This script loads a real image, constructs the AnomalyDetector (with fast-mode toggle),
warms up the detector, runs N timed inferences using `yolo_infer_frame`, and prints
mean/median/p95/min/max and approximate FPS.

Note: keep model loaded on GPU to avoid counting model load time.
"""
import time
import argparse
import numpy as np
import cv2
from pathlib import Path

from inference.engine import AnomalyDetector


def main():
    parser = argparse.ArgumentParser(description="Benchmark YOLO on a real frame")
    parser.add_argument('--frame', required=True, help='Path to a representative image')
    parser.add_argument('--model', default='models/best_model.pth', help='Path to research model (keeps API consistent)')
    parser.add_argument('--yolo', default='yolov8n.pt', help='YOLO model file')
    parser.add_argument('--device', default='cuda', help='Device: cuda or cpu')
    parser.add_argument('--runs', type=int, default=50, help='Number of timed runs')
    parser.add_argument('--warmup', type=int, default=3, help='Number of warmup runs (not timed)')
    parser.add_argument('--mode', choices=['detect', 'track'], default='detect', help='YOLO API to benchmark')
    parser.add_argument('--fast', action='store_true', help='Enable fast_mode (smaller imgsz, detect pref)')
    parser.add_argument('--yolo-imgsz', type=int, default=640, help='YOLO imgsz when not in fast mode')
    parser.add_argument('--yolo-fast-imgsz', type=int, default=320, help='YOLO imgsz when in fast mode')
    args = parser.parse_args()

    frame_path = Path(args.frame)
    if not frame_path.exists():
        print(f"ERROR: frame not found: {frame_path}")
        return

    # Load image
    frame = cv2.imread(str(frame_path))
    if frame is None:
        print(f"ERROR: cv2 failed to read image: {frame_path}")
        return

    print("Initializing detector (this may take a few seconds)...")
    det = AnomalyDetector(
        model_path=args.model,
        yolo_model=args.yolo,
        device=args.device,
        fast_mode=bool(args.fast),
        yolo_mode='detect' if args.fast else 'detect',
        yolo_fast_imgsz=int(args.yolo_fast_imgsz),
        yolo_imgsz=int(args.yolo_imgsz),
    )

    print(f"Warmup: {args.warmup} runs...")
    for i in range(args.warmup):
        _ = det.yolo_infer_frame(frame, mode=args.mode)

    timings = []
    obj_counts = []
    print(f"Timing {args.runs} runs (mode={args.mode})...")
    for i in range(args.runs):
        t0 = time.time()
        res = det.yolo_infer_frame(frame, mode=args.mode)
        t1 = time.time()
        timings.append(t1 - t0)
        if res is None:
            obj_counts.append(0)
        else:
            try:
                obj_counts.append(len(res.boxes))
            except Exception:
                # Fallback: try to convert to dict
                try:
                    d = det._convert_yolo_results(res, scale=1.0)
                    obj_counts.append(len(d.get('objects', [])))
                except Exception:
                    obj_counts.append(0)

    timings = np.array(timings)
    stats = {
        'mode': args.mode,
        'runs': int(args.runs),
        'mean_s': float(timings.mean()),
        'median_s': float(np.median(timings)),
        'p95_s': float(np.percentile(timings, 95)),
        'min_s': float(timings.min()),
        'max_s': float(timings.max()),
        'fps_approx': float(1.0 / timings.mean()) if timings.mean() > 0 else float('inf'),
        'mean_objects': float(np.mean(obj_counts))
    }

    print("\n=== YOLO Real-Frame Benchmark ===")
    for k, v in stats.items():
        print(f"{k}: {v}")

    print("\nTip: run `nvidia-smi -l 1` in another terminal to watch GPU utilization during the benchmark.")


if __name__ == '__main__':
    main()
