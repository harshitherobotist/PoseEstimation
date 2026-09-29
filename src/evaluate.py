"""
Full COCO-Pose evaluation.

Runs Ultralytics' official pose validation and additionally benchmarks
runtime. Ultralytics handles COCO/OKS AP evaluation; this script exports
a compact, machine-readable experiment summary.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolo26n-pose.pt")
    p.add_argument("--data", default="coco-pose.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--runtime-images", type=int, default=100)
    p.add_argument("--output", default="outputs/metrics/evaluation_metrics.json")
    return p.parse_args()


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def main():
    args = parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)

    results = model.val(
        data=args.data,
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
        conf=args.conf,
        plots=True,
        save_json=True,
        project="outputs/ultralytics",
        name="validation",
        exist_ok=True,
    )

    metrics = {
        "model": args.model,
        "dataset": args.data,
        "imgsz": args.imgsz,
        "device": args.device,
        "batch": args.batch,
        "pose_mAP50_95": safe_float(results.pose.map),
        "pose_mAP50": safe_float(results.pose.map50),
        "pose_mAP75": safe_float(results.pose.map75),
    }

    # Measure inference time separately from validation.
    source = None
    try:
        source = results.save_dir.parent.parent / "images"
    except Exception:
        source = None

    sample = "https://ultralytics.com/images/bus.jpg"
    warmup = 5
    times = []

    for i, result in enumerate(
        model.predict(
            source=sample,
            imgsz=args.imgsz,
            device=args.device,
            conf=args.conf,
            verbose=False,
            stream=True,
        )
    ):
        if i < warmup:
            continue
        times.append(float(result.speed.get("inference", np.nan)))
        if len(times) >= args.runtime_images:
            break

    if times:
        arr = np.asarray(times, dtype=float)
        metrics["runtime"] = {
            "n": int(len(arr)),
            "mean_inference_ms": float(np.nanmean(arr)),
            "median_inference_ms": float(np.nanmedian(arr)),
            "std_inference_ms": float(np.nanstd(arr)),
            "fps_from_mean_inference": float(1000.0 / np.nanmean(arr)),
        }

    try:
        metrics["parameters"] = int(sum(p.numel() for p in model.model.parameters()))
    except Exception:
        metrics["parameters"] = None

    out.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
