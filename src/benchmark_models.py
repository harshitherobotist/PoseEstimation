"""Evaluate several pose models on the same validation set."""

import argparse
import csv
import json
from pathlib import Path
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--models",
        nargs="+",
        default=["yolo26n-pose.pt", "yolo26s-pose.pt", "yolo26m-pose.pt"],
    )
    p.add_argument("--data", default="coco-pose.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--output", default="outputs/metrics/model_comparison.csv")
    args = p.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for model_name in args.models:
        print(f"\n===== {model_name} =====")
        model = YOLO(model_name)
        r = model.val(
            data=args.data,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            plots=False,
            save_json=False,
            project="outputs/ultralytics",
            name=f"benchmark_{Path(model_name).stem}",
            exist_ok=True,
        )

        try:
            params = sum(p.numel() for p in model.model.parameters())
        except Exception:
            params = None

        row = {
            "model": model_name,
            "imgsz": args.imgsz,
            "pose_mAP50_95": float(r.pose.map),
            "pose_mAP50": float(r.pose.map50),
            "pose_mAP75": float(r.pose.map75),
            "parameters": int(params) if params else None,
        }
        rows.append(row)

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(rows, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
