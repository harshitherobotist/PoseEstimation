"""Run pose inference and save the results."""

import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolo26n-pose.pt")
    p.add_argument("--source", required=True)
    p.add_argument("--device", default="0")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--project", default="outputs/predictions")
    args = p.parse_args()

    model = YOLO(args.model)
    model.predict(
        source=args.source,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        save=True,
        project=args.project,
        name="pose_results",
        exist_ok=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
