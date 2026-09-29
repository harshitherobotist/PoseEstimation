"""Run a small pose validation to check the setup."""

from pathlib import Path
from ultralytics import YOLO


def main():
    model = YOLO("yolo26n-pose.pt")
    results = model.val(
        data="coco8-pose.yaml",
        imgsz=640,
        batch=1,
        device="cpu",
        plots=False,
        verbose=False,
        project="outputs/ultralytics",
        name="smoke_test",
        exist_ok=True,
    )

    assert results.pose is not None, "Pose metrics were not returned."
    assert results.pose.map >= 0.0, "Invalid mAP value."

    print("PASS: model loaded and COCO8-Pose validation completed.")
    print(f"mAP50-95(P): {results.pose.map:.4f}")
    print(f"mAP50(P):    {results.pose.map50:.4f}")


if __name__ == "__main__":
    main()
