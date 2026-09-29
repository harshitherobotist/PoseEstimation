"""Calculate keypoint localization metrics."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

JOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def evaluate(pred, gt):
    assert pred.shape[:2] == gt.shape[:2]
    assert pred.shape[1] == 17

    xy = gt[..., :2]
    vis = gt[..., 2] > 0

    x_min = np.where(vis, xy[..., 0], np.nan).min(axis=1)
    x_max = np.where(vis, xy[..., 0], np.nan).max(axis=1)
    y_min = np.where(vis, xy[..., 1], np.nan).min(axis=1)
    y_max = np.where(vis, xy[..., 1], np.nan).max(axis=1)
    scale = np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)
    scale = np.maximum(scale, 1.0)

    dist = np.linalg.norm(pred - xy, axis=-1)
    norm_dist = dist / scale[:, None]

    metrics = {}
    for threshold in [0.05, 0.10, 0.20]:
        metrics[f"PCK@{threshold:.2f}"] = float(
            np.mean(norm_dist[vis] <= threshold)
        )

    metrics["MNKE"] = float(np.mean(norm_dist[vis]))
    metrics["mean_pixel_error"] = float(np.mean(dist[vis]))

    rows = []
    for k, name in enumerate(JOINTS):
        valid = vis[:, k]
        rows.append({
            "joint": name,
            "visible_samples": int(valid.sum()),
            "mean_pixel_error": float(np.mean(dist[valid, k])) if valid.any() else np.nan,
            "PCK@0.05": float(np.mean(norm_dist[valid, k] <= 0.05)) if valid.any() else np.nan,
            "PCK@0.10": float(np.mean(norm_dist[valid, k] <= 0.10)) if valid.any() else np.nan,
            "PCK@0.20": float(np.mean(norm_dist[valid, k] <= 0.20)) if valid.any() else np.nan,
        })

    return metrics, pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True)
    p.add_argument("--gt", required=True)
    p.add_argument("--out", default="outputs/metrics/custom")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pred = np.load(args.pred)
    gt = np.load(args.gt)

    metrics, joints = evaluate(pred, gt)
    (out / "summary.json").write_text(
        __import__("json").dumps(metrics, indent=2)
    )
    joints.to_csv(out / "pck_by_joint.csv", index=False)

    plt.figure(figsize=(12, 5))
    plt.bar(joints["joint"], joints["PCK@0.10"])
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("PCK@0.10")
    plt.title("Per-joint pose accuracy")
    plt.tight_layout()
    plt.savefig(out / "pck_by_joint.png", dpi=200)
    plt.close()

    print(__import__("json").dumps(metrics, indent=2))
    print(f"Saved metrics to {out}")


if __name__ == "__main__":
    main()
