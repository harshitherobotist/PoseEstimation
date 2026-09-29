import numpy as np
from src.custom_metrics import evaluate


def test_perfect_prediction():
    gt = np.zeros((2, 17, 3), dtype=float)
    gt[..., 0] = np.arange(17)
    gt[..., 1] = np.arange(17)
    gt[..., 2] = 2

    pred = gt[..., :2].copy()
    metrics, joints = evaluate(pred, gt)

    assert metrics["PCK@0.05"] == 1.0
    assert metrics["PCK@0.10"] == 1.0
    assert metrics["PCK@0.20"] == 1.0
    assert metrics["MNKE"] == 0.0
