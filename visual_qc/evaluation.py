"""Metrics utilities are part of the trusted workshop evaluation harness."""
import numpy as np


def segmentation_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: {pred.shape} != {gt.shape}")
    if pred.ndim != 2 or gt.ndim != 2:
        raise ValueError("expected 2D masks")
    if pred.dtype != np.uint8 or gt.dtype != np.uint8:
        raise ValueError("expected uint8 masks")
    if not np.isin(pred, (0, 255)).all():
        raise ValueError("prediction contains non-binary values")
    p, g = pred > 0, gt > 0
    tp = np.count_nonzero(p & g)
    fp = np.count_nonzero(p & ~g)
    fn = np.count_nonzero(~p & g)
    # Empty ground truth and empty prediction = perfect match.
    precision = float(tp / (tp + fp)) if (tp + fp) else (1.0 if not fn else 0.0)
    recall = float(tp / (tp + fn)) if (tp + fn) else 1.0
    f1 = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 1.0
    iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) else 1.0
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}
