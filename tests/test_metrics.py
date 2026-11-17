import numpy as np
from visual_qc.evaluation import segmentation_metrics


def test_perfect_match():
    m = np.array([[255, 0], [0, 255]], dtype=np.uint8)
    result = segmentation_metrics(m, m)
    assert result["f1"] == 1 and result["iou"] == 1


def test_empty_masks_are_perfect_match():
    z = np.zeros((4, 4), dtype=np.uint8)
    assert segmentation_metrics(z, z)["iou"] == 1


def test_no_prediction_when_defect_exists():
    pred = np.zeros((2, 2), dtype=np.uint8)
    gt = np.array([[255, 0], [0, 0]], dtype=np.uint8)
    assert segmentation_metrics(pred, gt)["recall"] == 0
