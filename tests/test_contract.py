import csv
from pathlib import Path
import cv2
import numpy as np
import pytest
from visual_qc.detector import detect_defects

ROOT = Path(__file__).resolve().parents[1]
with (ROOT / "data/dataset.csv").open(newline="") as f:
    SAMPLES = list(csv.DictReader(f))


@pytest.mark.parametrize("entry", SAMPLES[:5], ids=lambda x: x["id"])
def test_detector_contract(entry):
    img = cv2.imread(str(ROOT / entry["image"]), cv2.IMREAD_COLOR)
    mask = detect_defects(img)
    assert isinstance(mask, np.ndarray)
    assert mask.shape == img.shape[:2]
    assert mask.dtype == np.uint8
    assert np.isin(mask, [0, 255]).all()


def test_detector_finds_at_least_something_on_known_defect():
    entry = next(e for e in SAMPLES if e["split"] == "train" and e["has_defect"] == "1")
    img = cv2.imread(str(ROOT / entry["image"]), cv2.IMREAD_COLOR)
    assert np.count_nonzero(detect_defects(img)) > 0, "TODO: implement defect detection"
