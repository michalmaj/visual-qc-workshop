"""Student task: implement defect segmentation without accessing reference masks."""
import cv2
import numpy as np


def detect_defects(image_bgr: np.ndarray) -> np.ndarray:
    """Return a binary uint8 mask: 255 means defective, 0 means background.

    Args:
        image_bgr: H×W×3 uint8 OpenCV image (BGR channel order).

    Returns:
        An H×W uint8 array containing *only* the values 0 and 255.

    Requirements: deterministic, no reading dataset masks/labels, no network calls,
    no ML models downloaded at runtime. Student implementation goes here.
    """
    # TODO: ask an AI coding assistant for a strategy, review it, implement it,
    # then run tests and evaluation. This legal placeholder returns no detections.
    return np.zeros(image_bgr.shape[:2], dtype=np.uint8)
