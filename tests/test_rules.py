"""Rule checks: the detector must not read labels or use the network.

These tests enforce the README's "Important engineering rules" for people
*and* for coding agents. They are a guard, not a sandbox: a determined
cheater can get around them, so code review still matters.
"""
import contextlib
import csv
import functools
import importlib
import os
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

import visual_qc.detector

ROOT = Path(__file__).resolve().parents[1]
with (ROOT / "data/dataset.csv").open(newline="") as f:
    SAMPLES = list(csv.DictReader(f))

DEFECT = next(e for e in SAMPLES if e["split"] == "train" and e["has_defect"] == "1")
INTACT = next(e for e in SAMPLES if e["split"] == "train" and e["has_defect"] == "0")
MASK = ROOT / DEFECT["mask"]

LABEL_DIR = os.path.realpath(ROOT / "data" / "masks")
LABEL_FILES = {os.path.realpath(ROOT / "data" / "dataset.csv")}
NETWORK_EVENTS = {"socket.connect", "socket.getaddrinfo"}
CV2_READERS = [name for name in ("imread", "imreadmulti", "imreadanimation", "imcount")
               if hasattr(cv2, name)]
_STATE = {"active": False, "violations": []}


def _is_label_path(path) -> bool:
    if not isinstance(path, (str, bytes, os.PathLike)):
        return False  # e.g. an already-open file descriptor
    real = os.path.realpath(os.fsdecode(path))
    return real in LABEL_FILES or real.startswith(LABEL_DIR + os.sep)


def _violation(message: str) -> None:
    _STATE["violations"].append(message)
    raise PermissionError(message)


def _audit_hook(event, args):
    """Python-level file and socket access (open, pathlib, numpy, pandas, urllib...)."""
    if not _STATE["active"]:
        return
    if event == "open" and args and _is_label_path(args[0]):
        _violation(f"opened label file {args[0]}")
    elif event in NETWORK_EVENTS:
        _violation(f"network access ({event})")


# Audit hooks cannot be removed; this one does nothing outside guard().
sys.addaudithook(_audit_hook)


def _checked_cv2_reader(original):
    """cv2 reads files in C++, bypassing audit hooks, so wrap its readers."""
    @functools.wraps(original)
    def reader(filename, *args, **kwargs):
        if _STATE["active"] and _is_label_path(filename):
            _violation(f"cv2.{original.__name__} read label file {filename}")
        return original(filename, *args, **kwargs)
    return reader


@contextlib.contextmanager
def guard(monkeypatch):
    """Record (and block) label-file reads and network access inside the block."""
    violations: list[str] = []
    with monkeypatch.context() as patch:
        for name in CV2_READERS:
            patch.setattr(cv2, name, _checked_cv2_reader(getattr(cv2, name)))
        _STATE.update(active=True, violations=violations)
        try:
            yield violations
        finally:
            _STATE["active"] = False


def _load(entry):
    return cv2.imread(str(ROOT / entry["image"]), cv2.IMREAD_COLOR)


READERS = {
    "open": lambda: open(MASK, "rb").read(),
    "pathlib": lambda: MASK.read_bytes(),
    "numpy": lambda: np.fromfile(MASK, dtype=np.uint8),
    "pandas": lambda: pd.read_csv(ROOT / "data" / "dataset.csv"),
    "cv2": lambda: cv2.imread(str(MASK), cv2.IMREAD_GRAYSCALE),
    "path_trick": lambda: open(ROOT / "data" / "images" / ".." / "masks" / MASK.name, "rb").read(),
    "network": lambda: urllib.request.urlopen("http://example.com", timeout=1),
}


@pytest.mark.parametrize("reader", READERS.values(), ids=READERS.keys())
def test_guard_detects_label_and_network_access(monkeypatch, reader):
    with guard(monkeypatch) as violations:
        try:
            reader()
        except Exception:
            pass  # a cheating detector could swallow the error; the record remains
    assert violations, "the guard missed a forbidden access"


def test_guard_allows_reading_other_files(monkeypatch):
    with guard(monkeypatch) as violations:
        image = cv2.imread(str(ROOT / DEFECT["image"]), cv2.IMREAD_COLOR)
        (ROOT / "pyproject.toml").read_text()
    assert image is not None
    assert violations == []


def test_detector_respects_data_rules(monkeypatch):
    images = [_load(DEFECT), _load(INTACT)]
    with guard(monkeypatch) as violations:
        try:
            # Reload under the guard so reads at import time or cached reads count too.
            module = importlib.reload(visual_qc.detector)
            for image in images:
                module.detect_defects(image)
        except Exception:
            if not violations:
                raise
    assert not violations, (
        "Rule broken (README, 'Important engineering rules'): detect_defects() must use "
        "only its image argument - no reference masks, no dataset.csv, no network.\n"
        "Detected: " + "; ".join(violations)
    )


def test_detector_is_deterministic():
    image = _load(DEFECT)
    first = visual_qc.detector.detect_defects(image.copy())
    second = visual_qc.detector.detect_defects(image.copy())
    assert np.array_equal(first, second), "same input must give the same mask"
