"""Evaluate image segmentation and collect per-image execution timings."""
from __future__ import annotations
import argparse
import csv
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from .detector import detect_defects
from .evaluation import segmentation_metrics

BASE = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "test", "all"), default="test")
    parser.add_argument("--out", type=Path, default=Path("results/test"))
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")
    dest = args.out
    (dest / "predictions").mkdir(parents=True, exist_ok=True)
    rows = []
    with (BASE / "data" / "dataset.csv").open(newline="") as handle:
        dataset = list(csv.DictReader(handle))
    for entry in dataset:
        if args.split != "all" and entry["split"] != args.split:
            continue
        image = cv2.imread(str(BASE / entry["image"]), cv2.IMREAD_COLOR)
        gt = cv2.imread(str(BASE / entry["mask"]), cv2.IMREAD_GRAYSCALE)
        if image is None or gt is None:
            raise RuntimeError(f"Cannot read dataset sample {entry['id']}")
        # One untimed warmup; isolate console and disk writes from timing.
        detect_defects(image.copy())
        durations_ms: list[float] = []
        for _ in range(args.repeats):
            inp = image.copy()  # same input; defensive against accidental modification
            start = time.perf_counter_ns()
            pred = detect_defects(inp)
            durations_ms.append((time.perf_counter_ns() - start) / 1_000_000)
        m = segmentation_metrics(pred, gt)
        cv2.imwrite(str(dest / "predictions" / f"{entry['id']}.png"), pred)
        rows.append({"id":entry["id"], "split":entry["split"],
                     "has_defect":entry["has_defect"], "latency_ms":statistics.median(durations_ms), **m})
    if not rows:
        raise ValueError("No samples selected")
    with (dest / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Samples: {len(rows)} | mean IoU: {statistics.mean(r['iou'] for r in rows):.3f} | "
          f"mean F1: {statistics.mean(r['f1'] for r in rows):.3f} | "
          f"median latency: {statistics.median(r['latency_ms'] for r in rows):.3f} ms")
    print(f"CSV: {dest / 'metrics.csv'}")
    print("Warning: these are exploratory timings, not controlled microbenchmarks.")


if __name__ == "__main__":
    main()
