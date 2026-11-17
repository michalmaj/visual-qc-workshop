"""Small EDA exercise: generate a plot from the audit CSV."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("results/test/metrics.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/test/summary.png"))
    args = parser.parse_args()
    df = pd.read_csv(args.csv)
    print(df[["precision", "recall", "f1", "iou", "latency_ms"]].describe().round(3).to_string())
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].hist(df["iou"], bins=10)
    axs[0].set(xlabel="IoU", ylabel="Number of images", title="Segmentation quality")
    axs[1].scatter(df["latency_ms"], df["iou"], alpha=0.75)
    axs[1].set(xlabel="Median runtime per image (ms)", ylabel="IoU", title="Accuracy vs runtime")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"Plot: {args.out}")


if __name__ == "__main__":
    main()
