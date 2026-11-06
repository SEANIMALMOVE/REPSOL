"""Redraw the MobileNetV3-02 confusion matrices for the paper, with category names.

Usage: python -m src.make_confusion_figures outputs/evaluation/cache/MobileNetV3_02.npz outputs/paper
The .npz is outputs/evaluation/cache/MobileNetV3_02.npz from the REPSOL repo.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

NAMES = [
    "0 No pattern",
    "1 Pulses",
    "2 Marked cycles",
    "3 Continuous & 3.15 kHz (high)",
    "4 Continuous & 3.15 kHz (low)",
    "5 Blasts",
    "6 Machinery continuous",
    "7 Works, sirens & knocks",
]


def draw(matrix, fmt, out_path, colorbar_label):
    fig, ax = plt.subplots(figsize=(8, 7))
    norm = matrix / matrix.sum(axis=1, keepdims=True)  # colour always by row share
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(8):
        for j in range(8):
            value = matrix[i, j]
            text = fmt(value)
            colour = "white" if norm[i, j] > 0.5 else "black"
            ax.text(j, i, text, ha="center", va="center", color=colour, fontsize=10)
    ax.set_xticks(range(8), NAMES, rotation=45, ha="right")
    ax.set_yticks(range(8), NAMES)
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("Expert label")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    npz_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(npz_path)
    cm = np.zeros((8, 8), dtype=int)
    for t, p in zip(data["y_true"], data["y_pred"]):
        cm[t, p] += 1

    draw(cm, lambda v: str(v), out_dir / "confusion_matrix_counts.png",
         "Share of expert-labelled clips")
    share = cm / cm.sum(axis=1, keepdims=True)
    draw(share, lambda v: "0" if v == 0 else f"{v:.2f}",
         out_dir / "confusion_matrix_normalised.png", "Share of expert-labelled clips")
    print(cm)


if __name__ == "__main__":
    main()
