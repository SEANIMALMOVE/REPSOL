"""Paper-ready appendix figures for the REPSOL deep learning section.

Usage: python -m src.make_appendix_figures . outputs/paper

Reads only saved results (training histories + evaluation caches), no retraining:
  Figure X1  learning_curves_paper.png  train/val accuracy per model
  Figure X2  f1_heatmap_paper.png       per-category F1 of every model
  Figure X3  roc_paper.png              one-vs-rest ROC of the selected model
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

CATEGORIES = [
    "0 No pattern",
    "1 Pulses",
    "2 Marked cycles",
    "3 Continuous & 3.15 kHz (high)",
    "4 Continuous & 3.15 kHz (low)",
    "5 Blasts",
    "6 Machinery continuous",
    "7 Works, sirens & knocks",
]

# (paper name, training history file, evaluation cache file)
MODELS = [
    ("MobileNetV3-Large", "mobilenetv3_02_best_01", "MobileNetV3_02"),
    ("EfficientNet-B0", "efficientnet_04_best_01", "EfficientNet_04"),
    ("ResNet-50", "resnet50_02_best_01", "ResNet50_02"),
    ("EfficientNet-B0\n(no regularisation)", "efficientnet_05_best_01", "EfficientNet_05"),
    ("EfficientNet-B0\n(Watkins initialisation)", "pretrained_02_best_01", "PreTrained_02"),
]

# Reference palette (dataviz skill): categorical slots in fixed order, blue ramp
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE_RAMP = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
             "#256abf", "#184f95", "#0d366b"]
TEXT = "#2b2b2b"
MUTED = "#8a8a85"

plt.rcParams.update({
    "font.size": 10,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_cache(repo, name):
    d = np.load(repo / "outputs" / "evaluation" / "cache" / f"{name}.npz")
    return d["y_true"], d["y_pred"], d["y_probs"]


def per_class_f1(y_true, y_pred, n=8):
    f1 = []
    for c in range(n):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        denom = 2 * tp + fp + fn
        f1.append(2 * tp / denom if denom else 0.0)
    return np.array(f1)


def roc_curve(y_bin, scores):
    order = np.argsort(-scores, kind="mergesort")
    y = y_bin[order]
    s = scores[order]
    distinct = np.where(np.diff(s))[0]
    idx = np.r_[distinct, y.size - 1]
    tps = np.cumsum(y)[idx]
    fps = (idx + 1) - tps
    tpr = np.r_[0, tps / y.sum()]
    fpr = np.r_[0, fps / (y.size - y.sum())]
    return fpr, tpr, np.trapezoid(tpr, fpr)


def learning_curves(repo, out):
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), sharey=True)
    axes = axes.ravel()
    for ax, (label, hist, _) in zip(axes, MODELS):
        h = np.genfromtxt(repo / "Models_output" / f"{hist}_training_history.csv",
                          delimiter=",", names=True)
        ax.plot(h["epoch"], h["train_acc"], color=CATEGORICAL[0], lw=2, label="Training")
        ax.plot(h["epoch"], h["val_acc"], color=CATEGORICAL[1], lw=2, label="Validation")
        sel = "val_macro_f1" if "val_macro_f1" in h.dtype.names else "val_acc"
        best = h[np.argmax(h[sel])]
        ax.plot(best["epoch"], best["val_acc"], "o", ms=8, color=CATEGORICAL[1],
                markeredgecolor="white", markeredgewidth=2, zorder=3)
        ax.set_title(label, fontsize=10, color=TEXT)
        ax.set_ylim(0, 100)
        ax.set_xlim(0, 31)
        ax.grid(axis="y", color="#e6e5e0", lw=0.8)
        ax.set_xlabel("Epoch")
    for ax in axes[::3]:
        ax.set_ylabel("Accuracy (%)")
    legend_ax = axes[5]
    legend_ax.axis("off")
    handles = [
        plt.Line2D([], [], color=CATEGORICAL[0], lw=2, label="Training accuracy"),
        plt.Line2D([], [], color=CATEGORICAL[1], lw=2, label="Validation accuracy"),
        plt.Line2D([], [], color=CATEGORICAL[1], marker="o", ms=8, lw=0,
                   markeredgecolor="white", markeredgewidth=2, label="Selected checkpoint"),
    ]
    legend_ax.legend(handles=handles, loc="center", frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "learning_curves_paper.png", dpi=300)
    plt.close(fig)


def f1_heatmap(repo, out):
    scores = np.column_stack([per_class_f1(*load_cache(repo, c)[:2]) for _, _, c in MODELS])
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    im = ax.imshow(scores, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    wrapped = [m[0].replace("\n(", "\n(").replace(" initialisation", "\ninitialisation")
               .replace("(no regularisation)", "(no\nregularisation)") for m in MODELS]
    ax.set_xticks(range(len(MODELS)), wrapped)
    ax.set_yticks(range(8), CATEGORIES)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(MODELS)), minor=True)
    ax.set_yticks(np.arange(-0.5, 8), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="minor", length=0)
    for i in range(8):
        for j in range(len(MODELS)):
            v = scores[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > 0.55 else TEXT, fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("F1 score")
    cbar.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "f1_heatmap_paper.png", dpi=300)
    plt.close(fig)


def roc_best(repo, out):
    y_true, _, probs = load_cache(repo, MODELS[0][2])
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=MUTED)
    aucs = []
    for c in range(8):
        fpr, tpr, auc = roc_curve((y_true == c).astype(int), probs[:, c])
        aucs.append(auc)
        ax.plot(fpr, tpr, lw=2, color=CATEGORICAL[c],
                label=f"{CATEGORIES[c]} (AUC {auc:.2f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.grid(color="#e6e5e0", lw=0.8)
    ax.legend(loc="lower right", fontsize=9, frameon=False,
              title=f"Macro AUC {np.mean(aucs):.2f}", title_fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "roc_paper.png", dpi=300)
    plt.close(fig)
    return np.mean(aucs)


def main():
    repo, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    learning_curves(repo, out)
    f1_heatmap(repo, out)
    print("macro AUC", round(roc_best(repo, out), 3))


if __name__ == "__main__":
    main()
