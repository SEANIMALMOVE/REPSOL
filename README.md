# REPSOL Industrial Sound Classification Project

A deep learning project for classifying industrial sounds from REPSOL facilities using spectrogram-based audio analysis. Eight sound categories, four CNN architectures compared, with a full ablation study separating the effect of a critical data-pipeline fix from model/training improvements.

**Best model: MobileNetV3-Large — 86.5% test accuracy, 0.83 macro-F1** (up from a 70% baseline).

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [The Data Pipeline Fix (key finding)](#-the-data-pipeline-fix-key-finding)
- [Models & Results](#-models--results)
- [Findings from Evaluation](#-findings-from-evaluation)
- [Which Model to Use When](#-which-model-to-use-when)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Next Steps](#-next-steps)

## 🎯 Project Overview

This project classifies 60-second industrial acoustic recordings into 8 categories using transfer-learning CNNs on spectrogram images. Intended uses: predictive maintenance, anomaly detection, and pre-labeling large volumes of unlabeled monitoring audio (model predicts + human reviews low-confidence cases).

### Sound Categories

| # | Class | Train / Val / Test samples |
|---|-------|---------------------------|
| 0 | ActividadBASE_NO pattern activity (baseline/background) | 158 / 34 / 33 |
| 1 | Pulses HAMMERING | 30 / 7 / 6 |
| 2 | Marked cycles 3 segundos | 92 / 20 / 20 |
| 3 | Continuous activity & tone 3.15kHz — SPL ALTO | 492 / 105 / 106 |
| 4 | Continuous activity & tone 3.15kHz — SPL BAJO | 178 / 38 / 39 |
| 5 | Blasts | 51 / 11 / 11 |
| 6 | Machinery continuous activity | 352 / 75 / 76 |
| 7 | Works, sirens & knocks (high-frequency bursts at 3.15 kHz) | 29 / 6 / 6 |

1,975 samples total, 70/15/15 stratified split, **17× class imbalance** (492 vs 29 training samples).

## 📊 Dataset

### Audio specifications
- **Sample rate**: 96,000 Hz · **Duration**: 60 s per clip · **Original format**: WAV
- Source data arrives as pre-rendered MATLAB spectrogram images (`*_spectrogram_win16384.png`, win=16384, overlap=90%, nfft=32768, log-frequency axis 20 Hz–15 kHz)

### Structure
```
Data/
├── Annotations/
│   ├── audio_annotations.csv     # Metadata for all audio files
│   ├── train.csv / val.csv / test.csv
├── Spectrograms/                 # LEGACY: full-figure renders (3×1714×3156, ~65 MB each)
│   ├── train/ val/ test/         #   kept only as re-conversion source
└── Spectrograms_224/             # CURRENT: plot interior, 224×224, z-scored (~600 KB each)
    ├── train/ val/ test/         #   all current models train/evaluate on this
```

## 🚨 The Data Pipeline Fix (key finding)

**Every model up to and including EfficientNet-03 trained on ~13% of each spectrogram without anyone noticing.**

What happened:
1. The stored `.norm.pt` tensors were z-scored RGB renders of **entire MATLAB figures** (3×1714×3156): title text (containing the source filename), axis labels, and colorbar included.
2. `dataset.py` cropped every sample to a fixed width of **400 pixels** — a setting written for real mel-spectrogram matrices (~128×400) but applied to 3,156-pixel-wide figure screenshots. The model saw only the leftmost strip: white margin, the rotated "Frequency (Hz)" label, and roughly the **first 4 seconds** of a 60-second recording.
3. The 65 MB files also meant ~90 GB of disk reads per epoch → **~50 min/epoch on CPU**.

The fix (`src/preprocess/downsize_spectrograms.py`): crop the plot interior (rows 54:1578, cols 209:2969 — pixel-identical across all files), resize to 224×224 with antialiasing, re-z-score, save to `Data/Spectrograms_224`. Result: full time axis visible, no text/margins (removes a filename-shortcut-learning risk), 100× smaller files, **~4–5 min/epoch** — a ~10× faster experiment cycle.

**Measured impact (controlled ablation, EfficientNet-02 vs -05, identical config):** 70% → 81.1% test accuracy from the data fix alone, with zero hyperparameter changes.

## 🧠 Models & Results

All current models: ImageNet-pretrained backbones fine-tuned on `Spectrograms_224`. The **full anti-overfitting package** = SpecAugment (2 frequency + 2 time masks, train only), partial backbone freezing (frozen-block BatchNorm kept in eval mode), dropout 0.35, AdamW weight decay 1e-2, balanced class weights + ×1.5 boost for classes 1 & 7, label smoothing 0.1, batch 16, checkpoint & early-stop on **validation macro-F1**, ReduceLROnPlateau.

### Main results (fixed dataset, test set — 297 samples)

| Model | Recipe | Params (trainable) | Test acc | Macro-F1 | Weighted-F1 |
|---|---|---|---|---|---|
| **MobileNetV3-02** 🏆 | full package (freeze features 0–12) | 5.4M (81%) | **86.5%** | **0.83** | **0.87** |
| EfficientNet-04 | full package (freeze features 0–5) | 4.0M (79%) | 83.2% | 0.78 | 0.83 |
| EfficientNet-05 | run-02 baseline config (ablation control) | 4.0M (100%) | 81.1% | 0.82 | 0.81 |
| ResNet50-02 | full package (layer4 + fc only) | 23.5M (64%) | 80.8% | 0.73 | 0.82 |
| PreTrained-02 | Watkins warm-start + full package | 4.0M (79%) | 27.3% ❌ | 0.26 | 0.26 |

### Per-class F1 (best model, MobileNetV3-02)

| Class | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| F1 | 0.88 | 0.62 | 0.84 | 0.86 | 0.80 | 0.95 | 0.93 | 0.75 |

Class 7 recall is 6/6; class 4 (SPL BAJO) went from 0.46 (old baseline) to 0.80.

### Historical runs (broken data view — quoted from recorded outputs, not re-evaluated)

| Model | Test acc | Macro-F1 | Note |
|---|---|---|---|
| EfficientNet-01 | 67.7% | 0.63 | baseline config |
| EfficientNet-02 | 70.0% | 0.59 | baseline config — previous project best |
| EfficientNet-03a | — | — | squared inverse-freq class weights (~300× dynamic range) → training collapsed below the 36% majority-class baseline; aborted |
| EfficientNet-03b | — | — | balanced+boost weights, OneCycleLR: best val acc 71.6% (ep13), then overfit to 99% train acc; stopped ep18 |
| ResNet50-01 | 61.3% | 0.55 | 25M params fully fine-tuned on 1,382 images → memorised training set |
| PreTrained-01 | 0.3% | 0.00 | 44-class Watkins marine-species model evaluated zero-shot — disjoint class sets, invalid experiment |

## 🔍 Findings from Evaluation

1. **The data fix dominated everything.** +13 accuracy points (70→81%) with the identical baseline config. No amount of hyperparameter tuning on broken data came close — data quality beat every modelling effort combined.
2. **The regularisation package buys training stability more than raw accuracy.** On EfficientNet it added +2.1 pts (81.1→83.2) and, more importantly, healthy monotone learning curves — the unregularised control still shot to 99% train accuracy by epoch 14 and relied on early stopping to rescue a good checkpoint.
3. **Architecture choice mattered more than tuning: smaller generalises better.** MobileNetV3 (5.4M) > EfficientNet-B0 (4.0M, but higher capacity per image) > ResNet-50 (23.5M) on this 1,382-image dataset. ResNet-50 lost even with only layer4 trainable.
4. **Domain-mismatched transfer learning fails.** Warm-starting from the Watkins bioacoustics model (trained on librosa grayscale mel-spectrograms) collapsed to 27% — its features never matched these MATLAB RGB renders, and the frozen blocks locked the mismatch in. Generic ImageNet features transfer better than narrow-domain features from a different rendering pipeline.
5. **Remaining error mass is concentrated and explainable:**
   - **Class 3 ↔ 4 confusion**: the two 3.15 kHz tone classes differ only in amplitude (SPL), which per-sample z-score normalisation largely erases.
   - **Class 1 (Hammering)**: only 30 training samples; best F1 0.62.
6. **Checkpoint selection matters under imbalance**: weighted val loss was dominated by a handful of rare-class samples; switching model selection to val macro-F1 gave stabler, fairer checkpoints.
7. **Statistical honesty**: per-class F1 for classes 1, 5, 7 rests on 6–11 test samples — one flipped sample moves recall by ~17 points. Differences there are indicative, not significant; k-fold CV would be needed to defend rare-class claims.
8. **OneCycleLR and early stopping don't mix** (03b): early stopping cuts the schedule off before the low-LR anneal that OneCycle depends on. Use ReduceLROnPlateau with early stopping.

## ✅ Which Model to Use When

| Priority | Model | Choose when |
|---|---|---|
| **1** | **MobileNetV3-02** | Default & deployment: pre-labeling unlabeled data, CPU/edge inference. Best accuracy, best hard-class balance, smallest and fastest. |
| **2** | EfficientNet-04 | Ensemble partner (its errors differ — best class-7 F1 at 0.80) and the most stable recipe for future experiments (e.g. higher-resolution input). |
| **3** | EfficientNet-05 | Reporting only — the scientific control quantifying the data-fix effect. Not for deployment. |
| **4** | ResNet50-02 | No deployment scenario; keep as the architecture-comparison data point. |
| — | PreTrained-02 | Negative-result documentation only. |

**Deployment guidance** (pre-labeling unlabeled data): export softmax probabilities with each prediction; auto-accept high-confidence bulk classes, route low-confidence (max-prob < ~0.7) and all rare-class predictions (1, 7) to human review. Preprocessing parity is critical — unlabeled data must pass through the identical chain (same MATLAB render settings → plot-interior crop → 224×224 → z-score).

## 📁 Project Structure

```
REPSOL/
├── README.md
├── Dataset_exploration.ipynb              # EDA
├── Preprocess.ipynb                       # split + PNG→tensor conversion
├── Inference.ipynb                        # ⭐ deploy: run best model on new, unlabeled data
│
├── Data/                                  # (see Dataset section)
│
├── Models/                                # one notebook per training run
│   ├── EfficientNet_01..03.ipynb          # legacy (broken data) — kept for reproducibility
│   ├── EfficientNet_04.ipynb              # full package on fixed data
│   ├── EfficientNet_05.ipynb              # ablation control (baseline cfg, fixed data)
│   ├── MobileNetV3_01.ipynb / _02.ipynb   # _02 = best model
│   ├── ResNet50_01.ipynb / _02.ipynb
│   ├── PreTrained_model_01.ipynb / _02.ipynb  # Watkins zero-shot / warm-start
│   └── Evaluation.ipynb                   # ⭐ single source of truth: live eval + history + verdict
│
├── Models_output/                         # checkpoints (*.pth) + training histories (*.csv)
│
├── outputs/evaluation/                    # summary CSVs, comparison charts, confusion matrices,
│                                          # per-class F1 heatmap, learning curves, cache/
└── src/
    ├── dataset.py                         # SpectrogramPTDataset (transform + target_width params)
    ├── dataloaders.py                     # train/val/test loaders (train_transform for SpecAugment)
    ├── evaluate.py                        # metrics helpers
    ├── plot_analysis.py                   # visualisation utilities
    ├── EfficientNet/  MobileNet/  ResNet/ # one model.py per architecture (+ EfficientNet/train.py)
    ├── PreTrained_model/                  # Watkins checkpoint loading utilities
    └── preprocess/
        ├── generate_spectrograms_from_images.py  # PNG → full-figure .norm.pt (legacy path)
        ├── downsize_spectrograms.py       # ⭐ figure tensors → cropped 224×224 (the fix)
        ├── split.py / metadata.py / preprocess.py / normalize_pt.py
```

## 🚀 Installation

- Python 3.10+ · PyTorch, torchvision, torchaudio · scikit-learn, pandas, numpy, matplotlib, seaborn, tqdm

```bash
pip install torch torchvision torchaudio scikit-learn pandas numpy matplotlib seaborn tqdm
```

Training runs on CPU (~4–5 min/epoch for EfficientNet/MobileNet at 224×224); CUDA is used automatically if available.

## 💻 Usage

### 1. Build the dataset (one-time)
```bash
# PNG figures → full-figure tensors (needs the source image drive mounted)
python -m src.preprocess.generate_spectrograms_from_images
# full-figure tensors → model-ready 224×224 tensors
python -m src.preprocess.downsize_spectrograms
```

### 2. Train
Open a run notebook in `Models/` and run all cells — recommended starting points are `MobileNetV3_02.ipynb` or `EfficientNet_04.ipynb`. Each notebook is self-contained (config → data check → training → evaluation → learning curves) and auto-numbers its checkpoint so reruns never overwrite previous results.

### 3. Evaluate & compare
Run `Models/Evaluation.ipynb`. It live-evaluates every current checkpoint on the fixed test set (with caching — instant reruns), reproduces all comparison tables/charts, includes the historical runs, and ends with the verdict & decision guide. Set `FORCE_RERUN = True` after retraining a model.

### ⚠️ Legacy notebooks
`EfficientNet_01–03`, `ResNet50_01`, `MobileNetV3_01`, `PreTrained_model_01` point at the **broken** data view (`Data/Spectrograms` + 400-px crop). They are kept so their recorded results stay reproducible — do not use them for new experiments; copy from an `_02`/`04` notebook instead.

## 🔭 Next Steps

1. **Higher time-resolution input** — regenerate at 224×448 (`TARGET_SIZE` in `downsize_spectrograms.py`) to give transient classes (1, 5, 7) more temporal detail; compare against the 224² baseline.
2. **More data for classes 1 & 7** (30/29 training samples) — no ML technique substitutes for data; worth raising with REPSOL/INMAR. Overlapping-window slicing of source audio is an alternative (keep windows of one recording in one split to avoid leakage).
3. **Soft-voting ensemble** MobileNetV3-02 + EfficientNet-04 for offline batch labeling (see the oracle-ensemble ceiling in Evaluation.ipynb §11b).
4. **Class 3↔4**: their distinction is amplitude-only, which z-scoring erases — consider adding a global (non-per-sample) amplitude feature, or raise merging them as a domain question.
5. ~~**Inference script** for unlabeled data: folder → CSV of `filename, predicted_class, confidence, top-3` with a review threshold.~~ Done — see `Inference.ipynb`.
6. **K-fold cross-validation** if rare-class performance claims need statistical backing.

## 📄 License

This project is part of an INMAR internship program. Dataset contains industrial monitoring data from REPSOL facilities; handle accordingly.

---

**Last Updated**: July 2026
