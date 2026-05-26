# REPSOL Industrial Sound Classification Project

A deep learning project for classifying industrial sounds from REPSOL facilities using spectrogram-based audio analysis. This project implements multiple neural network architectures to identify different types of machinery and operational patterns.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Results](#results)
- [Data Pipeline](#data-pipeline)

## 🎯 Project Overview

This project aims to classify industrial sounds into 8 distinct categories using deep learning. The sound data is converted to spectrograms (frequency representations) and fed into neural networks for classification. This could be used for predictive maintenance, anomaly detection, or operational monitoring in industrial settings.

### Sound Categories

The dataset contains 8 classes of industrial sounds:

0. **ActividadBASE_NO pattern activity** - Baseline/background activity
1. **Pulses HAMMERING** - Hammering/pulsing sounds
2. **Marked cycles 3 segundos** - Marked cyclical patterns (3 seconds)
3. **Continuous activity & tone 3.15kHz_SPL ALTO** - Continuous tone (high SPL)
4. **Continuous activity & tone 3.15kHz_SPL BAJO** - Continuous tone (low SPL)
5. **Blasts** - Blast/explosion sounds
6. **Machinery continuous activity** - Steady machinery operation
7. **Works, sirens and knocks en altas frecuencias RAFAGAS a 3.15** - High-frequency sirens and knocks

## 📊 Dataset

### Structure

```
Data/
├── Annotations/
│   ├── audio_annotations.csv     # Metadata for all audio files
│   ├── train.csv                 # Training set annotations
│   ├── val.csv                   # Validation set annotations
│   └── test.csv                  # Test set annotations
└── Spectrograms/
    ├── train/                    # Training spectrograms (70%)
    ├── val/                      # Validation spectrograms (15%)
    └── test/                     # Test spectrograms (15%)
```

### Audio Specifications

- **Sample Rate**: 96,000 Hz
- **Duration**: 60 seconds per clip
- **Format**: WAV files (original), converted to spectrograms (.pt format)
- **Train/Val/Test Split**: 70/15/15

### Annotation Format

The CSV files contain:
- `category` - Sound class label
- `filename` - Original WAV filename
- `duration_sec` - Audio duration in seconds
- `sample_rate` - Audio sample rate in Hz

## 📁 Project Structure

```
REPSOL/
├── README.md                              # This file
├── Dataset_exploration.ipynb              # EDA notebook
├── Main.ipynb                             # Main training notebook
├── Preprocess.ipynb                       # Data preprocessing notebook
├── run_pretrained_test.py                 # Script to test pretrained model
│
├── Data/
│   ├── Annotations/                       # CSV metadata
│   └── Spectrograms/                      # Spectrogram tensors
│
├── Models/
│   ├── EfficientNet.ipynb                 # EfficientNet training notebook
│   ├── Main.ipynb                         # Main model training
│   └── PreTrained_model.ipynb             # Pre-trained model notebook
│
├── Models_output/
│   ├── efficientnet_best_01.pth           # Best EfficientNet weights
│   ├── efficientnet_best_training_history_01.csv  # Training history
│   ├── PreTrained_model_best_01.pth       # Best pre-trained model weights
│
└── src/
    ├── dataloaders.py                     # PyTorch DataLoader utilities
    ├── dataset.py                         # Custom Dataset classes
    ├── evaluate.py                        # Evaluation metrics
    ├── plot_analysis.py                   # Visualization utilities
    │
    ├── EfficientNet/
    │   ├── model.py                       # EfficientNet architecture
    │   └── train.py                       # Training loop
    │
    ├── preprocess/
    │   ├── generate_spectrograms_from_images.py  # Spectrogram generation
    │   ├── metadata.py                    # Metadata handling
    │   ├── normalize_pt.py                # Normalization utilities
    │   ├── preprocess.py                  # Main preprocessing
    │   ├── split_fold.py                  # K-fold splitting
    │   └── split.py                       # Train/val/test splitting
    │
    └── PreTrained_model/
        ├── model.py                       # Pre-trained model architecture
        ├── load_pretrained.py             # Model loading utilities
        └── test_load.py                   # Testing script
```

## 🚀 Installation

### Requirements

- Python 3.8+
- PyTorch with GPU support (recommended)
- torchvision
- torchaudio
- scikit-learn
- pandas
- numpy
- matplotlib

### Setup

1. **Clone or navigate to the project**:
   ```bash
   cd REPSOL
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install torch torchvision torchaudio
   pip install scikit-learn pandas numpy matplotlib
   ```

## 💻 Usage

### Data Preprocessing

1. **Generate spectrograms** from audio files:
   ```bash
   python src/preprocess/generate_spectrograms_from_images.py
   ```

2. **Split data** into train/val/test sets:
   ```bash
   python src/preprocess/split.py
   ```

3. **Normalize spectrograms**:
   ```bash
   python src/preprocess/normalize_pt.py
   ```

### Training

#### Option 1: Using Jupyter Notebooks

Run the notebooks in this order:

1. **Main.ipynb** - Full pipeline (recommended for first run)
   - Loads data
   - Trains EfficientNet
   - Evaluates on test set
   - Generates classification report

2. **Dataset_exploration.ipynb** - Exploratory Data Analysis
   - Visualizes spectrogram distribution
   - Checks data balance
   - Analyzes class statistics

#### Option 2: Using Scripts

Train EfficientNet directly:
```bash
cd Models
jupyter nbconvert --to script Main.ipynb
python Main.py
```

### Evaluation

Test a pre-trained model:
```bash
python run_pretrained_test.py
```

## 🧠 Models

### 1. EfficientNet (EfficientNet-B0)

**Architecture**: Transfer learning from ImageNet-pretrained EfficientNet-B0

**Key Features**:
- Pretrained backbone frozen (feature extractor)
- Custom classifier head for 8 sound classes
- Single-channel mel-spectrograms converted to 3-channel images
- Efficient scaling for good accuracy-to-parameter ratio

**Configuration** (from Main.ipynb):
- Batch Size: 16
- Epochs: 15
- Learning Rate: 1e-3
- Early Stopping Patience: 4 epochs
- Optimizer: Adam (default in training)

**Output**: `Models_output/efficientnet_best_01.pth`

### 2. Pre-trained Model

**Architecture**: Custom architecture with pre-trained ImageNet weights

**Key Features**:
- Specialized for spectrogram classification
- Can be loaded with custom checkpoint system

**Output**: `Models_output/PreTrained_model_best_01.pth`

## 📈 Results

The trained models' performance is tracked through:

1. **Training History**: `Models_output/efficientnet_best_training_history_01.csv`
   - Loss and accuracy per epoch
   - Validation metrics
   - Training curves

2. **Model Checkpoints**: `.pth` files containing best weights

3. **Evaluation Reports**: Classification reports and confusion matrices generated during evaluation

### Metrics

- **Accuracy**: Overall classification accuracy on test set
- **Precision/Recall/F1**: Per-class metrics
- **Confusion Matrix**: Misclassification patterns

## 🔧 Data Pipeline

### Spectrogram Generation

```
Raw WAV Audio (96 kHz, 60 sec)
    ↓
Mel-Spectrogram (converts time-domain to frequency-domain)
    ↓
Normalized Tensor (channel-first format)
    ↓
Saved as .pt file (PyTorch tensor)
```

### Dataset Loading

The `SpectrogramPTDataset` class:
- Loads pre-generated .pt spectrogram files
- Maintains class hierarchy
- Supports in-memory caching for faster training
- Applies optional transforms (augmentations)

```python
from src.dataset import SpectrogramPTDataset

dataset = SpectrogramPTDataset(
    root_dir="Data/Spectrograms/train",
    cache_in_memory=True  # Cache dataset in RAM for faster access
)
```

### Data Augmentation

Transforms can be applied during training for:
- Robustness improvement
- Better generalization
- Reducing overfitting

## 📝 Key Notebooks

### Main.ipynb
The primary notebook that runs the complete pipeline:
1. Configures hyperparameters
2. Loads and verifies spectrogram tensors
3. Trains EfficientNet with progress tracking
4. Evaluates on validation and test sets
5. Outputs detailed classification metrics

### Dataset_exploration.ipynb
Exploratory analysis:
- Data distribution across classes
- Spectrogram visualization
- Class imbalance analysis

### Preprocess.ipynb
Data preparation:
- Audio file validation
- Spectrogram generation
- Normalization and caching

## 🔍 Troubleshooting

### Out of Memory (OOM)

- Reduce `BATCH_SIZE` in training notebooks
- Set `cache_in_memory=False` in dataset loading
- Use gradient accumulation

### Model Not Improving

- Check data loading (verify spectrograms are correct)
- Adjust learning rate
- Increase training epochs
- Ensure classes are balanced

### Missing Spectrograms

- Run preprocessing pipeline first
- Verify audio files are in correct format
- Check file paths in configuration

## 📖 References

- **PyTorch Documentation**: https://pytorch.org/docs/stable/index.html
- **EfficientNet Paper**: https://arxiv.org/abs/1905.11946
- **Mel-Spectrograms**: https://en.wikipedia.org/wiki/Mel-scale

## 📞 Notes

- All paths in notebooks may need adjustment based on your local setup
- GPU recommended for faster training (CUDA support detected automatically)
- Dataset contains sensitive industrial data; handle accordingly

## 📄 License

This project is part of an INMAR internship program.

---

**Last Updated**: May 2026
