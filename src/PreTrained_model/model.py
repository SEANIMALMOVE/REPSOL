import torch
import torch.nn as nn
from torchvision import models # type: ignore
import numpy as np

try:
    from sklearn.utils.class_weight import compute_class_weight
    _HAS_SKLEARN = True
except Exception:
    compute_class_weight = None
    _HAS_SKLEARN = False


class EfficientNetSpectrogram(nn.Module):
    def __init__(self, num_classes: int, freeze_backbone: bool = True):
        super().__init__()

        # Load pretrained EfficientNet-B0
        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        # Replace classifier head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

        # Optionally freeze feature extractor
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        x shape: [B, 1, H, W]  (mel spectrogram)
        """
        x = x.repeat(1, 3, 1, 1)  # Makes 3 identical channels so the model can use knowledge from color images

        # Forward through EfficientNet
        return self.backbone(x)



# Deeper baseline CNN with Global Average Pooling and larger classifier
class BaselineCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=2):
        super().__init__()

        # Convolutional feature extractor: 16 -> 32 -> 64 -> 128
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),   # Looks for simple patterns in the sound-image, like edges or blobs
            nn.ReLU(inplace=True),                                    # Keeps only strong pattern signals, ignores weak/negative ones
            nn.MaxPool2d(2),                                          # Shrinks the image, keeping only the most important signals

            nn.Conv2d(16, 32, kernel_size=3, padding=1),              # Looks for more complex patterns, using previous layer's output
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),              # Even more complex pattern detection
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),             # Final set of pattern detectors
            nn.ReLU(inplace=True),
            # keep spatial resolution and use global pooling afterwards
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # Averages each pattern map into a single number (summarizes presence)
        self.classifier = nn.Sequential(
            nn.Flatten(),                  # Turns the summary numbers into a single list for decision making
            nn.Linear(128, 256),           # Mixes the pattern summaries to find combinations that matter for each animal
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),               # Randomly ignores some signals during training so the model doesn't become over-reliant
            nn.Linear(256, num_classes),   # Gives a score for each animal; the highest score is the model's guess
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)       # -> [B, 128, 1, 1]
        x = self.classifier(x)
        return x


def get_model(model_name: str, num_classes: int, input_channels: int = 1, freeze_backbone: bool = True):
    """Factory: returns the requested model.

    model_name: 'baseline' or 'efficientnet'
    """
    if model_name.lower() in ("efficientnet", "efficientnet_b0", "b0"):
        return EfficientNetSpectrogram(num_classes=num_classes, freeze_backbone=freeze_backbone)
    else:
        return BaselineCNN(input_channels=input_channels, num_classes=num_classes)


def compute_class_weights(labels, num_classes=None):
    """Compute per-class weights suitable for CrossEntropyLoss.

    labels: 1D array-like of integer class labels
    Returns a torch.FloatTensor of length `num_classes`.
    """
    labels = np.asarray(labels)
    if num_classes is None:
        num_classes = int(labels.max()) + 1

    if _HAS_SKLEARN:
        weights = compute_class_weight(class_weight="balanced", classes=np.arange(num_classes), y=labels)
    else:
        # Fallback: inverse frequency (balanced)
        counts = np.bincount(labels, minlength=num_classes)
        counts = counts.astype(float)
        total = counts.sum()
        # avoid division by zero
        counts[counts == 0] = 1.0
        weights = total / (num_classes * counts)

    return torch.tensor(weights, dtype=torch.float)


def get_weighted_criterion(labels, num_classes=None, device=None):
    """Return CrossEntropyLoss with weights computed from `labels`.

    `labels` can be a list/array of training labels.
    """
    weights = compute_class_weights(labels, num_classes=num_classes)
    if device is not None:
        weights = weights.to(device)
    return nn.CrossEntropyLoss(weight=weights)