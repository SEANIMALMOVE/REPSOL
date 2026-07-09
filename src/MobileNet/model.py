import torch
import torch.nn as nn
from torchvision import models
import numpy as np

try:
    from sklearn.utils.class_weight import compute_class_weight
    _HAS_SKLEARN = True
except Exception:
    compute_class_weight = None
    _HAS_SKLEARN = False


class MobileNetV3Spectrogram(nn.Module):
    """MobileNetV3-Large pretrained on ImageNet, adapted for spectrogram classification.

    Lightweight (~5.4M params) and CPU-friendly: hard-swish activations and
    squeeze-excitation blocks give near-EfficientNet accuracy at lower compute.
    Accepts 1-channel or 3-channel input; single-channel inputs are repeated
    across 3 channels before the backbone.
    """

    def __init__(self, num_classes: int, freeze_backbone: bool = False):
        super().__init__()

        self.backbone = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
        )

        # classifier = [Linear(960,1280), Hardswish, Dropout(0.2), Linear(1280,1000)]
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, num_classes)

        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(f"Expected [B, C, H, W], got {tuple(x.shape)}")
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != 3:
            raise ValueError(f"Expected 1 or 3 input channels, got {x.shape[1]}")
        return self.backbone(x)


def get_model(model_name: str, num_classes: int, freeze_backbone: bool = False, **kwargs):
    """Factory: returns the requested model.

    model_name: 'mobilenet', 'mobilenetv3', or 'v3'
    """
    if model_name.lower() in ("mobilenet", "mobilenetv3", "mobilenet_v3", "v3"):
        return MobileNetV3Spectrogram(num_classes=num_classes, freeze_backbone=freeze_backbone)
    raise ValueError(f"Unknown model_name: {model_name!r}. Use 'mobilenet', 'mobilenetv3', or 'v3'.")


def compute_class_weights(labels, num_classes=None):
    labels = np.asarray(labels)
    if num_classes is None:
        num_classes = int(labels.max()) + 1

    if _HAS_SKLEARN:
        weights = compute_class_weight(
            class_weight="balanced", classes=np.arange(num_classes), y=labels
        )
    else:
        counts = np.bincount(labels, minlength=num_classes).astype(float)
        counts[counts == 0] = 1.0
        weights = counts.sum() / (num_classes * counts)

    return torch.tensor(weights, dtype=torch.float)


def get_weighted_criterion(labels, num_classes=None, device=None, label_smoothing=0.0):
    weights = compute_class_weights(labels, num_classes=num_classes)
    if device is not None:
        weights = weights.to(device)
    return nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
