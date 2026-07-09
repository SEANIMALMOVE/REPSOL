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


class ResNet50Spectrogram(nn.Module):
    """ResNet-50 backbone pretrained on ImageNet, adapted for spectrogram classification.

    Accepts 1-channel or 3-channel input. Single-channel inputs are repeated
    across 3 channels before being passed to the backbone.
    """

    def __init__(self, num_classes: int, freeze_backbone: bool = False):
        super().__init__()

        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Replace final fully-connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )

        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if not name.startswith("fc"):
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

    model_name: 'resnet50', 'resnet', or 'r50'
    """
    if model_name.lower() in ("resnet50", "resnet", "r50"):
        return ResNet50Spectrogram(num_classes=num_classes, freeze_backbone=freeze_backbone)
    raise ValueError(f"Unknown model_name: {model_name!r}. Use 'resnet50', 'resnet', or 'r50'.")


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


def get_weighted_criterion(labels, num_classes=None, device=None):
    weights = compute_class_weights(labels, num_classes=num_classes)
    if device is not None:
        weights = weights.to(device)
    return nn.CrossEntropyLoss(weight=weights)
