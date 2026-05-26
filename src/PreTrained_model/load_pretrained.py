import re
from pathlib import Path
import torch
from typing import Tuple, Optional

from .model import get_model


def _extract_state_dict(checkpoint: dict) -> dict:
    # Common containers
    for key in ("state_dict", "model_state_dict", "model", "model_state"):
        if isinstance(checkpoint, dict) and key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]

    # If values are tensors, treat as state dict
    if isinstance(checkpoint, dict) and all(hasattr(v, "shape") for v in checkpoint.values()):
        return checkpoint

    # Try to find a nested dict that looks like a state dict
    if isinstance(checkpoint, dict):
        for v in checkpoint.values():
            if isinstance(v, dict) and all(hasattr(x, "shape") for x in v.values()):
                return v

    raise RuntimeError("Unable to locate state dict in checkpoint")


def infer_num_classes(state_dict: dict) -> int:
    candidates = []
    for k, v in state_dict.items():
        if k.lower().endswith(".weight") and getattr(v, "dim", lambda: 0)() == 2:
            if "classifier" in k.lower() or re.search(r"fc|head|classifier", k, re.I):
                candidates.append((k, v.shape[0], v.shape[1]))

    if candidates:
        # prefer the smallest output dim (likely num_classes)
        candidates.sort(key=lambda x: x[1])
        return int(candidates[0][1])

    # fallback: any linear-like weight with reasonably small out dim
    for k, v in state_dict.items():
        if k.lower().endswith(".weight") and getattr(v, "dim", lambda: 0)() == 2:
            if v.shape[0] <= 1024:
                return int(v.shape[0])

    raise RuntimeError("Couldn't infer num_classes from checkpoint; pass num_classes explicitly")


def _strip_prefix_keys(sd: dict, prefix: str) -> dict:
    out = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            out[k[len(prefix) :]] = v
        else:
            out[k] = v
    return out


def load_model(
    checkpoint_path: str,
    model_name: str = "efficientnet",
    num_classes: Optional[int] = None,
    device: str = "cpu",
    input_channels: int = 1,
    freeze_backbone: bool = False,
) -> Tuple[torch.nn.Module, dict]:
    """Load a model from a checkpoint file.

    Attempts to locate the state dict inside common checkpoint containers, infer
    the output `num_classes` when not provided, construct the model using
    `get_model(...)`, and load the weights (best-effort, uses strict=False
    fallback where necessary).
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    raw = torch.load(str(ckpt_path), map_location="cpu")
    sd = _extract_state_dict(raw)

    if num_classes is None:
        num_classes = infer_num_classes(sd)

    model = get_model(model_name=model_name, num_classes=int(num_classes), input_channels=input_channels, freeze_backbone=freeze_backbone)

    # Try direct load first
    try:
        model.load_state_dict(sd)
    except Exception:
        # try common fixes: strip 'module.' (from DataParallel) and 'backbone.' prefixes
        attempt = sd
        if any(k.startswith("module.") for k in sd.keys()):
            attempt = _strip_prefix_keys(attempt, "module.")

        try:
            model.load_state_dict(attempt, strict=False)
        except Exception:
            # try removing 'backbone.' prefix if present in checkpoint
            attempt2 = {k.replace("backbone.", ""): v for k, v in attempt.items()}
            model.load_state_dict(attempt2, strict=False)

    model.to(device)
    model.eval()
    return model, sd


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to .pth checkpoint")
    parser.add_argument("--model", default="efficientnet", help="Model name for factory (efficientnet|baseline)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-classes", type=int, default=None)
    args = parser.parse_args()

    model, sd = load_model(args.checkpoint, model_name=args.model, num_classes=args.num_classes, device=args.device)
    print("Loaded model:", args.model)
    param_count = sum(p.numel() for p in model.parameters())
    print("Parameter count:", param_count)
    # print found classifier keys
    cls_keys = [k for k in sd.keys() if "class" in k.lower() or "classifier" in k.lower() or re.search(r"fc|head", k, re.I)]
    print("Classifier-related keys found:", cls_keys)
