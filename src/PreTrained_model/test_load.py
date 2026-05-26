"""Small script to test loading a checkpoint and running a dummy forward pass.

Usage:
    python -m src.PreTrained_model.test_load <checkpoint> --model efficientnet --device cpu
"""
import argparse
import torch
from pathlib import Path

from .load_pretrained import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--model", default="efficientnet")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model, sd = load_model(args.checkpoint, model_name=args.model, device=args.device)
    print("Model loaded. Parameter count:", sum(p.numel() for p in model.parameters()))

    # Prepare a dummy input matching expected shape: [B, 1, H, W]
    # Try to infer shape from first conv layer if possible, otherwise use 128x128
    dummy = torch.randn(1, 1, 128, 128)
    device = torch.device(args.device)
    model = model.to(device)
    dummy = dummy.to(device)
    with torch.no_grad():
        out = model(dummy)
    print("Forward pass output shape:", out.shape)


if __name__ == "__main__":
    main()
