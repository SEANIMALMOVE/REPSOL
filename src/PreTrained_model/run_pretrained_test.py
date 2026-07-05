import sys
from pathlib import Path

# Ensure 'src' is on sys.path so imports using the project package work
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import argparse
from PreTrained_model.load_pretrained import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--model", default="efficientnet")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model, sd = load_model(args.checkpoint, model_name=args.model, device=args.device)
    print("Model loaded. Parameter count:", sum(p.numel() for p in model.parameters()))

    import torch
    dummy = torch.randn(1, 1, 128, 128)
    device = torch.device(args.device)
    model = model.to(device)
    dummy = dummy.to(device)
    with torch.no_grad():
        out = model(dummy)
    print("Forward pass output shape:", out.shape)


if __name__ == "__main__":
    main()
