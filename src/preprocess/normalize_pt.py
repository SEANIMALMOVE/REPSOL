"""
Offline normalization for spectrogram .pt files.

This script walks a spectrogram root (train/val/test) and normalizes each
.pt tensor using per-sample mean/std normalization and writes a new file
with suffix `.norm.pt` by default. Use `--inplace` to overwrite originals
and `--backup` to keep a `.bak` copy.

Usage (project root):
python -m src.preprocess.normalize_pt --root Data/Spectrograms --inplace --backup
"""
from pathlib import Path
import argparse
import torch # type: ignore
import shutil


def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.float()
    return (tensor - tensor.mean()) / (tensor.std() + 1e-6)


def process_file(path: Path, inplace: bool = False, backup: bool = False) -> Path:
    tensor = torch.load(path)

    # Match dataset shape handling: [H,W] -> [1,H,W]; permute if channels last
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3 and tensor.shape[0] not in (1, 3):
        tensor = tensor.permute(2, 0, 1)

    norm = normalize_tensor(tensor)

    if inplace:
        if backup:
            bak = path.with_suffix(path.suffix + ".bak")
            shutil.copy(path, bak)
        torch.save(norm, path)
        return path
    else:
        new_path = path.with_name(path.stem + ".norm.pt")
        torch.save(norm, new_path)
        return new_path


def find_pt_files(root: Path):
    for sub in ("train", "val", "test"):
        folder = root / sub
        if not folder.exists():
            continue
        for class_dir in folder.iterdir():
            if not class_dir.is_dir():
                continue
            for p in class_dir.rglob("*.pt"):
                yield p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Spectrogram root (contains train/val/test)")
    parser.add_argument("--inplace", action="store_true", help="Overwrite original .pt files")
    parser.add_argument("--backup", action="store_true", help="When used with --inplace, keep .bak copies of originals")
    args = parser.parse_args()

    root = args.root
    if not root.exists():
        raise SystemExit(f"Root path not found: {root}")

    files = list(find_pt_files(root))
    if not files:
        print("No .pt files found under", root)
        return

    print(f"Found {len(files)} .pt files. inplace={args.inplace}, backup={args.backup}")

    for i, p in enumerate(files, 1):
        out = process_file(p, inplace=args.inplace, backup=args.backup)
        print(f"[{i}/{len(files)}] -> {p} -> {out}")


if __name__ == "__main__":
    main()