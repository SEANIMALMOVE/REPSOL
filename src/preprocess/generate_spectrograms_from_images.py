"""
Convert spectrogram images from D:\REPSOL_Classification\{class}\Espectrogramas
into normalized .pt tensor files organized by train/val/test splits.

Run: python -m src.preprocess.generate_spectrograms_from_images
or : python src/preprocess/generate_spectrograms_from_images.py
"""
from pathlib import Path
import pandas as pd
import torch
import numpy as np
from PIL import Image, ImageFile
import shutil
import time

ImageFile.LOAD_TRUNCATED_IMAGES = True


def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor - tensor.mean()) / (tensor.std() + 1e-6)


def image_to_tensor(image_path):
    """Convert an image file to a normalized torch tensor."""
    try:
        try:
            from torchvision.io import read_image

            tensor = read_image(str(image_path)).float()
        except Exception:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                img.load()
                img_array = np.asarray(img, dtype=np.uint8)

            tensor = torch.from_numpy(np.ascontiguousarray(img_array)).permute(2, 0, 1).float()
        return normalize_tensor(tensor)
    except Exception as e:
        print(f"Error converting {image_path}: {type(e).__name__}: {e!r}")
        return None


def _check_disk_space(annotation_dir, source_images_dir, target_dir, splits):
    """Estimate required space by sampling one PNG → tensor, then extrapolate to all files."""
    # Count total files to generate and find one sample PNG
    file_count = 0
    sample_image = None
    for split in splits:
        csv_path = annotation_dir / f"{split}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            filename = str(row["filename"])
            class_name = str(row["category"])
            for candidate in _source_image_candidates(source_images_dir, class_name, filename):
                if candidate.exists():
                    file_count += 1
                    if sample_image is None:
                        sample_image = candidate
                    break

    if file_count == 0 or sample_image is None:
        print("\nDisk space check: no source files found, skipping.\n")
        return

    # Convert sample PNG to tensor, save to a temp file, measure actual on-disk size
    sample_tensor = image_to_tensor(sample_image)
    if sample_tensor is None:
        print("\nDisk space check: could not sample a file, skipping.\n")
        return
    tmp_sample = target_dir / "_space_check.tmp"
    try:
        torch.save(sample_tensor, tmp_sample)
        bytes_per_file = tmp_sample.stat().st_size
    finally:
        tmp_sample.unlink(missing_ok=True)

    estimated_bytes = int(bytes_per_file * file_count * 1.1)  # 10% buffer
    free_bytes = shutil.disk_usage(target_dir).free

    estimated_gb = estimated_bytes / 1024**3
    free_gb = free_bytes / 1024**3

    print(f"\nDisk space check:")
    print(f"  Files to generate : {file_count}")
    print(f"  Bytes per tensor  : {bytes_per_file / 1024**2:.1f} MB (sampled from one PNG)")
    print(f"  Estimated needed  : {estimated_gb:.2f} GB (incl. 10% buffer)")
    print(f"  Free on target    : {free_gb:.2f} GB")

    if free_bytes < estimated_bytes:
        raise RuntimeError(
            f"Not enough disk space on target drive.\n"
            f"  Needed : {estimated_gb:.2f} GB\n"
            f"  Free   : {free_gb:.2f} GB\n"
            f"  Short  : {(estimated_bytes - free_bytes) / 1024**3:.2f} GB\n"
            f"Free up space on '{Path(target_dir).anchor}' and try again."
        )

    print(f"  ✓ Sufficient space available.\n")


def _source_image_candidates(source_images_dir, class_name, filename):
    filename_base = filename[:-4] if filename.endswith(".wav") else filename
    for folder_name in ("Espectrogramas", "Espectrograma"):
        yield Path(source_images_dir) / class_name / folder_name / f"{filename_base}_spectrogram_win16384.png"


def generate_spectrograms_from_images(
    annotation_dir,
    source_images_dir,
    target_dir,
    splits=("train", "val", "test")
):
    """
    Convert spectrogram PNG images to normalized .pt files organized by split.
    Skips files that already exist and are valid.

    Args:
        annotation_dir: Path to folder with train.csv, val.csv, test.csv
        source_images_dir: D:\REPSOL_Classification (base dir with class subfolders)
        target_dir: Data/Spectrograms (output dir)
        splits: ("train", "val", "test")
    """
    annotation_dir = Path(annotation_dir)
    source_images_dir = Path(source_images_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    _check_disk_space(annotation_dir, source_images_dir, target_dir, splits)

    # Create target directories
    for split in splits:
        split_dir = target_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

    counts = {s: 0 for s in splits}
    skipped = {s: 0 for s in splits}
    missing = []
    errors = []
    start_time = time.time()

    # Process each split
    for split in splits:
        csv_path = annotation_dir / f"{split}.csv"
        if not csv_path.exists():
            print(f"⚠ Warning: {csv_path} not found — skipping {split}")
            continue

        df = pd.read_csv(csv_path)
        total = len(df)
        print(f"\n{'='*70}")
        print(f"Processing {split.upper()} ({total} rows)")
        print(f"{'='*70}")

        for idx, (_, row) in enumerate(df.iterrows(), 1):
            class_name = str(row["category"])
            filename = str(row["filename"])
            filename_base = filename[:-4] if filename.endswith(".wav") else filename

            # Check if already exists and is valid
            dst_dir = target_dir / split / class_name
            dst_path = dst_dir / (filename_base + ".wav.norm.pt")

            if dst_path.exists():
                try:
                    torch.load(dst_path, map_location="cpu")
                    skipped[split] += 1
                    pct = 100 * idx / total
                    print(f"  [{idx:4d}/{total}] {pct:5.1f}% SKIP {class_name:30s} {filename}")
                    continue
                except Exception:
                    print(f"  [{idx:4d}/{total}] Corrupt tensor, regenerating: {dst_path}")

            # Find source image
            source_image = None
            for candidate in _source_image_candidates(source_images_dir, class_name, filename):
                if candidate.exists():
                    source_image = candidate
                    break

            if source_image is None:
                missing.append(f"{class_name}/{filename}")
                pct = 100 * idx / total
                print(f"  [{idx:4d}/{total}] {pct:5.1f}% MISS {class_name:30s} {filename}")
                continue

            # Convert and normalize
            tensor = image_to_tensor(source_image)
            if tensor is None:
                errors.append(f"{class_name}/{filename}")
                pct = 100 * idx / total
                print(f"  [{idx:4d}/{total}] {pct:5.1f}% ERR  {class_name:30s} {filename}")
                continue

            # Save normalized tensor with atomic write (write to .tmp first, then rename)
            dst_dir.mkdir(parents=True, exist_ok=True)
            try:
                tmp_path = dst_path.with_suffix(dst_path.suffix + ".tmp")
                torch.save(tensor, tmp_path)
                tmp_path.replace(dst_path)
                counts[split] += 1
                pct = 100 * idx / total
                elapsed = time.time() - start_time
                print(f"  [{idx:4d}/{total}] {pct:5.1f}% OK   {class_name:30s} {filename} ({elapsed:.0f}s)")
            except Exception as e:
                errors.append(f"{class_name}/{filename}")
                pct = 100 * idx / total
                print(f"  [{idx:4d}/{total}] {pct:5.1f}% FAIL {class_name:30s} {filename} ({e})")
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass

        elapsed = time.time() - start_time
        print(f"  Completed {split}: {counts[split]} saved, {skipped[split]} skipped")

    # Summary
    elapsed_total = time.time() - start_time
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for split in splits:
        print(f"{split:10s}: {counts[split]:5d} saved, {skipped[split]:5d} skipped")
    print(f"{'─'*70}")
    print(f"{'Total':10s}: {sum(counts.values()):5d} saved, {sum(skipped.values()):5d} skipped")
    print(f"Time: {elapsed_total:.1f}s")

    if missing:
        print(f"\n⚠ Missing source images: {len(missing)}")
        for f in missing[:10]:
            print(f"  - {f}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    if errors:
        print(f"\n⚠ Conversion errors: {len(errors)}")
        for f in errors[:5]:
            print(f"  - {f}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")


def main(annotation_dir=None, source_images_dir=None, target_dir=None):
    if annotation_dir is None:
        repo_root = Path(__file__).resolve().parents[2]
        annotation_dir = repo_root / "Data" / "Annotations"
        source_images_dir = Path(r"D:\REPSOL_Classification")
        target_dir = repo_root / "Data" / "Spectrograms"

    print("\n" + "="*70)
    print("SPECTROGRAM GENERATION (PNG → normalized .pt)")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Annotations:  {annotation_dir}")
    print(f"  Source:       {source_images_dir}")
    print(f"  Target:       {target_dir}")
    print(f"\nValidation:")
    print(f"  Annotations exist:    {annotation_dir.exists()}")
    print(f"  Source exists:        {source_images_dir.exists()}")

    if not annotation_dir.exists():
        print(f"\n❌ Error: Annotations directory not found: {annotation_dir}")
        return

    if not source_images_dir.exists():
        print(f"\n❌ Error: Source directory not found: {source_images_dir}")
        return

    print(f"\n✓ All paths valid. Starting conversion...\n")
    generate_spectrograms_from_images(annotation_dir, source_images_dir, target_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert PNG spectrograms to normalized .pt files")
    parser.add_argument("--source", type=str, default=None, help="Source directory (default: D:\\REPSOL_Classification)")
    parser.add_argument("--target", type=str, default=None, help="Target directory (default: {repo}/Data/Spectrograms)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    annotation_dir = repo_root / "Data" / "Annotations"
    source_images_dir = Path(args.source) if args.source else Path(r"D:\REPSOL_Classification")
    target_dir = Path(args.target) if args.target else repo_root / "Data" / "Spectrograms"

    main(annotation_dir, source_images_dir, target_dir)
