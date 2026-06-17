"""
Convert spectrogram images from D:\REPSOL_Classification\{class}\Espectrogramas
into .pt tensor files organized by train/val/test splits.
"""
from pathlib import Path
import pandas as pd
import torch
import numpy as np
from PIL import Image, ImageFile
import sys

ImageFile.LOAD_TRUNCATED_IMAGES = True


def image_to_tensor(image_path):
    """Convert an image file to a torch tensor."""
    try:
        try:
            from torchvision.io import read_image

            return read_image(str(image_path)).float()
        except Exception:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                img.load()
                img_array = np.asarray(img, dtype=np.uint8)

            tensor = torch.from_numpy(np.ascontiguousarray(img_array)).permute(2, 0, 1).float()
            return tensor
    except Exception as e:
        print(f"Error converting {image_path}: {type(e).__name__}: {e!r}")
        return None


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
    Convert spectrogram images to .pt files organized by split.
    
    Args:
        annotation_dir: Path to folder with train.csv, val.csv, test.csv
        source_images_dir: D:\REPSOL_Classification (base dir with class subfolders)
        target_dir: C:\home\ben\REPSOL\Data\Spectrograms
        splits: ("train", "val", "test")
    """
    annotation_dir = Path(annotation_dir)
    source_images_dir = Path(source_images_dir)
    target_dir = Path(target_dir)
    
    # Clear or create target directories
    for split in splits:
        split_dir = target_dir / split
        if split_dir.exists():
            print(f"Note: {split_dir} already exists. Will add/overwrite files.")
        else:
            split_dir.mkdir(parents=True, exist_ok=True)
    
    counts = {s: 0 for s in splits}
    missing = []
    errors = []
    
    # Process each split
    for split in splits:
        csv_path = annotation_dir / f"{split}.csv"
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found — skipping {split}")
            continue
        
        df = pd.read_csv(csv_path)
        print(f"\nProcessing {split} ({len(df)} rows)...")
        
        for idx, (_, row) in enumerate(df.iterrows()):
            if (idx + 1) % 100 == 0:
                print(f"  {split}: {idx + 1}/{len(df)}")
            
            class_name = str(row["category"])
            filename = str(row["filename"])
            
            source_image = None
            for candidate in _source_image_candidates(source_images_dir, class_name, filename):
                if candidate.exists():
                    source_image = candidate
                    break
            
            if source_image is None:
                missing.append(f"{class_name}/{filename}")
                continue
            
            # Convert image to tensor
            tensor = image_to_tensor(source_image)
            if tensor is None:
                errors.append(f"{class_name}/{filename}")
                continue
            
            # Save to target
            dst_dir = target_dir / split / class_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            filename_base = filename[:-4] if filename.endswith(".wav") else filename
            dst_path = dst_dir / (filename_base + ".wav.pt")

            if dst_path.exists():
                try:
                    torch.load(dst_path, map_location="cpu")
                    continue
                except Exception:
                    print(f"Corrupt tensor detected, regenerating: {dst_path}")
            
            try:
                tmp_path = dst_path.with_suffix(dst_path.suffix + ".tmp")
                torch.save(tensor, tmp_path)
                tmp_path.replace(dst_path)
                counts[split] += 1
            except Exception as e:
                print(f"Error saving {dst_path}: {e}")
                errors.append(f"{class_name}/{filename}")
        
        print(f"  Completed {split}: {counts[split]} .pt files saved")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for split in splits:
        print(f"{split}: {counts[split]} files")
    print(f"Total: {sum(counts.values())} files")
    
    if missing:
        print(f"\nMissing source images: {len(missing)}")
        for f in missing[:10]:
            print(f"  {f}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
    
    if errors:
        print(f"\nConversion errors: {len(errors)}")
        for f in errors[:5]:
            print(f"  {f}")


def main():
    repo_root = Path(__file__).resolve().parents[2]
    annotation_dir = repo_root / "Data" / "Annotations"
    source_images_dir = Path(r"D:\REPSOL_Classification")
    target_dir = repo_root / "Data" / "Spectrograms"
    
    print("Configuration:")
    print(f"  Annotations: {annotation_dir}")
    print(f"  Source images: {source_images_dir}")
    print(f"  Target: {target_dir}")
    print(f"  Source exists: {source_images_dir.exists()}")
    
    if not source_images_dir.exists():
        print(f"\nError: Source directory does not exist: {source_images_dir}")
        return
    
    generate_spectrograms_from_images(annotation_dir, source_images_dir, target_dir)


if __name__ == "__main__":
    main()
