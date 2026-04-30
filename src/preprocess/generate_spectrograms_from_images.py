"""
Convert spectrogram images from D:\REPSOL_Classification\{class}\Espectrogramas
into .pt tensor files organized by train/val/test splits.
"""
from pathlib import Path
import pandas as pd
import torch
import numpy as np
from PIL import Image
import sys


def image_to_tensor(image_path):
    """Convert an image file to a torch tensor."""
    try:
        img = Image.open(image_path)
        # Convert to numpy array
        img_array = np.array(img)
        # Convert to tensor
        tensor = torch.from_numpy(img_array).float()
        return tensor
    except Exception as e:
        print(f"Error converting {image_path}: {e}")
        return None


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
            
            # Build source image path: D:\REPSOL_Classification\{class}\Espectrogramas\{filename}_spectrogram_win16384.png
            source_image = source_images_dir / class_name / "Espectrogramas" / (filename + "_spectrogram_win16384.png")
            
            if not source_image.exists():
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
            dst_path = dst_dir / (filename + ".wav.pt")
            
            try:
                torch.save(tensor, dst_path)
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
