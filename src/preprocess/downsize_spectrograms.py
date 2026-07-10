"""
Downsize the stored full-figure spectrogram tensors to model-ready inputs.

The original .norm.pt files are z-scored RGB renders of full MATLAB figures
(3 x 1714 x 3156, ~65 MB each) including title text, axis labels and colorbar.
This script:
  1. crops the plot interior (rows 54:1578, cols 209:2969 — constant across
     all files, verified on multiple classes),
  2. resizes to TARGET_SIZE with antialiasing,
  3. re-applies z-score normalization,
  4. saves to Data/Spectrograms_224 mirroring the split/class structure.

Result: ~600 KB per file, the model sees the FULL time axis instead of the
first 400 of 3156 pixel columns, and no filename/station text is visible.

Run: python -m src.preprocess.downsize_spectrograms
"""
from pathlib import Path
import time
import torch
import torch.nn.functional as F

# Plot interior inside the rendered figure (excludes axes box lines)
CROP_ROWS = (54, 1578)
CROP_COLS = (209, 2969)
TARGET_SIZE = (224, 224)

SOURCE_DIR = Path(r"D:\Work\Internships\INMAR\REPSOL\Data\Spectrograms")
TARGET_DIR = Path(r"D:\Work\Internships\INMAR\REPSOL\Data\Spectrograms_224")
SPLITS = ("train", "val", "test")


def downsize_tensor(t: torch.Tensor) -> torch.Tensor:
    """Crop plot interior, resize, re-normalize. Input [3, H, W] z-scored."""
    t = t[:, CROP_ROWS[0]:CROP_ROWS[1], CROP_COLS[0]:CROP_COLS[1]]
    t = F.interpolate(
        t.unsqueeze(0), size=TARGET_SIZE, mode="bilinear",
        align_corners=False, antialias=True,
    ).squeeze(0)
    return (t - t.mean()) / (t.std() + 1e-6)


def main():
    start = time.time()
    done, skipped, errors = 0, 0, []

    files = []
    for split in SPLITS:
        files.extend(sorted((SOURCE_DIR / split).rglob("*.norm.pt")))
    total = len(files)
    print(f"Found {total} source tensors under {SOURCE_DIR}")

    for i, src in enumerate(files, 1):
        rel = src.relative_to(SOURCE_DIR)
        dst = TARGET_DIR / rel
        if dst.exists():
            try:
                torch.load(dst, map_location="cpu")
                skipped += 1
                continue
            except Exception:
                pass  # corrupt — regenerate

        try:
            t = torch.load(src, map_location="cpu")
            if t.dim() == 3 and t.shape[0] not in (1, 3):
                t = t.permute(2, 0, 1)
            small = downsize_tensor(t.float())
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(dst.suffix + ".tmp")
            torch.save(small, tmp)
            tmp.replace(dst)
            done += 1
        except Exception as e:
            errors.append(f"{rel}: {type(e).__name__}: {e}")

        if i % 50 == 0 or i == total:
            elapsed = time.time() - start
            rate = i / elapsed if elapsed > 0 else 0
            eta = (total - i) / rate if rate > 0 else 0
            print(f"  [{i:4d}/{total}] done={done} skip={skipped} "
                  f"err={len(errors)}  {rate:.1f} files/s  ETA {eta/60:.1f} min",
                  flush=True)

    print(f"\nFinished in {(time.time()-start)/60:.1f} min: "
          f"{done} converted, {skipped} skipped, {len(errors)} errors")
    for e in errors[:10]:
        print("  ERR", e)


if __name__ == "__main__":
    main()
