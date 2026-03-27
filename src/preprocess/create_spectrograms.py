import pandas as pd  # type: ignore
from pathlib import Path
from .preprocess import generate_spectrogram


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def process_split(
    annotation_csv: str | Path,
    audio_dir: str | Path,
    spectrogram_dir: str | Path,
    split_name: str = "trainval",
):
    """
    Process a single split CSV and generate spectrograms.
    
    Args:
        annotation_csv: Path to CSV file (trainval.csv or test.csv)
        audio_dir: Root audio directory
        spectrogram_dir: Root spectrogram output directory
        split_name: Name of split (trainval or test)
    
    Output structure:
        spectrogram_dir / split_name / category / filename.pt
    """
    annotation_csv = Path(annotation_csv)
    audio_dir = Path(audio_dir)
    spectrogram_dir = Path(spectrogram_dir)
    
    if not annotation_csv.exists():
        raise FileNotFoundError(f"Annotation CSV not found: {annotation_csv}")
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    print(f"\nProcessing {split_name}...")
    df = pd.read_csv(annotation_csv)

    spec_paths = []
    errors = []

    for idx, row in df.iterrows():
        category = row["category"]
        filename = row["filename"]

        audio_path = audio_dir / category / filename
        output_dir = spectrogram_dir / split_name / category
        ensure_dir(output_dir)

        output_path = output_dir / f"{filename}.pt"

        # Skip if already exists
        if output_path.exists():
            spec_paths.append(str(output_path))
            continue

        try:
            print(f"[{idx+1}/{len(df)}] {filename} ({category})")
            generate_spectrogram(audio_path, output_path)
            spec_paths.append(str(output_path))
        except Exception as e:
            print(f"    ERROR: {e}")
            spec_paths.append(None)
            errors.append({"filename": filename, "category": category, "error": str(e)})

    # Save output CSV with spectrogram paths
    df["spectrogram_path"] = spec_paths
    df.to_csv(annotation_csv, index=False)

    print(f"{split_name} complete.")
    print(f"  Total: {len(df)}")
    print(f"  Generated: {sum(1 for p in spec_paths if p is not None)}")
    print(f"  Errors: {len(errors)}")

    if errors:
        error_df = pd.DataFrame(errors)
        print("\nErrors encountered:")
        print(error_df.to_string(index=False))

    return len(df), sum(1 for p in spec_paths if p is not None), errors


def main(
    annotation_dir,
    audio_dir,
    spectrogram_dir,
    splits=["trainval", "test"],
):
    """Process multiple splits (trainval and test)."""
    annotation_dir = Path(annotation_dir)
    spectrogram_dir = Path(spectrogram_dir)
    spectrogram_dir.mkdir(parents=True, exist_ok=True)

    for split_name in splits:
        csv_path = annotation_dir / f"{split_name}.csv"
        if csv_path.exists():
            process_split(csv_path, audio_dir, spectrogram_dir, split_name)
        else:
            print(f"Skipping {split_name} (not found): {csv_path}")


if __name__ == "__main__":
    from pathlib import Path

    # For direct execution, set these paths
    ANNOT_DIR = Path("Data/Annotations")
    AUDIO_DIR = Path("Data/Audio")
    SPEC_DIR = Path("Data/Spectrograms")

    main(ANNOT_DIR, AUDIO_DIR, SPEC_DIR)