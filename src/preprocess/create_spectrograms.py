import pandas as pd # type: ignore
from pathlib import Path
from .preprocess import generate_spectrogram  
import os

ROOT_PATH = Path("..")  
    
ANNOT_DIR = ROOT_PATH / "Data" / "Annotations"         # contains train.csv, val.csv, test.csv
AUDIO_DIR = ROOT_PATH / "Data" / "Audio"               # your raw wav files by category
SPEC_DIR = ROOT_PATH / "Data" / "Spectrograms"         # target directory

splits = ["train", "val", "test"]

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def process_split(split_name):
    print(f"\nProcessing {split_name}...")
    
    df = pd.read_csv(ANNOT_DIR / f"{split_name}.csv")

    spec_paths = []
    
    for idx, row in df.iterrows():
        
        category = row["category"]
        filename = row["filename"]

        print(f"[{idx+1}/{len(df)}] Processing: {filename} in {category}")
        
        audio_path = AUDIO_DIR / category / filename

        # Output path: spectrograms/split/category/file.pt
        output_dir = SPEC_DIR / split_name / category
        ensure_dir(output_dir)

        output_path = output_dir / f"{filename}.pt"

        # Skip if already exists
        if output_path.exists():
            print(f"    Skipping (already exists): {output_path}")
            spec_paths.append(str(output_path))
            continue

        try:
            generate_spectrogram(audio_path, output_path)
            spec_paths.append(str(output_path))
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            spec_paths.append(None)

    # Add the spectrogram path column
    df["spectrogram_path"] = spec_paths

    # Save modified CSV
    df.to_csv(ANNOT_DIR / f"{split_name}.csv", index=False)
    print(f"{split_name} complete. {df.shape[0]} spectrograms processed.")

def main():
    ensure_dir(SPEC_DIR)

    for split in splits:
        process_split(split)

if __name__ == "__main__":
    main()