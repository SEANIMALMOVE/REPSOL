import pandas as pd # type: ignore
from pathlib import Path
import soundfile as sf  # type: ignore

# annotation CSV
def build_annotation_file(audio_dir, annot_path):
    records = []

    for cat in audio_dir.iterdir():
        if not cat.is_dir():
            continue
        for audio_file in cat.rglob("*.wav"):
            try:
                info = sf.info(audio_file)
                duration = info.frames / info.samplerate
                if duration > 0:
                    records.append({
                        "category": cat.name,
                        "filename": audio_file.name,
                        "duration_sec": duration,
                        "sample_rate": info.samplerate
                    })
            except Exception as e:
                print(f"Error reading {audio_file}: {e}")

    df = pd.DataFrame(records)
    df.to_csv(annot_path, index=False)
    print("Total files processed:", len(df))

    df