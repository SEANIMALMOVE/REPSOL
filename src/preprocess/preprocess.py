import librosa # type: ignore
import numpy as np # type: ignore
import torch # type: ignore
from pathlib import Path

# generate spectograms normalization

def generate_spectrogram(audio_path, output_path, target_sr=44000, n_mels=128):
    # Load audio using target sample rate
    audio, sr = librosa.load(audio_path, sr=target_sr)

    # Compute mel spectrogram

    n_fft = min(1024, max(256, (len(audio)//2)*2))
    hop_length = n_fft // 2

    # Compute mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0
    )

    # Convert to dB
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Convert to a tensor [1, n_mels, time]
    mel_tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)

    # Save the tensor
    output_path = Path(output_path)
    torch.save(mel_tensor, output_path)

    return output_path