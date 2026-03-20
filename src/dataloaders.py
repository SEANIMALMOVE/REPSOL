from torch.utils.data import DataLoader # type: ignore
from pathlib import Path

# Import the dataset in a way that works when running as a script
# or as a package. Try package-relative first, then fall back.
try:
    from .dataset import SpectrogramPTDataset
except Exception:
    try:
        from dataset import SpectrogramPTDataset
    except Exception:
        from src.dataset import SpectrogramPTDataset

### batching 16 torch samples

def get_dataloaders(
    spectrogram_root: Path,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
    cache_in_memory: bool = False,
):
    """
    Creates train / val / test dataloaders.

    spectrogram_root/
        train/
        val/
        test/
    """

    train_ds = SpectrogramPTDataset(spectrogram_root / "train", cache_in_memory=cache_in_memory)
    val_ds = SpectrogramPTDataset(spectrogram_root / "val", cache_in_memory=False)
    test_ds = SpectrogramPTDataset(spectrogram_root / "test", cache_in_memory=False)

    # Only enable persistent workers when num_workers>0
    pw = bool(persistent_workers and num_workers > 0)
    # prefetch_factor is only valid when multiprocessing (num_workers>0) is enabled
    pf = prefetch_factor if num_workers > 0 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=pw,
        prefetch_factor=pf,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=pw,
        prefetch_factor=pf,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=pw,
        prefetch_factor=pf,
    )

    return train_loader, val_loader, test_loader