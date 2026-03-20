import torch    # type: ignore
from torch.utils.data import Dataset # type: ignore
import torchaudio # type: ignore
import os

### Get one torch sample from 1 spectogram

# Custom Dataset for loading spectrogram .pt files
class SpectrogramPTDataset(Dataset):
    def __init__(self, root_dir, transform=None, cache_in_memory: bool = False):
        self.root_dir = root_dir
        self.transform = transform
        self.cache_in_memory = bool(cache_in_memory)

        self.samples = []
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)
            for fname in os.listdir(class_path):
                if fname.endswith(".pt"):
                    self.samples.append((
                        os.path.join(class_path, fname),
                        self.class_to_idx[class_name]
                    ))

        # Optional in-memory cache to avoid repeated disk reads when dataset fits RAM
        self._cache = {} if self.cache_in_memory else None
        if self.cache_in_memory:
            for idx, (path, _lbl) in enumerate(self.samples):
                try:
                    t = torch.load(path)
                    # ensure channel-first format is preserved as on-disk
                    self._cache[idx] = t
                except Exception:
                    # if a single file fails to load, skip caching that one
                    pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path, label = self.samples[idx]

        if self.cache_in_memory and self._cache is not None and idx in self._cache:
            tensor = self._cache[idx]
        else:
            tensor = torch.load(path)  # loads spectrogram tensor

        # ensure tensor shape is [C, H, W]
        if tensor.dim() == 2:               # [H, W]
            tensor = tensor.unsqueeze(0)    # → [1, H, W]
        if tensor.dim() == 3 and tensor.shape[0] not in [1,3]:
            tensor = tensor.permute(2, 0, 1)

        # NOTE: offline normalization should be applied when generating .pt files.
        # Removed online normalization for faster training.

        # ------------------------------------------------------------
        # FIX: Pad or crop all spectrograms to a fixed width
        TARGET_WIDTH = 400
        _, H, W = tensor.shape

        if W < TARGET_WIDTH:
            pad_amount = TARGET_WIDTH - W
            tensor = torch.nn.functional.pad(tensor, (0, pad_amount))
        else:
            tensor = tensor[:, :, :TARGET_WIDTH]
        # ------------------------------------------------------------

        if self.transform:
            tensor = self.transform(tensor)

        return tensor.float(), label