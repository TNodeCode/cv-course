from pathlib import Path
from typing import Tuple

import numpy as np
from torch.utils.data import Dataset


class AlbumGenreDataset(Dataset):
    def __init__(self, root: str, split: str):
        self.root = Path(root)
        assert self.root.exists(), f'Root path {self.root} does not exist'
        self.split = split
        assert self.split in ['train', 'val', 'test'], 'split must be train, val or test'

        self.X_audio = np.load(str(self.root / f'X_{self.split}_audio.npy'))
        self.X_vision = np.load(str(self.root / f'X_{self.split}_visual.npy'))
        self.y = np.load(str(self.root / f'y_{self.split}.npy')).astype(np.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        return (self.X_audio[idx], self.X_vision[idx]), self.y[idx]
