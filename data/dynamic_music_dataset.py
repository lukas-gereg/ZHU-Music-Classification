import os
import torch
import librosa
import numpy as np
from PIL import Image
from typing import Callable
from torchvision import transforms
from audiomentations import Compose

from data.base_dataset import BaseDataset


class DynamicMusicDataset(BaseDataset):
    def __init__(self,
                 path: str,
                 music_transform:  Compose,
                 image_generation_method: Callable[[np.ndarray, float | int, Compose | None], Image.Image],
                 item_transform: transforms.Compose = transforms.Compose([transforms.ToTensor()]),
                 target_transform: any = None
                 ) -> None:
        super().__init__()
        self.music_transform = music_transform
        self.transform = item_transform
        self.target_transform = target_transform
        self.loader = image_generation_method
        self.data, self.classes = self._init_data(path)

    @staticmethod
    def _init_data(path: str):
        classes = dict()
        items = []

        for idx, class_path in enumerate(os.listdir(path)):
            classes.update({idx: class_path})

            items_path = os.path.join(path, class_path)

            if os.path.exists(items_path):
                item_paths = os.listdir(items_path)
                items.extend([((librosa.load(os.path.join(items_path, item_path)), os.path.join(items_path, item_path)), idx) for item_path in item_paths if item_path != 'jazz.00054.wav'])

        return items, classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> tuple[torch.Tensor, any, str]:
        ((y, sr), item_path), label = self.data[idx]
        item = self.loader(y,sr, self.music_transform)

        if self.transform is not None:
            item = self.transform(item)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return item, label, item_path