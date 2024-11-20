import os
import torch
from PIL import Image
from torchvision import transforms

from data.base_dataset import BaseDataset


class MusicDataset(BaseDataset):
    @staticmethod
    def default_loader(path: str, load_type: str = "RGB") -> Image:
        """
        Loads image in load_type, default RGB
        """

        return Image.open(path).convert(load_type)

    def __init__(self,
                 path: str,
                 color_channels: int = 3,
                 item_transform: transforms.Compose = transforms.Compose([transforms.ToTensor()]),
                 target_transform: any = None
                 ) -> None:
        super().__init__()

        self.load_type = "RGB" if color_channels == 3 else "L"
        self.transform = item_transform
        self.target_transform = target_transform
        self.data, self.classes = self._init_data(path)


    def _init_data(self, path: str):
        classes = dict()
        items = []

        for idx, class_path in enumerate(os.listdir(path)):
        # for idx, class_path in enumerate(["Lung_Opacity", "Normal"]):
            classes.update({idx: class_path})

            items_path = os.path.join(path, class_path)

            if os.path.exists(items_path):
                item_paths = os.listdir(items_path)
                items.extend([(os.path.join(items_path, item_path), idx) for item_path in item_paths])

        return items, classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> tuple[torch.Tensor, any, str]:
        item_path, label = self.data[idx]
        item = self.default_loader(item_path, load_type=self.load_type)

        if self.transform is not None:
            item = self.transform(item)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return item, label, item_path