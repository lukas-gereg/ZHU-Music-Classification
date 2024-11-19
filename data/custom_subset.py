from collections.abc import Sequence

from torch.utils.data import Subset

from data.base_dataset import BaseDataset


class CustomSubset(Subset, BaseDataset):
    def __init__(self, dataset: BaseDataset, indices: Sequence[int]):
        super().__init__(dataset, indices)
        self.classes = dataset.classes

    def find_y_by_index(self, idx: int):
        original_idx = self.indices[idx]
        _, label = self.dataset.data[original_idx]

        if self.dataset.target_transform is not None:
            label = self.dataset.target_transform(label)

        return label