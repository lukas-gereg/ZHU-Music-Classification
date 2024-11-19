from more_itertools import locate
from torch.utils.data import ConcatDataset

from data.base_dataset import BaseDataset


class CustomConcatDataset(ConcatDataset, BaseDataset):
    def __init__(self, datasets: list[BaseDataset]):
        super().__init__(datasets)

        classes_list = [dataset.classes for dataset in datasets]
        self.classes = dict()

        for class_dict in classes_list:
            collision_keys = list(set(self.classes.keys()).intersection(set(class_dict.keys())))

            equality_checks = [class_dict[collision_key] == self.classes[collision_key] for collision_key in collision_keys]
            value_collision_indexes = list(locate(equality_checks, lambda val: not val))

            assert all(equality_checks), \
                f"Meanings to labels {[collision_keys[i] for i in value_collision_indexes]} are in conflict with main meanings of class labels!"

            self.classes.update(class_dict)
