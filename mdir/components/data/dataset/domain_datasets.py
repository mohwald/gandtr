import numpy as np

from torch.utils.data import Dataset

from daan.ml.tools import path_join
from daan.core.path_resolver import resolve_path
from .tuple_datasets import imread


class RandomDomainsPairDataset(Dataset):
    """Adjusts the size of two datasets into single size.

        Each iteration outputs (img_X, img_Y).

        The probability of sampling specific image depends on the size of the dataset.

        Config example:
        ```yml
        name: RandomDomainsPair
        dataset_X: data/train/retrieval-SfM-120k/dataset/train_day.txt
        dataset_Y: data/train/retrieval-SfM-120k/dataset/train_night.txt
        image_dir: data/train/retrieval-SfM-120k/ims/*
        size: 10000
        ```
    """

    loader_params = {}

    def __init__(self, data, transform, dataset_X, dataset_Y, image_dir, size, image_dir_Y=None):
        assert not data
        image_dir = resolve_path(image_dir)
        image_dir_Y = image_dir if image_dir_Y is None else resolve_path(image_dir_Y)

        with open(resolve_path(dataset_X)) as f:
            images_X = [x.strip() for x in f.readlines()]
        with open(resolve_path(dataset_Y)) as f:
            images_Y = [y.strip() for y in f.readlines()]

        self.images_X = [path_join(image_dir, x) for x in images_X]
        self.images_Y = [path_join(image_dir_Y, y) for y in images_Y]
        self.transform = transform
        self.size = int(size) if size is not None else min(len(self.images_X), len(self.images_Y))

    def prepare_epoch(self, network, device):
        self.idxs_X = list(np.random.randint(len(self.images_X), size=self.size))
        self.idxs_Y = list(np.random.randint(len(self.images_Y), size=self.size))
        return None

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        images = tuple([imread(self.images_X[self.idxs_X[idx]]), imread(self.images_Y[self.idxs_Y[idx]])])

        if self.transform:
            images = self.transform(*images)

        return images
