import os.path
import pickle
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image

from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from torchvision.datasets.vision import VisionDataset
from torchvision import datasets

class Pizza_Steak_Sushi(VisionDataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
    ) -> None:

        super().__init__(root, transform=transform)

        self.train = train
        self.transform = transform  # training set or test set



        self.data: Any = []
        self.targets = []

        self.data = datasets.ImageFolder(root)
        self.transform = transform
        self.targets = self.data.targets

        self.class_to_idx = self.data.class_to_idx



    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index][0], self.data[index][1]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img)
        img = np.array(img)

        if self.transform is not None:
            img = self.transform(image=img)

        return img["image"], target

    def __len__(self) -> int:
        return len(self.data)