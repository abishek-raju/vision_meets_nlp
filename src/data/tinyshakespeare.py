import os.path
import pickle
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch


class TinyShakespeare(Dataset):

    def __init__(
        self,
        root: str,
        filename: str,
        train_ratio:float,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        assert os.path.isfile(root + filename)

        with open(root + filename,'r',encoding='utf-8') as f:
                text = f.read()
            
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)

        # create a mapping from characters to integers
        self.stoi = { ch:i for i,ch in enumerate(self.chars)}
        self.itos = { i:ch for i,ch in enumerate(self.chars)}

        self.encode = lambda s: [self.stoi[c] for c in s]
        self.decode = lambda l: ''.join([self.itos[i] for i in l])

        data = torch.tensor(self.encode(text), dtype=torch.long)
        # n = int(0.9*len(data)) # first 90% will be train, rest val
        self.block_size = 8
        n = int(train_ratio*len(data))
        if train:
            self.data = data[:n]
        else:
            self.data = data[n:]




    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        src = self.data[index : index + self.block_size]
        tgt = self.data[index+1 : index + self.block_size + 1]

        return src, tgt

    def __len__(self) -> int:
        return (len(self.data) - self.block_size)
        
