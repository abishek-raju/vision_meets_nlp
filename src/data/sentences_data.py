import os.path
import pickle
from typing import Any, Callable, Optional, Tuple, List

import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from os.path import exists
from collections import Counter
import random
import torch
import re


class SentencesDataset(Dataset):
    """`sentences dataset

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    def __init__(
        self,
        train_file_path: str,
        vocab_file_path: str,
        seq_len: int = 20,
        n_vocab: int = 40000,

    ) -> None:

        self.train_file_path = train_file_path
        self.n_vocab = n_vocab
        self.vocab_file_path = vocab_file_path
        self.seq_len = seq_len

        self.sentences = open(self.train_file_path).read().lower().split('\n')
        self.special_chars = ',?;.:/*!+-()[]{}"\'&'
        self.sentences = [re.sub(f'[{re.escape(self.special_chars)}]', ' \g<0> ', s).split(' ') for s in self.sentences]
        self.sentences = [[w for w in s if len(w)] for s in self.sentences]

        if not exists(vocab_file_path):
            self.words = [w for s in self.sentences for w in s]
            self.vocab = Counter(self.words).most_common(self.n_vocab) #keep the N most frequent words
            self.vocab = [w[0] for w in self.vocab]
            open(self.vocab_file_path, 'w+').write('\n'.join(self.vocab))
        else:
            self.vocab = open(self.vocab_file_path).read().split('\n')
        
        dataset = self
        
        dataset.sentences = self.sentences
        dataset.vocab = self.vocab + ['<ignore>', '<oov>', '<mask>']
        dataset.vocab = {e:i for i, e in enumerate(dataset.vocab)} 
        dataset.rvocab = {v:k for k,v in dataset.vocab.items()}
        dataset.seq_len = self.seq_len
        
        #special tags
        dataset.IGNORE_IDX = dataset.vocab['<ignore>'] #replacement tag for tokens to ignore
        dataset.OUT_OF_VOCAB_IDX = dataset.vocab['<oov>'] #replacement tag for unknown words
        dataset.MASK_IDX = dataset.vocab['<mask>'] #replacement tag for the masked word prediction task

    #fetch data
    def __getitem__(self, index, p_random_mask=0.15):
        dataset = self
        
        #while we don't have enough word to fill the sentence for a batch
        s = []
        while len(s) < dataset.seq_len:
            s.extend(dataset.get_sentence_idx(index % len(dataset)))
            index += 1
        
        #ensure that the sequence is of length seq_len
        s = s[:dataset.seq_len]
        [s.append(dataset.IGNORE_IDX) for i in range(dataset.seq_len - len(s))] #PAD ok
        
        #apply random mask
        s = [(dataset.MASK_IDX, w) if random.random() < p_random_mask else (w, dataset.IGNORE_IDX) for w in s]
        
        return {'input': torch.Tensor([w[0] for w in s]).long(),
                'target': torch.Tensor([w[1] for w in s]).long()}

    #return length
    def __len__(self):
        return len(self.sentences)

    #get words id
    def get_sentence_idx(self, index):
        dataset = self
        s = dataset.sentences[index]
        s = [dataset.vocab[w] if w in dataset.vocab else dataset.OUT_OF_VOCAB_IDX for w in s] 
        return s