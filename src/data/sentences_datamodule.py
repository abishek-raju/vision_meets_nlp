from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

from torchvision.transforms import transforms
from torchvision import datasets
import numpy
import torchvision
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from src.data.sentences_data import SentencesDataset


class Sentences_Datamodule(LightningDataModule):
    """Example of LightningDataModule for Pizza_Steak_Sushi dataset.

    A DataModule implements 6 key methods:
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        train_file_path: str = None,
        vocab_file_path: str = None,
        train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        seq_len: int = 20,
        n_vocab: int = 40000,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
    
    def get_samples(self,number_of_samples = 10):
        """Return sample images
            number_of_samples: int: 10
        """
        if not self.data_train:
            self.prepare_data()
            self.setup()

        sentences_data = SentencesDataset(self.hparams.train_file_path,self.hparams.vocab_file_path,
                                        self.hparams.seq_len,self.hparams.n_vocab)

        sample_count = 0
        text_samples = []
        output_samples = []
        for item in sentences_data:
            if sample_count <= number_of_samples:
                text_samples.append(item["input"])
                output_samples.append(item["target"])
                sample_count += 1 
            else:
                break
        res = dict((v,k) for k,v in sentences_data.vocab.items())
        text_samples = [' '.join([res[i.item()] for i in j]) for j in text_samples]
        output_samples = [' '.join([res[i.item()] for i in j]) for j in output_samples]



        return text_samples,output_samples
    
    def get_sample_images_transformed(self,number_of_samples = 10):
        """Return sample images
            number_of_samples: int: 10
        """
        if not self.data_train:
            self.prepare_data()
            self.setup()

        sentences_data = SentencesDataset(self.hparams.train_file_path,self.hparams.vocab_file_path,
                                        self.hparams.seq_len,self.hparams.n_vocab)

        text_samples = [item["input"] for item in sentences_data[:number_of_samples]] 

        return text_samples

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val:
            trainset = SentencesDataset(self.hparams.train_file_path,self.hparams.vocab_file_path,
                                        self.hparams.seq_len,self.hparams.n_vocab)
            testset = SentencesDataset(self.hparams.train_file_path,self.hparams.vocab_file_path,
                                        self.hparams.seq_len,self.hparams.n_vocab)
            # self.data_val, self.data_test = random_split(
            #     dataset=testset,
            #     lengths=self.hparams.train_val_test_split,
            #     generator=torch.Generator().manual_seed(42),
            # )
            self.data_train = trainset
            self.data_val = testset
            self.data_test = testset

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    _ = Sentences_Datamodule()
