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
from src.data.pizza_steak_sushi import Pizza_Steak_Sushi


class Pizza_Steak_Sushi_datamodule(LightningDataModule):
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
        train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations

        # self.train_transforms = transforms.Compose(
        #     [
        #     transforms.ToTensor(), 
        #     transforms.Normalize((0.49139968,0.48215827,0.44653124), 
        #                             (0.24703233,0.24348505,0.26158768))]
        # )
        self.train_transforms = A.Compose([
                A.Normalize(
                mean=[136,106,84],
                std=[70,68,68],
                ),
                A.Resize(224, 224),
                ToTensorV2()

        ])

        # self.test_transforms = transforms.Compose(
        #     [transforms.ToTensor(), transforms.Normalize((0.49139968,0.48215827,0.44653124), 
        #                                                     (0.24703233,0.24348505,0.26158768))]
        # )

        self.test_transforms = A.Compose([
            A.Normalize(
                mean=[136,106,84],
                std=[70,68,68],
                ),
                A.Resize(224, 224),
            ToTensorV2()

        ])

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self):
        return 3
    
    def calculate_mean_std_dev(self):
        """
        To calculate mean and std deviation of the given dataset
        """
        if not self.data_train:
            self.prepare_data()
            self.setup()
        # simple_transforms = transforms.Compose([
        #                               #  transforms.Resize((28, 28)),
        #                               #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
        #                                transforms.ToTensor(),
        #                               #  transforms.Normalize((0.1307,), (0.3081,)) # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values. 
        #                                # Note the difference between (0.1307) and (0.1307,)
        #                                ])

        simple_transforms = A.Compose([A.Resize(224, 224),
                ToTensorV2()

        ])

        pizza_steak_sushi = Pizza_Steak_Sushi(self.hparams.data_dir + "pizza_steak_sushi/train", transform=simple_transforms)

        imgs = [item[0] for item in pizza_steak_sushi] # item[0] and item[1] are image and its label
        imgs = torch.stack(imgs, dim=0).numpy()

        # calculate mean over each channel (r,g,b)
        mean_r = imgs[:,0,:,:].mean()
        mean_g = imgs[:,1,:,:].mean()
        mean_b = imgs[:,2,:,:].mean()
        print('These are the mean values to update.')
        print(mean_r,mean_g,mean_b)

        # calculate std over each channel (r,g,b)
        std_r = imgs[:,0,:,:].std()
        std_g = imgs[:,1,:,:].std()
        std_b = imgs[:,2,:,:].std()
        print('These are the standard deviation values to update.')
        print(std_r,std_g,std_b)
    
    def get_sample_images(self,number_of_samples = 10):
        """Return sample images
            number_of_samples: int: 10
        """
        if not self.data_train:
            self.prepare_data()
            self.setup()

        # simple_transforms = transforms.Compose([
        #                                transforms.Resize((64, 64)),
        #                               #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
        #                                transforms.ToTensor(),
        #                               #  transforms.Normalize((0.1307,), (0.3081,)) # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values. 
        #                                # Note the difference between (0.1307) and (0.1307,)
        #                                ])
        simple_transforms = A.Compose([A.Resize(224, 224),
                ToTensorV2()

        ])
        pizza_steak_sushi = Pizza_Steak_Sushi(self.hparams.data_dir + "pizza_steak_sushi/train", transform=simple_transforms)

        imgs = [item[0] for item in pizza_steak_sushi] 

        self.sample_dict = {}
        self.idx_to_key = {}

        for key in pizza_steak_sushi.class_to_idx:
            # print(key, '->', cifar_trainset.class_to_idx[key])
            self.sample_dict[key] = []
            self.idx_to_key[pizza_steak_sushi.class_to_idx[key]] = key

        for i,target in enumerate(pizza_steak_sushi.targets):
            # print(i,target,idx_to_key[target])
            self.sample_dict[self.idx_to_key[target]].append({"index" : i,"img" : imgs[i]})
        
        images_to_display = []
        for key in self.sample_dict:
            for img in self.sample_dict[key][:number_of_samples]:
                images_to_display.append(img["img"])

        
        grid_img = torchvision.utils.make_grid(images_to_display, nrow=number_of_samples)
        return grid_img
    
    def get_sample_images_transformed(self,number_of_samples = 10):
        """Return sample images
            number_of_samples: int: 10
        """
        if not self.data_train:
            self.prepare_data()
            self.setup()

        # simple_transforms = transforms.Compose([
        #                                transforms.Resize((64, 64)),
        #                               #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
        #                                transforms.ToTensor(),
        #                               #  transforms.Normalize((0.1307,), (0.3081,)) # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values. 
        #                                # Note the difference between (0.1307) and (0.1307,)
        #                                ])
        simple_transforms = A.Compose([A.Resize(224, 224),
                ToTensorV2()

        ])

        pizza_steak_sushi = Pizza_Steak_Sushi(self.hparams.data_dir + "pizza_steak_sushi/train", transform=simple_transforms)

        imgs = [item[0] for item in pizza_steak_sushi] 

        self.sample_dict = {}
        self.idx_to_key = {}

        for key in pizza_steak_sushi.class_to_idx:
            # print(key, '->', cifar_trainset.class_to_idx[key])
            self.sample_dict[key] = []
            self.idx_to_key[pizza_steak_sushi.class_to_idx[key]] = key

        for i,target in enumerate(pizza_steak_sushi.targets):
            # print(i,target,idx_to_key[target])
            self.sample_dict[self.idx_to_key[target]].append({"index" : i,"img" : imgs[i]})
        
        images_to_display = []
        for key in self.sample_dict:
            for img in self.sample_dict[key][:number_of_samples]:
                images_to_display.append(img["img"])

        
        grid_img = torchvision.utils.make_grid(images_to_display, nrow=number_of_samples)
        return grid_img

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
        if not self.data_train and not self.data_val and not self.data_test:
            trainset = Pizza_Steak_Sushi(self.hparams.data_dir + "pizza_steak_sushi/train", transform=self.test_transforms)
            testset = Pizza_Steak_Sushi(self.hparams.data_dir + "pizza_steak_sushi/test", transform=self.test_transforms)
            dataset = ConcatDataset(datasets=[trainset, testset])
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
    _ = CIFAR10DataModule()
