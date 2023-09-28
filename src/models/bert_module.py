from typing import Any

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
import torch.nn as nn
import torch.nn.functional as F
from src.models.bert import Transformer

class Bert_module(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Initialization (__init__)
        - Train Loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        n_code: int = 8, 
        n_heads: int = 8, 
        embed_size: int = 128, 
        inner_ff_size: int = 128 * 4, 
        n_embeddings: int = 100,
        seq_len: int = 20, 
        dropout: float = 0.1,

    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        dropout_value = 0.1

        self.model_ = Transformer(n_code, n_heads, embed_size, inner_ff_size, n_embeddings, seq_len, dropout)

        
        self.dropout = nn.Dropout(dropout_value)


        # loss function



        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor):

        x = self.model_.forward(x)
        
        return x

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        # self.val_loss.reset()
        # self.val_acc.reset()
        # self.val_acc_best.reset()
        pass
    
    # def on_train_end(self):
    #     print("**********************************End of training")

    def model_step(self, batch: Any):
        # x, y = batch
        masked_input = batch['input']
        masked_target = batch['target']
        output = self.forward(masked_input)
        output_v = output.view(-1,output.shape[-1])
        target_v = masked_target.view(-1,1).squeeze()
        loss = self.criterion(output_v, target_v)
        return loss, output_v, target_v

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        # self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/lr", self.get_lr(), on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        # loss, preds, targets = self.model_step(batch)

        # # update and log metrics
        # self.val_loss(loss)
        # self.val_acc(preds, targets)
        # self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        pass

    def on_validation_epoch_end(self):
        # acc = self.val_acc.compute()  # get current val acc
        # self.val_acc_best(acc)  # update best so far val acc
        # # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # # otherwise metric would be reset by lightning after each epoch
        # self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
        pass

    def test_step(self, batch: Any, batch_idx: int):
        # loss, preds, targets = self.model_step(batch)

        # # update and log metrics
        # self.test_loss(loss)
        # self.test_acc(preds, targets)
        # self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        pass

    def on_test_epoch_end(self):
        pass

    def get_lr(self):
        if self.lr_schedulers():
            for param_group in self.lr_schedulers().optimizer.param_groups:
                return param_group['lr']
        else:
            for param_group in self.optimizers().optimizer.param_groups:
                return param_group['lr']
    
    def configure_loss(self,ignore_index):
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train/loss",
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = CIFAR10LitModule(None, None, None)
