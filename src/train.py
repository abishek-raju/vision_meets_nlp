from typing import List, Optional, Tuple

import hydra
import lightning as L
import pyrootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from src import utils

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    if cfg.print_mean_and_std:
        datamodule.calculate_mean_std_dev()
    
    if not datamodule.data_train:
        datamodule.prepare_data()
        datamodule.setup()
    
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model,n_embeddings=len(datamodule.data_train.vocab))
    model.configure_loss(datamodule.data_train.IGNORE_IDX)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    if cfg.visualizations.generate and not cfg.trainer.fast_dev_run:
        log.info(f"Displaying sample")
        text_samples,output_samples = datamodule.get_samples(cfg.visualizations.display_number_of_sample)
        for count,i in enumerate(zip(text_samples,output_samples)):
            logger[0].experiment.add_text("Sample " + str(count), i[0] + "  \n", 0)
            logger[0].experiment.add_text("Sample " + str(count), i[1] + "  \n", 0)

        # log.info(f"Displaying Transformed sample image")
        # sample_image_grid = datamodule.get_sample_images_transformed(cfg.visualizations.display_number_of_sample)
        # logger[0].experiment.add_image("Sample Transformed input", sample_image_grid)
    else:
        log.info(f"Skipping Displaying sample image")
        log.info(f"Skipping Displaying Transformed sample image")
        

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("compile"):
        log.info("Compiling model!")
        model = torch.compile(model)
    
    if cfg.find_lr.find_lr:
        log.info("Finding optimal LR")
        optimizer = model.configure_optimizers()["optimizer"]
        criterion = model.criterion
        if not datamodule.data_train:
            datamodule.prepare_data()
            datamodule.setup()
        end_lr = cfg.find_lr.end_lr
        num_iter = cfg.find_lr.num_iter
        utils.lr_finder.get_lr(model,optimizer,criterion,datamodule.train_dataloader(),end_lr,num_iter,device = model.device)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    if cfg.visualizations.generate_output_images and cfg.visualizations.correctly_identified and cfg.visualizations.generate and not cfg.trainer.fast_dev_run:
        log.info(f"Generating {cfg.visualizations.correctly_identified} correctly identified samples")
        (correct_classified,
        mis_classified) = utils.model_performance.get_correct_and_misclassified_images_grid(model,
                                                                                datamodule.test_dataloader())
        logger[0].experiment.add_image("Correct Classified", correct_classified)
        logger[0].experiment.add_image("Mis Classified", mis_classified)

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
