import logging
import datetime

import torch
import lightning.pytorch as pl

import hydra
from omegaconf import DictConfig, OmegaConf

from src.config import (
    instantiate_collection,
    log_hyperparameters,
    print_config,
    save_config,
    reload_original_config,
)
from src.utils import str2timedelta, timedelta2str

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver("eval", eval)


@hydra.main(version_base=None, config_path="configs/", config_name="config.yaml")
def main(cfg: DictConfig) -> None:

    # Check if the original config should be reloaded and print
    if cfg.full_resume:
        log.info("Reloading original config")
        cfg = reload_original_config(cfg)

        if cfg.train:
            log.info("Add 7 days of training to the timer callback")
            duration = str2timedelta(cfg.callbacks.timer.duration)
            duration += datetime.timedelta(days=7)
            cfg.callbacks.timer.duration = timedelta2str(duration)

    print_config(cfg)

    # Set some PyTorch configurations
    torch.set_float32_matmul_precision(cfg.float32_precision)

    if cfg.seed:
        log.info(f"Setting seed to: {cfg.seed}")
        pl.seed_everything(cfg.seed, workers=True)

    log.info("Instantiating the data module")
    datamodule = hydra.utils.instantiate(cfg.data)

    log.info("Instantiating the model")
    model = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating all callbacks")
    callbacks = instantiate_collection(cfg.callbacks)

    log.info("Instantiating the loggers")
    loggers = instantiate_collection(cfg.loggers)

    log.info("Instantiating the trainer")
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=loggers,
    )

    log.info("Saving the config")
    save_config(cfg)
    if loggers:
        log.info("Logging all hyperparameters")
        log_hyperparameters(cfg, model, trainer)

    if cfg.train:
        log.info("Starting training!")
        if cfg.full_resume:
            log.info("Resuming training from checkpoint")
            ckpt_path = cfg.ckpt_path
        else:
            ckpt_path = None
        trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

    if cfg.test:
        if cfg.full_resume or cfg.train:
            log.info("Starting testing!")
            trainer.test(model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
        else:
            log.warning("No existing checkpoint, skipping testing")


if __name__ == "__main__":
    main()
