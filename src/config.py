from pathlib import Path
import rich
import rich.tree
import rich.syntax
import hydra
import logging
from typing import List, Any, Sequence
from omegaconf import OmegaConf, DictConfig
import wandb

from lightning.pytorch.utilities.rank_zero import rank_zero_only
import lightning.pytorch as pl

log = logging.getLogger(__name__)


@rank_zero_only
def reload_original_config(cfg: OmegaConf) -> OmegaConf:
    """
    Replaces the config with the one stored at the specified location,
    but keeps some flags unchanged:
        - cfg.full_resume
        - cfg.train
        - cfg.test
        - cfg.job_id
        - cfg.seed
        - cfg.model.job_id
        - cfg.model.do_test_loop
        - cfg.model.loss_delay_n

    Args:
        cfg: The current config with path to the config to reload.

    Returns:
        orig_cfg: The original config with some flags unchanged.
    """

    orig_cfg = OmegaConf.load(Path(cfg.cfg_path))

    orig_cfg.full_resume = cfg.full_resume
    orig_cfg.train = cfg.train
    orig_cfg.test = cfg.test
    orig_cfg.job_id = cfg.job_id
    orig_cfg.seed = cfg.seed
    orig_cfg.model.job_id = cfg.model.job_id
    orig_cfg.model.do_test_loop = cfg.model.do_test_loop
    orig_cfg.model.loss_delay_n = cfg.model.loss_delay_n

    return orig_cfg


@rank_zero_only
def print_config(
    cfg: DictConfig,
    print_order: Sequence[str] = (
        "data",
        "model",
        "callbacks",
        "loggers",
        "trainer",
        "paths",
    ),
    resolve: bool = True,
) -> None:
    """
    Prints content of DictConfig using Rich library and its tree structure.

    Args:
        cfg: Configuration composed by Hydra.
        print_order: Determines in what order config components are printed.
        resolve: Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # Add the fields from 'print_order' to the queue
    for field in print_order:
        queue.append(field) if field in cfg else log.warning(
            f"Field '{field}' not found in config. Skipping '{field}' printing..."
        )

    # Add all the other fields to the queue (not specified in 'print_order')
    for field in cfg:
        if field not in queue:
            queue.insert(0, field)

    # Generate the config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # Print the config tree
    rich.print(tree)


@rank_zero_only
def save_config(cfg: DictConfig, suffix="") -> None:
    """Saves the config to the output directory"""

    # In order to be able to resume the wandb logger session later,
    # one needs to keep track of the current run id.
    if hasattr(cfg, "loggers"):
        if hasattr(cfg.loggers, "wandb"):
            if wandb.run is not None:
                cfg.loggers.wandb.id = wandb.run.id

    path = Path(cfg.cfg_path).parent
    path /= Path(cfg.cfg_path).stem + suffix + Path(cfg.cfg_path).suffix
    OmegaConf.save(
        cfg,
        path,
        resolve=True,
    )


@rank_zero_only
def log_hyperparameters(
    cfg: DictConfig,
    model: pl.LightningModule,
    trainer: pl.Trainer,
) -> None:
    """
    Passes the config dict to the trainer's logger.
    Also calculates the number of parameters of the model.

    Args:
        cfg: The configuration dictionary.
        model: The model to calculate the number of parameters.
        trainer: The trainer object to log the hyperparameters.
    """

    # Convert the config object to a hyperparameter dict
    hparams = OmegaConf.to_container(cfg, resolve=True)

    # Calculate the number of trainable parameters in the model and add it
    hparams["model/params/total"] = sum(
        p.numel() for p in model.parameters()
    )
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    trainer.logger.log_hyperparams(hparams)


def instantiate_collection(cfg_coll: DictConfig) -> List[Any]:
    """
    Uses hydra to instantiate a collection of classes and return a list
    of instantiated objects.

    Args:
        cfg_coll: A collection of configurations.

    Returns:
        objs: A list of instantiated objects.
    """
    objs = []

    if not cfg_coll:
        log.warning("List of configs is empty")
        return objs

    if not isinstance(cfg_coll, DictConfig):
        raise TypeError("List of configs must be a DictConfig!")

    for _, cb_conf in cfg_coll.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating <{cb_conf._target_}>")
            objs.append(hydra.utils.instantiate(cb_conf))

    return objs
