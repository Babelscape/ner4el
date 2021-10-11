import os
from pathlib import Path
from typing import Dict, List, Optional

import dotenv
import numpy as np
import pytorch_lightning as pl
import torch
import pickle
import json
from tqdm import tqdm
import pandas as pd
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf


def read_alias_table(filename):
    with open(filename, 'rb') as f:
        alias_table = pickle.load(f)
    return alias_table

def read_descriptions_dict(filename):
    descriptions = pd.read_csv(filename)
    descriptions_dict = descriptions.set_index("text_id").text.to_dict()
    return descriptions_dict

def read_item_counts_dict(filename):
    id_counts_df = pd.read_csv(filename).set_index('page_id')
    item_counts_dict = id_counts_df.counts.to_dict()
    return item_counts_dict

def read_dataset(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            data.append(json.loads(line))

    return data

def get_title_dict(filename):
    '''id_counts_df = pd.read_csv(filename)
    title_dict = id_counts_df.set_index("page_id").title.to_dict()'''
    with open(filename, 'rb') as f:
        title_dict = pickle.load(f)
    return title_dict

def read_id2ner_dict(filename):
    with open(filename, 'rb') as f:
        id2ner = pickle.load(f)
    return id2ner


def get_env(env_name: str, default: Optional[str] = None) -> str:
    """
    Safely read an environment variable.
    Raises errors if it is not defined or it is empty.

    :param env_name: the name of the environment variable
    :param default: the default (optional) value for the environment variable

    :return: the value of the environment variable
    """
    if env_name not in os.environ:
        if default is None:
            raise KeyError(f"{env_name} not defined and no default value is present!")
        return default

    env_value: str = os.environ[env_name]
    if not env_value:
        if default is None:
            raise ValueError(
                f"{env_name} has yet to be configured and no default value is present!"
            )
        return default

    return env_value


def load_envs(env_file: Optional[str] = None) -> None:
    """
    Load all the environment variables defined in the `env_file`.
    This is equivalent to `. env_file` in bash.

    It is possible to define all the system specific variables in the `env_file`.

    :param env_file: the file that defines the environment variables to use. If None
                     it searches for a `.env` file in the project.
    """
    dotenv.load_dotenv(dotenv_path=env_file, override=True)


STATS_KEY: str = "stats"


# Adapted from https://github.com/hobogalaxy/lightning-hydra-template/blob/6bf03035107e12568e3e576e82f83da0f91d6a11/src/utils/template_utils.py#L125
def log_hyperparameters(
    cfg: DictConfig,
    model: pl.LightningModule,
    trainer: pl.Trainer,
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.
    Additionally saves:
        - sizes of train, val, test dataset
        - number of trainable model parameters
    Args:
        cfg (DictConfig): [description]
        model (pl.LightningModule): [description]
        trainer (pl.Trainer): [description]
    """
    hparams = OmegaConf.to_container(cfg, resolve=True)

    # save number of model parameters
    hparams[f"{STATS_KEY}/params_total"] = sum(p.numel() for p in model.parameters())
    hparams[f"{STATS_KEY}/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams[f"{STATS_KEY}/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

    # disable logging any more hyperparameters for all loggers
    # (this is just a trick to prevent trainer from logging hparams of model, since we already did that above)
    trainer.logger.log_hyperparams = lambda params: None


# Load environment variables
load_envs()

# Set the cwd to the project root
PROJECT_ROOT: Path = Path(get_env("PROJECT_ROOT"))
assert (
    PROJECT_ROOT.exists()
), "You must configure the PROJECT_ROOT environment variable in a .env file!"

os.chdir(PROJECT_ROOT)
