import random
from typing import Optional, Sequence

import hydra
from hydra import utils
import numpy as np
import omegaconf
import pytorch_lightning as pl
from pytorch_lightning.core import datamodule
import torch
from pprint import pprint
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from src.pl_data.dataset import MyDataset

from src.common.utils import PROJECT_ROOT

from src.common.utils import *
from transformers import BertTokenizer
from transformers import XLMRobertaTokenizer


def worker_init_fn(id: int):
    """
    DataLoaders workers init function.

    Initialize the numpy.random seed correctly for each worker, so that
    random augmentations between workers and/or epochs are not identical.

    If a global seed is set, the augmentations are deterministic.

    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))
    random.seed(uint64_seed)


class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        transformer_name: str,
        alias_table_path: str,
        descriptions_dict_path: str,
        item_counts_dict_path: str,
        title_dict_path: str,
        id2ner_dict_path: str,
        version: str,
        negative_samples: bool,
        ner_negative_samples: bool,
        ner_representation: bool,
        ner_filter_candidates: bool,
        ner_constrained_decoding: bool,
        processed: bool,
    ):
        super().__init__()
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.transformer_name = transformer_name
        self.alias_table_path = str(PROJECT_ROOT / alias_table_path)
        self.descriptions_dict_path = str(PROJECT_ROOT / descriptions_dict_path)
        self.item_counts_dict_path = str(PROJECT_ROOT / item_counts_dict_path)
        self.title_dict_path = str(PROJECT_ROOT / title_dict_path)
        self.id2ner_dict_path = str(PROJECT_ROOT / id2ner_dict_path)
        self.version = version
        self.negative_samples = negative_samples
        self.ner_negative_samples = ner_negative_samples
        self.ner_representation = ner_representation
        self.ner_filter_candidates = ner_filter_candidates
        self.ner_constrained_decoding = ner_constrained_decoding
        self.processed = processed

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def prepare_data(self) -> None:
        # download only
        pass

    def setup(self, stage: Optional[str] = None):
        # Here you should instantiate your datasets, you may also split the train into train and validation if needed.

        train_data = read_dataset(str(PROJECT_ROOT / self.datasets.train.path))
        dev_data = read_dataset(str(PROJECT_ROOT / self.datasets.val.path))
        test_data = read_dataset(str(PROJECT_ROOT / self.datasets.test.path))
        print("Datasets loaded.")

        if "bert-" in self.transformer_name:
            self.tokenizer = BertTokenizer.from_pretrained(self.transformer_name)
            special_tokens_dict = {'additional_special_tokens': ['[E]','[\E]']}
            self.tokenizer.add_special_tokens(special_tokens_dict)
        elif "xlm" in self.transformer_name:
            self.tokenizer = XLMRobertaTokenizer.from_pretrained(self.transformer_name)
            special_tokens_dict = {'additional_special_tokens': ['[E]','[\E]']}
            self.tokenizer.add_special_tokens(special_tokens_dict)

        self.alias_table = read_alias_table(self.alias_table_path)
        print("Alias table loaded.")

        self.descriptions_dict = read_descriptions_dict(self.descriptions_dict_path)
        print("Descriptions dict loaded.")

        self.item_counts_dict = read_item_counts_dict(self.item_counts_dict_path)
        print("Item counts dict loaded.")

        self.title_dict = get_title_dict(self.title_dict_path)
        self.title_dict_reverse = dict((v, k) for (k, v) in self.title_dict.items())
        print("Title dict loaded.")

        self.id2ner = read_id2ner_dict(self.id2ner_dict_path)
        print("NER dict loaded.")
   

        if stage is None or stage == "fit":
            self.train_dataset = hydra.utils.instantiate(
                self.datasets.train,
                data=train_data,
                datamodule = self
            )

            self.val_dataset = hydra.utils.instantiate(
                self.datasets.val,
                data=dev_data,
                datamodule = self
            )

        if stage is None or stage == "test":
            self.test_dataset = hydra.utils.instantiate(
                self.datasets.test,
                data=test_data,
                datamodule = self
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            worker_init_fn=worker_init_fn,
            pin_memory=True,
            collate_fn=self.collate
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size.val,
            num_workers=self.num_workers.val,
            worker_init_fn=worker_init_fn,
            pin_memory=True,
            collate_fn=self.collate
        )

    def test_dataloader(self) -> Sequence[DataLoader]:
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.batch_size.test,
            num_workers=self.num_workers.test,
            worker_init_fn=worker_init_fn,
            pin_memory=True,
            collate_fn=self.collate
        )

    def collate(self, elems: List[tuple]) -> List[tuple]:
        mentions, positions, candidates, descriptions, labels = list(zip(*elems))
    
        pad_mentions = pad_sequence(mentions, batch_first=True, padding_value=0)
        pad_candidates = pad_sequence(candidates, batch_first=True, padding_value=0)
        pad_descriptions = pad_sequence(descriptions, batch_first=True, padding_value=0)
 
        return pad_mentions, positions, pad_candidates, pad_descriptions, labels

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.datasets=}, "
            f"{self.num_workers=}, "
            f"{self.batch_size=})"
        )


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )
    datamodule.setup()


if __name__ == "__main__":
    main()
