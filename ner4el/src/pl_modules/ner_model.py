from typing import Any, Dict, Sequence, Tuple, Union, List

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from torch import nn
from omegaconf import DictConfig
from torch._C import device
from torch.optim import Optimizer
import os
import math
from sklearn.metrics import f1_score


from src.common.utils import PROJECT_ROOT
from src.common.utils import *

from transformers import BertTokenizer, BertModel, BertConfig

model_name_ner = 'bert-base-uncased'


class MyNERModel(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        #self.save_hyperparameters()  # populate self.hparams with args and kwargs automagically!
        global bert_tokenizer_ner

        id2ner_dict_path = "data/id2ner_dict.pickle"
        id2ner_dict_path = str(PROJECT_ROOT / id2ner_dict_path)
        id2ner = read_id2ner_dict(id2ner_dict_path)

        labels_vocab = {}
        i = 0
        for v in id2ner.values():
            if v not in labels_vocab:
                labels_vocab[v] = i
                i+=1

 
        bert_config_ner = BertConfig.from_pretrained(model_name_ner, output_hidden_states=True)
        bert_tokenizer_ner = BertTokenizer.from_pretrained(model_name_ner)
        special_tokens_dict = {'additional_special_tokens': ['[E]','[\E]']}
        num_added_toks = bert_tokenizer_ner.add_special_tokens(special_tokens_dict)
        bert_model_ner = BertModel.from_pretrained(model_name_ner, config=bert_config_ner)
        bert_model_ner.resize_token_embeddings(len(bert_tokenizer_ner))


        self.mention_encoder = bert_model_ner

        self.cosine_similarity = nn.CosineSimilarity(dim=-1, eps=1e-6)
        
        self.dropout = nn.Dropout(0.5)
            
        self.linear = nn.Linear(768, len(labels_vocab))

    def forward(
        self, mentions, positions, mask1, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Method for the forward pass.
        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.
        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.
        """

        embedding_mention = self.mention_encoder.forward(mentions, mask1)[0] #16x64x768
        embedding_mention2 = embedding_mention.gather(1, positions.reshape(-1, 1, 1).repeat(1, 1, 768)).squeeze(1)
        embedding_mention2 = self.dropout(embedding_mention2) #16x768
        
        predictions = self.linear(embedding_mention2)
                        
        return predictions

    def step(self, batch: Any, batch_idx: int, dataset_type:str):
        softmax_function = nn.Softmax(dim=1)

        mentions, positions, candidates, descriptions, labels = batch
        positions = torch.tensor(positions, device=self.device)

        mask1 = self.padding_mask(mentions)
        predictions = self.forward(mentions, positions, mask1)
        predictions = softmax_function(predictions)

        return {
            "pred": predictions,
        }
                    
                

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        step_output = self.step(batch, batch_idx, "train")
        return step_output

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        step_output = self.step(batch, batch_idx, "dev")
        return step_output

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        step_output = self.step(batch, batch_idx, "test")
        return step_output

        

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        Return:
            Any of these 6 options.
            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler'
              key whose value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency' key.
            - None - Fit will run without any optimizer.
        """
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return [opt], [scheduler]

    def padding_mask(self, batch):
        padding = torch.ones_like(batch)
        padding[batch == 0] = 0
        padding = padding.type(torch.int64)
        return padding

    def normalize(self, m):
        row_min, _ = m.min(dim=1, keepdim=True)
        row_max, _ = m.max(dim=1, keepdim=True)
        return (m - row_min) / (row_max - row_min)


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    )


if __name__ == "__main__":
    main()
