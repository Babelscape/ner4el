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
from pl_modules.ner_model import MyNERModel
from src.common.utils import *

from src.common.utils import PROJECT_ROOT

from transformers import BertTokenizer, BertModel, BertConfig


ner_model = MyNERModel().cuda()
ner_model.load_state_dict(torch.load("/mnt/data/NER_for_EL/ner4el/wandb/ner_predictor_only_aida.pt"))

id2ner_dict_path = "data/id2ner_dict.pickle"
id2ner_dict_path = str(PROJECT_ROOT / id2ner_dict_path)

id2ner = read_id2ner_dict(id2ner_dict_path)

labels_vocab = {}
i = 0
for v in id2ner.values():
    if v not in labels_vocab:
        labels_vocab[v] = i
        i+=1

class MyModel(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()  # populate self.hparams with args and kwargs automagically!

        self.bert_config = BertConfig.from_pretrained(
            self.hparams.transformer_name, output_hidden_states=True
        )

        self.bert_tokenizer = BertTokenizer.from_pretrained(
            self.hparams.transformer_name
        )

        '''self.mention_encoder = BertModel.from_pretrained(
            self.hparams.transformer_name, config=self.bert_config
        )

        self.entity_encoder = BertModel.from_pretrained(
            self.hparams.transformer_name, config=self.bert_config
        )

        special_tokens_dict = {"additional_special_tokens": ["[E]", "[\E]"]}
        self.bert_tokenizer.add_special_tokens(special_tokens_dict)
        self.mention_encoder.resize_token_embeddings(len(self.bert_tokenizer))
        self.entity_encoder.resize_token_embeddings(len(self.bert_tokenizer))'''

        #------------------------------------------------------------------------------------
        # Uncomment the above alternative block of code to use the dual-encoder architecture.
        # However, we observed that using a single encoder, we obtain very similar 
        # performances, while we have much shorter training times and less computational
        # resources are required

        self.bert_model = BertModel.from_pretrained(
            self.hparams.transformer_name, config=self.bert_config
        )

        special_tokens_dict = {"additional_special_tokens": ["[E]", "[\E]"]}
        self.bert_tokenizer.add_special_tokens(special_tokens_dict)
        self.bert_model.resize_token_embeddings(len(self.bert_tokenizer))

        self.mention_encoder = self.bert_model
        self.entity_encoder = self.bert_model
        
        #------------------------------------------------------------------------------------

        self.cosine_similarity = nn.CosineSimilarity(dim=-1, eps=1e-6)

        self.dropout = nn.Dropout(self.hparams.dropout)
        self.loss_function = nn.CrossEntropyLoss()

    def forward(
        self, mentions, positions, descriptions, mask1, mask2, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Method for the forward pass.
        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.
        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.
        """
        num_candidates = descriptions.shape[1]

        embedding_mention = self.mention_encoder.forward(mentions, mask1)[0]  # 16x64x768
        embedding_mention2 = (embedding_mention.gather(1, positions.reshape(-1, 1, 1).repeat(1, 1, self.bert_config.hidden_size),).squeeze(1))
        # embedding_mention2 = self.dropout(embedding_mention2) #16x768

        descriptions = descriptions.flatten(start_dim=0, end_dim=1)  # 16x20x64 -> 320x64
        mask2 = mask2.flatten(start_dim=0, end_dim=1)

        embedding_entities = self.entity_encoder.forward(descriptions, mask2)[0]  # 320x64x768
        # embedding_entities = self.dropout(embedding_entities)
        embedding_entities = embedding_entities[:, 0, :].squeeze(1)  # 320x768
        embedding_entities = embedding_entities.reshape(embedding_mention2.shape[0], num_candidates, -1)  # 16x20x768

        # mentions 16x768, entities #16x20x768
        embedding_mention2 = embedding_mention2.unsqueeze(1)  # 16x1x768
        embedding_mention2 = embedding_mention2.repeat_interleave(num_candidates, dim=1)  # 16x20x768

        similarities = self.cosine_similarity(embedding_mention2, embedding_entities)

        return similarities

    def step(self, batch: Any, batch_idx: int, dataset_type:str):
        
        if not self.hparams.ner_constrained_decoding:
            mentions, positions, candidates, descriptions, labels = batch
            positions = torch.tensor(positions, device=self.device)

            mask1 = self.padding_mask(mentions)
            mask2 = self.padding_mask(descriptions)

            similarities = self.forward(mentions, positions, descriptions, mask1, mask2)
            normalized_similarities = self.normalize(similarities)

            gold = torch.zeros(normalized_similarities.shape[0])
            for i in range(descriptions.shape[0]):  # i is the index of the batch
                for j in range(descriptions.shape[1]):
                    if candidates[i][j] == labels[i]:
                        gold[i] = j
            gold = gold.type(torch.LongTensor).to(self.device)


            loss = self.loss_function(normalized_similarities, gold)

            all_predictions = list()
            all_labels = list()
            
            if dataset_type != "train":
                for i in range(len(similarities)):
                    current_candidates = list(filter(lambda x: x!=0, candidates[i]))
                    normalized_similarities_line = normalized_similarities[i][:len(current_candidates)]
                    all_predictions.append(int(candidates[i][torch.argmax(normalized_similarities_line)]))
                    all_labels.append(int(labels[i]))

            if dataset_type=="train":
                if not math.isnan(loss):
                    return {
                        "loss": loss,
                        "pred": all_predictions,
                        "gold": all_labels,
                    }
                else:
                    return None
            else:
                return {
                        "pred": all_predictions,
                        "gold": all_labels,
                    }

        else:
            softmax_function = nn.Softmax(dim=1)

            mentions, positions, candidates, descriptions, labels = batch
            positions = torch.tensor(positions, device=self.device)

            mask1 = self.padding_mask(mentions)
            mask2 = self.padding_mask(descriptions)

            similarities = self.forward(mentions, positions, descriptions, mask1, mask2)
            normalized_similarities = self.normalize(similarities)

            predictions = ner_model.forward(mentions, positions, mask1)
            predictions = softmax_function(predictions)

            gold = torch.zeros(normalized_similarities.shape[0])
            for i in range(descriptions.shape[0]):  # i is the index of the batch
                for j in range(descriptions.shape[1]):
                    if candidates[i][j] == labels[i]:
                        gold[i] = j
            gold = gold.type(torch.LongTensor).to(self.device)


            loss = self.loss_function(normalized_similarities, gold)

            all_predictions = list()
            all_labels = list()
            
            if dataset_type != "train":
                for i in range(len(similarities)):
                    
                    top_k_ner = torch.topk(predictions[i], 3)[1]
                    target_ner_tag_id = top_k_ner[0].item()
                    confidence = predictions[i][target_ner_tag_id].item()
                    print(confidence)
                    target_ner_tags = []
                    for candidate in top_k_ner:
                        target_ner_tags.append(self.get_key(labels_vocab, candidate))
                                            
                    current_candidates = list(filter(lambda x: x!=0, candidates[i]))
                    normalized_similarities_line = normalized_similarities[i][:len(current_candidates)]
                    current_candidates_ner = [id2ner[str(c.item())] if str(c.item()) in id2ner else "" for c in current_candidates]
                    for k in range(len(current_candidates)):
                        if current_candidates_ner[k] not in target_ner_tags[0] and confidence>0.5: #confidence
                            normalized_similarities_line[k] = 0.0
                        elif current_candidates_ner[k] not in target_ner_tags: #top-k
                            normalized_similarities_line[k] = 0.0

                            
                    all_predictions.append(int(candidates[i][torch.argmax(normalized_similarities_line)]))       
                    all_labels.append(labels[i])

            if dataset_type=="train":
                if not math.isnan(loss):
                    return {
                        "loss": loss,
                        "pred": all_predictions,
                        "gold": all_labels,
                    }
                else:
                    return None
            else:
                return {
                        "pred": all_predictions,
                        "gold": all_labels,
                    }




    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        step_output = self.step(batch, batch_idx, "train")
        if step_output is not None:
            self.log_dict(
                {"train_loss": step_output["loss"]},
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
        return step_output

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        step_output = self.step(batch, batch_idx, "dev")
        return step_output

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        step_output = self.step(batch, batch_idx, "test")
        return step_output


    def my_epoch_end(self, outputs: List[Any], split:str) -> None:
        all_predictions = []
        all_labels = []

        for elem in outputs:
            all_predictions.extend(elem["pred"])
            all_labels.extend(elem["gold"])
        
        f1_micro = f1_score(all_labels, all_predictions, average='micro')
        self.log_dict(
                {f"{split}_acc": f1_micro},
                prog_bar=True
            )

        return super().validation_epoch_end(outputs)

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        return self.my_epoch_end(outputs, "val")

    def test_epoch_end(self, outputs: List[Any]) -> None:
        return self.my_epoch_end(outputs, "test")
        

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

    def get_key(self, dictionary, val):
        for key, value in dictionary.items():
            if val == value:
                return key
    
        return "key doesn't exist"


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
