from typing import Dict, Tuple, Union

import hydra
import omegaconf
import pytorch_lightning as pl
from pytorch_lightning.core import datamodule
import torch
from torch import nn
from omegaconf import ValueNode
from torch.utils.data import Dataset
from tqdm import tqdm
import pickle
import random

from src.common.utils import PROJECT_ROOT
from pl_modules.ner_model import MyNERModel



class MyDataset(Dataset):
    def __init__(self, name: ValueNode, 
                path: ValueNode, 
                data: list,
                num_candidates: int, 
                window: int,
                datamodule,
                dataset_type,
                **kwargs):

        from src.pl_data.datamodule import MyDataModule
        datamodule:MyDataModule

        super().__init__()
        self.path = path
        self.name = name
        self.data = data
        self.num_candidates = num_candidates
        self.window = window
        self.transformer_name = datamodule.transformer_name
        self.tokenizer = datamodule.tokenizer
        self.alias_table = datamodule.alias_table
        self.descriptions_dict = datamodule.descriptions_dict
        self.count_dict = datamodule.item_counts_dict
        self.id2ner = datamodule.id2ner
        self.dataset_type = dataset_type
        self.title_dict = datamodule.title_dict
        self.title_dict_reverse = datamodule.title_dict_reverse
        self.negative_samples = datamodule.negative_samples
        self.ner_negative_samples = datamodule.ner_negative_samples
        self.ner_representation = datamodule.ner_representation
        self.ner_filter_candidates = datamodule.ner_filter_candidates
        self.processed = datamodule.processed

        self.encoded_data = []
        self.__encode_data()
        print(f"LEN: {len(self.encoded_data)}")

    def __encode_data(self):

        if self.ner_filter_candidates:
            ner_classifier = MyNERModel()
            ner_classifier.load_state_dict(torch.load(str(PROJECT_ROOT / "wandb/ner_classifier.pt")))

            softmax_function = nn.Softmax(dim=1)

            labels_vocab = {}
            i = 0
            for v in self.id2ner.values():
                if v not in labels_vocab:
                    labels_vocab[v] = i
                    i+=1


        if self.processed == True and self.dataset_type=="train":
            with open(str(PROJECT_ROOT /  f"preprocessed_datasets/aida_kilt_train_{self.num_candidates}_{self.window}_{self.transformer_name}_negativesamples={self.negative_samples}_nernegativesamples={self.ner_negative_samples}_nerrepresentation={self.ner_representation}.pickle"), 'rb') as f:
                self.encoded_data = pickle.load(f)

        else:
            total_entities = 0
            target_between_candidates = 0

            #for negative samples
            if self.dataset_type == "train":
                count_dict_with_descriptions = {}
                dict_candidates_for_NER_type = {}

                for key in tqdm(self.count_dict):
                    if key in self.descriptions_dict:
                        count_dict_with_descriptions[key] = 1

                        if self.ner_negative_samples == True:
                            if str(key) in self.id2ner:
                                ner_tag = self.id2ner[str(key)]
                                if ner_tag not in dict_candidates_for_NER_type:
                                    dict_candidates_for_NER_type[ner_tag] = [key]
                                else:
                                    dict_candidates_for_NER_type[ner_tag].append(key)
                        
            
            for entry in tqdm(self.data):
                m = entry["mention"].lower()
                left_context = entry["left_context"]
                right_context = entry["right_context"]
                ent = self.title_dict_reverse[entry["output"]] if entry["output"] in self.title_dict_reverse else ""
                title = entry["output"]

                if self.ner_negative_samples == True:
                    if str(ent) in self.id2ner: #to see the upperbound
                            target_ner_tag = self.id2ner[str(ent)]
                    else:
                        target_ner_tag = ""

                tokenized_left_context = self.tokenize_mention("[CLS]" + left_context, self.tokenizer, self.window//2, False)
                mention_position = len(tokenized_left_context) + 1
                tokenized_m = self.tokenize_mention("[E]" + m + "[\E]", self.tokenizer, self.window//2, False)
                tokenized_right_context = self.tokenize_mention(right_context + "[SEP]", self.tokenizer, self.window//2, False)

                tokenized_mention = tokenized_left_context + tokenized_m + tokenized_right_context
                for _ in range(self.window-len(tokenized_mention)):
                    tokenized_mention.append(0)
                tokenized_mention = torch.tensor(tokenized_mention)


                if self.ner_filter_candidates:
                    pad_mentions = tokenized_mention.unsqueeze(0)
                    mask = self.padding_mask(pad_mentions)

                    position = torch.tensor(mention_position)

                    predictions = ner_classifier(pad_mentions, position, mask)
                    predictions = softmax_function(predictions)[0]
                    top_k_ner = torch.topk(predictions, 3)[1]
                    label_ner = top_k_ner[0].item()
                    confidence = predictions[label_ner]


                if(m in self.alias_table) and ent!="":
                    total_entities+=1
                    all_candidates = self.alias_table[m]

                    all_candidates_with_description = []
                    for c in all_candidates:
                        if c in self.descriptions_dict:
                            all_candidates_with_description.append(c)

                    all_candidates_counts = torch.tensor([self.count_dict.get(idx, 0) for idx in all_candidates_with_description])

                    if len(all_candidates_with_description)>self.num_candidates:
                        top_k = torch.topk(all_candidates_counts, self.num_candidates)[1].tolist()
                        top_k_candidates = []
                        for idx in top_k:
                            top_k_candidates.append(all_candidates_with_description[idx])
                    else:
                        top_k_candidates = all_candidates_with_description

                        if self.negative_samples == True and self.dataset_type == "train": #standard negative samples
                            tmp_count_dict_with_descriptions = count_dict_with_descriptions.copy()
                            for candidate in top_k_candidates:
                                del tmp_count_dict_with_descriptions[candidate]
                            negative_samples = random.sample(tmp_count_dict_with_descriptions.keys(), self.num_candidates-len(top_k_candidates))
                            top_k_candidates.extend(negative_samples)

                        elif self.ner_negative_samples == True and self.dataset_type == "train": #NER-enhanced negative samples
                            if target_ner_tag!="":
                                candidates_of_same_NER_type = dict_candidates_for_NER_type[target_ner_tag]
                                for candidate in top_k_candidates:
                                    if candidate in candidates_of_same_NER_type:
                                        candidates_of_same_NER_type.remove(candidate)
                                negative_samples = random.sample(candidates_of_same_NER_type, self.num_candidates-len(top_k_candidates))
                                top_k_candidates.extend(negative_samples)

                        elif self.ner_filter_candidates == True and self.dataset_type == "train":
                            top_k_candidates_filtered = []
                            for c in top_k_candidates:
                                if str(c) in self.id2ner and confidence>0.5:
                                    if labels_vocab[self.id2ner[str(c)]] == label_ner:
                                        top_k_candidates_filtered.append(c)
                                    elif labels_vocab[self.id2ner[str(c)]] in top_k_ner:
                                        top_k_candidates_filtered.append(c)
                                else:
                                    top_k_candidates_filtered.append(c)


                    if int(ent) in top_k_candidates:
                        target_between_candidates+=1

                    tokenized_descriptions = []
                    for c in top_k_candidates:
                        if self.ner_representation == True:
                            if str(c) in self.id2ner:
                                ner_tag = self.id2ner[str(c)]
                            else:
                                ner_tag = ""
                            d = ner_tag + "[SEP]" + self.descriptions_dict[c]
                        else:
                            d = self.descriptions_dict[c]

                        tokenized_descriptions.append(self.tokenize_description(d, self.tokenizer, self.window))


                    if self.dataset_type == "train":
                        if len(top_k_candidates)>0 and int(ent) in top_k_candidates:
                            self.encoded_data.append((tokenized_mention,
                                                mention_position,
                                                torch.tensor(top_k_candidates),
                                                torch.tensor(tokenized_descriptions),
                                                int(ent)))

                    else:
                        if len(top_k_candidates)>0:
                            self.encoded_data.append((tokenized_mention,
                                                mention_position,
                                                torch.tensor(top_k_candidates),
                                                torch.tensor(tokenized_descriptions),
                                                int(ent)))


                    if total_entities>0:
                        print(f"Percentage of target entities within the candidate set: {target_between_candidates/total_entities}")

                    #print(self.encoded_data)


            if self.dataset_type == "train":
                with open(str(PROJECT_ROOT /  f"preprocessed_datasets/aida_kilt_train_{self.num_candidates}_{self.window}_{self.transformer_name}_negativesamples={self.negative_samples}_nernegativesamples={self.ner_negative_samples}_nerrepresentation={self.ner_representation}.pickle"), 'wb') as f:
                    pickle.dump(self.encoded_data, f)
        
         
        return self.encoded_data


    def tokenize_mention(self, sent, tokenizer, window, special_tokens):
        encoded_sentence = tokenizer.encode(sent, add_special_tokens = special_tokens)
        if len(encoded_sentence)>=window:
            return encoded_sentence[:window]
        else:
            return encoded_sentence
    
    def tokenize_description(self, sent, tokenizer, window):
        encoded_sentence = tokenizer.encode(sent, add_special_tokens = True)
        if len(encoded_sentence)>=window:
            return encoded_sentence[:window]
        else:
            return encoded_sentence + [0]*(window-len(encoded_sentence))

    def padding_mask(self, batch):
        padding = torch.ones_like(batch)
        padding[batch == 0] = 0
        padding = padding.type(torch.int64)
        return padding


    def __len__(self) -> int:
        return len(self.encoded_data)

    def __getitem__(
        self, index
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        return self.encoded_data[index]

    def __repr__(self) -> str:
        return f"MyDataset({self.name=}, {self.path=})"


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    dataset: MyDataset = hydra.utils.instantiate(
        cfg.data.datamodule.datasets.train, _recursive_=False
    )


if __name__ == "__main__":
    main()
