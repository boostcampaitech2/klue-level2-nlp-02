import pickle as pickle
import os
import re
import pandas as pd
import collections
import random
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, Subset, random_split
from preprocessor import *

from time import sleep


class RE_Dataset(Dataset):
    """ Dataset 구성을 위한 class."""

    def __init__(self, pair_dataset, labels, val_ratio=0.2):
        self.pair_dataset = pair_dataset
        self.labels = labels
        self.val_ratio = val_ratio

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.pair_dataset.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

    def split(self):
        data_size = len(self)
        index_map = collections.defaultdict(list)
        for idx in range(data_size):
            label = self.labels[idx]
            index_map[label].append(idx)

        train_data = []
        val_data = []

        label_size = len(index_map)
        for label in range(label_size):
            idx_list = index_map[label]
            sample_size = int(len(idx_list) * self.val_ratio)

            val_index = random.sample(idx_list, sample_size)
            train_index = list(set(idx_list) - set(val_index))

            train_data.extend(train_index)
            val_data.extend(val_index)

        random.shuffle(train_data)
        random.shuffle(val_data)

        train_dset = Subset(self, train_data)
        val_dset = Subset(self, val_data)
        return train_dset, val_dset


class LM_Dataset(Dataset):
    """ Language Model Dataset 구성을 위한 class."""
    
    def __init__(self, pair_dataset, tokenizer):
        self.pair_dataset = pair_dataset
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.pair_dataset.items()}
        inputs, labels = mask_tokens(item['input_ids'], self.tokenizer)
        item['input_ids'] = inputs.squeeze()
        # item['input_mask'] = inputs != 0
        item['labels'] = labels.squeeze()
        return item

    def __len__(self):
        return len(self.pair_dataset['input_ids'])


def preprocessing_dataset(dataset):
    """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
    subject_entity = []
    object_entity = []

    # sentence에 entity 속성 추가
    sentence_flag = True
    if sentence_flag == True:
        new_sentence = sentence_processing(dataset)
        dataset.sentence = new_sentence

    for i, j in zip(dataset['subject_entity'], dataset['object_entity']):
        i = eval(i)['word']
        j = eval(j)['word']

        subject_entity.append(i)
        object_entity.append(j)

    out_dataset = pd.DataFrame({'id': dataset['id'],
                                'sentence': dataset['sentence'],
                                'subject_entity': subject_entity,
                                'object_entity': object_entity,
                                'label': dataset['label'], })
    return out_dataset


def load_data(dataset_dir):
    """ csv 파일을 경로에 맞게 불러 옵니다. """
    pd_dataset = pd.read_csv(dataset_dir)
    if 'train' in dataset_dir:
        # 완전 중복 제거 42개
        pd_dataset = pd_dataset.drop_duplicates(
            ['sentence', 'subject_entity', 'object_entity', 'label'], keep='first')
        # 라벨링이 다른 데이터 제거
        pd_dataset = pd_dataset.drop(index=[6749, 8364, 22258, 277, 25094])

    dataset = preprocessing_dataset(pd_dataset)
    return dataset


def load_lm_data(dataset_dir):
    """ csv 파일을 경로에 맞게 불러 옵니다. """
    pd_dataset = pd.read_csv(dataset_dir)
    if 'train' in dataset_dir:
        # 완전 중복 제거 42개
        pd_dataset = pd_dataset.drop_duplicates(
            ['sentence'], keep='first')

    # 전처리를 할 경우 preprocessor.py에 전처리 함수를 만들어서 넣어주자
    pd_dataset = pd.DataFrame({'id': pd_dataset['id'],
                                'sentence': pd_dataset['sentence'],
                                'label': pd_dataset['label'],
                                # 'input_mask': pd_dataset['label'],
                                })
    return pd_dataset


def tokenized_dataset(dataset, tokenizer, is_inference=False):
    """ tokenizer에 따라 sentence를 tokenizing 합니다."""
    concat_entity = []
    for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
        temp = ''
        temp = e01 + '[SEP]' + e02
        concat_entity.append(temp)

    if is_inference:
        """ Roberta TTI_flag """
        if 'roberta' in tokenizer.name_or_path and not 'xlm' in tokenizer.name_or_path:
            tokenized_sentences = tokenizer(
                concat_entity,
                list(dataset['sentence']),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
                add_special_tokens=True,
                return_token_type_ids=False
            )
        else:
            tokenized_sentences = tokenizer(
                concat_entity,
                list(dataset['sentence']),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
                add_special_tokens=True
            )
    else:
        """ Roberta TTI_flag """
        if 'roberta' in tokenizer.name_or_path and not 'xlm' in tokenizer.name_or_path:
            tokenized_sentences = tokenizer(
                concat_entity,
                list(dataset['sentence']),
                truncation=True,
                max_length=256,
                add_special_tokens=True,
                return_token_type_ids=False
            )
        else:
            tokenized_sentences = tokenizer(
                concat_entity,
                list(dataset['sentence']),
                truncation=True,
                max_length=256,
                add_special_tokens=True
            )

    return tokenized_sentences

def mask_tokens(inputs, tokenizer, mlm_probability=0.15, pad=True):
    inputs = torch.tensor(inputs)
    labels = inputs.clone()
    
    # mlm_probability은 15%로 BERT에서 사용하는 확률
    probability_matrix = torch.full(labels.shape, mlm_probability)
    # special_tokens_mask = [
    #     tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    # ] # 원래 코드
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(labels.tolist(), already_has_special_tokens=True)
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool).squeeze(), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]
    return inputs, labels


def tokenized_lm_dataset(dataset, tokenizer, is_inference=False):
    """ tokenizer에 따라 sentence를 tokenizing 합니다."""
    if is_inference:
        """ Roberta TTI_flag """
        if 'roberta' in tokenizer.name_or_path and not 'xlm' in tokenizer.name_or_path:
            tokenized_sentences = tokenizer(
                list(dataset['sentence']),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
                add_special_tokens=True,
                return_token_type_ids=False
            )
        else:
            tokenized_sentences = tokenizer(
                list(dataset['sentence']),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
                add_special_tokens=True
            )
    else:
        """ Roberta TTI_flag """
        if 'roberta' in tokenizer.name_or_path and not 'xlm' in tokenizer.name_or_path:
            tokenized_sentences = tokenizer(
                list(dataset['sentence']),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
                add_special_tokens=True,
                return_token_type_ids=False
            )
        else:
            tokenized_sentences = tokenizer(
                list(dataset['sentence']),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
                add_special_tokens=True
            )
    return tokenized_sentences