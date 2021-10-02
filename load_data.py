import pickle as pickle
import os
import re
import pandas as pd
import collections
import random
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, Subset
from Preprocessing.preprocessor import *


class RE_Dataset(Dataset):
    """ Dataset 구성을 위한 class."""

    def __init__(self, pair_dataset, labels,
                 val_ratio=0.2, seed=2):
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


def text_preprocessing(sentence, preprocessing_cmb):
    if 0 in preprocessing_cmb :
        sentence = remove_special_char(sentence)
    if 1 in preprocessing_cmb :
        sentence = substitution_date(sentence)
    if 2 in preprocessing_cmb :
        sentence = add_space_char(sentence)
    return sentence


def preprocessing_dataset(dataset, entity_flag=0, preprocessing_cmb=None, mecab_flag=0):
    """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
    subject_entity = []
    object_entity = []

    # sentence에 entity 속성 추가
    for i, j in zip(dataset['subject_entity'], dataset['object_entity']):
        i = eval(i)['word']
        j = eval(j)['word']

        subject_entity.append(i)
        object_entity.append(j)

    if entity_flag:
        new_sentence = sentence_processing(dataset)
        dataset.sentence = new_sentence

    if preprocessing_cmb != None :
        out_dataset = pd.DataFrame({'id': dataset['id'],
                                    'sentence': [text_preprocessing(sent, preprocessing_cmb) for sent in dataset['sentence']],
                                    'subject_entity': [text_preprocessing(entity, preprocessing_cmb) for entity in subject_entity],
                                    'object_entity': [text_preprocessing(entity, preprocessing_cmb) for entity in object_entity],
                                    'label': dataset['label'], })
        print('Finish text preprocessing!!!')
    else:
        out_dataset = pd.DataFrame({'id': dataset['id'],
                                    'sentence': (dataset['sentence']),
                                    'subject_entity': (subject_entity),
                                    'object_entity': (object_entity),
                                    'label': dataset['label'], })
        print('None text preprocessing')
    return out_dataset


def load_data(dataset_dir, entity_flag=0, preprocessing_flag=0, mecab_flag=0):
    """ csv 파일을 경로에 맡게 불러 옵니다. """
    pd_dataset = pd.read_csv(dataset_dir)
    if 'train' in dataset_dir:
        # 완전 중복 제거 42개
        pd_dataset = pd_dataset.drop_duplicates(
            ['sentence', 'subject_entity', 'object_entity', 'label'], keep='first')
        # 라벨링이 다른 데이터 제거
        pd_dataset = pd_dataset.drop(index=[6749, 8364, 22258, 277, 25094])

    dataset = preprocessing_dataset(
        pd_dataset, entity_flag, preprocessing_flag, mecab_flag)
    return dataset


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
