# -*- coding: utf-8 -*-
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


def text_preprocessing(sentence):
    sent = remove_special_char(sentence)
    sent = substitution_date(sent)
    sent = add_space_char(sent)
    return sent


def preprocessing_dataset(dataset, entity_flag=0, preprocessing_flag=0, mecab_flag=0):
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

    if preprocessing_flag and mecab_flag:
        out_dataset = pd.DataFrame({'id': dataset['id'],
                                    'sentence': [mecab_processing(text_preprocessing(sent)) for sent in dataset['sentence']],
                                    'subject_entity': [mecab_processing(text_preprocessing(entity)) for entity in subject_entity],
                                    'object_entity': [mecab_processing(text_preprocessing(entity)) for entity in object_entity],
                                    'label': dataset['label'], })
        print('Finish preprocessing and mecab !!!')

    elif preprocessing_flag and not mecab_flag:
        out_dataset = pd.DataFrame({'id': dataset['id'],
                                    'sentence': [text_preprocessing(sent) for sent in dataset['sentence']],
                                    'subject_entity': [text_preprocessing(entity) for entity in subject_entity],
                                    'object_entity': [text_preprocessing(entity) for entity in object_entity],
                                    'label': dataset['label'], })
        print('Finish data preprocessing!!!')

    elif mecab_flag and not preprocessing_flag:
        out_dataset = pd.DataFrame({'id': dataset['id'],
                                    'sentence': [mecab_processing(sent) for sent in dataset['sentence']],
                                    'subject_entity': [mecab_processing(entity) for entity in subject_entity],
                                    'object_entity': [mecab_processing(entity) for entity in object_entity],
                                    'label': dataset['label'], })
        print('Finish mecab preprocessing!!!')
    else:
        out_dataset = pd.DataFrame({'id': dataset['id'],
                                    'sentence': (dataset['sentence']),
                                    'subject_entity': (subject_entity),
                                    'object_entity': (object_entity),
                                    'label': dataset['label'], })
        print('None preprocessing')

    return out_dataset


def load_data(dataset_dir, entity_flag=0, preprocessing_flag=0, mecab_flag=0, model_type="big_sort"):
    """ csv 파일을 경로에 맡게 불러 옵니다. """
    pd_dataset = pd.read_csv(dataset_dir)
    if 'train' in dataset_dir:
        # 완전 중복 제거 42개
        pd_dataset = pd_dataset.drop_duplicates(
            ['sentence', 'subject_entity', 'object_entity', 'label'], keep='first')
        # 라벨링이 다른 데이터 제거
        pd_dataset = pd_dataset.drop(index=[6749, 8364, 22258, 277, 25094])
        pd_dataset = pd_dataset.reset_index(drop=True)
        print("Finish remove duplicated data")
    
    if model_type=="per_sort":
        pd_dataset=pd_dataset[pd_dataset['label'].str.contains('per')].reset_index(drop=True)
    elif model_type=="org_sort":
        pd_dataset=pd_dataset[pd_dataset['label'].str.contains('org')].reset_index(drop=True)
        

    dataset = preprocessing_dataset(
        pd_dataset, entity_flag, preprocessing_flag, mecab_flag)
    return dataset


def tokenized_dataset(dataset, tokenizer, is_inference=False):
    """ tokenizer에 따라 sentence를 tokenizing 합니다."""
#     sentences = []
#     questions = []
#     for sen, e01, e02 in zip(dataset['sentence'], dataset['subject_entity'], dataset['object_entity']):
#         sentences.append(sen)
          
#         아래의 question 둘 중 하나 선택    
#         questions.append('◈ ' + e01 + '◈ 과 ↑ ' + e02 + ' ↑ 는 무슨 관계일까요?')
#         questions.append('</e1> ' + e01 + '</e1> 과 </e2> ' + e02 + ' </e2> 는 무슨 관계일까요?')    
#     tokenized_sentences = tokenizer(
#         sentences,
#         questions,
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#         max_length=256,
#         add_special_tokens=True,
#         return_token_type_ids=False
#     )
    concat_entity = []
    for sen ,e01, e02 in zip(dataset['sentence'], dataset['subject_entity'], dataset['object_entity']):
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
