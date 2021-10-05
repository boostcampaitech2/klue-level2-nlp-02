import pickle as pickle
import os
import re
import pandas as pd
import collections
import random
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, Subset

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

class r_RE_Dataset(RE_Dataset) :
    def __init__(self, pair_dataset, labels, tokenizer, val_ratio=0.2, entity_ids=False):
        self.pair_dataset = pair_dataset
        self.labels = labels
        self.val_ratio = val_ratio
        self.entity_ids = entity_ids

        self.sub_id = tokenizer.get_vocab()['◈']
        self.obj_id = tokenizer.get_vocab()['↑']

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.pair_dataset.items()}
        item['labels'] = torch.tensor(self.labels[idx])

        if self.entity_ids :
            sub_flag, obj_flag = 0, 0
            e1_mask, e2_mask = [], []
            for enc in item['input_ids'] :
                if enc == self.sub_id :
                    sub_flag += 1
                    e1_mask.append(0)
                    e2_mask.append(0)
                    continue
                elif enc == self.obj_id :
                    obj_flag += 1
                    e1_mask.append(0)
                    e2_mask.append(0)
                    continue
                if sub_flag == 1 : e1_mask.append(1)
                else : e1_mask.append(0)
                if obj_flag == 1 : e2_mask.append(1)
                else : e2_mask.append(0)
            item["e1_mask"] = torch.tensor(e1_mask)
            item["e2_mask"] = torch.tensor(e2_mask)
        return item


def preprocessing_dataset(dataset, sen_preprocessor, entity_preprocessor):
    """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
    subject_entity = []
    object_entity = []
    # sentence에 entity 속성 추가
    for i, j in zip(dataset['subject_entity'], dataset['object_entity']):
        i = eval(i)['word']
        j = eval(j)['word']
        subject_entity.append(i)
        object_entity.append(j)

    # Data Sentence Entity processor
    dataset = entity_preprocessor(dataset)

    # Sentence Preprocessor
    sen_data = [sen_preprocessor(sen) for sen in dataset['sentence']]
    subject_data = [sen_preprocessor(sub) for sub in subject_entity]
    object_data = [sen_preprocessor(obj) for obj in object_entity]

    out_dataset = pd.DataFrame({'id': dataset['id'],
        'sentence': sen_data,
        'subject_entity' : subject_data,
        'object_entity' : object_data,
        'label' : dataset['label']}
    )
    return out_dataset


def load_data(dataset_dir, sen_preprocessor, entity_preprocessor):
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

    dataset = preprocessing_dataset(pd_dataset, sen_preprocessor, entity_preprocessor)
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

    return tokenized_sentences