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
import random


class RE_Dataset(Dataset):
    """ Dataset 구성을 위한 class."""

    def __init__(self, pair_dataset, labels,
                 val_ratio=0.2, seed=2):
        random.seed(seed)
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
    if '0' in preprocessing_cmb :
        sentence = remove_special_char(sentence)
    if '1' in preprocessing_cmb :
        sentence = substitution_special_char(sentence)
    if '2' in preprocessing_cmb :
        sentence = substitution_date(sentence)
    if '3' in preprocessing_cmb :
        sentence = add_space_char(sentence)
    return sentence


def preprocessing_dataset(dataset, entity_flag=0, preprocessing_cmb=None, mecab_flag=0):
    """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
    subject_entity = []
    object_entity = []

    # sentence에 entity 속성 추가
    for i, j in zip(dataset['subject_entity'], dataset['object_entity']):
        # i = eval(i)['word']
        # j = eval(j)['word']
        i = i['word']
        j = j['word']

        subject_entity.append(i)
        object_entity.append(j)

    if entity_flag:
        new_sentence = sentence_processing(dataset)
        dataset.sentence = new_sentence

    if preprocessing_cmb != None and mecab_flag:
        out_dataset = pd.DataFrame({'id': dataset['id'],
                                    'sentence': [mecab_processing(text_preprocessing(sent, preprocessing_cmb)) for sent in dataset['sentence']],
                                    'subject_entity': [mecab_processing(text_preprocessing(entity, preprocessing_cmb)) for entity in subject_entity],
                                    'object_entity': [mecab_processing(text_preprocessing(entity, preprocessing_cmb)) for entity in object_entity],
                                    'label': dataset['label'], })
        print('Finish preprocessing and mecab !!!')

    elif preprocessing_cmb != None and not mecab_flag:
        out_dataset = pd.DataFrame({'id': dataset['id'],
                                    'sentence': [text_preprocessing(sent, preprocessing_cmb) for sent in dataset['sentence']],
                                    'subject_entity': [text_preprocessing(entity, preprocessing_cmb) for entity in subject_entity],
                                    'object_entity': [text_preprocessing(entity, preprocessing_cmb) for entity in object_entity],
                                    'label': dataset['label'], })
        print('Finish data preprocessing!!!')

    elif mecab_flag and preprocessing_cmb == None :
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


def load_data(dataset_dir, entity_flag=0, preprocessing_flag=None, mecab_flag=0, augmentation_flag=False):
    """ csv 파일을 경로에 맡게 불러 옵니다. """
    pd_dataset = pd.read_csv(dataset_dir)
    
    if 'train' in dataset_dir:
        # 완전 중복 제거 42개
        pd_dataset = pd_dataset.drop_duplicates(['sentence', 'subject_entity', 'object_entity', 'label'], keep='first')
        # 라벨링이 다른 데이터 제거
        pd_dataset = pd_dataset.drop(index=[6749, 8364, 22258, 277, 25094])
        pd_dataset = pd_dataset.reset_index(drop=True)
        print("Finish remove duplicated data")

    #datatype 변경
    pd_dataset['subject_entity'] = pd_dataset.subject_entity.map(eval)
    pd_dataset['object_entity'] = pd_dataset.object_entity.map(eval)
    
    #data augmentation(동일한 label 및 subject_object 내에서 subject와 object entity를 sampling하여 데이터 증량)
    #단 특정 subject_object_type은 data가 1개이므로, 이 경우 임의로 값 지정
    if augmentation_flag is True:
        pd_dataset = augmentation_by_resampling(pd_dataset)
    
    # dataset = preprocessing_dataset(
        # pd_dataset, entity_flag, preprocessing_flag, mecab_flag)
    # return dataset
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

def augmentation_by_resampling(data):
    #subject_object 타입 및 word를 새로운 column에 추가(마지막에 삭제 예정)
    def sbj_obj_type(row):
        return row["subject_entity"]["type"] + "_" + row["object_entity"]["type"]
    def sbj_obj_word(row):
        return row["subject_entity"]["word"] + "_" + row["object_entity"]["word"]

    data["sbj_obj_entity_type"] = data.apply(sbj_obj_type,axis=1)
    data["sbj_obj_entity_word"] = data.apply(sbj_obj_word,axis=1)
       

    #라벨 종류 확인
    label_list = get_labels(data)

    #label_sbj_obj - dict 생성
    sbj_obj_dict = get_dict_frame(data,label_list)

    #dictionary에 값 입력 - label - sbj_obj_entity_type - sbj / obj
    for i in range(len(data)):
        tmp_row = data.iloc[i]
        label, sbj_obj_entity_type = tmp_row["label"],tmp_row["sbj_obj_entity_type"]
        sbj,obj = tmp_row["sbj_obj_entity_word"].split("_")
        sbj_obj_dict[label][sbj_obj_entity_type]["sbjs"].append(sbj)
        sbj_obj_dict[label][sbj_obj_entity_type]["objs"].append(obj)
    

    #sampling
    for label in label_list:
        for sbj_obj_entity_type in sbj_obj_dict[label].keys():
            if len(sbj_obj_dict[label][sbj_obj_entity_type]['sbjs'])>3: #data개수가 1개이면 shuffle이 불가하므로 pass
                new_sbj_obj = shuffling_data(data,sbj_obj_dict,label,sbj_obj_entity_type)
                data = augmentation(data,new_sbj_obj,label,sbj_obj_entity_type)
    # data = data.drop(['sbj_obj_entity_type','sbj_obj_entity_word'], axis=1)
    return data

             
def get_labels(data):
    return list(data.label.value_counts().index)


def get_dict_frame(data,label_list):
    sbj_obj_dict = collections.defaultdict(dict)
    for label in label_list:
        sbj_obj_type_list = data.loc[data.label==label,"sbj_obj_entity_type"].unique()
        for sbj_obj_type in sbj_obj_type_list:
            sbj_obj_dict[label][sbj_obj_type] = {}
            sbj_obj_dict[label][sbj_obj_type]["sbjs"] = []
            sbj_obj_dict[label][sbj_obj_type]["objs"] = []
    return sbj_obj_dict


def shuffling_data(data,sbj_obj_dict,label,sbj_obj_type):
    sbj_list = sbj_obj_dict[label][sbj_obj_type]["sbjs"]
    obj_list = sbj_obj_dict[label][sbj_obj_type]["objs"]
    resample=True
    while resample==True:
        #subject와 object entity 순서 shuffle
        random.shuffle(sbj_list)
        random.shuffle(obj_list)
        sbj_list_choice = [random.choice(sbj_list) for _ in range(len(sbj_list))]
        obj_list_choice = [random.choice(obj_list) for _ in range(len(obj_list))]

        #shuffle된 entity를 조합
        new_sb_ob_list = []
        for sb,ob in zip(sbj_list_choice,obj_list_choice):
            new_sb_ob_list.append(f"{sb}_{ob}")
        
        #기존 entity 조합과 비교
        data_cond = (data["label"]==label) & (data["sbj_obj_entity_type"]==sbj_obj_type)
        
        if not any(data.loc[data_cond,"sbj_obj_entity_word"]==new_sb_ob_list): #전체가 false(모든 element 불일치)일 경우 return
            return new_sb_ob_list

def augmentation(data,new_sb_ob_list, label,sbj_obj_type):
    data_cond = (data.label==label) & (data.sbj_obj_entity_type==sbj_obj_type)
    target_data = data[data_cond].copy()
    target_data['replaced'] = new_sb_ob_list
    new_data = pd.DataFrame(list(target_data.apply(change_entity,axis=1).values))
    data = pd.concat([data, new_data])
    return data


def change_entity(row):
    row = row.copy()
    sent = row['sentence']
    sbjs, objs = row['subject_entity'], row['object_entity']
    new_sb, new_ob = row['replaced'].split("_")

    sb_len, new_sb_len, ob_len, new_ob_lens = len(sbjs['word']), len(new_sb), len(objs['word']), len(new_ob)
    if sbjs['start_idx'] < objs['start_idx'] : 
        str1,str2,str3 = sent[:sbjs['start_idx']], sent[sbjs['end_idx']+1:objs['start_idx']], sent[objs['end_idx']+1:]
        new_sent = str1 + new_sb + str2 + new_ob + str3
        sbjs['word'] = new_sb
        sbjs["end_idx"] = sbjs['start_idx'] + len(new_sb)-1
        objs['word'] = new_ob
        objs['start_idx'] = new_sent.find(new_ob)
        objs['end_idx'] = objs['start_idx'] + len(new_ob)-1
    else:
        str1,str2,str3 = sent[:objs['start_idx']], sent[objs['end_idx']+1:sbjs['start_idx']], sent[sbjs['end_idx']+1:]
        new_sent = str1 + new_ob + str2 + new_sb + str3
        objs['word'] = new_ob
        objs["end_idx"] = objs['start_idx'] + len(new_sb)-1
        sbjs['word'] = new_sb
        sbjs['start_idx'] = new_sent.find(new_sb)
        sbjs['end_idx'] = sbjs['start_idx'] + len(new_sb)-1
    
    return {"id" : row['id'],
            "sentence" : new_sent,
            "subject_entity" : sbjs,
            "object_entity" : objs,
            "label" : row["label"],
            "source" : row["source"],
            "sbj_obj_entity_type" : row["sbj_obj_entity_type"],
            "sbj_obj_entity_word" : row["sbj_obj_entity_type"]}
