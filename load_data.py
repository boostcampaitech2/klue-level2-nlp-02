import pickle as pickle
import os
import re
import pandas as pd
import collections
import random
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, Subset
from time import sleep

TAG_DICT = {'PER' : '인물', 'ORG' : '기관', 'DAT' : '날짜', 'LOC' : '위치', 'NOH' : '수량' , 'POH' : '기타'}
SUB_TOKEN1 = '▲'
SUB_TOKEN2 = '▫'
OBJ_TOKEN1 = '◈'
OBJ_TOKEN2 = '☆'

class RE_Dataset(Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, pair_dataset, labels, val_ratio=0.2):
    self.pair_dataset = pair_dataset
    self.labels = labels
    self.val_ratio = val_ratio

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

  def split(self) :
    data_size = len(self)
    index_map = collections.defaultdict(list)
    for idx in range(data_size) :
      label = self.labels[idx]
      index_map[label].append(idx)
            
    train_data = []
    val_data = []
        
    label_size = len(index_map)
    for label in range(label_size) :
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

def add_sep_tok(sen, sub_start, sub_end, sub_type, obj_start, obj_end, obj_type) :
  sub_tok = TAG_DICT[sub_type]
  obj_tok = TAG_DICT[obj_type]

  sub_start_tok = ' ' + SUB_TOKEN1 + ' ' + SUB_TOKEN2 + ' ' + sub_tok + ' ' + SUB_TOKEN2 + ' '
  sub_end_tok = ' ' + SUB_TOKEN1 + ' '
  obj_start_tok = ' ' + OBJ_TOKEN1 + ' ' + OBJ_TOKEN2 + ' ' + obj_tok + ' ' + OBJ_TOKEN2 + ' '
  obj_end_tok = ' ' + OBJ_TOKEN1 + ' '
    
 if sub_start < obj_start :
  sen = sen[:sub_start] +  sub_start_tok + sen[sub_start:sub_end+1] + sub_end_tok + sen[sub_end+1:]
  obj_start += 13
  obj_end += 13
  sen = sen[:obj_start] + obj_start_tok + sen[obj_start:obj_end+1] + obj_end_tok + sen[obj_end+1:]
 else :
  sen = sen[:obj_start] + obj_start_tok + sen[obj_start:obj_end+1] + obj_end_tok + sen[obj_end+1:]
  sub_start += 13
  sub_end += 13
  sen = sen[:sub_start] + sub_start_tok + sen[sub_start:sub_end+1] + sub_end_tok + sen[sub_end+1:]
  
 return sen
  
  
def preprocessing_dataset(dataset):
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  subject_entity = []
  object_entity = []
  sen_data = []
  for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
    subject_entity.append(sub_info['word'])
    object_entity.append(obj_info['word'])
        
    sub_type = sub_info['type']
    sub_start = sub_info['start_idx']
    sub_end = sub_info['end_idx']
    obj_type = obj_info['type']
    obj_start = obj_info['start_idx']
    obj_end = obj_info['end_idx']
        
    sen = add_sep_tok(s, sub_start, sub_end, sub_type, obj_start, obj_end, obj_type)
    sen_data.append(sen)

  out_dataset = pd.DataFrame({'id':dataset['id'], 
    'sentence':dataset['sentence'],
    'subject_entity':subject_entity,
    'object_entity':object_entity,
    'label':dataset['label'],})
  return out_dataset

def load_data(dataset_dir):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  dataset = preprocessing_dataset(pd_dataset)
  return dataset

def tokenized_dataset(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  entity_data = []
  sen_data = []
  for e01, e02, s in zip(dataset['subject_entity'], dataset['object_entity'], dataset['sentence']):
    entity = e01 + ' [SEP] ' + e02
    entity_data.append(entity)
    sen_data.append(s)

  """ Roberta TTI_flag """
  if 'roberta' in tokenizer.name_or_path and not 'xlm' in tokenizer.name_or_path:
    tokenized_sentences = tokenizer(
      entity_data,
      sen_data,
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
      return_token_type_ids=False
    )
  else :
    tokenized_sentences = tokenizer(
      entity_data,
      sen_data,
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True
    )

  return tokenized_sentences
