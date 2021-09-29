import pickle as pickle
import os
import re
import pandas as pd
import collections
import random
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, Subset
from preprocessor import *

from time import sleep

class RE_Dataset(Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, pair_dataset, labels, val_size=20):
    self.pair_dataset = pair_dataset
    self.labels = labels
    self.val_size = val_size

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
        val_index = random.sample(idx_list, self.val_size)
        train_index = list(set(idx_list) - set(val_index))
            
        train_data.extend(train_index)
        val_data.extend(val_index)
        
    random.shuffle(train_data)
    random.shuffle(val_data)
        
    train_dset = Subset(self, train_data)
    val_dset = Subset(self, val_data)
        
    return train_dset, val_dset

def preprocessing_dataset(dataset):
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  subject_entity = []
  object_entity = []
  for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
    i = i[1:-1].split(',')[0].split(':')[1]
    j = j[1:-1].split(',')[0].split(':')[1]
    i = re.sub('[\']', '' , i)
    j = re.sub('[\']', '' , j)
        
    subject_entity.append(i)
    object_entity.append(j)

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
  concat_entity = []
  for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
    temp = ''
    temp = e01 + '[SEP]' + e02
    concat_entity.append(temp)

  concat_entity = [preprocess_sen(entity).strip() for entity in concat_entity]
  sen_data = [preprocess_sen(sen).strip() for sen in dataset['sentence']]

  """ UNK token counting """
  print('-'*100)
  print(f'{tokenizer.unk_token} token processing...')
  unk_idx = tokenizer.get_vocab()[tokenizer.unk_token]
  UNK_cnt = 0
  for sen in tqdm(dataset['sentence']) :
    UNK_cnt += tokenizer.encode(sen).count(unk_idx)
  sleep(0.01)
  print(f'UNK token count is :{UNK_cnt} in {tokenizer.name_or_path}')
  print('-'*100)
  sleep(3)
  
  """ Roberta TTI_flag """
  if 'roberta' in tokenizer.name_or_path and not 'xlm' in tokenizer.name_or_path:
    tokenized_sentences = tokenizer(
      concat_entity,
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
      concat_entity,
      sen_data,
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True
    )

  return tokenized_sentences