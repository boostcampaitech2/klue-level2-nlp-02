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
  re_data = []
  data_size = len(dataset)
  for i in range(data_size) :
    sub = dataset['subject_entity'][i]
    obj = dataset['object_entity'][i]
    sen = dataset['sentence'][i]

    re_sen = sub + ' [SEP] ' + obj + ' [SEP] ' + sen
    re_data.append(re_sen)

  print('Preprocess Text Data')
  re_sen = [preprocess_sen(sen) for sen in tqdm(re_data)]

  tokenized_sentences = tokenizer(
    re_sen,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=256,
  )

  return tokenized_sentences
