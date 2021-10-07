import pickle

import os
import re
import torch
import random
import argparse
import itertools
import pandas as pd
from collections import Counter
from typing import Dict, List, Tuple

from transformers import AutoTokenizer
from konlpy.tag import Mecab
from pororo import Pororo

# load Pororo model
model = Pororo(task="mt", lang="multi", model="transformer.large.multi.mtpg")

def cycling_translation_en(sentence):
    english = model(sentence, 'ko', 'en')
    korean = model(english, 'en', 'ko')
    return korean

def make_rtt_csv(args, dataset):
    data_num = len(dataset)
    count = 0 # 총 몇 개의 데이터가 추가되는지 확인용

    lbl_values = list(dataset['label'].unique()) # 모든 class 불러오기
    lbl_remove_list = ['no_relation', 'org:top_members/employees', 'per:employee_of'] # 개수가 3000개 이상인 라벨들 (augmentation 필요X)
    lbl_candidate_list = [val for val in lbl_values if val not in lbl_remove_list]
    lbl_dict = dict.fromkeys(lbl_candidate_list, 0) # lbl_candidate_list를 키로 갖는 dictionary를 0으로 초기화
    
    ind_list = []
    new_sen_list = []
    new_subj_list = []
    new_obj_list = []
    label_list = []
    source_list = []
    for ind in range(data_num):
        _, sen, subj_dict, obj_dict, label, source = dataset.loc[ind]
        
        if label not in lbl_candidate_list: # candidate list 안에 없는 건 패스!
            continue
        rtt_sen = cycling_translation_en(sen)
        rtt_subj = cycling_translation_en(eval(subj_dict)['word'])
        rtt_obj = cycling_translation_en(eval(obj_dict)['word'])
        
        if rtt_subj not in rtt_sen or rtt_obj not in rtt_sen: # 번역된 subj, obj가 번역된 문장에 없으면 추가 X
            continue
        if len(sen.split('.')) < len(rtt_sen.split('.')): # 문장이 여러 개로 만들어졌으면 추가 X (잘못 번역되었을 확률 높음)
            continue
        
        # entity의 위치 index 찾기 (TO DO: 똑같은 entity가 한 문장에 여러 개 있는 경우는?)
        subj_start = rtt_sen.find(rtt_subj)
        subj_end = subj_start + len(rtt_subj) - 1
        rtt_subj = str(dict({'word':rtt_subj,
                            'start_idx':subj_start,
                            'end_idx':subj_end,
                            'type':eval(subj_dict)['type'],
                            }))
        obj_start = rtt_sen.find(rtt_obj)
        obj_end = obj_start + len(rtt_obj) - 1
        rtt_obj = str(dict({'word':rtt_obj,
                            'start_idx':obj_start,
                            'end_idx':obj_end,
                            'type':eval(subj_dict)['type'],
                           }))

        ind_list.append(count)
        new_sen_list.append(rtt_sen)
        new_subj_list.append(rtt_subj)
        new_obj_list.append(rtt_obj)
        label_list.append(label)
        source_list.append(source)
        lbl_dict[label] += 1
        count += 1
        
        # 번역된 문장을 확인하고 싶으시면 주석을 풀어주세요
        # print(f'[{ind} / {count} 개]\nORIGINAL: {sen}\nRTT: {rtt_sen}\nsubject: {rtt_subj}, object: {rtt_obj}, label: {label}')

        print(f'[{ind} / {count} 개]')
    print(lbl_dict, '\n')
    new_data = {'sentence': new_sen_list,
               'subject_entity': new_subj_list,
               'object_entity': new_obj_list,
               'label': label_list,
               'source': source_list}
    new_data_df = pd.DataFrame(new_data)
    new_data_df.to_csv(os.path.join('/opt/ml/dataset/train', args.save_name+'.csv'))
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/opt/ml/dataset/train/train.csv',
                        help='source data to apply rtt')
    parser.add_argument('--save_name', type=str, default='rtt_data',
                        help='output csv file name')
    
    args = parser.parse_args()
    
    # load csv file
    source_data = pd.read_csv(args.data_path)
    
    make_rtt_csv(args, source_data)
    print('Finish !')