# -*- coding: utf-8 -*-
import random
import re
import pandas as pd
import collections
from train import label_to_num

def aeda(original_data):
    data = original_data.copy()
    new_sentence = []
    punctuations = ('.', ',', '!', '?', ';', ':')
    pattern_for_split = re.compile(r'#\s↑\s\w+\s↑\s.+\s#|@\s◈\s\w+\s◈\s.+\s@|\S+')
    pattern_for_check_entitized = re.compile(r'#\s↑\s\w+\s↑\s.+\s#|@\s◈\s\w+\s◈\s.+\s@')
    entity_check = re.findall(pattern_for_check_entitized, data.values[0][1])

    is_entitized = False
    if entity_check:
      is_entitized = True

    for row in data.values:
        sentence = row[1]

        word_count = len(sentence.split())

        number_of_punctuations_to_add = random.randint(0, word_count//3)

        if is_entitized:
            splited_sentence = re.findall(pattern_for_split, sentence)
            for i in range(number_of_punctuations_to_add):
                index_to_be_added = random.randint(0, len(punctuations)-1)
                splited_sentence.insert(random.randint(0, len(splited_sentence)), punctuations[index_to_be_added])
        else:
            splited_sentence = sentence.split()
            for i in range(number_of_punctuations_to_add):
                punctuation_index = random.randint(0, len(punctuations)-1)
                index_to_be_inserted = random.randint(0, len(splited_sentence))
                splited_sentence.insert(index_to_be_inserted, punctuations[punctuation_index])
              
        new_sentence.append(' '.join(splited_sentence))
    
    data.sentence = new_sentence
    
    return data

def aeda_dataset(dataset):
    return aeda(dataset)
    # dataset = pd.concat([dataset, aug_data])
    # dataset = dataset.sample(frac=1, random_state=2).reset_index(drop=True)
    # return dataset

def augmentation_by_resampling(data):
    data = data.copy()
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
    new_df = pd.DataFrame()
    for label in label_list:
        for sbj_obj_entity_type in sbj_obj_dict[label].keys():
            if len(sbj_obj_dict[label][sbj_obj_entity_type]['sbjs'])>3: #data개수가 1개이면 shuffle이 불가하므로 pass
                new_sbj_obj = shuffling_data(data,sbj_obj_dict,label,sbj_obj_entity_type)
                new_data = augmentation(data,new_sbj_obj,label,sbj_obj_entity_type)
                new_df = pd.concat([new_df,new_data])
    new_df = new_df.drop(['sbj_obj_entity_type','sbj_obj_entity_word'], axis=1)
    return new_df


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
    return new_data
    # data = pd.concat([data, new_data])
    # return data


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
