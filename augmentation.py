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
    aug_data = aeda(dataset)
    dataset = pd.concat([dataset, aug_data])
    dataset = dataset.sample(frac=1, random_state=2).reset_index(drop=True)
    return dataset

def split_data(dataset, val_ratio=0.2):

    data_size = len(dataset)
    index_map = collections.defaultdict(list)
    labels = label_to_num(dataset['label'].values)
    for idx in range(data_size):
        label = labels[idx]
        index_map[label].append(idx)

    train_data = []
    val_data = []

    label_size = len(index_map)
    for label in range(label_size):
        idx_list = index_map[label]
        sample_size = int(len(idx_list) * val_ratio)

        val_index = random.sample(idx_list, sample_size)
        train_index = list(set(idx_list) - set(val_index))

        train_data.extend(train_index)
        val_data.extend(val_index)
        
    random.shuffle(train_data)
    random.shuffle(val_data)

    train_dataset = dataset.iloc[train_data]
    val_dataset = dataset.iloc[val_data]

    return train_dataset, val_dataset