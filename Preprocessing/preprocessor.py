import pandas as pd

import re
import itertools
from collections import Counter
from typing import Dict, List, Tuple

from tqdm import tqdm
from transformers import AutoTokenizer
from konlpy.tag import Mecab

# UKN tokenizer 처리
""" 
Roberta tokenizer 기준으로 unk token 줄이는 방향의 preprocessing
다른 tokenizer에서 추가 기능이 필요하시면 말씀해주세요

해당 전처리는 train, test에 동일하게 적용되어야함
"""


def remove_special_char(sentence):
    """ 특수문자 및 독일어 제거, 수정"""
    sentence = re.sub(r'[À-ÿ]+', '', sentence)  # 독일어
    sentence = re.sub(r'[\u0600-\u06FF]+', '', sentence)  # 사우디어
    sentence = re.sub(r'[ß↔Ⓐب€☎☏±]+', '', sentence)
    sentence = re.sub('–', '─', sentence)
    sentence = re.sub('⟪', '《', sentence)
    sentence = re.sub('⟫', '》', sentence)
    sentence = re.sub('･', '・', sentence)
    return sentence


def add_space_char(sentence):
    def add_space(match):
        res_str = ', '.join(match.group().split(',')).rstrip()
        return res_str
    p = re.compile(r'([기-힣\w\-]+,)+[기-힣\w\-]+')
    sentence = p.sub(add_space, sentence)
    return sentence


def substitution_date(sentence):
    """
    기간 표시 '-' => '~'
    1223년 – => 1223년 ~ 
    """
    def sub_tibble(match):
        res_str = re.sub('[–\-]', '~', match.group())
        return res_str
    re_patterns = [
        r'(\d{2,4}년\s*)(\d{1,2}[월|일]\s*)(\d{1,2}[월|일])\s*[–\-]',
        r'(\d{2,4}년\s*)(\d{1,2}[월|일]\s*)\s*[–\-]',
        r'(\d{2,4}년\s*)\s*[–\-]',
        r'\((\d{4}[–\-]\d{2,4})\)'
    ]
    for re_pattern in re_patterns:
        p = re.compile(re_pattern)
        sentence = p.sub(sub_tibble, sentence)
    return sentence


def sentence_processing(data):
    new_sentence = []
    for row in tqdm(data.values):
        sentence, subject_entity, object_entity = row[1], eval(
            row[2]), eval(row[3])
        sub_start_idx, sub_end_idx, sub_type = subject_entity[
            'start_idx'], subject_entity['end_idx'], subject_entity['type']
        ob_start_idx, ob_end_idx, ob_type = object_entity[
            'start_idx'], object_entity['end_idx'], object_entity['type']

        if sub_start_idx < ob_start_idx:
            sentence = sentence[:sub_start_idx] + ' @ * ' + sub_type + ' * ' + sentence[sub_start_idx:sub_end_idx+1] + ' @ ' + \
                sentence[sub_end_idx+1:ob_start_idx] + ' # ~ ' + ob_type + ' ~ ' + \
                sentence[ob_start_idx:ob_end_idx+1] + \
                ' # ' + sentence[ob_end_idx+1:]
        else:
            sentence = sentence[:ob_start_idx] + ' # ~ ' + ob_type + ' ~ ' + sentence[ob_start_idx:ob_end_idx+1] + ' # ' + sentence[ob_end_idx +
                                                                                                                                    1:sub_start_idx] + ' @ * ' + sub_type + ' * ' + sentence[sub_start_idx:sub_end_idx+1] + ' @ ' + sentence[sub_end_idx+1:]

        sentence = re.sub('\s+', " ", sentence)
        new_sentence.append(sentence)

    print("Finish type entity processing!!!")

    return new_sentence


def mecab_processing(sentence):
    tokenizer = Mecab()

    tokens = tokenizer.morphs(sentence)
    mecab_sentence = " ".join(tokens)

    return mecab_sentence
