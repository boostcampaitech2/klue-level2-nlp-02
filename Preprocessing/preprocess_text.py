import pandas as pd

import re
import itertools
from collections import Counter
from typing import  Dict, List, Tuple

from tqdm import tqdm
from transformers import AutoTokenizer


""" 
Roberta tokenizer 기준으로 unk token 줄이는 방향의 preprocessing
다른 tokenizer에서 추가 기능이 필요하시면 말씀해주세요

해당 전처리는 train, test에 동일하게 적용되어야함
"""

def remove_special_char(sentence):
    """ 특수문자 및 독일어 제거, 수정"""
    sentence = re.sub(r'[À-ÿß↔Ⓐب☎☏±]+','', sentence)
    sentence = re.sub('–','─', sentence)
    return sentence


def add_space_char(sentence) :
    def add_space(match) :
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
    def sub_tibble(match) :
        res_str = re.sub('[–\-]','~',match.group())
        return res_str
    re_patterns = [
        r'(\d{2,4}년\s*)(\d{1,2}[월|일]\s*)(\d{1,2}[월|일])\s*[–\-]',
        r'(\d{2,4}년\s*)(\d{1,2}[월|일]\s*)\s*[–\-]',
        r'(\d{2,4}년\s*)\s*[–\-]',
        r'\((\d{4}[–\-]\d{2,4})\)'
    ]
    for re_pattern in re_patterns :
        p = re.compile(re_pattern)
        sentence = p.sub(sub_tibble, sentence)   
    return sentence

def add_space_year(sentence):
    """
    숫자와 년 사이에 공백
    1223년 => 1223 년 => ⑦ 년
    """
    def add_space(match) :
        # res_str = '⑦ ' + match.group()[4:]
        res_str =  match.group()[:4] +' ' + match.group()[4:]
        return res_str
    p = re.compile(r'\d{4}년')
    sentence = p.sub(add_space, sentence)
    return sentence