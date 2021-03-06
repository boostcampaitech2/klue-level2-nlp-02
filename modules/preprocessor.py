import re
from itertools import chain
from collections import Counter
from typing import Dict, List, Tuple

from tqdm import tqdm
from konlpy.tag import Mecab


class SenPreprocessor:
    def __init__(self, preprocessing_cmb, mecab_flag):
        self.preprocessing_cmb = preprocessing_cmb
        self.mecab_flag = mecab_flag
        if mecab_flag == 1:
            self.mecab = Mecab()

    def __call__(self, sentence):
        if self.preprocessing_cmb != None:
            if "0" in self.preprocessing_cmb:
                sentence = self.remove_special_char(sentence)
            if "1" in self.preprocessing_cmb:
                sentence = self.substitution_special_char(sentence)
            if "2" in self.preprocessing_cmb:
                sentence = self.substitution_date(sentence)
            if "3" in self.preprocessing_cmb:
                sentence = self.add_space_char(sentence)

        if self.mecab_flag == True:
            sentence = self.mecab_processing(sentence)

        return sentence

    def remove_special_char(self, sentence):
        sentence = re.sub(r"[À-ÿ]+", "", sentence)
        sentence = re.sub(r"[\u0600-\u06FF]+", "", sentence)
        sentence = re.sub(r"[\u00C0-\u02B0]+", "", sentence)
        sentence = re.sub(r"[ß↔Ⓐب€☎☏±∞]+", "", sentence)
        return sentence

    def substitution_special_char(self, sentence):
        sentence = re.sub("–", "─", sentence)
        sentence = re.sub("⟪", "《", sentence)
        sentence = re.sub("⟫", "》", sentence)
        sentence = re.sub("･", "・", sentence)
        sentence = re.sub("µ", "ℓ", sentence)
        sentence = re.sub("®", "㈜", sentence)
        sentence = re.sub("～", "㈜", sentence)
        return sentence

    def add_space_char(self, sentence):
        def add_space(match):
            res_str = ", ".join(match.group().split(",")).rstrip()
            return res_str

        p = re.compile(r"([기-힣\w\-]+,)+[기-힣\w\-]+")
        sentence = p.sub(add_space, sentence)
        return sentence

    def substitution_date(self, sentence):
        def sub_tibble(match):
            res_str = re.sub("[–\-]", "~", match.group())
            return res_str

        re_patterns = [
            r"(\d{2,4}년\s*)(\d{1,2}[월|일]\s*)(\d{1,2}[월|일])\s*[–\-]",
            r"(\d{2,4}년\s*)(\d{1,2}[월|일]\s*)\s*[–\-]",
            r"(\d{2,4}년\s*)\s*[–\-]",
            r"\((\d{4}[–\-]\d{2,4})\)",
        ]

        for re_pattern in re_patterns:
            p = re.compile(re_pattern)
            sentence = p.sub(sub_tibble, sentence)
        return sentence

    def mecab_processing(self, sentence):
        tokens = self.mecab.morphs(sentence)
        mecab_sentence = " ".join(tokens)
        return mecab_sentence


# UKN tokenizer 처리
class UnkPreprocessor:
    """
    tokenizer가 unk으로 분류하는 단어들 찾기 및 추가
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, sent: List, sub: List, obj: List):
        print("-" * 100)
        print("before add unk, tokenizer count:", len(self.tokenizer.vocab.keys()))
        print("sentence unknown token searching...")
        UNK_sentence_list = chain(*[self.UNK_word_and_chr(s) for s in tqdm(sent)])

        # for_add = [token for token, cnt in Counter(UNK_sentence_list).items() if cnt >= 10]
        for_add = [token for token, cnt in Counter(UNK_sentence_list).items()]
        print(for_add[:20])

        print("entity unknown token searching...")
        UNK_entity_list = chain(*[self.UNK_word_and_chr(w) for w in tqdm(sub + obj)])
        # for_add += [token for token, cnt in Counter(UNK_entity_list).items() if cnt >= 2]
        for_add += [token for token, cnt in Counter(UNK_entity_list).items()]
        print("add unk example:", for_add[:20])

        for_add = list(set(for_add))
        added_token_num = self.tokenizer.add_tokens(for_add)

        print("after add unk, toknizer count:", len(self.tokenizer.vocab.keys()))
        print("added_token_num:", added_token_num)
        return self.tokenizer, added_token_num

    def UNK_word_and_chr(self, text: str) -> Tuple[List[str], List[str]]:
        """
        정규식을 이용하여 문장기호 띄어쓰기 및 split을 이용한UNK subword 찾기
        """
        sub_word_UNK_list = []

        def add_space(match):
            bracket = match.group()
            added = " " + bracket + " "
            return added

        p = re.compile(r'[\([)\]|,|-|~|-|‘|’|"|\']')
        words_list = p.sub(add_space, text).split()
        for word in words_list:
            subwordpieces_ID_encoded = self.tokenizer.tokenize(word)
            Known_subword = self.subword_parsing(subwordpieces_ID_encoded)

            for sub_char, NK_char in zip(word, Known_subword):
                if sub_char != NK_char and len(word) == len(Known_subword):
                    sub_word_UNK_list.append(sub_char)
                elif sub_char != NK_char and len(word) != len(Known_subword):
                    sub_word_UNK_list.append(word)
                    break
        return sub_word_UNK_list

    def subword_parsing(self, wordpiece: List) -> List[str]:
        """subword # 제거용"""
        Known_char = []
        for subword in wordpiece:
            if subword == self.tokenizer.unk_token:
                Known_char.append(self.tokenizer.unk_token)
            else:
                string = subword.replace("#", "")
                Known_char.extend(string)
        return Known_char


# Typed Entity Marker
class EntityPreprocessor:
    def __init__(self, entity_flag):
        self.entity_flag = entity_flag

    def __call__(self, dataset):
        if self.entity_flag:
            sen_preprocessed = self.preprocess(dataset)
            dataset.sentence = sen_preprocessed
        return dataset

    def preprocess(self, data):
        new_sentence = []
        for idx in tqdm(range(len(data))):
            row = data.iloc[idx]
            sentence, subject_entity, object_entity = (
                row["sentence"],
                row["subject_entity"],
                row["object_entity"],
            )
            sub_start_idx, sub_end_idx, sub_type = (
                subject_entity["start_idx"],
                subject_entity["end_idx"],
                subject_entity["type"],
            )
            ob_start_idx, ob_end_idx, ob_type = (
                object_entity["start_idx"],
                object_entity["end_idx"],
                object_entity["type"],
            )

            if sub_start_idx < ob_start_idx:
                sentence = (
                    sentence[:sub_start_idx]
                    + " @ ◈ "
                    + sub_type
                    + " ◈ "
                    + sentence[sub_start_idx : sub_end_idx + 1]
                    + " @ "
                    + sentence[sub_end_idx + 1 : ob_start_idx]
                    + " # ↑ "
                    + ob_type
                    + " ↑ "
                    + sentence[ob_start_idx : ob_end_idx + 1]
                    + " # "
                    + sentence[ob_end_idx + 1 :]
                )
            else:
                sentence = (
                    sentence[:ob_start_idx]
                    + " # ↑ "
                    + ob_type
                    + " ↑ "
                    + sentence[ob_start_idx : ob_end_idx + 1]
                    + " # "
                    + sentence[ob_end_idx + 1 : sub_start_idx]
                    + " @ ◈ "
                    + sub_type
                    + " ◈ "
                    + sentence[sub_start_idx : sub_end_idx + 1]
                    + " @ "
                    + sentence[sub_end_idx + 1 :]
                )

            sentence = re.sub("\s+", " ", sentence)
            new_sentence.append(sentence)

        print("Finish type entity processing!!!")
        return new_sentence
