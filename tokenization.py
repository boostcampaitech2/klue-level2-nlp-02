from transformers.file_utils import PaddingStrategy

def tokenized_dataset(dataset, tokenizer, is_inference=False, is_mlm=False):
    """ tokenizer에 따라 sentence를 tokenizing 합니다."""
    concat_entity = []
    
    concat_entity = list(dataset.apply(lambda x : x['subject_entity'] + "[SEP]" + x['object_entity'], axis= 1))
    """ Roberta TTI_flag with dynamic_padding"""
    if 'roberta' in tokenizer.name_or_path and not 'xlm' in tokenizer.name_or_path:
        tokenized_sentences = tokenizer(
            concat_entity,
            list(dataset['sentence']),
            return_tensors='pt' if is_mlm or is_inference else None,
            padding=True if is_mlm or is_inference else PaddingStrategy.DO_NOT_PAD.value,
            truncation=True,
            max_length=256,
            add_special_tokens=True,
            return_token_type_ids=False
        )
    else:
        tokenized_sentences = tokenizer(
            concat_entity,
            list(dataset['sentence']),
            return_tensors='pt' if is_mlm or is_inference else None,
            padding=True if is_mlm or is_inference else PaddingStrategy.DO_NOT_PAD.value,
            truncation=True,
            max_length=256,
            add_special_tokens=True
        )
    return tokenized_sentences

def tokenized_mlm_dataset(dataset, tokenizer):
    """ tokenizer에 따라 sentence를 tokenizing 합니다."""
    if 'roberta' in tokenizer.name_or_path and not 'xlm' in tokenizer.name_or_path:
        tokenized_sentences = tokenizer(
            list(dataset['sentence']),
            return_tensors="pt",
            padding=True, # TO DO: dynamic_padding 사용하는데도 에러 발생 (mlm)
            truncation=True,
            max_length=256,
            add_special_tokens=True,
            return_token_type_ids=False
        )
    else:
        tokenized_sentences = tokenizer(
            list(dataset['sentence']),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            add_special_tokens=True
        )
    return tokenized_sentences