import pandas as pd
import collections
import random
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold


class RE_Dataset(Dataset):
    """
    Relation Extraction Dataset 구성을 위한 class
    """

    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: val[idx] for key, val in self.dataset.items()
        }  # id,sentence, subject_entity,object_entity,label
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class r_RE_Dataset(RE_Dataset):
    """rRobert도 그렇고. .r_RE_Dataset이 무엇입니까..ㅠㅠ - 문영기 멘토님
    subject, object entity embedding을 만들어주기 위한 클래스
    subject : e1_mask
    object : e2_mask
    """

    def __init__(self, dataset, labels, tokenizer):
        self.dataset = dataset
        self.labels = labels

        self.sub_id = tokenizer.get_vocab()["◈"]
        self.obj_id = tokenizer.get_vocab()["↑"]

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.dataset.items()}
        item["labels"] = torch.tensor(self.labels[idx])

        sub_flag, obj_flag = 0, 0
        e1_mask, e2_mask = [], []
        for enc in item["input_ids"]:
            if enc == self.sub_id:
                sub_flag += 1
                e1_mask.append(0)
                e2_mask.append(0)
                continue
            elif enc == self.obj_id:
                obj_flag += 1
                e1_mask.append(0)
                e2_mask.append(0)
                continue
            if sub_flag == 1:
                e1_mask.append(1)
            else:
                e1_mask.append(0)
            if obj_flag == 1:
                e2_mask.append(1)
            else:
                e2_mask.append(0)
        item["e1_mask"] = torch.tensor(e1_mask)
        item["e2_mask"] = torch.tensor(e2_mask)
        return item


def preprocessing_dataset(dataset, sen_preprocessor, entity_preprocessor):
    """처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""

    subject_entity, object_entity = list(
        zip(
            *dataset.apply(
                lambda x: [x["subject_entity"]["word"], x["object_entity"]["word"]],
                axis=1,
            )
        )
    )

    # Data Sentence Entity processor
    dataset = entity_preprocessor(dataset)

    # Sentence Preprocessor
    sen_data = [sen_preprocessor(sen) for sen in dataset["sentence"]]
    subject_data = [sen_preprocessor(sub) for sub in subject_entity]
    object_data = [sen_preprocessor(obj) for obj in object_entity]

    out_dataset = pd.DataFrame(
        {
            "id": dataset["id"],
            "sentence": sen_data,
            "subject_entity": subject_data,
            "object_entity": object_data,
            "label": dataset["label"],
        }
    )
    return out_dataset


def load_data(dataset_dir, k_fold=0, val_ratio=0, train=True, model_type="default"):
    """
    csv 파일을 경로에 맡게 불러 옵니다.
    k_fold와 val_ratio를 통해 train data와 validation data를 나눕니다.
    단, k_fold가 우선순위고, k_fold가 없을 경우 val_ratio에 따라 split을 진행합니다.
    """
    dataset = pd.read_csv(dataset_dir)
    if train == True:
        dataset = dataset.drop_duplicates(
            ["sentence", "subject_entity", "object_entity", "label"], keep="first"
        )

        # 라벨링이 다른 데이터 제거
        dataset = dataset.drop(index=[6749, 8364, 22258, 277, 25094])
        dataset = dataset.reset_index(drop=True)

    if model_type == "per_sort":
        dataset = dataset[dataset["label"].str.contains("per")].reset_index(drop=True)
    elif model_type == "org_sort":
        dataset = dataset[dataset["label"].str.contains("org")].reset_index(drop=True)

    # datatype 변경
    dataset["subject_entity"] = dataset.subject_entity.map(eval)
    dataset["object_entity"] = dataset.object_entity.map(eval)
    print(
        ">>>>>>>>>>Finish pre processing loaded data(drop duplicated and miss-labeled data"
    )

    if k_fold > 0 and train == True:  # train, split by kfold
        return split_by_kfolds(dataset, k_fold)
    elif val_ratio > 0:  # train, split by val_ratio
        return split_by_val_ratio(dataset, val_ratio)
    elif train == False:  # inference
        return dataset
    else:  # train, not split
        return [[dataset, None]]


def split_by_kfolds(dataset, k_fold):
    X = dataset.drop(["label"], axis=1)
    y = dataset["label"]
    skf = StratifiedKFold(n_splits=k_fold, shuffle=True)
    return [
        [dataset.iloc[train_dset], dataset.iloc[val_dset]]
        for train_dset, val_dset in skf.split(X, y)
    ]


def split_by_val_ratio(dataset, val_ratio):
    data_size = len(dataset)
    index_map = collections.defaultdict(list)
    for idx in range(data_size):
        label = dataset.iloc[idx]["label"]
        index_map[label].append(idx)
    train_indices = []
    val_indices = []

    for label in index_map.keys():
        idx_list = index_map[label]
        val_size = int(len(idx_list) * val_ratio)

        val_index = random.sample(idx_list, val_size)
        train_index = list(set(idx_list) - set(val_index))

        train_indices.extend(train_index)
        val_indices.extend(val_index)

    random.shuffle(train_indices)
    random.shuffle(val_indices)
    train_dset = dataset.iloc[train_indices]
    val_dset = dataset.iloc[val_indices]
    return [[train_dset, val_dset]]


#### for mlm
class MLM_Dataset(Dataset):
    """Masked Language Model Dataset 구성을 위한 class."""

    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.dataset.items()}
        inputs, labels = mask_tokens(item["input_ids"], self.tokenizer)
        item["input_ids"] = inputs.squeeze()
        item["labels"] = labels.squeeze()
        return item

    def __len__(self):
        return len(self.dataset["input_ids"])


def load_mlm_data(dataset_dir):
    """csv 파일을 경로에 맞게 불러 옵니다."""
    pd_dataset = pd.read_csv(dataset_dir)
    if "train" in dataset_dir:
        pd_dataset = pd_dataset.drop_duplicates(
            ["sentence"], keep="first"
        )  # 중복되는 문장 제거

    pd_dataset = pd.DataFrame(
        {
            "id": pd_dataset["id"],
            "sentence": pd_dataset["sentence"],
            "label": pd_dataset["label"],
        }
    )
    return pd_dataset


def mask_tokens(inputs, tokenizer, mlm_probability=0.15, pad=True):
    inputs = torch.tensor(inputs)
    labels = inputs.clone()

    # mlm_probability은 15%로 BERT에서 사용하는 확률
    probability_matrix = torch.full(labels.shape, mlm_probability)

    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(
            labels.tolist(), already_has_special_tokens=True
        )
    ]
    probability_matrix.masked_fill_(
        torch.tensor(special_tokens_mask, dtype=torch.bool).squeeze(), value=0.0
    )

    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens tokenizer.mask_token ([MASK])
    indices_replaced = (
        torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    )
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]
    return inputs, labels
