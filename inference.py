from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    BertTokenizer,
)
from torch.utils.data import DataLoader
from load_data import *
from modules.preprocessor import EntityPreprocessor, SenPreprocessor, UnkPreprocessor
import pandas as pd
import torch
import torch.nn.functional as F
import importlib

import pickle as pickle
import numpy as np
import argparse
import os
from tqdm import tqdm
from tokenization import *


def inference(model, tokenized_sent, device, args, is_roberta=False):
    dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)
    model.eval()

    output_pred = []
    output_prob = []
    for data in tqdm(dataloader):
        with torch.no_grad():
            if args.model_name == "Rroberta":
                outputs = model(
                    input_ids=data["input_ids"].to(device),
                    attention_mask=data["attention_mask"].to(device),
                    e1_mask=data["e1_mask"].to(device),
                    e2_mask=data["e2_mask"].to(device),
                )
            elif is_roberta:
                outputs = model(
                    input_ids=data["input_ids"].to(device),
                    attention_mask=data["attention_mask"].to(device),
                )
            else:
                outputs = model(
                    input_ids=data["input_ids"].to(device),
                    attention_mask=data["attention_mask"].to(device),
                    token_type_ids=data["token_type_ids"].to(device),
                )
        logits = outputs[0]
        prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)

        output_pred.append(result)
        output_prob.append(prob)

    return (
        np.concatenate(output_pred).tolist(),
        np.concatenate(output_prob, axis=0).tolist(),
    )


def inference_ensemble(model_dir, tokenized_sent, device, args, is_roberta=False):
    dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)

    dirs = os.listdir(model_dir)
    dirs = sorted(dirs)

    final_output_prob, final_output_pred = [], []
    for i in range(len(dirs)):
        model_d = os.path.abspath(os.path.join(model_dir, dirs[i]))
        if args.model_name is not None:
            model_config = AutoConfig.from_pretrained(args.PLM)
            model_config.num_labels = 30
            mm = importlib.import_module("model")
            MyModel = getattr(mm, args.model_name)

            if MyModel.__name__ == "ConcatFourClsModel":
                model_config.update({"output_hidden_states": True})

            model = MyModel(args.PLM, config=model_config)
            model.load_state_dict(torch.load(os.path.join(model_d, "pytorch_model.pt")))
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_d)

        model.eval()
        model.to(device)

        fold_prob = []
        fold_pred = []
        for data in tqdm(dataloader):
            with torch.no_grad():
                if args.model_name == "Rroberta":
                    outputs = model(
                        input_ids=data["input_ids"].to(device),
                        attention_mask=data["attention_mask"].to(device),
                        e1_mask=data["e1_mask"].to(device),
                        e2_mask=data["e2_mask"].to(device),
                    )
                elif is_roberta:
                    outputs = model(
                        input_ids=data["input_ids"].to(device),
                        attention_mask=data["attention_mask"].to(device),
                    )
                else:
                    outputs = model(
                        input_ids=data["input_ids"].to(device),
                        attention_mask=data["attention_mask"].to(device),
                        token_type_ids=data["token_type_ids"].to(device),
                    )
            logits = outputs[0]
            prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
            logits = logits.detach().cpu().numpy()

            fold_pred.extend(logits.tolist())
            fold_prob.append(prob)

        final_output_pred.append(fold_pred)
        final_output_prob.append(np.concatenate(fold_prob, axis=0).tolist())

    return final_output_pred, final_output_prob


pers_id_index = {
    0: 4,
    1: 6,
    2: 8,
    3: 10,
    4: 11,
    5: 12,
    6: 13,
    7: 14,
    8: 15,
    9: 16,
    10: 17,
    11: 21,
    12: 23,
    13: 24,
    14: 25,
    15: 26,
    16: 27,
    17: 29,
}
orgs_id_index = {0: 1, 1: 2, 2: 3, 3: 5, 4: 7, 5: 9, 6: 18, 7: 19, 8: 20, 9: 22, 10: 28}


def inference_three_step(model_dir, Re_test_dataset, device, args, is_roberta):
    """
    ?????????(no_relation, org, per ??????) -> ?????????(per ?????? label ?????? ??????), ?????????(org ?????? label ?????? ??????) ??????
    """
    dataloader = DataLoader(Re_test_dataset, batch_size=16, shuffle=False)

    dirs = os.listdir(model_dir)
    dirs = sorted(dirs)

    # ????????? ????????? list
    final_output_prob = [[0] * 30] * 7765
    final_output_pred = [31] * 7765

    # ????????? ???, ????????? ??? label??? ????????? lsit
    no_relation_index, per_index, org_index = [], [], []

    ## ????????? ?????? ##
    fold_big_prob, fold_big_pred = [], []

    ## per?????? label ?????? ?????? ##
    fold_per_prob, fold_per_pred = [], []

    # org ?????? label ?????? ??????!
    fold_org_prob, fold_org_pred = [], []

    for what_model in dirs:
        print(what_model)
        model_d = os.path.abspath(os.path.join(model_dir, what_model))
        # big sort ?????? ????????? ???????????????! (????????? ?????? ?????? ????????? big????????? ????????? ???????????? ?????? ??????!)
        prob = []
        pred = []

        if "big" in what_model:
            model_config.num_labels = 3
        elif "per" in what_model:
            model_config.num_labels = 18
        else:
            model_config.num_labels = 11

        if args.model_name is not None:
            model_config = AutoConfig.from_pretrained(args.PLM)
            mm = importlib.import_module("model")
            MyModel = getattr(mm, args.model_name)

            if MyModel.__name__ == "ConcatFourClsModel":
                model_config.update({"output_hidden_states": True})

            model = MyModel(args.PLM, config=model_config)
            model.load_state_dict(torch.load(os.path.join(model_d, "pytorch_model.pt")))

        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_d)
        model.parameters
        model.to(device)

        model.eval()
        for data in tqdm(dataloader):
            with torch.no_grad():
                if is_roberta:
                    outputs = model(
                        input_ids=data["input_ids"].to(device),
                        attention_mask=data["attention_mask"].to(device),
                    )
                else:
                    outputs = model(
                        input_ids=data["input_ids"].to(device),
                        attention_mask=data["attention_mask"].to(device),
                        token_type_ids=data["token_type_ids"].to(device),
                    )
            logits = outputs[0]
            prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
            logits = logits.detach().cpu().numpy()
            result = np.argmax(logits, axis=-1)

            pred.extend(logits.tolist())
            prob.append(prob)

        if "big" in what_model:
            fold_big_pred.append(pred)
            fold_big_prob.append(np.concatenate(prob, axis=0).tolist())

        elif "per" in what_model:
            fold_per_pred.append(pred)
            fold_per_prob.append(np.concatenate(prob, axis=0).tolist())

        else:
            fold_org_pred.append(pred)
            fold_org_prob.append(np.concatenate(prob, axis=0).tolist())

    fold_big_pred = np.argmax(np.mean(fold_big_pred, axis=0), axis=-1)
    fold_big_prob = np.mean(fold_big_prob, axis=0).tolist()

    fold_per_pred = np.argmax(np.mean(fold_per_pred, axis=0), axis=-1)
    fold_per_prob = np.mean(fold_per_prob, axis=0).tolist()

    fold_org_pred = np.argmax(np.mean(fold_org_pred, axis=0), axis=-1)
    fold_org_prob = np.mean(fold_org_prob, axis=0).tolist()

    ## no_relation, per, org ????????? ?????? -> no_relation??? ?????? ?????? ??????
    for idx in range(7765):
        idx_prob = []
        if fold_big_pred[idx] == 0:
            final_output_pred[idx] = 0
            idx_prob.append(fold_big_prob[idx][0])
            for c in range(1, 30):
                if c in pers_id_index.values():
                    idx_prob.append((fold_big_prob[idx][1]) / 18)
                else:
                    idx_prob.append((fold_big_prob[idx][2]) / 11)
            final_output_prob[idx] = idx_prob

        elif fold_big_pred[idx] == 1:
            idx_prob = [0] * 30
            final_output_pred[idx] = orgs_id_index[fold_org_pred[idx]]
            # org ?????? ????????? ??????
            for orgsid in orgs_id_index.keys():
                idx_prob[orgs_id_index[orgsid]] = fold_org_prob[idx][orgsid]

            # org??? ???????????? ??? ??????
            for i in range(30):
                if idx_prob[i] == 0:
                    idx_prob[i] = min(fold_org_prob[idx]) / 19
            final_output_prob[idx] = idx_prob

        else:
            idx_prob = [0] * 30
            final_output_pred[idx] = pers_id_index[fold_per_pred[idx]]
            # per ?????? ????????? ??????
            for persid in pers_id_index.keys():
                idx_prob[pers_id_index[persid]] = fold_per_prob[idx][persid]

            # per??? ???????????? ????????? ??????
            for i in range(30):
                if idx_prob[i] == 0:
                    idx_prob[i] = min(fold_per_prob[idx]) / 12
            final_output_prob[idx] = idx_prob

    return final_output_pred, final_output_prob


def num_to_label(label):
    """
    ????????? ?????? ?????? class??? ?????? ????????? ????????? ?????? ?????????.
    """
    origin_label = []
    with open("dict_num_to_label.pkl", "rb") as f:
        dict_num_to_label = pickle.load(f)
    for v in label:
        origin_label.append(dict_num_to_label[v])

    return origin_label


def load_test_dataset(dataset_dir, tokenizer, sen_preprocessor, entity_preprocessor):
    """
    test dataset??? ????????? ???,
    tokenizing ?????????.
    """
    test_dataset = load_data(dataset_dir, train=False)
    test_dataset = preprocessing_dataset(
        test_dataset, sen_preprocessor, entity_preprocessor
    )
    test_label = list(map(int, test_dataset["label"].values))

    # tokenizing dataset
    tokenized_test = tokenized_dataset(test_dataset, tokenizer, is_inference=True)
    return test_dataset["id"], tokenized_test, test_label


def select_checkpoint(args):
    models_dir = args.model_dir
    dirs = os.listdir(models_dir)
    dirs = sorted(dirs)

    for i, d in enumerate(dirs, 0):
        print("(%d) %s" % (i, d))
    d_idx = input("Select directory you want to load: ")

    checkpoint_dir = os.path.abspath(os.path.join(models_dir, dirs[int(d_idx)]))
    print("checkpoint_dir is: {}".format(checkpoint_dir))

    return checkpoint_dir


def main(args):
    """
    ????????? dataset csv ????????? ?????? ????????? ?????? inference ????????? ???????????????.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.PLM)

    # load my model
    model_dir = select_checkpoint(args)

    # load test datset
    test_dataset_dir = "/opt/ml/dataset/test/test_data.csv"

    # preprocessor
    sen_preprocessor = SenPreprocessor(args.preprocessing_cmb, args.mecab_flag)
    entity_preprocessor = EntityPreprocessor(
        args.entity_flag if args.model_name != "Rroberta" else True
    )

    if args.PLM in ["klue/roberta-base", "klue/roberta-small", "klue/roberta-large"]:
        is_roberta = True
        if args.add_unk_token:
            print(model_dir + "/tokenizer")
            tokenizer = BertTokenizer.from_pretrained(model_dir + "/tokenizer")
            print(
                "new vocab size:",
                len(tokenizer.vocab) + len(tokenizer.get_added_vocab()),
            )
    else:
        is_roberta = False

    test_id, test_dataset, test_label = load_test_dataset(
        test_dataset_dir, tokenizer, sen_preprocessor, entity_preprocessor
    )
    Re_test_dataset = (
        RE_Dataset(test_dataset, test_label)
        if args.model_name != "Rroberta"
        else DatasetForRRoBERTa(test_dataset, test_label, tokenizer)
    )

    if args.model_type:
        pred_answer, output_prob = inference_three_step(
            model_dir, Re_test_dataset, device, args, is_roberta
        )  # model?????? class ??????
        pred_answer = num_to_label(pred_answer)

    elif args.k_fold:
        pred_answer, output_prob = inference_ensemble(
            model_dir, Re_test_dataset, device, args, is_roberta
        )  # model?????? class ??????
        pred_answer = np.mean(pred_answer, axis=0)
        pred_answer = np.argmax(pred_answer, axis=-1)
        pred_answer = num_to_label(pred_answer)
        output_prob = np.mean(output_prob, axis=0).tolist()

    else:
        if args.model_name is not None:
            model_config = AutoConfig.from_pretrained(args.PLM)
            model_config.num_labels = 30
            mm = importlib.import_module("model")
            MyModel = getattr(mm, args.model_name)

            if MyModel.__name__ == "ConcatFourClsModel":
                model_config.update({"output_hidden_states": True})

            model = MyModel(args.PLM, config=model_config)
            model.load_state_dict(
                torch.load(os.path.join(model_dir, "pytorch_model.pt"))
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.parameters
        model.to(device)

        pred_answer, output_prob = inference(
            model, Re_test_dataset, device, args, is_roberta
        )  # model?????? class ??????
        pred_answer = num_to_label(pred_answer)  # ????????? ??? class??? ?????? ????????? ????????? ??????.

    # make csv file with predicted answer
    #########################################################
    # ?????? directory??? columns??? ????????? ??????????????? ????????????.
    output = pd.DataFrame(
        {
            "id": test_id,
            "pred_label": pred_answer,
            "probs": output_prob,
        }
    )

    sub_name = model_dir.split("/")[-1]
    # ??????????????? ????????? ????????? ?????? csv ?????? ????????? ??????.
    output.to_csv(f"./prediction/submission_{sub_name}.csv", index=False)
    #### ??????!! ##############################################
    print("---- Finish! ----")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model dir
    parser.add_argument("--model_dir", type=str, default="./best_models")
    parser.add_argument(
        "--PLM", type=str, help="model type (example: klue/bert-base)", required=True
    )
    parser.add_argument(
        "--entity_flag",
        default=False,
        action="store_true",
        help="Train??? ?????????????????? ????????? (default: False)",
    )
    parser.add_argument(
        "--preprocessing_cmb", nargs="+", help="<Required> Set flag (example: 0 1 2)"
    )
    parser.add_argument(
        "--mecab_flag",
        default=False,
        action="store_true",
        help="Train??? ?????????????????? ????????? (default: False)",
    )
    parser.add_argument(
        "--add_unk_token",
        default=False,
        action="store_true",
        help="add unknown token in vocab (default: False)",
    )
    parser.add_argument("--k_fold", type=int, default=0, help="not k fold(defalut: 0)")
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="if want, you have to enter your model class name",
    )
    parser.add_argument(
        "--model_type",
        default=False,
        action="store_true",
        help="(default: False - ?????????/????????? X)",
    )

    args = parser.parse_args()
    print(args)

    os.makedirs("./prediction", exist_ok=True)
    main(args)
