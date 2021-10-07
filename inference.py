from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, BertTokenizer
from torch.utils.data import DataLoader
from load_data import *
from Preprocessing.preprocessor import EntityPreprocessor, SenPreprocessor, UnkPreprocessor
import pandas as pd
import torch
import torch.nn.functional as F

import pickle as pickle
import numpy as np
import argparse
import os
from tqdm import tqdm
from tokenization import tokenized_dataset


def inference(model, tokenized_sent, device, is_roberta=False):
    """
    test dataset을 DataLoader로 만들어 준 후,
    batch_size로 나눠 model이 예측 합니다.
    """
    dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)
    model.eval()
    output_pred = []
    output_prob = []
    for data in tqdm(dataloader):
        with torch.no_grad():
            if is_roberta:
                outputs = model(
                    input_ids=data['input_ids'].to(device),
                    attention_mask=data['attention_mask'].to(device)
                )
            else:
                outputs = model(
                    input_ids=data['input_ids'].to(device),
                    attention_mask=data['attention_mask'].to(device),
                    token_type_ids=data['token_type_ids'].to(device)
                )
        logits = outputs[0]
        prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)

        output_pred.append(result)
        output_prob.append(prob)

    return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()

def inference_ensemble(model_dir, tokenized_sent, device, is_roberta=False):
    dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)
    
    dirs = os.listdir(model_dir)
    dirs = sorted(dirs)
    
    final_output_prob=[]
    final_output_pred=[]
    for i in range(len(dirs)):
        model_d = os.path.abspath(os.path.join(model_dir, dirs[i]))
        model = AutoModelForSequenceClassification.from_pretrained(model_d)
        model.parameters
        model.to(device)
        
        model.eval()
        fold_prob=[]
        fold_pred=[]
        for data in tqdm(dataloader):
            with torch.no_grad():
                if is_roberta:
                    outputs = model(
                        input_ids=data['input_ids'].to(device),
                        attention_mask=data['attention_mask'].to(device),
                    )
                else:
                    outputs = model(
                        input_ids=data['input_ids'].to(device),
                        attention_mask=data['attention_mask'].to(device),
                        token_type_ids=data['token_type_ids'].to(device)
                    )
            logits = outputs[0]
            prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
            logits = logits.detach().cpu().numpy()
            
            fold_pred.extend(logits.tolist())
            fold_prob.append(prob)
        
        final_output_pred.append(fold_pred)
        final_output_prob.append(np.concatenate(fold_prob, axis=0).tolist())
        
    return final_output_pred, final_output_prob


def num_to_label(label):
    """
      숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
    """
    origin_label = []
    with open('dict_num_to_label.pkl', 'rb') as f:
        dict_num_to_label = pickle.load(f)
    for v in label:
        origin_label.append(dict_num_to_label[v])

    return origin_label


def load_test_dataset(dataset_dir, tokenizer, sen_preprocessor, entity_preprocessor):
    """
      test dataset을 불러온 후,
      tokenizing 합니다.
    """
    test_dataset = load_data(dataset_dir, train=False)
    test_dataset = preprocessing_dataset(test_dataset, sen_preprocessor, entity_preprocessor)
    test_label = list(map(int, test_dataset['label'].values))
    
    # tokenizing dataset
    tokenized_test = tokenized_dataset(
        test_dataset, tokenizer, is_inference=True)
    return test_dataset['id'], tokenized_test, test_label


def select_checkpoint(args):
    models_dir = args.model_dir
    dirs = os.listdir(models_dir)
    dirs = sorted(dirs)

    for i, d in enumerate(dirs, 0):
        print("(%d) %s" % (i, d))
    d_idx = input("Select directory you want to load: ")

    checkpoint_dir = os.path.abspath(
        os.path.join(models_dir, dirs[int(d_idx)]))
    print("checkpoint_dir is: {}".format(checkpoint_dir))

    return checkpoint_dir


def main(args):
    """
      주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.PLM)

    # load my model
    model_dir = select_checkpoint(args)

    # load test datset
    test_dataset_dir = "/opt/ml/dataset/test/test_data.csv"

    # preprocessor
    sen_preprocessor = SenPreprocessor(args.preprocessing_cmb, args.mecab_flag)
    entity_preprocessor = EntityPreprocessor(args.entity_flag)

    if args.PLM in ['klue/roberta-base', 'klue/roberta-small', 'klue/roberta-large']:
        is_roberta = True
        if args.add_unk_token :
            print(model_dir+'/tokenizer')
            tokenizer = BertTokenizer.from_pretrained(model_dir+'/tokenizer')
            print('new vocab size:', len(tokenizer.vocab)+len(tokenizer.get_added_vocab()))
    else:
        is_roberta = False

    test_id, test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer, sen_preprocessor, entity_preprocessor)
    Re_test_dataset = RE_Dataset(test_dataset, test_label)

    if args.k_fold:
        pred_answer, output_prob = inference_ensemble(model_dir, Re_test_dataset, device, is_roberta)  # model에서 class 추론
        pred_answer = np.mean(pred_answer,axis=0)
        pred_answer = np.argmax(pred_answer,axis=-1)
        pred_answer = num_to_label(pred_answer)
        output_prob = np.mean(output_prob,axis=0).tolist()
    
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.parameters
        model.to(device)
        
        pred_answer, output_prob = inference(
        model, Re_test_dataset, device, is_roberta)  # model에서 class 추론
        pred_answer = num_to_label(pred_answer)  # 숫자로 된 class를 원래 문자열 라벨로 변환.

    # make csv file with predicted answer
    #########################################################
    # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
    output = pd.DataFrame(
        {'id': test_id, 'pred_label': pred_answer, 'probs': output_prob, })

    sub_name = model_dir.split('/')[-1]
    # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
    output.to_csv(f"./prediction/submission_{sub_name}.csv", index=False)
    #### 필수!! ##############################################
    print('---- Finish! ----')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model dir
    parser.add_argument('--model_dir', type=str, default="./best_models")
    parser.add_argument(
        '--PLM', type=str, help='model type (example: klue/bert-base)', required=True)
    parser.add_argument(
        '--entity_flag', default=False, action='store_true', help='Train에 사용했던거랑 똑같이 (default: False)')
    # parser.add_argument(
    #     '--preprocessing_flag', default=False, action='store_true', help='Train에 사용했던거랑 똑같이 (default: False)')
    parser.add_argument(
        '--preprocessing_cmb', nargs='+', help='<Required> Set flag (example: 0 1 2)')
    parser.add_argument(
        '--mecab_flag', default=False, action='store_true', help='Train에 사용했던거랑 똑같이 (default: False)')
    parser.add_argument(
        '--add_unk_token', default=False, action='store_true', help='add unknown token in vocab (default: False)')
    
    parser.add_argument("--k_fold", type=int, default=0, help='not k fold(defalut: 0)')
    
    args = parser.parse_args()
    print(args)

    os.makedirs('./prediction', exist_ok=True)
    main(args)
