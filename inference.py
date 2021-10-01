from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import torch.nn.functional as F

import pickle as pickle
import numpy as np
import argparse
import os
from tqdm import tqdm


def inference(model, tokenized_sent, device, is_roberta=False):
    dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)
    model.eval()
    output_pred = []
    output_prob = []
    for i, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            if is_roberta:
                outputs = model(
                    input_ids=data['input_ids'].to(device),
                    attention_mask=data['attention_mask'].to(device),
                    # token_type_ids=data['token_type_ids'].to(device)
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
        for i1, data in enumerate(tqdm(dataloader)):
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
    with open('/opt/ml/klue-level2-nlp-02/dict_num_to_label.pkl', 'rb') as f:
        dict_num_to_label = pickle.load(f)
    for v in label:
        origin_label.append(dict_num_to_label[v])

    return origin_label


def load_test_dataset(dataset_dir, tokenizer):
    """
      test dataset을 불러온 후,
      tokenizing 합니다.
    """
    test_dataset = load_data(dataset_dir)
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
    Tokenizer_NAME = args.PLM
    tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)
    # load test datset
    test_dataset_dir = "/opt/ml/dataset/test/test_data.csv"
    
    if Tokenizer_NAME in ['klue/roberta-base', 'klue/roberta-small', 'klue/roberta-large']:
        is_roberta = True
    else:
        is_roberta = False
        
    test_id, test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
    Re_test_dataset = RE_Dataset(test_dataset, test_label)
    model_dir = select_checkpoint(args)

    # load my model
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
        
        pred_answer, output_prob = inference(model, Re_test_dataset, device, is_roberta)  # model에서 class 추론
        pred_answer = num_to_label(pred_answer)  # 숫자로 된 class를 원래 문자열 라벨로 변환.

    
    output = pd.DataFrame(
        {'id': test_id, 'pred_label': pred_answer, 'probs': output_prob, })

    output.to_csv('/opt/ml/klue-level2-nlp-02/prediction/submission.csv', index=False)
    print('---- Finish! ----')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model dir
    parser.add_argument('--model_dir', type=str, default="/opt/ml/klue-level2-nlp-02/best_models")
    parser.add_argument('--PLM', type=str, help='model type (example: klue/bert-base)', required=True)
    
    parser.add_argument("--k_fold", type=int, default=0, help='not k fold(defalut: 0)')

    args = parser.parse_args()
    print(args)

    os.makedirs('./prediction', exist_ok=True)
    main(args)
