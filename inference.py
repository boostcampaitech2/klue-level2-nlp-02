# -*- coding: utf-8 -*-
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
    """
    test dataset을 DataLoader로 만들어 준 후,
    batch_size로 나눠 model이 예측 합니다.
    """
    dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)
    model.eval()
    output_pred = []
    output_prob = []
    for i, data in enumerate(tqdm(dataloader)):
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



def inference_three_step(model_dir, tokenized_sent, device, is_roberta=False):
    pers_id_index = {0:4,1:6,2:8,3:10,4:11,5:12,6:13,7:14,8:15,9:16,10:17,11:21,12:23,13:24,14:25,15:26,16:27,17:29}
    orgs_id_index = {0:1,1:2,2:3,3:5,4:7,5:9,6:18,7:19,8:20,9:22,10:28}
    
    dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)
    
    dirs = os.listdir(model_dir)
    dirs = sorted(dirs)
    
    # 최종값 저장할 list
    final_output_prob=[[0]*30]*7765
    final_output_pred=[31]*7765
    
    # 대분류 시, 분류된 각 label을 저장할 lsit
    no_relation_index=[]
    per_index=[]
    org_index=[]
    
    
    ## 대분류 시작 ##
    big_prob=[]
    big_pred=[]
    for bigmodel in dirs:
        
        # big sort 관련 모델을 불러옵니다! (대분류 시에 모델 이름이 big이라는 단어가 들어가야 수행 가능!)
        if "big" in bigmodel:
            
            model_d = os.path.abspath(os.path.join(model_dir, bigmodel))
            model = AutoModelForSequenceClassification.from_pretrained(model_d)
            model.parameters
            model.to(device)
            
            model.eval()
            
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
                result = np.argmax(logits, axis=-1)
                
                big_pred.append(result)
                big_prob.append(prob)
    
    
    per_index=[]
    org_index=[]
    ## no_relation, per, org 분류 저장
    big_prob_concatenate = np.concatenate(big_prob,axis=0).tolist()
    big_pred_concatenate = np.concatenate(big_pred).tolist()
    for idx in range(7765):
        idx_prob=[]
        if big_pred_concatenate[idx]==0:
            final_output_pred[idx]=0
            k=big_prob_concatenate[idx][0]
            idx_prob.append(k)
            ## 제출 후 작성한 코드 ## -> 좀 더 prob를 정교하게 주기 위해서!
            perprob=(big_prob_concatenate[idx][1])/18
            orgprob=(big_prob_concatenate[idx][2])/11
            for c in range(1,30):
                if c in pers_id_index.values():
                    idx_prob.append(perprob)
                else:
                    idx_prob.append(orgprob)
            final_output_prob[idx]=idx_prob
            ##############################################################
        
        elif big_pred_concatenate[idx]==1:
            per_index.append(idx)
        
        else:
            org_index.append(idx)
    
    print("Finish big sort")
            
    ## per관련 label 시작 ##
    
    
    per_prob=[]
    per_pred=[]
    for permodel in dirs:
        if "per" in permodel:
            model_d = os.path.abspath(os.path.join(model_dir, permodel))
            model = AutoModelForSequenceClassification.from_pretrained(model_d)
            model.parameters
            model.to(device)
                    
            model.eval()
            
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
                result = np.argmax(logits, axis=-1)
                    
                per_prob.append(prob)
                per_pred.append(result)
                    
            
            per_prob_concatenate = np.concatenate(per_prob,axis=0).tolist()
            per_pred_concatenate = np.concatenate(per_pred).tolist()
            for iz in range(7765):
                if iz in per_index:
                    idx_prob=[0]*30
                    final_output_pred[iz] = pers_id_index[per_pred_concatenate[iz]]
                    min_prob = min(per_prob_concatenate[iz])
                    for persid in pers_id_index.keys():
                        idx_prob[pers_id_index[persid]]=per_prob_concatenate[iz][persid]
                    for i in range(30):
                        if idx_prob[i]==0:
                            idx_prob[i]=min_prob/12
                    final_output_prob[iz] = idx_prob
            
    
    print("Finish per sort")
    
    org_prob=[]
    org_pred=[]
    for orgmodel in dirs:
        if "org" in orgmodel:
            model_d = os.path.abspath(os.path.join(model_dir, orgmodel))
            model = AutoModelForSequenceClassification.from_pretrained(model_d)
            model.parameters
            model.to(device)
                    
            model.eval()
            
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
                result = np.argmax(logits, axis=-1)
                    
                org_prob.append(prob)
                org_pred.append(result)
            
            org_prob_concatenate = np.concatenate(org_prob,axis=0).tolist()
            org_pred_concatenate = np.concatenate(org_pred).tolist()
            for iz in range(7765):
                if iz in org_index:
                    idx_prob=[0]*30
                    final_output_pred[iz] = orgs_id_index[org_pred_concatenate[iz]]
                    min_prob = min(org_prob_concatenate[iz])
                    
                    for orgsid in orgs_id_index.keys():
                        idx_prob[orgs_id_index[orgsid]]=org_prob_concatenate[iz][orgsid]
                                                                                 
                    for i in range(30):
                        if idx_prob[i]==0:
                            idx_prob[i]=min_prob/19
                    final_output_prob[iz]=idx_prob
    
    
    print("Finish org sort")        
    return final_output_pred, final_output_prob



def inference_three_step_ensemble(model_dir, Re_test_dataset, device, is_roberta):
    pers_id_index = {0:4,1:6,2:8,3:10,4:11,5:12,6:13,7:14,8:15,9:16,10:17,11:21,12:23,13:24,14:25,15:26,16:27,17:29}
    orgs_id_index = {0:1,1:2,2:3,3:5,4:7,5:9,6:18,7:19,8:20,9:22,10:28}
    
    dataloader = DataLoader(Re_test_dataset, batch_size=16, shuffle=False)
    
    dirs = os.listdir(model_dir)
    dirs = sorted(dirs)
    
    # 최종값 저장할 list
    final_output_prob=[[0]*30]*7765
    final_output_pred=[31]*7765
    
    # 대분류 시, 분류된 각 label을 저장할 lsit
    no_relation_index=[]
    per_index=[]
    org_index=[]
    
    
    ## 대분류 시작 ##
    fold_big_prob=[]
    fold_big_pred=[]
    for bigmodel in dirs:
        
        # big sort 관련 모델을 불러옵니다! (대분류 시에 모델 이름이 big이라는 단어가 들어가야 수행 가능!)
        if "big" in bigmodel:
            print(bigmodel)
            model_d = os.path.abspath(os.path.join(model_dir, bigmodel))
            model = AutoModelForSequenceClassification.from_pretrained(model_d)
            model.parameters
            model.to(device)
            
            model.eval()
            big_prob=[]
            big_pred=[]
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
                result = np.argmax(logits, axis=-1)
                
                big_pred.extend(logits.tolist())
                big_prob.append(prob)
            
            fold_big_pred.append(big_pred)
            fold_big_prob.append(np.concatenate(big_prob, axis=0).tolist())
    
    fold_big_pred = np.mean(fold_big_pred,axis=0)
    fold_big_pred = np.argmax(fold_big_pred,axis=-1)
    print(len(fold_big_pred))
    fold_big_prob = np.mean(fold_big_prob,axis=0).tolist()
    print(len(fold_big_prob))
    per_index=[]
    org_index=[]
    ## no_relation, per, org 분류 저장
    for idx in range(7765):
        idx_prob=[]
        if fold_big_pred[idx]==0:
            final_output_pred[idx]=0
            k=fold_big_prob[idx][0]
            idx_prob.append(k)
            ## 제출 후 작성한 코드 ## -> 좀 더 prob를 정교하게 주기 위해서!
            perprob=(fold_big_prob[idx][1])/18
            orgprob=(fold_big_prob[idx][2])/11
            for c in range(1,30):
                if c in pers_id_index.values():
                    idx_prob.append(perprob)
                else:
                    idx_prob.append(orgprob)
            final_output_prob[idx]=idx_prob
            ##############################################################
            
            
        if fold_big_pred[idx]==1:
            per_index.append(idx)
        
        if fold_big_pred[idx]==2:
            org_index.append(idx)
    
    print("Finish big sort")
            
    ## per관련 label 시작 ##
    
    
    fold_per_prob=[]
    fold_per_pred=[]
    for permodel in dirs:
        if "per" in permodel:
            print(permodel)
            model_d = os.path.abspath(os.path.join(model_dir, permodel))
            model = AutoModelForSequenceClassification.from_pretrained(model_d)
            model.parameters
            model.to(device)
                    
            model.eval()
            per_prob=[]
            per_pred=[]
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
                result = np.argmax(logits, axis=-1)
                
                per_pred.extend(logits.tolist())
                per_prob.append(prob)
            
            fold_per_pred.append(per_pred)
            fold_per_prob.append(np.concatenate(per_prob, axis=0).tolist())
    
    fold_per_pred = np.mean(fold_per_pred,axis=0)
    fold_per_pred = np.argmax(fold_per_pred,axis=-1)
    print(len(fold_per_pred))
    fold_per_prob = np.mean(fold_per_prob,axis=0).tolist()
    print(per_index[0:10])
    for iz in range(7765):
        if iz in per_index:
            idx_prob=[0]*30
            final_output_pred[iz] = pers_id_index[fold_per_pred[iz]]
            min_prob = min(fold_per_prob[iz])
            for persid in pers_id_index.keys():
                idx_prob[pers_id_index[persid]]=fold_per_prob[iz][persid]
            
            for i in range(30):
                if idx_prob[i]==0:
                    idx_prob[i]=min_prob/12
            final_output_prob[iz]=idx_prob
            
    
    print("Finish per sort")
    
    fold_org_prob=[]
    fold_org_pred=[]
    for orgmodel in dirs:
        if "org" in orgmodel:
            print(orgmodel)
            model_d = os.path.abspath(os.path.join(model_dir, orgmodel))
            model = AutoModelForSequenceClassification.from_pretrained(model_d)
            model.parameters
            model.to(device)
                    
            model.eval()
            org_prob=[]
            org_pred=[]
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
                result = np.argmax(logits, axis=-1)
                    
                org_pred.extend(logits.tolist())
                org_prob.append(prob)
            
            fold_org_pred.append(org_pred)
            fold_org_prob.append(np.concatenate(org_prob, axis=0).tolist())
    
    fold_org_pred = np.mean(fold_org_pred,axis=0)
    fold_org_pred = np.argmax(fold_org_pred,axis=-1)
    print(len(fold_org_pred))
    fold_org_prob = np.mean(fold_org_prob,axis=0).tolist()
            
    print(org_index[0:10])
    for iz in range(7765):
        if iz in org_index:
            idx_prob=[0]*30
            final_output_pred[iz] = orgs_id_index[fold_org_pred[iz]]
            min_prob = min(fold_org_prob[iz])
            for orgsid in orgs_id_index.keys():
                idx_prob[orgs_id_index[orgsid]]=fold_org_prob[iz][orgsid]
            
            for i in range(30):
                if idx_prob[i]==0:
                    idx_prob[i]=min_prob/19
            final_output_prob[iz]=idx_prob
    
    
    print("Finish org sort")        
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


def load_test_dataset(dataset_dir, tokenizer, entity_flag, preprocessing_flag, mecab_flag):
    """
      test dataset을 불러온 후,
      tokenizing 합니다.
    """
    test_dataset = load_data(dataset_dir, entity_flag,
                             preprocessing_flag, mecab_flag)
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
#     # load tokenizer
    Tokenizer_NAME = args.PLM
    tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)

#     # load my model
    model_dir = select_checkpoint(args)
#     model = AutoModelForSequenceClassification.from_pretrained(model_dir)
#     model.parameters
#     model.to(device)

    # load test datset
    test_dataset_dir = "/opt/ml/dataset/test/test_data.csv"

    if Tokenizer_NAME in ['klue/roberta-base', 'klue/roberta-small', 'klue/roberta-large']:
        is_roberta = True
    else:
        is_roberta = False

    test_id, test_dataset, test_label = load_test_dataset(
        test_dataset_dir, tokenizer, args.entity_flag, args.preprocessing_flag, args.mecab_flag)
    Re_test_dataset = RE_Dataset(test_dataset, test_label)
    
    if args.model_type and args.k_fold==0:
        # big sort -> per sort-> org sort 순서대로 진행(같은 폴더 안에 3개의 모델이 들어가 있어야 한다!)
        pred_answer, output_prob = inference_three_step(model_dir, Re_test_dataset, device, is_roberta)
        pred_answer = num_to_label(pred_answer)
    
    elif args.model_type and args.k_fold:
        pred_answer, output_prob = inference_three_step_ensemble(model_dir, Re_test_dataset, device, is_roberta)
        pred_answer = num_to_label(pred_answer)
        
    elif args.k_fold:
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

    # make csv file with predicted answer
    #########################################################
    # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
    output = pd.DataFrame(
        {'id': test_id, 'pred_label': pred_answer, 'probs': output_prob, })

    sub_name = model_dir.split('/')[-1]
    # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
    output.to_csv(f"/opt/ml/bigsmall/prediction/submission_{sub_name}.csv", index=False)
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
    parser.add_argument(
        '--preprocessing_flag', default=False, action='store_true', help='Train에 사용했던거랑 똑같이 (default: False)')
    parser.add_argument(
        '--mecab_flag', default=False, action='store_true', help='Train에 사용했던거랑 똑같이 (default: False)')
    
    parser.add_argument("--k_fold", type=int, default=0, help='not k fold(defalut: 0)')
    
    ### model type (대분류/소분류 추가) ###
    parser.add_argument(
        '--model_type', default=False, action='store_true', help='(default: False - 대분류/소분류 X)')
    
    args = parser.parse_args()
    print(args)

    os.makedirs('./prediction', exist_ok=True)
    main(args)
