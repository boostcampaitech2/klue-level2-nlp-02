# -*- coding: utf-8 -*-
import pickle as pickle
import os
import pandas as pd
import torch.nn as nn
import torch
import torch.nn.functional as F
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from loss import use_criterion
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer, DataCollatorWithPadding
from load_data import *

import argparse
from pathlib import Path

import random
import wandb
from dotenv import load_dotenv

dictionary_big_sort = {'no_relation':0, "per":1, "org":2}
dictionary_per_sort = {'per:title': 0, 'per:employee_of': 1,  'per:product': 2, 
                       'per:children': 3, 'per:place_of_residence': 4, 
                       'per:alternate_names': 5, 'per:other_family': 6, 'per:colleagues': 7, 
                       'per:origin': 8, 'per:siblings': 9, 'per:spouse': 10, 
                       'per:parents': 11,  'per:schools_attended': 12, 
                       'per:date_of_death': 13, 'per:date_of_birth': 14, 'per:place_of_birth': 15, 
                       'per:place_of_death': 16,  'per:religion': 17}
dictionary_org_sort = {'org:top_members/employees': 0, 'org:members': 1, 'org:product': 2,
                       'org:alternate_names': 3, 'org:place_of_headquarters': 4, 'org:number_of_employees/members': 5, 'org:founded': 6,
                       'org:political/religious_affiliation': 7, 'org:member_of': 8, 'org:dissolved': 9, 'org:founded_by': 10}

def klue_re_micro_f1(preds, labels, model_type):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = ['no_relation', 'org:top_members/employees', 'org:members',
                  'org:product', 'per:title', 'org:alternate_names',
                  'per:employee_of', 'org:place_of_headquarters', 'per:product',
                  'org:number_of_employees/members', 'per:children',
                  'per:place_of_residence', 'per:alternate_names',
                  'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
                  'per:spouse', 'org:founded', 'org:political/religious_affiliation',
                  'org:member_of', 'per:parents', 'org:dissolved',
                  'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
                  'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
                  'per:religion']
    
    if model_type=="big_sort":
        label_list=list(dictionary_big_sort.keys())
        no_relation_label_idx = label_list.index("no_relation")
        label_indices = list(range(len(label_list)))
        label_indices.remove(no_relation_label_idx)
        
    elif model_type=="per_sort":
        label_list=list(dictionary_per_sort.keys())
        label_indices = list(range(len(label_list)))
        
    elif model_type=="org_sort":
        label_list=list(dictionary_org_sort.keys())
        label_indices = list(range(len(label_list)))
        
    else:
        no_relation_label_idx = label_list.index("no_relation")
        label_indices = list(range(len(label_list)))
        label_indices.remove(no_relation_label_idx)
        
    # no_relation_label_idx = label_list.index("no_relation")
    # label_indices = list(range(len(label_list)))
    # label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0


def klue_re_auprc(probs, labels, model_type):
    """KLUE-RE AUPRC (with no_relation)"""
    k=30
    if model_type=="big_sort":
        k=3
    elif model_type=="per_sort":
        k=18
    elif model_type=="org_sort":
        k=11
    
    labels = np.eye(k)[labels]

    score = np.zeros((k,))
    for c in range(k):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(
            targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0


def compute_metrics(pred):
    """ validation을 위한 metrics function """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    print(len(set(preds)))
    probs = pred.predictions

    # calculate accuracy using sklearn's function
    f1 = klue_re_micro_f1(preds, labels, args.model_type)
    auprc = klue_re_auprc(probs, labels, args.model_type)
    acc = accuracy_score(labels, preds)  # 리더보드 평가에는 포함되지 않습니다.

    return {
        'micro f1 score': f1,
        'auprc': auprc,
        'accuracy': acc,
    }


def label_to_num(label, model_type="per_sort"):
    num_label = []
    with open('dict_label_to_num.pkl', 'rb') as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        
        if model_type=="default":
            num_label.append(dict_label_to_num[v])
        
        elif model_type=="big_sort": # no_relation: 0, per: 1, org: 2
            if "per" in v:
                v="per"
            if "org" in v:
                v="org"
            num_label.append(dictionary_big_sort[v])
        
        elif model_type=="per_sort":
            num_label.append(dictionary_per_sort[v])
            
        elif model_type=="org_sort":
            num_label.append(dictionary_org_sort[v])

    return num_label


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



#https://huggingface.co/transformers/main_classes/trainer.html
class MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels") # "labels" in inputs 기능으로 label 가져오기!
        outputs = model(**inputs) # input으로 모델이 예측!
        logits = outputs.logits # model 결과값
        loss_fn = use_criterion(args.criterion)
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def train(args):
    # load model and tokenizer
    MODEL_NAME = args.PLM
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # dynamic padding
    dynamic_padding = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataset = load_data("/opt/ml/dataset/train/train.csv",
                              args.entity_flag, args.preprocessing_flag, args.mecab_flag, args.model_type)
    
    train_label = label_to_num(train_dataset['label'].values, args.model_type)
    
    if args.k_fold:
        skf = StratifiedKFold(n_splits=args.k_fold, shuffle=True)
        
        for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(train_dataset,train_label),1):
            train_lists, train_labels = train_dataset.loc[train_idx], list(np.array(train_label)[train_idx])
            valid_lists, valid_labels = train_dataset.loc[valid_idx], list(np.array(train_label)[valid_idx])
            
            tokenized_train = tokenized_dataset(train_lists, tokenizer)  # UNK token count
            tokenized_valid = tokenized_dataset(valid_lists, tokenizer)  # UNK token count
            RE_train_dataset = RE_Dataset(tokenized_train, train_labels, args.seed)
            RE_dev_dataset = RE_Dataset(tokenized_valid, valid_labels, args.eval_ratio, args.seed)
            
            load_dotenv(dotenv_path=args.dotenv_path)
            WANDB_AUTH_KEY = os.getenv('WANDB_AUTH_KEY')
            wandb.login(key=WANDB_AUTH_KEY)
            
            wandb.init(
                entity="klue-level2-nlp-02",
                project="Relation-Extraction",
                name=args.wandb_unique_tag+"_"+str(fold_idx),
                group=args.PLM+"-k_fold")
            
            wandb.config.update(args)
            train_model(args=args,RE_train_dataset=RE_train_dataset, RE_dev_dataset=RE_dev_dataset, fold_idx=fold_idx, dynamic_padding=dynamic_padding, tokenizer=tokenizer, model_type=args.model_type)
            wandb.finish()
            
    else:
        # tokenizing dataset
        tokenized_train = tokenized_dataset(train_dataset, tokenizer)  # UNK token count
        
        RE_train_dataset = RE_Dataset(tokenized_train, train_label, args.eval_ratio, args.seed)
        
        train_model(args, RE_train_dataset, RE_dev_dataset=0, fold_idx=0, dynamic_padding=dynamic_padding, tokenizer=tokenizer, model_type=args.model_type)


def train_model(args,RE_train_dataset,RE_dev_dataset,fold_idx,dynamic_padding,tokenizer,model_type):
    
    # Split validation dataset
    if args.eval_flag == True and args.k_fold==0:
        RE_train_dataset, RE_dev_dataset = RE_train_dataset.split()

    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print(device)
    # setting model hyperparameter
    model_config = AutoConfig.from_pretrained(args.PLM)
    if model_type=="big_sort":
        model_config.num_labels=3
    elif model_type=="per_sort":
        model_config.num_labels=18
    elif model_type=="org_sort":
        model_config.num_labels=11
    else:
        model_config.num_labels = 30

    model = AutoModelForSequenceClassification.from_pretrained(
        args.PLM, ignore_mismatched_sizes=args.ignore_mismatched, config=model_config)
    print(model.config)
    model.parameters
    model.to(device)
    
    if args.eval_flag == True or args.k_fold:
        training_args = TrainingArguments(
            output_dir='./results',          # output directory
            save_total_limit=5,              # number of total save model.
            save_steps=500,                  # model saving step.
            num_train_epochs=args.epochs,              # total number of training epochs
            learning_rate=args.lr,                     # learning_rate
            # batch size per device during training
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,    # batch size for evaluation
            # number of warmup steps for learning rate scheduler
            warmup_steps=args.warmup_steps,
            weight_decay=args.weight_decay,                # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
            logging_steps=100,               # log saving step.
            # evaluation strategy to adopt during training
            evaluation_strategy=args.evaluation_strategy,
            # `no`: No evaluation during training.
            # `steps`: Evaluate every `eval_steps`.
            # `epoch`: Evaluate every end of epoch.
            eval_steps=500,           # evaluation step.
            load_best_model_at_end=True,
            report_to="wandb"
        )
        
        if args.criterion=='default':
            trainer = Trainer(
                # the instantiated 🤗 Transformers model to be trained
                model=model,
                args=training_args,                  # training arguments, defined above
                train_dataset=RE_train_dataset,         # training dataset
                eval_dataset=RE_dev_dataset,             # evaluation dataset
                compute_metrics=compute_metrics,         # define metrics function
                data_collator=dynamic_padding,
                tokenizer=tokenizer,
            )
        else:
            trainer = MultilabelTrainer(
                # the instantiated 🤗 Transformers model to be trained
                model=model,
                args=training_args,                  # training arguments, defined above
                train_dataset=RE_train_dataset,         # training dataset
                eval_dataset=RE_dev_dataset,             # evaluation dataset
                compute_metrics=compute_metrics,         # define metrics function
                data_collator=dynamic_padding,
                tokenizer=tokenizer,
            )

    else:
        training_args = TrainingArguments(
            output_dir='./results',          # output directory
            save_total_limit=5,              # number of total save model.
            save_steps=500,                  # model saving step.
            num_train_epochs=args.epochs,              # total number of training epochs
            learning_rate=args.lr,                     # learning_rate
            # batch size per device during training
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,    # batch size for evaluation
            # number of warmup steps for learning rate scheduler
            warmup_steps=args.warmup_steps,
            weight_decay=args.weight_decay,                # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
            logging_steps=100,               # log saving step.
            # evaluation strategy to adopt during training
            evaluation_strategy=args.evaluation_strategy,
            # `no`: No evaluation during training.
            # `steps`: Evaluate every `eval_steps`.
            # `epoch`: Evaluate every end of epoch.
            eval_steps=500,           # evaluation step.
            load_best_model_at_end=True,
            report_to="wandb"
        )
        
        if args.criterion=='default':
            trainer = Trainer(
                # the instantiated 🤗 Transformers model to be trained
                model=model,
                args=training_args,                  # training arguments, defined above
                train_dataset=RE_train_dataset,         # training dataset
                eval_dataset=RE_train_dataset,             # evaluation dataset
                compute_metrics=compute_metrics,         # define metrics function
                data_collator=dynamic_padding,
                tokenizer=tokenizer,
            )
        
        else:
            trainer = MultilabelTrainer(
                # the instantiated 🤗 Transformers model to be trained
                model=model,
                args=training_args,                  # training arguments, defined above
                train_dataset=RE_train_dataset,         # training dataset
                eval_dataset=RE_train_dataset,             # evaluation dataset
                compute_metrics=compute_metrics,         # define metrics function
                data_collator=dynamic_padding,
                tokenizer=tokenizer,
            )

    # train model
    trainer.train()
    
    if args.k_fold:
        if model_type!="default":
            folder_name = re.sub("orgsort|persort|bigsort","",args.wandb_unique_tag)
            model_save_pth = os.path.join(args.save_dir, args.PLM.replace('/', '-') + '-' + folder_name.replace('/', '-') + "/" + model_type + "_" + str(fold_idx))
        else:
            model_save_pth = os.path.join(args.save_dir, args.PLM.replace('/', '-') + '-' + args.wandb_unique_tag.replace('/', '-') + "/" + str(fold_idx))
        os.makedirs(model_save_pth, exist_ok=True)
        model.save_pretrained(model_save_pth)
    
    elif model_type!="default":
        folder_name = re.sub("orgsort|persort|bigsort","",args.wandb_unique_tag)
        model_save_pth = os.path.join(args.save_dir, args.PLM.replace('/', '-') + '-' + folder_name.replace('/', '-') + "/" + model_type)
        os.makedirs(model_save_pth, exist_ok=True)
        model.save_pretrained(model_save_pth)
        
        
    else:
        model_save_pth = os.path.join(args.save_dir, args.PLM.replace('/', '-') + '-' + args.wandb_unique_tag.replace('/', '-'))
        os.makedirs(model_save_pth, exist_ok=True)
        model.save_pretrained(model_save_pth)



def main(args):
    if args.k_fold==0:
        load_dotenv(dotenv_path=args.dotenv_path)
        WANDB_AUTH_KEY = os.getenv('WANDB_AUTH_KEY')
        wandb.login(key=WANDB_AUTH_KEY)

        wandb.init(
            entity="klue-level2-nlp-02",
            project="Relation-Extraction_1001",
            name=args.wandb_unique_tag,
            group=args.PLM)
        wandb.config.update(args)
        train(args)
        wandb.finish()
    else:
        train(args)


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--save_dir', default='./best_models',
                        help='model save at save_dir/PLM-wandb_unique_tag')
    parser.add_argument('--PLM', type=str, default='klue/bert-base',
                        help='model type (default: klue/bert-base)')
    parser.add_argument('--epochs', type=int, default=3,
                        help='number of epochs to train (default: 3)')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='learning rate (default: 5e-5)')
    parser.add_argument('--train_batch_size', type=int,
                        default=16, help='train batch size (default: 16)')
    parser.add_argument('--warmup_steps', type=int, default=500,
                        help='number of warmup steps for learning rate scheduler (default: 500)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='strength of weight decay (default: 0.01)')
    parser.add_argument('--evaluation_strategy', type=str, default='steps',
                        help='evaluation strategy to adopt during training, steps or epoch (default: steps)')
    parser.add_argument('--ignore_mismatched', type=bool, default=False,
                        help='ignore mismatched size when load pretrained model')

    # Validation
    parser.add_argument('--eval_flag', default=True, action='store_true',
                        help='eval flag (default: True)')
    parser.add_argument('--eval_ratio', type=float, default=0.2,
                        help='eval data size ratio (default: 0.2)')
    parser.add_argument('--eval_batch_size', type=int,
                        default=16, help='eval batch size (default: 16)')

    # Seed
    parser.add_argument('--seed', type=int, default=2,
                        help='random seed (default: 2)')

    # Wandb
    parser.add_argument(
        '--dotenv_path', default='/opt/ml/wandb.env', help='input your dotenv path')
    parser.add_argument('--wandb_unique_tag', default='bert-base-high-lr',
                        help='input your wandb unique tag (default: bert-base-high-lr)')

    # Running mode
    parser.add_argument('--entity_flag', default=False, action='store_true',
                        help='add Entity flag (default: False)')
    parser.add_argument('--preprocessing_flag', default=False, action='store_true',
                        help='input text pre-processing (default: False)')
    parser.add_argument('--mecab_flag', default=False, action='store_true',
                        help='input text pre-processing (default: False)')
    
    parser.add_argument("--k_fold", type=int, default=0, help='not k fold(defalut: 0)')
    parser.add_argument("--criterion", type=str, default='default', help='criterion type: label_smoothing, focal_loss')
    
    ## 주의사항 ##
    ## model_type 사용 시, wandb_unique_tag는 use_prepro_entity_mecab_orgsort, use_prepro_entity_mecab_persort 등으로 끝부분 제외하고 통일시켜주시면 감사합니다!
    ## 이유는 같은 폴더 안에 넣기 위함입니다!
    parser.add_argument("--model_type", type=str, default='default', help='criterion type: big_sort, per_sort, org_sort')
    
    args = parser.parse_args()

    # Start
    seed_everything(args.seed)

    main(args)
