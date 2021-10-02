import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForMaskedLM, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer, DataCollatorWithPadding
from datasets import load_metric
from load_data import *
from torch.utils.data.dataset import random_split

import argparse
from pathlib import Path

import random
import wandb
from dotenv import load_dotenv


def train(args):
    # load model and tokenizer
    MODEL_NAME = args.PLM
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # dynamic padding
    dynamic_padding = DataCollatorWithPadding(tokenizer=tokenizer)

    # load dataset
    train_dataset = load_lm_data("/opt/ml/dataset/train/train.csv")
    
    # tokenizing dataset
    tokenized_train = tokenized_lm_dataset(
        train_dataset, tokenizer)
    LM_train_dataset = LM_Dataset(
        tokenized_train, tokenizer)

    # Split validation dataset
    if args.eval_flag == True:
        val_num = int(len(LM_train_dataset) * args.eval_ratio)
        train_num = len(LM_train_dataset) - val_num
        LM_train_dataset, LM_dev_dataset = random_split(LM_train_dataset, [train_num, val_num])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(device)
    # setting model hyperparameter
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    
    model = AutoModelForMaskedLM.from_pretrained(
        MODEL_NAME, ignore_mismatched_sizes=args.ignore_mismatched, config=model_config)
    print(model.config)
    model.parameters
    model.to(device)

    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.

    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        save_total_limit=5,              # number of total save model.
        save_steps=500,                  # model saving step.
        num_train_epochs=args.epochs,              # total number of training epochs
        learning_rate=args.lr,                     # learning_rate
        # batch size per device during training
        per_device_train_batch_size=args.train_batch_size,
        # number of warmup steps for learning rate scheduler
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,                # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=100,               # log saving step.
        report_to="wandb"
    )
    trainer = Trainer(
        # the instantiated 🤗 Transformers model to be trained
        model=model,
        args=training_args,                  # training arguments, defined above
        train_dataset=LM_train_dataset,         # training dataset
        data_collator=dynamic_padding,
        tokenizer=tokenizer,
    )

    # train model
    trainer.train()
    model_save_pth = os.path.join(args.save_dir, args.PLM.replace(
        '/', '-') + '-' + args.wandb_unique_tag.replace('/', '-'))
    os.makedirs(model_save_pth, exist_ok=True)
    model.save_pretrained(model_save_pth)


def main(args):
    load_dotenv(dotenv_path=args.dotenv_path)
    WANDB_AUTH_KEY = os.getenv('WANDB_AUTH_KEY')
    wandb.login(key=WANDB_AUTH_KEY)

    wandb.init(
        entity="klue-level2-nlp-02",
        project="Relation-Extraction",
        name=args.wandb_unique_tag,
        group='LM_finetuning')
    wandb.config.update(args)

    train(args)
    wandb.finish()


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
    parser.add_argument('--PLM', type=str, default='klue/roberta-large',
                        help='model type (default: klue/roberta-large)')
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
    parser.add_argument('--eval_flag', action='store_false',
                        default=True, help='eval flag (default: True)')
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

    args = parser.parse_args()

    # Start
    seed_everything(args.seed)

    main(args)
