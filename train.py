import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer, DataCollatorWithPadding
from Preprocessing.preprocessor import EntityPreprocessor, SenPreprocessor, UnkPreprocessor
import argparse
from pathlib import Path
import random
import wandb
from dotenv import load_dotenv

from load_data import *
from augmentation import *
from custom_model import *
from tokenization import *

def klue_re_micro_f1(preds, labels):
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
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0


def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
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
    probs = pred.predictions

    # calculate accuracy using sklearn's function
    f1 = klue_re_micro_f1(preds, labels)
    auprc = klue_re_auprc(probs, labels)
    acc = accuracy_score(labels, preds)  # 리더보드 평가에는 포함되지 않습니다.

    return {
        'micro f1 score': f1,
        'auprc': auprc,
        'accuracy': acc,
    }


def label_to_num(label):
    num_label = []
    with open('dict_label_to_num.pkl', 'rb') as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])

    return num_label


def train(args):
    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.PLM)

    # dynamic padding
    dynamic_padding = DataCollatorWithPadding(tokenizer=tokenizer)

    # Preprocessor
    sen_preprocessor = SenPreprocessor(args.preprocessing_cmb, args.mecab_flag)
    unk_preprocessor = UnkPreprocessor(tokenizer)
    entity_preprocessor = EntityPreprocessor(args.entity_flag)

    # load dataset
    datasets = load_data("/opt/ml/dataset/train/train.csv", args.k_fold, args.eval_ratio)
    
    for fold_idx, (train_dataset, test_dataset) in enumerate(datasets):
        
        #agumentation and preprocessing
        aug_data_by_mixing_entity = None
        if args.augmentation_flag is True:
            aug_data_by_mixing_entity = augmentation_by_resampling(train_dataset)
            aug_data_by_mixing_entity = preprocessing_dataset(aug_data_by_mixing_entity, sen_preprocessor, entity_preprocessor)
        train_dataset = preprocessing_dataset(train_dataset, sen_preprocessor, entity_preprocessor)
        aug_data_by_aeda = aeda_dataset(train_dataset) if args.aeda_flag is True else None

        #concatenate augmentation data and train data
        train_dataset = pd.concat([train_dataset, aug_data_by_mixing_entity, aug_data_by_aeda])

        added_token_num = 0
        if args.add_unk_token :
            tokenizer, added_token_num =  unk_preprocessor(list(train_dataset['sentence']),
                                                           list(train_dataset['subject_entity']),
                                                           list(train_dataset['object_entity']))
        #shuffle rows
        train_dataset = train_dataset.sample(frac=1,random_state=args.seed).reset_index(drop=True)

        train_label = label_to_num(train_dataset['label'].values)
        tokenized_train = tokenized_dataset(train_dataset, tokenizer)
        RE_train_dataset = RE_Dataset(tokenized_train, train_label)
      
        if args.r_roberta :
            tokenized_train = r_tokenized_dataset(train_dataset, tokenizer)
            RE_train_dataset = r_RE_Dataset(tokenized_train, train_label, tokenizer) ## r_robert

        #eval set이 없으면 RE_dev_dataset에 RE_train_dataset 복사하여 사용
        if test_dataset is not None:
            test_dataset = preprocessing_dataset(test_dataset, sen_preprocessor, entity_preprocessor)
            test_label = label_to_num(test_dataset['label'].values)
            tokenized_test = tokenized_dataset(test_dataset, tokenizer)
            RE_dev_dataset = RE_Dataset(tokenized_test, test_label)
            
            if args.r_roberta :
                tokenized_test = r_tokenized_dataset(test_dataset, tokenizer)
                RE_dev_dataset = r_RE_Dataset(tokenized_test, test_label, tokenizer) ## r_robert
        else:
            RE_dev_dataset = RE_train_dataset
        
        #wandb
        load_dotenv(dotenv_path=args.dotenv_path)
        WANDB_AUTH_KEY = os.getenv('WANDB_AUTH_KEY')
        wandb.login(key=WANDB_AUTH_KEY)
        
        wandb.init(
            entity="klue-level2-nlp-02",
            project="Relation-Extraction_1001",
            name=args.wandb_unique_tag+"_"+str(fold_idx),
            group=args.PLM+"-k_fold" if args.k_fold > 0 else args.PLM)
        wandb.config.update(args)

        train_model(args, RE_train_dataset, RE_dev_dataset, fold_idx=0, dynamic_padding=dynamic_padding,
                     tokenizer=tokenizer, added_token_num=added_token_num)

        wandb.finish()
        
    
def train_model(
    args,
    RE_train_dataset,
    RE_dev_dataset,
    fold_idx,
    dynamic_padding,
    tokenizer,
    added_token_num):
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # setting model hyperparameter
    model_config = AutoConfig.from_pretrained(args.PLM)
    model_config.num_labels = 30

    if args.use_mlm:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.MLM_checkpoint, config=model_config)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.PLM, ignore_mismatched_sizes=args.ignore_mismatched, config=model_config)
    
    if args.use_mlm and args.r_roberta : 
        model = r_roberta(args.MLM_checkpoint, model_config, args.dropout_rate)
    elif not args.use_mlm and args.r_roberta : 
        model = r_roberta(args.PLM, model_config, args.dropout_rate)
    
    if args.use_mlm and args.add_unk_token :
        model.resize_token_embeddings(tokenizer.vocab_size + added_token_num)

    print(model.config)
    model.parameters
    model.to(device)

    training_args = TrainingArguments(
            output_dir='./results',          # output directory
            save_total_limit=5,              # number of total save model.
            save_steps=500,                  # model saving step.
            num_train_epochs=args.epochs,              # total number of training epochs
            learning_rate=args.lr,                     # learning_rate
            per_device_train_batch_size=args.train_batch_size, # batch size per device during training
            per_device_eval_batch_size=args.eval_batch_size,    # batch size for evaluation
            warmup_steps=args.warmup_steps, # number of warmup steps for learning rate scheduler
            weight_decay=args.weight_decay,                # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
            logging_steps=100,               # log saving step.
            evaluation_strategy=args.evaluation_strategy, # evaluation strategy to adopt during training
            # `no`: No evaluation during training.
            # `steps`: Evaluate every `eval_steps`.
            # `epoch`: Evaluate every end of epoch.
            eval_steps=500,           # evaluation step.
            load_best_model_at_end=True,
            report_to="wandb"
        )
    trainer = Trainer(
        # the instantiated 🤗 Transformers model to be trained
        model=model,
        args=training_args,                  # training arguments, defined above
        train_dataset=RE_train_dataset,         # training dataset
        eval_dataset=RE_dev_dataset, # evaluation dataset
        compute_metrics=compute_metrics,         # define metrics function
        data_collator=dynamic_padding if args.r_roberta == False else None,
        tokenizer=tokenizer)


    # train model
    trainer.train()
    
    if args.k_fold:
        model_save_pth = os.path.join(args.save_dir, args.PLM.replace(
        '/', '-') + '-' + args.wandb_unique_tag.replace('/', '-') + "/" + str(fold_idx))
        os.makedirs(model_save_pth, exist_ok=True)
        model.save_pretrained(model_save_pth)
        if args.r_roberta :
            torch.save(model.state_dict(), os.path.join(
                model_save_pth, 'pytorch_model.pt'))
        if args.add_unk_token :
            tokenizer.save_pretrained(model_save_pth+'/tokenizer')
    
    else:
        model_save_pth = os.path.join(args.save_dir, args.PLM.replace(
        '/', '-') + '-' + args.wandb_unique_tag.replace('/', '-'))
        os.makedirs(model_save_pth, exist_ok=True)
        model.save_pretrained(model_save_pth)
        if args.r_roberta :
            torch.save(model.state_dict(), os.path.join(
                model_save_pth, 'pytorch_model.pt'))

        if args.add_unk_token :
            tokenizer.save_pretrained(model_save_pth+'/tokenizer')


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
    parser.add_argument('--MLM_checkpoint', type=str, default='./best_models/klue-roberta-large-pem-mlm',
                        help='MaskedLM pretrained model path')
    parser.add_argument('--use_mlm', default=False, action='store_true',
                        help='whether or not use MaskedLM pretrained model')
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
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                        help='dropout_rate (default: 0.1)')

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
    parser.add_argument('--dotenv_path', default='/opt/ml/wandb.env',
                        help='input your dotenv path')
    parser.add_argument('--wandb_unique_tag', default='bert-base-high-lr',
                        help='input your wandb unique tag (default: bert-base-high-lr)')

    # Running mode
    parser.add_argument('--entity_flag', default=False, action='store_true',
                        help='add Entity flag (default: False)')
    
    parser.add_argument('--preprocessing_cmb', nargs='+',
                        help='<Required> Set flag (example: 0 1 2)')

    parser.add_argument('--mecab_flag', default=False, action='store_true',
                        help='input text pre-processing (default: False)')
    
    parser.add_argument('--add_unk_token', default=False, action='store_true',
                        help='add unknown token in vocab (default: False)')
    
    parser.add_argument("--k_fold", type=int, default=0,
                        help='not k fold(defalut: 0)')

    parser.add_argument('--aeda_flag', type=bool, default=False,
                        help='Number of adea agmentations (default: 0)')

    parser.add_argument('--augmentation_flag', type=bool, default=False,
                        help="data augmentation by resampling")

    parser.add_argument('--r_roberta', default=False, action='store_true',
                        help=' custom model to r_roberta(default: False)')
    
    args = parser.parse_args()

    # Start
    seed_everything(args.seed)

    # main(args)
    train(args)