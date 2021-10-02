#!/bin/bash

python train.py --PLM klue/roberta-large --eval_ratio 0.1 --wandb_unique_tag use_mecab --mecab_flag
python train.py --PLM klue/roberta-large --eval_ratio 0.1 --wandb_unique_tag use_prepro_mecab --preprocessing_flag --mecab_flag
python train.py --PLM klue/roberta-large --eval_ratio 0.1 --wandb_unique_tag use_prepro_entity_mecab --entity_flag --preprocessing_flag --mecab_flag