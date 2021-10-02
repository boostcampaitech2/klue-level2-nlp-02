#!/bin/bash

python train.py --PLM klue/roberta-large --eval_ratio 0.1 --wandb_unique_tag raw_data
python train.py --PLM klue/roberta-large --eval_ratio 0.1 --wandb_unique_tag use_prepro --preprocessing_flag
python train.py --PLM klue/roberta-large --eval_ratio 0.1 --wandb_unique_tag use_entity --entity_flag
python train.py --PLM klue/roberta-large --eval_ratio 0.1 --wandb_unique_tag use_mecab --mecab_flag
python train.py --PLM klue/roberta-large --eval_ratio 0.1 --wandb_unique_tag use_prepro_entity --preprocessing_flag --entity_flag
python train.py --PLM klue/roberta-large --eval_ratio 0.1 --wandb_unique_tag use_prepro_mecab --preprocessing_flag --mecab_flag
python train.py --PLM klue/roberta-large --eval_ratio 0.1 --wandb_unique_tag use_entity_mecab --entity_flag --mecab_flag
python train.py --PLM klue/roberta-large --eval_ratio 0.1 --wandb_unique_tag use_prepro_entity_mecab --entity_flag --preprocessing_flag --mecab_flag