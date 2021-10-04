#!/bin/bash

python train_custom.py --PLM klue/roberta-large --eval_ratio 0.1 --wandb_unique_tag custom_model_epoch3 --entity_flag --preprocessing_flag --mecab_flag
python train_custom.py --PLM klue/roberta-large --eval_ratio 0.1 --wandb_unique_tag custom_model_epoch5 --entity_flag --preprocessing_flag --mecab_flag --epochs 5
python train_custom.py --PLM klue/roberta-large --eval_ratio 0.1 --wandb_unique_tag custom_model_epoch5 --entity_flag --preprocessing_flag --mecab_flag --epochs 20