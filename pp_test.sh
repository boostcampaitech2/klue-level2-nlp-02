#! /bin/bash

python train.py --PLM klue/roberta-large --eval_ratio 0.1 --wandb_unique_tag base

python train.py --PLM klue/roberta-large --eval_ratio 0.1 --wandb_unique_tag pp0_AUT --preprocessing_cmb 0 --add_unk_token
python train.py --PLM klue/roberta-large --eval_ratio 0.1 --wandb_unique_tag pp1_AUT --preprocessing_cmb 1 --add_unk_token
python train.py --PLM klue/roberta-large --eval_ratio 0.1 --wandb_unique_tag pp2_AUT --preprocessing_cmb 2 --add_unk_token
python train.py --PLM klue/roberta-large --eval_ratio 0.1 --wandb_unique_tag pp01_AUT --preprocessing_cmb 0 1 --add_unk_token
python train.py --PLM klue/roberta-large --eval_ratio 0.1 --wandb_unique_tag pp02_AUT --preprocessing_cmb 0 2 --add_unk_token
python train.py --PLM klue/roberta-large --eval_ratio 0.1 --wandb_unique_tag pp12_AUT --preprocessing_cmb 1 2 --add_unk_token
python train.py --PLM klue/roberta-large --eval_ratio 0.1 --wandb_unique_tag pp012_AUT --preprocessing_cmb 0 1 2 --add_unk_token

python train.py --PLM klue/roberta-large --eval_ratio 0.1 --wandb_unique_tag pp0 --preprocessing_cmb 0 
python train.py --PLM klue/roberta-large --eval_ratio 0.1 --wandb_unique_tag pp1 --preprocessing_cmb 1
python train.py --PLM klue/roberta-large --eval_ratio 0.1 --wandb_unique_tag pp2 --preprocessing_cmb 2
python train.py --PLM klue/roberta-large --eval_ratio 0.1 --wandb_unique_tag pp01 --preprocessing_cmb 0 1
python train.py --PLM klue/roberta-large --eval_ratio 0.1 --wandb_unique_tag pp02 --preprocessing_cmb 0 2
python train.py --PLM klue/roberta-large --eval_ratio 0.1 --wandb_unique_tag pp12 --preprocessing_cmb 1 2
python train.py --PLM klue/roberta-large --eval_ratio 0.1 --wandb_unique_tag pp012 --preprocessing_cmb 0 1 2


