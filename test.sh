# python train.py --PLM klue/roberta-large --eval_ratio 0.1 --wandb_unique_tag klue/roberta-large-normal
# python train.py --PLM klue/roberta-large --eval_ratio 0.1 --wandb_unique_tag klue/roberta-large-augmen --augmentation_flag True
python train.py --PLM klue/roberta-large --eval_ratio 0.1 --wandb_unique_tag klue/roberta-large-augmen --augmentation_flag True --epoch 7
python train.py --PLM klue/roberta-large --eval_ratio 0.1 --wandb_unique_tag klue/roberta-large-augmen --augmentation_flag True --epcoh 10
python train.py --PLM klue/roberta-large --eval_ratio 0.1 --wandb_unique_tag klue/roberta-large-augmen --augmentation_flag True --entity_flag True --preprocessing_cmb 1 2 3 --mecab_flag True --epoch 5
python train.py --PLM klue/roberta-large --eval_ratio 0.1 --wandb_unique_tag klue/roberta-large-augmen --augmentation_flag True --entity_flag True --preprocessing_cmb 1 2 3 --mecab_flag True --epcoh 7
python train.py --PLM klue/roberta-large --eval_ratio 0.1 --wandb_unique_tag klue/roberta-large-augmen --augmentation_flag True --entity_flag True --preprocessing_cmb 1 2 3 --mecab_flag True --epcoh 10
python train.py --PLM klue/roberta-large --eval_ratio 0.1 --wandb_unique_tag klue/roberta-large-augmen --augmentation_flag True --entity_flag True --preprocessing_cmb 1 2 3 --mecab_flag True --k_fold 5
