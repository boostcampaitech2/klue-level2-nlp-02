# python train.py --PLM klue/roberta-large --eval_ratio 0.1 --wandb_unique_tag klue/roberta-large-augmen --augmentation_flag True
#epoch 횟수 변경
# python train.py --PLM klue/roberta-large --eval_ratio 0.1 --wandb_unique_tag klue/roberta-large-augmen-epoch1 --augmentation_flag True --epochs 1 
# python train.py --PLM klue/roberta-large --eval_ratio 0.1 --wandb_unique_tag klue/roberta-large-augmen-epcoh2 --augmentation_flag True --epochs 2 

# #val 비율 변경
# python train.py --PLM klue/roberta-large --eval_ratio 0.2 --wandb_unique_tag klue/roberta-large-augmen-val0.2 --augmentation_flag True 
# python train.py --PLM klue/roberta-large --eval_ratio 0.3 --wandb_unique_tag klue/roberta-large-augmen-val0.3 --augmentation_flag True

#augmentation_flag True, entity_flag True,  preprocessing_cmb
python train.py --PLM klue/roberta-large --eval_ratio 0.1 --wandb_unique_tag klue/roberta-large-augmen-cmbs --augmentation_flag True --entity_flag --preprocessing_cmb 0 1 2 --mecab_flag 
python train.py --PLM klue/roberta-large --eval_ratio 0.2 --wandb_unique_tag klue/roberta-large-augmen-eval0.2 --augmentation_flag True --entity_flag  --preprocessing_cmb 1 2 3 --mecab_flag 

#kfold 
python train.py --PLM klue/roberta-large --eval_ratio 0.1 --wandb_unique_tag klue/roberta-large-augmen-cmbd-fkolds --augmentation_flag True --entity_flag  --preprocessing_cmb 1 2 3 --mecab_flag  --k_fold 5
