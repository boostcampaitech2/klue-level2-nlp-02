# Pstage_02_KLUE

Solution for KLUE Competitions in 2nd BoostCamp AI Tech 2기 by **메타몽팀 (2조)**

## Content
- [Competition Abstract](#competition-abstract)
- [Hardware](#hardware)
- [Archive Contents](#archive-contents)
- [Getting Started](#getting-started)
  * [Dependencies](#dependencies)
  * [Install Requirements](#install-requirements)
  * [Training](#training)
  * [Inference](#inference)
  * [Soft-voting Ensemble](#soft-voting-ensemble)
- [Architecture](#architecture)
- [Result](#result)

## Competition Abstract

- 문장과 subject_entity, object_entity가 주어졌을 때, 해당 문장 내의 subject_entity와 object_entity 사이의 관계를 예측하는 Relation Extraction task
- 데이터셋 통계:
    - train.csv: 총 32470개
    - test_data.csv: 총 7765개 (정답 라벨 blind = 100으로 임의 표현)
- 데이터 예시:

![DataExample](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/eebd028b-24f4-44ce-a784-e0d3bc0d3bc4/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20211008%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20211008T042928Z&X-Amz-Expires=86400&X-Amz-Signature=acf0146dcde629bb186e4f527c1488ae9f6d432e4965281769a3978e5ba6a378&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

## Hardware

- Ubuntu 18.04.5 LTS
- Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz
- NVIDIA Tesla V100-SXM2-32GB

## Archive Contents

- klue-level2-nlp-02 : 구현 코드와 모델 checkpoint 및 모델 결과를 포함하는 디렉토리

```
klue-level2-nlp-02/
├── best_models/
├── results/
├── prediction/
│   └── all/
├── modules/
│   ├── preprocesor.py
│   ├── augmentation.py
│   ├── loss.py
│   ├── make_rtt_csv.py
│   ├── concat_csv.py
│   ├── UNK_token_text_search.ipynb
│   └── crawling_papago_rtt.ipynb
├── dict_label_to_num.pkl
├── dict_num_to_label.pkl
├── train.py
├── train_mlm.py
├── load_data.py
├── tokenization.py
├── model.py
├── model_ensemble.py
└── inference.py
```

- `best_models/` : train.py/train_mlm.py 실행 후 모델의 loss가 가장 낮은 checkpoint가 저장되는 디렉토리
- `results/` : train.py/train_mlm.py 실행 중 모델의 checkpoint가 임시로 저장되는 디렉토리
- `prediction/` : inference.py 실행 후 모델 예측 결과 csv가 저장되는 디렉토리
    - `all/` : model_ensemble.py 실행 시에 args.dir=='all'인 경우 불러오는 csv 파일들이 저장되어있는 디렉토리
- `modules/` : 데이터 전처리나 augmentation과 관련된 모듈 혹은 train시 사용되는 모듈들이 있는 디렉토리
    - `preprocessor.py` : 데이터 전처리
    - `augmentation.py` : 데이터 augmentation
    - `loss.py` : custom loss
    - `make_rtt_csv.py` : Round-trip Translation을 이용한 데이터 추가 생성
    - `concat_csv.py` : 추가 생성 데이터 csv 파일을 concatenation (RTT or TAPT)
    - 
- `train.py` : RE task 단일 모델 및 k-fold 모델 학습
- `train_mlm.py` : MLM task 사전 학습
- `load_data.py` : 데이터 클래스 정의
- `tokenization.py` : tokenization
- `model.py` : custom model
- `model_ensemble.py` : 모델 결과 csv 파일들을 soft-voting ensemble
- `inference.py` : RE task 단일 모델 및 k-fold 모델 결과 생성

## Getting Started

### Dependencies

- torch==1.6.0
- pandas==1.1.5
- scikit-learn~=0.24.1
- transformers ~
- wandb==0.12.1

### Install Requirements

```
sh requirement_install.sh
```

### Training

#### Options-train.py

|argument        |description|default|
|:--------------:|:----------|:------|
|save_dir|모델 저장 경로 설정| ./best_models|
|PLM|사용할 모델 선택(checkpoint)|klue/bert-base|
|MLM_checkpoint  | MLM 모델 불러오기 |./best_models/klue-roberta-large-rtt-pem-mlm|
|entity_flag| typed entity marker punct 사용 | False |
| use_mlm | MaskedLM pretrained model 사용 유무 | False |
| epochs | train epoch 횟수 지정 | 3 |
| lr | learning rate 지정 | 5e-5 |
| train_batch_size | train batch size 설정 | 16 |
| warmup_steps | warmup step 설정 | 500 |
| weigth_decay | weight decay 설정 | 0.01 |
| evaluatoin_stratgey   | evaluation_strategy 설정 | steps |
| ignore_mismatched | pretrained model load 시, mismatched size 무시 유무 | False|
| eval_flag | validation data 사용 유무 | False |
| eval_ratio | evalation data size ratio 설정 | 0.2 |
| seed | random seed 설정 | 2 |
| dotenv_path | 사용자 env 파일 경로 설정   | /opt/ml/wandb.env |
| wandb_unique_tag | wandb tag 설정 | bert-base-high-lr |
| entity_flag | 사용자 env 파일 경로 설정   | False |
| preprocessing_cmb | 데이터 전처리 방식 선택(0: 특수 문자 제거, 1: 특수 문자 치환, 2: date 보정, 3: 한글 띄워주기)| set ex: 0 1 2 |
| mecab_flag | mecab을 활용한 형태소 분리 | False |
| add_unk_token | unk token vocab에 저장 | False |
| k_fold | Stratified K Fold 사용 | 0 |
| adea_flag | adea 사용 유무 | False |
| augmentation_flag | rtt augmentation dataset 사용 유무 | False |
| model_type | 대,소분류 진행할 class 입력 | default |
| model_name | custom 모델 입력 | None |

#### Options-train_mlm.py

**train.py의 --use_pem, --model_type 제외 동일**

|argument  |description|default|
|:------------:|:----------|:------|
|--use_pem| 데이터 전처리 방식 선택 유무 |False|


```
$ python train_mlm.py --PLM klue/roberta-large --use_pem --preprocessing_cmb 0 1 2 --use_rtt
$ python train.py --PLM klue/roberta-large --wandb_unique_tag AddLayerNorm_lr_2e5_k_fold_10 --entity_flag --preprocessing_cmb 0 1 3 --mecab_flag --model_name AddLayerNorm --lr 2e-5 --k_fold 10
```

### Inference

#### Options-inference.py

|argument  |description|default|
|:------------:|:----------|:------|
|--model_dir| 선택할 모델 경로 |./best_models|
|--PLM | 모델 checkpoint |klue/bert-base|
|--entity_flag | typed entity marker punct 사용 유무 |False|
|--preprocessing_cmb|데이터 전처리 방식 선택(0: 특수 문자 제거, 1: 특수 문자 치환, 2: date 보정, 3: 한글 띄워주기)| set ex: 0 1 2 |
|--mecab_flag | Mecab을 활용해 형태소를 분리 유무 | False |
| --add_unk_token | unk token vocab에 저장한 tokenizer 사용 유무 | False |
| --k_fold | Stratified K Fold 사용 | 0 |
| --model_type | 대,소분류 진행 유무 | False |
| --model_name | custom 모델 입력 | None |

```
$ python inference.py --PLM klue-roberta-large --k_fold 10 --entity_flag --preprocessing_cmb 0 1 2 --mecab_flag
```

### Soft-voting Ensemble

단일 모델(혹은 k-fold 모델)의 결과 csv 파일들에서 probs열을 soft-voting 하여 최종 ensemble 결과 csv를 생성합니다.

#### Options-model_ensemble.py

|argument  |description|default|
|:------------:|:----------|:------|
|--dir| 앙상블할 모델 경로 선택 |all|

```
$ python model_ensemble.py --dir all
```

## Architecture

1. Model
    1. Model 1
        
        ![스크린샷 2021-10-08 오전 10.59.06.png](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/2686f3c1-33a9-4826-b8d2-98dd75d31ab4/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7_2021-10-08_%EC%98%A4%EC%A0%84_10.59.06.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20211008%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20211008T043041Z&X-Amz-Expires=86400&X-Amz-Signature=739a2e59cdd77d8107c4355c133c99b98961838ae453a8a31e085b4604ef8c62&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22%25EC%258A%25A4%25ED%2581%25AC%25EB%25A6%25B0%25EC%2583%25B7%25202021-10-08%2520%25EC%2598%25A4%25EC%25A0%2584%252010.59.06.png%22)
        
    2. Model 2
        
        ![스크린샷 2021-10-08 오전 11.00.00.png](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/7426081d-0ba4-4a72-8d41-b0ead6c06ecf/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7_2021-10-08_%EC%98%A4%EC%A0%84_11.00.00.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20211008%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20211008T043433Z&X-Amz-Expires=86400&X-Amz-Signature=0dc056c691ccd07f8a81c5a2193103449b777f6fe8010e7378d68ab4eed0764a&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22%25EC%258A%25A4%25ED%2581%25AC%25EB%25A6%25B0%25EC%2583%25B7%25202021-10-08%2520%25EC%2598%25A4%25EC%25A0%2584%252011.00.00.png%22)
        
    3. Model 3
        
        ![스크린샷 2021-10-08 오전 11.00.27.png](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/953e0b67-1e1c-452e-8881-1e9df078b0ae/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7_2021-10-08_%EC%98%A4%EC%A0%84_11.00.27.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20211008%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20211008T043537Z&X-Amz-Expires=86400&X-Amz-Signature=69f836f5af693aa87e7469c82ce8c525b7011f2a31413d41aa01cc95d76a1ee2&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22%25EC%258A%25A4%25ED%2581%25AC%25EB%25A6%25B0%25EC%2583%25B7%25202021-10-08%2520%25EC%2598%25A4%25EC%25A0%2584%252011.00.27.png%22)
        
    4. Model 4
        
        ![스크린샷 2021-10-08 오전 11.00.48.png](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/d86689bb-8896-41dc-be7b-46f1a78faee5/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7_2021-10-08_%EC%98%A4%EC%A0%84_11.00.48.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20211008%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20211008T043609Z&X-Amz-Expires=86400&X-Amz-Signature=1d6e89bf105e5cf92b32acef19fe7400edddc3da87b3d2cbc9a58a47c5807e0d&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22%25EC%258A%25A4%25ED%2581%25AC%25EB%25A6%25B0%25EC%2583%25B7%25202021-10-08%2520%25EC%2598%25A4%25EC%25A0%2584%252011.00.48.png%22)
        


    5. Model 5

![스크린샷 2021-10-08 오전 11.02.01.png](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/a0840bc8-cf4f-4a62-b4a7-f2ed95ec67a2/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7_2021-10-08_%EC%98%A4%EC%A0%84_11.02.01.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20211008%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20211008T043640Z&X-Amz-Expires=86400&X-Amz-Signature=7a8c818fd06c4bc95c9ed04d14dcfad4f52463cdfd203f5d9a53173d13cfeb44&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22%25EC%258A%25A4%25ED%2581%25AC%25EB%25A6%25B0%25EC%2583%25B7%25202021-10-08%2520%25EC%2598%25A4%25EC%25A0%2584%252011.02.01.png%22)

1. Model Ensemble
    
    ![스크린샷 2021-10-08 오전 11.02.36.png](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/7fa47087-1c91-46a0-8ec0-299b63f7235c/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7_2021-10-08_%EC%98%A4%EC%A0%84_11.02.36.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20211008%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20211008T043749Z&X-Amz-Expires=86400&X-Amz-Signature=540cef97c69524bc78f0d85f434dc54df698e4c5c61414548e3a0b1075baef0a&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22%25EC%258A%25A4%25ED%2581%25AC%25EB%25A6%25B0%25EC%2583%25B7%25202021-10-08%2520%25EC%2598%25A4%25EC%25A0%2584%252011.02.36.png%22)
    

## Result
| |micro_f1|AUPRC|RANK
|:---:|:---:|:---:|:---:|
|Public|73.860|81.085|11|
|Private|73.069|82.295|7|
