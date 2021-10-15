from glob import glob
import os
import pickle as pickle
import argparse

import pandas as pd


def num_to_label(label):
    """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
    """
    origin_label = []
    with open("dict_num_to_label.pkl", "rb") as f:
        dict_num_to_label = pickle.load(f)
    for v in label:
        origin_label.append(dict_num_to_label[v])

    return origin_label


def ensemble():
    csv_files = glob(f"./prediction/all/*.csv")
    for index, csv_name in enumerate(csv_files, 1):
        print("{:02d}_dir is: {}".format(index, os.path.basename(csv_name)))

    for i, csv_file in enumerate(csv_files):
        df = pd.read_csv(csv_file)
        if i == 0:
            probs = pd.DataFrame(df.probs.map(eval).to_list())
        else:
            probs += pd.DataFrame(df.probs.map(eval).to_list())

    probs = probs.div(probs.sum(axis=1), axis=0)
    probs_argmax = probs.idxmax(axis=1).values.tolist()
    pred_answer = num_to_label(probs_argmax)
    output_prob = probs.values.tolist()

    output = pd.DataFrame(
        {
            "id": [i for i in range(7765)],
            "pred_label": pred_answer,
            "probs": output_prob,
        }
    )
    output.to_csv(f"./prediction/ensemble.csv", index=False)


if __name__ == "__main__":
    ensemble()
    print("Finish Ensemble !")
