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
    with open('dict_num_to_label.pkl', 'rb') as f:
        dict_num_to_label = pickle.load(f)
    for v in label:
        origin_label.append(dict_num_to_label[v])

    return origin_label

def select_csv(base_path):
    csv_dir = base_path
    # dirs = os.listdir(csv_dir)
    dirs = glob(os.path.join(csv_dir, '*.csv'))
    dirs = sorted(dirs)

    for i, d in enumerate(dirs, 0):
        dirs[i] = os.path.basename(d)
        print("(%d) %s" % (i, dirs[i]))
    idx_list = input("Select csv files you want to ensemble: ").split()
    
    csv_lists = []
    for index, file_idx in enumerate(idx_list, 1):
        csv_file = os.path.abspath(
            os.path.join(csv_dir, dirs[int(file_idx)]))
        csv_lists.append(csv_file)
        print("{:02d}_dir is: {}".format(index, csv_file))

    return csv_lists

def ensemble(args):
    if args.dir == 'all':
        csv_files = glob(f"./prediction/all/*.csv")
        for index, csv_name in enumerate(csv_files, 1):
            print("{:02d}_dir is: {}".format(index, os.path.basename(csv_name)))
    else:
        csv_files = select_csv("./prediction")
    
    for i, csv_file in enumerate(csv_files) :
        df = pd.read_csv(csv_file)
        if i == 0:
            probs = pd.DataFrame(df.probs.map(eval).to_list())
        else :
            probs += pd.DataFrame(df.probs.map(eval).to_list())
    
    probs = probs.div(probs.sum(axis=1), axis=0)
    probs_argmax = probs.idxmax(axis=1).values.tolist()
    pred_answer = num_to_label(probs_argmax)
    output_prob = probs.values.tolist()

    output = pd.DataFrame(
        {'id': [i for i in range(7765)], 'pred_label': pred_answer, 'probs': output_prob, })
    output.to_csv(f"./prediction/ensemble_{args.dir}.csv", index=False)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='all', help='dir type(default: all)')
    args = parser.parse_args()

    ensemble(args)
    print('Finish Ensemble !')