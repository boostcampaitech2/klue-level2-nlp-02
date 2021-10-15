import pandas as pd
import argparse
import os

def concat_csv(csv1_path, csv2_path):
    csv1 = pd.read_csv(csv1_path)
    csv2 = pd.read_csv(csv2_path)
    
    csv1.columns = ['id', 'sentence', 'subject_entity', 'object_entity', 'label', 'source']
    csv2.columns = ['id', 'sentence', 'subject_entity', 'object_entity', 'label', 'source']
    
    concat_result = pd.concat([csv1, csv2])
    concat_result.reset_index(drop=True)
    
    return concat_result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--base_dir', default='/opt/ml/dataset/train', help='base directory has csv files')
    parser.add_argument('--csv1', default='train', help='first csv file name')
    parser.add_argument('--csv2', default='rtt_data', help='second csv file name')
    parser.add_argument('--save_name', default='train_final', help='output csv file name')
    
    args = parser.parse_args()
    
    
    # make path
    csv1_path = os.path.join(args.base_dir, args.csv1+'.csv')
    csv2_path = os.path.join(args.base_dir, args.csv2+'.csv')
    result_path = os.path.join(args.base_dir, args.save_name+'.csv')
    
    result = concat_csv(csv1_path, csv2_path)
    result.to_csv(result_path, index=False)
    
    print('Finish concat csv files !')