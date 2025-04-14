from __future__ import absolute_import, division, print_function
import sys
sys.path.append('../../')
sys.path.append('../../python_parser')
import warnings
warnings.filterwarnings("ignore")
import argparse
from tqdm import tqdm, trange
import json
import os
from run_parser import get_identifiers_ori


def generate(args,attack_method, model_name):
    source_codes = []
    with open(args.train_data_file) as rf:
        for line in rf:
            source_codes.append(line.split(' <CODESPLIT> ')[0].strip().replace("\\n", "\n").replace('\"', '"'))
    with open(args.eval_data_file) as rf:
        for line in rf:
            source_codes.append(line.split(' <CODESPLIT> ')[0].strip().replace("\\n", "\n").replace('\"', '"'))
    results = set()
    for i in tqdm(source_codes):
        identifiers, code_tokens = get_identifiers_ori(i, args.language_type)
        for j in identifiers:
            results.add(j)
    results = [i+'\n' for i in results]
    print(len(results))
    w_path = '../dataset/All_vars/%s/%s/' % (attack_method, model_name)
    if not os.path.exists(w_path):
        os.makedirs(w_path)
    open(w_path + '/all_vars.txt', 'w').writelines(results)




def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.train_data_file = '../dataset/train.txt'
    args.language_type = 'c'
    args.eval_data_file = '../CodeDenoise/adv_train/defec/ALERT/codebert/test.txt'
    attack_methods = ['CODA', 'ALERT', 'CodeTAE']
    model_names = ['codebert', 'graphcodebert', 'codet5', 'codet5p']
    for model_name in model_names:
        for attack_method in attack_methods:
            if attack_method == 'CodeTAE' and model_name == 'codet5p':
                continue
            args.eval_data_file = '../CodeDenoise/adv_train/defec/%s/%s/test.txt' % (attack_method, model_name)
            generate(args, attack_method, model_name)


if __name__ == "__main__":
    main()