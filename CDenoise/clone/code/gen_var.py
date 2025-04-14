from __future__ import absolute_import, division, print_function
import sys
sys.path.append('../../')
sys.path.append('../../python_parser')
import warnings
warnings.filterwarnings("ignore")
import argparse
from tqdm import tqdm, trange
import json
from run_parser import get_identifiers_ori
import os


def generate(args,attack_method, model_name):
    source_codes = []
    with open(args.eval_data_file) as rf:
        for line in rf:
            json_data = json.loads(line)
            code = json_data['func']
            source_codes.append(code.strip().replace("\\n", "\n").replace('\"', '"'))

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
    args.language_type = 'java'
    attack_methods = ['CODA', 'ALERT', 'CodeTAE']
    model_names = ['codebert', 'graphcodebert', 'codet5','codet5p']
    for model_name in model_names:
        for attack_method in attack_methods:
            if attack_method=='CodeTAE' and model_name=='codet5p':
                continue
            args.eval_data_file = '/workspace/CodeDenoise/adv_train/clone/%s/%s/data.jsonl' % (attack_method, model_name)
            generate(args, attack_method, model_name)

if __name__ == "__main__":
    main()