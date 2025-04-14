import json
import sys
import os

sys.path.append('../../../')
sys.path.append('../../../python_parser')
retval = os.getcwd()
import argparse
import warnings
import pickle
import torch
from model import CodeBERT, GraphCodeBERT, CodeT5
from l_for_inference_run import CodeBertTextDataset, GraphCodeBertTextDataset, CodeT5TextDataset, set_seed

#from run import CodeBertTextDataset, GraphCodeBertTextDataset, CodeT5TextDataset, set_seed
from attacker import Attacker, infrence_Attacker
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer, RobertaForMaskedLM,
                          RobertaForSequenceClassification, T5Config, T5ForConditionalGeneration)
import fasttext
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)  # Only report warning

MODEL_CLASSES = {
    'codebert_roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'graphcodebert_roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer)
}


def get_code_pairs(args):
    file_path = args.eval_data_file
    postfix = file_path.split('/')[-1].split('.txt')[0]
    folder = '/'.join(file_path.split('/')[:-1])
    code_pairs_file_path = os.path.join(folder, '{}_cached_{}.pkl'.format(args.model_name, postfix))
    with open(code_pairs_file_path, 'rb') as f:
        code_pairs = pickle.load(f)
    return code_pairs


def main():
    current_time = datetime.now()
    print('Current time1: {}'.format(current_time))
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--model_name", default="", type=str,
                        help="model name")

    args = parser.parse_args()
    args.device = torch.device("cuda")
    # Set seed
    args.seed = 123456
    args.eval_batch_size = 32
    args.language_type = 'java'
    args.n_gpu = 2
    args.block_size = 512
    args.use_ga = True

    if args.model_name == 'codebert':
        args.output_dir = './saved_models'
        args.model_type = 'codebert_roberta'
        args.config_name = 'CODA/microsoft/codebert-base'  # '/root/CODA/microsoft/codebert-base'
        args.model_name_or_path = 'CODA/microsoft/codebert-base'  # '/root/CODA/microsoft/codebert-base'
        args.tokenizer_name = 'CODA/roberta-base'  # '/root/CODA/roberta-base'
        args.base_model = 'CODA/microsoft/codebert-base-mlm'  # '/root/CODA/microsoft/codebert-base-mlm'
        args.number_labels = 2
    if args.model_name == 'graphcodebert':
        args.output_dir = './saved_models'
        args.model_type = 'graphcodebert_roberta'
        args.config_name = 'CODA/microsoft/graphcodebert-base'
        args.tokenizer_name = 'CODA/microsoft/graphcodebert-base'
        args.model_name_or_path = 'CODA/microsoft/graphcodebert-base'
        args.base_model = 'CODA/microsoft/graphcodebert-base'
        args.code_length = 448
        args.data_flow_length = 64
        args.number_labels = 1
    if args.model_name == 'codet5' or args.model_name == 'codet5p':
        args.output_dir = './saved_models'
        args.model_type = 'codet5'
        args.config_name = 'CODA/codet5-base-multi-sum'
        args.model_name_or_path = 'CODA/codet5-base-multi-sum'
        args.tokenizer_name = 'CODA/codet5-base-multi-sum'
        args.base_model = 'CODA/microsoft/codebert-base-mlm'
        args.number_labels = 2

    # Set seed
    set_seed(args)
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels = args.number_labels
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=False,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    if args.model_name_or_path:
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)
    else:
        model = model_class(config)

    if args.model_name == 'codebert':
        model = CodeBERT(model, config, tokenizer, args)
    elif args.model_name == 'graphcodebert':
        model = GraphCodeBERT(model, config, tokenizer, args)
    elif args.model_name == 'codet5' or args.model_name == 'codet5p':
        model = CodeT5(model, config, tokenizer, args)

    args.output_dir = 'shilihua_ori'
    checkpoint_prefix = 'checkpoint-best-f1/%s_model.bin' % args.model_name
    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))

    model.load_state_dict(torch.load(output_dir))
    model.to(args.device)

    tokenizer_mlm = RobertaTokenizer.from_pretrained(args.base_model)
    # Load CodeBERT (MLM) model
    codebert_mlm = RobertaForMaskedLM.from_pretrained(args.base_model)
    codebert_mlm.to('cuda')

    # load fasttext
    fasttext_model = fasttext.load_model("../../../fasttext_model.bin")
    if args.model_name == 'codet5p':
        aaa = '../dataset/codet5_all_subs.json'
    else:
        aaa = '../dataset/%s_all_subs.json' % args.model_name
    generated_substitutions = json.load(open(aaa, 'r'))

    attacker = infrence_Attacker(args, model, tokenizer, tokenizer_mlm, codebert_mlm, fasttext_model,
                                 generated_substitutions)

    ## Load tensor features
    if args.model_name == 'codebert':
        eval_dataset = CodeBertTextDataset(tokenizer, args, args.eval_data_file)
    elif args.model_name == 'graphcodebert':
        eval_dataset = GraphCodeBertTextDataset(tokenizer, args, args.eval_data_file)
    elif args.model_name == 'codet5' or args.model_name == 'codet5p':
        eval_dataset = CodeT5TextDataset(tokenizer, args, args.eval_data_file)

    success_attack = 0
    total_cnt = 0
    not_passed = []

    for index, example in enumerate(eval_dataset):

        is_success = attacker.attack(
            example
        )
        total_cnt += 1
        if is_success == 1:
            success_attack += 1
        else:
            not_passed.append(index)
        print("inference successful rate: %.2f%%" % ((1.0 * success_attack / total_cnt) * 100))
        print("successful items count: ", success_attack)
        print("Total count: ", total_cnt)
        print("Index: ", index)
        print()
    for item in not_passed:
        print(item, end=' ')  # 每个元素后面加空格，不换行


if __name__ == '__main__':
    main()
