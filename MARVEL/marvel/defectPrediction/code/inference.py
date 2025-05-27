from tree_sitter import Language, Parser
import sys, os

sys.path.append('../../../')
sys.path.append('../../')
# print(sys.path)
from load_data import *
import multiprocessing
import fasttext
import torch
import json
import argparse
import time
import random
import numpy as np
from model import *
from utils.utils_alert import build_vocab
from attacker import AlertAttacker, CodaAttacker, MHMAttacker
from transformers import RobertaForMaskedLM
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer, RobertaForSequenceClassification,
                          T5Config, T5ForConditionalGeneration)

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_CLASSES = {
    'codebert': (RobertaConfig, RobertaModel, RobertaTokenizer, CodeBERT, CodeBERTnoise, CodeBERT_twoContact),
    'graphcodebert': (
    RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, GraphCodeBERT, GraphCodeBERTnoise,
    GraphCodeBERT_twoContact),
    'unixcoder': (RobertaConfig, RobertaModel, RobertaTokenizer, UniXCoder, UniXCodernoise, UniXCoder_twoContact),
    'codet5':(T5Config, T5ForConditionalGeneration, RobertaTokenizer,CodeT5,CodeT5noise, CodeT5_twoContact),
    'codet5p':(T5Config, T5ForConditionalGeneration, RobertaTokenizer,CodeT5,CodeT5noise, CodeT5_twoContact)
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.model_type == 'codebert':
        os.environ['PYHTONHASHSEED'] = str(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    elif args.model_type == 'graphcodebert':
        if args.n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)
    elif args.model_type == 'unixcoder':
        os.environ['PYHTONHASHSEED'] = str(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--eval_data_file", default='../dataset/valid.txt', type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--model_name", default="", type=str,
                        help="model name")
    parser.add_argument("--model_type", default="", type=str,
                        help="model name")
    parser.add_argument("--attack_name", default="", type=str,
                        help="attack name")
    parser.add_argument("--info", default="", type=str,
                        help="info")
    parser.add_argument("--model_dir", default=None, type=str,
                        help="model_dir")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="eval batch size")
    parser.add_argument("--index", nargs='+',
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--save_name", default='model1_attention.bin', type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    args = parser.parse_args()
    args.device = torch.device(device)

    # Set seed
    args.seed = 123456
    args.number_labels = 4
    # args.eval_batch_size = 32
    args.language_type = 'c'
    args.n_gpu = torch.cuda.device_count()
    args.block_size = 512
    args.use_ga = True
    args.code_length = 448
    args.data_flow_length = 64

    set_seed(args)

    print("==========Loading Model===========", flush=True)


    if args.model_type == 'codebert':
        args.config_name = "/data/CODA_new_conrtainer/CODA_new/CODA/microsoft/codebert-base"
        args.model_name_or_path = args.config_name
        args.tokenizer_name = args.config_name
        args.base_model = "/data/CODA_new_conrtainer/CODA_new/CODA/microsoft/codebert-base-mlm"
        args.model_name = 'codebert'
        args.code_length = 512
        args.data_flow_length = 0
    elif args.model_type == "graphcodebert":
        args.config_name = "/data/CODA_new_conrtainer/CODA_new/CODA/microsoft/graphcodebert-base"
        args.model_name_or_path = args.config_name
        args.tokenizer_name = args.config_name
        args.base_model = args.config_name
        args.model_name = 'graphcodebert'
    elif args.model_type == 'codet5':
        args.config_name = "/data/CODA_new_conrtainer/CODA_new/CODA/codet5-base-multi-sum"
        args.model_name_or_path = args.config_name
        args.tokenizer_name = args.config_name
        args.learning_rate = 2e-5
        args.code_length = 512
        args.data_flow_length = 0
        args.model_name = 'codet5'
        args.base_model = "/data/CODA_new_conrtainer/CODA_new/CODA/microsoft/codebert-base-mlm"
    elif args.model_type == 'codet5p':
        args.config_name = "/data/CODA_new_conrtainer/CODA_new/CODA/codet5p-220m"
        args.model_name_or_path = args.config_name
        args.tokenizer_name = args.config_name
        args.learning_rate = 2e-5
        args.code_length = 512
        args.data_flow_length = 0
        args.model_name = 'codet5p'
        args.base_model = "/data/CODA_new_conrtainer/CODA_new/CODA/microsoft/codebert-base-mlm"

    config_class, model_class, tokenizer_class, Model, Model_Noise, Model_Contact = MODEL_CLASSES[args.model_type]

    config = config_class.from_pretrained(args.config_name)
    config.num_labels = args.number_labels
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)

    outputdir = "../model/{}/checkpoint-best-acc/{}_{}_model.bin".format(args.model_type, args.model_type,
                                                                         args.save_name)

    if args.save_name == 'original':
        model = Model(model, config, tokenizer, args)
        model.config.ouput_attentions = True
        model.load_state_dict(torch.load(outputdir, map_location=torch.device(device)), strict=False)
        model.to(device)
    elif args.save_name == 'marvel':
        model1 = Model_Noise(model, config, tokenizer, args)
        model2 = Model_Noise(model, config, tokenizer, args)

        model1.config.ouput_attentions = True
        model2.config.ouput_attentions = True

        model = Model_Contact(config, model1, model2, args)
        model.load_state_dict(torch.load(outputdir, map_location=torch.device(device)), strict=False)
        model.to(device)
    print("=======Loading Model Finished=====", flush=True)

    config_class, model_class, tokenizer_class, _, _, _ = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels = args.number_labels
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=False,
                                                cache_dir=args.cache_dir if args.cache_dir else None)


    pool = multiprocessing.Pool(cpu_cont)
    if args.model_type == 'codebert':
        eval_dataset = CodeBertTextDataset(tokenizer, args, args.eval_data_file, pool=pool)
    elif args.model_type == 'graphcodebert':
        eval_dataset = GraphCodeBertTextDataset(tokenizer, args, args.eval_data_file, pool=pool)
    elif args.model_type == 'unixcoder':
        eval_dataset = UniXCoderTextDataset(tokenizer, args, args.eval_data_file, pool=pool)
    elif args.model_type == 'codet5' or args.model_type == 'codet5p':
        eval_dataset = CodeT5TextDataset(tokenizer, args, args.eval_data_file, pool=pool)

    start_time = time.time()
    total_cnt = 0
    is_success=0
    for index, example in enumerate(eval_dataset):
        # if index < int(args.index[0]) or index >= int(args.index[1]):
        #     continue
        total_cnt += 1
        orig_prob, orig_label = model.get_results([example], args.eval_batch_size)
        orig_prob = orig_prob[0]
        orig_label = orig_label[0]
        if args.model_name == 'codebert':
            true_label = example[1].item()
        elif args.model_name == 'graphcodebert':
            true_label = example[3].item()
        elif args.model_name == 'unixcoder':
            true_label = example[1].item()
        elif args.model_name == 'codet5' or args.model_name == 'codet5p':
            true_label = example[1].item()
        if true_label == orig_label:
            is_success += 1
            print('num %d SUCCESS!\n' % index, flush=True)
        else:
            print('num %d FAILED!\n' % index, flush=True)
        end_time = time.time()
        print("Success rate: {}".format(1.0 * is_success / total_cnt), flush=True)
        print("Successful items count: {}".format(is_success))
        print("Total count: {} \nIndex: {} \nTime: {} min".format(total_cnt, index,
                                                                  round((end_time - start_time) / 60, 2)), flush=True)

    print("Over!   Success rate: {}".format(1.0 * is_success / total_cnt))
    print("Successful items count: {}".format(is_success))


if __name__ == '__main__':
    main()