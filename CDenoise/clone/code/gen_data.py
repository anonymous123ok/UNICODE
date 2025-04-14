from __future__ import absolute_import, division, print_function
import sys
sys.path.append('../../')
sys.path.append('../../python_parser')
import warnings
warnings.filterwarnings("ignore")
import os
import re
import torch
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import CodeBERT, GraphCodeBERT, CodeT5
from run import CodeBertTextDataset, GraphCodeBertTextDataset, CodeT5TextDataset
from run_parser import get_identifiers_ori, get_example, remove_comments_and_docstrings
from transformers import (RobertaModel, RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, T5Config,
                          T5ForConditionalGeneration)


MODEL_CLASSES = {
    'codebert_roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'graphcodebert_roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def generate(args, model, tokenizer, mlm_tokenizer, data_file, method,attack_method):
    idx2jsonl={}
    with open(args.datajsonl) as rf:
        for line in rf:
            json_data = json.loads(line)
            code = json_data['func']
            idx = json_data['idx']
            idx2jsonl[idx] = code
    source_codes = []
    with open(data_file) as rf:
        for line in rf:
            line = line.strip()
            url1, url2, label = line.split('\t')
            code1 = idx2jsonl[url1]
            code2 = idx2jsonl[url2]
            content = '<s>' + code1 + '</s><s>' + code2 + '</s>'.replace("\\n", "\n").replace('\"', '"')
            source_codes.append(content)

    if args.model_name == 'codebert':
        print("+++++++++++++")
        dataset = CodeBertTextDataset(tokenizer, args, data_file)
    elif args.model_name == 'graphcodebert':
        print("-----------")
        dataset = GraphCodeBertTextDataset(tokenizer, args, data_file)
    elif args.model_name == 'codet5' or args.model_name == 'codet5p':
        dataset = CodeT5TextDataset(tokenizer, args, data_file)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
    model = torch.nn.DataParallel(model)
    model.eval()
    results = []
    clean_train_data = []
    for batch_i, batch in enumerate(tqdm(dataloader)):
        if args.model_name == 'codebert':
            inputs = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            with torch.no_grad():
                lm_loss, logit, _ = model(inputs, labels)
        elif args.model_name == 'graphcodebert':

            inputs_ids1 = batch[0].to(args.device)
            attn_mask1 = batch[1].to(args.device)
            position_idx1 = batch[2].to(args.device)
            labels = batch[3].to(args.device)
            inputs_ids2 = batch[4].to(args.device)
            attn_mask2 = batch[5].to(args.device)
            position_idx2 = batch[6].to(args.device)

            with torch.no_grad():
                lm_loss, logit, _ = model(inputs_ids1,position_idx1,attn_mask1,inputs_ids2,position_idx2,attn_mask2,labels)
        elif args.model_name == 'codet5' or args.model_name == 'codet5p':
            inputs = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            with torch.no_grad():
                lm_loss, logit, _ = model(inputs, labels)
        label = labels.cpu().numpy()[0]
        label_pred = np.argmax(logit.cpu().numpy()[0])
        if method == 'train' and label_pred == label:
            ori_code_temp = source_codes[batch_i]
            try:
                code_temp = remove_comments_and_docstrings(ori_code_temp, args.language_type).strip()
            except:
                code_temp = ori_code_temp.strip()
            dic = {}
            dic['text'] = code_temp
            clean_train_data.append(dic)
            filename = "../Finetune_mlm/data/clone/" + args.model_name
            if not os.path.exists(filename):
                os.makedirs(filename)
            with open(filename + "/clean_train.json", "w", encoding="utf-8") as f:
                json.dump(clean_train_data, f, ensure_ascii=False)
        #     code_temp = re.sub(' +', ' ', code_temp)
        #     t = mlm_tokenizer.tokenize(code_temp)[:args.block_size]
        #     code_temp = mlm_tokenizer.convert_tokens_to_string(t)
        #     identifiers, code_tokens = get_identifiers_ori(code_temp, args.language_type)
        #     for iden in identifiers:
        #         results.append(ori_code_temp.replace('\n', '\\n').replace("\t", "\\t").replace('"', '\"') + ' <CODESPLIT> ' +
        #                        get_example(code_temp, iden, '<mask>', args.language_type).replace('\n', '\\n').replace("\t", "\\t").replace('"', '\"') + ' <CODESPLIT> ' +
        #                        iden + ' <CODESPLIT> ' + str(label)+ ' <CODESPLIT> ' + str(label_pred)+'\n')
        if method == 'test' and label_pred != label:
            ori_code_temp = source_codes[batch_i]
            try:
                code_temp = remove_comments_and_docstrings(ori_code_temp, args.language_type).strip()
            except:
                code_temp = ori_code_temp.strip()
            code_temp = re.sub(' +', ' ', code_temp)
            t = mlm_tokenizer.tokenize(code_temp)[:args.block_size-2]
            code_temp = mlm_tokenizer.convert_tokens_to_string(t)
            results.append(ori_code_temp.replace('\n', '\\n').replace("\t", "\\t").replace('"', '\"')+' <CODESPLIT> '+
                           code_temp.replace('\n', '\\n').replace("\t","\\t").replace('"', '\"')+' <CODESPLIT> '+
                           '' + ' <CODESPLIT> ' + str(label) + ' <CODESPLIT> ' + str(label_pred)+'\n')
    w_path = '../dataset/All_vars/%s/%s/' % (attack_method, args.model_name)
    if not os.path.exists(w_path):
        os.makedirs(w_path)
    open(w_path + '/mlm_%s_%s.txt' % (args.model_name, method), 'w').writelines(results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=None, type=str, required=True, help="model name.")
    args = parser.parse_args()
    args.output_dir = './saved_models/'
    args.block_size = 512
    args.seed = 123456
    args.batch_size = 1
    args.number_labels = 2
    if args.model_name == 'codebert':
        args.model_type = 'codebert_roberta'
        args.config_name = 'microsoft/codebert-base'
        args.model_name_or_path = 'microsoft/codebert-base'
        args.tokenizer_name = 'microsoft/codebert-base'
    elif args.model_name == 'graphcodebert':
        args.model_type = 'graphcodebert_roberta'
        args.config_name = 'microsoft/graphcodebert-base'
        args.model_name_or_path = 'microsoft/graphcodebert-base'
        args.tokenizer_name = 'microsoft/graphcodebert-base'
        args.code_length = 448
        args.data_flow_length = 64
        args.number_labels = 1
    elif args.model_name == 'codet5':
        args.model_type = 'codet5'
        args.config_name = 'codet5-base-multi-sum'
        args.model_name_or_path = 'codet5-base-multi-sum'
        args.tokenizer_name = 'codet5-base-multi-sum'
    elif args.model_name == 'codet5p':
        args.model_type = 'codet5'
        args.config_name = 'codet5p-220m'
        args.tokenizer_name = 'codet5p-220m'
        args.model_name_or_path = 'codet5p-220m'
    args.language_type = 'java'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    set_seed(args)
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    config.num_labels = args.number_labels
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)

    model = model_class.from_pretrained(args.model_name_or_path, config=config)
    if args.model_name == 'codebert':
        model = CodeBERT(model, config, tokenizer, args)
    elif args.model_name == 'graphcodebert':
        model = GraphCodeBERT(model, config, tokenizer, args)
    elif args.model_name == 'codet5' or args.model_name == 'codet5p':
        model = CodeT5(model, config, tokenizer, args)

    checkpoint_prefix = 'checkpoint-best-f1/%s_model.bin' % args.model_name
    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))

    model.load_state_dict(torch.load(output_dir))
    model.to(args.device)


    mlm_tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base-mlm')

    attack_methods = ['CODA', 'ALERT', 'CodeTAE']
    for attack_method in attack_methods:
        if args.model_name == 'codet5p' and attack_method == 'CodeTAE': continue
        path1 = "../adv_train/clone/%s/%s" % (attack_method, args.model_name)
        args.train_data_file = path1 + '/train_sampled.txt'
        args.eval_data_file = path1 + '/test_sampled.txt'
        args.datajsonl = path1 + '/data.jsonl'
        generate(args, model, tokenizer, mlm_tokenizer, args.eval_data_file, 'test',attack_method)
        #generate(args, model, tokenizer, mlm_tokenizer, args.train_data_file, 'train',attack_method)



if __name__ == "__main__":
    main()