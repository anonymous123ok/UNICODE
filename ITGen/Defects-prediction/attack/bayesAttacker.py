"""
Search via Bayesian Optimization
===============

"""
import os
import sys

sys.path.append('../../../')
# sys.path.append('../code')
sys.path.append('../../../algorithms')
sys.path.append('../../../python_parser')
# sys.path.append('../../../data_preprocessors')
retval = os.getcwd()

from collections import defaultdict
import numpy as np
import torch
import random

# from algorithms.bayesopt.acquisition.algorithm import kmeans_pp, acquisition_maximization_with_indices
# from algorithms.bayesopt.historyboard import HistoryBoard
# from algorithms.bayesopt.surrogate_model.gp_model import MyGPModel
from algorithms.gp_model import MyGPModel
from algorithms.hb import HistoryBoard
from algorithms.kmeanspp import kmeans_pp
from algorithms.greedy_ascent import acquisition_maximization_with_indices

from utils import CodeDataset, get_identifier_posistions_from_code, is_valid_identifier, get_code_tokens, _tokenize, get_masked_code_by_position
# from run_parser import get_identifiers, get_gen_code, get_example_batch, extract_dataflow
from python_parser.run_parser import get_identifiers, get_gen_code, get_example_batch,extract_dataflow
import copy
from copy import deepcopy
import gc
import operator
import time


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 idx,
                 label,

    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.idx=str(idx)
        self.label=label


def convert_code_to_features(code, tokenizer, label, args):
    code=' '.join(code.split())
    code_tokens=tokenizer.tokenize(code)[:args.block_size-2]
    source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids+=[tokenizer.pad_token_id]*padding_length
    return InputFeatures(source_tokens,source_ids, 0, label)


class Bayes_Attacker(object):
    """An attack based on Bayesian Optimization
    """

    def __init__(self, args, model, tokenizer):
        
        self.args = args
        self.block_num = args.block_num
        self.batch_size = args.batch_size
        self.update_step = args.update_step
        self.max_patience = args.max_patience
        self.post_opt = args.post_opt 
        self.use_sod = args.use_sod
        self.dpp_type = args.dpp_type
        self.max_loop = args.max_loop
        self.fit_iter = args.fit_iter
        self.max_budget_key_type = args.max_budget_key_type
        self.model = model
        self.tokenizer = tokenizer

        self.memory_count = 0
    
    def filter_identifier(self, code, identifiers):
        # code_token = get_code_tokens(code)
        _, _, code_token = extract_dataflow(code, "c")
        filter_identifiers = []
        for identifier in identifiers:
            if is_valid_identifier(identifier):
                position = []
                for index, token in enumerate(code_token):
                    if identifier == token:
                        position.append(index)
                if not all(x > self.args.block_size - 2 for x in position):
                    filter_identifiers.append(identifier)
        return filter_identifiers

    def check_query_const(self):
        start_time = deepcopy(self.start_time)
        if self.model.query - self.query_times == self.query_budget or time.time() - start_time > 120:
            return True
        else:
            return False
    
    def seq2code(self, seq):
        assert type(seq) == torch.Tensor, f"type(seq) is {type(seq)}"
        if len(seq.shape) == 1:
            assert seq.shape[0] == self.len_seq, "indices length should be one of target indices length or seq length"
            seq_ = seq
        elif len(seq.shape) == 2:
            assert seq.shape[0] == 1 and seq.shape[1] == self.len_seq, "indices length should be one of target indices length or seq length"
            seq_ = seq.view(-1)

        cur_words = deepcopy(self.code_tokens_1)
        if len(seq_) == self.len_seq:
            for ct, ind in enumerate(seq_):
                if ind > 0 and ct in self.target_indices:
                    # cur_words[ct] = self.word_substitution_cache[ct][int(ind)]
                    idx = self.names_positions_dict[self.code_tokens_1[ct]]
                    for i in idx:
                        cur_words[i] = self.word_substitution_cache[ct][int(ind)]
        return cur_words
    
    def get_score(self,x,require_transform=True):
        if require_transform:
            x_ = self.seq2code(x.view(-1))
        else:
            x_ = x
        if self.model.query - self.query_times >= self.query_budget: return None
        if operator.eq(self.code_tokens_1, x_):
            logits = deepcopy(self.orig_logits)
            output_else_max = np.max(np.concatenate([logits[0][:self.true_label], logits[0][self.true_label+1:]]))
            return float(output_else_max - logits[0][self.true_label])
        else:
            replaced_words = self.replaced_words(self.code_tokens_1, x_)
            temp_code = get_example_batch(self.orig_code_1, replaced_words, 'c')
            new_feature = convert_code_to_features(temp_code, self.tokenizer, self.true_label, self.args)
            logits, _ = self.model.get_results(CodeDataset([new_feature]), self.args.eval_batch_size)
            output_else_max = np.max(np.concatenate([logits[0][:self.true_label], logits[0][self.true_label+1:]]))
            return float(output_else_max - logits[0][self.true_label])
    
    def get_scores(self, xs, require_transform=True):
        if require_transform:
            xs_ = []
            for x in xs:
                x_ = self.seq2code(x.view(-1))
                xs_.append(x_)
        else:
            xs_ = xs
        if self.model.query - self.query_times >= self.query_budget: return [None for _ in range(len(xs_))]

        new_example = []

        for code_tokens_1 in xs_:
            replaced_words = self.replaced_words(self.code_tokens_1, code_tokens_1)
            # print(replaced_words)
            temp_code = get_example_batch(self.orig_code_1, replaced_words, 'c')
            new_feature = convert_code_to_features(temp_code, self.tokenizer, self.true_label, self.args)
            new_example.append(new_feature)

        new_dataset = CodeDataset(new_example)
        logits, _ = self.model.get_results(new_dataset, self.args.eval_batch_size)
        scores = []
        for logit in logits:
            # gini = 1 - np.sum(logit**2)
            output_else_max = np.max(np.concatenate([logit[:self.true_label], logit[self.true_label+1:]]))
            scores.append(float(output_else_max - logit[self.true_label]))
            # scores.append(float(gini))
        return scores
    
    def get_query_budget(self, n_vertices, baseline='textfooler'):
        '''
            input
                syndict : synonym dictionary
                baseline : one of ['pwws','textfooler','pso','lsh','bae']
        '''
        assert baseline in ['pwws','textfooler','pso','lsh','bae'], f"max-budget-key-type {baseline} is not in ['pwws','textfooler','pso','lsh','bae']"
        if baseline == 'pso':
            query_budget = float('inf') # 最大查询预算设置为无穷大
        else:
            query_budget_count = n_vertices
            if baseline in ['textfooler','bae']:
                # 最大查询预算设置为词的数量（用于计算重要性）加上所有词的替换词数量减1的总和（用于搜索）
                query_budget = len(query_budget_count) # queries for importance calculation
                query_budget += sum([qc-1 for qc in query_budget_count]) # max queries for search
            elif baseline == 'pwws':
                # 最大查询预算设置为所有词的替换词数量减1的总和的两倍（用于计算重要性和搜索）
                query_budget = sum([qc-1 for qc in query_budget_count]) # queries for importance calculation
                query_budget += sum([qc-1 for qc in query_budget_count]) # max queries for search
        # print("max query budget : ", query_budget)
        return query_budget
    
    def get_initial_block_order(self, inds_list):
        leave_block_code = []
        for inds in inds_list:
            start, end = self.target_indices[inds[0]], self.target_indices[inds[-1]]
            del_code = deepcopy(self.code_tokens_1)
            for i in range(start,end+1):
                # TODO: is_identifiers?
                del_code[i] = '<MASK>'
                # if is_valid_identifier(del_code[i]):
                    # del_code[i] = '[MASK]'
                # else:
                    # continue
            leave_block_code.append(
                del_code
            )
        
        new_example = []
        for item in leave_block_code:
            replaced_words = self.replaced_words(self.code_tokens_1, item)
            # print(replaced_words)
            temp_code = get_example_batch(self.orig_code_1, replaced_words, 'c')
            new_feature = convert_code_to_features(temp_code, self.tokenizer, self.true_label, self.args)
            new_example.append(new_feature)
        new_dataset = CodeDataset(new_example)

        logits, _ = self.model.get_results(new_dataset, self.args.eval_batch_size)

        init_logits = deepcopy(self.orig_logits)
        # init_gini = 1 - np.sum(init_logits**2,axis=1)
        init_score = float(np.max(np.concatenate([init_logits[0][:self.true_label], init_logits[0][self.true_label+1:]])) - init_logits[0][self.true_label])
        tmp_scores = []
        for logit in logits:
            # gini = 1 - np.sum(logit**2)
            output_else_max = np.max(np.concatenate([logit[:self.true_label], logit[self.true_label+1:]]))
            tmp_scores.append(float(output_else_max - logit[self.true_label]))
            # tmp_scores.append(float(gini))

        index_scores = np.array([abs(score-init_score) for score in tmp_scores])
        # index_scores = np.array([abs(score-init_gini) for score in tmp_scores])
        index_order = (-index_scores).argsort()
        return index_order
    
    def init_before_loop(self):
        # Bayesian Optimization Loop
        # set best evaluated seq as initial seq.
        best_ind = self.hb.best_in_history()[3][0]
        initial_seq = self.hb.eval_X[best_ind]
        self.eff_len = self.hb.eff_len_seq
        
        D_0 = []
        index_order = self.get_index_order_for_block_decomposition()

        self.NB_INIT = int(np.ceil(self.eff_len / self.block_num))
        self.INDEX_DICT = defaultdict(list)
        self.HISTORY_DICT = defaultdict(list)
        self.BLOCK_QUEUE = [(0,int(i)) for i in range(self.NB_INIT)]
            
        center_seq = initial_seq
        center_ind = 0

        ALL_IND = index_order
        for KEY in self.BLOCK_QUEUE:
            self.INDEX_DICT[KEY] = deepcopy(ALL_IND[self.block_num*KEY[1]:self.block_num*(KEY[1]+1)])
            
        # print('self.INDEX_DICT',self.INDEX_DICT)
        LOCAL_OPTIMUM = defaultdict(list)
        stage = -1
        
        return D_0, center_seq, center_ind, LOCAL_OPTIMUM, stage

    def init_in_loop(self, KEY, center_ind):
        self.clean_memory_cache()
        opt_indices = deepcopy(self.INDEX_DICT[KEY])
        # print('opt_indices',opt_indices)
        fix_indices = list( set(list(range(self.eff_len))) - set(opt_indices) )
        # print('fix_indices',fix_indices)

        self.HISTORY_DICT[KEY].append(int(center_ind))

        if not opt_indices: cont = True
        else: cont = False

        stage_init_ind = len(self.hb.eval_Y)
        stage_iter = sum([self.hb.reduced_n_vertices[ind]-1 for ind in opt_indices]) 
        ex_ball_size = 10000
        n_samples = int(stage_iter / len(opt_indices)) if len(opt_indices)<=3 else int(stage_iter / len(opt_indices)) * 2
        next_len = self.block_num

        return opt_indices, fix_indices, stage_init_ind, stage_iter, ex_ball_size, n_samples, next_len, cont
    
    def set_x(self, code, substitutes):
        self.len_seq = len(code)
        self.word_substitution_cache = [[] for _ in range(self.len_seq)]
        for ind in range(self.len_seq):
            # transformed_texts = self.transformer(self.x0, original_text=self.x0, indices_to_modify=[ind])
            # 将原始的code token放入候选token列表中
            self.word_substitution_cache[ind].append(code[ind])
            # for key in list(substitutes.keys()):
            for key in list(self.names_positions_dict.keys()):
                # 不将重复的token放入候选token列表中
                # if key == code[ind]:
                if key == code[ind] and ind not in self.names_positions_dict[key][1:]:
                    self.word_substitution_cache[ind].extend(substitutes[key])
                else:
                    continue
        
        # print("word_substitution_cache",self.word_substitution_cache)
        # self.n_vertices 是每个词的替换词个数
        self.n_vertices = [len(w_candids) for w_candids in self.word_substitution_cache]
        # self.target_indices 是每个词的替换词个数大于1的词的索引
        self.target_indices = [ind for ind in range(self.len_seq) if self.n_vertices[ind]>1]
        # print("target_indices",self.target_indices)

    def is_success(self, new_code_tokens):
        replaced_words = self.replaced_words(self.code_tokens_1, new_code_tokens)
        # print(replaced_words)
        temp_code = get_example_batch(self.orig_code_1, replaced_words, 'c')
        new_feature = convert_code_to_features(temp_code, self.tokenizer, self.true_label, self.args)
        logits, preds = self.model.get_results(CodeDataset([new_feature]), self.args.eval_batch_size)
        # print(logits)
        # print(preds)
        # print(self.true_label)
        if preds[0] != self.true_label:
            print("攻击之后模型的输出：", preds[0])
            return 1
        else:
            return 0
    
    def adv_code(self, old_code, new_code_tokens):
        replaced_words = self.replaced_words(self.code_tokens_1, new_code_tokens)
        # print(replaced_words)
        temp_code = get_example_batch(self.orig_code_1, replaced_words, 'c')
        return temp_code
    
    def replaced_words(self, old_code_tokens, new_code_tokens):
        differences = [(a1, a2) for a1, a2 in zip(old_code_tokens, new_code_tokens) if a1 != a2]
        return dict(differences)
      
    def bayes_attack(self, example, substitutes, code, query_times, logits, example_start_time):
        self.true_label = example[1].item()
        self.query_times = query_times
        # print("query_times",self.query_times)
        self.orig_logits = logits
        self.orig_code_1 = deepcopy(code)

        self.start_time = example_start_time

        _, self.code_tokens_1 = get_identifiers(code, 'c')
        processed_code_1 = " ".join(deepcopy(self.code_tokens_1))
        words, _, _ = _tokenize(processed_code_1, self.tokenizer)

        variable_names = list(substitutes.keys())
        variable_names = self.filter_identifier(code, variable_names)
        # print(variable_names)

        self.names_positions_dict = get_identifier_posistions_from_code(self.code_tokens_1, variable_names)

        self.set_x(self.code_tokens_1, substitutes)

        # self.orig_X, self.orig_Y = example[0].unsqueeze(0), example[1].unsqueeze(0)
        # torch.zeros(1, len(x0.words))

        self.orig_X = torch.zeros(1, len(self.code_tokens_1))

        # 设置查询限制
        self.query_budget = self.get_query_budget(self.n_vertices, baseline=self.max_budget_key_type)

        # TODO:self.orig_X的长度如何处理
        self.hb = HistoryBoard(orig_X = self.orig_X, n_vertices=self.n_vertices)
        self.eff_len = self.hb.eff_len_seq

        # print("query budget is ", self.query_budget) 
        # TODO: 重写 eval_and_add_datum 函数
        self.eval_and_add_datum(self.orig_X)
        if self.check_query_const() or len(self.hb.target_indices)==0: return None, 0, None
        
        # Initialize surrogate model wrapper.
        self.surrogate_model = MyGPModel(fit_iter=self.fit_iter)

        D_0, center_seq, center_ind, LOCAL_OPTIMUM, stage = self.init_before_loop()

        while self.BLOCK_QUEUE:
            if self.BLOCK_QUEUE[0][0] != stage:
                self.BLOCK_QUEUE = self.update_queue(self.BLOCK_QUEUE, self.INDEX_DICT)
                stage += 1
            if not self.BLOCK_QUEUE: break

            KEY = self.BLOCK_QUEUE.pop(0)
            opt_indices, fix_indices, stage_init_ind, stage_iter, ex_ball_size, n_samples, next_len, cont = self.init_in_loop(KEY,center_ind)
            if cont: continue

            # Exploration.
            prev_qr = len(self.hb.eval_Y)
            stage_call = 0 

            if self.check_query_const(): break
            stage_call, fX, X, fidx = self.exploration_ball_with_indices(center_seq=center_seq,n_samples=n_samples,ball_size=ex_ball_size,stage_call=stage_call, opt_indices=opt_indices, KEY=KEY, stage_init_ind=stage_init_ind)

            # print('===========')
            # print('stage_call',stage_call)
            if stage_call == -1:
                new_code_tokens = self.seq2code(X)
                is_success = self.is_success(new_code_tokens)
                if is_success == 1:
                    return self.adv_code(code, new_code_tokens), is_success, self.replaced_words(self.code_tokens_1, new_code_tokens)
                else:
                    return None, 0, None
            if len(self.hb.eval_Y) == prev_qr:
                if KEY[0] < self.max_loop: 
                    new = (KEY[0]+1, KEY[1])
                    self.BLOCK_QUEUE.append(new)
                    self.INDEX_DICT[new] = deepcopy(opt_indices[:next_len])
                    self.HISTORY_DICT[KEY].extend([ int(i) for i in range(stage_init_ind, len(self.hb.eval_X))])
                continue
            if self.check_query_const(): break

            # union parent histories and prev best ind (center ind)
            parent_history = set(D_0)
            for n in range(KEY[0]):
                key = (int(n),KEY[1])
                parent_history |= set(self.HISTORY_DICT[key])

                inds = self.HISTORY_DICT[key]
                loc_fix_indices = list( set(list(range(self.eff_len))) - set(deepcopy(self.INDEX_DICT[key])) )
                if loc_fix_indices:
                    history = self.hb.eval_X_reduced[inds]
                    uniq = torch.unique(history[:,loc_fix_indices],dim=0)
                    assert uniq.shape[0] == 1, f'{uniq.shape},{uniq[:,:5]}'

            parent_history.add(center_ind)
            parent_history = list(parent_history)
            # print('parent_history',parent_history)

            if self.use_sod:
                parent_history = self.subset_of_dataset(parent_history, stage_iter)
                assert len(parent_history) <= stage_iter, f'something wrong {stage_iter}, {len(parent_history)}'
            
            # Exploitation.
            num_candids = 10
            init_cent_indiced = deepcopy(self.hb.reduce_seq(center_seq))

            count = 0
            prev_size = len(self.hb.eval_Y)
            iter_patience = 5
            while stage_call < stage_iter and iter_patience:
                self.clean_memory_cache()
                if prev_size == len(self.hb.eval_Y):
                    iter_patience -= 1
                else:
                    iter_patience = 5
                    prev_size = len(self.hb.eval_Y)
                self.surrogate_model.fit_partial(self.hb, list(range(self.eff_len)), stage_init_ind, prev_indices=parent_history) 

                if count  % self.update_step == 0:
                    best_inds = self.hb.topk_in_history_with_fixed_indices(len(self.hb.eval_Y), init_cent_indiced, fix_indices)[3]          
                    for best_ind in best_inds:
                        if not (best_ind in LOCAL_OPTIMUM[tuple(opt_indices)]):
                            break

                    best_val = self.hb.eval_Y[best_inds[0]][0].item()
                    best_seq = [self.hb.eval_X[best_ind]]
                    reference = best_val
                    best_indiced = self.hb.reduce_seqs(best_seq)
                    
                    best_seqs = self.find_greedy_init_with_indices_v2(cand_indices=best_indiced, max_radius=self.eff_len, num_candids=num_candids, reference=reference)
                best_candidates = acquisition_maximization_with_indices(best_seqs, opt_indices=opt_indices, batch_size=self.batch_size, stage=self.eff_len, hb=self.hb, surrogate_model=self.surrogate_model, reference=reference, dpp_type=self.dpp_type, acq_with_opt_indices=False)

                if type(best_candidates) == type(None):
                    LOCAL_OPTIMUM[tuple(opt_indices)].append(best_ind)
                    rand_indices = self.hb.nbd_sampler(best_indiced, self.batch_size, 2, 1, fix_indices=fix_indices)
                    best_candidates = [self.hb.seq_by_indices(inds) for inds in random.sample(list(rand_indices), self.batch_size)]                        

                if stage_call >= stage_iter or self.check_query_const(): break

                prev_len = len(self.hb.eval_Y) 
                if self.eval_and_add_data(best_candidates):
                    self.HISTORY_DICT[KEY].extend([ int(i) for i in range(stage_init_ind, len(self.hb.eval_X))])
                    
                    final_init_ind = len(self.hb.eval_Y)-1
                    # print("final_init_ind",final_init_ind)
                    X = self.final_exploitation(self.hb.eval_X[-1], final_init_ind)
                    new_code_tokens = self.seq2code(X)
                    is_success = self.is_success(new_code_tokens)
                    if is_success == 1:
                        return self.adv_code(code, new_code_tokens), is_success, self.replaced_words(self.code_tokens_1, new_code_tokens)
                    else:
                        return None, 0, None
                stage_call += len(self.hb.eval_Y) - prev_len
                if stage_call >= stage_iter or self.check_query_const(): break
                count += 1

            best_inds = self.hb.topk_in_history(len(self.hb.eval_Y))[3] 
            
            for center_ind in best_inds:
                if not (center_ind in LOCAL_OPTIMUM[tuple(opt_indices)]):
                    center_ind = int(center_ind)
                    break
            center_seq = self.hb.eval_X[center_ind]
            if self.check_query_const(): break

            if KEY[0] < self.max_loop: 
                new = (KEY[0]+1, KEY[1])
                self.BLOCK_QUEUE.append(new)
                self.INDEX_DICT[new] = deepcopy(opt_indices[:next_len])
                self.HISTORY_DICT[KEY].extend([ int(i) for i in range(stage_init_ind, len(self.hb.eval_X))])
            continue
                  
        if self.query_budget < float('inf'):
            Ys = self.hb.eval_Y[:self.query_budget]
        else:
            Ys = self.hb.eval_Y
        max_ind = torch.argmax(Ys)
        new_code_tokens = self.seq2code(self.hb.eval_X[max_ind])
        is_success = self.is_success(new_code_tokens)
        if is_success == 1:
            return self.adv_code(code, new_code_tokens), is_success, self.replaced_words(self.code_tokens_1, new_code_tokens)
        else:
            return None, 0, None 

    def subset_of_dataset(self, history, num_samples):
        if len(history) <= num_samples:
            return history
        else:
            history_X_reduced = self.hb.eval_X_reduced[history].numpy()
            _, selected_indices_ = kmeans_pp(history_X_reduced, num_samples, dist='hamming')
            history = [history[ind] for ind in selected_indices_]
            return history

    def greedy_step(self, seq, index_order, is_shuffle=True):
        best_seq = seq
        best_indiced = self.hb.reduce_seq(seq)[0]
        best_score = self.eval_and_add_datum(best_seq, return_scores=True)

        order = deepcopy(index_order)
        while True:
            prev_best_seq = best_seq
            if is_shuffle:
                random.shuffle(order)
            for ind in order:
                nv = self.hb.reduced_n_vertices[ind]
                candids = [deepcopy(best_indiced) for i in range(nv)] 
                for i, cand in enumerate(candids):
                    cand[ind] = i 
                candid_seqs = [self.hb.seq_by_indices(cand) for cand in candids]
                candid_scores = self.eval_and_add_data(candid_seqs, return_scores=True)
                max_ind = np.argmax(candid_scores)
                if candid_scores[max_ind]>=0: return candid_seqs[max_ind], candid_scores[max_ind], True
                if best_score < candid_scores[max_ind]:
                    best_seq = candid_seqs[max_ind]
                    best_indiced = self.hb.reduce_seq(best_seq)[0]
                    best_score = candid_scores[max_ind]
            if torch.all(prev_best_seq == best_seq):
                break 
        return best_seq, best_score, False

    def ind_score(self, ind, beta, stage):
        score = sum([float(beta[i]) for i in ind]) + 1e6 * stage
        return score

    def update_queue(self, Q, I):
        if Q[0][0] == 0:
            if len(Q)==1:
                return Q
            else:
                inds_list = [I[KEY] for KEY in Q]
                order = self.get_initial_block_order(inds_list)
                Q_ = [Q[ind] for ind in order]
                return Q_
        if len(Q)==1:
            return Q
        self.fit_surrogate_model_by_block_history()

        beta = 1/(self.surrogate_model.model.covar_module.base_kernel.lengthscale[0].detach().cpu()+1e-6)
    
        for KEY in Q:
            if not I[KEY]:
                Q.remove(KEY)

        def f(KEY):
            return self.ind_score(I[KEY],beta,KEY[0])

        Q_ = sorted(Q,key=f)
        return Q_
    
    def get_index_order_for_block_decomposition(self, rank_policy='straight'):
        if rank_policy == 'straight':
            index_order = list(range(len(self.hb.target_indices)))
        elif rank_policy == 'beta':
            beta = 1/(self.surrogate_model.model.covar_module.base_kernel.lengthscale.detach().cpu()+1e-6)
            index_order = torch.argsort(beta)[0]
        index_order = [int(i) for i in index_order]
        return index_order

    def exploration_ball_with_indices(self, center_seq, n_samples, ball_size, stage_call, opt_indices, KEY, stage_init_ind):
        if n_samples == 0:
            return stage_call, None, None
        fix_indices = list(set(list(range(len(self.hb.target_indices)))) - set(opt_indices))
        prev_len = self.hb.eval_Y.shape[0]
        rand_candidates = self.hb.sample_ball_candidates_from_seq(center_seq, n_samples=n_samples, ball_size=ball_size, fix_indices=fix_indices)

        for rand_candidate in rand_candidates:
            if self.eval_and_add_datum(rand_candidate):
                self.HISTORY_DICT[KEY].extend([ int(i) for i in range(stage_init_ind, len(self.hb.eval_X))])
                fX = self.hb.eval_X[-1]
                best_candidate = self.hb.eval_X[-1]
                fidx = len(self.hb.eval_Y)-1
                return -1, fX, self.final_exploitation(best_candidate, fidx), fidx

        if self.check_query_const():
            stage_call += self.hb.eval_Y.shape[0] - prev_len
            return stage_call, None, None, None

        # If any query were not evaluated, hardly sample non orig examples.
        if self.hb.eval_Y.shape[0] == prev_len:
            # print('*******************')
            center_indiced = self.hb.reduce_seq(center_seq)[0]
            rand_indiced = copy.deepcopy(center_indiced)
            for ind in opt_indices:
                rand_indiced[ind] = int(random.sample(list(range(self.hb.reduced_n_vertices[ind]-1)), 1)[0] + 1)
            rand_candidate = self.hb.seq_by_indices(rand_indiced)
            if self.eval_and_add_datum(rand_candidate):
                self.HISTORY_DICT[KEY].extend([ int(i) for i in range(stage_init_ind, len(self.hb.eval_X))])
                fX = self.hb.eval_X[-1]
                fidx = len(self.hb.eval_Y)-1
                return -1, fX, self.final_exploitation(rand_candidate, fidx), fidx
        stage_call += self.hb.eval_Y.shape[0] - prev_len
        return stage_call, None, None, None

    def find_greedy_init_with_indices_v2(self, cand_indices, max_radius, num_candids, reference=None):
        # calculate acquisition
        if reference is None:
            _, reference, best_ind = self.hb.best_of_hamming_orig(distance=max_radius)
        ei = self.surrogate_model.acquisition(cand_indices, bias=reference)
        topk_values, topk_indices = torch.topk(ei, min(len(ei),num_candids))
        center_indices_list = [cand_indices[idx].view(1,-1) for idx in topk_indices]
        return center_indices_list
    
    def eval_and_add_datum(self, seq, return_scores=False):
        score = self.get_score(seq)
        # print("score",score)
        if type(score) == type(None): return 0
        if not self.hb.is_seq_in_hb(seq):
            self.hb.add_datum(seq, score)
        if return_scores: return score
        if score >= 0:
            return 1
        else: 
            return 0
    
    def eval_and_add_data(self, seqs, return_scores=False):
        scores = self.get_scores(seqs)
        # print(scores)
        for seq, score in zip(seqs, scores):
            if type(score) == type(None): 
                if return_scores: return scores
                else: return 0
            if not self.hb.is_seq_in_hb(seq):
                self.hb.add_datum(seq, score)
            if score >= 0 and not return_scores:
                return 1
        if return_scores: return scores
        return 0

    def eval_and_add_data_best_ind(self, seqs, cur_seq, best_ind, tmp, tmp_modif, patience):
        scores = self.get_scores(seqs)
        # print(scores)
        for seq, score in zip(seqs,scores):
            if type(score) == type(None): 
                return cur_seq, best_ind, patience
            if not self.hb.is_seq_in_hb(seq):
                self.hb.add_datum(seq, score)
                if score >= 0:
                    modif = self.hb._hamming(self.orig_X, seq)
                    if tmp < self.hb.eval_Y[-1].item() or (tmp == self.hb.eval_Y[-1].item() and tmp_modif > modif):
                        tmp = self.hb.eval_Y[-1].item()
                        tmp_modif = modif
                        cur_seq = seq
                        best_ind = len(self.hb.eval_X) - 1 
            patience -= 1
        
        return cur_seq, best_ind, patience

    def final_exploitation(self, seq, ind):
        seq = seq.view(1,-1)
        if 'v3' in self.post_opt:
            return self.final_exploitation_v3(seq, ind)
        else:
            return self.hb.eval_X[ind].view(1,-1)
    
    def final_exploitation_v3(self, seq, ind, forced_inds=[]):
        cur_seq = seq
        best_ind = ind
        self.eff_len = self.hb.eff_len_seq
        max_patience = self.max_patience
        prev_radius = self.hb._hamming(self.orig_X, cur_seq)
        patience = max_patience

        i=0
        nbd_size = 2
        opt_indices = [idx for idx in range(self.eff_len)] 
        whole_indices = [idx for idx in range(self.eff_len)] 
        _, sum_history = self.fit_surrogate_model_by_block_history(forced_inds)
        init_idx = len(self.hb.eval_Y)

        
        # del it.
        setattr(self.hb,"expl", [])
        while True:
            self.clean_memory_cache()
            self.hb.expl.append((self.hb._hamming(self.orig_X, cur_seq)/self.hb.len_seq, patience, self.model.query - self.query_times))
            max_radius = self.hb._hamming(self.orig_X, cur_seq) - 1
            if prev_radius == max_radius:
                if patience <= 0:
                    break
            else:
                patience = max_patience
                prev_radius = max_radius
                nbd_size = 2
            # del it.
            if max_radius == 0:
                return self.hb.eval_X[best_ind].view(1,-1)
            
            self.surrogate_model.fit_partial(self.hb, whole_indices, init_idx, sum_history)
            best_candidate = cur_seq

            # best in ball seq
            bib_seq, bib_score, _ = self.hb.best_of_hamming_orig(distance=max_radius)

            best_indiced = self.hb.reduce_seq(best_candidate)
            bib_indiced = self.hb.reduce_seq(bib_seq)  
            orig_indiced = self.hb.reduce_seq(self.orig_X)
            rand_indices = self.hb.subset_sampler(best_indiced, 300, nbd_size)

            cand_indices = torch.cat([orig_indiced.view(1,-1), best_indiced.view(1,-1), bib_indiced.view(1,-1), rand_indices], dim=0)
            cand_indices = torch.unique(cand_indices.long(),dim=0).float()
            center_candidates = self.find_greedy_init_with_indices_v2(cand_indices, max_radius, num_candids=self.batch_size, reference=0.0)
            # print("center_candidates",center_candidates)  
            reference = self.hb.eval_Y[best_ind].item()
            best_candidates = acquisition_maximization_with_indices(center_candidates, opt_indices=opt_indices, batch_size=self.batch_size, stage=max_radius-1, hb=self.hb, surrogate_model=self.surrogate_model, reference=reference, dpp_type=self.dpp_type, acq_with_opt_indices=False)
            if best_candidates == None:
                if max_radius + 1 == nbd_size:
                    break
                else:
                    nbd_size += 1
                    continue

            tmp = 0.0
            tmp_modif = self.eff_len
            if self.check_query_const(): break

            cur_seq, best_ind, patience = self.eval_and_add_data_best_ind(best_candidates, cur_seq, best_ind, tmp, tmp_modif, patience)
            
            if self.check_query_const() or patience <= 0: break
            i += 1
        return self.hb.eval_X[best_ind].view(1,-1)

    def block_history_dict(self, forced_inds=[]):
        bhl = defaultdict(list)
        for KEY, INDEX in self.INDEX_DICT.items():
            HISTORY = self.HISTORY_DICT[KEY]
            bhl[KEY[1]].extend(HISTORY)
        for key in bhl:
            bhl[key] = list(dict.fromkeys(bhl[key]))
            opt_indices = list(range(key*self.block_num,min((key+1)*self.block_num,len(self.hb.reduced_n_vertices))))
            # if self.eff_len < 3:
            #     opt_indices = list(range(key*self.block_num,min((key+1)*self.block_num,len(self.hb.reduced_n_vertices))))
            # else:
            #     opt_indices = self.idx[key]
            num_samples = sum([self.hb.reduced_n_vertices[ind]-1 for ind in opt_indices]) 
            bhl[key] = self.subset_of_dataset(bhl[key],num_samples)
        sum_history = copy.deepcopy(forced_inds)
        for key, l in bhl.items():
            sum_history.extend(l)
        return bhl, sum_history
    
    def fit_surrogate_model_by_block_history(self, forced_inds=[]):
        self.eff_len = len(self.hb.target_indices)
        bhl, sum_history = self.block_history_dict(forced_inds)
        whole_indices = [idx for idx in range(self.eff_len)] # nonzero indices
        self.surrogate_model.fit_partial(self.hb, whole_indices, len(self.hb.eval_Y), sum_history)
        return bhl, sum_history

    def clean_memory_cache(self,debug=False):
        # Clear garbage cache for memory.
        if self.memory_count == 10:
            gc.collect()
            torch.cuda.empty_cache()
            self.memory_count = 0
        else:
            self.memory_count += 1   
        if debug:
            print(torch.cuda.memory_allocated(0))
