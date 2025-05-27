import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import SequentialSampler, DataLoader
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig, RobertaForSequenceClassification
import numpy as np
import sys

sys.path.append("../../../")
from language_parser.run_parser import get_identifiers_from_tokens


def calculate_noise(model, inputs_ids, attentions, args):
    if args.model_type == 'codebert':
        embeddings = model.encoder.embeddings.word_embeddings(inputs_ids)
    elif args.model_type == 'graphcodebert':
        embeddings = model.encoder.roberta.embeddings.word_embeddings(inputs_ids)
    elif args.model_type == 'unixcoder':
        embeddings = model.encoder.embeddings.word_embeddings(inputs_ids)
    elif args.model_type == 'codet5' or args.model_type == 'codet5p':
        embeddings = model.encoder.encoder.embed_tokens(input_ids)

    bsz, seqlen, hsz = embeddings.shape
    noise = torch.zeros_like(embeddings).to(args.device)  # (bsz, seqlen, hsz)

    code_tokens = [model.tokenizer.convert_ids_to_tokens(inputs_ids[i]) for i in range(bsz)]

    identifiers = [get_identifiers_from_tokens(code_token, args.language_type) for code_token in code_tokens]

    token_index_list = []  # (bsz, token_num)
    token_index_map_list = []  # (bsz, token_num)
    attention_rank_list = []  # (bsz, token_num)，存储对应 token 位置的 attention 排名

    for batch_id in range(bsz):
        # 获取每个标识符的位置
        token_index_list.append([])
        token_index_map_list.append({})
        attention_rank_list.append({})
        temp_code_token_list = code_tokens[batch_id]
        code_token_list = []
        for code_token in temp_code_token_list:
            if len(code_token) > 1 and code_token[0] == 'Ġ':
                code_token_list.append(code_token[1:])
            else:
                code_token_list.append(code_token)

        identifiers_list = identifiers[batch_id]
        identifiers_list.remove('Ġ') if 'Ġ' in identifiers_list else None

        for identifier in identifiers_list:
            indexex = [i for i in range(len(code_token_list) - 1) if code_token_list[i] == identifier]
            token_index_map_list[batch_id][identifier] = indexex

        for identifier in identifiers_list:
            attention_val = 0
            for layer in attentions[batch_id]:
                temp_val = 0
                for idn in token_index_map_list[batch_id][identifier]:
                    temp_val += layer[idn][idn].item()
                attention_val += (temp_val / len(token_index_map_list[batch_id][identifier]))
            for idn in token_index_map_list[batch_id][identifier]:
                attention_rank_list[batch_id][idn] = attention_val

        attention_rank_list[batch_id] = sorted(attention_rank_list[batch_id].items(), key=lambda x: x[1], reverse=True)
    random_noise = torch.randn(hsz).to(args.device)
    for batch_id in range(bsz):
        attentions_values = [t[1] for t in attention_rank_list[batch_id]]
        if len(attentions_values) > 0:
            average = sum(attentions_values) / len(attentions_values)
            for id, weight in attention_rank_list[batch_id]:
                cur_noise = random_noise * (weight * 1 / average + 1)
                noise[batch_id][id] += cur_noise
    return noise


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = x.reshape(-1, x.size(-1) * 2)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaClassificationHead_twoContact(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(2 * config.hidden_size * 2, 2 * config.hidden_size)
        self.dropout = nn.Dropout(2 * config.hidden_dropout_prob)
        self.out_proj = nn.Linear(2 * config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = x.reshape(-1, x.size(-1) * 2)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class CodeT5RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = features.reshape(-1, features.size(-1) * 2)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.out_proj(x)
        return x


class CodeT5RobertaClassificationHead_twoContact(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(2 * config.hidden_size * 2, 2 * config.hidden_size)
        self.out_proj = nn.Linear(2 * config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = features.reshape(-1, features.size(-1) * 2)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.out_proj(x)
        return x


class CodeBERT(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(CodeBERT, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args
        self.query = 0

    def forward(self, input_ids=None, labels=None):
        # print("input_ids.size()", input_ids.size())
        input_ids = input_ids.view(-1, self.args.block_size)
        # input_ids = input_ids.view(-1, self.args.block_size - 2)  # add by lsr
        opt = self.encoder(input_ids=input_ids, attention_mask=input_ids.ne(1), output_attentions=output_attentions)
        outputs = opt[0]
        logits = self.classifier(outputs)
        prob = F.softmax(logits)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob

    def get_results(self, dataset, batch_size, threshold=0.5):
        '''Given a dataset, return probabilities and labels.'''
        self.query += len(dataset)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size, num_workers=0,
                                     pin_memory=False)

        ## Evaluate Model
        eval_loss = 0.0
        self.eval()
        logits = []
        for batch in eval_dataloader:
            inputs = batch[0].to("cuda")
            label = batch[1].to("cuda")
            with torch.no_grad():
                lm_loss, logit = self.forward(inputs, label)
                eval_loss += lm_loss.mean().item()
                logits.append(logit.cpu().numpy())
        logits = np.concatenate(logits, 0)
        probs = logits
        pred_labels = [0 if first_softmax > threshold else 1 for first_softmax in logits[:, 0]]

        return probs, pred_labels


class GraphCodeBERT(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(GraphCodeBERT, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args
        self.query = 0

    def forward(self, inputs_ids_1, position_idx_1, attn_mask_1, inputs_ids_2, position_idx_2, attn_mask_2,
                labels=None):
        bs, l = inputs_ids_1.size()
        inputs_ids = torch.cat((inputs_ids_1.unsqueeze(1), inputs_ids_2.unsqueeze(1)), 1).view(bs * 2, l)
        position_idx = torch.cat((position_idx_1.unsqueeze(1), position_idx_2.unsqueeze(1)), 1).view(bs * 2, l)
        attn_mask = torch.cat((attn_mask_1.unsqueeze(1), attn_mask_2.unsqueeze(1)), 1).view(bs * 2, l, l)

        # embedding
        nodes_mask = position_idx.eq(0)
        token_mask = position_idx.ge(2)
        inputs_embeddings = self.encoder.roberta.embeddings.word_embeddings(inputs_ids)
        nodes_to_token_mask = nodes_mask[:, :, None] & token_mask[:, None, :] & attn_mask
        nodes_to_token_mask = nodes_to_token_mask / (nodes_to_token_mask.sum(-1) + 1e-10)[:, :, None]
        avg_embeddings = torch.einsum("abc,acd->abd", nodes_to_token_mask, inputs_embeddings)
        inputs_embeddings = inputs_embeddings * (~nodes_mask)[:, :, None] + avg_embeddings * nodes_mask[:, :, None]
        opt = self.encoder.roberta(inputs_embeds=inputs_embeddings, attention_mask=attn_mask,
                                   position_ids=position_idx)
        outputs = opt[0]
        logits = self.classifier(outputs)
        prob = F.softmax(logits)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob

    def get_results(self, dataset, batch_size, threshold=0.5):
        '''Given a dataset, return probabilities and labels.'''
        self.query += len(dataset)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size, num_workers=0,
                                     pin_memory=False)
        self.eval()
        logits = []
        for batch in eval_dataloader:
            (inputs_ids_1, position_idx_1, attn_mask_1,
             inputs_ids_2, position_idx_2, attn_mask_2,
             label) = [x.to("cuda") for x in batch]

            with torch.no_grad():
                lm_loss, logit = self.forward(inputs_ids_1, position_idx_1, attn_mask_1, inputs_ids_2,
                                              position_idx_2, attn_mask_2, label)
                logits.append(logit.cpu().numpy())

        logits = np.concatenate(logits, 0)
        probs = logits
        pred_labels = [0 if first_softmax > threshold else 1 for first_softmax in logits[:, 0]]

        return probs, pred_labels


class CodeT5(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(CodeT5, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = CodeT5RobertaClassificationHead(config)
        self.args = args
        self.query = 0

    def forward(self, input_ids=None, labels=None):
        input_ids = input_ids.view(-1, self.args.block_size)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                               labels=input_ids, decoder_attention_mask=attention_mask,
                               output_hidden_states=True)
        opt = outputs
        hidden_states = outputs['decoder_hidden_states'][-1]
        eos_mask = input_ids.eq(self.config.eos_token_id)
        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        outputs = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                                  hidden_states.size(-1))[:, -1, :]

        logits = self.classifier(outputs)
        prob = F.softmax(logits)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob

    def get_results(self, dataset, batch_size, threshold=0.5):
        '''Given a dataset, return probabilities and labels.'''
        self.query += len(dataset)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size, num_workers=0,
                                     pin_memory=False)
        eval_loss = 0.0
        self.eval()
        logits = []

        for batch in eval_dataloader:
            inputs = batch[0].to("cuda")
            label = batch[1].to("cuda")
            with torch.no_grad():
                lm_loss, logit = self.forward(inputs, label)
                eval_loss += lm_loss.mean().item()
                logits.append(logit.cpu().numpy())

        logits = np.concatenate(logits, 0)

        probs = logits
        pred_labels = [0 if first_softmax > threshold else 1 for first_softmax in logits[:, 0]]

        return probs, pred_labels


class CodeBERTnoise(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(CodeBERTnoise, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args
        self.query = 0

    def forward(self, input_ids=None, labels=None, noise=None):
        padding_idx = self.encoder.embeddings.padding_idx
        mask = input_ids.ne(padding_idx).int()
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask)) * mask
        position_ids = incremental_indices.long() + padding_idx
        # print("input_ids.size()", input_ids.size())
        input_ids = input_ids.view(-1, self.args.block_size)
        # input_ids = input_ids.view(-1, self.args.block_size - 2)  # add by lsr
        inputs_embeddings = self.encoder.embeddings.word_embeddings(input_ids)
        if noise is not None:
            inputs_embeddings += noise
            # noise = noise.view(-1, self.args.block_size, noise.size(-1))
            # inputs_embeddings += noise[:, :inputs_embeddings.size(1), :]  # 针对克隆
        opt = self.encoder(inputs_embeds=inputs_embeddings, attention_mask=input_ids.ne(1), output_attentions=True)
        outputs = opt[0]
        logits = self.classifier(outputs)
        prob = F.softmax(logits)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob, outputs, opt.attentions[0]
        else:
            return prob, logits

    def get_outputs(self, input_ids=None, labels=None, noise=None):
        padding_idx = self.encoder.embeddings.padding_idx
        mask = input_ids.ne(padding_idx).int()
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask)) * mask
        position_ids = incremental_indices.long() + padding_idx
        # print("input_ids.size()", input_ids.size())
        input_ids = input_ids.view(-1, self.args.block_size)
        # input_ids = input_ids.view(-1, self.args.block_size - 2)  # add by lsr
        inputs_embeddings = self.encoder.embeddings.word_embeddings(input_ids)
        if noise is not None:
            inputs_embeddings += noise
            # noise = noise.view(-1, self.args.block_size, noise.size(-1))
            # inputs_embeddings += noise[:, :inputs_embeddings.size(1), :]  # 针对克隆
        # opt = self.encoder(inputs_embeds=inputs_embeddings, attention_mask=input_ids.ne(1), position_ids=position_ids,
        #                    output_attentions=True)  # 我怀疑之前的报错可以通过修改这一行解决，先待定
        opt = self.encoder(inputs_embeds=inputs_embeddings, attention_mask=input_ids.ne(1),output_attentions=True)  # 我怀疑之前的报错可以通过修改这一行解决，先待定
        outputs = opt[0]
        return outputs

    def get_results(self, dataset, batch_size, new_infer=False, threshold=0.5):
        '''Given a dataset, return probabilities and labels.'''
        self.query += len(dataset)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size, num_workers=0,
                                     pin_memory=False)

        ## Evaluate Model
        eval_loss = 0.0
        self.eval()
        logits = []
        for batch in eval_dataloader:
            inputs = batch[0].to("cuda")
            label = batch[1].to("cuda")
            if new_infer:
                _, _, _, attentions = self.forward(inputs, label)
                noise = calculate_noise(self, inputs, attentions, self.args)
            with torch.no_grad():
                lm_loss, logit, _, _ = self.forward(inputs, label, noise=noise if new_infer else None)
                eval_loss += lm_loss.mean().item()
                logits.append(logit.cpu().numpy())
        logits = np.concatenate(logits, 0)
        probs = logits
        pred_labels = [0 if first_softmax > threshold else 1 for first_softmax in logits[:, 0]]

        return probs, pred_labels


class GraphCodeBERTnoise(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(GraphCodeBERTnoise, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args
        self.query = 0
        self.config.output_attentions = True

    def forward(self, inputs_ids_1, position_idx_1, attn_mask_1, inputs_ids_2, position_idx_2, attn_mask_2,
                labels=None, noise=None):
        bs, l = inputs_ids_1.size()
        inputs_ids = torch.cat((inputs_ids_1.unsqueeze(1), inputs_ids_2.unsqueeze(1)), 1).view(bs * 2, l)
        position_idx = torch.cat((position_idx_1.unsqueeze(1), position_idx_2.unsqueeze(1)), 1).view(bs * 2, l)
        attn_mask = torch.cat((attn_mask_1.unsqueeze(1), attn_mask_2.unsqueeze(1)), 1).view(bs * 2, l, l)

        # embedding
        nodes_mask = position_idx.eq(0)
        token_mask = position_idx.ge(2)
        inputs_embeddings = self.encoder.roberta.embeddings.word_embeddings(inputs_ids)
        nodes_to_token_mask = nodes_mask[:, :, None] & token_mask[:, None, :] & attn_mask
        nodes_to_token_mask = nodes_to_token_mask / (nodes_to_token_mask.sum(-1) + 1e-10)[:, :, None]
        avg_embeddings = torch.einsum("abc,acd->abd", nodes_to_token_mask, inputs_embeddings)
        inputs_embeddings = inputs_embeddings * (~nodes_mask)[:, :, None] + avg_embeddings * nodes_mask[:, :, None]

        if noise is not None:
            inputs_embeddings += noise

        opt = self.encoder.roberta(inputs_embeds=inputs_embeddings, attention_mask=attn_mask,
                                   position_ids=position_idx,output_attentions=True)
        outputs = opt[0]
        logits = self.classifier(outputs)
        prob = F.softmax(logits)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob,outputs,opt.attentions[0]
        else:
            return prob,logits

    def get_outputs(self, inputs_ids_1, position_idx_1, attn_mask_1, inputs_ids_2, position_idx_2, attn_mask_2,
                labels=None, noise=None):
        bs, l = inputs_ids_1.size()
        inputs_ids = torch.cat((inputs_ids_1.unsqueeze(1), inputs_ids_2.unsqueeze(1)), 1).view(bs * 2, l)
        position_idx = torch.cat((position_idx_1.unsqueeze(1), position_idx_2.unsqueeze(1)), 1).view(bs * 2, l)
        attn_mask = torch.cat((attn_mask_1.unsqueeze(1), attn_mask_2.unsqueeze(1)), 1).view(bs * 2, l, l)

        # embedding
        nodes_mask = position_idx.eq(0)
        token_mask = position_idx.ge(2)
        inputs_embeddings = self.encoder.roberta.embeddings.word_embeddings(inputs_ids)
        nodes_to_token_mask = nodes_mask[:, :, None] & token_mask[:, None, :] & attn_mask
        nodes_to_token_mask = nodes_to_token_mask / (nodes_to_token_mask.sum(-1) + 1e-10)[:, :, None]
        avg_embeddings = torch.einsum("abc,acd->abd", nodes_to_token_mask, inputs_embeddings)
        inputs_embeddings = inputs_embeddings * (~nodes_mask)[:, :, None] + avg_embeddings * nodes_mask[:, :, None]

        if noise is not None:
            inputs_embeddings += noise

        opt = self.encoder.roberta(inputs_embeds=inputs_embeddings, attention_mask=attn_mask,
                                   position_ids=position_idx, output_attentions=True)
        outputs = opt[0]

        return outputs

    def get_results(self, dataset, batch_size, new_infer=False, threshold=0.5):
        '''Given a dataset, return probabilities and labels.'''
        self.query += len(dataset)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size, num_workers=0,
                                     pin_memory=False)
        self.eval()
        logits = []
        for batch in eval_dataloader:
            (inputs_ids_1, position_idx_1, attn_mask_1,
             inputs_ids_2, position_idx_2, attn_mask_2,
             label) = [x.to("cuda") for x in batch]
            if new_infer:
                _,_,_, attentions = self.forward(inputs_ids_1, position_idx_1, attn_mask_1, inputs_ids_2,
                                              position_idx_2, attn_mask_2, label)
                bs, l = inputs_ids_1.size()
                inputs_ids = torch.cat((inputs_ids_1.unsqueeze(1), inputs_ids_2.unsqueeze(1)), 1).view(bs * 2, l)
                noise = calculate_noise(self, inputs_ids, attentions, self.args)

            with torch.no_grad():
                lm_loss, logit, _, _  = self.forward(inputs_ids_1, position_idx_1, attn_mask_1, inputs_ids_2,
                                              position_idx_2, attn_mask_2, label, noise = noise if new_infer else None)
                logits.append(logit.cpu().numpy())

        logits = np.concatenate(logits, 0)
        probs = logits
        pred_labels = [0 if first_softmax > threshold else 1 for first_softmax in logits[:, 0]]

        return probs, pred_labels


class CodeT5noise(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(CodeT5noise, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = CodeT5RobertaClassificationHead(config)
        self.args = args
        self.query = 0
        self.config.output_attentions = True

    def forward(self, input_ids=None, labels=None, noise=None):
        input_ids = input_ids.view(-1, self.args.block_size)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        # 先拿原始 embedding
        inputs_embeddings = self.encoder.encoder.embed_tokens(input_ids)

        # 加 noise
        if noise is not None:
            inputs_embeddings += noise
            # noise = noise.view(-1, self.args.block_size, noise.size(-1))
            # inputs_embeddings += noise[:, :inputs_embeddings.size(1), :]  # 针对克隆

        # outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
        #                        labels=input_ids, decoder_attention_mask=attention_mask,
        #                        output_hidden_states=True)
        # opt = outputs

        # 编码过程，注意这里用 inputs_embeds 而不是 input_ids
        outputs = self.encoder.encoder(
            inputs_embeds=inputs_embeddings,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True
        )
        hidden_states = outputs['hidden_states'][-1]
        eos_mask = input_ids.eq(self.config.eos_token_id)
        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        sequence_outputs = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                                           hidden_states.size(-1))[:, -1, :]

        logits = self.classifier(sequence_outputs)
        prob = F.softmax(logits)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob, sequence_outputs, outputs.attentions
        else:
            return prob, logits

    def get_outputs(self, input_ids=None, labels=None, noise=None):
        input_ids = input_ids.view(-1, self.args.block_size)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        # 先拿原始 embedding
        inputs_embeddings = self.encoder.encoder.embed_tokens(input_ids)

        # 加 noise
        if noise is not None:
            inputs_embeddings += noise
            # noise = noise.view(-1, self.args.block_size, noise.size(-1))
            # inputs_embeddings += noise[:, :inputs_embeddings.size(1), :]  # 针对克隆

        # outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
        #                        labels=input_ids, decoder_attention_mask=attention_mask,
        #                        output_hidden_states=True)
        # opt = outputs

        # 编码过程，注意这里用 inputs_embeds 而不是 input_ids
        outputs = self.encoder.encoder(
            inputs_embeds=inputs_embeddings,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True
        )
        hidden_states = outputs['decoder_hidden_states'][-1]
        eos_mask = input_ids.eq(self.config.eos_token_id)
        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        sequence_outputs = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                                           hidden_states.size(-1))[:, -1, :]
        return sequence_outputs

    def get_results(self, dataset, batch_size, new_infer=False, threshold=0.5):
        '''Given a dataset, return probabilities and labels.'''
        self.query += len(dataset)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size, num_workers=0,
                                     pin_memory=False)
        eval_loss = 0.0
        self.eval()
        logits = []

        for batch in eval_dataloader:
            inputs = batch[0].to("cuda")
            label = batch[1].to("cuda")
            if new_infer:
                _, _, _, attentions = self.forward(inputs_ids, attn_mask, position_idx, label)
                noise = calculate_noise(self, inputs_ids, attentions, self.args)
            with torch.no_grad():
                lm_loss, logit, _, _ = self.forward(inputs, label, noise=noise if new_infer else None)
                eval_loss += lm_loss.mean().item()
                logits.append(logit.cpu().numpy())

        logits = np.concatenate(logits, 0)

        probs = logits
        pred_labels = [0 if first_softmax > threshold else 1 for first_softmax in logits[:, 0]]

        return probs, pred_labels


class CodeBERT_twoContact(nn.Module):
    def __init__(self, config, model1, model2, args):
        super(CodeBERT_twoContact, self).__init__()
        self.args = args
        self.config = config
        self.model1 = model1
        self.model2 = model2
        self.classifier = RobertaClassificationHead_twoContact(self.config)
        self.query = 0

    def forward(self, input_ids=None, labels=None, noise=None):
        outputs1 = self.model1.get_outputs(input_ids, labels, noise)[:, 0, :]
        outputs2 = self.model2.get_outputs(input_ids, labels, noise=None)[:, 0, :]
        outputs = torch.cat((outputs1, outputs2), dim=1)
        logits = self.classifier(outputs)
        prob = F.softmax(logits)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob, logits

    def get_results(self, dataset, batch_size):
        '''
        给定example和tgt model，返回预测的label和probability
        '''
        self.query += len(dataset)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size, num_workers=0,
                                     pin_memory=False)

        ## Evaluate Model
        eval_loss = 0.0
        nb_eval_steps = 0
        self.eval()
        logits = []
        for batch in eval_dataloader:
            inputs = batch[0].to("cuda" if torch.cuda.is_available() else "cpu")
            label = batch[1].to("cuda" if torch.cuda.is_available() else "cpu")
            _, _, _, attentions = self.model1(inputs, label)
            noise = calculate_noise(self.model1, inputs, attentions, self.args)
            with torch.no_grad():
                lm_loss, logit = self.forward(inputs, label, noise)
                # 调用这个模型. 重写了反前向传播模型.
                eval_loss += lm_loss.mean().item()
                logits.append(logit.cpu().numpy())
            nb_eval_steps += 1
        logits = np.concatenate(logits, 0)

        probs = logits
        pred_labels = []
        for logit in logits:
            pred_labels.append(np.argmax(logit))

        return probs, pred_labels


class CodeT5_twoContact(nn.Module):
    def __init__(self, config, model1, model2, args):
        super(CodeT5_twoContact, self).__init__()
        self.args = args
        self.config = config
        self.model1 = model1
        self.model2 = model2
        self.classifier = CodeT5RobertaClassificationHead_twoContact(self.config)
        self.query = 0

    def forward(self, input_ids=None, labels=None, noise=None):
        # model1 和 model2 的 get_outputs 要返回最后一层 decoder_hidden_states
        outputs1 = self.model1.get_outputs(input_ids, labels, noise)  # shape: [batch, seq_len, hidden]
        outputs2 = self.model2.get_outputs(input_ids, labels, noise=None)

        if outputs1.dim() == 3:
            eos_mask = input_ids.eq(self.config.eos_token_id)
            if len(torch.unique(eos_mask.sum(1))) > 1:
                raise ValueError("All examples must have the same number of <eos> tokens.")

            rep1 = outputs1[eos_mask].view(outputs1.size(0), -1, outputs1.size(-1))[:, -1, :]
            rep2 = outputs2[eos_mask].view(outputs2.size(0), -1, outputs2.size(-1))[:, -1, :]
        elif outputs1.dim() == 2:
            rep1 = outputs1
            rep2 = outputs2
        else:
            raise ValueError(f"Unexpected outputs1 shape: {outputs1.shape}")

        concat_rep = torch.cat((rep1, rep2), dim=1)  # shape: [batch, 2 * hidden]

        logits = self.classifier(concat_rep)
        prob = F.softmax(logits, dim=-1)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob, logits

    def get_results(self, dataset, batch_size):
        '''
        给定example和tgt model，返回预测的label和probability
        '''
        self.query += len(dataset)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size, num_workers=0,
                                     pin_memory=False)

        eval_loss = 0.0
        nb_eval_steps = 0
        self.eval()
        logits = []

        for batch in eval_dataloader:
            inputs = batch[0].to("cuda" if torch.cuda.is_available() else "cpu")
            label = batch[1].to("cuda" if torch.cuda.is_available() else "cpu")

            # 用 model1 生成 noise
            _, _, _, attentions = self.model1.forward(inputs, label)
            noise = calculate_noise(self.model1, inputs, attentions, self.args)

            with torch.no_grad():
                lm_loss, logit = self.forward(inputs, label, noise)
                eval_loss += lm_loss.mean().item()
                logits.append(logit.cpu().numpy())

            nb_eval_steps += 1

        logits = np.concatenate(logits, 0)
        probs = logits
        pred_labels = [np.argmax(logit) for logit in logits]

        return probs, pred_labels
