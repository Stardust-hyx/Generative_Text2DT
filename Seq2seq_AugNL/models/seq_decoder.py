import torch
import torch.nn as nn
import random
from torch.nn import functional as F
from models.modeling_cpt import CPTConfig, CPTDecoder, CPTPretrainedModel, CPTAttention, _expand_mask
from utils.functions import scan_seq
from typing import Union
from copy import deepcopy

class SeqDecoder(CPTPretrainedModel):

    def __init__(self, config, structure_token_ids, bos_token_id=1, eos_token_id=0, pad_token_id=0):
        super().__init__(config)
        self.config = config
        self.decoder = CPTDecoder(config)
        self.dropout_layer = nn.Dropout(0.1)
        # self.dropout_layer = nn.Dropout(config.dropout)
        print(config.dropout)

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        structure_token_embeds = self.decoder.embed_tokens(torch.LongTensor(structure_token_ids)).clone()
        self.structure_token_embeds = nn.Parameter(structure_token_embeds)
        print(self.structure_token_embeds.shape)

        self.structure_token_ids = structure_token_ids
        self.src_start_index = len(structure_token_ids)

        # self.trans_triple = nn.Linear(config.d_model, config.d_model)
        # self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, tgt_seq_ids, state):
        encoder_outputs = state.encoder_output
        encoder_pad_mask = state.encoder_mask
        h_triple = state.project_triple
        triples_pad_mask = state.triples_pad_mask

        # print('encoder_outputs.shape', encoder_outputs.shape)
        # print('encoder_pad_mask.shape', encoder_pad_mask.shape)

        use_cache = not self.training

        if use_cache:
            tgt_seq_ids = tgt_seq_ids[:, -1:]

        # 把输入做一下映射
        mapping_token_mask = tgt_seq_ids.lt(self.src_start_index)  # 为1的地方应该为structure_tokens
        mapped_tokens = tgt_seq_ids.masked_fill(tgt_seq_ids.ge(self.src_start_index), 0)
        tag_mapped_embeds = self.structure_token_embeds[mapped_tokens] * self.decoder.embed_scale

        src_tokens_index = tgt_seq_ids - self.src_start_index
        src_tokens_index = src_tokens_index.masked_fill(mapping_token_mask, 0)
        # src_tokens_index = src_tokens_index.masked_fill(src_tokens_index.ge(src_tokens.size(1)), 0)

        triple_mapped_embeds = h_triple[torch.arange(src_tokens_index.size(0)).unsqueeze(-1), src_tokens_index]
        # triple_mapped_embeds = self.trans_triple(triple_mapped_embeds)

        input_embeds = torch.where(mapping_token_mask.unsqueeze(-1).expand(-1, -1, tag_mapped_embeds.size(-1)),
                                    tag_mapped_embeds,
                                    triple_mapped_embeds)  # bsz x max_len

        # input_embeds = self.LayerNorm(input_embeds)

        if self.training:
            input_embeds = input_embeds[:, :-1]
            decoder_pad_mask = tgt_seq_ids[:, :-1].ne(self.pad_token_id)
            dict = self.decoder(input_ids=None,
                                encoder_hidden_states=encoder_outputs,
                                encoder_attention_mask=encoder_pad_mask,
                                attention_mask=decoder_pad_mask,
                                inputs_embeds=input_embeds,
                                return_dict=True)
        else:
            past_key_values = state.past_key_values
            dict = self.decoder(input_ids=None,
                                encoder_hidden_states=encoder_outputs,
                                encoder_attention_mask=encoder_pad_mask,
                                attention_mask=None,
                                past_key_values=past_key_values,
                                inputs_embeds=input_embeds,
                                use_cache=True,
                                return_dict=True)
            
        hidden_state = dict.last_hidden_state  # bsz x max_len x hidden_size
        hidden_state = self.dropout_layer(hidden_state)

        if not self.training:
            state.past_key_values = dict.past_key_values

        logits = hidden_state.new_full((hidden_state.size(0), hidden_state.size(1), self.src_start_index+h_triple.size(1)),
                                       fill_value=-1e24)

        tag_scores = F.linear(hidden_state, self.dropout_layer(self.structure_token_embeds))

        triple_scores = torch.einsum('blh,bnh->bln', hidden_state, self.dropout_layer(h_triple))
        # print('triple_scores', triple_scores.shape)
        if not self.training:
            # print('triples_pad_mask', triples_pad_mask)
            triple_scores = triple_scores.masked_fill(triples_pad_mask.unsqueeze(1), torch.finfo(triple_scores.dtype).min)
            # print('triple_scores', triple_scores)

        logits[:, :, :self.src_start_index] = tag_scores
        logits[:, :, self.src_start_index:] = triple_scores

        return logits

    def decode(self, tgt_seq_ids, state):
        logits = self.forward(tgt_seq_ids, state)
        return logits[:, -1]

    def no_beam_search_generate(self, state, tokens=None, max_length=10, max_len_a=0.2,
                                repetition_penalty=1.0, length_penalty=1.0):
        
        encoder_pad_mask = state.encoder_mask
        device = self.device
        batch_size = encoder_pad_mask.size(0)
        eos_token_id = self.eos_token_id

        if tokens is None:
            tokens = torch.full([batch_size, 1], fill_value=self.bos_token_id, dtype=torch.long).to(device)

        scores = self.decode(tokens, state)  # 主要是为了update state
        next_tokens = scores.argmax(dim=-1, keepdim=True)

        token_ids = torch.cat([tokens, next_tokens], dim=1)
        cur_len = token_ids.size(1)
        dones = token_ids.new_zeros(batch_size).eq(1).__or__(next_tokens.squeeze(1).eq(eos_token_id))

        if max_len_a!=0:
            if state.encoder_mask is not None:
                max_lengths = (encoder_pad_mask.sum(dim=1).float()*max_len_a).long() + max_length
            else:
                max_lengths = tokens.new_full((tokens.size(0), ), fill_value=max_length, dtype=torch.long)
            real_max_length = max_lengths.max().item()
        else:
            real_max_length = max_length
            if state.encoder_mask is not None:
                max_lengths = encoder_pad_mask.new_ones(encoder_pad_mask.size(0)).long()*max_length
            else:
                max_lengths = tokens.new_full((tokens.size(0),), fill_value=max_length, dtype=torch.long)

        while cur_len < real_max_length:
            scores = self.decode(token_ids, state)  # batch_size x vocab_size

            if repetition_penalty != 1.0:
                token_scores = scores.gather(dim=1, index=token_ids)
                lt_zero_mask = token_scores.lt(0).float()
                ge_zero_mask = lt_zero_mask.eq(0).float()
                token_scores = lt_zero_mask * repetition_penalty * token_scores + ge_zero_mask / repetition_penalty * token_scores
                scores.scatter_(dim=1, index=token_ids, src=token_scores)

            if eos_token_id is not None and length_penalty != 1.0:
                token_scores = scores / cur_len ** length_penalty  # batch_size x vocab_size
                eos_mask = scores.new_ones(scores.size(1))
                eos_mask[eos_token_id] = 0
                eos_mask = eos_mask.unsqueeze(0).eq(1)
                scores = scores.masked_scatter(eos_mask, token_scores)  # 也即除了eos，其他词的分数经过了放大/缩小

            next_tokens = scores.argmax(dim=-1, keepdim=True)
            next_tokens = next_tokens.squeeze(-1)
            # 如果已经达到对应的sequence长度了，就直接填为eos了
            if eos_token_id!=-1:
                next_tokens = next_tokens.masked_fill(max_lengths.eq(cur_len+1), eos_token_id)
                
            next_tokens = next_tokens.masked_fill(dones, self.pad_token_id)  # 对已经搜索完成的sample做padding
            tokens = next_tokens.unsqueeze(1)
            token_ids = torch.cat([token_ids, tokens], dim=-1)  # batch_size x max_len

            end_mask = next_tokens.eq(eos_token_id)
            dones = dones.__or__(end_mask)
            cur_len += 1

            if dones.min() == 1:
                break

        return token_ids

    def beam_search_generate(self, state, tokens=None, max_length=10, max_len_a=0.2, num_beams=4,
                            repetition_penalty=1.0, length_penalty=1.0):
        
        device = self.device
        batch_size = state.encoder_mask.size(0)
        eos_token_id = self.eos_token_id

        if tokens is None:
            tokens = torch.full([batch_size, 1], fill_value=self.bos_token_id, dtype=torch.long).to(device)

        scores = self.decode(tokens, state)  # 这里要传入的是整个句子的长度
        vocab_size = scores.size(1)
        assert vocab_size >= num_beams, "num_beams should be smaller than the number of vocabulary size."

        scores = F.log_softmax(scores, dim=-1)  # (batch_size, vocab_size)
        # 得到(batch_size, num_beams), (batch_size, num_beams)
        _next_scores, _next_tokens = torch.topk(scores, num_beams+1, dim=1, largest=True, sorted=True)

        # 根据index来做顺序的调转
        indices = torch.arange(batch_size, dtype=torch.long).to(device)
        indices = indices.repeat_interleave(num_beams)
        state.reorder_state(indices)
        tokens = tokens.index_select(dim=0, index=indices)  # batch_size * num_beams x length

        if max_len_a!=0:
            if state.encoder_mask is not None:
                max_lengths = (state.encoder_mask.sum(dim=1).float()*max_len_a).long() + max_length
            else:
                max_lengths = tokens.new_full((batch_size*num_beams, ), fill_value=max_length, dtype=torch.long)
            real_max_length = max_lengths.max().item()
        else:
            real_max_length = max_length
            if state.encoder_mask is not None:
                max_lengths = state.encoder_mask.new_ones(state.encoder_mask.size(0)).long()*max_length
            else:
                max_lengths = tokens.new_full((batch_size*num_beams,), fill_value=max_length, dtype=torch.long)
        hypos = [
            BeamHypotheses(num_beams, real_max_length, length_penalty, early_stopping=False) for _ in range(batch_size)
        ]

        not_eos_mask = _next_tokens.ne(eos_token_id)  # 为1的地方不是eos
        keep_mask = not_eos_mask.cumsum(dim=1).le(num_beams)  # 为1的地方需要保留
        keep_mask = not_eos_mask.__and__(keep_mask)  # 为1的地方是需要进行下一步search的

        next_tokens = _next_tokens.masked_select(keep_mask).view(batch_size, num_beams)  # 这是真的接下来要继续的
        next_scores = _next_scores.masked_select(keep_mask).view(batch_size, num_beams)

        rows, cols = not_eos_mask.eq(0)[:, :num_beams].nonzero(as_tuple=True)

        if len(rows)>0:  # 说明有的开头就结束了
            for row, col in zip(rows.tolist(), cols.tolist()):
                _token = torch.cat([tokens[row*num_beams], _next_tokens[row, col:col+1]], dim=0)
                hypos[row].add(_token.clone(), _next_scores[row, col].item())

        # 记录生成好的token (batch_size', cur_len)
        token_ids = torch.cat([tokens, next_tokens.view(-1, 1)], dim=-1)
        dones = [False] * batch_size

        beam_scores = next_scores.view(-1)  # batch_size * num_beams

        #  用来记录已经生成好的token的长度
        cur_len = token_ids.size(1)

        # 0, num_beams, 2*num_beams, ...
        batch_inds_with_numbeams_interval = (torch.arange(batch_size) * num_beams).view(-1, 1).to(token_ids)

        while cur_len < real_max_length:
            scores = self.decode(token_ids, state)  # (bsz x num_beams, vocab_size)
            if repetition_penalty != 1.0:
                token_scores = scores.gather(dim=1, index=token_ids)
                lt_zero_mask = token_scores.lt(0).float()
                ge_zero_mask = lt_zero_mask.eq(0).float()
                token_scores = lt_zero_mask * repetition_penalty * token_scores + ge_zero_mask / repetition_penalty * token_scores
                scores.scatter_(dim=1, index=token_ids, src=token_scores)

            if eos_token_id!=-1:
                max_len_eos_mask = max_lengths.eq(cur_len+1)
                eos_scores = scores[:, eos_token_id]
                # 如果已经达到最大长度，就把eos的分数加大
                scores[:, eos_token_id] = torch.where(max_len_eos_mask, eos_scores+1e32, eos_scores)

            scores = F.log_softmax(scores, dim=-1)  # (batch_size * num_beams, vocab_size)
            _scores = scores + beam_scores[:, None]  # (batch_size * num_beams, vocab_size)
            _scores = _scores.view(batch_size, -1)  # (batch_size, num_beams*vocab_size)
            
            next_scores, ids = torch.topk(_scores, 2 * num_beams, dim=1, largest=True, sorted=True)  # (bsz, 2*num_beams)
            from_which_beam = ids // vocab_size  # (batch_size, 2*num_beams)
            next_tokens = ids % vocab_size  # (batch_size, 2*num_beams)

            #  接下来需要组装下一个batch的结果。
            #  需要选定哪些留下来
            # next_scores, sorted_inds = next_scores.sort(dim=-1, descending=True)
            # next_tokens = next_tokens.gather(dim=1, index=sorted_inds)
            # from_which_beam = from_which_beam.gather(dim=1, index=sorted_inds)

            not_eos_mask = next_tokens.ne(eos_token_id)  # 为1的地方不是eos
            keep_mask = not_eos_mask.cumsum(dim=1).le(num_beams)  # 为1的地方需要保留
            keep_mask = not_eos_mask.__and__(keep_mask)  # 为1的地方是需要进行下一步search的

            _next_tokens = next_tokens.masked_select(keep_mask).view(-1, 1)
            _from_which_beam = from_which_beam.masked_select(keep_mask).view(batch_size, num_beams)  # 上面的token是来自哪个beam
            _next_scores = next_scores.masked_select(keep_mask).view(batch_size, num_beams)
            beam_scores = _next_scores.view(-1)

            flag = True
            if cur_len+1 == real_max_length:
                eos_batch_idx = torch.arange(batch_size).to(next_tokens).repeat_interleave(repeats=num_beams, dim=0)
                eos_beam_ind = torch.arange(num_beams).to(token_ids).repeat(batch_size)  # 表示的是indice
                eos_beam_idx = from_which_beam[:, :num_beams].reshape(-1)  # 表示的是从哪个beam获取得到的
            else:
                # 将每个batch中在num_beam内的序列添加到结束中, 为1的地方需要结束了
                effective_eos_mask = next_tokens[:, :num_beams].eq(eos_token_id)  # batch_size x num_beams
                if effective_eos_mask.sum().gt(0):
                    eos_batch_idx, eos_beam_ind = effective_eos_mask.nonzero(as_tuple=True)
                    # 是由于from_which_beam是 (batch_size, 2*num_beams)的，所以需要2*num_beams
                    eos_beam_idx = eos_batch_idx * num_beams * 2 + eos_beam_ind
                    eos_beam_idx = from_which_beam.view(-1)[eos_beam_idx]  # 获取真实的从哪个beam获取的eos
                else:
                    flag = False

            if flag:
                _token_ids = torch.cat([token_ids, _next_tokens], dim=-1)
                for batch_idx, beam_ind, beam_idx in zip(eos_batch_idx.tolist(), eos_beam_ind.tolist(),
                                                        eos_beam_idx.tolist()):
                    if not dones[batch_idx]:
                        score = next_scores[batch_idx, beam_ind].item()
                        # 之后需要在结尾新增一个eos
                        if eos_token_id!=-1:
                            hypos[batch_idx].add(_token_ids[batch_idx * num_beams + beam_idx, :cur_len].clone(), score)
                        else:
                            hypos[batch_idx].add(_token_ids[batch_idx * num_beams + beam_idx].clone(), score)

            # 更改state状态, 重组token_ids
            reorder_inds = (batch_inds_with_numbeams_interval + _from_which_beam).view(-1)  # flatten成一维
            state.reorder_state(reorder_inds)
            # 重新组织token_ids的状态
            token_ids = torch.cat([token_ids.index_select(index=reorder_inds, dim=0), _next_tokens], dim=-1)

            for batch_idx in range(batch_size):
                dones[batch_idx] = dones[batch_idx] or hypos[batch_idx].is_done(next_scores[batch_idx, 0].item()) or \
                                max_lengths[batch_idx*num_beams]==cur_len+1

            cur_len += 1

            if all(dones):
                break

        # select the best hypotheses
        tgt_len = token_ids.new_zeros(batch_size)
        best = []

        for i, hypotheses in enumerate(hypos):
            best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
            # 把上面替换为非eos的词替换回eos
            if eos_token_id!=-1:
                best_hyp = torch.cat([best_hyp, best_hyp.new_ones(1)*eos_token_id])
            tgt_len[i] = len(best_hyp)
            best.append(best_hyp)

        # generate target batch
        decoded = token_ids.new_zeros(batch_size, tgt_len.max().item()).fill_(self.pad_token_id)
        for i, hypo in enumerate(best):
            decoded[i, :tgt_len[i]] = hypo

        return decoded


    def decode_tree(self, tokenizer, batch_seq, dict_triples, info):
        batch_seq = batch_seq.cpu().tolist()
        batch_text = info['text']
        sent_idxes = info['sent_idx']

        res = []
        for text, seq, sent_idx in zip(batch_text, batch_seq, sent_idxes):
            triples = dict_triples[sent_idx]

            mapped_seq = []
            for idx in seq:
                if idx < self.src_start_index:
                    token_id = self.structure_token_ids[idx]
                    token = tokenizer.convert_ids_to_tokens([token_id])[0]
                else:
                    token = idx - self.src_start_index

                mapped_seq.append(token)

            tree, _, _ = scan_seq([], mapped_seq, 1, triples)

            # print(text)
            # print(mapped_seq)
            # print(tree)
            # print('', flush=True)

            res.append({'text': text, 'tree': tree})

        return res
        


class State:
    def __init__(self, encoder_output, encoder_mask, project_triple, rel_indices=None, relID2triples=None):
        """
        每个Decoder都有对应的State对象用来承载encoder的输出以及当前时刻之前的decode状态。

        :param Union[torch.Tensor, list, tuple] encoder_output: 如果不为None，内部元素需要为torch.Tensor, 默认其中第一维是batch
            维度
        :param Union[torch.Tensor, list, tuple] encoder_mask: 如果部位None，内部元素需要torch.Tensor, 默认其中第一维是batch
            维度
        :param kwargs:
        """
        self.past_key_values = None

        self.encoder_output = encoder_output
        self.encoder_mask = encoder_mask

        self.LayerNorm = nn.LayerNorm(encoder_output.size(-1), eps=1e-5).to(encoder_output.device)

        if rel_indices is not None:
            project_triple_ = []
            # print([len(indices[0]) for indices in rel_indices])
            batch_size, num_gen_triples = project_triple.size(0), project_triple.size(1)
            if relID2triples[0] is not None:
                num_triples = [len(x) for x in relID2triples]
            else:
                num_triples = [len(indices[0]) for indices in rel_indices]
            max_num_triples = max(num_triples)

            for i, (src, tgt) in enumerate(rel_indices):
                num_rel = num_triples[i]
                assert len(tgt) % num_rel == 0

                _, sorted_indices = torch.sort(tgt, dim=0)
                batch_idx = src.new_full((max_num_triples,), fill_value=i)

                if relID2triples[i] is not None:
                    chosen_src = []
                    relID2triples_ = relID2triples[i]
                    # print(relID2triples_)
                    # print('sorted_indices')
                    # for s, index in enumerate(sorted_indices):
                    #     print(s, index)
                    # print()
                    # print('src')
                    # for s, index in enumerate(src):
                    #     print(s, index)
                    # print()
                    
                    for relID in range(num_rel):
                        chosen_triple = random.choice(relID2triples_[relID])
                        # print(sorted_indices[chosen_triple], src[sorted_indices[chosen_triple]])
                        chosen_src.append(src[sorted_indices[chosen_triple]])
                    # print()

                    chosen_src = torch.stack(chosen_src)
                else:
                    chosen_src = src[sorted_indices]

                # print(chosen_src)
                # print()

                pad_full_idx = [idx for idx in torch.randperm(num_gen_triples) if idx not in chosen_src]
                k = (max_num_triples-num_rel) // len(pad_full_idx)
                v = (max_num_triples-num_rel) % len(pad_full_idx)
                pad_full_idx = pad_full_idx * k + pad_full_idx[:v]

                # pad_full_idx = [idx for idx in range(num_gen_triples) if idx not in sorted_indices]
                # pad_full_idx = pad_full_idx[:1] * (max_num_triples-len(src))

                pad_full_idx = torch.LongTensor(pad_full_idx, device=src.device)

                src_idx = torch.cat([chosen_src, pad_full_idx])

                # print(chosen_src)
                # print(pad_full_idx)
                # print(flush=True)

                tmp = project_triple[batch_idx, src_idx]
                # print(tmp.shape)
                project_triple_.append(tmp)

            self.project_triple = torch.stack(project_triple_)

            mask = torch.arange(max_num_triples).expand(batch_size, -1) >= torch.LongTensor(num_triples).unsqueeze(1)
            self.triples_pad_mask = mask.to(project_triple.device)
            
        else:
            self.project_triple = project_triple
            self.triples_pad_mask = None

    def concat_context_triples(self, h_triples=None):
        if h_triples is None:
            h_triples = self.project_triple
        self.encoder_output = torch.cat([h_triples, self.encoder_output], dim=1)
        self.encoder_output = self.LayerNorm(self.encoder_output)
        self.encoder_mask = torch.cat([~self.triples_pad_mask, self.encoder_mask], dim=1)
    
    def set_triples(self, triple_idxes):
        project_triple = self.project_triple

        project_triple_ = []
        batch_size, num_gen_triples = project_triple.size(0), project_triple.size(1)
        num_triples = [len(triple_idx) for triple_idx in triple_idxes]
        max_num_triples = max(num_triples)
        for i, triple_idx in enumerate(triple_idxes):
            # print(i, triple_idx)
            pad_full_idx = [idx for idx in range(num_gen_triples) if idx not in triple_idx]
            pad_full_idx = pad_full_idx[:1] * (max_num_triples-len(triple_idx))

            triple_idx_ = triple_idx + pad_full_idx
            project_triple_.append(project_triple[i, triple_idx_])

        self.project_triple = torch.stack(project_triple_)

        mask = torch.arange(max_num_triples).expand(batch_size, -1) >= torch.LongTensor(num_triples).unsqueeze(1)
        self.triples_pad_mask = mask.to(project_triple.device)

        # print('num_triples', num_triples)

    def _reorder_state(self, state: Union[torch.Tensor, list, tuple], indices: torch.LongTensor, dim: int = 0):
        if isinstance(state, torch.Tensor):
            state = state.index_select(index=indices, dim=dim)
        elif isinstance(state, list):
            for i in range(len(state)):
                assert state[i] is not None
                state[i] = self._reorder_state(state[i], indices, dim)
        elif isinstance(state, tuple):
            tmp_list = []
            for i in range(len(state)):
                assert state[i] is not None
                tmp_list.append(self._reorder_state(state[i], indices, dim))
            state = tuple(tmp_list)
        else:
            raise TypeError(f"Cannot reorder data of type:{type(state)}")

        return state

    def reorder_state(self, indices: torch.LongTensor):
        if self.encoder_mask is not None:
            self.encoder_mask = self._reorder_state(self.encoder_mask, indices)
        if self.encoder_output is not None:
            self.encoder_output = self._reorder_state(self.encoder_output, indices)
            
        if self.project_triple is not None:
            self.project_triple = self._reorder_state(self.project_triple, indices)
        if self.triples_pad_mask is not None:
            self.triples_pad_mask = self._reorder_state(self.triples_pad_mask, indices)

        if self.past_key_values is not None:
            # print(len(self.past_key_values))
            reordered_past = ()
            for layer_past in self.past_key_values:
                # print(len(layer_past))
                # cached cross_attention states don't have to be reordered -> they are always the same
                reordered_past += (
                    tuple(past_state.index_select(0, indices) for past_state in layer_past),
                )
            self.past_key_values = reordered_past


class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.hyp = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.num_beams or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            return self.worst_score >= best_sum_logprobs / self.max_length ** self.length_penalty
