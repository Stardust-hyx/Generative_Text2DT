from numpy import dtype
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import random
from models.set_criterion import PIQNLoss, ConsistencyLoss, Seq2SeqLoss
from models.seq_encoder import EntityAwareBertConfig, SeqEncoder
from models.seq_decoder import CPTConfig, SeqDecoder, State, CPTAttention, _expand_mask
from utils.functions import generate_triple, generate_entity
from utils.data import padded_stack


class SetPred4RE(nn.Module):

    def __init__(self, args, relational_alphabet, entity_type_alphabet, tokenizer, structure_token_ids):
        super(SetPred4RE, self).__init__()
        self.args = args

        self.relational_alphabet = relational_alphabet
        self.entity_type_alphabet = entity_type_alphabet
        num_classes = relational_alphabet.size()
        num_ent_types = entity_type_alphabet.size()


        config = EntityAwareBertConfig.from_pretrained(args.cpt_directory, hidden_dropout_prob=args.dropout, num_generated_triples=args.num_generated_triples,
                    entity_queries_num = args.entity_queries_num, mask_ent2tok = args.mask_ent2tok,  mask_tok2ent = args.mask_tok2ent, mask_ent2ent = args.mask_ent2ent, mask_entself = args.mask_entself, entity_aware_attention = args.entity_aware_attention, entity_aware_selfout = args.entity_aware_selfout, entity_aware_intermediate = args.entity_aware_intermediate, entity_aware_output = args.entity_aware_output, use_entity_pos = args.use_entity_pos, use_entity_common_embedding = args.use_entity_common_embedding)

        self.PIQN = SeqEncoder.from_pretrained(args.cpt_directory,
                                                config = config,
                                                fix_bert_embeddings=args.fix_bert_embeddings,
                                                relation_type_count = num_classes,
                                                # piqn model parameters
                                                entity_type_count = num_ent_types,
                                                prop_drop = args.prop_drop,
                                                pos_size = args.pos_size,
                                                char_lstm_layers = args.char_lstm_layers,
                                                char_lstm_drop = args.char_lstm_drop, 
                                                char_size = args.char_size, 
                                                use_glove = args.use_glove, 
                                                use_pos = args.use_pos, 
                                                use_char_lstm = args.use_char_lstm,
                                                lstm_layers = args.lstm_layers,
                                                pool_type = args.pool_type,
                                                word_mask_tok2ent = args.word_mask_tok2ent,
                                                word_mask_ent2tok = args.word_mask_ent2tok,
                                                word_mask_ent2ent = args.word_mask_ent2ent,
                                                word_mask_entself = args.word_mask_entself,
                                                share_query_pos = args.share_query_pos,
                                                use_token_level_encoder = args.use_token_level_encoder,
                                                num_token_ent_rel_layer = args.num_token_ent_rel_layer,
                                                num_token_ent_layer = args.num_token_ent_layer,
                                                num_token_rel_layer = args.num_token_rel_layer,
                                                num_token_head_tail_layer = args.num_token_head_tail_layer,
                                                use_entity_attention = args.use_entity_attention,
                                                use_aux_loss =  args.use_aux_loss,
                                                use_lstm = args.use_lstm,
                                            )
        
        config = CPTConfig.from_pretrained(args.cpt_directory)
        
        self.decoder = SeqDecoder.from_pretrained(args.cpt_directory,
                                                config = config,
                                                structure_token_ids=structure_token_ids,
                                            )
        
        self.triples_attn_to_context = CPTAttention(
            config.d_model,
            config.decoder_attention_heads,
            dropout=0.1,
            bias=True,
            do_out_proj=False
        )

        self.gate = nn.Parameter(torch.tensor(0.5))

        self.tokenizer = tokenizer

        self.num_classes = num_classes
        self.num_ent_types = num_ent_types

        self.re_criterion = PIQNLoss(num_classes, loss_weight=self.get_loss_weight(args), na_coef=args.na_rel_coef, losses=["entity", "relation", "head_tail_part", "head_tail_type"],
                                        matcher=args.matcher, loss_type='RE', boundary_softmax=True)
        self.ner_criterion = PIQNLoss(num_ent_types, loss_weight=self.get_loss_weight(args), na_coef=args.na_ent_coef, losses=["ner_type", "ner_span", "ner_part", "ent_have_rel"],
                                        matcher=args.matcher, loss_type='NER', boundary_softmax=False)

        self.consistency_criterion = ConsistencyLoss()

        self.seq2seq_loss = Seq2SeqLoss(num_structure_token=len(structure_token_ids),
                                        tag_coef=args.tag_coef,
                                        triple_coef=args.triple_coef
                                    )

    def forward(self, input_ids, attention_mask, seg_encoding, context2token_masks, token_masks, targets=None, tgt_seq_ids=None, tgt_seq_len=None, relID2triples=None,
                epoch=None, gen=True):

        h_token, ent_outputs, rel_outputs, project_entity, project_triple = self.PIQN(input_ids, attention_mask, seg_encoding, context2token_masks, token_masks)

        if targets is not None:
            re_loss, _, rel_indices = self.re_criterion(rel_outputs, targets)
            ner_loss, ent_indices, _ = self.ner_criterion(ent_outputs, targets)

            # print('re_loss', re_loss)
            # print('ner_loss', ner_loss)
            # print('gen_loss', gen_loss)
            # print()

            if not gen:
                loss = re_loss + ner_loss
                if epoch >= self.args.start_consistency_epoch:
                    consistency_loss = self.consistency_criterion(ent_outputs[-1], rel_outputs[-1], ent_indices, rel_indices)
                    # print(re_loss, ner_loss, consistency_loss, flush=True)
                    loss += consistency_loss * self.args.consistency_loss_weight
            else:
                state = State(h_token, attention_mask, project_triple, rel_indices, relID2triples)

                encoder_attention_mask = _expand_mask(state.encoder_mask, state.project_triple.dtype, tgt_len=state.project_triple.size(1))
                h_triples, _, _ = self.triples_attn_to_context(
                    hidden_states=state.project_triple,
                    key_value_states=state.encoder_output.detach(),
                    attention_mask=encoder_attention_mask,
                )
                state.concat_context_triples(h_triples=state.project_triple)
                # state.project_triple = h_triples
                state.project_triple = self.gate * state.project_triple + (1-self.gate) * h_triples
                
                decoder_output = self.decoder(tgt_seq_ids, state)
                gen_loss = self.seq2seq_loss(tgt_seq_ids, tgt_seq_len, decoder_output)

                consistency_loss = self.consistency_criterion(ent_outputs[-1], rel_outputs[-1], ent_indices, rel_indices)
                loss = re_loss + ner_loss + consistency_loss * self.args.consistency_loss_weight + gen_loss * self.args.Gen_loss_weight

            return loss, rel_outputs[-1], ent_outputs[-1]
        else:
            state = State(h_token, attention_mask, project_triple)

            return state, rel_outputs[-1], ent_outputs[-1]

    def predict(self, input_ids, attention_mask, seg_encoding, context2token_masks, token_masks, info, gen):
        with torch.no_grad():
            state, rel_output, ent_output = self.forward(input_ids, attention_mask, seg_encoding, context2token_masks, token_masks)

            pred_triple, triple_idxes, triples_  = \
                generate_triple(rel_output, info, self.args, self.num_classes, self.relational_alphabet)

            pred_entity = None
            # pred_entity = generate_entity(ent_output, info, self.args, self.num_ent_types)
            res = []
            
            if gen:
                state.set_triples(triple_idxes)

                encoder_attention_mask = _expand_mask(state.encoder_mask, state.project_triple.dtype, tgt_len=state.project_triple.size(1))
                h_triples, _, _ = self.triples_attn_to_context(
                    hidden_states=state.project_triple,
                    key_value_states=state.encoder_output,
                    attention_mask=encoder_attention_mask,
                )
                state.concat_context_triples(h_triples=state.project_triple)
                # state.project_triple = h_triples
                state.project_triple = self.gate * state.project_triple + (1-self.gate) * h_triples

                # gen_seq = self.decoder.no_beam_search_generate(state)
                gen_seq = self.decoder.beam_search_generate(state, num_beams=4)
                # print(gen_seq)
                res = self.decoder.decode_tree(self.tokenizer, gen_seq, triples_, info)

        return pred_triple, pred_entity, res

    def batchify(self, batch_list, is_test=False):
        batch_size = len(batch_list)
        sent_idx = [ele[0] for ele in batch_list]
        sent_ids = [ele[1] for ele in batch_list]
        targets = [ele[2] for ele in batch_list]
        # bio_labels = [ele[3] for ele in batch_list]
        tree = [ele[3] for ele in batch_list]
        text = [ele[4] for ele in batch_list]
        bep_to_char = [ele[5] for ele in batch_list]
        seg_encoding = [ele[6] for ele in batch_list]
        context2token_masks = [ele[7] for ele in batch_list]
        token_masks = [ele[8] for ele in batch_list]
        tgt_seq_ids = [ele[9] for ele in batch_list] if not is_test else None
        relID2triples = [ele[10] for ele in batch_list] if not is_test else None
        tgt_seq_len = None
        
        sent_lens = list(map(len, sent_ids))
        max_sent_len = max(sent_lens)
        input_ids = torch.zeros((batch_size, max_sent_len), requires_grad=False).long()
        attention_mask = torch.zeros((batch_size, max_sent_len), requires_grad=False, dtype=torch.float32)
        for idx, (seq, seqlen) in enumerate(zip(sent_ids, sent_lens)):
            input_ids[idx, :seqlen] = torch.LongTensor(seq)
            attention_mask[idx, :seqlen] = 1

        seg_encoding = [torch.tensor(x, dtype=torch.long) for x in seg_encoding]
        seg_encoding = padded_stack(seg_encoding)
        if context2token_masks[0]:
            context2token_masks = [torch.stack([torch.tensor(x, dtype=torch.bool) for x in masks]) for masks in context2token_masks]
            context2token_masks = padded_stack(context2token_masks)
        else:
            context2token_masks = None
        token_masks = [torch.tensor(x, dtype=torch.bool) for x in token_masks]
        token_masks = padded_stack(token_masks)

        targets_ = copy.deepcopy(targets)
        for t in targets_:
            for k in t.keys():
                if 'labels' in k:
                    padded = []
                    for labels in t[k]:
                        padded.append(labels + [0] * (max_sent_len - len(labels)))
                    t[k] = padded

        if not is_test:
            tgt_seq_len = [len(ids) for ids in tgt_seq_ids]
            batch_max_length = max(tgt_seq_len)
            tgt_seq_ids = torch.LongTensor([ids + [0 for _ in range(batch_max_length - len(ids))]   # padding with id of [SEP]
                                            for ids in tgt_seq_ids])
            tgt_seq_len = torch.LongTensor(tgt_seq_len)

        if self.args.use_gpu:
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            seg_encoding = seg_encoding.cuda()
            context2token_masks = context2token_masks.cuda() if context2token_masks else None
            token_masks = token_masks.cuda()

            targets_ = [{k: torch.tensor(v, dtype=(torch.float if 'labels' in k else torch.long), requires_grad=False).cuda()
            for k, v in t.items() if 'mention' not in k} for t in targets_]
            if not is_test:
                tgt_seq_ids = tgt_seq_ids.cuda()
                tgt_seq_len = tgt_seq_len.cuda()
            
        else:
            targets_ = [{k: torch.tensor(v, dtype=(torch.float if 'labels' in k else torch.long), requires_grad=False)
            for k, v in t.items() if 'mention' not in k} for t in targets_]
                
        info = {"seq_len": sent_lens, "sent_idx": sent_idx, "text": text, "bep_to_char": bep_to_char}

        info.update({"head_mention": [t['head_mention'] for t in targets]})
        info.update({"tail_mention": [t['tail_mention'] for t in targets]})
        info.update({"tree": tree})
        info.update({'relID2triples': relID2triples})

        return input_ids, attention_mask, seg_encoding, context2token_masks, token_masks, targets_, tgt_seq_ids, tgt_seq_len, info

    def get_CL_sample(self, projections, target_labels, indices, type_embeddings):
        idx = self._get_src_permutation_idx(indices)
        projections = projections[idx]
        target_labels = torch.cat([t[i] for t, (_, i) in zip(target_labels, indices)])
        
        instances = [[] for type_embedding in type_embeddings]
        min_instance = projections.size(0)
        for projection, target_label in zip(projections, target_labels):
            instances[target_label].append(projection)

        for instance in instances:
            num = len(instance)
            if num > 1:
                min_instance = min(min_instance, num)

        instances_ = []
        labels_ = []
        for i, instance in enumerate(instances):
            num = len(instance)
            if num > 1:
                num_split = num // min_instance
                splits = [instance[k*min_instance: (k+1)*min_instance] for k in range(num_split)]
                for split in splits:
                    instances_.append(torch.stack(split))
                    labels_.append(i)

        instances_ = torch.stack(instances_)
        instances_ = F.normalize(instances_, dim=-1)
        labels_ = torch.LongTensor(labels_)
        
        return instances_, labels_
        
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def init_type_embeddings(self, tokenizer, ent_type_names, rel_type_names):
        CLS_id= tokenizer.cls_token_id
        CLS_embedding = self.PIQN.bert.embeddings.word_embeddings.weight.data[CLS_id]

        for i, ent_type_name in enumerate(ent_type_names):
            tokens = tokenizer.tokenize(ent_type_name)
            indexs = tokenizer.convert_tokens_to_ids(tokens)
            embed = self.PIQN.bert.embeddings.word_embeddings.weight.data[indexs]
            embed = embed.max(dim=0).values
            # print(tokens)
            self.PIQN.ent_type_embeddings.weight.data[i] = embed

        for i, rel_type_name in enumerate(rel_type_names):
            tokens = tokenizer.tokenize(rel_type_name)
            indexs = tokenizer.convert_tokens_to_ids(tokens)
            embed = self.PIQN.bert.embeddings.word_embeddings.weight.data[indexs]
            embed = embed.max(dim=0).values
            # print(tokens)
            self.PIQN.rel_type_embeddings.weight.data[i] = embed

    @staticmethod
    def get_loss_weight(args):
        return {"relation": args.rel_loss_weight,
                "head_entity": args.head_ent_loss_weight,
                "tail_entity": args.tail_ent_loss_weight,
                "ent_type": args.ent_type_loss_weight,
                "ent_span": args.ent_span_loss_weight,
                "ent_part": args.ent_part_loss_weight,
                "head_part": args.head_part_loss_weight,
                "tail_part": args.tail_part_loss_weight,
                "head_tail_type": args.head_tail_type_loss_weight,
                "ent_have_rel": 0,
                }
