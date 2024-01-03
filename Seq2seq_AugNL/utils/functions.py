import torch, collections
from collections import Counter

def filteration(prediction, remove_overlap=True):
    triple_idxes, triples_ = [], {}
    
    for sent_id, triples in prediction.items():
        triples = [(triple_id,
                    (ele.head_mention, ele.pred_rel, ele.tail_mention, ele.head_start_index, ele.head_end_index,
                    ele.tail_start_index, ele.tail_end_index),
                    ele.rel_prob + 0.5*(ele.head_start_prob + ele.head_end_prob) + 0.5*(ele.tail_start_prob + ele.tail_end_prob)
                    ) for triple_id, ele in triples]

        triples = sorted(triples, key=lambda x: x[-1], reverse=True)

        res = []
        triple_idx = []
        triples_[sent_id] = []
        for triple_id, pred, score in triples:
            remove = False
            for ele in res:
                if remove_overlap and max(ele[3], pred[3]) <= min(ele[4], pred[4]) and \
                    max(ele[5], pred[5]) <= min(ele[6], pred[6]):
                    remove = True
                elif ele[3] == pred[3] and ele[4] == pred[4] and ele[5] == pred[5] and ele[6] == pred[6]:
                    remove = True
            if not remove:
                res.append(pred)

                triple_idx.append(triple_id)
                triples_[sent_id].append(pred[:3])

        triple_idxes.append(triple_idx)

    return triple_idxes, triples_


def generate_span(start_logits, end_logits, info, args, num_queries):
    seq_lens = info["seq_len"] # including [CLS] and [SEP]
    sent_idxes = info["sent_idx"]
    texts = info["text"]
    bep_to_chars = info["bep_to_char"]
    _Prediction = collections.namedtuple(
        "Prediction", ["start_index", "end_index", "start_prob", "end_prob", "mention"]
    )
    output = {}
    start_probs = start_logits.softmax(-1)
    end_probs = end_logits.softmax(-1)
    # start_probs = start_probs.cpu().tolist()
    # end_probs = end_probs.cpu().tolist()
    for (start_prob, end_prob, seq_len, sent_idx, text, bep_to_char) in zip(start_probs, end_probs, seq_lens, sent_idxes, texts, bep_to_chars):
        output[sent_idx] = {}
        K = min(start_prob.size(-1), args.n_best_size)
        for query_id in range(num_queries):
            predictions = []
            start_indexes = torch.topk(start_prob[query_id], K).indices
            end_indexes = torch.topk(end_prob[query_id], K).indices
            found = False
            for start_index in start_indexes:
                if found:
                    break
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the sentence. We throw out all
                    # invalid predictions.
                    if start_index == 0:
                        continue
                    if start_index >= (seq_len-1): # [SEP]
                        continue
                    if end_index >= (seq_len-1):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > args.max_span_length:
                        continue
                    start_index_, end_index_ = start_index.item(), end_index.item()

                    if isinstance(text, list):
                        mention = ' '.join(text[bep_to_char[start_index_]: bep_to_char[end_index_]+1])
                    else:
                        mention = text[bep_to_char[start_index_][0]: bep_to_char[end_index_][-1]+1]
                    predictions.append(
                        _Prediction(
                            start_index=start_index_,
                            end_index=end_index_,
                            start_prob=start_prob[query_id][start_index].item(),
                            end_prob=end_prob[query_id][end_index].item(),
                            mention=mention
                        )
                    )
                    found = True
                    break
            output[sent_idx][query_id] = predictions
    return output


def generate_relation(pred_rel_logits, info, args, relational_alphabet):
    rel_probs, pred_rels = torch.max(pred_rel_logits.softmax(-1), dim=2)
    rel_probs = rel_probs.cpu().tolist()
    pred_rels = pred_rels.cpu().tolist()
    sent_idxes = info["sent_idx"]
    output = {}
    _Prediction = collections.namedtuple(
        "Prediction", ["pred_rel", "rel_prob"]
    )
    for (rel_prob, pred_rel, sent_idx) in zip(rel_probs, pred_rels, sent_idxes):
        output[sent_idx] = {}
        for triple_id in range(args.num_generated_triples):
            relation = pred_rel[triple_id]
            if relation < relational_alphabet.size():
                relation = relational_alphabet.get_instance(pred_rel[triple_id])
                
            output[sent_idx][triple_id] = _Prediction(
                            pred_rel=relation,
                            rel_prob=rel_prob[triple_id])
    return output


def generate_triple(output, info, args, num_classes, relational_alphabet):
    _Pred_Triple = collections.namedtuple(
        "Pred_Triple", ["pred_rel", "rel_prob", "head_start_index", "head_end_index", "head_start_prob", "head_end_prob", "tail_start_index", "tail_end_index", "tail_start_prob", "tail_end_prob", "head_mention", "tail_mention"]
    )
    pred_head_ent_dict = generate_span(output["head_start_logits"], output["head_end_logits"], info, args, args.num_generated_triples)
    pred_tail_ent_dict = generate_span(output["tail_start_logits"], output["tail_end_logits"], info, args, args.num_generated_triples)
    pred_rel_dict = generate_relation(output['pred_rel_logits'], info, args, relational_alphabet)
    triples = {}
    triples_ = {}
    for sent_idx in pred_rel_dict:
        triples[sent_idx] = []
        triples_[sent_idx] = []
        for triple_id in range(args.num_generated_triples):
            pred_rel = pred_rel_dict[sent_idx][triple_id]
            pred_head = pred_head_ent_dict[sent_idx][triple_id]
            pred_tail = pred_tail_ent_dict[sent_idx][triple_id]
            triple = generate_strategy(pred_rel, pred_head, pred_tail, num_classes, _Pred_Triple)
            if triple:
                triples[sent_idx].append(triple)
                triples_[sent_idx].append((triple_id, triple))

    # print(triples)
    triple_idxes, triples_ = filteration(triples_)

    return triples, triple_idxes, triples_


def generate_strategy(pred_rel, pred_head, pred_tail, num_classes, _Pred_Triple):
    if pred_rel.pred_rel != num_classes:
        if pred_head and pred_tail:
            # for ele in pred_head:
            #     if ele.start_index != 0:
            #         break
            head = pred_head[0]
            # for ele in pred_tail:
            #     if ele.start_index != 0:
            #         break
            tail = pred_tail[0]
            return _Pred_Triple(pred_rel=pred_rel.pred_rel, rel_prob=pred_rel.rel_prob, head_start_index=head.start_index, head_end_index=head.end_index, head_start_prob=head.start_prob, head_end_prob=head.end_prob, tail_start_index=tail.start_index, tail_end_index=tail.end_index, tail_start_prob=tail.start_prob, tail_end_prob=tail.end_prob, head_mention=head.mention, tail_mention=tail.mention)
        else:
            return
    else:
        return

def formulate_gold(target, info):
    sent_idxes = info["sent_idx"]
    gold = {}
    for i in range(len(sent_idxes)):
        gold[sent_idxes[i]] = []
        for j in range(len(target[i]["relation"])):
            gold[sent_idxes[i]].append(
                (target[i]["relation"][j].item(), target[i]["head_start_index"][j].item(), target[i]["head_end_index"][j].item(), target[i]["tail_start_index"][j].item(), target[i]["tail_end_index"][j].item())
            )
    return gold

def formulate_gold_(target, info, relational_alphabet):
    sent_idxes = info["sent_idx"]
    gold = {}
    for i in range(len(sent_idxes)):
        gold[sent_idxes[i]] = []
        for j in range(len(target[i]["relation"])):
            tuple = (relational_alphabet.get_instance(target[i]["relation"][j].item()), info["head_mention"][i][j], info["tail_mention"][i][j])
            # if tuple in gold[sent_idxes[i]]:
            #     print(info['text'][i])
            if tuple not in gold[sent_idxes[i]]:
                gold[sent_idxes[i]].append(tuple)
    return gold


def generate_ent_type(pred_type_logits, info, args):
    type_probs, pred_types = torch.max(pred_type_logits.softmax(-1), dim=2)
    type_probs = type_probs.cpu().tolist()
    pred_types = pred_types.cpu().tolist()
    sent_idxes = info["sent_idx"]
    output = {}
    _Prediction = collections.namedtuple(
        "Prediction", ["pred_type", "type_prob"]
    )
    for (type_prob, pred_type, sent_idx) in zip(type_probs, pred_types, sent_idxes):
        output[sent_idx] = {}
        for entity_id in range(args.entity_queries_num):
            output[sent_idx][entity_id] = _Prediction(
                            pred_type=pred_type[entity_id],
                            type_prob=type_prob[entity_id])
    return output

def generate_entity(output, info, args, num_classes):
    _Pred_Entity = collections.namedtuple(
        "Pred_Triple", ["pred_type", "type_prob", "start_index", "end_index", "start_prob", "end_prob", "entity_mention"]
    )
    pred_span_dict = generate_span(output["ent_start_logits"], output["ent_end_logits"], info, args, args.entity_queries_num)
    pred_type_dict = generate_ent_type(output['ent_type_logits'], info, args)
    entities = {}
    for sent_idx in pred_type_dict:
        entities[sent_idx] = []
        for entity_id in range(args.entity_queries_num):
            pred_type = pred_type_dict[sent_idx][entity_id]
            pred_span = pred_span_dict[sent_idx][entity_id]
            entity = generate_ent_strategy(pred_type, pred_span, num_classes, _Pred_Entity)
            if entity:
                entities[sent_idx].append(entity)
    # print(triples)
    return entities

def generate_ent_strategy(pred_type, pred_span, num_classes, _Pred_Entity):
    if pred_type.pred_type != num_classes:
        if pred_span:
            span = pred_span[0]
            return _Pred_Entity(pred_type=pred_type.pred_type, type_prob=pred_type.type_prob, start_index=span.start_index, end_index=span.end_index, start_prob=span.start_prob, end_prob=span.end_prob, entity_mention=span.mention)
        else:
            return
    else:
        return

def formulate_gold_ent(target, info):
    sent_idxes = info["sent_idx"]
    gold = {}
    for i in range(len(sent_idxes)):
        # print(target[i])
        gold[sent_idxes[i]] = set()
        for j in range(len(target[i]["ent_type"])):
            gold[sent_idxes[i]].add(
                (target[i]["ent_type"][j].item(), target[i]["ent_start_index"][j].item(), target[i]["ent_end_index"][j].item())
            )
    return gold


def scan_seq(tgt, seq, start, id2triple, num_leaf=0, num_inner=0):
    flag, _ = is_completed(tgt, 0)
    if start >= len(seq) or flag:
        return tgt, num_leaf, num_inner

    if seq[start: start+4] == ['否', '则', '，', '若']:
        role = 'C'
        node_triples, logical_rel, end = get_node(seq, start+4, id2triple)
        tgt.append({'role': role, 'triples': node_triples, 'logical_rel': logical_rel})

        tgt, num_leaf_, num_inner_ = scan_seq(tgt, seq, end+1, id2triple, 0, 1)

        if num_leaf_ < num_inner_ + 1:
            tgt += [{'role': 'D', 'triples': [], 'logical_rel': 'null'}] * (num_inner_ + 1 - num_leaf_)
            num_leaf_ = num_inner_ + 1

        return tgt, num_leaf+num_leaf_, num_inner+num_inner_

    elif seq[start: start+2] == ['否', '则']:
        role = 'D'
        node_triples, logical_rel, end = get_node(seq, start+2, id2triple)

        if len(node_triples) > 0:
            tgt.append({'role': role, 'triples': node_triples, 'logical_rel': logical_rel})
            num_leaf += 1
        tgt, num_leaf, num_inner = scan_seq(tgt, seq, end+1, id2triple, num_leaf, num_inner)

        return tgt, num_leaf, num_inner

    elif seq[start: start+1] == ['则']:
        role = 'D'
        node_triples, logical_rel, end = get_node(seq, start+1, id2triple)

        if len(node_triples) > 0:
            tgt.append({'role': role, 'triples': node_triples, 'logical_rel': logical_rel})
            num_leaf += 1
        tgt, num_leaf, num_inner = scan_seq(tgt, seq, end+1, id2triple, num_leaf, num_inner)

        return tgt, num_leaf, num_inner

    elif seq[start: start+1] == ['若']:
        role = 'C'
        node_triples, logical_rel, end = get_node(seq, start+1, id2triple)
        tgt.append({'role': role, 'triples': node_triples, 'logical_rel': logical_rel})

        tgt, num_leaf_, num_inner_ = scan_seq(tgt, seq, end+1, id2triple, 0, 1)

        if num_leaf_ < num_inner_ + 1:
            tgt += [{'role': 'D', 'triples': [], 'logical_rel': 'null'}] * (num_inner_ + 1 - num_leaf_)
            num_leaf_ = num_inner_ + 1

        return tgt, num_leaf+num_leaf_, num_inner+num_inner_

    else:
        return tgt, num_leaf, num_inner

def get_node(seq, start, id2triple):
    cursor = start
    node_triples = []
    logical_rels = []

    while cursor < len(seq):

        if isinstance(seq[cursor], int):
            triple_id = seq[cursor]
            assert triple_id < len(id2triple)
            triple = id2triple[triple_id]
            if triple not in node_triples:
                node_triples.append(triple)
            cursor += 1

        elif seq[cursor] in ['或']:
            logical_rels.append('or')
            cursor += 1

        elif seq[cursor] in ['且', '和']:
            logical_rels.append('and')
            cursor += 1

        elif seq[cursor] in ['，', '。']:
            break

        else:
            cursor += 1

    # node_triples = list(set(node_triples))

    logical_rel = 'null'
    if len(node_triples) > 1:
        if len(logical_rels) > 0:
            logical_rel = Counter(logical_rels).most_common(1)[0][0]
        else:
            logical_rel = 'and'

    return node_triples, logical_rel, cursor

def is_completed(tree, start):
    if start >= len(tree):
        return False, start

    if tree[start]['role'] == 'D':
        return True, start

    elif tree[start]['role'] == 'C':
        left_flag, left_end = is_completed(tree, start+1)
        right_flag, right_end = is_completed(tree, left_end+1)

        flag = left_flag and right_flag
        return flag, right_end
