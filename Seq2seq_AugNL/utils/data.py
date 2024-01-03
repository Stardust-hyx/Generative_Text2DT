import os, pickle, copy, sys
import torch, random
from utils.alphabet import Alphabet
from transformers import BertTokenizer, BertTokenizerFast

def preorder_traverse(tree, start, list_triple2relID, relations, side='left'):
    if start >= len(tree):
        return None, ''

    root = tree[start]
    triple2relID = list_triple2relID[start]
    role = root['role']
    tgt_seq = ''

    if role == 'C':
        if side == 'left':
            tgt_seq += '若'
        else:
            tgt_seq += '否则，若'

        tgt_seq += read_node(root, triple2relID, relations)
        tgt_seq += '，'

        left_end, left_tgt_seq = preorder_traverse(tree, start+1, list_triple2relID, relations)
        tgt_seq += left_tgt_seq

        right_end, right_tgt_seq = preorder_traverse(tree, left_end+1, list_triple2relID, relations, side='right')
        tgt_seq += right_tgt_seq

        end = right_end

    elif role == 'D':
        if len(root['triples']) > 0:
            if side == 'left':
                tgt_seq += '则'
            else:
                tgt_seq += '否则'

            tgt_seq += read_node(root, triple2relID, relations)
            tgt_seq += '，' if any([len(x['triples']) > 0 for x in tree[start+1:]]) else '。'
        
        end = start

    return end, tgt_seq


def read_node(node, triple2relID, relations):
    role = node['role']
    triples = [tuple(t) for t in node['triples']]
    logical_rel = node['logical_rel']

    if logical_rel == 'or':
        conjunction = '或'
    elif role == 'C':
        conjunction = '且'
    else:
        conjunction = '和'

    # if logical_rel == 'or':
    #     conjunction = '或'
    # else:
    #     conjunction = '和'

    clauses = []
    visited = set()
    for i, t1 in enumerate(triples):
        if t1 in visited:
            continue
        id1 = triple2relID[t1]

        clause = [id1]
        visited.add(t1)

        for t2 in triples[i+1:]:
            if t2 in visited:
                continue

            if t1[-1] == t2[0] or t1[0] == t2[-1]:
                id2 = triple2relID[t2]
                clause.append(id2)
                visited.add(t2)

        # if len(clause) > 2:
        #     print([relations[id] for id in clause])

        clause = sorted(clause, key=lambda x: location(x, relations))
        clauses.append(clause)

    clauses = sorted(clauses, key=lambda x: clause_location(x, relations))
    clauses = conjunction.join([''.join([enclose(x) for x in clause]) for clause in clauses])

    return clauses

def location(relID, relations):
    rel = relations[relID]
    return sum(rel[0][:2] + rel[2][:2])
    # return (sum(rel[2][:2]), sum(rel[0][:2]))

def clause_location(clause, relations):
    return min([location(relID, relations) for relID in clause])
    # return sum([location(triple, triple2relID, relations) for triple in clause]) / len(clause)


def is_ahead_of(a, b):
    return sum(a[0][:2] + a[2][:2]) < sum(b[0][:2] + b[2][:2])

def enclose(x):
    return '<<' + str(x) + '>>'


def text2dt_data_process(input_doc, relational_alphabet, entity_type_alphabet, tokenizer, structure_token_mapping=None,
                        evaluate=False, repeat_gt_entities=-1, repeat_gt_triples=-1):

    samples = []
    total_triples = 0
    max_triples = 0
    max_entities = 0
    max_len_mention = 0
    num_samples = 0
    print(input_doc)

    type_len_2_mentions = dict()

    with open(input_doc) as f:
        lines = f.readlines()
        lines = [eval(ele) for ele in lines]
    for idx, line in enumerate(lines):
        # print(line)
        text = line["text"]
        enc = tokenizer(text, add_special_tokens=True)
        sent_id = enc['input_ids']

        sent_seg_encoding = [0] * len(sent_id)
        context2token_masks = None
        token_masks = [1] * len(sent_id)

        char_to_bep = dict()
        bep_to_char = dict()
        for i in range(len(text)):
            bep_index = enc.char_to_token(i)
            char_to_bep[i] = bep_index
            if bep_index in bep_to_char:
                left, right = bep_to_char[bep_index][0], bep_to_char[bep_index][-1]
                bep_to_char[bep_index] = [left, max(right, i)]
            else:
                bep_to_char[bep_index] = [i, i]
        
        target = {"relation": [], "head_start_index": [], "head_end_index": [], "tail_start_index": [], "tail_end_index": [],
                    "head_mention": [], "tail_mention": [], 'head_part_labels': [], 'tail_part_labels': [], 'head_type': [], 'tail_type': [],
                    "ent_type": [], "ent_start_index": [], "ent_end_index": [], "ent_part_labels": [], 'ent_have_rel': [],
                }


        tree = line["tree"]

        if evaluate:
            triples = line["relations"]
            for triple in triples:
                relation_id = relational_alphabet.get_index(triple[1])
                h_mention = triple[0][-1]
                t_mention = triple[2][-1]

                max_len_mention = max(max_len_mention, len(h_mention))
                max_len_mention = max(max_len_mention, len(t_mention))

                target["relation"].append(relation_id)
                target["head_mention"].append(h_mention)
                target["tail_mention"].append(t_mention)

            samples.append([idx, sent_id, target, tree, text, bep_to_char, sent_seg_encoding, context2token_masks, token_masks, None, None])

            total_triples += len(triples)
            max_triples = max(max_triples, len(triples))
            num_samples += 1

        else:
            repeat_num = 1
            triples = line["relations"]
            entities = line["entities"]
            set_head_tail = set()
            mention2etype = {}
            for ent in entities:
                mention = ent[2]
                mention_len = len(mention)
                etype = ent[-1]

                mention2etype[mention] = etype

                if (etype, mention_len) not in type_len_2_mentions:
                    type_len_2_mentions[(etype, mention_len)] = []
                if mention not in type_len_2_mentions[(etype, mention_len)]:
                    type_len_2_mentions[(etype, mention_len)].append(mention)

            list_triple2relID = []
            visited = set()
            for node in tree:
                triple2relID = dict()
                for triple in node['triples']:
                    triple = tuple(triple)
                    candidates = []
                    for relID, rel in enumerate(line["relations"]):
                        if (rel[0][-1], rel[1], rel[2][-1]) == triple:
                            candidates.append(relID)
                            if relID not in visited:
                                triple2relID[triple] = relID
                                visited.add(relID)
                                break
                    if triple not in triple2relID:
                        triple2relID[triple] = candidates[-1]

                list_triple2relID.append(triple2relID)

            _, tgt_seq = preorder_traverse(tree, 0, list_triple2relID, triples)
            enc = tokenizer(tgt_seq, add_special_tokens=True)
            token_ids = enc['input_ids']
            tgt_seq_ids = []

            num_structure_tokens = len(structure_token_mapping)
            ori_vocab_size = tokenizer.vocab_size
            for token_id in token_ids:
                if token_id in structure_token_mapping:
                    tgt_seq_ids.append(structure_token_mapping[token_id])
                else:
                    assert token_id >= ori_vocab_size
                    tgt_seq_ids.append(token_id - ori_vocab_size + num_structure_tokens)

            # print(text)
            # print(tgt_seq)
            # print(token_ids)
            # print(tokenizer.convert_ids_to_tokens(token_ids))
            # print(tgt_seq_ids)
            # print()

            relID2triples = None

            triples_ = triples
            if repeat_gt_triples != -1 and len(triples) > 0:
                # unfixed
                repeat_gt_triples = max(repeat_gt_triples, len(triples))
                k = repeat_gt_triples // len(triples)
                triples_ = triples * k

                relID2triples = dict([(i, [len(triples)*j + i for j in range(k)])
                                        for i in range(len(triples))])

            for triple in triples_:
                head, rel_type, tail = triple[0], triple[1], triple[2]
                relation_id = relational_alphabet.get_index(rel_type)
                h_start, h_end, h_mention = head
                t_start, t_end, t_mention = tail

                assert text[h_start: h_end] == h_mention
                assert text[t_start: t_end] == t_mention

                head_start_index, head_end_index = char_to_bep[h_start], char_to_bep[h_end-1]
                tail_start_index, tail_end_index = char_to_bep[t_start], char_to_bep[t_end-1]

                max_len_mention = max(max_len_mention, h_end-h_start+1)
                max_len_mention = max(max_len_mention, t_end-t_start+1)

                target["relation"].append(relation_id)
                target["head_start_index"].append(head_start_index)
                target["head_end_index"].append(head_end_index)
                target["tail_start_index"].append(tail_start_index)
                target["tail_end_index"].append(tail_end_index)
                target["head_mention"].append(h_mention)
                target["tail_mention"].append(t_mention)
                target["head_type"].append(entity_type_alphabet.get_index(mention2etype[h_mention]))
                target["tail_type"].append(entity_type_alphabet.get_index(mention2etype[t_mention]))

                head_part_labels = [0.0] * len(sent_id)
                for index in range(head_start_index, head_end_index+1):
                    head_part_labels[index] = 0.5
                head_part_labels[head_start_index] = 1.0
                head_part_labels[head_end_index] = 1.0
                target["head_part_labels"].append(head_part_labels)
                
                tail_part_labels = [0.0] * len(sent_id)
                for index in range(tail_start_index, tail_end_index+1):
                    tail_part_labels[index] = 0.5
                tail_part_labels[tail_start_index] = 1.0
                tail_part_labels[tail_end_index] = 1.0
                target["tail_part_labels"].append(tail_part_labels)

                set_head_tail.add(tuple(head[:2]))
                set_head_tail.add(tuple(tail[:2]))

            entities_ = entities
            if repeat_gt_entities != -1 and len(entities) > 0:
                assert repeat_gt_entities > len(entities)
                k = repeat_gt_entities // len(entities)
                m = repeat_gt_entities % len(entities)
                entities_ = entities * k
                entities_ += entities[:m]

            for ent in entities_:
                ent_type = 'ENTITY' if len(ent) == 3 else ent[-1]

                ent_type_id = entity_type_alphabet.get_index(ent_type)
                ent_start, ent_end = ent[0], ent[1]
                ent_start_index, ent_end_index = char_to_bep[ent_start], char_to_bep[ent_end-1]

                if text[ent_start: ent_end] != ent[2]:
                    print(idx)
                    print(line)
                    print(ent)
                    exit(0)

                target["ent_type"].append(ent_type_id)
                target["ent_start_index"].append(ent_start_index)
                target["ent_end_index"].append(ent_end_index)

                ent_have_rel = 1 if ent[:2] in set_head_tail else 0
                target["ent_have_rel"].append(ent_have_rel)

                # bio_labels[ent_start_index] = 1
                # for index in range(ent_start_index+1, ent_end_index+1):
                #     bio_labels[index] = 2

                ent_part_labels = [0.0] * len(sent_id)
                for index in range(ent_start_index, ent_end_index+1):
                    ent_part_labels[index] = 0.5
                ent_part_labels[ent_start_index] = 1.0
                ent_part_labels[ent_end_index] = 1.0
                target["ent_part_labels"].append(ent_part_labels)

            for _ in range(repeat_num):
                samples.append([idx, sent_id, target, tree, text, bep_to_char, sent_seg_encoding, context2token_masks, token_masks, tgt_seq_ids, relID2triples])

            total_triples += len(triples)
            max_triples = max(max_triples, len(triples))
            max_entities = max(max_entities, len(entities))
            num_samples += 1

    print('[num samples]:', num_samples)
    print('[avg triples]:', total_triples / num_samples)
    print('[max triples]:', max_triples)
    if not evaluate:
        print('[max entities]:', max_entities)
    print('[max len mention]:', max_len_mention)
    print()

    return samples, type_len_2_mentions

class Data:
    def __init__(self):
        self.relational_alphabet = Alphabet("Relation", unkflag=False, padflag=False)
        self.entity_type_alphabet = Alphabet("Entity", unkflag=False, padflag=False)
        self.train_loader = []
        self.valid_loader = []
        self.test_loader = []
        self.weight = {}

        self.tokenizer = None
        self.structure_token_ids = None

    def show_data_summary(self):
        print("DATA SUMMARY START:")
        print("     Relation Alphabet Size: %s" % self.relational_alphabet.size())
        print("     Ent Type Alphabet Size: %s" % self.entity_type_alphabet.size())
        print("     Train  Instance Number: %s" % (len(self.train_loader)))
        print("     Valid  Instance Number: %s" % (len(self.valid_loader)))
        print("     Test   Instance Number: %s" % (len(self.test_loader)))
        print("DATA SUMMARY END.")
        sys.stdout.flush()

    def generate_instance(self, args):
        if args.dataset_name == 'Text2DT':
            tokenizer = BertTokenizerFast.from_pretrained(args.cpt_directory, do_lower_case=True)
            print('tokenizer.vocab_size', tokenizer.vocab_size)

            structure_tokens = [tokenizer.sep_token, tokenizer.cls_token] + ['若', '则', '否', '或', '且', '和', '，', '。']
            structure_token_ids = tokenizer.convert_tokens_to_ids(structure_tokens)
            structure_token_mapping = dict()
            for i, token_id in enumerate(structure_token_ids):
                assert token_id != tokenizer.unk_token_id
                structure_token_mapping[token_id] = i

            sorted_add_tokens = ['<<' + str(i) + '>>' for i in range(args.num_generated_triples)]
            # sorted_add_tokens = sorted(, key=lambda x: len(x), reverse=True)
            for tok in sorted_add_tokens:
                assert tokenizer.convert_tokens_to_ids([tok])[0] == tokenizer.unk_token_id
            tokenizer.add_tokens(sorted_add_tokens)

            self.train_loader, type_len_2_mentions = text2dt_data_process(args.train_file, self.relational_alphabet, self.entity_type_alphabet, 
                                                                          tokenizer, structure_token_mapping,
                                                                          repeat_gt_entities=args.repeat_gt_entities, repeat_gt_triples=args.repeat_gt_triples)
            self.weight = copy.deepcopy(self.relational_alphabet.index_num)

            if not os.path.exists(args.aug_train_file):
                data_augmentation(args.train_file, args.aug_train_file, type_len_2_mentions, repeat=1)

            self.aug_train_loader, _ = text2dt_data_process(args.aug_train_file, self.relational_alphabet, self.entity_type_alphabet,
                                                            tokenizer, structure_token_mapping,
                                                            repeat_gt_entities=args.repeat_gt_entities, repeat_gt_triples=args.repeat_gt_triples)
            
            self.train_loader += self.aug_train_loader

            self.valid_loader, _ = text2dt_data_process(args.valid_file, self.relational_alphabet, self.entity_type_alphabet, tokenizer, evaluate=True)
            self.test_loader, _ = text2dt_data_process(args.test_file, self.relational_alphabet, self.entity_type_alphabet, tokenizer, evaluate=True)

        self.relational_alphabet.close()
        self.tokenizer = tokenizer
        self.structure_token_ids = structure_token_ids
        print('structure_token_ids', structure_token_ids)
        print('sorted_add_tokens', sorted_add_tokens)
        print('tokenizer.vocab_size', tokenizer.vocab_size)

        # exit(0)


def build_data(args):

    file = args.generated_data_directory + args.dataset_name + "_" + args.model_name + "_data.pickle"
    # if os.path.exists(file) and not args.refresh:
    #     data = load_data_setting(args)
    # else:
    data = Data()
    data.generate_instance(args)
    save_data_setting(data, args)
    return data


def save_data_setting(data, args):
    new_data = copy.deepcopy(data)
    data.show_data_summary()
    if not os.path.exists(args.generated_data_directory):
        os.makedirs(args.generated_data_directory)
    saved_path = args.generated_data_directory + args.dataset_name + "_" + args.model_name + "_data.pickle"
    with open(saved_path, 'wb') as fp:
        pickle.dump(new_data, fp)
    print("Data setting is saved to file: ", saved_path)


def load_data_setting(args):

    saved_path = args.generated_data_directory + args.dataset_name + "_" + args.model_name + "_data.pickle"
    with open(saved_path, 'rb') as fp:
        data = pickle.load(fp)
    print("Data setting is loaded from file: ", saved_path)
    data.show_data_summary()
    return data


def extend_tensor(tensor, extended_shape, fill=0):
    tensor_shape = tensor.shape

    extended_tensor = torch.zeros(extended_shape, dtype=tensor.dtype).to(tensor.device)
    extended_tensor = extended_tensor.fill_(fill)

    if len(tensor_shape) == 1:
        extended_tensor[:tensor_shape[0]] = tensor
    elif len(tensor_shape) == 2:
        extended_tensor[:tensor_shape[0], :tensor_shape[1]] = tensor
    elif len(tensor_shape) == 3:
        extended_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2]] = tensor
    elif len(tensor_shape) == 4:
        extended_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2], :tensor_shape[3]] = tensor

    return extended_tensor

def padded_stack(tensors, padding=0):
    dim_count = len(tensors[0].shape)

    max_shape = [max([t.shape[d] for t in tensors]) for d in range(dim_count)]
    padded_tensors = []

    for t in tensors:
        e = extend_tensor(t, max_shape, fill=padding)
        padded_tensors.append(e)

    stacked = torch.stack(padded_tensors)
    return stacked

def overlap(span1, span2):
    return not (span1[1] <= span2[0] or span2[1] <= span1[0])

def LCS(text1: str, text2: str) -> int:
    n = len(text1)
    m = len(text2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i][j - 1], dp[i - 1][j])

    return dp[n][m]

def data_augmentation(ori_fn, tgt_fn, type_len_2_mentions, repeat=1):
    random.seed(2021)

    with open(ori_fn) as f:
        lines = f.readlines()
        lines = [eval(ele) for ele in lines]

    new_lines = []

    for line in lines:
        text = line["text"]
        relations = line["relations"]
        entities = line["entities"]
        tree = line["tree"]

        skip_mentions = set()
        for i, e1 in enumerate(entities):
            for j, e2 in enumerate(entities):
                if e1 == e2:
                    continue
                mention1 = e1[2]
                mention2 = e2[2]
                max_len_LCS = LCS(mention1, mention2)
                if overlap(e1, e2) or max_len_LCS > max(len(mention1), len(mention2))/2:
                    skip_mentions.add(mention1)
                    skip_mentions.add(mention2)

        for _ in range(repeat):
            type_len_2_mentions_ = copy.deepcopy(type_len_2_mentions)
            text_ = text
            entites_ = []
            mention_2_substitute = dict()
            for i, ent in enumerate(entities):
                ent_start, ent_end, mention, etype = ent

                if mention in mention_2_substitute:
                    substitute = mention_2_substitute[mention]

                else:
                    candidates = type_len_2_mentions_[(etype, len(mention))]
                    
                    if len(candidates) == 1 or mention in skip_mentions:
                        substitute = mention
                    else:
                        substitute = random.choice(candidates)

                    mention_2_substitute[mention] = substitute
                
                text_ = text_[:ent_start] + substitute + text_[ent_end:]
                entites_.append((ent_start, ent_end, substitute, etype))

            relations_ = []
            triple2substitute = dict()

            for relation in relations:
                head, rel_type, tail = relation
                h_start, h_end, h_mention = head
                t_start, t_end, t_mention = tail
                triple = (h_mention, rel_type, t_mention)

                if triple in triple2substitute:
                    h_mention_ = triple2substitute[triple][0]
                    t_mention_ = triple2substitute[triple][-1]

                else:
                    h_mention_ = mention_2_substitute[h_mention]
                    t_mention_ = mention_2_substitute[t_mention]
                    triple2substitute[triple] = (h_mention_, rel_type, t_mention_)

                head_ = (h_start, h_end, h_mention_)
                tail_ = (t_start, t_end, t_mention_)
                relations_.append((head_, rel_type, tail_))

            tree_ = []
            for node in tree:
                role = node['role']
                triples = node['triples']
                logical_rel = node['logical_rel']

                triples_ = []
                for triple in triples:
                    triple_ = triple2substitute[tuple(triple)]
                    triples_.append(triple_)
                
                tree_.append({'role': role, 'triples': triples_, 'logical_rel': logical_rel})

            # print(text)
            # print(text_)
            # print()
                
            new_line = {'text': text_, 'relations': relations_, 'entities': entites_, 'tree': tree_}
            new_lines.append(new_line)
    
    
    out_f = open(tgt_fn, 'w')
    for new_line in new_lines:
        print(new_line, file=out_f)
