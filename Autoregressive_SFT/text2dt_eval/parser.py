import re

rel_name_2_label = {
    "临床表现": "临床表现",
    "用药": "治疗药物",
    "治疗方案": "治疗方案",
    "用法": "用法用量",
    "基本情况": "基本情况",
    "慎用": "禁用药物",
}

def parsing_RE(seq, prefix=''):
    match = re.search(f'{prefix}.*', seq)
    res = {'entities': [], 'triples': []}

    if not match:
        print(seq)
        return res

    cursor = match.span()[0] + len(prefix)

    while cursor < len(seq):
        if seq[cursor] == "(":
            # matching for (xxx, xxx, xxx)
            triple_span = re.match(r'\(.*?,.*?,[^\n\r\(]*?(\(.*?\))*?[^\n\r\(]*?\)', seq[cursor:])
            if triple_span is None:
                # print(seq[cursor:])
                cursor += 1
                continue

            triple_span = triple_span.span()
            triple_str = seq[cursor+triple_span[0]: cursor+triple_span[1]]
            len_span = len(triple_str)
            cursor += len_span
            
            triple_split = triple_str[1:-1].split(',')
            triple = [x.strip() for x in triple_split]

            if len(triple_split) == 3 and triple[1] in rel_name_2_label:
                rel_label = rel_name_2_label[triple[1]]
            else:
                print(triple_split)
                continue

            triple = (triple[0], rel_label, triple[2])
            if triple not in res['triples']:
                res['triples'].append(triple)

            if triple[0] not in res['entities']:
                res['entities'].append(triple[0])
            if triple[2] not in res['entities']:
                res['entities'].append(triple[2])

        else:
            cursor += 1

    return res


def parsing(seq, prefix='', output_dict=False):
    # print(seq)
    match = re.search(f'{prefix}.*', seq)
    res = []

    if match:
        start = match.span()[0] + len(prefix)
        res, _, _ = scan_seq([], seq[start:], 0)
        
    if res == []:
        # empty tree
        # print(seq)
        res = [{'role': 'C', 'triples': [], 'logical_rel': 'null'}]
        res += [{'role': 'D', 'triples': [], 'logical_rel': 'null'}] * 2

    if output_dict:
        res = {"tree": res}
    
    return res

def scan_seq(tgt, seq, start, num_leaf=0, num_inner=0):

    flag, _ = is_completed(tgt, 0)
    if start >= len(seq) or flag:
        return tgt, num_leaf, num_inner

    if seq[start: start+4] == '否则,若':
        role = 'C'
        node_triples, logical_rel, end = get_node(seq, start+4, role)
        tgt.append({'role': role, 'triples': node_triples, 'logical_rel': logical_rel})

        tgt, num_leaf_, num_inner_ = scan_seq(tgt, seq, end+1, 0, 1)

        if num_leaf_ < num_inner_ + 1:
            tgt += [{'role': 'D', 'triples': [], 'logical_rel': 'null'}] * (num_inner_ + 1 - num_leaf_)
            num_leaf_ = num_inner_ + 1

        return tgt, num_leaf+num_leaf_, num_inner+num_inner_

    elif seq[start: start+2] == '否则':
        role = 'D'
        node_triples, logical_rel, end = get_node(seq, start+2, role)

        if len(node_triples) > 0:
            tgt.append({'role': role, 'triples': node_triples, 'logical_rel': logical_rel})
            num_leaf += 1
        tgt, num_leaf, num_inner = scan_seq(tgt, seq, end+1, num_leaf, num_inner)

        return tgt, num_leaf, num_inner

    elif seq[start: start+1] == '则':
        role = 'D'
        node_triples, logical_rel, end = get_node(seq, start+1, role)

        if len(node_triples) > 0:
            tgt.append({'role': role, 'triples': node_triples, 'logical_rel': logical_rel})
            num_leaf += 1
        tgt, num_leaf, num_inner = scan_seq(tgt, seq, end+1, num_leaf, num_inner)

        return tgt, num_leaf, num_inner

    elif seq[start: start+1] == '若':
        role = 'C'
        node_triples, logical_rel, end = get_node(seq, start+1, role)
        tgt.append({'role': role, 'triples': node_triples, 'logical_rel': logical_rel})

        tgt, num_leaf_, num_inner_ = scan_seq(tgt, seq, end+1, 0, 1)

        if num_leaf_ < num_inner_ + 1:
            tgt += [{'role': 'D', 'triples': [], 'logical_rel': 'null'}] * (num_inner_ + 1 - num_leaf_)
            num_leaf_ = num_inner_ + 1

        return tgt, num_leaf+num_leaf_, num_inner+num_inner_

    else:
        return tgt, num_leaf, num_inner

def get_node(seq, start, role):
    cursor = start
    node_triples = []
    logical_rels = []
    num_or = 0
    num_and = 0

    while cursor < len(seq):
        if seq[cursor] == "(":
            # matching for (xxx, xxx, xxx)
            triple_span = re.match(r'\(.*?,.*?,[^\n\r\(]*?(\(.*?\))*?[^\n\r\(]*?\)', seq[cursor:])
            if triple_span is None:
                # print(seq[cursor:])
                cursor += 1
                continue

            triple_span = triple_span.span()
            triple_str = seq[cursor+triple_span[0]: cursor+triple_span[1]]
            len_span = len(triple_str)
            cursor += len_span
            
            triple_split = triple_str[1:-1].split(',')
            triple = [x.strip() for x in triple_split]

            if len(triple_split) == 3 and triple[1] in rel_name_2_label:
                rel_label = rel_name_2_label[triple[1]]
            else:
                # print(triple_split)
                continue

            triple = (triple[0], rel_label, triple[2])
            if triple not in node_triples:
                node_triples.append(triple)

        elif seq[cursor] in ['或']:
            logical_rels.append('or')
            num_or += 1
            cursor += 1

        elif seq[cursor] in ['且']:
            logical_rels.append('and')
            num_and += 1
            cursor += 1
        
        elif seq[cursor] in ['和']:
            logical_rels.append('and')
            num_and += 1
            cursor += 1

        elif seq[cursor] in [',', '。']:
            break

        else:
            cursor += 1

    logical_rel = 'null'
    if len(node_triples) > 1:
        if num_or > num_and:
            logical_rel = 'or'
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
