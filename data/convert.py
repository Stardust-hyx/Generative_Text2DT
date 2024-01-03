import os, json

ent_label_mapping = {
    "症状": "临床表现",
    "情况": "基本情况",
    "药物": "药物",
    "用法": "用法用量",
    "治疗": "治疗方法",
}

# 临床表现，用药，治疗方案，用法，基本情况，慎用
rel_label_mapping = {
    "临床表现": "临床表现",
    "治疗药物": "用药",
    "治疗方案": "治疗方案",
    "用法用量": "用法",
    "基本情况": "基本情况",
    "禁用药物": "慎用",
}

def preorder_traverse(tree, start, list_triple2relID, relations, side='left', TreeS=False, triple_2_clauses=None):
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

        tgt_seq += read_node(root, triple2relID, relations, TreeS, triple_2_clauses)
        tgt_seq += '，'

        left_end, left_tgt_seq = preorder_traverse(tree, start+1, list_triple2relID, relations, TreeS=TreeS, triple_2_clauses=triple_2_clauses)
        tgt_seq += left_tgt_seq

        right_end, right_tgt_seq = preorder_traverse(tree, left_end+1, list_triple2relID, relations, side='right', TreeS=TreeS, triple_2_clauses=triple_2_clauses)
        tgt_seq += right_tgt_seq

        end = right_end

    elif role == 'D':
        if len(root['triples']) > 0:
            if side == 'left':
                tgt_seq += '则'
            else:
                tgt_seq += '否则'

            tgt_seq += read_node(root, triple2relID, relations, TreeS, triple_2_clauses)
            tgt_seq += '，' if any([len(x['triples']) > 0 for x in tree[start+1:]]) else '。'
        
        end = start

    return end, tgt_seq


def read_node(node, triple2relID, relations, TreeS=False, triple_2_clauses=None):
    role = node['role']
    triples = [tuple(t) for t in node['triples']]
    logical_rel = node['logical_rel']

    if logical_rel == 'or':
        conjunction = '或'
    elif role == 'C':
        conjunction = '且'
    else:
        conjunction = '和'

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

        try:
            clause = sorted(clause, key=lambda x: location(x, relations))
        except:
            pass

        clauses.append(clause)

        if not TreeS:
            clause_ = [(tuple(relations[x][0]), relations[x][1], tuple(relations[x][2])) for x in clause]
            if clause_[0] not in triple_2_clauses:
                triple_2_clauses[clause_[0]] = []
            triple_2_clauses[clause_[0]].append(clause_)

    try:
        clauses = sorted(clauses, key=lambda x: clause_location(x, relations))
    except:
        pass

    if TreeS:
        clauses = conjunction.join(['...' for clause in clauses])
    else:
        clauses = conjunction.join([''.join([linearize_triple(relations[x]) for x in clause]) for clause in clauses])

    return clauses

def location(relID, relations):
    rel = relations[relID]
    return sum(rel[0][:2] + rel[2][:2])
    # return (sum(rel[2][:2]), sum(rel[0][:2]))

def clause_location(clause, relations):
    return min([location(relID, relations) for relID in clause])

# def linearize_clause(clause, relations):
#     for tripleID in clause:
#         triple = relations[tripleID]
#     return

def linearize_triple(triple):
    if isinstance(triple[0], str):
        return '('+ triple[0] + ', ' + rel_label_mapping[triple[1]] + ', ' + triple[2] + ')'
    else:
        return '('+ triple[0][-1] + ', ' + rel_label_mapping[triple[1]] + ', ' + triple[2][-1] + ')'

# def linearize_triple(triple):
#     if isinstance(triple[0], str):
#         return '(' + rel_label_mapping[triple[1]] + ', ' + triple[2] + ')'
#     else:
#         return '(' + rel_label_mapping[triple[1]] + ', ' + triple[2][-1] + ')'

def convert_sample(input_doc, NER=False, RE=False, TreeS=False, COT=False):
    samples = []

    with open(input_doc) as f:
        lines = f.readlines()
        lines = [eval(ele) for ele in lines]
    for idx, line in enumerate(lines):
        # print(line)
        text = line["text"]
        tree = line["tree"]
        entities = line['entities'] if 'entities' in line else None
        triples = line["relations"]
        triples_in_tree = []
        
        list_triple2relID = []
        visited = set()
        for node in tree:
            triple2relID = dict()
            for triple in node['triples']:
                triple = tuple(triple)
                # print(triple)
                if triple not in triples_in_tree:
                    triples_in_tree.append(triple)

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

        triple_2_clauses = dict()
        _, tree_seq = preorder_traverse(tree, 0, list_triple2relID, triples, triple_2_clauses=triple_2_clauses)

        sample = {'input': text, 'target': tree_seq, 'NER_target': '', 'RE_target': '', 'TreeS_target': '', 'COT_target': '', 'rel_2_triples': ''}

        if NER and entities is not None:
            # mentions = []
            # for ent in entities:
            #     mention = ent[2]
            #     etype = ent[-1]
            #     if etype in ent_label_mapping and mention not in mentions:
            #         mentions.append(mention)

            # NER_target = '[' + ', '.join(["\"" + mention + "\"" for mention in mentions]) + ']'

            etype_2_entities = dict([(etype, []) for etype in ent_label_mapping.values()])
            for entity in entities:
                try:
                    etype = ent_label_mapping[entity[-1]]
                except:
                    continue
                mention = entity[-2]
                if mention not in etype_2_entities[etype]:
                    etype_2_entities[etype].append(mention)

            # NER_target = '\n'.join([etype + '：' + ('，'.join(entities) if entities else '无') for (etype, entities) in etype_2_entities.items()])
            # NER_target = json.dumps(etype_2_entities, ensure_ascii=False)

            sample['NER_target'] = etype_2_entities

        if RE:
            try:
                triples_ = sorted(triples, key=lambda x: sum(x[0][:2] + x[2][:2]))
            except:
                triples_ = triples

            rel_2_triples = dict([(rel, []) for rel in rel_label_mapping.values()])
            for triple in triples_:
                head, rel, tail = triple
                rel = rel_label_mapping[rel]
                head = head[-1]
                tail = tail[-1]
                if (head, rel, tail) not in rel_2_triples[rel]:
                    rel_2_triples[rel].append((head, rel, tail))

            sample['rel_2_triples'] = rel_2_triples

            RE_target = '['
            visited = []
            for i, triple in enumerate(triples_):
                triple = (tuple(triple[0]), triple[1], tuple(triple[2]))
                if triple in visited:
                    continue

                if triple in triple_2_clauses:
                    clauses = triple_2_clauses[triple]
                    try:
                        clauses = sorted(clauses, key=lambda x: max(sum(t[0][:2] + t[0][:2]) for t in x))
                    except:
                        pass
                    
                    clause = clauses[0]
                    linearized = ', '.join([linearize_triple(x) for x in clause])
                    for t in clause:
                        visited.append(t)
                else:
                    linearized = linearize_triple(triple)
                    visited.append(triple)

                if i != 0:
                    RE_target += ', '
                RE_target += linearized
            RE_target += ']'

            sample['RE_target'] = RE_target

        if TreeS:
            _, TreeS_target = preorder_traverse(tree, 0, list_triple2relID, triples, TreeS=True)
            sample['TreeS_target'] = TreeS_target

        if COT:
            COT_target = '诊疗逻辑框架:' + TreeS_target + '\n\n关系:' + RE_target + '\n\n诊疗决策过程:' + tree_seq
            sample['COT_target'] = COT_target
        
        samples.append(sample)

    return samples

if __name__ == '__main__':
    in_dir = './Text2DT'
    out_dir = './Text2DT_SFT'

    os.makedirs(out_dir, exist_ok=True)

    for root, dirs, files in os.walk(in_dir):
        for fn in files:
            if fn[-4:] != '.txt':
                continue

            samples = convert_sample(os.path.join(in_dir, fn), NER=True, RE=True, TreeS=True, COT=True)
            fname = fn.split('.')[0]
            print(f'{fn} -> {fname}.json')

            json_f = open(os.path.join(out_dir, fname + '.json'), 'w', encoding='utf-8')
            for sample in samples:
                json_line = json.dumps(sample, ensure_ascii=False)
                print(json_line, file=json_f)
