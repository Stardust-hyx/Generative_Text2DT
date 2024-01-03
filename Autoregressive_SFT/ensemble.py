import os, json
from text2dt_eval.metric import text2dt_metric, edit_distance

def statistic_edit_dist(dir, list_pred_files):
    list_pred_data = []
    list_edit_dist = []

    for fn in list_pred_files:
        fpath = os.path.join(dir, fn)
        f = open(fpath, 'r', encoding='utf-8')
        pred_data = json.load(f)
        list_pred_data.append(pred_data)

    print('-'*50, 'Edit Distance Statistic')

    for i, pred_data in enumerate(list_pred_data):
        for j, pred_data_ in enumerate(list_pred_data[i+1:]):
            edit_dist_ij = edit_distance(pred_data, pred_data_)
            list_edit_dist.append(edit_dist_ij)
            assert edit_dist_ij == edit_distance(pred_data_, pred_data)

            print(f'[i={i}, j={i+1+j}]', edit_dist_ij)
        print()

    print('[Avg Edit Distance]:', sum(list_edit_dist)/len(list_edit_dist))
    print('-'*50)
    print()

    return list_edit_dist

def read_pred_file(fn):
    f = open(fn, 'r', encoding='utf-8')
    pred_data = json.load(f)
    
    list_text = []
    list_tree_structure = []
    list_node_triples = []
    list_node_logic = []
    
    for pred in pred_data:
        text = pred['text']
        tree = pred['tree']

        tree_structure = ''
        node_triples = []
        node_logic = []
        for node in tree:
            role, triples, logic = node['role'], node['triples'], node['logical_rel']

            tree_structure += role
            node_triples.append(triples)
            node_logic.append(logic)

        list_text.append(text)
        list_tree_structure.append(tree_structure)
        list_node_triples.append(node_triples)
        list_node_logic.append(node_logic)

    return list_text, list_tree_structure, list_node_triples, list_node_logic


def ensemble(dir, list_pred_files, ths=None):
    text_2_list_tree_structure = dict()
    text_2_list_node_triples = dict()
    text_2_list_node_logic = dict()

    for fn in list_pred_files:
        list_text, list_tree_structure, list_node_triples, list_node_logic = read_pred_file(os.path.join(dir, fn))

        for (text, tree_structure, node_triples, node_logic) in \
            zip(list_text, list_tree_structure, list_node_triples, list_node_logic):

            if text not in text_2_list_tree_structure:
                text_2_list_tree_structure[text] = []
                text_2_list_node_triples[text] = []
                text_2_list_node_logic[text] = []

            text_2_list_tree_structure[text].append(tree_structure)
            text_2_list_node_triples[text].append(node_triples)
            text_2_list_node_logic[text].append(node_logic)

    ensemble_pred_data = []

    for (text, list_tree_structure) in text_2_list_tree_structure.items():
        tree_structure_cnt = dict()
        for tree_structure in list_tree_structure:
            if tree_structure not in tree_structure_cnt:
                tree_structure_cnt[tree_structure] = 0
            tree_structure_cnt[tree_structure] += 1

        best_tree_structure = None
        max_cnt = -1
        for (tree_structure, cnt) in tree_structure_cnt.items():
            if cnt > max_cnt:
                max_cnt = cnt
                best_tree_structure = tree_structure

        list_node_triples = text_2_list_node_triples[text]
        list_node_logic = text_2_list_node_logic[text]

        node_triples_cnt = [dict() for _ in range(len(best_tree_structure))]
        node_logic_cnt = [dict() for _ in range(len(best_tree_structure))]
        for (tree_structure, node_triples, node_logic) in \
            zip(list_tree_structure, list_node_triples, list_node_logic):

            if tree_structure != best_tree_structure:
                continue

            for i, triples in enumerate(node_triples):
                for triple in triples:
                    triple = tuple(triple)

                    if triple not in node_triples_cnt[i]:
                        node_triples_cnt[i][triple] = 0
                    node_triples_cnt[i][triple] += 1

            for i, logic in enumerate(node_logic):
                if logic not in node_logic_cnt:
                    node_logic_cnt[i][logic] = 0
                node_logic_cnt[i][logic] += 1
        
        best_node_triples = []
        for triples_cnt in node_triples_cnt:
            best_triples = []
            for triple, cnt in triples_cnt.items():
                if cnt >= min(2, max_cnt/2):
                    best_triples.append(triple)
            best_node_triples.append(best_triples)

        best_node_logic = []
        for logic_cnt in node_logic_cnt:
            tgt = None
            max_cnt_ = -1
            for (logic, cnt) in logic_cnt.items():
                if cnt > max_cnt_:
                    max_cnt_ = cnt
                    tgt = logic
            best_node_logic.append(tgt)

        tree = []
        for node_role, node_triples, node_logic in zip(best_tree_structure, best_node_triples, best_node_logic):
            node_triples_ = remove_overlap(node_triples)
            if node_logic in ['or', 'and'] and len(node_triples_) == 1:
                node_triples_ = node_triples

            tree.append({'role': node_role, 'triples': node_triples_, 'logical_rel': node_logic})

        ensemnle_pred = {'text': text, 'tree': tree}
        ensemble_pred_data.append(ensemnle_pred)

    return ensemble_pred_data

def remove_overlap(triples):
    triples_ = []
    for i, triple1 in enumerate(triples):
        overlap = False
        for j, triple2 in enumerate(triples):
            if i == j:
                continue
            if triple1[0] in triple2[0] and triple1[2] in triple2[2]:
                overlap = True

        if not overlap:
            triples_.append(triple1)

    return triples_


if __name__ == '__main__':
    dir = './outputs'

    out_fn = 'lora_RE_TreeS-aug'
    list_pred_files = ['lora_RE_TreeS-aug-2e-4-2028/pred.json', 'lora_RE_TreeS-aug-2e-4-2022/pred.json', 'lora_RE_TreeS-aug-2e-4-2024/pred.json', 'lora_RE_TreeS-aug-2e-4-2025/pred.json', 'lora_RE_TreeS-aug-2e-4-2026/pred.json']

    if len(list_pred_files) > 1:
        statistic_edit_dist(dir, list_pred_files)

    pred_data = ensemble(dir, list_pred_files)

    out_f = open(out_fn, 'w', encoding='utf-8')
    json.dump(pred_data, out_f, ensure_ascii=False, indent=2)

    gold_json_path = '../json/Text2DT_test.json'
    gold_data = json.load(open(gold_json_path, "r"))
    text2dt_metric(gold_data, pred_data)
