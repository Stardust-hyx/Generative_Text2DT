
from text2dt_eval.eval_func import eval

def text2dt_metric(gold_data, predict_data, depth=None):
    gold_tree_num, correct_tree_num = 0.000001, 0.000001
    gold_triplet_num, predict_triplet_num, correct_triplet_num = 0.000001, 0.000001, 0.000001
    gold_path_num, predict_path_num, correct_path_num= 0.000001, 0.000001, 0.000001
    gold_node_num, predict_node_num, correct_node_num = 0.000001, 0.000001, 0.000001

    edit_dis = 0

    for i in range(len(predict_data)):
        if depth and get_depth(gold_data[i]) != depth:
            continue
        # print(i)
        tmp= eval(predict_data[i]['tree'], gold_data[i]['tree'])
        gold_tree_num += tmp[0]
        correct_tree_num += tmp[1]
        correct_triplet_num += tmp[2]
        predict_triplet_num += tmp[3]
        gold_triplet_num += tmp[4]
        correct_path_num += tmp[5]
        predict_path_num += tmp[6]
        gold_path_num += tmp[7]
        edit_dis += tmp[8]
        correct_node_num += tmp[9]
        predict_node_num += tmp[10]
        gold_node_num += tmp[11]

    tree_acc= correct_tree_num/gold_tree_num
    triple_p = correct_triplet_num/predict_triplet_num
    triple_r = correct_triplet_num/gold_triplet_num
    triple_f1 = 2 * triple_p * triple_r / (triple_p + triple_r)
    path_f1 =2* (correct_path_num/predict_path_num) *(correct_path_num/gold_path_num)/(correct_path_num/predict_path_num + correct_path_num/gold_path_num)
    tree_edit_distance=edit_dis/gold_tree_num
    node_f1 =2* (correct_node_num/predict_node_num) *(correct_node_num/gold_node_num)/(correct_node_num/predict_node_num + correct_node_num/gold_node_num)

    print('[Triple_P]: %.6f;\t [Triple_R]: %.6f\t [Triple_F1]: %.6f' % (triple_p, triple_r, triple_f1))
    print("[Node_F1] : %.6f;\t [Path_F1] : %.6f\t [Edit_Dist]: %.6f" % (node_f1, path_f1, tree_edit_distance))
    print('[Tree_ACC]: %.6f' % tree_acc)

    return {'f1': triple_f1, 'tree_acc': tree_acc, 'path_f1': path_f1}


def error_count(gold_data, predict_data):
    structure_error = 0
    logic_rel_error = 0
    triple_error = 0

    rel_type_error = 0
    entity_error = 0

    for i in range(len(predict_data)):
        pred, gold = predict_data[i], gold_data[i]
        if 'tree' in pred:
            pred = pred['tree']
        if 'tree' in gold:
            gold = gold['tree']

        pred_node_roles = [x['role'] for x in pred]
        gold_node_roles = [x['role'] for x in gold]
        if pred_node_roles != gold_node_roles:
            structure_error += 1
            continue

        pred_logical_rels = [x['logical_rel'] for x in pred]
        gold_logical_rels = [x['logical_rel'] for x in gold]
        pred_triples = [[tuple(triple) for triple in x['triples']] for x in pred if len(x['triples']) > 0]
        gold_triples = [[tuple(triple) for triple in x['triples']] for x in gold if len(x['triples']) > 0]
        pred_triples = sum(pred_triples, [])
        gold_triples = sum(gold_triples, [])

        all_triples_same = len(pred_triples) == len(gold_triples) and all(x in gold_triples for x in pred_triples)
        if all_triples_same and pred_logical_rels != gold_logical_rels:
            logic_rel_error += 1
            continue

        if not all_triples_same:
            triple_error += 1
            
        if len(pred_triples) < len(gold_triples):
            rel_type_error += len(gold_triples) - len(pred_triples)
        
        for x in pred_triples:
            x_type = x[1]
            x_ent = (x[0], x[2])
            flag_found = False
            for y in gold_triples:
                if x_ent == (y[0], y[2]):
                    flag_found = True
                    if x_type != y[1]:
                        rel_type_error += 1
                    break
                # elif overlap(x_ent, (y[0], y[2])):
                #     flag_found = True
                #     entity_error += 1
                #     if 
                #     break
            
            if not flag_found:
                entity_error += 1

    overall_errors = structure_error + logic_rel_error + triple_error

    print(f'# overall errors: {overall_errors}')
    print(f'structure_error: {structure_error} ({structure_error/overall_errors})')
    print(f'logic_rel_error: {logic_rel_error} ({logic_rel_error/overall_errors})')
    print(f'triple_error: {triple_error} ({triple_error/overall_errors})')
    print()
    print(f'rel_type_error: {rel_type_error} ({rel_type_error/(rel_type_error+entity_error)})')
    print(f'entity_error: {entity_error} ({entity_error/(rel_type_error+entity_error)})')

def get_depth(tree):
    if 'tree' in tree:
        tree = tree['tree']
    num_nodes = len(tree)
    return (num_nodes+1)/2