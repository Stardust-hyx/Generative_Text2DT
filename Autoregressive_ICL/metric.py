
from eval_func import eval
import json

def text2dt_metric(gold_data, predict_data):
    gold_tree_num, correct_tree_num = 0.000001, 0.000001
    gold_triplet_num, predict_triplet_num, correct_triplet_num = 0.000001, 0.000001, 0.000001
    gold_path_num, predict_path_num, correct_path_num= 0.000001, 0.000001, 0.000001
    gold_node_num, predict_node_num, correct_node_num = 0.000001, 0.000001, 0.000001

    edit_dis = 0

    for i in range(len(predict_data)):
        print(i)
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

    print('[Triple_P]: %.6f;\t [Triple_R]: %.6f\t [Triple_F1]: %.6f' % (triple_p, triple_r, triple_f1), flush=True)
    print("[Node_F1] : %.6f;\t [Path_F1] : %.6f\t [Edit_Dist]: %.6f" % (node_f1, path_f1, tree_edit_distance), flush=True)
    print('[Tree_ACC]: %.6f' % tree_acc, flush=True)

    return {'triple_f1': triple_f1, 'node_f1': node_f1, 'path_f1': path_f1, 'tree_acc': tree_acc, 'path_tree_avg': (path_f1+tree_acc)/2}

if __name__ == '__main__':
    pred_fn = "pred_trees2.json"
    with open(pred_fn, "r", encoding='utf-8') as f:
        predict_data = json.load(f)
    with open("json/Text2DT_test.json", "r", encoding='utf-8') as f:
        gold_data = json.load(f)

    text2dt_metric = text2dt_metric(gold_data, predict_data)
