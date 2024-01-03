
from eval_func import eval
import json

with open("predict_example.json", "r") as f:
    predict_data = json.load(f)
with open("gold_example.json", "r") as f:
    gold_data = json.load(f)


gold_tree_num, correct_tree_num = 0.000001, 0.000001
gold_triplet_num, predict_triplet_num, correct_triplet_num = 0.000001, 0.000001, 0.000001
gold_path_num, predict_path_num, correct_path_num= 0.000001, 0.000001, 0.000001
gold_node_num, predict_node_num, correct_node_num = 0.000001, 0.000001, 0.000001

edit_dis = 0

for i in range(len(predict_data)):
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
triplet_f1 =2* (correct_triplet_num/predict_triplet_num) *(correct_triplet_num/gold_triplet_num)/(correct_triplet_num/predict_triplet_num + correct_triplet_num/gold_triplet_num)
path_f1 =2* (correct_path_num/predict_path_num) *(correct_path_num/gold_path_num)/(correct_path_num/predict_path_num + correct_path_num/gold_path_num)
tree_edit_distance=edit_dis/gold_tree_num
node_f1 =2* (correct_node_num/predict_node_num) *(correct_node_num/gold_node_num)/(correct_node_num/predict_node_num + correct_node_num/gold_node_num)

print("tree_acc, triplet_f1, path_f1, tree_edit_distance, node_f1",tree_acc, triplet_f1, path_f1, tree_edit_distance, node_f1 )
