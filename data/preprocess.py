import os
import random
import json
# from tokenization import GeneralTokenizer

def obtain_entity(entity_text, text):
    pos = text.find(entity_text)
    if pos == -1:
        entity = ['UNFOUND', entity_text]
        return entity

    entity = [pos, pos + len(entity_text), entity_text]
    
    remain = text[pos + len(entity_text):].find(entity_text)
    if remain != -1:
        entity = ['MULTI', entity_text]

    return entity


def preprocess(in_fpath, out_fpath):
    in_f = open(in_fpath, 'r', encoding='utf-8')
    out_f = open(out_fpath, 'w', encoding='utf-8')

    data = json.load(in_f)
    for d in data:
        sample = {'text': d["text"], 'relations': []}
        for node in d["tree"]:
            for triple in node["triples"]:
                h_text, rel, t_text = triple
                head = obtain_entity(h_text, sample['text'])
                tail = obtain_entity(t_text, sample['text'])

                sample['relations'].append((head, rel, tail))
        print(sample, file=out_f)


def check(fpath1, fpath2):
    """
        检查'tmp.txt'和'exact.txt'是否一致，并统计关系类别分布
    """
    f1 = open(fpath1, 'r', encoding='utf-8')
    f2 = open(fpath2, 'r', encoding='utf-8')

    data1 = []
    data2 = []
    for line in f1.readlines():
        data1.append(eval(line.strip()))
    for line in f2.readlines():
        data2.append(eval(line.strip()))

    assert len(data1) == len(data2)
    print('[num instances]:\t', len(data1))

    count_relation = dict()

    for i, sample in enumerate(data1):
        sample_ = data2[i]

        text = sample['text']
        text_ = sample['text']
        assert text == text_

        relations = sample['relations']
        relations_ = sample_['relations']
        assert len(relations) == len(relations_)

        for j, relation in enumerate(relations):
            relation_ = relations_[j]

            head, rel, tail = relation[0], relation[1], relation[2]
            head_, rel_, tail_ = relation_[0], relation_[1], relation_[2]
            if text[head_[0]: head_[1]] != head_[2] or text[tail_[0]: tail_[1]] != tail_[2]:
                print(text)
                print(relation_)
                exit(0)

            if (head[-1], rel, tail[-1]) != (head_[-1], rel_, tail_[-1]):
                print(text)
                print(relation_)
                exit(0)

            if rel not in count_relation:
                count_relation[rel] = 0
            count_relation[rel] += 1

    print(count_relation)


def remove_duplicate(in_fpath, out_fpath):
    """
        三元组去重
    """
    in_f = open(in_fpath, 'r', encoding='utf-8')
    out_f = open(out_fpath, 'w', encoding='utf-8')

    for line in in_f.readlines():
        d = eval(line.strip())
        
        sample = {"text": d["text"]}
        relations = []
        for relation in d['relations']:
            if relation not in relations:
                relations.append(relation)

        sample["relations"] = relations
        print(sample, file=out_f)


def overlap(span1, span2):
    return not (span1[1] <= span2[0] or span2[1] <= span1[0])

def add_field_entities(fpath1, fpath2):
    """
        对'exact.txt'中的每个样本，根据'relations'信息增加'entities'域，得到'ent_rel.txt'
    """
    f1 = open(fpath1, 'r', encoding='utf-8')
    f2 = open(fpath2, 'w', encoding='utf-8')

    data = []
    for line in f1.readlines():
        data.append(eval(line.strip()))

    for sample in data:
        text = sample['text']
        relations = sample['relations']

        entities = set()
        for relation in relations:
            head, rel, tail = relation[0], relation[1], relation[2]
            entities.add(tuple(head))
            entities.add(tuple(tail))

        entities = sorted(list(entities), key=lambda x: x[1]-x[0], reverse=True)
        entities_ = entities
        # if fname == '268.txt':
        #     print(entities_)

        # 扩增entities
        span_set = set([(ent[0], ent[1]) for ent in entities])
        for entity in entities:
            mention = entity[2]
            length = len(mention)
            
            for i in range(len(text)-length+1):
                if text[i: i+length] == mention:
                    ent = (i, i+length, mention)

                    # 避免标出重叠实体
                    flag_overlap = False
                    for span_ in span_set:
                        if overlap((i, i+length), span_):
                            flag_overlap = True
                    if flag_overlap:
                        continue

                    if ent not in entities_:
                        entities_.append(ent)
                        span_set.add((ent[0], ent[1]))

        # if fname == '268.txt':
        #     print(entities_)

        sample['entities'] = sorted(entities_, key=lambda x: (x[0], x[1]))
        print(sample, file=f2)


def add_entity_type(fpath1, fpath2):
    """
        对'ent_rel_full.txt'中的每个样本，根据'relations'信息增加'entities'域，得到'ent_rel_with_type.txt'
    """
    f1 = open(fpath1, 'r', encoding='utf-8')
    f2 = open(fpath2, 'w', encoding='utf-8')

    data = []
    for line in f1.readlines():
        data.append(eval(line.strip()))

    for sample in data:
        text = sample['text']
        relations = sample['relations']
        entities = sample['entities']

        set_patient, set_symptom, set_drug, set_treatment, set_usage, set_situation = set(), set(), set(), set(), set(), set()

        for relation in relations:
            head, rel_type, tail = relation[0], relation[1], relation[2]

            if rel_type == '临床表现':
                set_patient.add(head[-1])
                set_symptom.add(tail[-1])
            elif rel_type == '治疗药物':
                set_patient.add(head[-1])
                set_drug.add(tail[-1])
            elif rel_type == '治疗方案':
                set_patient.add(head[-1])
                set_treatment.add(tail[-1])
            elif rel_type == '用法用量':
                set_drug.add(head[-1])
                set_usage.add(tail[-1])
            elif rel_type == '基本情况':
                set_patient.add(head[-1])
                set_situation.add(tail[-1])
            elif rel_type == '禁用药物':
                set_patient.add(head[-1])
                set_drug.add(tail[-1])

        entities_ = []
        for ent in entities:
            ent_mention = ent[-1]
            if ent_mention in set_patient:
                ent_type = '病患'
            elif ent_mention in set_drug:
                ent_type = '药物'
            elif ent_mention in set_treatment:
                ent_type = '治疗'
            elif ent_mention in set_usage:
                ent_type = '用法'
            elif ent_mention in set_symptom:
                ent_type = '症状'
            elif ent_mention in set_situation:
                ent_type = '情况'
            else:
                print(ent[-1])
                ent_type = 'others'

            entities_.append(ent + (ent_type,))

        sample['entities'] = entities_
        print(sample, file=f2)


def add_dt(in_file1, in_file2, out_file):
    in_f1 = open(in_file1, 'r', encoding='utf-8')
    in_f2 = open(in_file2, 'r', encoding='utf-8')
    out_f = open(out_file, 'w', encoding='utf-8')

    in_data1 = json.load(in_f1)
    in_data2 = []
    for line in in_f2.readlines():
        in_data2.append(eval(line.strip()))

    for d1, d2 in zip(in_data1, in_data2):
        text1 = d1["text"]
        tree = d1['tree']

        text2 = d2["text"]

        if text1 != text2:
            print(text1)
            print(text2)
            exit(0)

        if 'relations' in d2:
            relations = d2['relations']
            relations_ = [[rel[0][-1], rel[1], rel[2][-1]] for rel in relations]
            for node in tree:
                for triple in node["triples"]:
                    if triple not in relations_:
                        print(tree)
                        print(relations)
                        exit(0)

        d2['tree'] = tree

        print(d2, file=out_f)

def concat(list_fpaths, out_fpath):
    out_f = open(out_fpath, 'w', encoding='utf-8')
    for fpath in list_fpaths:
        for line in open(fpath).readlines():
            print(line.strip(), file=out_f)


if __name__ == '__main__':
    """
    对Text2DT_dev.json订正：
        AVNRT患者@合并低血压者可应用升压药物（如去氧肾上腺素、甲氧明或间羟胺），通过反射性兴奋迷走神经终止心动过速。但老年患者、高血压、急性心肌梗死患者等禁用升压药物。
        肺炎链球菌肺炎患者@抗菌药物治疗:轻症病人，可用青霉素240万U/d分3次肌内注射，或用普鲁卡因青霉素每12小时肌内注射60万U。病情稍重者，宜用青霉素240万一480万U/d分次静脉滴注每6一8小时1次；重症及并发脑膜炎者，可增至1000万~3000万U/d分4次静脉滴注。
        膜性肾病患者@免疫抑制治疗首选糖皮质激素与烷化剂的联合使用；有糖皮质激素禁忌症时，可用环孢素与小剂量糖皮质激素联合治疗，其疗效相当。
        非ST段抬高急性冠状动脉综合征患者@无禁忌证者，在阿司匹林基础上联合应用一种P2Y12受体抑制剂。否则口服阿司匹林首剂负荷量150-300mg，并以75-100mg/d长期服用。

    对Text2DT_train.json订正：
        局灶节段性肾小球硬化症患者@首选治疗方案是大剂量糖皮质激素联合环磷酰胺。对于糖皮质激素依赖、反复复发者，环磷酰胺、环孢素、硫唑嘌呤、吗替麦考酚酯可能有利于延长维持缓解时间。
        非中性粒细胞减少成人患者念珠菌血症患者@对于血流动力学稳定的非危重感染，未使用过唑类药物者首选氟康唑，血流动力学稳定的中重度感染，近期使用过唑类药物者宜首选棘白菌素类。
        癫痫患者@对于肌阵挛发作患者，给予丙戊酸、左乙拉西坦或左乙拉西作为一线治疗药物；对于局灶性发作患者，给予卡马西平、拉莫三嗪、奥卡西平坦，左乙拉西坦或丙戊酸作为一线治疗药物。
        肉芽性多血管炎患者@对轻型或局限型早期病例可单用糖皮质激素治疗，若糖皮质激素疗效不佳应尽早使用CTX。对有肾受累或下呼吸道病变者，开始治疗即应联合应用糖皮质激素与CTX。
        甲真菌病患儿@如体重＜20kg，口服特比萘芬62.5mg/d，如体重20～40kg，口服特比萘芬125mg/d。
        哮喘病患儿@根据年龄分为大于等于6岁儿童哮喘的长期治疗方案：包括以β受体激动剂为代表的缓解药物和以ICS及白三烯调节剂为代表的抗炎药物；对于小于6岁儿童哮喘的长期治疗方案：推荐使用低剂量ICS作为初始控制治疗。
        主动脉瓣重度狭窄患者@对于有明显症状者，可以进行主动脉瓣置换手术。对于无明显症状的患者，表现为收缩压较基线降低或收缩压较基线不能增加20mmHg以上，或与年龄性别正常标准相比运动耐力明显降低，对此类患者考虑择期手术置换主动脉瓣是合理。
        局灶节段性肾小球硬化症患者@首选治疗方案是大剂量糖皮质激素联合环磷酰胺。对于糖皮质激素依赖、反复复发者，环磷酰胺、环孢素、硫唑嘌呤、吗替麦考酚酯可能有利于延长维持缓解时间。
        支气管肺炎患儿@高热患儿可用物理降温，口服对乙酰线基酚或布洛芬等。
        过敏性紫癜患儿@有荨麻疹或血管神经性水肿时，应用抗组胺药物和钙剂。腹痛时应用解痉剂,消化道出血时应禁食，可静脉滴注西咪替丁，每日20~40mg/kg,必要时输血。
        异位型输尿管口囊肿患者@如果同侧肾功能良好，可先选择经尿道输尿管口囊肿切开术或囊壁部分切除术，术后复查提示如果有膀胱输尿管反流，可行输尿管膀胱再吻合术。
        急性乙醇中毒患者@急性意识障碍者可考虑静脉注射葡萄糖，肌注维生素B1，维生素B6，以加速乙醇在体内氧化。对烦躁不安或过度兴奋者，可用地西洋，避免用吗啡、氣丙嗪、苯巴比妥类镇静药。
        嗜酸性食管炎患者@局固醇是嗜酸性食管炎的一线药物治疗。如果局部类固醇无效，或要快速改善症状的患者，强的松可用于治疗。
        输尿管口囊肿患者@对于囊肿体积小，无临床症状和无相关并发症的，无需特殊治疗，可定期复查。对于并发尿路梗阻或尿路感染的患者，可先行经尿道输尿管口囊肿切开术或囊壁部分切除术。
        支气管扩张症患者@无铜绿假单胞菌感染高危因素的病人应立即经验性使用对流感嗜血杆菌有活性的抗菌药物。对于存在铜绿假单胞菌感染高危因素的病人，可选择具有抗假单胞菌活性的内酰胺类抗生素。

    存疑：
        血管内念珠菌感染患者@对于敏感念珠菌，临床稳定和血中念珠菌已被清除的患者，如果氟康唑耐药，可以将口服伏立康唑200-300mg[3-4mg/kg]，否则推荐应用氟康唑400-800mg[6-12mg/kg]作为治疗药物。
    """

    preprocess("../json/Text2DT_dev.json", "dev_tmp.txt")
    preprocess("../json/Text2DT_train.json", "train_tmp.txt")
    preprocess("../json/Text2DT_test.json", "test.txt")

    """
    对'tmp.txt'中，在同一段文本多次出现的实体('MULTI')，人工标出其span位置，得到'exact.txt'
    """

    # 检查'tmp.txt'和'exact.txt'是否一致
    check('dev_tmp.txt', 'dev_exact.txt')
    check('train_tmp.txt', 'train_exact.txt')

    # 三元组去重
    remove_duplicate('dev_exact.txt', 'dev_rel.txt')
    remove_duplicate('train_exact.txt', 'train_rel.txt')

    # 对每个样本，根据'relations'信息增加'entities'域，得到'ent_rel.txt'
    add_field_entities('dev_rel.txt', 'dev_ent_rel_full.txt')
    add_field_entities('train_rel.txt', 'train_ent_rel_full.txt')

    # 对每个样本，根据关系类型为头尾实体增加实体类别伪标注
    add_entity_type('dev_ent_rel_full.txt', 'dev_ent_rel_with_type.txt')
    add_entity_type('train_ent_rel_full.txt', 'train_ent_rel_with_type.txt')

    """
    人工对不含关系的实体人工增加实体类别标注，得到dev_ent_rel_exact_type.txt和train_ent_rel_exact_type.txt
    """

    # 增加决策树标注信息
    os.makedirs("Text2DT", exist_ok=True)
    add_dt("../json/Text2DT_dev.json", "dev_ent_rel_exact_type.txt", "Text2DT/dev_dt.txt")
    add_dt("../json/Text2DT_train.json", "train_ent_rel_exact_type.txt", "Text2DT/train_dt.txt")
    concat(["Text2DT/train_dt.txt", "Text2DT/dev_dt.txt"], "Text2DT/train_dev_dt.txt")
    add_dt("../json/Text2DT_test.json", "test.txt", "Text2DT/test_dt.txt")
