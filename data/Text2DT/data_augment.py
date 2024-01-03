import random
import copy

def data_augmentation(ori_fn, tgt_fn, type_len_2_mentions=None, repeat=1):
    random.seed(2021)

    with open(ori_fn) as f:
        lines = f.readlines()
        lines = [eval(ele) for ele in lines]

    if type_len_2_mentions is None:
        type_len_2_mentions = dict()
        for line in lines:
            entities = line["entities"]
            for ent in entities:
                mention = ent[2]
                mention_len = len(mention)
                etype = ent[-1]

                if (etype, mention_len) not in type_len_2_mentions:
                    type_len_2_mentions[(etype, mention_len)] = []
                if mention not in type_len_2_mentions[(etype, mention_len)]:
                    type_len_2_mentions[(etype, mention_len)].append(mention)

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

if __name__ == '__main__':
    data_augmentation("train_dev_dt.txt", "aug_dt.txt")
    