# coding=utf-8
import os, time
import json
import zhipuai
# your api key
zhipuai.api_key = ""

model = "chatglm_pro"

template_fn = 'template_JSON.txt'
out_fn = 'chatglm_json_pred.json'

log_fn = 'chatglm_json.log'
log_f = open(log_fn, 'w', encoding='utf-8')

template_f = open(template_fn, encoding='utf-8')
template = template_f.read()

in_fn = '../json/Text2DT_test.json'
list_text = []
samples = json.load(open(in_fn, encoding='utf-8'))
for sample in samples:
    text = sample['text']
    list_text.append(text)

samples = []
for i, text in enumerate(list_text):
    # if i >= 1:
    #     continue

    query = template.format(text)

    prompt = [
        {"role": "user", "content": query}
    ]

    response = zhipuai.model_api.invoke(
        model=model,
        prompt=prompt,
        temperature=0.01,
    )
    reply = response['data']['choices'][0]['content']
    assert isinstance(reply, str)
    assert reply[0] == '\"'
    reply = eval(reply).strip()

    print('-'*100, i, file=log_f)
    print('[Query]', file=log_f)
    print(query, file=log_f)
    print('[Response]', file=log_f)
    print(reply, file=log_f)
    print('-'*100, i, file=log_f, flush=True)

    try:
        tree = eval(reply)
    except:
        tree = []

    assert isinstance(tree, list)

    sample = {
        "text": text,
        "tree": tree
    }
    samples.append(sample)

out_f = open(out_fn, 'w', encoding='utf-8')
json.dump(samples, out_f, ensure_ascii=False, indent=2)
