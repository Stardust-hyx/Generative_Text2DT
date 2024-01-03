# coding=utf-8
import os, time
import json
import openai
from my_parser import parsing

#这里是你的获取到的api
openai.api_key = ''

model = "gpt-3.5-turbo"

template_fn = 'template_NL.txt'
out_fn = 'chatgpt_nl_pred.json'

log_fn = 'chatgpt_nl.log'
log_f = open(log_fn, 'w', encoding='utf-8')

template_f = open(template_fn, encoding='utf-8')
template = template_f.read()

in_fn = 'json/Text2DT_test.json'
list_text = []
samples = json.load(open(in_fn, encoding='utf-8'))
for sample in samples:
    text = sample['text']
    list_text.append(text)

samples = []
for i, text in enumerate(list_text):
    # if i >= 1:
    #     continue

    text_ = text[:-1].replace('；', '，').replace('。', '，') + text[-1]

    query = template.format(text_)

    messages = [
        {"role": "system", "content": "你是一个细心的自然语言处理助手。"},
        {"role": "user", "content": query}
    ]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.0,
    )
    reply = response["choices"][0]["message"]["content"].strip()

    print('-'*100, i, file=log_f)
    print('[Query]', file=log_f)
    print(query, file=log_f)
    print('[Response]', file=log_f)
    print(reply, file=log_f)
    print('-'*100, i, file=log_f, flush=True)

    try:
        tree = parsing(reply)
    except:
        print('Invalid Format!', file=log_f, flush=True)
        exit(0)

    sample = {
        "text": text,
        "tree": tree
    }
    samples.append(sample)

    time.sleep(18)

out_f = open(out_fn, 'w', encoding='utf-8')
json.dump(samples, out_f, ensure_ascii=False, indent=2)
