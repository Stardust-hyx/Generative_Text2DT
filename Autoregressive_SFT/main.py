#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys
import json

import numpy as np
from datasets import load_dataset
import jieba 
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from text2dt_eval.parser import parsing, parsing_RE
from text2dt_eval.metric import text2dt_metric, RE_metric
import torch

import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)

import sys
sys.path.append("./")

from tokenization_chatglm import ChatGLMTokenizer
from configuration_chatglm import ChatGLMConfig
from modeling_chatglm import ChatGLMForConditionalGeneration
from trainer_seq2seq import Seq2SeqTrainer
from arguments import ModelArguments, DataTrainingArguments

from peft import PeftModel, LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict
from data_utils import MyTrainDataProcessor, MyDataCollator

logger = logging.getLogger(__name__)

def main():
    instruct = "请形式化地描述以下医疗文本所表达的诊疗决策过程，目标输出格式为“若...[若...]，则...，否则，若...[若...]，则...，否则...”。三元组关系类型包括“临床表现，用药，治疗方案，用法，基本情况，慎用”。\n输入文本:"

    # NER_instruct = "抽取出以下医疗文本中的“病人”，“临床表现”，“基本情况”，“药物”，“用法用量”，“治疗方法”类型实体。\n输入文本:"
    NER_instruct = "请找出以下医疗文本中与诊疗相关的实体，并以“[\"实体1\", \"实体2\", ...]”的格式按顺序输出。\n输入文本:"

    RE_instruct = "请找出以下医疗文本中出现的关系三元组，并按照“[(头实体1, 关系类型1, 尾实体1), (头实体2, 关系类型2, 尾实体2), ...]”的格式输出，目标关系类型包括“临床表现，用药，治疗方案，用法，基本情况，慎用”。\n输入文本:"

    TreeS_instruct = "请用“若”，“则”，“否则”，“或”，“且”，“和”大致表示以下医疗文本的诊疗逻辑框架。\n输入文本:"
    
    # COT_instruct = "请先找出以下医疗文本中出现的关系三元组，并按照“[(头实体1, 关系类型1, 尾实体1), (头实体2, 关系类型2, 尾实体2), ...]”的格式输出，目标关系类型包括“临床表现，用药，治疗方案，用法，基本情况，慎用”；再用“若”，“则”，“否则”，“或”，“且”，“和”大致表示该医疗文本的诊疗逻辑框架；最后再根据关系三元组列表和诊疗逻辑框架形式化地描述该医疗文本所表达的诊疗决策过程。\n输入文本:"
    COT_instruct = "请先用“若”，“则”，“否则”，“或”，“且”，“和”大致表示以下医疗文本的诊疗逻辑框架；再找出该医疗文本中出现的关系三元组，并按照“[(头实体1, 关系类型1, 尾实体1), (头实体2, 关系类型2, 尾实体2), ...]”的格式输出，目标关系类型包括“临床表现，用药，治疗方案，用法，基本情况，慎用”；最后再根据诊疗逻辑框架和关系三元组列表形式化地描述该医疗文本所表达的诊疗决策过程。\n输入文本:"

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    # datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load dataset
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
        extension = data_args.train_file.split(".")[-1]
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
        extension = data_args.validation_file.split(".")[-1]
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
        extension = data_args.test_file.split(".")[-1]

    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
        streaming=True
    )
    print("raw_datasets: ", raw_datasets)
    # print("raw_datasets: ", len(raw_datasets["train"]))

    # Load pretrained model and tokenizer
    config = ChatGLMConfig.from_pretrained(
        model_args.model_name_or_path,
        # trust_remote_code=True
    )
    config.pre_seq_len = model_args.pre_seq_len
    config.prefix_projection = model_args.prefix_projection
    print("config.pre_seq_len", config.pre_seq_len)
    print("config.prefix_projection", config.prefix_projection)

    tokenizer = ChatGLMTokenizer.from_pretrained(
        model_args.model_name_or_path,
        # trust_remote_code=True
    )
    print(tokenizer.tokenize(instruct))

    model = ChatGLMForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
    ).half().cuda()

    # for n, p in model.named_parameters():
    #     print(n, p.requires_grad)

    if model_args.peft_path is not None:
        logger.info("Peft from pre-trained model")
        model = PeftModel.from_pretrained(model, model_args.peft_path)
    else:
        logger.info("Init new peft model")
        target_modules = model_args.trainable.split(',')
        modules_to_save = model_args.modules_to_save.split(',') if model_args.modules_to_save!="null" else None
        lora_rank = model_args.lora_rank
        lora_dropout = model_args.lora_dropout
        lora_alpha = model_args.lora_alpha
        print(target_modules)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            inference_mode=False,
            r=lora_rank, lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            modules_to_save=modules_to_save
        )
        model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # for n, p in model.named_parameters():
    #     print(n, p.requires_grad, p.numel())

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
        column_names = list(next(iter(raw_datasets["train"])).keys())
        print("[Train Data] column_names:\n", column_names)
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
        column_names = list(next(iter(raw_datasets["validation"])).keys())
        print("[Eval Data] column_names:\n", column_names)
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
        column_names = list(next(iter(raw_datasets["test"])).keys())
        print("[Test Data] column_names:\n", column_names)
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target.
    prompt_column = data_args.prompt_column
    response_column = data_args.response_column
    history_column = data_args.history_column
    COT_response_column = 'COT_target'
    NER_response_column = 'NER_target'
    RE_response_column = 'RE_target'
    TreeS_response_column = 'TreeS_target'
    
    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length

    def preprocess_function_eval(examples):
        inputs, targets = [], []
        for i in range(len(examples[prompt_column])):
            if not examples[response_column][i]:
                targets.append("filled in !")
            else:
                targets.append(examples[response_column][i])

            if examples[prompt_column][i]:
                query = examples[prompt_column][i]
                query = instruct + query + '\n输出:'

                if history_column is None or len(examples[history_column][i]) == 0:
                    prompt = query
                else:
                    prompt = ""
                    history = examples[history_column][i]
                    for turn_idx, (old_query, response) in enumerate(history):
                        prompt += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, old_query, response)
                    prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
                inputs.append(prompt)

        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs,
                                 max_length=data_args.max_source_length,
                                 truncation=True,
                                 padding=True)
        labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True)

        if data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs
        
    def print_dataset_example(example):
        print("\ninput_ids:", example["input_ids"])
        print("inputs:\n" + tokenizer.decode(example["input_ids"]))
        print("label_ids:", example["labels"])
        print("labels:\n" + tokenizer.decode(example["labels"]))

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            preprocess_function_train = MyTrainDataProcessor(
                tokenizer,
                data_args.max_source_length + data_args.max_target_length,
                instruct, COT_instruct, NER_instruct, RE_instruct, TreeS_instruct,
                prompt_column, response_column,
            )
            train_dataset = train_dataset.map(
                preprocess_function_train,
                fn_kwargs={'aug_start_index': 400},
                batched=True,
                remove_columns=column_names,
            )
        train_dataset = train_dataset.shuffle(seed=training_args.seed, buffer_size=10000)
        # print_dataset_example(train_dataset[0])

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function_eval,
                batched=True,
                remove_columns=column_names,
            )
        # print_dataset_example(eval_dataset[0])

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function_eval,
                batched=True,
                remove_columns=column_names,
            )
        # print_dataset_example(predict_dataset[0])

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = MyDataCollator(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
        padding=True
    )

    # Metric
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # print('decoded_labels', decoded_labels)
        # print('decoded_preds', decoded_preds)

        score_dict = dict()
        gold_trees = list(map(lambda x: parsing(x, prefix=''), decoded_labels))
        pred_trees = list(map(lambda x: parsing(x, prefix=''), decoded_preds))
        text2dt_scores = text2dt_metric(gold_trees, pred_trees)
        score_dict.update(text2dt_scores)

        # score_dict = dict()
        # gold_data = list(map(lambda x: parsing_RE(x, prefix=''), decoded_labels))
        # pred_data = list(map(lambda x: parsing_RE(x, prefix=''), decoded_preds))
        # scores = RE_metric(gold_data, pred_data)
        # score_dict.update(scores)
        
        return score_dict

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    training_args.generation_num_beams = (
        data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    )
    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        save_prefixencoder=model_args.pre_seq_len is not None
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        # elif last_checkpoint is not None:
        #     checkpoint = last_checkpoint
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        # trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    max_seq_length = data_args.max_source_length + data_args.max_target_length + 1
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval", use_cache=True, max_length=max_seq_length, do_sample=False, top_p=0.7, temperature=0.95)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        # 读取原test file
        list_test_samples = []
        try:
            f = open(data_args.test_file, "r", encoding="utf-8")
            for line in f:
                line = json.loads(line)
                list_test_samples.append(line)
        except:
            f = open(data_args.test_file, "r", encoding="utf-8")
            list_test_samples = json.load(f)


        predict_results = trainer.predict(
            predict_dataset,
            metric_key_prefix="predict",
            use_cache=True,
            max_length=max_seq_length,
            do_sample=False,
            num_beams=1,
            top_p=0.7,
            temperature=0.95,
        )
        metrics = predict_results.metrics

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                labels = tokenizer.batch_decode(
                    predict_results.label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                labels = [label.strip() for label in labels]
                assert len(labels) == len(list_test_samples)

                output_gold_file = os.path.join(training_args.output_dir, "gold.json")
                output_pred_file = os.path.join(training_args.output_dir, "pred.json")
                gold_file = open(output_gold_file, "w", encoding="utf-8")
                pred_file = open(output_pred_file, "w", encoding="utf-8")
                list_gold = []
                list_pred = []
                for idx, (p, l) in enumerate(zip(predictions, labels)):
                    text = list_test_samples[idx]['input']
                    gold = {'text': text}
                    gold.update(parsing(l, prefix='', output_dict=True))
                    pred = {'text': text}
                    pred.update(parsing(p, prefix='', output_dict=True))
                    list_gold.append(gold)
                    list_pred.append(pred)
                json.dump(list_gold, gold_file, ensure_ascii=False, indent=2)
                json.dump(list_pred, pred_file, ensure_ascii=False, indent=2)

                # output_gold_file = os.path.join(training_args.output_dir, "gold.json")
                # output_pred_file = os.path.join(training_args.output_dir, "pred.json")
                # gold_file = open(output_gold_file, "w", encoding="utf-8")
                # pred_file = open(output_pred_file, "w", encoding="utf-8")
                # list_gold = []
                # list_pred = []
                # for idx, (l, p) in enumerate(zip(labels, predictions)):
                #     text = list_test_samples[idx]['input']
                #     gold = {'text': text}
                #     gold.update(parsing_RE(l, prefix=''))
                #     pred = {'text': text}
                #     pred.update(parsing_RE(p, prefix=''))
                #     list_gold.append(gold)
                #     list_pred.append(pred)
                # json.dump(list_gold, gold_file, ensure_ascii=False, indent=2)
                # json.dump(list_pred, pred_file, ensure_ascii=False, indent=2)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
