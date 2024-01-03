import random
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from random import randint
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase

@dataclass
class MyTrainDataProcessor:
    """
        Train data preprocessor that works in pair with IterableDataset.map() to dynamically prepare the training data for each epoch.

        Args:
            
    """

    tokenizer: PreTrainedTokenizerBase
    max_seq_length: int
    
    instruct: str
    COT_instruct: str
    NER_instruct: str
    RE_instruct: str
    TreeS_instruct: str

    prompt_column: Optional[str] = 'input'
    response_column: Optional[str] = 'target'
    history_column: Optional[str] = None
    COT_response_column: Optional[str] = 'COT_target'
    NER_response_column: Optional[str] = 'NER_target'
    rel_2_triples_column: Optional[str] = 'rel_2_triples'
    RE_response_column: Optional[str] = 'RE_target'
    TreeS_response_column: Optional[str] = 'TreeS_target'

    epoch: float = 0

    main_ths: float = 1.0
    COT_ths: float = 0
    NER_ths: float = 0
    REsub_ths: float = 0.6
    RE_ths: float = 0.3
    TreeS_ths: float = 1.0

    main_delta: float = 0.0
    COT_delta: float = 0
    NER_delta: float = 0
    REsub_delta: float = -0.1
    RE_delta: float = 0.15
    TreeS_delta: float = 0.0

    def __call__(self, examples, aug_start_index=1e10):
        model_inputs = {
            "input_ids": [],
            "labels": [],
        }

        print(len(examples[self.prompt_column]))
        for i in range(len(examples[self.prompt_column])):
            if (i < aug_start_index or examples[self.response_column][i].count('若') > 2):
                random_num = np.random.random()
            # sample for main task
            if examples[self.prompt_column][i] and examples[self.response_column][i]:
                # using all non-augmented data as well as hard augmented data
                if (i < aug_start_index or examples[self.response_column][i].count('若') > 2) and random_num < self.main_ths:
                    input_ids, labels = construct_train_sample(
                        self.tokenizer,
                        self.instruct,
                        examples[self.prompt_column][i],
                        examples[self.response_column][i],
                        self.max_seq_length,
                        None if self.history_column is None else examples[self.history_column][i],
                        suffix='\n输出:'
                    )
                    # oversampling to balance # main-task-samples amd # auxilary-task-samples
                    for _ in range(2):
                        model_inputs["input_ids"].append(input_ids)
                        model_inputs["labels"].append(labels)

            # # sample for COT task
            # if examples[self.prompt_column][i] and examples[self.COT_response_column][i]:
            #     if i < aug_start_index:
            #         input_ids, labels = construct_train_sample(
            #             self.tokenizer,
            #             self.COT_instruct,
            #             examples[self.prompt_column][i],
            #             examples[self.COT_response_column][i],
            #             self.max_seq_length,
            #             None if self.history_column is None else examples[self.history_column][i],
            #             suffix='\n'
            #         )
            #         # oversampling to balance # main-task-samples amd # auxilary-task-samples
            #         for _ in range(3):
            #             model_inputs["input_ids"].append(input_ids)
            #             model_inputs["labels"].append(labels)
            
            # # sample for auxiliary NER task
            # if examples[self.prompt_column][i] and examples[self.NER_response_column][i]:
            #     # random_num = np.random.random()
            #     if (i < aug_start_index) and random_num < self.NER_ths:
            #         input_ids, labels = construct_NER_train_sample(
            #             self.tokenizer,
            #             examples[self.prompt_column][i],
            #             examples[self.NER_response_column][i],
            #             self.max_seq_length,
            #         )
            #         model_inputs["input_ids"] += input_ids
            #         model_inputs["labels"] += labels

            # sample for auxiliary RE task
            if examples[self.prompt_column][i] and examples[self.RE_response_column][i]:
                # # random_num = np.random.random()
                if (i < aug_start_index or examples[self.response_column][i].count('若') > 2) and random_num < self.REsub_ths:
                    input_ids, labels = construct_RE_train_sample(
                        self.tokenizer,
                        examples[self.prompt_column][i],
                        examples[self.rel_2_triples_column][i],
                        self.max_seq_length,
                    )
                    model_inputs["input_ids"] += input_ids
                    model_inputs["labels"] += labels
            
                if (i < aug_start_index or examples[self.response_column][i].count('若') > 2) and random_num < self.RE_ths:
                    # print(i, 'RE_response_column', examples[self.RE_response_column][i])
                    input_ids, labels = construct_train_sample(
                        self.tokenizer,
                        self.RE_instruct,
                        examples[self.prompt_column][i],
                        examples[self.RE_response_column][i],
                        self.max_seq_length,
                        None if self.history_column is None else examples[self.history_column][i],
                        suffix='\n输出:'
                    )
                    # oversampling to balance # main-task-samples amd # auxilary-task-samples
                    for _ in range(1):
                        model_inputs["input_ids"].append(input_ids)
                        model_inputs["labels"].append(labels)


            # sample for auxiliary TreeS task
            if examples[self.prompt_column][i] and examples[self.TreeS_response_column][i]:
                # random_num = np.random.random()
                if (i < aug_start_index or examples[self.response_column][i].count('若') > 2) and random_num < self.TreeS_ths:
                    input_ids, labels = construct_train_sample(
                        self.tokenizer,
                        self.TreeS_instruct,
                        examples[self.prompt_column][i],
                        examples[self.TreeS_response_column][i],
                        self.max_seq_length,
                        None if self.history_column is None else examples[self.history_column][i],
                        suffix='\n输出:'
                    )
                    model_inputs["input_ids"].append(input_ids)
                    model_inputs["labels"].append(labels)

        self.epoch += 1

        self.main_ths += self.main_delta
        self.COT_ths += self.COT_delta
        self.NER_ths += self.NER_delta
        self.REsub_ths += self.REsub_delta
        self.RE_ths += self.RE_delta
        self.TreeS_ths += self.TreeS_delta

        return model_inputs


def construct_train_sample(tokenizer, instruct, query, answer, max_seq_length, history=None, suffix=''):
    query = instruct + query + suffix

    if history is None:
        prompt = query
    else:
        prompt = ""
        for turn_idx, (old_query, response) in enumerate(history):
            prompt += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, old_query, response)
        prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)

    a_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
    b_ids = tokenizer.encode(text=answer, add_special_tokens=False)

    if len(a_ids) + len(b_ids) > max_seq_length - 3:
        print('len(a_ids)', len(a_ids))
        print(prompt)
        print('len(b_ids)', len(b_ids))
        print(answer)
        print()
        offset = len(a_ids) + len(b_ids) - (max_seq_length - 3)
        b_ids = b_ids[: -offset]

    input_ids = tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)

    context_length = input_ids.index(tokenizer.bos_token_id)
    mask_position = context_length - 1
    labels = [-100] * context_length + input_ids[mask_position+1:]
    
    # pad_len = max_seq_length - len(input_ids)
    # input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
    # labels = labels + [tokenizer.pad_token_id] * pad_len
    # if data_args.ignore_pad_token_for_loss:
    #     labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]

    # print(tokenizer.decode(input_ids))
    # print(tokenizer.decode(labels))
    # print()

    return input_ids, labels


def construct_NER_train_sample(tokenizer, text, info_dict, max_seq_length, num_positive=1, num_negative=1):
    assert isinstance(info_dict, dict)
    presented_types = [etype for (etype, mentions) in info_dict.items() if len(mentions) > 0]
    presented_types_ = [etype for (etype, mentions) in info_dict.items() if len(mentions) > 1]
    unpresented_types = [etype for (etype, mentions) in info_dict.items() if len(mentions) == 0]

    chosen_types = []
    if len(presented_types) > 0:
        chosen_type = np.random.choice(presented_types)
        chosen_types.append(chosen_type)
        # if len(info_dict[chosen_type]) == 1 and len(presented_types_) > 0:
        #     chosen_type = np.random.choice(presented_types_)
        #     chosen_types.append(chosen_type)

    if len(unpresented_types) > 0:
        chosen_type = np.random.choice(unpresented_types)
        chosen_types.append(chosen_type)

    assert len(chosen_types) > 0

    input_ids_, labels_ = [], []
    for etype in chosen_types:
        query = text + f"抽取出上述文本中的“{etype}”类型实体。{etype}："
        mentions = info_dict[etype]
        if len(mentions) > 0:
            answer = ', '.join(['“' + mention + '”' for mention in mentions])
        else:
            answer = '无'

        # print(query + answer)
        a_ids = tokenizer.encode(text=query, add_special_tokens=False)
        b_ids = tokenizer.encode(text=answer, add_special_tokens=False)

        if len(a_ids) + len(b_ids) > max_seq_length - 3:
            print('len(a_ids)', len(a_ids))
            print(query)
            print('len(b_ids)', len(b_ids))
            print(answer)
            print()
            offset = len(a_ids) + len(b_ids) - (max_seq_length - 3)
            b_ids = b_ids[: -offset]

        input_ids = tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)
        input_ids_.append(input_ids)

        context_length = input_ids.index(tokenizer.bos_token_id)
        mask_position = context_length - 1
        labels = [-100] * context_length + input_ids[mask_position+1:]
        labels_.append(labels)

    return input_ids_, labels_

def linearize_triple(triple):
    return '('+ triple[0] + ', ' + triple[1] + ', ' + triple[2] + ')'

def construct_RE_train_sample(tokenizer, text, info_dict, max_seq_length, num_positive=1, num_negative=1):
    assert isinstance(info_dict, dict)
    presented_types = [rtype for (rtype, triples) in info_dict.items() if len(triples) > 0]
    presented_types_ = [rtype for (rtype, triples) in info_dict.items() if len(triples) > 1]
    unpresented_types = [rtype for (rtype, triples) in info_dict.items() if len(triples) == 0]

    chosen_types = []
    if len(presented_types) > 0:
        chosen_type = np.random.choice(presented_types)
        chosen_types.append(chosen_type)
        # if len(info_dict[chosen_type]) == 1 and len(presented_types_) > 0:
        #     chosen_type = np.random.choice(presented_types_)
        #     chosen_types.append(chosen_type)

    if len(unpresented_types) > 0:
        chosen_type = np.random.choice(unpresented_types)
        chosen_types.append(chosen_type)

    assert len(chosen_types) > 0

    input_ids_, labels_ = [], []
    for rtype in chosen_types:
        query = text + f"找出上述文本中出现的“{rtype}”类型关系三元组。输出："
        triples = info_dict[rtype]
        if len(triples) > 0:
            answer = ', '.join(['('+ x[0] + ', ' + x[1] + ', ' + x[2] + ')' for x in triples])
        else:
            answer = '无'

        # print(query + answer)
        a_ids = tokenizer.encode(text=query, add_special_tokens=False)
        b_ids = tokenizer.encode(text=answer, add_special_tokens=False)

        if len(a_ids) + len(b_ids) > max_seq_length - 3:
            print('len(a_ids)', len(a_ids))
            print(query)
            print('len(b_ids)', len(b_ids))
            print(answer)
            print()
            offset = len(a_ids) + len(b_ids) - (max_seq_length - 3)
            b_ids = b_ids[: -offset]

        input_ids = tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)
        input_ids_.append(input_ids)

        context_length = input_ids.index(tokenizer.bos_token_id)
        mask_position = context_length - 1
        labels = [-100] * context_length + input_ids[mask_position+1:]
        labels_.append(labels)

    return input_ids_, labels_


@dataclass
class MyDataCollator:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            min_label_length = min(len(l) for l in labels)
            if min_label_length != max_label_length:
                for feature in features:
                    remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                    feature["labels"] = feature["labels"] + remainder


        # pad input_ids
        max_length = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of is not None:
            max_length = (
                (max_length + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )

        min_length = min(len(feature["input_ids"]) for feature in features)
        if min_length != max_length:
            for feature in features:
                remainder = [self.tokenizer.pad_token_id] * (max_length - len(feature["input_ids"]))
                feature["input_ids"] = feature["input_ids"] + remainder

        # print(len(features))
        features_ = dict()
        features_['input_ids'] = torch.LongTensor([feature["input_ids"] for feature in features])
        if labels is not None:
            features_['labels'] = torch.LongTensor([feature["labels"] for feature in features])

        return features_
