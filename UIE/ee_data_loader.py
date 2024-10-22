import json
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from config import get_tokenizer
import random
import copy
from transformers import BertTokenizer
from utils.question_maker import (
    creat_demo,
    creat_argu_token,
    creat_argu_labels,
)
from utils.alphabet import Alphabet
from utils.gazetteer import Gazetteer
from utils.lexicon_functions import (
    generate_instance_with_gaz,
    batchify_augment_ids,
)


import os

NULLKEY = "-null-"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def find_all(sub_str, string):
    positions = []
    pos = string.find(sub_str)
    while pos != -1:
        positions.append(pos)
        pos = string.find(sub_str, pos + 1)
    return positions


def replace_text(role_tuple, old, new):
    text = role_tuple["text"]
    all_index = find_all(old, text)
    bias = len(new) - len(old)
    multi = 0
    for index in all_index:
        if index < role_tuple["trigger_start_index"]:
            multi += 1
    role_tuple["trigger_start_index"] += bias * multi
    for argu in role_tuple["arguments"]:
        multi = 0
        for index in all_index:
            if index < argu["argument_start_index"]:
                multi += 1
        argu["argument_start_index"] += bias * multi
        if argu["argument"] == old:
            argu["argument"] = new
    role_tuple["text"] = text.replace(old, new)


def random_replace(role_tuple, argument_label_dict, replace_set):
    random_argu = random.choice(role_tuple["arguments"])["argument"]
    label = argument_label_dict[role_tuple["event_type"] + "-" + role_tuple["role"]]
    new_argument = random.choice(replace_set[label])
    replace_text(role_tuple, random_argu, new_argument)
    return role_tuple


def merge_evt(evt_origin, evt_aug):
    ret = {"text": "", "id": evt_origin["id"], "event_list": []}

    random_number = random.choice([4, 5])
    evt_origin = copy.deepcopy(evt_origin)
    evt_aug = copy.deepcopy(evt_aug)
    # logging.info(random_number)
    if random_number == 4:
        ret["text"] = evt_origin["text"] + "。" + evt_aug["text"]
        bias = len(evt_origin["text"]) + 1  ## 因为加了一个。分隔
        for evt in evt_aug["event_list"]:
            evt["trigger_start_index"] += bias

    else:
        ret["text"] = evt_aug["text"] + "。" + evt_origin["text"]
        bias = len(evt_aug["text"]) + 1
        for evt in evt_origin["event_list"]:
            evt["trigger_start_index"] += bias

    ret["event_list"] += evt_origin["event_list"]
    ret["event_list"] += evt_aug["event_list"]
    return ret


def merge_argu(argu_origin, argu_aug):
    ret = copy.deepcopy(argu_origin)
    random_number = random.choice([4, 5])
    if random_number == 4:
        ret["text"] = argu_origin["text"] + "。" + argu_aug["text"]
    else:
        ret["text"] = argu_aug["text"] + "。" + argu_origin["text"]
        bias = len(argu_aug["text"]) + 1
        argu_origin["trigger_start_index"] += bias
        for argu in argu_origin["arguments"]:
            argu["argument_start_index"] += bias

    return ret


class ListDataset(Dataset):
    def __init__(self, file_path=None, data=None, args=None, **kwargs):
        self.args = args
        self.kwargs = kwargs
        if isinstance(file_path, (str, list)):
            self.data = self.load_data(file_path)
        elif isinstance(data, list):
            self.data = self.process_list_data(data)
        else:
            raise ValueError(
                "The input args shall be str format file_path / list format dataset"
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_data(file_path):
        return file_path

    def process_list_data(self, data):
        return data


# 加载实体识别数据集
class EeDataset(ListDataset):
    def __init__(self, file_path=None, data=None, args=None, test=False, **kwargs):
        super().__init__(file_path, data, args, **kwargs)
        self.test = test

    def __getitem__(self, index):
        data = self.data[index]
        mode = self.args.aug_mode
        if self.test:
            mode = None
        if "obj" == self.args.task:  ## 如果采用辅助学习增强策略
            data = self.convert_argu_data(data, mode)
        elif "tri" == self.args.task:  ## 如果采用拼接增强策略
            data = self.convert_tri_data(data, mode)
        elif "ner" == self.args.task:  ## 如果采用拼接增强策略
            data = self.convert_evt_data(data, mode)
        # if self.test:
        #     logging.debug(data)
        return data

    def convert_argu_data(self, role_tuple, mode=None):
        max_len = self.args.max_seq_len

        demo_tuples = self.args.demo_tuples
        sim_id = random.choice(role_tuple["sim_ids"])
        demo = None

        if mode == "demo":
            demo = creat_demo(demo_tuples[sim_id])
            # logging.debug(demo)
            if (
                len(demo) + len(role_tuple["question"] + role_tuple["text"]) + 1
                > max_len - 2
            ):  ### +1是因为中间有个[SEP]
                demo = None

        if mode == "merge":
            # random_number = random.choice([1, 2, 3])
            # if random_number == 1:
            #     role_tuple = random_replace(
            #         role_tuple, self.args.argument_label_dict, self.args.replace_set
            #     )

            random_number = random.choice([1, 2, 3, 4])
            if random_number == 3:  ## 随机选取事件拼接
                sim_ids = random.sample(role_tuple["sim_ids"], 2)
                role_augs = [self.args.demo_tuples[sim_id] for sim_id in sim_ids]
                for role_aug in role_augs:
                    role_tuple = merge_argu(role_tuple, role_aug)
            elif random_number == 4:  ## 选择相似事件
                sim_id = random.choice(role_tuple["sim_ids"])
                role_aug = self.args.demo_tuples[sim_id]
                role_tuple = merge_argu(role_tuple, role_aug)
            else:  ## 什么都不做
                role_tuple = role_tuple

        # logging.info('='*20)
        # logging.info(demo_tuples[sim_id])
        # logging.info(role_tuple)
        # logging.info('='*20)

        argu_token, obj_token_type_ids = creat_argu_token(role_tuple, demo, max_len)
        argu_start_labels, argu_end_labels, argu_tuples = creat_argu_labels(
            argu_token, demo, role_tuple, max_len
        )

        if self.args.use_lexicon:
            augment_Ids = generate_instance_with_gaz(
                argu_token,
                self.args.pos_alphabet,
                self.args.word_alphabet,
                self.args.biword_alphabet,
                self.args.gaz_alphabet,
                self.args.gaz_alphabet_count,
                self.args.gaz,
                max_len,
            )
        else:
            augment_Ids = []

        argu_data = {
            "obj_tokens": argu_token,
            "obj_start_labels": argu_start_labels,
            "obj_end_labels": argu_end_labels,
            "obj_token_type_ids": obj_token_type_ids,
            "augment_Ids": augment_Ids,
            "sim_score": 0,
            "argu_tuples": argu_tuples,
            "sim_id": sim_id,
        }
        # logging.info(role_tuple)
        # logging.info(argu_data)
        return argu_data

    def convert_tri_data(self, evt_tuple, mode=None):
        ### 用于识别文本中同类事件的触发词
        max_len = self.args.max_seq_len
        text_tokens = [i for i in evt_tuple["text"]]
        pre_tokens = [i for i in evt_tuple["question"]] + ["[SEP]"]
        trigger_bias = len(["[CLS]"] + pre_tokens)

        concat_token = pre_tokens + text_tokens
        # 防止太长
        if len(concat_token) > max_len - 2:
            concat_token = concat_token[: max_len - 2]

        concat_token = ["[CLS]"] + concat_token + ["[SEP]"]
        token_type_ids = [0] * len(concat_token)  # [CLS] +　pre_tokens

        ### 添加首尾标签
        tri_start_labels = [0] * len(concat_token)
        tri_end_labels = [0] * len(concat_token)

        for evt in evt_tuple["events"]:
            trigger_start_index = evt["trigger_start_index"] + trigger_bias
            trigger_end_index = (
                trigger_start_index + len(evt["trigger"]) - 1
            )  ## 注意长度要-1，否则出界

            ## 长文本特例
            if trigger_end_index < max_len:
                tri_start_labels[trigger_start_index] = 1
                tri_end_labels[trigger_end_index] = 1

        if self.args.use_lexicon:
            augment_Ids = generate_instance_with_gaz(
                concat_token,
                self.args.pos_alphabet,
                self.args.word_alphabet,
                self.args.biword_alphabet,
                self.args.gaz_alphabet,
                self.args.gaz_alphabet_count,
                self.args.gaz,
                max_len,
            )
        else:
            augment_Ids = []

        tri_data = {
            "tri_tokens": concat_token,
            "tri_start_labels": tri_start_labels,
            "tri_end_labels": tri_end_labels,
            "tri_token_type_ids": token_type_ids,
            "augment_Ids": augment_Ids,
        }
        # logging.info(tri_data)
        return tri_data

    def convert_evt_data(self, evt, mode=None):
        max_len = self.args.max_seq_len
        entity_label = self.args.entity_label
        sim_id = random.choice(evt["sim_ids"])
        ent_label2id = {label: i for i, label in enumerate(entity_label)}

        if mode == "merge":
            random_number = random.choice([1, 2, 3, 4])
            # logging.debug(random_number)
            if random_number == 3:  ## 随机选取事件拼接
                sim_ids = random.sample(evt["sim_ids"], 2)
                evt_augs = [self.args.demo_tuples[sim_id] for sim_id in sim_ids]
                for evt_aug in evt_augs:
                    evt = merge_evt(evt, evt_aug)
            elif random_number == 4:  ## 选择相似事件
                evt_aug = self.args.demo_tuples[sim_id]
                evt = merge_evt(evt_origin=evt, evt_aug=evt_aug)

        text = evt["text"]
        event_list = evt["event_list"]

        event_start_labels = np.zeros((len(ent_label2id), max_len))
        event_end_labels = np.zeros((len(ent_label2id), max_len))

        # 词典增强的向量
        bert_token = [i for i in text]
        bert_token = ["[CLS]"] + bert_token + ["[SEP]"]

        if len(bert_token) > max_len - 2:
            bert_token = bert_token[: max_len - 2]

        if self.args.use_lexicon:
            augment_Ids = generate_instance_with_gaz(
                bert_token,
                self.args.pos_alphabet,
                self.args.word_alphabet,
                self.args.biword_alphabet,
                self.args.gaz_alphabet,
                self.args.gaz_alphabet_count,
                self.args.gaz,
                max_len,
            )
        else:
            augment_Ids = []

        # 真实标签
        for event in event_list:
            event_type = event["event_type"]
            trigger = event["trigger"]
            trigger_start_index = event["trigger_start_index"]

            event_tokens = [i for i in text]
            if len(event_tokens) > max_len - 2:
                event_tokens = event_tokens[: max_len - 2]
            event_tokens = ["[CLS]"] + event_tokens + ["[SEP]"]

            if trigger_start_index + len(trigger) >= max_len - 1:
                continue

            event_start_labels[ent_label2id[event_type]][trigger_start_index + 1] = 1
            event_end_labels[ent_label2id[event_type]][
                trigger_start_index + len(trigger)  ## 这里是+1和-1抵消了
            ] = 1

        event_data = {
            "ner_tokens": event_tokens,
            "ner_start_labels": event_start_labels,
            "ner_end_labels": event_end_labels,
            "augment_Ids": augment_Ids,
        }

        return event_data

    def load_data(self, filename):
        logging.info("...构造文本embedding")
        sim_scorer = self.args.sim_scorer
        embs, data_list = sim_scorer.create_embs_and_tuples(filename, self.args.task,self.args.add_trigger)
        logging.info("文本embedding构造完毕")
        # 此处用训练集作为demo库
        ignore_first = True if filename == self.args.demo_path else False
        most_sim = sim_scorer.sim_match(
            embs, self.args.demo_embs, ignore_first
        )  # {corpus_id,score}
        logging.info("相似度匹配完成")
        for idx, data in enumerate(data_list):
            data["sim_ids"] = most_sim[idx]

        logging.info("数据集构建完毕")
        return data_list


def convert_list_to_tensor(alist, dtype=torch.long):
    return torch.tensor(
        np.array(alist) if isinstance(alist, list) else alist, dtype=dtype
    )


class EeCollate:
    def __init__(self, max_len, task, args):
        self.maxlen = max_len
        self.tokenizer = get_tokenizer(args)
        self.task = task
        self.args = args

    def collate_fn(self, batch):
        batch_ner_token_ids = []
        batch_raw_tokens = []  # 用于bad-case分析时获取结果
        batch_ner_attention_mask = []
        batch_ner_token_type_ids = []
        batch_ner_start_labels = []
        batch_ner_end_labels = []
        batch_obj_token_ids = []
        batch_obj_attention_mask = []
        batch_obj_token_type_ids = []
        batch_obj_start_labels = []
        batch_obj_end_labels = []
        batch_obj_argu_tuples = []
        batch_augment_Ids = []
        batch_sim_score = []

        batch_tri_token_ids = []
        batch_tri_attention_mask = []
        batch_tri_token_type_ids = []
        batch_tri_start_labels = []
        batch_tri_end_labels = []
        batch_tri_argu_tuples = []
        for i, data in enumerate(batch):
            augment_Ids = data["augment_Ids"]
            batch_augment_Ids.append(augment_Ids)

            if "ner" == self.task:
                ner_token_type_ids = [0] * self.maxlen
                ner_tokens = data["ner_tokens"]
                raw_tokens = ner_tokens
                ner_tokens = self.tokenizer.convert_tokens_to_ids(ner_tokens)
                ner_start_labels = data["ner_start_labels"]
                ner_end_labels = data["ner_end_labels"]

                # 对齐操作
                if len(ner_tokens) < self.maxlen:
                    # 第0个token的第0个gaz列表长度
                    ner_attention_mask = [1] * len(ner_tokens) + [0] * (
                        self.maxlen - len(ner_tokens)
                    )
                    ner_tokens = ner_tokens + [0] * (self.maxlen - len(ner_tokens))
                else:
                    ner_attention_mask = [1] * self.maxlen

                batch_ner_token_ids.append(ner_tokens)
                batch_ner_attention_mask.append(ner_attention_mask)
                batch_ner_token_type_ids.append(ner_token_type_ids)
                batch_ner_start_labels.append(ner_start_labels)
                batch_ner_end_labels.append(ner_end_labels)
                batch_raw_tokens.append(raw_tokens)

            elif "obj" == self.task:
                obj_tokens = data["obj_tokens"]
                raw_tokens = obj_tokens
                obj_tokens = self.tokenizer.convert_tokens_to_ids(obj_tokens)
                obj_start_labels = data["obj_start_labels"]
                obj_end_labels = data["obj_end_labels"]
                obj_token_type_ids = data["obj_token_type_ids"]
                obj_argu_tuples = data["argu_tuples"]
                sim_score = data["sim_score"]

                if len(obj_tokens) < self.maxlen:
                    obj_start_labels = obj_start_labels + [0] * (
                        self.maxlen - len(obj_tokens)
                    )
                    obj_end_labels = obj_end_labels + [0] * (
                        self.maxlen - len(obj_tokens)
                    )
                    obj_attention_mask = [1] * len(obj_tokens) + [0] * (
                        self.maxlen - len(obj_tokens)
                    )
                    obj_token_type_ids = obj_token_type_ids + [0] * (
                        self.maxlen - len(obj_tokens)
                    )
                    obj_tokens = obj_tokens + [0] * (self.maxlen - len(obj_tokens))
                else:
                    obj_attention_mask = [1] * self.maxlen

                batch_obj_token_ids.append(obj_tokens)
                batch_obj_attention_mask.append(obj_attention_mask)
                batch_obj_token_type_ids.append(obj_token_type_ids)
                batch_obj_start_labels.append(obj_start_labels)
                batch_obj_end_labels.append(obj_end_labels)
                batch_raw_tokens.append(raw_tokens)
                batch_sim_score.append(sim_score)
                batch_obj_argu_tuples.append(obj_argu_tuples)
            elif "tri" == self.task:
                tri_tokens = data["tri_tokens"]
                raw_tokens = tri_tokens
                tri_tokens = self.tokenizer.convert_tokens_to_ids(tri_tokens)
                tri_start_labels = data["tri_start_labels"]
                tri_end_labels = data["tri_end_labels"]
                tri_token_type_ids = data["tri_token_type_ids"]
                sim_score = 0

                # 对齐操作
                if len(tri_tokens) < self.maxlen:
                    tri_start_labels = tri_start_labels + [0] * (
                        self.maxlen - len(tri_tokens)
                    )
                    tri_end_labels = tri_end_labels + [0] * (
                        self.maxlen - len(tri_tokens)
                    )
                    tri_attention_mask = [1] * len(tri_tokens) + [0] * (
                        self.maxlen - len(tri_tokens)
                    )
                    tri_token_type_ids = tri_token_type_ids + [0] * (
                        self.maxlen - len(tri_tokens)
                    )
                    tri_tokens = tri_tokens + [0] * (self.maxlen - len(tri_tokens))
                else:
                    tri_attention_mask = [1] * self.maxlen

                batch_tri_token_ids.append(tri_tokens)
                batch_tri_attention_mask.append(tri_attention_mask)
                batch_tri_token_type_ids.append(tri_token_type_ids)
                batch_tri_start_labels.append(tri_start_labels)
                batch_tri_end_labels.append(tri_end_labels)
                batch_raw_tokens.append(raw_tokens)
                batch_sim_score.append(sim_score)

        res = {}

        if "ner" == self.task:
            batch_ner_token_ids = convert_list_to_tensor(batch_ner_token_ids)
            batch_ner_attention_mask = convert_list_to_tensor(batch_ner_attention_mask)
            batch_ner_token_type_ids = convert_list_to_tensor(batch_ner_token_type_ids)
            batch_ner_start_labels = convert_list_to_tensor(
                batch_ner_start_labels, dtype=torch.float
            )
            batch_ner_end_labels = convert_list_to_tensor(
                batch_ner_end_labels, dtype=torch.float
            )

            ner_res = {
                "ner_input_ids": batch_ner_token_ids,
                "ner_attention_mask": batch_ner_attention_mask,
                "ner_token_type_ids": batch_ner_token_type_ids,
                "ner_start_labels": batch_ner_start_labels,
                "ner_end_labels": batch_ner_end_labels,
            }

            res = ner_res

        elif "obj" == self.task:
            batch_obj_token_ids = convert_list_to_tensor(batch_obj_token_ids)
            batch_obj_attention_mask = convert_list_to_tensor(batch_obj_attention_mask)

            batch_obj_token_type_ids = convert_list_to_tensor(batch_obj_token_type_ids)
            batch_obj_start_labels = convert_list_to_tensor(
                batch_obj_start_labels, dtype=torch.float
            )
            batch_obj_end_labels = convert_list_to_tensor(
                batch_obj_end_labels, dtype=torch.float
            )
            batch_sim_score = convert_list_to_tensor(batch_sim_score, dtype=torch.float)

            sbj_obj_res = {
                "re_obj_input_ids": batch_obj_token_ids,
                "re_obj_attention_mask": batch_obj_attention_mask,
                "re_obj_token_type_ids": batch_obj_token_type_ids,
                "re_obj_start_labels": batch_obj_start_labels,
                "re_obj_end_labels": batch_obj_end_labels,
                "batch_sim_score": batch_sim_score,
                "argu_tuples": batch_obj_argu_tuples,
            }
            res = sbj_obj_res

        elif "tri" == self.task:
            batch_tri_token_ids = convert_list_to_tensor(batch_tri_token_ids)
            batch_tri_attention_mask = convert_list_to_tensor(batch_tri_attention_mask)

            batch_tri_token_type_ids = convert_list_to_tensor(batch_tri_token_type_ids)
            batch_tri_start_labels = convert_list_to_tensor(
                batch_tri_start_labels, dtype=torch.float
            )
            batch_tri_end_labels = convert_list_to_tensor(
                batch_tri_end_labels, dtype=torch.float
            )
            batch_sim_score = convert_list_to_tensor(batch_sim_score, dtype=torch.float)

            tri_res = {
                "re_obj_input_ids": batch_tri_token_ids,
                "re_obj_attention_mask": batch_tri_attention_mask,
                "re_obj_token_type_ids": batch_tri_token_type_ids,
                "re_obj_start_labels": batch_tri_start_labels,
                "re_obj_end_labels": batch_tri_end_labels,
                "batch_sim_score": batch_sim_score,
                "argu_tuples": [[]],  ## code smell来了
            }
            res = tri_res
        # 用于错误输出
        res["raw_tokens"] = batch_raw_tokens
        # 用于词典增强
        if self.args.use_lexicon:
            batch_augment_Ids = batchify_augment_ids(batch_augment_Ids, self.maxlen)
        res["batch_augment_Ids"] = batch_augment_Ids

        return res


if __name__ == "__main__":
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained("model_hub/chinese-bert-wwm-ext")
    # 测试实体识别
    # ============================
    max_seq_len = 256
    with open("data/ee/duee/labels.txt", "r", encoding="utf-8") as fp:
        entity_label = fp.read().strip().split("\n")

    print(entity_label)
    task = "ner"
    gaz_file = "/home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/data/embs/ctb.50d.vec"
    train_dataset = EeDataset(
        file_path="data/ee/duee/duee_dev.json",
        tokenizer=tokenizer,
        max_len=max_seq_len,
        entity_label=entity_label,
        task=task,
        gaz_file=gaz_file,
    )

    print(train_dataset[:5])
    # for k, v in train_dataset[0].items():
    #     print(k, v)

    collate = EeCollate(max_len=max_seq_len, tokenizer=tokenizer, task=task)
    batch_size = 16
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate.collate_fn,
    )
    # for eval_step, batch_data in enumerate(train_dataloader):
    #   print(batch_data['raw_tokens'])
    #   exit(0)
