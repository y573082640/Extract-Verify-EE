import json
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from LAC import LAC
import jieba
import jieba.posseg as pseg
from sentence_transformers import SentenceTransformer, util
from transformers import BertTokenizer
import time
from utils.question_maker import Sim_scorer, creat_demo, creat_argu_token, creat_argu_labels
from utils.alphabet import Alphabet
from utils.gazetteer import Gazetteer
from utils.lexicon_functions import build_alphabet, build_gaz_alphabet, build_gaz_file, generate_instance_with_gaz, batchify_augment_ids

NULLKEY = "-null-"


class ListDataset(Dataset):
    def __init__(self,
                 file_path=None,
                 data=None,
                 tokenizer=None,
                 max_len=None,
                 entity_label=None,
                 tasks=None,
                 args=None,
                 **kwargs):
        self.args = args
        self.kwargs = kwargs
        if isinstance(file_path, (str, list)):
            self.data = self.load_data(
                file_path, tokenizer, max_len, entity_label, tasks)
        elif isinstance(data, list):
            self.data = data
        else:
            raise ValueError(
                'The input args shall be str format file_path / list format dataset')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_data(file_path, tokenizer, max_len, entity_label, tasks):
        return file_path


# 加载实体识别数据集
class EeDataset(ListDataset):

    def load_data(self, filename, tokenizer, max_len, entity_label=None, tasks=None, gaz_file=None):
        ner_data = []
        obj_data = []
        ent_label2id = {label: i for i, label in enumerate(entity_label)}
        if "ner" in tasks:
            with open(filename, encoding='utf-8') as f:
                f = f.read().strip().split("\n")
                for d in f:
                    d = json.loads(d)
                    text = d["text"]

                    event_list = d["event_list"]
                    if len(text) == 0:
                        continue

                    event_start_labels = np.zeros((len(ent_label2id), max_len))
                    event_end_labels = np.zeros((len(ent_label2id), max_len))

                    # 词典增强的向量
                    bert_token = [i for i in text]
                    bert_token = ['[CLS]'] + bert_token + ['[SEP]']

                    if len(bert_token) > max_len - 2:
                        bert_token = bert_token[:max_len - 2]

                    if self.args.use_lexicon:
                        augment_Ids = generate_instance_with_gaz(
                            bert_token, self.args.pos_alphabet, self.args.word_alphabet, self.args.biword_alphabet,
                            self.args.gaz_alphabet, self.args.gaz_alphabet_count, self.args.gaz, max_len)
                    else:
                        augment_Ids = []

                    # 真实标签
                    for event in event_list:
                        event_type = event["event_type"]
                        trigger = event["trigger"]
                        trigger_start_index = event["trigger_start_index"]

                        event_tokens = [i for i in text]
                        if len(event_tokens) > max_len - 2:
                            event_tokens = event_tokens[:max_len - 2]
                        event_tokens = ['[CLS]'] + event_tokens + ['[SEP]']

                        if trigger_start_index+len(trigger) >= max_len - 1:
                            continue

                        event_start_labels[ent_label2id[event_type]
                                           ][trigger_start_index+1] = 1  # TODO: 如果丢入分类器前要去掉CLS，则不用+1
                        event_end_labels[ent_label2id[event_type]
                                         ][trigger_start_index+len(trigger)] = 1

                    event_data = {
                        "ner_tokens": event_tokens,
                        "ner_start_labels": event_start_labels,
                        "ner_end_labels": event_end_labels,
                        "augment_Ids": augment_Ids
                    }

                    ner_data.append(event_data)

        elif "obj" in tasks:
            logging.info('...构造文本embedding')
            sim_scorer = self.args.sim_scorer
            embs, tuples = sim_scorer.create_embs_and_tuples(filename)
            logging.info('文本embedding构造完毕')
            # 此处用训练集作为demo库
            demo_embs = self.args.demo_embs
            demo_tuples = self.args.demo_tuples
            most_sim = sim_scorer.sim_match(
                embs, demo_embs)  # {corpus_id,score}
            logging.info('相似度匹配完成')
            for idx, text_tuple in enumerate(tuples):
                sim_id = most_sim[idx]['corpus_id']
                demo = creat_demo(demo_tuples[sim_id])
                argu_token = creat_argu_token(text_tuple, demo, max_len)
                argu_start_labels, argu_end_labels = creat_argu_labels(
                    argu_token, demo, text_tuple, max_len)

                if len(argu_start_labels) == 0:
                    continue

                if self.args.use_lexicon:
                    augment_Ids = generate_instance_with_gaz(
                        argu_token,
                        self.args.pos_alphabet,
                        self.args.word_alphabet,
                        self.args.biword_alphabet,
                        self.args.gaz_alphabet,
                        self.args.gaz_alphabet_count,
                        self.args.gaz,
                        max_len)
                else:
                    augment_Ids = []

                argu_data = {
                    "obj_tokens": argu_token,
                    "obj_start_labels": argu_start_labels,
                    "obj_end_labels": argu_end_labels,
                    "augment_Ids": augment_Ids
                }

                obj_data.append(argu_data)

        logging.info("数据集构建完毕")
        return ner_data if "ner" in tasks else obj_data


def convert_list_to_tensor(alist, dtype=torch.long):
    return torch.tensor(np.array(alist) if isinstance(alist, list) else alist, dtype=dtype)


class EeCollate:
    def __init__(self,
                 max_len,
                 tokenizer,
                 tasks,
                 args):
        self.maxlen = max_len
        self.tokenizer = tokenizer
        self.tasks = tasks
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
        batch_augment_Ids = []
        for i, data in enumerate(batch):

            augment_Ids = data["augment_Ids"]
            batch_augment_Ids.append(augment_Ids)

            if "ner" in self.tasks:
                ner_token_type_ids = [0] * self.maxlen
                ner_tokens = data["ner_tokens"]
                raw_tokens = ner_tokens
                ner_tokens = self.tokenizer.convert_tokens_to_ids(ner_tokens)
                ner_start_labels = data["ner_start_labels"]
                ner_end_labels = data["ner_end_labels"]

                # 对齐操作
                if len(ner_tokens) < self.maxlen:
                    # 第0个token的第0个gaz列表长度
                    ner_attention_mask = [
                        1] * len(ner_tokens) + [0] * (self.maxlen - len(ner_tokens))
                    ner_tokens = ner_tokens + [0] * \
                        (self.maxlen - len(ner_tokens))
                else:
                    ner_attention_mask = [1] * self.maxlen

                batch_ner_token_ids.append(ner_tokens)
                batch_ner_attention_mask.append(ner_attention_mask)
                batch_ner_token_type_ids.append(ner_token_type_ids)
                batch_ner_start_labels.append(ner_start_labels)
                batch_ner_end_labels.append(ner_end_labels)
                batch_raw_tokens.append(raw_tokens)

            elif "obj" in self.tasks:
                obj_tokens = data["obj_tokens"]
                raw_tokens = obj_tokens
                obj_tokens = self.tokenizer.convert_tokens_to_ids(obj_tokens)
                obj_start_labels = data["obj_start_labels"]
                obj_end_labels = data["obj_end_labels"]
                obj_token_type_ids = [0] * self.maxlen

                if len(obj_tokens) < self.maxlen:
                    obj_start_labels = obj_start_labels + \
                        [0] * (self.maxlen - len(obj_tokens))
                    obj_end_labels = obj_end_labels + \
                        [0] * (self.maxlen - len(obj_tokens))
                    obj_attention_mask = [
                        1] * len(obj_tokens) + [0] * (self.maxlen - len(obj_tokens))
                    obj_tokens = obj_tokens + [0] * \
                        (self.maxlen - len(obj_tokens))
                else:
                    obj_attention_mask = [1] * self.maxlen

                batch_obj_token_ids.append(obj_tokens)
                batch_obj_attention_mask.append(obj_attention_mask)
                batch_obj_token_type_ids.append(obj_token_type_ids)
                batch_obj_start_labels.append(obj_start_labels)
                batch_obj_end_labels.append(obj_end_labels)
                batch_raw_tokens.append(raw_tokens)

        res = {}

        if "ner" in self.tasks:
            batch_ner_token_ids = convert_list_to_tensor(batch_ner_token_ids)
            batch_ner_attention_mask = convert_list_to_tensor(
                batch_ner_attention_mask)
            batch_ner_token_type_ids = convert_list_to_tensor(
                batch_ner_token_type_ids)
            batch_ner_start_labels = convert_list_to_tensor(
                batch_ner_start_labels, dtype=torch.float)
            batch_ner_end_labels = convert_list_to_tensor(
                batch_ner_end_labels, dtype=torch.float)

            ner_res = {
                "ner_input_ids": batch_ner_token_ids,
                "ner_attention_mask": batch_ner_attention_mask,
                "ner_token_type_ids": batch_ner_token_type_ids,
                "ner_start_labels": batch_ner_start_labels,
                "ner_end_labels": batch_ner_end_labels,
            }

            res = ner_res

        elif "obj" in self.tasks:
            batch_obj_token_ids = convert_list_to_tensor(batch_obj_token_ids)
            batch_obj_attention_mask = convert_list_to_tensor(
                batch_obj_attention_mask)
            batch_obj_token_type_ids = convert_list_to_tensor(
                batch_obj_token_type_ids)
            batch_obj_start_labels = convert_list_to_tensor(
                batch_obj_start_labels, dtype=torch.float)
            batch_obj_end_labels = convert_list_to_tensor(
                batch_obj_end_labels, dtype=torch.float)

            sbj_obj_res = {
                "re_obj_input_ids": batch_obj_token_ids,
                "re_obj_attention_mask": batch_obj_attention_mask,
                "re_obj_token_type_ids": batch_obj_token_type_ids,
                "re_obj_start_labels": batch_obj_start_labels,
                "re_obj_end_labels": batch_obj_end_labels,
            }

            res = sbj_obj_res

        # 用于错误输出
        res['raw_tokens'] = batch_raw_tokens
        # 用于词典增强
        if self.args.use_lexicon:
            batch_augment_Ids = batchify_augment_ids(
                batch_augment_Ids, self.maxlen)
        res['batch_augment_Ids'] = batch_augment_Ids

        return res


if __name__ == "__main__":
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained('model_hub/chinese-bert-wwm-ext')
    tokenizer.add_special_tokens(
        {'additional_special_tokens': ['[DEMO]', '[ARG]', '[TGR]']})
    # 测试实体识别
    # ============================
    max_seq_len = 256
    with open("data/ee/duee/labels.txt", "r", encoding="utf-8") as fp:
        entity_label = fp.read().strip().split("\n")

    print(entity_label)
    tasks = ["ner"]
    gaz_file = "/home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/data/embs/ctb.50d.vec"
    train_dataset = EeDataset(file_path='data/ee/duee/duee_dev.json',
                              tokenizer=tokenizer,
                              max_len=max_seq_len,
                              entity_label=entity_label,
                              tasks=tasks,
                              gaz_file=gaz_file)

    print(train_dataset[:5])
    # for k, v in train_dataset[0].items():
    #     print(k, v)

    collate = EeCollate(max_len=max_seq_len, tokenizer=tokenizer, tasks=tasks)
    batch_size = 16
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate.collate_fn)
    # for eval_step, batch_data in enumerate(train_dataloader):
    #   print(batch_data['raw_tokens'])
    #   exit(0)
