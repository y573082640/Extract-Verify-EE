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
from utils.question_maker import get_question_for_argument,create_role_tuple,create_role_tuple_for_predict
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
            self.process_list_data()
        else:
            raise ValueError(
                'The input args shall be str format file_path / list format dataset')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_data(file_path, tokenizer, max_len, entity_label, tasks):
        return file_path

    def process_list_data(self):
        pass

# 加载实体识别数据集


class EeDatasetPredictor(ListDataset):

    def process_list_data(self):
        """
        event = {
            "event_type":e_type,
            'text':text,
            'trigger':trg_tuple[0],
            'trigger_start_index':trg_tuple[1],
            'event_id':text_id+"——"+str(i), # 聚合事件论元 和 聚合事件列表
        }
        """
        ret = []
        if self.args.use_demo:
            sim_scorer = self.args.sim_scorer
            text_embs, text_tuples = sim_scorer.create_embs_and_tuples(
                data_list=self.data, label2role=self.args.label2role)
            demo_embs = self.args.demo_embs
            demo_tuples = self.args.demo_tuples
            most_sim = sim_scorer.sim_match(
                text_embs, demo_embs, rank=0)  # {corpus_id,score} rank表示取最相似的还是次相似的
            for idx, text_tuple in enumerate(text_tuples):
                sim_id = most_sim[idx]['corpus_id']
                demo = creat_demo(demo_tuples[sim_id])
                obj_tokens, token_type_ids = creat_argu_token(
                    text_tuple, demo, self.args.max_seq_len)
                if self.args.use_lexicon:
                    augment_Ids = generate_instance_with_gaz(
                        obj_tokens,
                        self.args.pos_alphabet,
                        self.args.word_alphabet,
                        self.args.biword_alphabet,
                        self.args.gaz_alphabet,
                        self.args.gaz_alphabet_count,
                        self.args.gaz,
                        self.args.max_seq_len)
                    augment_Ids = batchify_augment_ids(
                        [augment_Ids], self.args.max_seq_len)
                else:
                    augment_Ids = []

                ret.append({
                    'obj_tokens': obj_tokens,
                    'token_type_ids': token_type_ids,
                    'augment_Ids': augment_Ids,
                    'event_id': text_tuple['event_id'],
                    'role': text_tuple['role'],
                })
        else:
            text_tuples,_ = create_role_tuple_for_predict(data_list=self.data, label2role=self.args.label2role)
            for idx, text_tuple in enumerate(text_tuples):
                obj_tokens, token_type_ids = creat_argu_token(
                    text_tuple, None, self.args.max_seq_len)
                if self.args.use_lexicon:
                    augment_Ids = generate_instance_with_gaz(
                        obj_tokens,
                        self.args.pos_alphabet,
                        self.args.word_alphabet,
                        self.args.biword_alphabet,
                        self.args.gaz_alphabet,
                        self.args.gaz_alphabet_count,
                        self.args.gaz,
                        self.args.max_seq_len)
                    augment_Ids = batchify_augment_ids(
                        [augment_Ids], self.args.max_seq_len)
                else:
                    augment_Ids = []

                ret.append({
                    'obj_tokens': obj_tokens,
                    'token_type_ids': token_type_ids,
                    'augment_Ids': augment_Ids,
                    'event_id': text_tuple['event_id'],
                    'role': text_tuple['role'],
                })

        self.data = ret
        logging.info('data_list数据预处理完成')
        return

    def load_data(self,  filename, tokenizer, max_len, entity_label=None, tasks=None):
        ret = []
        if 'ner' in self.args.tasks:
            with open(filename, encoding='utf-8') as f:
                f = f.read().strip().split("\n")
                for d in f:
                    d = json.loads(d)
                    text = d["text"]
                    id = d['id']

                    # 简单预处理
                    text = text.replace('\n', "。")
                    if text[-1] != '。':
                        text += '。'

                    tokens_b = [i for i in text]

                    if self.args.use_lexicon:
                        augment_Ids = generate_instance_with_gaz(
                            ['CLS']+tokens_b[:self.args.max_seq_len-2]+[
                                'SEP'], self.args.pos_alphabet, self.args.word_alphabet, self.args.biword_alphabet,
                            self.args.gaz_alphabet, self.args.gaz_alphabet_count, self.args.gaz, self.args.max_seq_len)
                    else:
                        augment_Ids = []

                    ret.append({
                        'ner_tokens': tokens_b,
                        "text_id": id,
                        'augment_Ids': augment_Ids
                    })

        elif 'obj' in self.args.tasks:
            # obj直接读取list了
            pass

        return ret


class EeCollatePredictor:
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
        batch_ner_attention_mask = []
        batch_ner_token_type_ids = []
        batch_obj_token_ids = []
        batch_obj_attention_mask = []
        batch_obj_token_type_ids = []
        batch_argu_roles = []  # 用于构建提交文件
        batch_raw_tokens = []  # 用于构建输出
        batch_text_ids = []  # 用于构建提交文件
        batch_augment_Ids = []

        for i, data in enumerate(batch):

            augment_Ids = data["augment_Ids"]
            batch_augment_Ids.append(augment_Ids)

            if "ner" in self.tasks:
                tokens = data["ner_tokens"]
                ids = data["text_id"]
                encode_dict = self.args.tokenizer.encode_plus(text=tokens,
                                                              max_length=self.args.max_seq_len,
                                                              padding="max_length",
                                                              truncation=True,
                                                              return_token_type_ids=True,
                                                              return_attention_mask=True)
                ner_token_ids = encode_dict['input_ids']
                ner_token_type_ids = encode_dict['token_type_ids']
                ner_attention_mask = encode_dict['attention_mask']

                batch_ner_token_ids.append(ner_token_ids)
                batch_ner_attention_mask.append(ner_attention_mask)
                batch_ner_token_type_ids.append(ner_token_type_ids)
                batch_raw_tokens.append(tokens)
                batch_text_ids.append(ids)

            elif "obj" in self.tasks:
                obj_tokens = data["obj_tokens"]
                role = data['role']
                event_id = data['event_id']
                raw_tokens = obj_tokens
                obj_tokens = self.tokenizer.convert_tokens_to_ids(obj_tokens)
                obj_token_type_ids = data["token_type_ids"]

                if len(obj_tokens) < self.maxlen:
                    obj_attention_mask = [
                        1] * len(obj_tokens) + [0] * (self.maxlen - len(obj_tokens))
                    obj_token_type_ids = obj_token_type_ids + [0] * \
                        (self.maxlen - len(obj_tokens))
                    obj_tokens = obj_tokens + [0] * \
                        (self.maxlen - len(obj_tokens))
                else:
                    obj_attention_mask = [1] * self.maxlen

                batch_obj_token_ids.append(obj_tokens)
                batch_obj_attention_mask.append(obj_attention_mask)
                batch_obj_token_type_ids.append(obj_token_type_ids)
                batch_raw_tokens.append(raw_tokens)
                batch_argu_roles.append(role)
                batch_text_ids.append(event_id)

        res = {}

        if "ner" in self.tasks:
            try:
                batch_ner_token_ids = convert_list_to_tensor(
                    batch_ner_token_ids)
                batch_ner_attention_mask = convert_list_to_tensor(
                    batch_ner_attention_mask)
                batch_ner_token_type_ids = convert_list_to_tensor(
                    batch_ner_token_type_ids)
            except Exception as e:
                logging.info(e)
                logging.info(batch_ner_token_ids)
                for i in batch_ner_token_ids:
                    logging.info(len(i))
                    logging.info(isinstance(i, list))
                exit(0)

            ner_res = {
                "ner_input_ids": batch_ner_token_ids,
                "ner_attention_mask": batch_ner_attention_mask,
                "ner_token_type_ids": batch_ner_token_type_ids,
            }

            res = ner_res

        elif "obj" in self.tasks:
            batch_obj_token_ids = convert_list_to_tensor(batch_obj_token_ids)
            batch_obj_attention_mask = convert_list_to_tensor(
                batch_obj_attention_mask)
            batch_obj_token_type_ids = convert_list_to_tensor(
                batch_obj_token_type_ids)

            sbj_obj_res = {
                "re_obj_input_ids": batch_obj_token_ids,
                "re_obj_attention_mask": batch_obj_attention_mask,
                "re_obj_token_type_ids": batch_obj_token_type_ids,
                "argu_roles": batch_argu_roles,
            }

            res = sbj_obj_res

        # 用于解码
        res['raw_tokens'] = batch_raw_tokens
        # 用于构建最终输出
        res['text_ids'] = batch_text_ids
        # 用于词典增强
        if self.args.use_lexicon:
            batch_augment_Ids = batchify_augment_ids(
                batch_augment_Ids, self.maxlen)
        res['batch_augment_Ids'] = batch_augment_Ids

        return res


# 加载实体识别数据集
class EeDataset(ListDataset):

    def load_data(self, filename, tokenizer, max_len, entity_label=None, tasks=None):
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
                        
                        # span_tuple[event_type].append()
                    event_data = {
                        "ner_tokens": event_tokens,
                        "ner_start_labels": event_start_labels,
                        "ner_end_labels": event_end_labels,
                        "augment_Ids": augment_Ids
                    }

                    ner_data.append(event_data)

        elif "obj" in tasks:
            if self.args.use_demo: ### 如果检索相似样例，则需要为文本构建检索向量
                logging.info('...构造文本embedding')
                sim_scorer = self.args.sim_scorer
                embs, tuples = sim_scorer.create_embs_and_tuples(filename)
                logging.info('文本embedding构造完毕')
                # 此处用训练集作为demo库
                demo_embs = self.args.demo_embs
                demo_tuples = self.args.demo_tuples
                most_sim = sim_scorer.sim_match(
                    embs, demo_embs, rank=1 if filename == self.args.demo_path else 0)  # {corpus_id,score}
                logging.info('相似度匹配完成')
                # logging.info(1 if filename == self.args.demo_path else 0)
                for idx, text_tuple in enumerate(tuples):
                    sim_id = most_sim[idx]['corpus_id']
                    sim_score = most_sim[idx]['score']

                    demo = creat_demo(demo_tuples[sim_id])

                    if len(demo) + len(text_tuple['question']+text_tuple['text']) + 1> max_len - 2: ### +1是因为中间有个[SEP]
                        demo = None

                    argu_token, obj_token_type_ids = creat_argu_token(
                        text_tuple, demo, max_len)
                    argu_start_labels, argu_end_labels, argu_tuples = creat_argu_labels(
                        argu_token, demo, text_tuple, max_len)

                    # logging.info('============')
                    # logging.info(sim_score)
                    # logging.info("".join(
                    #     demo_tuples[sim_id]['question'] + "--" + demo_tuples[sim_id]['trigger']))
                    # logging.info(
                    #     "".join(text_tuple['question'] + "--" + text_tuple['trigger']))

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
                        'obj_token_type_ids': obj_token_type_ids,
                        "augment_Ids": augment_Ids,
                        'sim_score': 1 if sim_score > 0.75 else 0,
                        "argu_tuples" : argu_tuples
                    }

                    obj_data.append(argu_data)
            else:
                tuples = []
                with open(filename,'r',encoding='utf-8') as f:
                    f = f.read().strip().split("\n")
                    for evt_idx, line in enumerate(f):
                        evt = json.loads(line)
                        text = evt["text"]
                        if len(text) == 0:
                            continue
                        batch_tuples,_ = create_role_tuple(evt)
                        tuples += batch_tuples
                for text_tuple in tuples:
                    argu_token, obj_token_type_ids = creat_argu_token(
                        text_tuple=text_tuple, 
                        demo=None, 
                        max_len=max_len)
                    
                    argu_start_labels, argu_end_labels,argu_tuples = creat_argu_labels(
                        argu_token=argu_token, 
                        demo=None, 
                        text_tuple=text_tuple, 
                        max_len=max_len)

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
                        'obj_token_type_ids': obj_token_type_ids,
                        "augment_Ids": augment_Ids,
                        'sim_score': 0,
                        "argu_tuples":argu_tuples
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
        batch_obj_argu_tuples = []
        batch_augment_Ids = []
        batch_sim_score = []
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
                obj_token_type_ids = data["obj_token_type_ids"]
                obj_argu_tuples = data['argu_tuples']
                sim_score = data['sim_score']

                if len(obj_tokens) < self.maxlen:
                    obj_start_labels = obj_start_labels + \
                        [0] * (self.maxlen - len(obj_tokens))
                    obj_end_labels = obj_end_labels + \
                        [0] * (self.maxlen - len(obj_tokens))
                    obj_attention_mask = [
                        1] * len(obj_tokens) + [0] * (self.maxlen - len(obj_tokens))
                    obj_token_type_ids = obj_token_type_ids + [0] * \
                        (self.maxlen - len(obj_tokens))
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
                batch_sim_score.append(sim_score)
                batch_obj_argu_tuples.append(obj_argu_tuples)

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
            batch_sim_score = convert_list_to_tensor(
                batch_sim_score, dtype=torch.float)

            sbj_obj_res = {
                "re_obj_input_ids": batch_obj_token_ids,
                "re_obj_attention_mask": batch_obj_attention_mask,
                "re_obj_token_type_ids": batch_obj_token_type_ids,
                "re_obj_start_labels": batch_obj_start_labels,
                "re_obj_end_labels": batch_obj_end_labels,
                'batch_sim_score': batch_sim_score,
                "argu_tuples" : batch_obj_argu_tuples
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
