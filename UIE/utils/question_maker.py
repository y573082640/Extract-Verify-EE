import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import util
import time
import re
import logging


def find_all(sub, ori):
    return [substr.start() for substr in re.finditer(sub, ori)]


# 1 分别编码所有的文本 触发词 和 问题
# 2 针对问题拼接

# 论元转化为问题
argument2question_path = "data/ee/duee/argument2question.json"
with open(argument2question_path, "r") as fp:
    argument2question = json.load(fp)


def apply_label_smoothing(label_list, factor=0.1):
    smoothed_labels = np.full((len(label_list),), factor / (len(label_list) - 1))
    true_label_indices = np.nonzero(label_list)[0]
    smoothed_labels[true_label_indices] = 1 - factor
    return smoothed_labels.tolist()


def create_role_tuple(data, add_trigger=False):
    tuples = []
    concat_texts = []
    text = data["text"]
    event_list = data["event_list"]

    if add_trigger:
        for tgr_idx, event in enumerate(event_list):
            event_type = event["event_type"]
            trigger = event["trigger"]
            role_dict = {}
            for aru_idx, argument in enumerate(event["arguments"]):
                key = event_type + ":" + argument["role"]
                if key not in role_dict:
                    role_dict[key] = [argument]
                else:
                    role_dict[key].append(argument)

            for key, arguments in role_dict.items():
                event_type = key.split(":")[0]
                role = key.split(":")[1]
                question = get_question_for_argument(event_type=event_type, role=role)
                concat_texts.append(
                    "文本长度为%d,事件类型为%s，触发词为%s，%s。"
                    % (len(text) // 30, event_type, trigger, question)
                )
                tuples.append(
                    {
                        "text": text,
                        "trigger": trigger,
                        "question": question,
                        "trigger_start_index": event["trigger_start_index"],
                        "arguments": arguments,
                        "role": role,
                        "event_type": event_type,
                    }
                )
    else:
        role_dict = {}
        for tgr_idx, event in enumerate(event_list):
            event_type = event["event_type"]
            trigger = event["trigger"]
            for aru_idx, argument in enumerate(event["arguments"]):
                key = event_type + ":" + argument["role"]
                if key not in role_dict:
                    role_dict[key] = [argument]
                else:
                    role_dict[key].append(argument)

        for key, arguments in role_dict.items():
            event_type = key.split(":")[0]
            role = key.split(":")[1]
            question = get_question_for_argument(event_type=event_type, role=role)
            concat_texts.append(
                "文本长度为%d,事件类型为%s，%s。" % (len(text) // 30, event_type, question)
            )
            tuples.append(
                {
                    "text": text,
                    "question": question,
                    "arguments": arguments,
                    "role": role,
                    "event_type": event_type,
                }
            )
    return tuples, concat_texts


def create_tri_tuple(data):
    tuples = []
    concat_texts = []
    text = data["text"]
    event_list = data["event_list"]
    evt_dict = {}
    for tgr_idx, event in enumerate(event_list):
        event_type = event["event_type"]
        if event_type not in evt_dict:
            evt_dict[event_type] = [event]
        else:
            evt_dict[event_type].append(event)

    for etype, events in evt_dict.items():
        question = "{}事件的关键词是什么？".format(etype)
        concat_texts.append(text)
        tuples.append(
            {
                "text": text,
                "question": question,
                "events": events,
                "event_type": etype,
            }
        )
    return tuples, concat_texts


def create_role_tuple_for_predict(data_list, label2role):
    """
    用于推理
    event = {
        "event_type":e_type,
        'text':text,
        'trigger':trg_tuple[0],
        'trigger_start_index':trg_tuple[1],
        'event_id':text_id+"——"+str(i), # 聚合事件论元 和 聚合事件列表
    }
    """
    tuples, concat_texts = [], []
    for tgr_idx, event in enumerate(data_list):
        event_type = event["event_type"]
        argu_types = label2role[event_type]
        textb = event["text"]
        event_id = event["event_id"]
        for role in argu_types:
            # 组织行为-游行_时间
            # 此处是为了配合get_question_for_argument函数，转变为"时间"
            role = role.split("_")[-1]
            q = get_question_for_argument(event_type, role)
            if "trigger_start_index" in event:
                trigger_start_index = event["trigger_start_index"]
                trigger = event["trigger"]
                text_tuple = {
                    "text": textb,
                    "trigger": trigger,
                    "question": q,
                    "trigger_start_index": trigger_start_index,
                    "arguments": None,
                    "role": role,
                    "event_type": event_type,
                    "event_id": event_id,
                }
                concat_texts.append(
                        "文本长度为%d,事件类型为%s，触发词为%s，%s。"
                        % (len(textb) // 30, event_type, trigger, q)
                )
            else:
                text_tuple = {
                    "text": textb,
                    "question": q,
                    "arguments": None,
                    "role": role,
                    "event_type": event_type,
                    "event_id": event_id,
                }
                concat_texts.append(
                    "文本长度为%d,事件类型为%s，%s。" % (len(textb) // 30, event_type, q)
                )
            tuples.append(text_tuple)
    return tuples, concat_texts


def get_question_for_verify_trigger(event_type, role):
    complete_slot_str = event_type + "-" + role
    query_str = argument2question.get(complete_slot_str)
    event_type_str = event_type.split("-")[-1]
    if query_str.__contains__("？"):
        query_str = query_str.replace("？", "")
    if query_str == role:
        query_str_final = "前文包含{}事件中的{}吗？".format(event_type_str, role)
    elif role == "时间":
        query_str_final = "前文包含{}{}吗？".format(event_type_str, query_str)
    else:
        query_str_final = "前文包含{}事件中的{},包括{}吗？".format(event_type_str, role, query_str)
    return query_str_final


def get_question_for_verify(event_type, role):
    complete_slot_str = event_type + "-" + role
    query_str = argument2question.get(complete_slot_str)
    event_type_str = event_type.split("-")[-1]
    if query_str.__contains__("？"):
        query_str = query_str.replace("？","")
    if query_str == role:
        query_str_final = "前文包含{}事件中的{}吗？".format(event_type_str, role)
    elif role == "时间":
        query_str_final = "前文包含{}{}吗？".format(event_type_str, query_str)
    else:
        query_str_final = "前文包含{}事件中的{},包括{}吗？".format(
            event_type_str, role, query_str)
    return query_str_final

    ## 新版不如旧版，所以取消了
    # complete_slot_str = event_type + "-" + role
    # query_str = argument2question.get(complete_slot_str)
    # event_type_str = event_type.split("-")[-1]
    # if query_str.__contains__("？"):
    #     query_str = query_str.replace("？", "")
    # if query_str == role:
    #     query_str_final = "{}事件的{}".format(event_type_str, role)
    # elif role == "时间":
    #     query_str_final = "{}{}".format(event_type_str, query_str)
    # else:
    #     query_str_final = "{}事件的{},比如{}".format(event_type_str, role, query_str)
    # return query_str_final


def get_question_for_argument(event_type, role):
    complete_slot_str = event_type + "-" + role
    query_str = argument2question.get(complete_slot_str)
    event_type_str = event_type.split("-")[-1]
    if query_str.__contains__("？"):
        query_str_final = query_str
    if query_str == role:
        query_str_final = "找到{}事件中的{}".format(event_type_str, role)
    elif role == "时间":
        query_str_final = "找到{}{}".format(event_type_str, query_str)
    else:
        query_str_final = "找到{}事件中的{},包括{}".format(event_type_str, role, query_str)
    # query_str_final = " unused9 " + query_str_final + " unused10 "
    return query_str_final


# def get_question_for_argument(event_type, role):
#     complete_slot_str = event_type + "-" + role
#     query_str = argument2question.get(complete_slot_str)
#     event_type_str = event_type.split("-")[-1]
#     if query_str.__contains__("？"):
#         query_str_final = query_str
#     if query_str == role:
#         query_str_final = "{}的{}".format(event_type_str, role)
#     elif role == "时间":
#         query_str_final = "{}{}".format(event_type_str, query_str)
#     else:
#         query_str_final = "{}的{},包括{}".format(event_type_str, role, query_str)
#     query_str_final = " unused9 " + query_str_final + " unused10 "
#     return query_str_final


def creat_demo(sim_tuple):
    """
    从sim_tuple中提取相关信息，并插入特殊占位符，包括[TGR] ，[ARG]，[DEMO]
    """
    # sim_trigger = sim_tuple["trigger"]
    # sim_trigger_start_index = sim_tuple["trigger_start_index"]
    # sim_text_tokens = [i for i in sim_tuple["text"]]
    # sim_text_tokens.insert(sim_trigger_start_index, "[TGR]")
    # sim_text_tokens.insert(sim_trigger_start_index + 1 + len(sim_trigger), "[TGR]")

    # answers = []
    # for argu in sim_tuple["arguments"][:3]:
    #     answers.append("[ARG]")
    #     answers.append(argu["argument"])

    # demo = (
    #     ["[DEMO]"]
    #     + [i for i in sim_tuple["question"]]
    #     + ["[SEP]"]
    #     + sim_text_tokens
    #     + ["[SEP]", "答案是："]
    #     + answers
    #     + ["[DEMO]"]
    # )

    demo = [i for i in sim_tuple["text"]]
    insert_indexes = []
    argu_set = dict()
    for argu in sim_tuple["arguments"]:
        if (
            argu["argument"] + ":{}".format(argu["argument_start_index"])
            not in argu_set
        ):
            insert_indexes.append(
                {
                    "argument_start_index": argu["argument_start_index"],
                    "length": len(argu["argument"]),
                }
            )
            argu_set[argu["argument"] + ":{}".format(argu["argument_start_index"])] = 1

    insert_indexes.sort(key=lambda x: x["argument_start_index"])
    for i in range(len(insert_indexes)):
        insert_indexes[i]["argument_start_index"] = (
            insert_indexes[i]["argument_start_index"] + i * 2
        )  ## 需要插入2个特殊标签[ARG]

    for argu in insert_indexes:
        index = argu["argument_start_index"]
        length = argu["length"]
        demo.insert(index, "[ARG]")
        demo.insert(index + length + 1, "[ARG]")
    # if len(insert_indexes) >=2 :
    #     print(demo)
    #     exit(0)
    # logging.debug(demo)
    # logging.debug(sim_tuple)
    return demo


def creat_argu_labels(argu_token, demo, text_tuple, max_len):
    argu_start_labels = [0] * len(argu_token)
    argu_end_labels = [0] * len(argu_token)

    # 因为text中多加了[TGR]
    if "trigger" in text_tuple:
        trigger = text_tuple["trigger"]
        trigger_start_index = text_tuple["trigger_start_index"]
        ## 用于增加对arg的偏置
        tgr1_index = trigger_start_index
        tgr2_index = trigger_start_index + 1 + len(trigger)
    else:
        tgr1_index = None
        tgr2_index = None

    ## 用于计算应该给argument_start_index加多少偏置
    question = text_tuple["question"]
    question = [i for i in question if i != " "]
    # 计算文本的偏置pre_tokens，用于构造label
    if demo is not None:
        pre_tokens = ["[CLS]"] + question + ["[SEP]"] + demo + ["[SEP]"]
    else:
        pre_tokens = ["[CLS]"] + question + ["[SEP]"]

    ## 用于计算所有事件论的起止位置
    argu_tuples = []
    for argu in text_tuple["arguments"]:
        argument_start_index = argu["argument_start_index"]
        argument_text = argu["argument"]
        if tgr2_index and tgr1_index:
            if tgr1_index <= argument_start_index:
                argument_start_index += 1
            if tgr2_index <= argument_start_index:
                argument_start_index += 1

        ## 注意-1
        argu_start = len(pre_tokens) + argument_start_index
        argu_end = argu_start + len(argument_text) - 1

        ## 长文本特例
        if argu_end < max_len:
            argu_start_labels[argu_start] = 1
            argu_end_labels[argu_end] = 1
            argu_tuples.append((argu_start, argu_end + 1))

    return argu_start_labels, argu_end_labels, argu_tuples


def creat_argu_token(text_tuple, demo, max_len):
    question = text_tuple["question"]

    text = text_tuple["text"]
    text_tokens = [i for i in text]
    if "trigger" in text_tuple:
        trigger = text_tuple["trigger"]
        trigger_start_index = text_tuple["trigger_start_index"]
        # 用于增加对arg的偏置
        tgr1_index = trigger_start_index
        tgr2_index = trigger_start_index + 1 + len(trigger)
        text_tokens.insert(tgr1_index, "[TGR]")
        text_tokens.insert(tgr2_index, "[TGR]")

    # 适应拼接或者不拼接的状态

    question = [i for i in question if i != " "]

    if demo is not None:
        pre_tokens = question + ["[SEP]"] + demo + ["[SEP]"]
    else:
        pre_tokens = question + ["[SEP]"]

    if len(text_tokens) + len(pre_tokens) > max_len - 2:
        argu_token = (pre_tokens + text_tokens)[: max_len - 2]
    else:
        argu_token = pre_tokens + text_tokens

    # 首尾补充特殊字符
    argu_token = ["[CLS]"] + argu_token + ["[SEP]"]
    token_type_ids = [0] * len(argu_token)  # [CLS] +　pre_tokens

    return argu_token, token_type_ids


class Sim_scorer:
    def __init__(self, sim_model) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(sim_model)
        self.model = AutoModel.from_pretrained(sim_model)

    def _mean_pooling(self, model_output, attention_mask):
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def _batch_encode(self, batch_size, dataset):
        ret = []
        it = iter(range(0, len(dataset), batch_size))
        for i in it:
            ret += self.get_sentence_embedding(dataset[i : i + batch_size])
        return ret

    def get_sentence_embedding(self, sentences, cuda=True):
        # Tokenize sentences
        encoded_input = self.tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        )

        if cuda:
            self.model = self.model.cuda()
            for key in encoded_input:
                encoded_input[key] = encoded_input[key].to(torch.device("cuda"))
        else:
            self.model = self.model.cpu()

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling. In this case, max pooling.
        sentence_embeddings = self._mean_pooling(
            model_output, encoded_input["attention_mask"]
        )

        return sentence_embeddings.cpu()

    def create_embs_and_tuples_for_obj_predict(self, data_list, label2role):
        concat_texts = []
        embs = []
        tuples = []
        # 用于训练
        text_tuples, concat_texts = create_role_tuple_for_predict(data_list, label2role)
        embs = self._batch_encode(batch_size=32, dataset=concat_texts)
        return embs, text_tuples

    def create_embs_and_tuples(self, filename=None, task="obj", add_trigger=False):
        concat_texts = []
        embs = []
        tuples = []
        # 用于训练
        if task == "obj":
            with open(filename, encoding="utf-8") as f:
                f = f.read().strip().split("\n")
                for evt_idx, line in enumerate(f):
                    evt = json.loads(line)
                    text = evt["text"]
                    if len(text) == 0:
                        continue
                    batch_tuples, batch_concat_texts = create_role_tuple(
                        evt, add_trigger
                    )
                    concat_texts += batch_concat_texts
                    tuples += batch_tuples

            embs = self._batch_encode(batch_size=32, dataset=concat_texts)
        elif task == "tri":
            with open(filename, encoding="utf-8") as f:
                f = f.read().strip().split("\n")
                for evt_idx, line in enumerate(f):
                    evt = json.loads(line)
                    batch_tuples, batch_concat_texts = create_tri_tuple(evt)
                    concat_texts += batch_concat_texts
                    tuples += batch_tuples
            embs = self._batch_encode(batch_size=32, dataset=concat_texts)
        elif task == "ner":
            with open(filename, encoding="utf-8") as f:
                f = f.read().strip().split("\n")
                for evt_idx, line in enumerate(f):
                    evt = json.loads(line)
                    text = evt["text"]
                    if len(text) == 0:
                        continue
                    cnt = len(evt["event_list"])
                    key_for_embs = "事件数量为{},文本长度为{},包含事件:".format(cnt, len(text))
                    for e in evt["event_list"]:
                        key_for_embs += e["event_type"] + "-" + e["trigger"] + ","
                        del e["arguments"]
                    concat_texts.append(key_for_embs)

                    tuples.append(evt)
            embs = self._batch_encode(batch_size=32, dataset=concat_texts)
        else:
            raise AttributeError("【create_embs_and_tuples】")
        return embs, tuples

    def sim_match(self, text_embs, demo_embs, ignore_first=False):
        most_sim = util.semantic_search(text_embs, demo_embs, top_k=3)
        ret = []
        for top_list in most_sim:
            tmp = []
            if ignore_first:
                top_list = top_list[1:]
            for i in range(len(top_list)):
                tmp.append(top_list[i]["corpus_id"])
            ret.append(tmp)
        return ret


if __name__ == "__main__":
    test_file = "/home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/data/ee/duee/duee_train.json"
    model = "model_hub/paraphrase-MiniLM-L6-v2"
    t1 = time.time()
    scorer = Sim_scorer(model)
    scorer.sim_match(test_file, test_file)
    t2 = time.time()
    running_time = t2 - t1
    print("time cost : %.5f sec" % running_time)
