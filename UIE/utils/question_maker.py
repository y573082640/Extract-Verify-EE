import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import util
import time

# 1 分别编码所有的文本 触发词 和 问题
# 2 针对问题拼接

# 论元转化为问题
argument2question_path = 'data/ee/duee/argument2question.json'
with open(argument2question_path, 'r') as fp:
    argument2question = json.load(fp)


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
        query_str_final = "找到{}事件中的{},包括{}".format(
            event_type_str, role, query_str)
    return query_str_final


def creat_demo(sim_tuple):
    """
        从sim_tuple中提取相关信息，并插入特殊占位符，包括[TGR]，[ARG]，[DEMO]
    """
    sim_trigger = sim_tuple['trigger']
    sim_trigger_start_index = sim_tuple['trigger_start_index']
    sim_text_tokens = [i for i in sim_tuple['text']]
    sim_text_tokens.insert(sim_trigger_start_index, '[TGR]')
    sim_text_tokens.insert(
        sim_trigger_start_index + 1 + len(sim_trigger), '[TGR]')
    demo = ['[DEMO]'] + [i for i in sim_tuple['question']] + ['[SEP]'] + \
        sim_text_tokens + ['[SEP]', '[ARG]'] + \
        [i for i in sim_tuple['argument']] + ['[DEMO]']
    return demo


def creat_argu_labels(argu_token, demo, text_tuple, max_len):
    argu_start_labels = [0] * len(argu_token)
    argu_end_labels = [0] * len(argu_token)

    # 因为text中多加了[TGR]
    trigger = text_tuple["trigger"]
    trigger_start_index = text_tuple['trigger_start_index']
    # 用于增加对arg的偏置
    tgr1_index = trigger_start_index
    tgr2_index = trigger_start_index + 1 + len(trigger)
    argu = text_tuple["argument"]
    argument_start_index = text_tuple['argument_start_index']

    if tgr1_index <= argument_start_index:
        argument_start_index += 1
    if tgr2_index <= argument_start_index:
        argument_start_index += 1

    question = text_tuple['question']
    pre_tokens = demo + [i for i in question] + ['[SEP]']
    argu_start = len(pre_tokens) + 1 + argument_start_index
    argu_end = argu_start + len(argu) - 1

    if argu_end >= max_len - 1:
        return [], []

    argu_start_labels[argu_start] = 1
    argu_end_labels[argu_end] = 1

    return argu_start_labels, argu_end_labels


def creat_argu_token(text_tuple, demo, max_len):
    question = text_tuple['question']
    text = text_tuple['text']
    trigger = text_tuple["trigger"]
    trigger_start_index = text_tuple['trigger_start_index']
    text_tokens = [i for i in text]
    # 用于增加对arg的偏置
    tgr1_index = trigger_start_index
    tgr2_index = trigger_start_index + 1 + len(trigger)
    text_tokens.insert(tgr1_index, '[TGR]')
    text_tokens.insert(tgr2_index, '[TGR]')
    pre_tokens = demo + [i for i in question] + ['[SEP]']
    if len(text_tokens) + len(pre_tokens) > max_len - 2:
        argu_token = (pre_tokens + text_tokens)[:max_len-2]
    else:
        argu_token = pre_tokens + text_tokens
    argu_token = ['[CLS]'] + argu_token + ['[SEP]']
    return argu_token


class Sim_scorer:
    def __init__(self, sim_model) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(sim_model)
        self.model = AutoModel.from_pretrained(sim_model)

    def _mean_pooling(self, model_output, attention_mask):
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(
            -1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _batch_encode(self, batch_size, dataset):
        ret = []
        it = iter(range(0, len(dataset), batch_size))
        for i in it:
            ret += self.get_sentence_embedding(dataset[i:i+batch_size])
        return ret

    def get_sentence_embedding(self, sentences, cuda=True):

        # Tokenize sentences
        encoded_input = self.tokenizer(
            sentences, padding=True, truncation=True, return_tensors='pt')

        if cuda:
            self.model = self.model.cuda()
            for key in encoded_input:
                encoded_input[key] = encoded_input[key].to(
                    torch.device("cuda"))
        else:
            self.model = self.model.cpu()

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling. In this case, max pooling.
        sentence_embeddings = self._mean_pooling(
            model_output, encoded_input['attention_mask'])

        return sentence_embeddings.cpu()

    def create_embs_and_tuples(self, filename):
        texts = []
        triggers = []
        questions = []
        text_trigger_question_map = []
        embs = []
        tuples = []
        # 记录所有的
        with open(filename, encoding='utf-8') as f:
            f = f.read().strip().split("\n")
            for evt_idx, d in enumerate(f):
                d = json.loads(d)
                text = d["text"]
                texts.append(text)

                event_list = d["event_list"]
                if len(text) == 0:
                    continue

                for tgr_idx, event in enumerate(event_list):
                    event_type = event["event_type"]
                    arguments = event["arguments"]
                    trigger = event["trigger"]

                    triggers.append(trigger)
                    for aru_idx, argument in enumerate(arguments):
                        role = argument["role"]
                        question = get_question_for_argument(
                            event_type=event_type, role=role)
                        questions.append(question)
                        text_trigger_question_map.append({
                            'text_idx': evt_idx,
                            'argu_idx': aru_idx,
                            'trigger_idx': tgr_idx
                        })
                        tuples.append({
                            'text': text,
                            'trigger': trigger,
                            'question': question,
                            'trigger_start_index': event["trigger_start_index"],
                            'argument': argument["argument"],
                            'argument_start_index': argument["argument_start_index"],
                            'role': argument["role"],
                            'event_type': event["event_type"]
                        })

        # 分批转化为embs

        text_embs = self._batch_encode(batch_size=1000, dataset=texts)
        question_embs = self._batch_encode(batch_size=1000, dataset=questions)
        trigger_embs = self._batch_encode(batch_size=1000, dataset=triggers)

        # 根据map拼接embs，用于相似度计算
        for idx, m in enumerate(text_trigger_question_map):
            ques_emb = question_embs[idx]
            text_emb = text_embs[m['text_idx']]
            tri_emb = trigger_embs[m['trigger_idx']]
            embs.append(torch.from_numpy(np.concatenate(
                (text_emb, tri_emb, ques_emb), axis=None)))

        return embs, tuples

    def sim_match(self, text_embs, demo_embs):
        most_sim = util.semantic_search(text_embs, demo_embs, top_k=2)
        ret = []
        for top2_list in most_sim:
            ret.append(top2_list[1])
        return ret


if __name__ == '__main__':

    test_file = '/home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/data/ee/duee/duee_train.json'
    model = 'model_hub/paraphrase-MiniLM-L6-v2'
    t1 = time.time()
    scorer = Sim_scorer(model)
    scorer.sim_match(test_file, test_file)
    t2 = time.time()
    running_time = t2-t1
    print('time cost : %.5f sec' % running_time)
