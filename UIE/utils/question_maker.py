import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import util
import time

# 1 分别编码所有的文本 触发词 和 问题
# 2 针对问题拼接

class Question_maker:
    def __init__(self, sim_model, demo_file, argument2question_path='data/ee/duee/argument2question.json') -> None:

        texts = []
        triggers = []
        questions = []
        text_trigger_question_map = []
        demo_embs = []
        demo_tuples = []

        with open(argument2question_path, 'r') as fp:
            self.argument2question = json.load(fp)

        self.sim_model = sim_model

        # 记录所有的
        with open(demo_file, encoding='utf-8') as f:
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
                        question = self.get_question_for_argument(
                            event_type=event_type, role=role)
                        questions.append(question)
                        text_trigger_question_map.append({
                            'text_idx': evt_idx,
                            'argu_idx': aru_idx,
                            'trigger_idx': tgr_idx
                        })
                        demo_tuples.append([
                            text,trigger,question
                        ])

        # 分批转化为embs
        
        text_embs = self.batch_encode(batch_size=1000, dataset=texts)
        question_embs = self.batch_encode(batch_size=1000, dataset=questions)
        trigger_embs = self.batch_encode(batch_size=1000, dataset=triggers)
       
        # 根据map拼接embs，用于相似度计算
        for idx, m in enumerate(text_trigger_question_map):
            ques_emb = question_embs[idx]
            text_emb = text_embs[m['text_idx']]
            tri_emb = trigger_embs[m['trigger_idx']]
            demo_embs.append(torch.from_numpy(np.concatenate((text_emb,tri_emb,ques_emb),axis=None)))
        
        self.demo_embs = demo_embs
        self.demo_tuples = demo_tuples

    def batch_encode(self, batch_size, dataset):
        ret = []
        it = iter(range(0, len(dataset), batch_size))
        for i in it:
            print(i)
            ret += self.get_sentence_embedding(dataset[i:i+batch_size])
        return ret

    def get_question_for_argument(self, event_type, role):
        complete_slot_str = event_type + "-" + role
        query_str = self.argument2question.get(complete_slot_str)
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

    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(
            -1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_sentence_embedding(self, sentences, cuda=True):

        # Load model from HuggingFace Hub
        tokenizer = AutoTokenizer.from_pretrained(self.sim_model)
        model = AutoModel.from_pretrained(self.sim_model)

        # Tokenize sentences
        encoded_input = tokenizer(
            sentences, padding=True, truncation=True, return_tensors='pt')

        if cuda:
            model = model.cuda()
            for key in encoded_input:
                encoded_input[key] = encoded_input[key].to(torch.device("cuda"))

        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Perform pooling. In this case, max pooling.
        sentence_embeddings = self.mean_pooling(
            model_output, encoded_input['attention_mask'])

        return sentence_embeddings.cpu()

    def semantic_search(self, context, question, trigger):
        t1 = time.time()
        tuple_emb = self.get_sentence_embedding([context, trigger,question],cuda=False)
        tuple_emb = torch.from_numpy(np.concatenate((tuple_emb[0],tuple_emb[1],tuple_emb[2]),axis=None))
        most_sim = util.semantic_search(
            [tuple_emb], self.demo_embs, top_k=2)[0][1]
        sim_context, sim_question, sim_ans = self.demo_tuples[most_sim['corpus_id']]
        t2 = time.time()
        running_time = t2-t1
        print('time cost : %.5f sec' % running_time)
        return sim_context, sim_question, sim_ans
