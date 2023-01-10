import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import util


class Question_maker:
    def __init__(self, sim_model, demo_file, argument2question_path='data/ee/duee/argument2question.json') -> None:
        argu_texts = []
        argu_embs = []

        with open(argument2question_path, 'r') as fp:
            self.argument2question = json.load(fp)

        self.sim_model = sim_model

        with open(demo_file, encoding='utf-8') as f:
            f = f.read().strip().split("\n")
            cnt = 0
            for d in f:
                d = json.loads(d)
                text = d["text"]
                event_list = d["event_list"]
                if len(text) == 0:
                    continue
                print(cnt)
                cnt+=1
                for event in event_list:
                    event_type = event["event_type"]
                    arguments = event["arguments"]
                    for idx, argument in enumerate(arguments):
                        role = argument["role"]
                        question = self.get_question_for_argument(
                            event_type=event_type, role=role)
                        argu_texts.append(
                            [text, question, argument['argument'], d['id'], str(idx)])
                        argu_embs.append(self.get_sentence_embedding([text,question,event['trigger']]))

        self.argu_texts = argu_texts
        self.argu_embs = argu_embs


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

    def get_sentence_embedding(self, sentences):

        # Load model from HuggingFace Hub
        tokenizer = AutoTokenizer.from_pretrained(self.sim_model)
        model = AutoModel.from_pretrained(self.sim_model)

        # Tokenize sentences
        encoded_input = tokenizer(
            sentences, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Perform pooling. In this case, max pooling.
        sentence_embeddings = self.mean_pooling(
            model_output, encoded_input['attention_mask'])
        
        total = np.concatenate((sentence_embeddings[0],  # 文本
                        sentence_embeddings[1],  # 问题
                        sentence_embeddings[2]),  # 触发词
            axis=None)

        return torch.from_numpy(total)

    def semantic_search(self, context, question, trigger):
        tuple_emb = self.get_sentence_embedding([context, question, trigger])
        most_sim = util.semantic_search([tuple_emb], self.argu_embs, top_k=2)[0][1]
        sim_context, sim_question, sim_ans, _, _ = self.argu_texts[most_sim['corpus_id']]

        return sim_context, sim_question, sim_ans
