import sys
import time
sys.path.append("..")
import numpy as np
import json
import torch
from transformers import BertTokenizer
from UIE.ee_main import EePipeline
from UIE.model import UIEModel
from UIE.config import EeArgs
import logging
import tqdm

class Predictor:
    def __init__(self, ner_args=None, obj_args=None):
        self.obj_args = obj_args
        self.ner_args = ner_args
    
        model = UIEModel(ner_args)
        self.ner_pipeline = EePipeline(model, ner_args)
        self.ner_pipeline.load_model()

        model = UIEModel(obj_args)
        self.obj_pipeline = EePipeline(model, obj_args)
        self.obj_pipeline.load_model()

    def predict_ner(self, text):
        entities = self.ner_pipeline.predict(text)
        return entities

    def predict_obj(self, text, event_type, trigger, trigger_start_index):
        ret = self.obj_pipeline.predict(text, event_type, trigger, trigger_start_index)
        return ret 

    def joint_predict(self, texts):
        ret = []
        for t in tqdm.tqdm(texts):
            event_list = {
                "id":t[1],
                # 'text':t[0],
                'event_list':[]
            }
            t_text = t[0]
            entities = self.predict_ner(t_text)
            # {"灾害/意外-坍/垮塌": [('坍塌', 7)], "人生-失联": [('失联', 17)]}
            for e_type, e_trgs in entities.items():
                for trg_tuple in e_trgs:
                    event = {
                        "event_type":e_type,
                        'arguments':[]
                    }
                    argus = self.predict_obj(t_text, event_type=e_type, trigger=trg_tuple[0], trigger_start_index=trg_tuple[1])
                    event['arguments'] = argus
                    event_list['event_list'].append(event)
            ret.append(event_list)
        fp = open('log/output-' + str(time.time())[:10]+ '.json','a+')
        for r in ret:
            json.dump(r,fp,ensure_ascii=False)
            fp.write('\n')
        return 
            

if __name__ == "__main__":
    ner_args = EeArgs('ner', use_lexicon=True,log=False)
    obj_args = EeArgs('obj', use_lexicon=False,log=True)
    predict_tool = Predictor(ner_args, obj_args)
    texts = []
    with open('/home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/data/ee/duee/duee_test2.json', encoding='utf-8') as f:
        f = f.read().strip().split("\n")
        for d in f:
            d = json.loads(d)
            text = d["text"]
            id = d['id']
            if len(text) < ner_args.max_seq_len and len(text) < obj_args.max_seq_len:
                # TODO:换更长的
                text = text.replace('\n',"。")
                texts.append((text,id))
    predict_tool.joint_predict(texts)
