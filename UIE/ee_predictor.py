import sys
sys.path.append("..")
import numpy as np
import json
import torch
from transformers import BertTokenizer
from UIE.ee_main import EePipeline
from UIE.model import UIEModel
from UIE.config import EeArgs



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
        self.obj_pipeline.predict(text, event_type, trigger, trigger_start_index)
        return 

    def joint_predict(self, texts):
        for t in texts:
            entities = self.predict_ner(t)
            # {"灾害/意外-坍/垮塌": [('坍塌', 7)], "人生-失联": [('失联', 17)]}
            for e_type, e_trgs in entities.items():
                for trg_tuple in e_trgs:
                    self.predict_obj(t, event_type=e_type, trigger=trg_tuple[0], trigger_start_index=trg_tuple[1])

        return 
            

if __name__ == "__main__":
    texts = [
        "目前，《名侦探柯南》第23部剧场版《绀青之拳》已确定8月份在国内上映，但具体日期还没决定，不过这个消息对于柯南迷来说确实非常激动",
        "2019年7月12日，国家市场监督管理总局缺陷产品管理中心，在其官方网站和微信公众号上发布了《上海施耐德低压终端电器有限公司召回部分剩余电流保护装置》，看到这条消息，确实令人震惊！\n作为传统的三大外资品牌之一，竟然发生如此大规模质量问题的召回，而且生产持续时间长达一年！从采购，检验，生产，测试，包装，销售，这么多环节竟没有反馈出问题，处于无人知晓状态，问题出在哪里？希望官方能有一个解释了"
    ]

    ner_args = EeArgs('ner', use_lexicon=True)
    obj_args = EeArgs('obj')
    predict_tool = Predictor(ner_args, obj_args)

    entities = predict_tool.joint_predict(texts)
    exit(0)
    event_types = []
    print("实体：")
    for k, v in entities.items():
        if len(v) != 0:
            print(k, v)
            event_types.append(k)
    for event_type in event_types:
        print("事件类型：", event_type)
        subjects = obj_args.label2role[event_type]
        sbj_obj = predict_tool.predict_obj(text, subjects)
        print("实体：", sbj_obj)
        print("="*100)
