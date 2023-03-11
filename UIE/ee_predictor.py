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

from torch.utils.data import DataLoader, RandomSampler
from UIE.ee_data_loader import EeDataset, EeCollate
class Predictor:
    def __init__(self, ner_args=None, obj_args=None):
        self.obj_args = obj_args
        self.ner_args = ner_args
    
        model = UIEModel(ner_args)
        self.ner_pipeline = EePipeline(model, ner_args)
        self.ner_pipeline.load_model()

        # model = UIEModel(obj_args)
        # self.obj_pipeline = EePipeline(model, obj_args)
        # self.obj_pipeline.load_model()

    def predict_ner(self, filepath):
        entities = self.ner_pipeline.predict(filepath)
        return entities

    def predict_obj(self,ner_result):
        pass

    def joint_predict(self, filepath):

        ret = []
        entities = self.predict_ner(filepath)
        for i in entities:
            logging.info(i)
        # {"灾害/意外-坍/垮塌": [('坍塌', 7)], "人生-失联": [('失联', 17)]}
        # for e_type, e_trgs in entities.items():
        #     for trg_tuple in e_trgs:
        #         event = {
        #             "event_type":e_type,
        #             'arguments':[]
        #         }
        #         argus = self.predict_obj(t_text, event_type=e_type, trigger=trg_tuple[0], trigger_start_index=trg_tuple[1])
        #         event['arguments'] = argus
        #         event_list['event_list'].append(event)
        # ret.append(event_list)
        # fp = open('log/output-' + str(time.time())[:10]+ '.json','a+')
        # for r in ret:
        #     json.dump(r,fp,ensure_ascii=False)
        #     fp.write('\n')
        # return 
            

if __name__ == "__main__":
    ner_args = EeArgs('ner', use_lexicon=True,log=True)
    # obj_args = EeArgs('obj', use_lexicon=False,log=False)
    # predict_tool = Predictor(ner_args, obj_args)
    predict_tool = Predictor(ner_args, None)
    t_path = '/home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/data/ee/duee/duee_test2.json'
    predict_tool.joint_predict(t_path)
