import sys
sys.path.append("..")

from transformers import BertForMaskedLM, BertTokenizer, pipeline
import tqdm
from datetime import datetime
import numpy as np
import json
import torch
from transformers import BertTokenizer
from UIE.ee_main import EePipeline
from UIE.model import UIEModel
from UIE.config import EeArgs
import logging
from time import time
from torch.utils.data import DataLoader, RandomSampler
from UIE.ee_data_loader import EeDataset, EeCollate
from utils.question_maker import get_question_for_verify
from UIE.ee_postprocessing import remove_duplicates

def chunks(lst, n):
    # Yield successive n-sized chunks from lst.
    for i in tqdm.tqdm(range(0, len(lst), n)):
        yield lst[i:i + n]

def name_with_date(name):
    return 'log/%s %s.json' % (name,datetime.fromtimestamp(int(time())))
    
def map_fn(example , mode='correct'):
    # 在文本中插入sptgr特殊标志
    text = example['text']
    trigger_start_index = example['trigger_start_index']
    trigger = example['trigger']
    text_tokens = [b for b in text]
    tgr1_index = trigger_start_index
    tgr2_index = trigger_start_index + 1 + len(trigger)
    text_tokens.insert(tgr1_index, '[TGR]')
    text_tokens.insert(tgr2_index, '[TGR]')
    if not text_tokens[-1] == '。':
        text_tokens.append("。")

    text_tokens = ''.join(text_tokens)
    # 构建问题和回答
    if mode == 'correct':
        event_type = example['event_type'].split('-')[-1]
        que_str = '前文的{}事件包含的{}是[ARG]{}[ARG]吗？'.format(
            event_type, example['role'], example['argument'])
        que_str = '[SEP]问题：' + que_str + \
                            "答案： unused4 unused5 [MASK] unused6 unused7 。"
    elif mode == 'exist':
        que_str = get_question_for_verify(example['event_type'], example['role'])
        que_str = '[SEP]问题：' + que_str + "答案： unused4 unused5 [MASK] unused6 unused7 。"
    else:
        raise AttributeError('提供的mode应该为exist 或者 correct')
    
    most_len = 512 - 2 - len(que_str) ### [CLS] + [SEP]
    v_tokens = text_tokens[:most_len] + que_str
    return v_tokens

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

    def decode_ner_to_obj(self, ner_result):
        # {"灾害/意外-坍/垮塌": [('坍塌', 7)], "人生-失联": [('失联', 17)]}
        argu_list = []
        for res in ner_result:
            # {'event_dict': {'交往-道歉': [(['致', '歉'], 3)]}, 'text_id': '5aec...'}
            event_dict = res['event_dict']
            text_id = res['text_id']
            text = res['text']
            for e_type, e_trgs in event_dict.items():
                for i, trg_tuple in enumerate(e_trgs):
                    event = {
                        "event_type": e_type,
                        'text': text,
                        'trigger': trg_tuple[0],
                        'trigger_start_index': trg_tuple[1],
                        # 聚合事件论元 和 聚合事件列表
                        'event_id': text_id + "——" + e_type + str(i),
                        'text_id': text_id
                    }
                    argu_list.append(event)

        return argu_list

    def _create_verified_dataset(self, argu_input, obj_result):
        dataset = []
        ret = {}
        # 构建事件
        for e in argu_input:
            if not ret.__contains__(e['event_id']):
                ret[e['event_id']] = e

        for argu in obj_result:
            e = ret[argu['event_id']]
            merged_object = {**e, **argu}
            dataset.append(map_fn(merged_object,mode='exist'))

        return dataset

    def accumulate_answer(self, argu_input, obj_result):
        ret = {}
        # 构建事件
        for e in argu_input:
            if not ret.__contains__(e['text_id']):
                ret[e['text_id']] = {
                    'id': e['text_id'],
                    'event_list': {}
                }
            ret[e['text_id']]['event_list'][e['event_id']] = {
                "event_type": e['event_type'],
                "arguments": []
            }
        # 将论元填入事件
        for argu in obj_result:
            text_id = argu['event_id'].split('——')[0]
            event_id = argu['event_id']
            argu_list = ret[text_id]['event_list'][event_id]['arguments']
            argu_list.append(argu)
        # 转换为评测需要的形式
        output = []
        for text_id, text_events in ret.items():
            tmp = {}
            tmp['id'] = text_id
            tmp['event_list'] = []
            for _, event in text_events['event_list'].items():
                tmp['event_list'].append(event)
            output.append(tmp)
        return output

    def predict_ner(self, filepath):
        entities = self.ner_pipeline.predict(filepath=filepath)
        return entities

    def predict_obj(self, argu_input):
        ret = self.obj_pipeline.predict(data=argu_input)
        return ret

    def verify_result(self, argu_input, obj_result, batch_size=64, model_path="checkpoints/ee/mlm_label"):

        datas = self._create_verified_dataset(argu_input, obj_result)
        # load your model and tokenizer from a local directory or a URL

        model = BertForMaskedLM.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(model_path)

        # create an unmasker object with the model and tokenizer
        unmasker = pipeline("fill-mask", model=model, tokenizer=tokenizer,
                            device=0, top_k=1)

        results = []
        for batch_data in chunks(datas, batch_size):
            result = unmasker(batch_data, targets=['不', '是'])
            # append the result to the list
            results.extend(result)

        ret = []
        for i, result in enumerate(results):
            if result[0]['token_str'] == '是':
                ret.append(obj_result[i])
        return ret

    def joint_predict(self, filepath, output):
        st = time()
        ner_result = self.predict_ner(filepath)
        argu_input = self.decode_ner_to_obj(ner_result)

        torch.cuda.empty_cache()
        obj_result = self.predict_obj(argu_input)
        torch.cuda.empty_cache()

        logging.info('...进行验证过滤')
        verified_result = self.verify_result(argu_input, obj_result)
        torch.cuda.empty_cache()
        logging.info('...转换为评测需要的形式...')
        answer = self.accumulate_answer(argu_input, verified_result)

        ### 暂时保存中间输出用于分析
        with open(name_with_date('ner_result'), 'w') as fp:
            for a in ner_result:
                json.dump(a, fp, ensure_ascii=False, separators=(',', ':'))
                fp.write('\n')
        with open(name_with_date('obj_result'), 'w') as fp:
            for a in obj_result:
                json.dump(a, fp, ensure_ascii=False, separators=(',', ':'))
                fp.write('\n')
        with open(name_with_date('verified_result'), 'w') as fp:
            for a in answer:
                json.dump(a, fp, ensure_ascii=False, separators=(',', ':'))
                fp.write('\n')
        ##

        logging.info('...后处理...')
        answer = remove_duplicates(answer)
        logging.info('...预测结果输入文件:' + str(output))
        with open(output, 'w') as fp:
            for a in answer:
                json.dump(a, fp, ensure_ascii=False, separators=(',', ':'))
                fp.write('\n')
        ed = time()
        logging.info('推理完成，总计用时%.5f秒' % (ed-st))
        return


if __name__ == "__main__":
    ner_args = EeArgs('ner', use_lexicon=False, log=True, weight_path='/home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/checkpoints/ee/ner_duee_roberta_no_lexicon_len256_bs32.pt')
    obj_args = EeArgs('obj', use_demo=False, log=False, weight_path='/home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/checkpoints/ee/obj_duee_roberta_mergedRole_noLexicon_noDemo_allMatch_len512_bs32.pt')
    predict_tool = Predictor(ner_args, obj_args)
    t_path = 'data/ee/duee/duee_test2.json'
    output_path = name_with_date('output')
    predict_tool.joint_predict(t_path, output_path)


