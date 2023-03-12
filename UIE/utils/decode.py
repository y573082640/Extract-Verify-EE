import numpy as np
from collections import defaultdict


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ner_decode_batch(ner_s_logits, ner_e_logits, raw_tokens, ent_id2label):
    ret = []
    for s_logit, e_logit, text in zip(ner_s_logits, ner_e_logits, raw_tokens):
        predict_entities = ner_decode(s_logit, e_logit, text, ent_id2label)
        ret.append(predict_entities)
    return ret


def obj_decode_batch(s_logits, e_logits, masks, id2label, raw_tokens, event_ids, roles):
    ret = []
    for s_logit, e_logit, mask, text, e_id, role in zip(s_logits, e_logits, masks, raw_tokens, event_ids, roles):
        length = sum(mask)
        pred_entities = bj_decode(s_logit, e_logit, length, id2label)
        values = pred_entities["答案"]
        for v in values:
            start = v[0]
            end = v[1]
            ans = "".join(text[start:end]).replace("[TGR]", "")
            ret.append({
                'role': role,
                'argument': ans,
                'event_id': e_id,
            })
    return ret


def ner_decode(start_logits, end_logits, raw_text, id2label):
    """
    不确定有哪些事件类型，用于实际预测
    """
    predict_entities = defaultdict(list)
    for label_id in range(len(id2label)):
        start_logit = np.where(sigmoid(start_logits[label_id]) > 0.5, 1, 0)
        end_logit = np.where(sigmoid(end_logits[label_id]) > 0.5, 1, 0)
        start_pred = start_logit[1:len(raw_text)+1]
        end_pred = end_logit[1:len(raw_text)+1]
        for i, s_type in enumerate(start_pred):
            if s_type == 0:
                continue
            for j, e_type in enumerate(end_pred[i:]):
                if s_type == e_type:
                    tmp_ent = raw_text[i:i + j + 1]
                    if tmp_ent == '':
                        continue
                    predict_entities[id2label[label_id]].append((tmp_ent, i))
                    break
    # {'事件':[('触发词',index),('触发词',index)]}
    return dict(predict_entities)


def ner_decode2(start_logits, end_logits, length, id2label):
    """
    确定有哪些事件类型，用于训练
    """
    predict_entities = {x: [] for x in list(id2label.values())}
    # predict_entities = defaultdict(list)
    # print(start_pred)
    # print(end_pred)

    for label_id in range(len(id2label)):
        start_logit = np.where(sigmoid(start_logits[label_id]) > 0.5, 1, 0)
        end_logit = np.where(sigmoid(end_logits[label_id]) > 0.5, 1, 0)
        # print(start_logit)
        # print(end_logit)
        # print("="*100)
        start_pred = start_logit
        end_pred = end_logit

        for i, s_type in enumerate(start_pred):
            if s_type == 0:
                continue
            for j, e_type in enumerate(end_pred[i:]):
                if s_type == e_type:
                    predict_entities[id2label[label_id]].append((i, i+j+1))
                    # 找到距离自己最近的结束符号就停止了
                    break
    return predict_entities


def bj_decode(start_logits, end_logits, length, id2label):
    predict_entities = {x: [] for x in list(id2label.values())}
    start_pred = np.where(sigmoid(start_logits) > 0.5, 1, 0)
    end_pred = np.where(sigmoid(end_logits) > 0.5, 1, 0)
    # print(start_pred)
    # print(end_pred)
    for i, s_type in enumerate(start_pred):
        if s_type == 0:
            continue
        for j, e_type in enumerate(end_pred[i:]):
            if s_type == e_type:
                predict_entities[id2label[0]].append((i, i+j+1))
                break

    return predict_entities


def depart_ner_output_batch(output, batch_data, ner_s_logits, ner_e_logits, raw_tokens):
    # (num_labels,batch_size,max_len)
    ner_start_logits = output["ner_output"]["ner_start_logits"]
    ner_end_logits = output["ner_output"]["ner_end_logits"]
    batch_size = batch_data['ner_input_ids'].size(0)

    for i in range(batch_size):
        # 对一条数据上不同label的预测值
        ner_s_logits.append([logit[i, :]
                            for logit in ner_start_logits])
        ner_e_logits.append([logit[i, :] for logit in ner_end_logits])

    raw_tokens += batch_data['raw_tokens']

    return ner_s_logits, ner_e_logits, raw_tokens
