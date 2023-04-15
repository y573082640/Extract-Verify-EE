from transformers import AutoModelForMaskedLM, AutoTokenizer,BertTokenizer, pipeline
from utils.question_maker import get_question_for_argument
import json
import tqdm
from datasets import load_dataset
import torch


def chunks(lst, n):
    # Yield successive n-sized chunks from lst.
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def map_fn(example):
    # 在文本中插入[TGR]特殊标志
    text = example['text']
    trigger_start_index = example['trigger_start_index']
    trigger = example['trigger']
    text_tokens = [b for b in text]
    tgr1_index = trigger_start_index
    tgr2_index = trigger_start_index + 1 + len(trigger)
    text_tokens.insert(tgr1_index, ' [TGR] ')
    text_tokens.insert(tgr2_index, ' [TGR] ')
    if not text_tokens[-1] == '。':
        text_tokens.append("。")

    text_tokens = ''.join(text_tokens)
    # 构建问题和回答
    event_type = example['event_type']
    event_type = event_type.split('-')[-1]
    que_str = '前文的{}事件包含的{}是 [ARG] {} [ARG] 吗？'.format(
        event_type, example['role'], example['argument'])
    v_tokens = text_tokens + '[SEP]问题：' + que_str + \
        "答案： unused4 unused5 [MASK] unused6 unused7 。"
    return v_tokens


def verify_result(datas, batch_size=32, model_path='"checkpoints/ee/mlm_label"'):

    # 设置批量大小为32
    # load your model and tokenizer from a local directory or a URL
    model = AutoModelForMaskedLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # create an unmasker object with the model and tokenizer
    unmasker = pipeline("fill-mask", model=model, tokenizer=tokenizer, top_k=1)

    results = []
    # loop over each line in the file
    for batch in chunks(datas, batch_size):
        # call the unmasker on the batch
        result = unmasker(batch)
        # append the result to the list
        results.extend(result)

    ret = []
    for i, result in enumerate(results):
        if result[0]['token_str'] == '不':
            ret.append(datas[i])
    return ret

# # optionally, write the results to another file
# with open("output/obj_verify_toy.json", "w") as f:
#     # loop over each result in the list
#     cnt = 0
#     total = 0
#     for i,result in enumerate(results):
#         # write the result as a string to the file
#         total += 1
#         if result[0]['score'] < 0.7:
#             cnt += 1
#         result[0]['sequence'] = result[0]['sequence'].replace(" ", "").replace(
#             "unused4unused5", "").replace("unused6unused7", "").replace("[TGR]", "").replace("[ARG]", "")
#         json.dump(result[0], f, ensure_ascii=False, separators=(',', ':'))
#         f.write("\n")
#     f.write("得分是：")
#     f.write(str(cnt/total))
#     f.write("\n")
# from transformers import BertTokenizer

# bert_dir = '/home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/model_hub/chinese-bert-wwm-ext'
# sp_tokens = ['[TGR]','[ARG]','[DEMO]']
# tokenizer = BertTokenizer.from_pretrained(bert_dir,additional_special_tokens=sp_tokens)
# tokenizer.save_pretrained(bert_dir)

model_path="checkpoints/ee/mlm_label"
model = AutoModelForMaskedLM.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)