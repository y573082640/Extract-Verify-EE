import torch
from transformers import BertTokenizer
from utils.lexicon_functions import build_alphabet, build_gaz_alphabet, build_gaz_file, build_gaz_pretrain_emb, build_biword_pretrain_emb, build_word_pretrain_emb
from utils.alphabet import Alphabet
from utils.gazetteer import Gazetteer
from utils.logging import logger_init
from utils.question_maker import Sim_scorer
import logging
import os
import numpy as np
import json
gaz_dict = {
    50: "./data/embs/ctb.50d.vec",
    100: "./data/embs/tencent-d100/tencent-ailab-embedding-zh-d100-v0.2.0-s.txt",
    200: "./data/embs/tencent-d200/tencent-ailab-embedding-zh-d200-v0.2.0-s.txt"
}

bert_dir = "model_hub/chinese-bert-wwm-ext/"
tokenizer = BertTokenizer.from_pretrained(
    bert_dir, add_special_tokens=True, do_lower_case=False)

text = '又一起 sptgr 坠机事故 sptgr 发生，美国居民区燃起熊熊大火，机上人员已经丧生。[SEP]问题：前文包含坠机事件发生的时间,包含年、月、日、天、周、时、分、秒等吗？答案： unused4 unused5 不unused6 unused7 。'
encode_dict = tokenizer.encode(text)
decode_dict = tokenizer.decode(encode_dict)
print(decode_dict)
print(encode_dict)
