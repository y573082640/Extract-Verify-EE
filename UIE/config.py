import torch
from transformers import BertTokenizer
from utils.lexicon_functions import (
    build_alphabet,
    build_gaz_alphabet,
    build_gaz_file,
    build_gaz_pretrain_emb,
    build_biword_pretrain_emb,
    build_word_pretrain_emb,
)
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
    200: "./data/embs/tencent-d200/tencent-ailab-embedding-zh-d200-v0.2.0-s.txt",
}

model_dict = {
    "bert": "model_hub/chinese-bert-wwm-ext/",
    "roberta": "model_hub/chinese-roberta-wwm-ext/",
    "macbert": "model_hub/chinese-macbert-base/",
    "mlmbert": "checkpoints/ee/mlm_label",
}
aug_modes = [None, "merge", "demo"]


def get_tokenizer(arg):
    tokenizer = BertTokenizer.from_pretrained(arg.bert_dir)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["[TGR]", "[DEMO]", "[ARG]"]}
    )
    return tokenizer


class EeArgs:
    def __init__(
        self,
        task,
        use_lexicon=False,
        gaz_dim=50,
        log=True,
        model="bert",
        output_name=None,
        aug_mode=None,
        weight_path=None,
        add_trigger=False
    ):
        self.task = task
        self.data_name = "duee"
        self.data_dir = "ee"
        self.add_trigger = add_trigger ## 是否使用触发词信息
        if weight_path is not None:
            self.save_dir = weight_path
            if "roberta" in weight_path:
                self.bert_dir = model_dict["roberta"]
            elif "mlmbert" in weight_path:
                self.bert_dir = model_dict["mlmbert"]
            elif "macbert" in weight_path:
                self.bert_dir = model_dict["macbert"]
            else:
                self.bert_dir = model_dict["bert"]
            print("[推理模式] 加载模型{}".format(self.bert_dir))

        else:
            if model in model_dict:
                self.bert_dir = model_dict[model]
            else:
                self.bert_dir = model_dict["bert"]
                print("[警告] 模型{}不存在,自动加载chinese-bert-wwm-ext".format(model))

            if output_name is None:
                self.save_dir = "./checkpoints/{}/{}_{}_{}_{}_default.pt".format(
                    self.data_dir, self.task, self.data_name, model , aug_mode, 
                )
            else:
                self.save_dir = "./checkpoints/{}/{}_{}_{}_{}_{}.pt".format(
                    self.data_dir, self.task, self.data_name, model, aug_mode, output_name
                )

        self.train_path = "./data/{}/{}/duee_train.json".format(
            self.data_dir, self.data_name
        )
        self.dev_path = "./data/{}/{}/duee_dev.json".format(
            self.data_dir, self.data_name
        )
        self.test_path = "./data/{}/{}/duee_dev.json".format(
            self.data_dir, self.data_name
        )
        self.infer_path = "./data/{}/{}/duee_dev.json".format(
            self.data_dir, self.data_name
        )
        self.label_path = "./data/{}/{}/labels.txt".format(
            self.data_dir, self.data_name
        )
        self.demo_path = "./data/{}/{}/duee_train.json".format(
            self.data_dir, self.data_name
        )
        self.sim_model = "model_hub/paraphrase-MiniLM-L6-v2"
        self.ignore_key = [
            "argu_roles",
            "raw_tokens",
            "argu_tuples",
            "batch_augment_Ids",
            "text_ids",
            "event_types",
            "text_bias"
        ]
        self.replace_set_path = "./data/{}/{}/replace_set.json".format(
            self.data_dir, self.data_name
        )
        self.argument_label_dict_path = "./data/{}/{}/argument_label_dict.json".format(
            self.data_dir, self.data_name
        )
        with open(self.replace_set_path, "r") as fp:
            self.replace_set = json.load(fp)
        with open(self.argument_label_dict_path, "r") as fp:
            self.argument_label_dict = json.load(fp)
        with open(self.label_path, "r") as fp:
            self.entity_label = fp.read().strip().split("\n")
        self.ent_label2id = {}
        self.ent_id2label = {}

        for i, label in enumerate(self.entity_label):
            self.ent_label2id[label] = i
            self.ent_id2label[i] = label
        self.ner_num_labels = len(self.entity_label)
        self.train_epoch = 20
        self.train_batch_size = 32
        self.eval_batch_size = 32
        self.eval_step = 300
        self.max_seq_len = 512
        self.weight_decay = 0.01
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 5.0
        self.lr = 3e-5
        self.other_lr = 3e-4
        self.warmup_proportion = 0.01
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")
        # 下面是lexicon的
        # 完成alphabet构建
        self.use_lexicon = use_lexicon
        self.use_count = True
        self.gaz_lower = False
        self.word_emb_dim = 50
        self.biword_emb_dim = 50
        self.gaz_emb_dim = 50
        self.pos_emb_dim = 24
        self.gaz_dropout = 0.5
        self.norm_word_emb = True
        self.norm_biword_emb = True
        self.norm_gaz_emb = True
        self.char_emb = "./data/embs/gigaword_chn.all.a2b.uni.ite50.vec"
        self.bichar_emb = "./data/embs/gigaword_chn.all.a2b.bi.ite50.vec"
        gaz_dim = gaz_dim if gaz_dim in [50, 100, 200] else 50
        self.gaz_file = gaz_dict.get(gaz_dim, 50)
        self.aug_mode = aug_mode

        if log:
            self.logs_save_dir = "log"
            logger_init(
                "ee-" + self.task,
                log_level=logging.DEBUG,
                log_dir=self.logs_save_dir,
                only_file=False,
            )
            logging.info("\n\n\n\n\n########  <----------------------->")
            for key, value in self.__dict__.items():
                if key == 'replace_set':
                    continue
                logging.info(f"########  {key} = {value}")

        if self.use_lexicon:
            self.word_alphabet = Alphabet("word")  # 单字
            self.biword_alphabet = Alphabet("biword")  # 双字
            self.pos_alphabet = Alphabet("pos")  # 双字
            self.gaz = Gazetteer(self.gaz_lower)
            self.gaz_alphabet = Alphabet("gaz")
            self.gaz_alphabet_count = {}
            self.gaz_alphabet_count[1] = 0
            self.init_lexicon()

        if self.task == "obj":
            label2role_path = "./data/ee/{}/label2role.json".format(self.data_name)
            with open(label2role_path, "r", encoding="utf-8") as fp:
                self.label2role = json.load(fp)

        # 相似度计算，用于辅助训练和文本增强
        # logging.info("...加载相似度匹配模型:" + self.sim_model)
        # self.sim_scorer = Sim_scorer(self.sim_model)
        # logging.info("...构造提示库embedding")
        # self.demo_embs, self.demo_tuples = self.sim_scorer.create_embs_and_tuples(
        #     self.demo_path, self.task
        # )
        # logging.info("...提示库embedding构造完毕")
        # logging.info("【增强模式】" + str(aug_mode))

    def init_lexicon(self):
        build_gaz_file(self.gaz_file, self.gaz)
        for filename in [
            self.train_path,
            self.dev_path,
            self.test_path,
            self.infer_path,
        ]:
            build_alphabet(
                filename, self.word_alphabet, self.biword_alphabet, self.pos_alphabet
            )
            build_gaz_alphabet(
                filename,
                self.gaz,
                self.gaz_alphabet,
                self.gaz_alphabet_count,
                count=self.use_count,
            )

        # 停止自动增长
        self.word_alphabet.keep_growing = False
        self.biword_alphabet.keep_growing = False
        self.pos_alphabet.keep_growing = False
        self.gaz_alphabet.keep_growing = False
        print("-词典和字母表构建完毕-")

        self.pretrain_gaz_embedding, self.gaz_emb_dim = build_gaz_pretrain_emb(
            self.gaz_file, self.gaz_alphabet, self.gaz_emb_dim, self.norm_gaz_emb
        )
        self.pretrain_word_embedding, self.word_emb_dim = build_word_pretrain_emb(
            self.char_emb, self.word_alphabet, self.word_emb_dim, self.norm_word_emb
        )
        self.pretrain_biword_embedding, self.biword_emb_dim = build_biword_pretrain_emb(
            self.bichar_emb,
            self.biword_alphabet,
            self.biword_emb_dim,
            self.norm_biword_emb,
        )

        # if os.path.exists('./storage/word_embedding.npy'):
        #     self.pretrain_word_embedding = np.load(
        #         'storage/word_embedding.npy')
        # else:
        #     self.pretrain_word_embedding, self.word_emb_dim = build_word_pretrain_emb(
        #         self.char_emb, self.word_alphabet, self.word_emb_dim, self.norm_word_emb)
        #     np.save('storage/word_embedding.npy', self.pretrain_word_embedding)

        # if os.path.exists('./storage/biword_embedding.npy'):
        #     self.pretrain_biword_embedding = np.load(
        #         'storage/biword_embedding.npy')
        # else:
        #     self.pretrain_biword_embedding, self.biword_emb_dim = build_biword_pretrain_emb(
        #         self.bichar_emb, self.biword_alphabet, self.biword_emb_dim, self.norm_biword_emb)
        #     np.save('storage/biword_embedding.npy',
        #             self.pretrain_biword_embedding)

        # if os.path.exists('./storage/gaz_embedding.npy'):
        #     self.pretrain_gaz_embedding = np.load('storage/gaz_embedding.npy')
        # else:
        #     self.pretrain_gaz_embedding, self.gaz_emb_dim = build_gaz_pretrain_emb(
        #         self.gaz_file, self.gaz_alphabet, self.gaz_emb_dim, self.norm_gaz_emb)
        #     np.save('storage/gaz_embedding.npy', self.pretrain_gaz_embedding)

        self.hidden_dim = self.word_emb_dim + self.biword_emb_dim + 4 * self.gaz_emb_dim
        print("-预训练向量加载完毕-维数为：" + str(self.hidden_dim))


class NerArgs:
    tasks = ["ner"]
    data_name = "cner"
    data_dir = "ner"
    bert_dir = "model_hub/chinese-bert-wwm-ext/"
    save_dir = "./checkpoints/{}/{}_{}_model.pt".format(data_dir, tasks, data_name)
    train_path = "./data/{}/{}/train.json".format(data_dir, data_name)
    dev_path = "./data/{}/{}/dev.json".format(data_dir, data_name)
    test_path = "./data/{}/{}/test.json".format(data_dir, data_name)
    label_path = "./data/{}/{}/labels.txt".format(data_dir, data_name)
    with open(label_path, "r") as fp:
        labels = fp.read().strip().split("\n")
    label2id = {}
    id2label = {}
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label
    ner_num_labels = len(labels)
    train_epoch = 20
    train_batch_size = 32
    eval_batch_size = 32
    eval_step = 100
    max_seq_len = 150
    weight_decay = 0.01
    adam_epsilon = 1e-8
    max_grad_norm = 5.0
    lr = 3e-5
    other_lr = 3e-4
    warmup_proportion = 0.01
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(bert_dir)


class ReArgs:
    tasks = ["obj"]
    bert_dir = "model_hub/chinese-bert-wwm-ext/"
    data_name = "ske"
    save_dir = "./checkpoints/re/{}_{}_model.pt".format(tasks, data_name)
    train_path = "./data/re/{}/train.json".format(data_name)
    dev_path = "./data/re/{}/dev.json".format(data_name)
    test_path = "./data/re/{}/dev.json".format(data_name)
    relation_label_path = "./data/re/{}/relation_labels.txt".format(data_name)

    entity_label_path = "data/re/{}/entity_labels.txt".format(data_name)
    with open(entity_label_path, "r", encoding="utf-8") as fp:
        entity_label = fp.read().strip().split("\n")
    ner_num_labels = len(entity_label)
    ent_label2id = {}
    ent_id2label = {}
    for i, label in enumerate(entity_label):
        ent_label2id[label] = i
        ent_id2label[i] = label

    with open(relation_label_path, "r", encoding="utf-8") as fp:
        relation_label = fp.read().strip().split("\n")
    relation_label.append("没有关系")

    rel_label2id = {}
    rel_id2label = {}
    for i, label in enumerate(relation_label):
        rel_label2id[label] = i
        rel_id2label[i] = label

    re_num_labels = len(relation_label)
    train_epoch = 3
    train_batch_size = 8
    eval_batch_size = 8
    eval_step = 100
    max_seq_len = 256
    weight_decay = 0.01
    adam_epsilon = 1e-8
    max_grad_norm = 5.0
    lr = 3e-5
    other_lr = 3e-4
    warmup_proportion = 0.01
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(bert_dir)
