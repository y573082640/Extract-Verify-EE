import torch
from transformers import BertTokenizer
from utils.lexicon_functions import build_alphabet, build_gaz_alphabet, build_gaz_file, build_gaz_pretrain_emb, build_biword_pretrain_emb, build_word_pretrain_emb
from utils.alphabet import Alphabet
from utils.gazetteer import Gazetteer
import os
import numpy as np

class EeArgs:
    tasks = ["ner"]
    data_name = "duee"
    data_dir = "ee"
    bert_dir = "model_hub/chinese-bert-wwm-ext/"
    save_dir = "./checkpoints/{}/{}_{}_model_retrival_wo_analogy.pt".format(
        data_dir, tasks[0], data_name)
    train_path = "./data/{}/{}/duee_train.json".format(data_dir, data_name)
    dev_path = "./data/{}/{}/duee_dev.json".format(data_dir, data_name)
    test_path = "./data/{}/{}/duee_dev.json".format(data_dir, data_name)
    label_path = "./data/{}/{}/labels.txt".format(data_dir, data_name)
    ignore_key = ['raw_tokens', 'batch_augment_Ids']
    with open(label_path, "r") as fp:
        entity_label = fp.read().strip().split("\n")
    ent_label2id = {}
    ent_id2label = {}
    for i, label in enumerate(entity_label):
        ent_label2id[label] = i
        ent_id2label[i] = label
    ner_num_labels = len(entity_label)
    train_epoch = 40
    train_batch_size = 32
    eval_batch_size = 32
    eval_step = 500
    max_seq_len = 256
    weight_decay = 0.01
    adam_epsilon = 1e-8
    max_grad_norm = 5.0
    lr = 3e-5
    other_lr = 3e-4
    warmup_proportion = 0.01
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    tokenizer = BertTokenizer.from_pretrained(bert_dir)
    tokenizer.add_special_tokens(
        {'additional_special_tokens': ['[DEMO]', '[ARG]', '[TGR]']})
    # 下面是lexicon的
    # 完成alphabet构建
    word_emb_dim = 50
    biword_emb_dim = 50
    gaz_emb_dim = 50
    pos_emb_dim = 24
    hidden_dim = word_emb_dim + biword_emb_dim + 4 * gaz_emb_dim #TODO:加上pos_emb
    gaz_dropout = 0.5
    use_count=True
    norm_word_emb = True
    norm_biword_emb = True
    norm_gaz_emb = False
    char_emb = "./data/embs/gigaword_chn.all.a2b.uni.ite50.vec"
    bichar_emb = "./data/embs/gigaword_chn.all.a2b.bi.ite50.vec"
    gaz_file = "./data/embs/ctb.50d.vec"
    word_alphabet = Alphabet('word')  # 单字
    biword_alphabet = Alphabet('biword')  # 双字
    pos_alphabet = Alphabet('pos')  # 双字
    gaz_lower = False
    gaz = Gazetteer(gaz_lower)
    gaz_alphabet = Alphabet('gaz')
    gaz_alphabet_count = {}
    build_gaz_file(gaz_file, gaz)
    for filename in [train_path, dev_path, test_path]:
        build_alphabet(filename, word_alphabet, biword_alphabet, pos_alphabet)
        build_gaz_alphabet(filename, gaz, gaz_alphabet, gaz_alphabet_count, count=use_count)
    word_alphabet.keep_growing = False
    biword_alphabet.keep_growing = False
    pos_alphabet.keep_growing = False
    gaz_alphabet.keep_growing = False
    print("-词典和字母表构建完毕-")

    if os.path.exists('./storage/word_embedding.npy'):
        pretrain_word_embedding = np.load('storage/word_embedding.npy')
    else:
        pretrain_word_embedding,word_emb_dim = build_word_pretrain_emb(char_emb,word_alphabet,word_emb_dim,norm_word_emb)
        np.save('storage/word_embedding.npy',pretrain_word_embedding)

    if os.path.exists('./storage/biword_embedding.npy'):
        pretrain_biword_embedding = np.load('storage/biword_embedding.npy')
    else:
        pretrain_biword_embedding,biword_emb_dim = build_biword_pretrain_emb(bichar_emb,biword_alphabet,biword_emb_dim,norm_biword_emb)
        np.save('storage/biword_embedding.npy',pretrain_biword_embedding)

    if os.path.exists('./storage/gaz_embedding.npy'):
        pretrain_gaz_embedding = np.load('storage/gaz_embedding.npy')
    else:
        pretrain_gaz_embedding,gaz_emb_dim = build_gaz_pretrain_emb(gaz_file,gaz_alphabet,gaz_emb_dim,norm_gaz_emb)
        np.save('storage/gaz_embedding.npy',pretrain_gaz_embedding)
    
    print("-预训练向量加载完毕-")

class NerArgs:
    tasks = ["ner"]
    data_name = "cner"
    data_dir = "ner"
    bert_dir = "model_hub/chinese-bert-wwm-ext/"
    save_dir = "./checkpoints/{}/{}_{}_model.pt".format(
        data_dir, tasks[0], data_name)
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
    save_dir = "./checkpoints/re/{}_{}_model.pt".format(tasks[0], data_name)
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

    with open(relation_label_path, "r", encoding='utf-8') as fp:
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
