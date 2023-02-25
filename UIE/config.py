import torch
from transformers import BertTokenizer

class EeArgs:
    tasks = ["ner"]
    data_name = "duee"
    data_dir = "ee"
    bert_dir = "model_hub/chinese-bert-wwm-ext/"
    save_dir = "./checkpoints/{}/{}_{}_model_retrival_wo_analogy.pt".format(data_dir, tasks[0], data_name)
    train_path = "./data/{}/{}/duee_train.json".format(data_dir, data_name)
    dev_path = "./data/{}/{}/duee_dev.json".format(data_dir, data_name)
    test_path = "./data/{}/{}/duee_dev.json".format(data_dir, data_name)
    label_path = "./data/{}/{}/labels.txt".format(data_dir, data_name)
    ignore_key = ['raw_tokens']
    with open(label_path, "r") as fp:
        entity_label = fp.read().strip().split("\n")
    ent_label2id = {}
    ent_id2label = {}
    for i, label in enumerate(entity_label):
        ent_label2id[label] = i
        ent_id2label[i] = label
    ner_num_labels = len(entity_label)
    train_epoch = 20
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
    tokenizer = BertTokenizer.from_pretrained(bert_dir)
    tokenizer.add_special_tokens({'additional_special_tokens':['[DEMO]','[ARG]','[TGR]']})
    use_bigram = True
    word_emb_dim = 50
    biword_emb_dim = 50
    char_emb_dim = 30
    gaz_emb_dim = 50
    pos_emb_dim = 24
    gaz_dropout = 0.5
    pretrain_word_embedding = None
    pretrain_biword_embedding = None
    pretrain_gaz_embedding = None
    gaz_embedding = None
    word_embedding = None
    biword_embedding = None
    label_size = 0
    word_alphabet_size = 0
    biword_alphabet_size = 0
    char_alphabet_size = 0
    label_alphabet_size = 0
    hidden_dim=300
    norm_word_emb = True
    norm_biword_emb = True
    norm_gaz_emb = False
    char_emb = "/home/aistudio/data/data190512/gigaword_chn.all.a2b.uni.ite50.vec"
    bichar_emb = "/home/aistudio/data/data190512/gigaword_chn.all.a2b.bi.ite50.vec"
    gaz_file = "/home/aistudio/data/data190512/ctb.50d.vec"
    


class NerArgs:
    tasks = ["ner"]
    data_name = "cner"
    data_dir = "ner"
    bert_dir = "model_hub/chinese-bert-wwm-ext/"
    save_dir = "./checkpoints/{}/{}_{}_model.pt".format(data_dir, tasks[0], data_name)
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

