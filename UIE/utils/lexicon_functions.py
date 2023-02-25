import json
from utils.alphabet import Alphabet
from utils.gazetteer import Gazetteer
from LAC import LAC
import jieba
import jieba.posseg as pseg
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


def build_gaz_file(gaz_file, gaz, gaz_lower=False):
    if gaz_file:
        fins = open(gaz_file, "r", encoding="utf-8").readlines()
        for f in fins:
            f = f.strip().split()[0]
            if f:
                gaz.insert(f, 'one_source')
        print("Load gaz file: ", gaz_file, " total size:", gaz.size())
    else:
        print("Gaz file is None, load nothing")
    return gaz


def build_gaz_alphabet(input_file, gaz, gaz_alphabet, gaz_alphabet_count, count=False):
    in_lines = open(input_file, 'r', encoding="utf-8").readlines()
    for line in in_lines:
        d = json.loads(line)
        text = d['text']
        text = list(text)
        w_length = len(text)
        entitys = []

        for idx in range(w_length):
            matched_entity = gaz.enumerateMatchList(text[idx:])
            entitys += matched_entity
            for entity in matched_entity:
                gaz_alphabet.add(entity)
                index = gaz_alphabet.get_index(entity)
                gaz_alphabet_count[index] = gaz_alphabet_count.get(
                    index, 0)  # initialize gaz count

        if count:
            entitys.sort(key=lambda x: -len(x))
            while entitys:
                longest = entitys[0]
                longest_index = gaz_alphabet.get_index(longest)
                gaz_alphabet_count[longest_index] = gaz_alphabet_count.get(
                    longest_index, 0) + 1
                gazlen = len(longest)
                for i in range(gazlen):
                    for j in range(i+1, gazlen+1):
                        covering_gaz = longest[i:j]
                        if covering_gaz in entitys:
                            entitys.remove(covering_gaz)

    print("gaz alphabet size:", gaz_alphabet.size())
    return gaz_alphabet, gaz_alphabet_count


def build_alphabet(input_file, word_alphabet, biword_alphabet, pos_alphabet, NULLKEY="-null"):
    in_lines = open(input_file, 'r', encoding="utf-8").readlines()
    for line in in_lines:
        d = json.loads(line)
        text = d['text']
        words_and_flags = pseg.cut(text)  # jieba默认模式
        for w, f in words_and_flags:
            pos_alphabet.add(f)

        text = list(text)
        for idx in range(len(text)):
            w = text[idx]
            # 单词
            word_alphabet.add(w)
            # 双词
            if idx < len(text) - 1:
                biword_alphabet.add(w + text[idx+1])
            else:
                biword_alphabet.add(w + NULLKEY)

    return word_alphabet, biword_alphabet, pos_alphabet


def generate_instance_with_gaz(text, pos_alphabet, word_alphabet,
                               biword_alphabet, gaz_alphabet, gaz_alphabet_count, gaz, max_len,
                               char_padding_size=-1, NULLKEY="-null-", char_padding_symbol='</pad>'):
    """
    Generate word with gaz tag

    :param file_path:路径
    """
    instence_Ids = []  # 训练数据转为ids
    words = []  # 单词
    word_pos = []  # 词性标注
    biwords = []  # 双词
    word_Ids = []  # 单词转为id（词表中）
    biword_Ids = []  # 双词转为id（词表中）
    # pos
    words_and_flags = pseg.cut(text)  # jieba默认模式
    for w, f in words_and_flags:
        f_index = pos_alphabet.get_index(f)
        word_pos += len(w) * [f_index]
    assert len(text) == len(word_pos)
    text = list(text)
    # 引入词、双词信息
    # batch_size = 1
    for idx in range(len(text)):
        w = text[idx]
        words.append(w)
        word_Ids.append(word_alphabet.get_index(w))
        # 双词
        if idx < len(text) - 1:
            biword = w + text[idx+1]
        else:
            biword = w + NULLKEY
        biwords.append(biword)
        biword_Ids.append(biword_alphabet.get_index(biword))

    # 引入BMSE信息
    matched_gaz_Ids = []
    layergazmasks = []
    w_length = len(text)
    # gazs:[c1,c2,...,cn]  ci:[B,M,E,S]  B/M/E/S :[w_id1,w_id2,...]  None:0
    gazs = [[[] for i in range(4)] for _ in range(w_length)]
    gazs_count = [[[] for i in range(4)] for _ in range(w_length)]
    max_gazlist = 0
    # 遍历可能的单词，找到匹配，然后给每个字符引入BMSE信息
    for idx in range(len(text)):
        matched_entitys = gaz.enumerateMatchList(text[idx:])
        matched_Ids = [gaz_alphabet.get_index(
            entity) for entity in matched_entitys]
        matched_lengths = [len(entity) for entity in matched_entitys]

        for entity_idx in range(len(matched_Ids)):
            id = matched_Ids[entity_idx]
            w_len = matched_lengths[entity_idx]

            if w_len == 1:  # Single
                gazs[idx][3].append(id)
                gazs_count[idx][3].append(1)
            else:
                gazs[idx][0].append(id)  # Begin
                gazs_count[idx][0].append(gaz_alphabet_count[id])

                gazs[idx + w_len - 1][2].append(id)  # End
                # End
                gazs_count[idx + w_len - 1][2].append(gaz_alphabet_count[id])

                for l in range(1, w_len-1):
                    gazs[idx + l][1].append(id)  # M
                    # M
                    gazs_count[idx + l][1].append(gaz_alphabet_count[id])

        for label in range(4):  # NULLKEY
            if not gazs[idx][label]:
                gazs[idx][label].append(0)
                gazs_count[idx][label].append(1)

            max_gazlist = max(len(gazs[idx][label]), max_gazlist)  # 最多的gaz数量

        if matched_lengths:  # 从这个idx之后匹配到的所有词
            matched_gaz_Ids.append([matched_Ids, matched_lengths])
        else:
            matched_gaz_Ids.append([])

    for idx in range(len(text)):
        gazmask = []  # 对某个tokend的mask

        for label in range(4):
            label_len = len(gazs[idx][label])  # 这个label中包含的gaz个数
            count_set = set(gazs_count[idx][label])  # 其中每个gaz的出现次数
            if len(count_set) == 1 and 0 in count_set:
                gazs_count[idx][label] = [1]*label_len  # 没有匹配到 gaz
            mask = label_len*[0]
            mask += (max_gazlist-label_len)*[1]
            gazs[idx][label] += (max_gazlist-label_len)*[0]  # 单词级别的的补齐
            gazs_count[idx][label] += (max_gazlist-label_len)*[0]  # 单词级别的的补齐
            gazmask.append(mask)  # 对这个token的所有mask

        layergazmasks.append(gazmask)  # 对整段文本的所有mask

    instence_Ids = [word_Ids, biword_Ids, matched_gaz_Ids,
                    word_pos, gazs, gazs_count, layergazmasks]

    return instence_Ids


def batchify_augment_ids(input_batch_list, max_len, volatile_flag=False):
    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    biwords = [sent[1] for sent in input_batch_list]
    matched_gaz_Ids = [sent[2] for sent in input_batch_list]
    word_pos = [sent[3] for sent in input_batch_list]
    gazs = [sent[4] for sent in input_batch_list]
    gaz_count = [sent[5] for sent in input_batch_list]
    gaz_mask = [sent[6] for sent in input_batch_list]

    # 文本级别
    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    word_pos_tensor = torch.zeros((batch_size, max_len))
    word_seq_tensor = torch.zeros((batch_size, max_len))
    biword_seq_tensor = torch.zeros((batch_size, max_len))
    mask = torch.zeros((batch_size, max_len))

    # gaz单词级别
    # 第i段文本第0个词的第0个label，文本内的label已经对齐了
    gaz_num = [len(gazs[i][0][0]) for i in range(batch_size)]
    max_gaz_num = max(gaz_num)
    layer_gaz_tensor = torch.zeros(batch_size, max_len, 4, max_gaz_num).long()
    gaz_count_tensor = torch.zeros(batch_size, max_len, 4, max_gaz_num).long()
    gaz_mask_tensor = torch.zeros(batch_size, max_len, 4, max_gaz_num).long()

    for b, (seq, pos, biseq, seqlen, layergaz, gazmask, gazcount, gaznum) \
            in enumerate(zip(words, word_pos, biwords, word_seq_lengths, gazs, gaz_mask, gaz_count, gaz_num)):
        assert len(pos) == len(seq)
        word_pos_tensor[b, :seqlen] = torch.LongTensor(pos)
        word_seq_tensor[b, :seqlen] = torch.LongTensor(seq)
        biword_seq_tensor[b, :seqlen] = torch.LongTensor(biseq)

        mask[b, :seqlen] = torch.LongTensor([1]*int(seqlen))
        layer_gaz_tensor[b, :seqlen, :, :gaznum] = torch.LongTensor(layergaz)
        gaz_count_tensor[b, :seqlen, :, :gaznum] = torch.LongTensor(gazcount)
        gaz_count_tensor[:, seqlen:] = 1
        gaz_mask_tensor[b, :seqlen, :, :gaznum] = torch.LongTensor(gazmask)

    word_pos_tensor.requires_grad_(True)
    word_seq_tensor.requires_grad_(True)
    biword_seq_tensor.requires_grad_(True)
    mask.requires_grad_(True)

    return gazs, word_seq_tensor, biword_seq_tensor, word_pos_tensor, word_seq_lengths, layer_gaz_tensor, gaz_count_tensor, gaz_mask_tensor, mask


def build_pretrain_embedding(embedding_path, word_alphabet, embedd_dim=100, norm=True):
    embedd_dict = {}
    if embedding_path != None:
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)
    scale = np.sqrt(3.0/embedd_dim)
    pretrain_emb = np.empty([word_alphabet.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    # -NULLKEY-
    pretrain_emb[0, :] = np.random.uniform(-scale, scale, [1, embedd_dim])
    for word, index in word_alphabet.instance2index.items():
        if word in embedd_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embedd_dict[word])
            else:
                pretrain_emb[index, :] = embedd_dict[word]
            perfect_match += 1
        elif word.lower() in embedd_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embedd_dict[word.lower()])
            else:
                pretrain_emb[index, :] = embedd_dict[word.lower()]
            case_match += 1
        else:
            pretrain_emb[index,
                         :] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1
    pretrained_size = len(embedd_dict)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s" % (
        pretrained_size, perfect_match, case_match, not_match, (not_match+0.)/word_alphabet.size()))
    return pretrain_emb, embedd_dim


def load_pretrain_emb(embedding_path):
    embedd_dim = -1
    embedd_dict = {}
    with open(embedding_path, "r", encoding="utf-8") as file_object:
        for line in file_object:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            embedd_dim = len(tokens) - 1
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            embedd_dict[tokens[0]] = embedd

    return embedd_dict, embedd_dim


def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec/root_sum_square


def build_word_pretrain_emb(file_path, word_alphabet, word_emb_dim, norm_word_emb):
    print("build word pretrain emb...")
    return build_pretrain_embedding(file_path, word_alphabet, word_emb_dim, norm_word_emb)


def build_biword_pretrain_emb(file_path, biword_alphabet, biword_emb_dim, norm_biword_emb):
    print("build biword pretrain emb...")
    return build_pretrain_embedding(file_path, biword_alphabet, biword_emb_dim, norm_biword_emb)


def build_gaz_pretrain_emb(file_path, gaz_alphabet, gaz_emb_dim, norm_gaz_emb):
    print("build gaz pretrain emb...")
    return build_pretrain_embedding(file_path, gaz_alphabet, gaz_emb_dim, norm_gaz_emb)
