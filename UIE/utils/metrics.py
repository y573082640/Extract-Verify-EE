from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

import numpy as np

def longest_common_substring(str1, str2):
    m = len(str1)
    n = len(str2)
    dp = [[0] * n for _ in range(m)]
    max_len = 0
    for i in range(m):
        for j in range(n):
            if str1[i] == str2[j]:
                if i == 0 or j == 0:
                    dp[i][j] = 1
                else:
                    dp[i][j] = dp[i-1][j-1] + 1
                max_len = max(max_len, dp[i][j])
            else:
                dp[i][j] = 0
    # print(str1, str2, max_len)
    return max_len

def lcs(A, B):
    n = len(A)
    m = len(B)
    L = np.zeros((n+1, m+1), dtype=np.int64)
    for i in range(1, n+1):
        for j in range(1, m+1):
            if A[i-1] == B[j-1]:
                L[i, j] = L[i-1, j-1] + 1
            else:
                L[i, j] = max(L[i, j-1], L[i-1, j])
    return L[n, m]


def calculate_metric(predict, gt, text=None):

    tp, fp, fn = 0, 0, 0
    for entity_predict in predict:
        flag = 0
        for entity_gt in gt:
            # pred_entities[_type] = [(start_index,end_index+1)...]
            # 要保证首尾一致吗？
            # 起点=起点，终点=终点
            if entity_predict[0] == entity_gt[0] and entity_predict[1] == entity_gt[1]:
                flag = 1
                tp += 1
                break
        if flag == 0:
            fp += 1

    fn = len(gt) - tp
    return np.array([tp, fp, fn])

def word_level_calculate_metric(predict, gt, text):
    """
    计算 tp fp fn
    字级别匹配F1值 = 2 * 字级别匹配P值 * 字级别匹配R值 / (字级别匹配P值 + 字级别匹配R值)
    字级别匹配P值 = 预测论元和人工标注论元共有字的数量/ 预测论元字数
    字级别匹配R值 = 预测论元和人工标注论元共有字的数量/ 人工标注论元字数
    """
    predict_texts = []
    groudtruth_texts = []
    for entity_predict in predict:
        predict_texts.append(
            "".join(text[entity_predict[0]:entity_predict[1]]).replace("[TGR]", ""))
    for entity_gt in gt:
        groudtruth_texts.append("".join(text[entity_gt[0]:entity_gt[1]]).replace("[TGR]", ""))

    # print('------------------------------')
    # print('predict_texts')
    # print(predict_texts)
    # print('groudtruth_texts')
    # print(groudtruth_texts)
    # print('gt')
    # print(gt)
    # print('text')
    # print(text)
    # print('------------------------------')

    count_predict, count_groundtruth, count_share_in_predict_texts,count_share_in_groudtruth_texts = 0, 0, 0 , 0

    for gt in groudtruth_texts: #人工标注论元字数
        count_groundtruth += len(gt)
    for pre_t in predict_texts: # 预测论元字数
        count_predict += len(pre_t)

    ## 计算最大匹配
    for i,pre_t in enumerate(predict_texts): ### 实际上此时只有一个论元角色
        count_max = 0
        for j,gt in enumerate(groudtruth_texts):
            count_tmp =  longest_common_substring(pre_t, gt) # 共有字数
            if count_tmp > count_max:
                count_max = count_tmp
        count_share_in_predict_texts += count_max

    for i,gt in enumerate(groudtruth_texts): ### 实际上此时只有一个论元角色
        count_max = 0
        for j,pre_t in enumerate(predict_texts):
            count_tmp =  longest_common_substring(pre_t, gt) # 共有字数
            if count_tmp > count_max:
                count_max = count_tmp
        count_share_in_groudtruth_texts += count_max
    ## 根据最大匹配

    return np.array([count_predict, count_groundtruth, count_share_in_predict_texts , count_share_in_groudtruth_texts])


def get_argu_p_r_f(count_predict, count_groundtruth, count_share_in_predict_texts , count_share_in_groudtruth_texts):
    p = count_share_in_predict_texts/count_predict if count_predict != 0 else 0
    r = count_share_in_groudtruth_texts/count_groundtruth if count_groundtruth != 0 else 0
    f1 = 2.0*p*r/(p+r) if p + r != 0 else 0
    return np.array([p, r, f1])


def get_p_r_f(tp, fp, fn):
    # print(tp, fp, fn)
    p = tp / (tp + fp) if tp + fp != 0 else 0
    r = tp / (tp + fn) if tp + fn != 0 else 0
    f1 = 2 * p * r / (p + r) if p + r != 0 else 0
    return np.array([p, r, f1])


def classification_report(metrics_matrix, label_list, id2label, total_count, digits=2, metrics_type='obj'):
    name_width = max([len(label) for label in label_list])
    last_line_heading = 'micro-f1'
    width = max(name_width, len(last_line_heading), digits)

    headers = ["precision", "recall", "f1-score", "support"]
    head_fmt = u'{:>{width}s} ' + u' {:>9}' * len(headers)
    report = head_fmt.format(u'', *headers, width=width)
    report += u'\n\n'

    row_fmt = u'{:>{width}s} ' + u' {:>9.{digits}f}' * 3 + u' {:>9}\n'

    ps, rs, f1s, s = [], [], [], []
    for label_id, label_matrix in enumerate(metrics_matrix):
        type_name = id2label[label_id]
        if metrics_type == 'ner':
            p, r, f1 = get_p_r_f(label_matrix[0], label_matrix[1], label_matrix[2])
        elif metrics_type == 'obj':
            p, r, f1 = get_argu_p_r_f(label_matrix[0], label_matrix[1], label_matrix[2], label_matrix[3])
        else:
            raise AttributeError("【classification_report】metric类型只能为obj或者ner")            
        nb_true = total_count[label_id]
        report += row_fmt.format(*[type_name, p, r,
                                 f1, nb_true], width=width, digits=digits)
        ps.append(p)
        rs.append(r)
        f1s.append(f1)
        s.append(nb_true)

    report += u'\n'
    mirco_metrics = np.sum(metrics_matrix, axis=0)
    if metrics_type == 'ner':
        mirco_metrics = get_p_r_f(
            mirco_metrics[0], mirco_metrics[1], mirco_metrics[2])
    elif metrics_type == 'obj':
        mirco_metrics = get_argu_p_r_f(
            mirco_metrics[0], mirco_metrics[1], mirco_metrics[2], mirco_metrics[3])   
    else:
        raise AttributeError("【classification_report】metric类型只能为obj或者ner")     
    # compute averages
    report += row_fmt.format(last_line_heading,
                             mirco_metrics[0],
                             mirco_metrics[1],
                             mirco_metrics[2],
                             np.sum(s),
                             width=width, digits=digits)

    return report
