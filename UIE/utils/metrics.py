from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

import numpy as np


def _lcs(A, B):
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
    """
    计算 tp fp fn
    字级别匹配F1值 = 2 * 字级别匹配P值 * 字级别匹配R值 / (字级别匹配P值 + 字级别匹配R值)
    字级别匹配P值 = 预测论元和人工标注论元共有字的数量/ 预测论元字数
    字级别匹配R值 = 预测论元和人工标注论元共有字的数量/ 人工标注论元字数
    """
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
    predict_texts = []
    gt_texts = []
    for entity_predict in predict:
        predict_texts.append(
            "".join(text[entity_predict[0]:entity_predict[1]]))
    for entity_gt in gt:
        gt_texts.append("".join(text[entity_gt[0]:entity_gt[1]]))
    predict_texts = set(predict_texts)
    gt_texts = set(gt_texts)

    for gt in gt_texts: ### 实际上此时只有一个论元角色
        count_pre, count_gt, count_share = 0, 0, 0
        count_gt = len(gt)
        f_max = 0
        rt1, rt2, rt3 = 0, count_gt, 0

        for pre_t in predict_texts:
            count_pre = len(pre_t)
            count_share = max(count_share, _lcs(pre_t, gt))
            p = count_share/count_pre if count_pre != 0 else 0
            r = count_share/count_gt if count_gt != 0 else 0
            f = 2*p*r/(p+r) if p + r != 0 else 0
            if f > f_max:
                f_max = f
                rt1 = count_pre
                rt2 = count_gt
                rt3 = count_share

    return np.array([rt1, rt2, rt3])


def get_argu_p_r_f(count_pre, count_gt, count_share):
    p = count_share/count_pre
    r = count_share/count_gt
    f1 = 2.0*p*r/(p+r)
    return np.array([p, r, f1])


def get_p_r_f(tp, fp, fn):
    print(tp, fp, fn)
    p = tp / (tp + fp) if tp + fp != 0 else 0
    r = tp / (tp + fn) if tp + fn != 0 else 0
    f1 = 2 * p * r / (p + r) if p + r != 0 else 0
    return np.array([p, r, f1])


def classification_report(metrics_matrix, label_list, id2label, total_count, digits=2, suffix=False):
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
        p, r, f1 = get_p_r_f(label_matrix[0], label_matrix[1], label_matrix[2])
        nb_true = total_count[label_id]
        report += row_fmt.format(*[type_name, p, r,
                                 f1, nb_true], width=width, digits=digits)
        ps.append(p)
        rs.append(r)
        f1s.append(f1)
        s.append(nb_true)

    report += u'\n'
    mirco_metrics = np.sum(metrics_matrix, axis=0)
    mirco_metrics = get_p_r_f(
        mirco_metrics[0], mirco_metrics[1], mirco_metrics[2])
    # compute averages
    report += row_fmt.format(last_line_heading,
                             mirco_metrics[0],
                             mirco_metrics[1],
                             mirco_metrics[2],
                             np.sum(s),
                             width=width, digits=digits)

    return report
