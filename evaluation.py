# coding: utf-8
#
# Copyright 2021 Yequan Wang
# Author: Yequan Wang (tshwangyequan@gmail.com)
#
# model evaluation

from __future__ import unicode_literals, print_function, division

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from typing import List
import ast


def get_element(labels: List[str]):
    dict_main = {}

    def reset(flag_find: bool = False, tmp_label: str = '', tmp_idx_start: int = -1):
        return flag_find, tmp_label, tmp_idx_start

    flag_find, tmp_label, tmp_idx_start = reset()

    for idx, label in enumerate(labels):
        if label == '<IGN>':
            if flag_find:
                dict_main[tmp_label].append([tmp_idx_start, idx])
            break
        if label.startswith('B-'):
            if flag_find:
                dict_main[tmp_label].append([tmp_idx_start, idx])

            tmp_label = label[2:]
            if tmp_label not in dict_main:
                dict_main[tmp_label] = []
            flag_find = True
            tmp_idx_start = idx
        elif label.startswith('I-'):
            if flag_find and (label[2:] == tmp_label):
                continue
            else:
                # avoid the pattern such as, 'B-source', 'I-content'
                if flag_find:
                    dict_main[tmp_label].append([tmp_idx_start, idx])

                flag_find, tmp_label, tmp_idx_start = reset()
        else:
            if flag_find:
                dict_main[tmp_label].append([tmp_idx_start, idx])

            flag_find, tmp_label, tmp_idx_start = reset()
    else:
        if flag_find:
            dict_main[tmp_label].append([tmp_idx_start, len(labels)])
    return dict_main


def get_spans(token: List[str], dict_idx: dict, lang: str):
    dict_span = dict()
    for k, v in dict_idx.items():
        if k not in dict_span:
            dict_span[k] = []
        for start, end in v:
            tmp = '' if lang == 'zh' else ' '
            dict_span[k].append(tmp.join(token[start: end]))
    return dict_span


def eva_classifier(list_t, list_p, labels=None, average='binary'):
    c_m = confusion_matrix(list_t, list_p, labels=labels)
    acc = accuracy_score(list_t, list_p)
    rec = recall_score(list_t, list_p, labels=labels, average=average)
    pre = precision_score(list_t, list_p, labels=labels, average=average)
    f1 = f1_score(list_t, list_p, labels=labels, average=average)
    f1_micro = f1_score(list_t, list_p, labels=labels, average='micro')
    f1_macro = f1_score(list_t, list_p, labels=labels, average='macro')

    return {'c_m': c_m, 'acc': acc, 'f1': f1, 'pre': pre, 'rec': rec, 'f1_macro': f1_macro, 'f1_micro': f1_micro}


def evaluate_bio_multiple_classification(y_true, y_pred):
    dict_eva = dict()
    bio_y, bio_p = [], []

    for list_t1, list_t2 in zip(y_true, y_pred):
        for t1, t2 in zip(list_t1, list_t2):
            if t1 == '<IGN>':
                break
            bio_y.append(t1)
            bio_p.append(t2)
    dict_eva['bio'] = eva_classifier(
        bio_y, bio_p, labels=['O', 'B-source', 'I-source', 'B-cue', 'I-cue', 'B-content', 'I-content'], average='macro')
    dict_eva['begin_multiple'] = eva_classifier(
        bio_y, bio_p, labels=['B-source', 'B-cue', 'B-content'], average='macro')
    return dict_eva


def evaluate_extraction(y_true_bio: List[List[str]], y_pred_bio: List[List[str]]):
    dict_y_true = {
        'begin': {'source': [], 'cue': [], 'content': []},
        'exact_match': {'source': [], 'cue': [], 'content': []},
        'jaccard': {'source': [], 'cue': [], 'content': []}
    }
    dict_y_pred = {
        'begin': {'source': [], 'cue': [], 'content': []},
        'exact_match': {'source': [], 'cue': [], 'content': []},
        'jaccard': {'source': [], 'cue': [], 'content': []}
    }
    for inst_true, inst_pred in zip(y_true_bio, y_pred_bio):
        dict_tmp_true = get_element(inst_true)
        dict_tmp_pred = get_element(inst_pred)

        for k in ['source', 'cue', 'content']:
            v_tmp_true = dict_tmp_true[k] if k in dict_tmp_true else []
            v_tmp_pred = dict_tmp_pred[k] if k in dict_tmp_pred else []
            for type_match in ['begin', 'exact_match', 'jaccard']:
                tmp_true, tmp_pred = cal_evaluation_detail(v_tmp_true, v_tmp_pred, type_match)
                dict_y_true[type_match][k].extend(tmp_true)
                dict_y_pred[type_match][k].extend(tmp_pred)

    dict_eva = evaluate_bio_multiple_classification(y_true_bio, y_pred_bio)
    for k1 in ['begin', 'exact_match', 'jaccard']:
        if k1 not in dict_eva:
            dict_eva[k1] = dict()
        for k2 in ['source', 'cue', 'content']:
            if k1 != 'jaccard':
                dict_eva[k1][k2] = eva_classifier(dict_y_true[k1][k2], dict_y_pred[k1][k2], average='binary')
            else:
                dict_eva[k1][k2] = np.mean(np.array(dict_y_true[k1][k2]) * np.array(dict_y_pred[k1][k2]))
    return dict_eva


def cal_evaluation_detail(list_idx_true, list_idx_pred, type_match: str):
    y_exact_match_true, y_exact_match_pred = [], []
    if type_match == 'begin':
        list_idx_true = [tmp[0] for tmp in list_idx_true]
        list_idx_pred = [tmp[0] for tmp in list_idx_pred]
    for tmp in list_idx_true:
        y_exact_match_true.append(1)
        if tmp in list_idx_pred:
            y_exact_match_pred.append(1)
        else:
            if type_match == 'exact_match':
                y_exact_match_pred.append(0)
            elif type_match == 'begin':
                y_exact_match_pred.append(0)
            else:
                y_exact_match_pred.append(
                    max([cal_overlap(tmp, _tmp) for _tmp in list_idx_pred]) if len(list_idx_pred) > 0 else 0)
    for tmp in list_idx_pred:
        if tmp in list_idx_true:
            continue
        else:
            if type_match == 'begin':
                y_exact_match_true.append(0)
                y_exact_match_pred.append(1)
            elif (len(list_idx_true) > 0) and (max([cal_overlap(tmp, _tmp) for _tmp in list_idx_true]) > 0):
                # do nothing, avoid the
                pass
            else:
                y_exact_match_true.append(0)
                y_exact_match_pred.append(1)
    assert len(y_exact_match_true) == len(y_exact_match_pred)
    return y_exact_match_true, y_exact_match_pred


def cal_overlap(a: [int, int], b: [int, int]):
    assert len(a) == len(b) == 2
    assert a[1] > a[0]
    assert b[1] > b[0]
    tmp = min(a[1] - b[0], b[1] - a[0], b[1] - b[0], a[1] - a[0]) / (max(a[1], b[1]) - min(a[0], b[0]))
    return max(0, tmp)


if __name__ == '__main__':
    true_test = []
    pred_test = []

    with open('0train.txt', 'r', encoding='utf-8') as f:
        values = f.readlines()
        values = [item for item in values]
        for i in range(len(values)):
            values_i = eval(values[i])
            true_test.append(values_i['labels'])

    with open('final_highest.txt', 'r', encoding='utf-8') as f:
        values = f.readlines()
        values = [item for item in values]
        for i in range(len(values)):
            values_i = ast.literal_eval(values[i])
            pred_test.append(values_i['labels'])

    # true_test = [["B-source", "I-cue", "B-cue", "I-cue", "I-cue", "B-content", "I-content", "I-content", "O", "O",
    #              "O", "O", "O", "O", "O", "B-source", "B-cue", "I-cue", "I-cue", "B-content", "I-content", "I-content",
    #              "O", "O", "O", "O", "O", "O", "O", "B-content", "O", "O", "O", "O", "O", "O", "O"]]
    # pred_test = [["B-source", "I-source", "B-cue", "I-cue", "I-cue", "B-content", "I-content", "I-content", "O", "O",
    #              "O", "O", "O", "O", "O", "B-source", "B-cue", "I-cue", "I-cue", "B-content", "I-content", "I-content",
    #              "O", "O", "O", "O", "O", "O", "O", "B-content", "O", "O", "O", "O", "O", "O", "O"]]

    # print(true_test)
    # print(pred_test)
    # exit()

    print(evaluate_extraction(true_test, pred_test))
