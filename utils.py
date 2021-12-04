import os
import numpy as np
import torch
import logging
from torch.autograd import Variable
PAD = np.random.uniform(-1 / 300, 1 / 300, 300)


def get_glove(filepath):
    glove = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            s = line.strip()
            word = s[:s.find(' ')]
            vec = np.array(s[s.find(' ')+1:].split(), dtype=float)
            if '\xa0' in str(word):
                word = word.replace('\xa0', '-')
            word_ = word.lower()
            glove[word_] = vec
    glove['<unk>'] = PAD
    return glove


def save_model(self, dir, idx):
    os.mkdir(dir) if not os.path.isdir(dir) else None
    torch.save(self, '%s/%smodel.pkl' % (dir, idx))


def get_feature(sent_dict, glove):
    # sent_dict是一个列表，列表元素是字典
    features = []
    for dic in sent_dict:
        text = dic['text']
        for word in text:
            print(glove[word].shape)
        feature = np.concatenate([glove[word] for word in text], axis=0)
        print(len(feature))
        features.append(feature)
    print(len(features))
    return features


def label2idx(str):
    labels = ['O', 'B-source', 'I-source', 'B-cue', 'I-cue', 'B-content', 'I-content']
    for idx, label in enumerate(labels):
        if str == label:
            return [float(idx)]


def idx2label(idx):
    labels = ['O', 'B-source', 'I-source', 'B-cue', 'I-cue', 'B-content', 'I-content']
    return labels[idx.data.item()]


def bio(pred, nums):
    print('-- BIO --')
    f1 = open('test0.txt', 'r', encoding='utf-8')
    f2 = open('final.txt', 'a', encoding='utf-8')

    sums = 0
    for i in nums:
        sums += i
    print(len(pred))
    print(sums)

    lines = f1.readlines()
    for i in range(len(nums)):
        dict = {}
        line = lines[i]
        line = eval(line)
        dict['tokens'] = line['tokens']
        dict['labels'] = []
        for j in range(nums[i]):
            idx = pred.pop(0)
            dict['labels'].append(idx2label(idx))
        f2.write(str(dict) + '\n')


class Logger(object):
    def __init__(self, log_file_name):
        super(Logger, self).__init__()
        self.__logger = logging.getLogger('my_logger')
        self.__logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(log_file_name)  # 存文件
        # console_handler = logging.StreamHandler()  # 控制台

        formatter = logging.Formatter("[%(asctime)s]:%(message)s")
        file_handler.setFormatter(formatter)
        # console_handler.setFormatter(formatter)
        self.__logger.addHandler(file_handler)
        # self.__logger.addHandler(console_handler)

    def get_log(self):
        return self.__logger


def getVariable(ori_data, inference):
    va_data = []
    for data in ori_data:
        input, label = data
        va_input = Variable(torch.from_numpy(input), requires_grad=True if not inference else False)
        va_label = Variable(label)
        va_data.append((va_input, va_label))
    assert len(ori_data) == len(va_data)
    return va_data


def draw():
    pass
