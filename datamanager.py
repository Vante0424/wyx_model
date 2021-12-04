import torch
from utils import get_feature, label2idx


def win3(token_list, label_list):
    tokens = token_list
    labels = label_list
    all_dict = []
    tokens.insert(0, '<unk>')
    tokens.append('<unk>')
    for i in range(1, len(tokens) - 1):
        tmp_dict = {}
        text = [tokens[i - 1], tokens[i], tokens[i + 1]]
        tmp_dict['text'] = text
        tmp_dict['label'] = labels[i - 1]

        all_dict.append(tmp_dict)
    return all_dict


def win5(token_list, label_list):
    tokens = token_list
    labels = label_list
    all_dict = []
    tokens.insert(0, '<unk>')
    tokens.insert(0, '<unk>')
    tokens.append('<unk>')
    tokens.append('<unk>')
    for i in range(2, len(tokens)-2):
        tmp_dict = {}
        text = [tokens[i-2], tokens[i-1], tokens[i], tokens[i+1], tokens[i+2]]
        tmp_dict['text'] = text
        tmp_dict['label'] = labels[i-2]

        all_dict.append(tmp_dict)
    return all_dict


def win9(token_list, label_list):
    tokens = token_list
    labels = label_list
    all_dict = []
    tokens.insert(0, '<unk>')
    tokens.insert(0, '<unk>')
    tokens.insert(0, '<unk>')
    tokens.insert(0, '<unk>')
    tokens.append('<unk>')
    tokens.append('<unk>')
    tokens.append('<unk>')
    tokens.append('<unk>')
    for i in range(4, len(tokens)-4):
        tmp_dict = {}
        text = [tokens[i-4], tokens[i-3], tokens[i-2], tokens[i-1],
                tokens[i], tokens[i+1], tokens[i+2], tokens[i+3], tokens[i+4]]
        tmp_dict['text'] = text
        tmp_dict['label'] = labels[i-4]

        all_dict.append(tmp_dict)
    return all_dict


def preprocessing(filepath, glove_dict, window_size):
    data = []
    nums = []
    with open(filepath, 'r', encoding='utf-8') as f:
        word_dict = []
        total_labels = []
        for idx, line in enumerate(f.readlines()):
            line = eval(line)
            tokens_ = line['tokens']
            labels = line['labels']
            for i, word in enumerate(tokens_):
                if '\xa0' in str(word):
                    tokens_.pop(i)
                    a = word.replace('\xa0', '-')
                    tokens_.insert(i, a)
            tokens = [s.lower() for s in tokens_]
            if window_size == 3:
                sent_dict_seq = win3(tokens, labels)
            elif window_size == 5:
                sent_dict_seq = win5(tokens, labels)
            else:
                sent_dict_seq = win9(tokens, labels)

            nums.append(len(sent_dict_seq))
            for item in sent_dict_seq:
                word_dict.append(item)
            for item in labels:
                total_labels.append(label2idx(item))

        data_x = get_feature(word_dict, glove_dict)
        print(len(data_x))
        data_y = torch.tensor(total_labels)
        print(len(data_y))
        assert len(data_x) == len(data_y)
        for i in range(len(data_x)):
            data.append((data_x[i], data_y[i]))

    return data, nums
