import torch
import time
import argparse
from model import basicModel
import torch.optim as optim
import torch.nn as nn
from utils import get_glove, Logger, getVariable, draw
from datamanager import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import random


parser = argparse.ArgumentParser()
parser.add_argument('--word_dim', type=int, default=300)
parser.add_argument('--hidden_dim1', type=int, default=512)
parser.add_argument('--hidden_dim2', type=int, default=256)
parser.add_argument('--learning_rate', type=float, default=1e-5)
parser.add_argument('--glove_file', type=str, default='/data/home/wangyuxiao/wyx_model/data/glove.840B.300d.txt')
parser.add_argument('--train', type=str, default='/data/home/wangyuxiao/wyx_model/data/train0.txt')
parser.add_argument('--val', type=str, default='/data/home/wangyuxiao/wyx_model/data/valid0.txt')
parser.add_argument('--test', type=str, default='/data/home/wangyuxiao/wyx_model/data/test0.txt')
parser.add_argument('--window_size', type=int, default=3, choices=[3, 5, 9])
parser.add_argument('--EPOCH_MAX', type=int, default=6000)
parser.add_argument('--optim', type=str, default="Adam", choices=['SGD', 'Adam', 'Adadelta'])
parser.add_argument('--per_checkpoint', type=int, default=10)
parser.add_argument('--name_model', type=str, default='wyx_model')
parser.add_argument('--log_path', type=str, default='./log')
FLAGS = parser.parse_args()
print(FLAGS)


def train(train_data, inference):
    train_data = getVariable(train_data, inference)
    model.eval() if inference else model.train()
    running_loss = 0.0
    labels = []
    pred = []

    if inference:
        selected_data = [random.choice(train_data) for i in range(200)]

    for idx, data in enumerate(train_data if not inference else selected_data):
        print(len(data))
        inputs, target = data
        labels.append(target)

        if not inference:
            optimizer.zero_grad()

        outputs = model(inputs)
        outputs = outputs.reshape([1, 7])
        loss = criterion(outputs, target)

        if not inference:
            loss.backward()
            optimizer.step()

        running_loss += loss.data.item()

        pred_ = torch.argmax(outputs).data.item()
        pred.append(pred_)

    return running_loss/len(train_data), labels, pred


def evaluate(data):
    model.eval()
    loss = 0.0
    st, times = 0, 0
    while st < len(data):
        with torch.no_grad():
            out_loss, labels, pred_y = train(data, inference=True)
        loss += out_loss
        st += 1
        times += 1
    acc = accuracy_score(labels, pred_y)
    pre = precision_score(labels, pred_y, average='micro')
    rec = recall_score(labels, pred_y, average='micro')
    f1_micro = f1_score(labels, pred_y, average='micro')
    f1_macro = f1_score(labels, pred_y, average='macro')
    c_m = confusion_matrix(labels, pred_y)
    loss /= times

    return loss, acc, pre, rec, f1_micro, f1_macro, c_m


if __name__ == '__main__':
    dataset_name = ['train', 'val', 'test']
    data = {}
    data['train'], _ = preprocessing(FLAGS.train, get_glove(FLAGS.glove_file), window_size=FLAGS.window_size)
    data['val'], _ = preprocessing(FLAGS.val, get_glove(FLAGS.glove_file), window_size=FLAGS.window_size)
    data['test'], nums = preprocessing(FLAGS.test, get_glove(FLAGS.glove_file), window_size=FLAGS.window_size)

    print('model parameters: %s' % str(FLAGS))
    print('train data: %s, val data: %s, test data: %s' % (len(data['train']), len(data['val']), len(data['test'])))

    model = basicModel(
        FLAGS.window_size*300,
        FLAGS.hidden_dim1,
        FLAGS.hidden_dim2
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), FLAGS.learning_rate)

    loss_step, time_step = 0.0, 1e10
    highest_f1_micro = 0
    still_f1 = 0

    start_time = time.time()
    for step in range(FLAGS.EPOCH_MAX):
        if step % FLAGS.per_checkpoint == 0:
            time_step = time.time() - start_time
            start_time = time.time()
            log_file_name = FLAGS.log_path + '/log%s.txt' % (step / FLAGS.per_checkpoint)
            print("------------------------------------------------------------------")
            print('Time of iter training %.2f s' % time_step)
            print("On iter step %s:, global step %d learning rate %.4f Loss-step %s"
                  % (step / FLAGS.per_checkpoint, step, FLAGS.learning_rate, loss_step))

            logger = Logger(log_file_name).get_log()
            logger.info('Iter Step %s     Global Step %s' % (step / FLAGS.per_checkpoint, step))
            logger.info('Learning rate %.6f' % FLAGS.learning_rate)
            logger.info('Loss-step %s' % loss_step)

            for name in dataset_name:
                loss, acc, pre, rec, f1_micro, f1_macro, c_m = evaluate(data[name])
                print('In dataset %s: Loss=%s, Accuracy=%s, Precision=%s, Recall=%s, F1=%s' % (name, loss, acc, pre, rec, f1_micro))
                if name == 'train' or name == 'test':
                    logger.info('=============%s=============' % name)
                elif name == 'val':
                    logger.info('=============%s=============' % 'validate')
                logger.info('Loss = %s' % loss)
                logger.info('Accuracy = %s' % acc)
                logger.info('Precision = %s' % pre)
                logger.info('Recall = %s' % rec)
                logger.info('F1_micro = %s', f1_micro)
                logger.info('F1_macro = %s', f1_macro)
                logger.info('Confusion_matrix')
                logger.info(c_m)
                logger.info('\n\n\n')

            loss_step = 0.0

            if f1_micro > highest_f1_micro:
                highest_f1 = f1_micro
                # save_model(model, "./model", int(step / FLAGS.per_checkpoint))
                still_f1 = 0
            else:
                still_f1 += 1

        if still_f1 == 30:
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('F1值长时间未更新，终止训练')
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            break

        loss_step += train(data['train'], inference=False)[0] / FLAGS.per_checkpoint
