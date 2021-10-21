# coding=utf-8
import logging

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.ERROR)


def classification(embedding, lbl_path, split_ratio=0.7, loop=100):
    eval_dict = {
        'acc': 0.0,
        'f1-micro': 0.0,
        'f1-macro': 0.0,
    }
    label = pd.read_csv(lbl_path, header=None, sep=' ').values
    for _ in range(loop):
        labels_np = shuffle(label)
        nodes = labels_np[:, 0]
        labels = labels_np[:, 1]

        lb = LabelBinarizer()
        labels = lb.fit_transform(labels)
        train_size = int(labels_np.shape[0] * split_ratio)
        features = embedding[nodes]
        train_x = features[:train_size, :]
        train_y = labels[:train_size, :]
        test_x = features[train_size:, :]
        test_y = labels[train_size:, :]
        clf = OneVsRestClassifier(
            LogisticRegression(class_weight='balanced', solver='liblinear', n_jobs=-1))
        clf.fit(train_x, train_y)
        y_pred = clf.predict_proba(test_x)
        y_pred = lb.transform(np.argmax(y_pred, 1))
        acc = np.sum(np.argmax(y_pred, 1) == np.argmax(test_y, 1)) / len(y_pred)
        eval_dict['acc'] += acc
        eval_dict['f1-micro'] += metrics.f1_score(np.argmax(test_y, 1), np.argmax(y_pred, 1),
                                                  average='micro')
        eval_dict['f1-macro'] += metrics.f1_score(np.argmax(test_y, 1), np.argmax(y_pred, 1),
                                                  average='macro')
    for key in eval_dict.keys():
        eval_dict[key] = round(1.0 * eval_dict[key] / loop, 4)
    print('split_ratio: {}'.format(split_ratio))
    print(eval_dict)
    return eval_dict


def _k_precision(embedding, lbl_path, k, lbl):
    label = pd.read_csv(lbl_path, header=None, sep=' ').values
    nodes = label[np.where(label[:, 1] == lbl)][:, 0]
    acc = 0.0
    for node in nodes:
        distance = {}
        for i in range(embedding.shape[0]):
            if i == node:
                continue
            distance[i] = np.linalg.norm(embedding[i] - embedding[node])
        distance = sorted(distance.items(), key=lambda x: x[1])
        distance = np.array(distance)[:k]
        acc += distance[np.isin(distance[:, 0], nodes)].shape[0] / k
    acc /= len(nodes)
    return acc


def k_precision(embedding, lbl_path, k=50):
    eval_dict = {
        'precision': k,
        'bots_acc': _k_precision(embedding, lbl_path, k, 1),
        'admins_acc': _k_precision(embedding, lbl_path, k, 2)
    }
    print(eval_dict)
