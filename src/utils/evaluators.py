import numpy as np
from sklearn.metrics import f1_score

import chainer
import chainer.functions as chaFunc


def softmax_cross_entropy_flatten(y, t):
    t = t.flatten()
    ignore_label = -1
    loss = chaFunc.softmax_cross_entropy(y, t,
                                         enable_double_backprop=True,
                                         ignore_label=ignore_label)
    return loss


def accuracy_flatten(y, t):
    t = t.flatten()
    ignore_label = -1
    acc = chaFunc.accuracy(y, t, ignore_label=ignore_label)
    return acc


def classification_summary_flatten(y, t):
    t = t.flatten()
    summary = chaFunc.classification_summary(y, t)
    return summary


def fscore_binary(ys, ts, max_n_spans):
    t_all = []
    y_all = []

    ts = chainer.cuda.to_cpu(ts)
    ys = chainer.cuda.to_cpu(ys.data)
    ys = chaFunc.argmax(ys, axis=-1)
    mask = ts > -1
    t_len = np.sum(mask, axis=-1).astype(np.int32)
    t_section = np.cumsum(t_len[:-1])
    ts = ts[mask]
    ys = ys[mask.flatten()]
    ts = chaFunc.split_axis(ts, t_section, 0)
    ys = chaFunc.split_axis(ys, t_section, 0)

    for y, t in zip(ys, ts):

        n = len(t)

        t_root_row = np.where(t.data == max_n_spans)

        t.data[t.data == max_n_spans] = 0

        eye = np.identity(n).astype(np.int32)
        t = eye[t.data]

        t[t_root_row] = np.zeros(n)

        t[np.identity(n).astype(np.bool)] = -1
        t = t[t > -1]

        t = t.flatten().tolist()
        t_all.extend(t)

        y_root_row = np.where(y.data == max_n_spans)
        y.data[y_root_row] = 0
        y = eye[y.data]
        y[y_root_row] = np.zeros(n)
        y[np.identity(n).astype(np.bool)] = -1
        y = y[y > -1]
        y = y.flatten().tolist()
        y_all.extend(y)

    f1_link = f1_score(y_all, t_all, pos_label=1)
    f1_no_link = f1_score(y_all, t_all, pos_label=0)
    return f1_link, f1_no_link


def count_prediction(y, i):
    y = chaFunc.argmax(y, axis=1)
    return len(y[y.data == i])
