import os
import shutil

import numpy as np

import torch


def make_dirs(path, replace):
    if not os.path.isdir(path):
        os.makedirs(path)
    else:
        if replace:
            shutil.rmtree(path)
            os.makedirs(path)


def calculate_acc(x, y):
    y_hat_class = torch.argmax(x, dim=-1)
    acc = (y_hat_class == y).sum() / y.shape[0]
    return acc


def get_average(a):
    return np.mean(np.asarray(a))


def append(*args):
    for arg in args:
        arg[0].append(arg[1])
