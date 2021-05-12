# @Time : 2020/12/4 11:01
# @Author: ZWX
# @Email: 935721546@qq.com
# @File : FuncTool.py

import sys
import time

import numpy as np
import scipy.io as scio
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_index(para_k=10, para_num_bags=10):
    """
    Get the training set index and test set index.
    @param
        para_k:
            The number of k-th fold.
    :return
        ret_tr_idx:
            The training set index, and its type is dict.
        ret_te_idx:
            The test set index, and its type is dict.
    """
    temp_rand_idx = np.random.permutation(para_num_bags)
    temp_fold = int(para_num_bags / para_k)
    ret_tr_idx = {}
    ret_te_idx = {}
    for i in range(para_k):
        temp_tr_idx = temp_rand_idx[0: i * temp_fold].tolist()
        temp_tr_idx.extend(temp_rand_idx[(i + 1) * temp_fold:])
        ret_tr_idx[i] = temp_tr_idx
        ret_te_idx[i] = temp_rand_idx[i * temp_fold: (i + 1) * temp_fold].tolist()
    return ret_tr_idx, ret_te_idx


def load_file(para_path):
    """
    Load file.
    @param:
    ------------
        para_file_name: the path of the given file.
    ------------
    @return:
    ------------
        The data.
    ------------
    """
    temp_type = para_path.split('.')[-1]

    if temp_type == 'mat':
        ret_data = scio.loadmat(para_path)
        return ret_data['data']
    else:
        with open(para_path) as temp_fd:
            ret_data = temp_fd.readlines()

        return ret_data


def dis_euclidean(ins1, ins2):
    """
    Calculate the distance between two instances
    :param ins1: the first instance
    :param ins2: the second instance
    :return: the distance between two instances
    """
    dis_instances = np.sqrt(np.sum((ins1 - ins2) ** 2))
    return dis_instances


def get_bar(i, j):
    k = i * 10 + j + 1
    str = '>' * ((j + 10 * i) // 2) + ' ' * ((100 - k) // 2)
    sys.stdout.write('\r' + str + '[%s%%]' % (i * 10 + j + 1))
    sys.stdout.flush()
    time.sleep(0.0001)


def get_matrix(para_bags):
    """

    :param para_bags:
    :return:
    """
    ins_matrix = []
    for i in range(para_bags.shape[0]):
        for ins in para_bags[i, 0][:, :-1]:
            ins_matrix.append(ins)
    # print(ins_matrix)
    return np.array(ins_matrix)


def print_progress_bar(para_idx, para_len):
    """
    Print the progress bar.
    :param
        para_idx:
            The current index.
        para_len:
            The loop length.
    """
    print('\r' + '▇' * int(para_idx // (para_len / 50)) + str(np.ceil((para_idx + 1) * 100 / para_len)) + '%', end='')


def mnist_bag_loader(train, mnist_path=None):
    """"""
    if mnist_path is None:
        mnist_path = "../Data"
    return DataLoader(datasets.MNIST(mnist_path,
                                     train=train,
                                     download=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.1307,), (0.3081,))])),
                      batch_size=1,
                      shuffle=False)
