# @Time : 2020/12/30 15:41
# @Author: ZWX
# @Email: 935721546@qq.com
# @File : FvGmm.py
import math

import numpy as np

np.set_printoptions(suppress=True)


class FvGm:

    def __init__(self, para_bag, para_weights, para_mean, para_covariances, para_K, para_gamma):
        """

        :param para_weights:
        :param para_mean:
        :param para_covariances:
        """
        self.bag = para_bag
        self.weights = para_weights
        self.mean = para_mean
        self.covariance = para_covariances
        self.K = para_K
        self.gamma = para_gamma

    def compute_fisher(self):
        """
        :return:
        """
        f_X = []
        # gamma = self.cal_gamma()
        gamma = self.gamma
        # print(np.array(gamma))
        for k in range(0, self.K):
            f_weight = np.zeros(1).astype('float64')
            f_mean_vector = np.zeros(self.bag.shape[1]).astype('float64')
            f_covariance = np.zeros(self.bag.shape[1]).astype('float64')
            for instance in range(0, self.bag.shape[0]):
                f_weight += gamma[instance][k] - self.weights[k]
                f_mean_vector += gamma[instance][k] * (self.bag[instance] - self.mean[k]) / np.sqrt(
                    np.diag(self.covariance[k]))
                f_covariance += gamma[instance][k] * 1 / np.sqrt(2) * (
                        (self.bag[instance] - self.mean[k]) ** 2 / np.diag(self.covariance[k]) - 1)
            temp_f_x = []
            temp_f_x.extend(f_weight)
            temp_f_x.extend(f_mean_vector)
            temp_f_x.extend(f_covariance)
            f_X.extend([(1 / np.sqrt(self.weights[k])) * i for i in temp_f_x])
        return f_X

    def __cal_gamma(self):
        gamma = []
        for instance in self.bag:
            instance_gamma = []
            p_k_instance = self.__p_k_instance(instance)
            for k in range(0, self.K):
                if p_k_instance == 0:
                    instance_gamma.append(0)
                else:
                    instance_gamma.append(self.weights[k] * self.__p_k(instance, k) / p_k_instance)
            gamma.append(instance_gamma)
        return gamma

    def __p_k_instance(self, instance):
        reslut = 0.0
        for k in range(0, self.K):
            reslut = reslut + self.weights[k] * self.__p_k(instance, k)
        return reslut

    def __p_k(self, instance, k):
        temp_dis = instance - self.mean[k]
        numerator = np.exp(-0.5 * np.dot(np.dot(temp_dis, np.linalg.inv(self.covariance[k])), temp_dis))
        denominator = (2 * math.pi) ** ((self.bag.shape[1]) * self.K * 0.5) * np.sqrt(
            np.linalg.norm(self.covariance[k], ord=1))
        return numerator / denominator
