# @Time : 2020/12/31 11:10
# @Author: ZWX
# @Email: 935721546@qq.com
# @File : miFV.py

import numpy as np
import warnings
from sklearn.mixture import GaussianMixture

from FuncTool import get_index, load_file, get_matrix
from FvGmm import FvGm

warnings.filterwarnings("ignore")


class miFV:

    def __init__(self, file_name, para_k=1, k_cv=10, bags=None):
        if bags is None:
            self.bags = load_file(file_name)
        else:
            self.bags = bags
        self.tr_bags = [[]]
        self.te_bags = [[]]
        self.num_bags = self.bags.shape[0]
        self.k_cv = k_cv
        self.K = para_k

    def get_mapping(self):
        tr_index, te_index = get_index(para_k=10, para_num_bags=self.num_bags)  # Get the index of train and test.
        for j in range(0, self.k_cv):
            self.tr_bags = self.bags[tr_index[j]]  # The train bag set.
            self.te_bags = self.bags[te_index[j]]  # The test bag set.
            vector_tr, vector_te = self.__Gmm(self.tr_bags, self.te_bags)  # The mapping vector.

            yield vector_tr[:, :-1], vector_tr[:, -1], vector_te[:, :-1], vector_te[:, -1], None

    def __Gmm(self, tr_bags, te_bags):
        """
        The mapping vector.
        :param tr_bags:
        :param te_bags:
        :return:
        """
        bags_instance = get_matrix(tr_bags)  # The space of instance.
        fv_model = GaussianMixture(n_components=self.K)  # The GMM model.
        fv_model.fit(bags_instance)
        data_mean = fv_model.means_  # The mean of gmm.
        data_weight = fv_model.weights_  # The weights of gmm.
        data_diag = fv_model.covariances_  # The covariances.

        # Step 1. Calculate the mapping vector of train bag.
        vector_tr = []
        for i in range(0, tr_bags.shape[0]):
            # Step 1.2. Predict the gamma of each bag.
            gamma = fv_model.predict_proba(tr_bags[i, 0][:, :-1])
            temp_vector_tr = np.zeros(((tr_bags[i, 0].shape[1] - 1) * 2 + 1) * self.K + 1).astype('float64')
            # Step 1.3 Call function to get vector.
            test_FvGm = FvGm(para_bag=tr_bags[i, 0][:, :-1], para_weights=data_weight, para_mean=data_mean,
                             para_covariances=data_diag,
                             para_K=self.K, para_gamma=gamma)
            temp_vector_tr[:-1] = test_FvGm.compute_fisher()
            temp_vector_tr[:-1] = self.__sqrt_norm(temp_vector_tr[:-1])
            temp_vector_tr[-1] = tr_bags[i, -1]
            vector_tr.append(temp_vector_tr)
        # Step 2. Calculate the mapping vector of train bag.
        vector_te = []
        for i in range(0, te_bags.shape[0]):
            # Step 2.2. Predict the gamma of each bag.
            gamma = fv_model.predict_proba(te_bags[i, 0][:, :-1])
            temp_vector_te = np.zeros(((te_bags[i, 0].shape[1] - 1) * 2 + 1) * self.K + 1).astype('float64')
            # Step 2.3 Call function to get vector.
            test_FvGm = FvGm(para_bag=te_bags[i, 0][:, :-1], para_weights=data_weight, para_mean=data_mean,
                             para_covariances=data_diag,
                             para_K=self.K, para_gamma=gamma)
            temp_vector_te[:-1] = test_FvGm.compute_fisher()
            temp_vector_te[:-1] = self.__sqrt_norm(temp_vector_te[:-1])
            temp_vector_te[-1] = te_bags[i, -1]
            vector_te.append(temp_vector_te)
        return np.array(vector_tr), np.array(vector_te)

    def __sqrt_norm(self, vector):
        """
        Normalized the mapping vector.
        :param vector:The mapping vector of single bag.
        :return:
        """
        vector = np.sign(vector) * (np.sqrt(np.abs(vector)))
        vector = vector / np.linalg.norm(vector)
        return vector
