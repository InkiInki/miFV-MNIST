# coding: utf-8
import numpy as np
from FuncTool import mnist_bag_loader, print_progress_bar


class MnistLoader:

    def __init__(self, po_label=0, bag_size=(10, 50), po_range=(2, 8), bag_num=(100, 100), seed=None,
                 mnist_path=None):
        self.po_label = po_label
        self.bag_size = bag_size
        self.po_range = po_range
        self.bag_num = bag_num
        self.seed = seed
        self.mnist_path = mnist_path
        self.__init_mnist_loader()

    def __init_mnist_loader(self):
        print("Loading mnist...")
        self.data_space = []
        self.label_space = []
        self.bag_space = []
        if self.seed is not None:
            np.random.seed(self.seed)

        self.data_space, self.label_space = self.__load_data(True)
        data_space, label_space = self.__load_data(False)
        self.data_space.extend(data_space)
        self.label_space.extend(label_space)
        self.data_space, self.label_space = np.array(self.data_space), np.array(self.label_space)
        self.po_idx = np.where(self.label_space == self.po_label)[0]
        self.ot_idx = np.where(self.label_space != self.po_label)[0]
        self.__generate_po_bag(self.bag_num[0])
        self.__generate_ot_bag(self.bag_num[1])
        self.bag_space = np.array(self.bag_space)

    def __load_data(self, train):
        flag = "train" if train else "test"
        print("Loading MNIST %s data..." % flag)

        data_loader = mnist_bag_loader(train, self.mnist_path)
        num_data = len(data_loader)

        ret_data, ret_label = [], []
        for i, (data, label) in enumerate(data_loader):
            print_progress_bar(i, num_data)
            data, label = data.reshape(-1).numpy().tolist(), int(label.numpy()[0])
            ret_data.append(data)
            ret_label.append(label)
        print()
        return ret_data, ret_label

    def __generate_po_bag(self, bag_num):
        print("Generating positive bag...")
        for i in range(bag_num):
            print_progress_bar(i, bag_num)
            bag = []
            bag_size = np.random.randint(self.po_range[0], self.po_range[1] + 1)
            for j in range(bag_size):
                ins = self.data_space[np.random.choice(self.po_idx)].tolist() + [1]
                bag.append(ins)
            bag_size = np.random.randint(self.bag_size[0] - self.po_range[0], self.bag_size[1] - self.po_range[1] + 1)
            for j in range(bag_size):
                ins = self.data_space[np.random.choice(self.ot_idx)].tolist() + [0]
                bag.append(ins)
            bag = np.array(bag)
            bag = np.array([bag, np.array([[1]])])
            self.bag_space.append(bag)
        print()

    def __generate_ot_bag(self, bag_num):
        print("Generate other class bag...")
        for i in range(bag_num):
            print_progress_bar(i, bag_num)
            bag = []
            bag_size = np.random.randint(self.bag_size[0], self.bag_size[1] + 1)
            for j in range(bag_size):
                ins = self.data_space[np.random.choice(self.ot_idx)].tolist() + [0]
                bag.append(ins)
            bag = np.array(bag)
            bag = np.array([bag, np.array([[0]])])
            self.bag_space.append(bag)
        print()


if __name__ == '__main__':
    ml = MnistLoader(po_label=0, seed=1)
