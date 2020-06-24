# coding: utf-8
from optimizer import SGD, Momentum, Nesterov, AdaGrad, RMSprop, Adam
import numpy as np
import sys
import os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定


class Trainer:
    def __init__(
        self, network,
        epochs=20, mini_batch_size=100,
        optimizer='adam', optimizer_param={'lr': 0.01},
        evaluate_sample_num_per_epoch=None, verbose=True
    ):
        self.network = network
        self.x_train = None         # init in train function
        self.t_train = None         # init in train function
        self.verbose = verbose
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch

        # optimizer
        optimizer_class_dict = {
            'sgd': SGD,
            'momentum': Momentum,
            'nesterov': Nesterov,
            'adagrad': AdaGrad,
            'rmsprop': RMSprop,
            'adam': Adam}
        self.optimizer = optimizer_class_dict[optimizer.lower()](** optimizer_param)

        self.train_size = 0         # init in train function
        self.iter_per_epoch = 0     # init in train function
        self.max_iter = 0           # init in train function
        self.current_iter = 1
        self.current_epoch = 1

        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train_step(self):
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]

        grads = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)

        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)
        if self.verbose:
            print("train loss:{}".format(loss))

        if self.current_iter % self.iter_per_epoch == 0:
            x_train_sample, t_train_sample = self.x_train, self.t_train

            if self.evaluate_sample_num_per_epoch is not None:
                t = self.evaluate_sample_num_per_epoch
                x_train_sample, t_train_sample = self.x_train[:t], self.t_train[:t]

            train_acc = self.network.accuracy(x_train_sample, t_train_sample)
            self.train_acc_list.append(train_acc)

            if self.verbose:
                print("epoch:{}, train acc:{}".format(self.current_epoch, train_acc))

            self.current_epoch += 1

        self.current_iter += 1

    def train(self, x_train, t_train, epochs=100, mini_batch_size=10):
        self.x_train = x_train
        self.t_train = t_train
        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        self.max_iter = int(epochs * self.iter_per_epoch)

        for i in range(self.max_iter):
            self.train_step()

    def test(self, x_test, t_test):
        test_acc = self.network.accuracy(x_test, t_test)

        if self.verbose:
            print("test acc:" + str(test_acc))
