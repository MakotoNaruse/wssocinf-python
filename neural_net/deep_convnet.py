# coding: utf-8
import numpy as np
import pickle
from layers import Relu, Affine, SoftmaxWithLoss, Dropout, Convolution
from functions import init_he


class DeepConvNet:
    def __init__(
        self, input_dim=(3, 32, 32),
        conv_params=[
            {'filter_num': 16, 'filter_size': 9, 'pad': 1, 'stride': 1},
            {'filter_num': 32, 'filter_size': 5, 'pad': 1, 'stride': 1},
            {'filter_num': 64, 'filter_size': 7, 'pad': 1, 'stride': 1}],
        hidden_size=[128, 64],
        output_size=5
    ):
        self.params = {}
        self.layers = {}
        pre_shape = input_dim
        for idx, conv_param in enumerate(conv_params):
            # init parameters
            self.params['W' + str(idx + 1)] = init_he(pre_shape[0] * conv_param['filter_size']**2) *\
                np.random.randn(
                    conv_param['filter_num'],
                    pre_shape[0],
                    conv_param['filter_size'],
                    conv_param['filter_size'])
            self.params['b' + str(idx + 1)] = np.zeros(conv_param['filter_num'])

            # set layers
            self.layers['Conv' + str(idx + 1)] = Convolution(
                self.params['W' + str(idx + 1)],
                self.params['b' + str(idx + 1)],
                conv_param['stride'],
                conv_param['pad'])
            self.layers['Relu' + str(idx + 1)] = Relu()

            # calc output image size of conv layers
            pre_shape = self.layers['Conv' + str(idx + 1)].output_size(pre_shape)

        # init parameters and set layers Affine4
        self.params['W4'] = init_he(pre_shape[0] * pre_shape[1]**2) *\
            np.random.randn(pre_shape[0] * pre_shape[1]**2, hidden_size[0])
        self.params['b4'] = np.zeros(hidden_size[0])
        self.layers['Affine4'] = Affine(self.params['W4'], self.params['b4'])
        self.layers['Relu4'] = Relu()
        self.layers['Dropout4'] = Dropout(0.5)

        # init parameters and set layers Affine5
        self.params['W5'] = init_he(
            hidden_size[0]) * np.random.randn(hidden_size[0], hidden_size[1])
        self.params['b5'] = np.zeros(hidden_size[1])
        self.layers['Affine5'] = Affine(self.params['W5'], self.params['b5'])
        self.layers['Relu5'] = Relu()
        self.layers['Dropout5'] = Dropout(0.5)

        # init parameters and set layers output
        self.params['W6'] = init_he(
            hidden_size[1]) * np.random.randn(hidden_size[1], output_size)
        self.params['b6'] = np.zeros(output_size)
        self.layers['Affine6'] = Affine(self.params['W6'], self.params['b6'])
        self.layers['Dropout6'] = Dropout(0.5)

        # set loss function layer
        self.loss_layer = SoftmaxWithLoss()

    def predict(self, x, train_flg=False):
        for layer in self.layers.values():
            if isinstance(layer, Dropout):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x, train_flg=True)
        return self.loss_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size:(i + 1) * batch_size]
            tt = t[i * batch_size:(i + 1) * batch_size]
            y = self.predict(tx, train_flg=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.loss_layer.backward(dout)

        tmp_layers = self.layers.copy()
        tmp_layers.reverse()
        for layer in tmp_layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
            grads['W' + str(i + 1)] = self.layers[layer_idx].dW
            grads['b' + str(i + 1)] = self.layers[layer_idx].db

        return grads

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
            self.layers[layer_idx].W = self.params['W' + str(i + 1)]
            self.layers[layer_idx].b = self.params['b' + str(i + 1)]
