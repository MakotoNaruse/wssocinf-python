# coding: utf-8
import numpy as np
import pickle
from layers import Relu, Affine, SoftmaxWithLoss, Dropout, Convolution, Pooling, BatchNormalization
from functions import init_he


class DeepConvNet:
    def __init__(
        self, input_dim,
        conv_params=[
            {'filter_num': 16, 'filter_size': 9, 'pad': 4, 'stride': 1},
            {'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 1},
            {'filter_num': 32, 'filter_size': 5, 'pad': 2, 'stride': 1},
            {'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 1},
            {'filter_num': 64, 'filter_size': 7, 'pad': 3, 'stride': 1},
            {'filter_num': 64, 'filter_size': 3, 'pad': 1, 'stride': 1},],
        hidden_size=[128, 64],
        dropout_ratio=[0.4, 0.4, 0.4],
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
            self.params['b' + str(idx + 1)
                        ] = np.zeros(conv_param['filter_num'])

            # set layers
            self.layers['Conv' + str(idx + 1)] = Convolution(
                self.params['W' + str(idx + 1)],
                self.params['b' + str(idx + 1)],
                conv_param['stride'],
                conv_param['pad'])
            self.layers['BatchNorm' + str(idx + 1)] = BatchNormalization()
            self.layers['Relu' + str(idx + 1)] = Relu()

            # calc output image size of conv layers
            pre_shape = self.layers['Conv' +
                                    str(idx + 1)].output_size(pre_shape)

        idx = len(conv_params)

        # init parameters and set layers Affine
        self.params['W' + str(idx + 1)] = init_he(pre_shape[0] * pre_shape[1]**2) *\
            np.random.randn(pre_shape[0] * pre_shape[1]**2, hidden_size[0])
        self.params['b' + str(idx + 1)] = np.zeros(hidden_size[0])
        self.layers['Affine' + str(idx + 1)] = Affine(self.params['W' + str(idx + 1)], self.params['b' + str(idx + 1)])
        self.layers['Relu' + str(idx + 1)] = Relu()
        self.layers['Dropout' + str(idx + 1)] = Dropout(dropout_ratio[0])
        idx += 1

        # init parameters and set layers Affine
        self.params['W' + str(idx + 1)] = init_he(
            hidden_size[0]) * np.random.randn(hidden_size[0], hidden_size[1])
        self.params['b' + str(idx + 1)] = np.zeros(hidden_size[1])
        self.layers['Affine' + str(idx + 1)] = Affine(self.params['W' + str(idx + 1)], self.params['b' + str(idx + 1)])
        self.layers['Relu' + str(idx + 1)] = Relu()
        self.layers['Dropout' + str(idx + 1)] = Dropout(dropout_ratio[1])
        idx += 1

        # init parameters and set layers output
        self.params['W' + str(idx + 1)] = init_he(
            hidden_size[1]) * np.random.randn(hidden_size[1], output_size)
        self.params['b' + str(idx + 1)] = np.zeros(output_size)
        self.layers['Affine' + str(idx + 1)] = Affine(self.params['W' + str(idx + 1)], self.params['b' + str(idx + 1)])
        self.layers['Dropout' + str(idx + 1)] = Dropout(dropout_ratio[2])

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

        tmp_layers = list(self.layers.values())
        tmp_layers.reverse()
        for layer in tmp_layers:
            dout = layer.backward(dout)

        # setting
        grads = {}
        for i, layer_name in enumerate(self.get_layer_names()):
            grads['W' + str(i + 1)] = self.layers[layer_name].dW
            grads['b' + str(i + 1)] = self.layers[layer_name].db

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

        for i, layer_name in enumerate(self.get_layer_names()):
            self.layers[layer_name].W = self.params['W' + str(i + 1)]
            self.layers[layer_name].b = self.params['b' + str(i + 1)]

    def get_layer_names(self):
        lst = []
        for layer_name in self.layers.keys():
            if 'Conv' in layer_name or 'Affine' in layer_name:
                lst.append(layer_name)

        return np.array(lst)
