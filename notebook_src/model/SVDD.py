#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2022/02/13 5:03 PM
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : SVDD.py
# @Software  : PyCharm
import torch

from .base_ae import Encoder
from torch import nn
from ..base import BaseNet
from .rnn_ae import HSCRNNEncoder
from .mnist_LeNet import MNIST_LeNet
from .cifar10_LeNet import CIFAR10_LeNet, CIFAR10_FE
from .cnn_lstm import C_LSTM

class SVDD(BaseNet):
    def __init__(self, rep_model_type: str='SimpleEncoder', window_size=128, in_dim: int=115, hidden_dims: list=None, num_rep_layer: int=2, num_svdd_layer: int=3, is_fc=False):
        super(SVDD, self).__init__()

        self.rep_dim = hidden_dims[-1]

        if rep_model_type == 'SimpleEncoder':
            self.encoder = Encoder(in_dim=in_dim, hidden_dims=hidden_dims)
        elif rep_model_type == 'RNNEncoder':
            self.encoder = HSCRNNEncoder(in_dim=in_dim, hidden_dim=hidden_dims[0], rep_dim=self.rep_dim, num_layers=num_rep_layer, window_size=window_size)
        elif rep_model_type == 'mnist':
            # print('mnistencoder')
            self.encoder = MNIST_LeNet()
        elif rep_model_type == 'cifar10':
            # print('cifar10')
            self.encoder = CIFAR10_LeNet()

        elif rep_model_type == 'toniot' or rep_model_type == 'cicids':
            self.encoder = C_LSTM(self.rep_dim)

        self.c = torch.zeros(self.rep_dim)
        # self.R = None

        svdd_layer = []
        for i in range(num_svdd_layer):
            svdd_layer.append(nn.Linear(hidden_dims[-1], hidden_dims[-1]))
            svdd_layer.append(nn.BatchNorm1d(hidden_dims[-1]))
            svdd_layer.append(nn.ReLU())

        if is_fc:
            svdd_layer.append(nn.Linear(hidden_dims[-1], 1))

        self.svdd = nn.Sequential(*svdd_layer)

    def encode(self, X):
        return self.encoder(X)

    def svdd_mapping(self, X):
        return self.svdd(X)

    def forward(self, X):
        z = self.encoder(X)
        svdd = self.svdd(z)
        return svdd



