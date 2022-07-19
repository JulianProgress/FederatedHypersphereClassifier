#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2022/02/14 20:43
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : vae.py
# @Software  : PyCharm
import torch
from torch import nn


class VAE(nn.Module):
    def __init__(self, input_dim=8, output_len=1, paddings=(0,) * 3, output_paddings=(1, 1, 0), strides=(2, 2, 2),
                 kernel_sizes=(5,) * 3, dilations=(1, 2, 4), embedding_dim=64):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.output_len = output_len
        self.hidden_dim = [self.input_dim, 32, 64, 64]
        self.paddings = paddings
        self.output_paddings = output_paddings
        self.strides = strides
        self.kernel_sizes = kernel_sizes
        self.dilations = dilations
        self.embedding_dim = embedding_dim
        self.enc_l_out = lambda l_in, padding, kernel_size, stride, dilation: int(
            (l_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)

        self.dec_l_out = lambda l_in, padding, output_padding, kernel_size, stride, dilation: int(
            (l_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1)

        self.l_in = 7 * 24

        self.inp = torch.randn(100, 8, self.l_in)  # n, c_in, l_in

        # Encoder

        self.encoder = nn.Sequential(
            nn.Conv1d(66, 32, kernel_size=5, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Conv1d(66, 32, kernel_size=5, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Conv1d(66, 32, kernel_size=5, stride=2, padding=0),
            nn.LeakyReLU(),
        )

        enc_convs = nn.Sequential()
        for i in range(3):
            enc_convs.add_module('conv%d' % (i + 1),
                                 nn.Conv1d(self.hidden_dim[i], self.hidden_dim[i + 1], self.kernel_sizes[i],
                                           self.strides[i], self.paddings[i],
                                           self.dilations[i]))
            enc_convs.add_module('relu%d' % (i + 1), nn.LeakyReLU())
            self.l_in = self.enc_l_out(self.l_in, self.paddings[i], self.kernel_sizes[i], self.strides[i],
                                       self.dilations[i])

        self.fc_in = self.l_in * self.hidden_dim[-1]
        # self.fc_dims = (self.fc_in, self.embedding_dim * 2)

        self.fc1 = nn.Linear(self.fc_in, self.embedding_dim)
        self.fc2 = nn.Linear(self.fc_in, self.embedding_dim)
        self.fc3 = nn.Linear(self.embedding_dim, self.fc_in)

        # enc_fcs = nn.Sequential()
        # enc_fcs.add_module('fc%d' % (i + 1), nn.Linear(self.fc_dims[i], self.fc_dims[i + 1]))

        self.l_out = self.l_in

        # Decoder
        # dec_fcs = nn.Sequential()
        # for i in range(2):
        #     if i == 0:
        #         in_c = self.embedding_dim
        #     else:
        #         in_c = self.fc_dims[-1 - i]
        # dec_fcs.add_module('fc%d' % (i + 1), nn.Linear(self.embedding_dim, self.l_in))

        dec_convs = nn.Sequential()
        for i in range(3):
            if i < 2:
                dec_convs.add_module('conv%d' % (i + 1),
                                     nn.ConvTranspose1d(self.hidden_dim[-1 - i], self.hidden_dim[-2 - i],
                                                        self.kernel_sizes[-1 - i],
                                                        self.strides[-1 - i], self.paddings[-1 - i],
                                                        self.output_paddings[-1 - i],
                                                        dilation=self.dilations[-1 - i]))
            else:
                dec_convs.add_module('conv%d' % (i + 1),
                                     nn.ConvTranspose1d(self.hidden_dim[-1 - i], 1,
                                                        self.kernel_sizes[-1 - i],
                                                        self.strides[-1 - i], self.paddings[-1 - i],
                                                        self.output_paddings[-1 - i],
                                                        dilation=self.dilations[-1 - i]))
            dec_convs.add_module('relu%d' % (i + 1), nn.LeakyReLU())
            self.l_out = self.dec_l_out(self.l_out, self.paddings[i], self.output_paddings[i], self.kernel_sizes[i],
                                        self.strides[i], self.dilations[i])

        self.enc_convs = enc_convs
        # self.enc_fcs = enc_fcs
        # self.dec_fcs = dec_fcs
        self.dec_convs = dec_convs
        # self.final_fc = nn.Linear(66*168, self.output_len)

        self.initialize()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def forward(self, inp):
        # print(inp.shape)
        out = self.enc_convs(inp)
        # print(out.shape)
        out = out.view(out.shape[0], -1)
        # print(out.shape)
        out, mu, logvar = self.bottleneck(out)
        # print(out.shape)
        # mu, logvar = out[:, :self.embedding_dim], out[:, self.embedding_dim:]
        # print(mu.shape, logvar.shape)
        # out = self.reparameterize(mu, logvar)
        # print(out.shape)
        out = self.fc3(out)
        # print(out.shape)
        out = out.reshape(out.shape[0], int(out.shape[1] / self.l_in), self.l_in)
        # print(out.shape)
        out = self.dec_convs(out)
        # print(out.shape)
        # out = out.view(out.shape[0], -1)
        # out = self.final_fc(out)
        # print(out.shape)
        return out, mu, logvar