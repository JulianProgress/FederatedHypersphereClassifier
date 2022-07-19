#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2022/02/14 14:36
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : aetrainer.py
# @Software  : PyCharm

from copy import deepcopy

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from base import BaseTrainer, BaseNet
from utils import AverageMeter



class RepAETrainer(BaseTrainer):
    def __init__(self, optimizer_name: str, lr: float, n_epochs: int, lr_milestones: tuple,
                 weight_decay: float, device: str, X_idx, y_idx, patience=7, transpose=True):
        super(RepAETrainer, self).__init__(optimizer_name, lr, n_epochs, lr_milestones, weight_decay, device)
        self.patience = patience
        self.X_idx = X_idx
        self.y_idx = y_idx
        self.transpose = transpose

    def init_optim(self, net: BaseNet) -> optim.Optimizer:
        return getattr(optim, self.optimizer_name)(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def prepare_data(self, batch, X_idx: any = 0, y_idx: any = 1):
        X = batch[X_idx].float().to(self.device)
        y = batch[y_idx].float().to(self.device)
        return X, y

    def train(self, dataset: DataLoader, validset: DataLoader, net: BaseNet) -> BaseNet:

        net.to(self.device)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()
        net.train()

        best_loss = 999999

        best_ae = None
        last_update = 0

        for epoch in tqdm(range(self.n_epochs)):
            ae_train_loss = AverageMeter()
            num_examples_train = 0

            """
            Train step
            """
            net.train()
            for batch in dataset:
                X, y = self.prepare_data(batch, X_idx=self.X_idx, y_idx=self.y_idx)

                X_hat = net(X, transpose=self.transpose)

                loss = criterion(X, X_hat)
                ae_train_loss.update(loss.item(), X.size(0))
                num_examples_train += X.size(0)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            """
            Validation step
            """
            ae_val_loss = self.test(validset, net)

            if ae_val_loss.avg < best_loss:
                last_update = epoch
                best_loss = ae_val_loss.avg
                best_ae = deepcopy(net)

            print("Pretrain Epoch %d | train mse: %.8f | val mse: %.8f" % (
                epoch + 1, ae_train_loss.avg, ae_val_loss.avg))

            if epoch - last_update > self.patience:
                print('Early stopping.')
                break
        return net, num_examples_train

    def test(self, dataset: DataLoader, net: BaseNet):
        net.eval()
        ae_val_loss = AverageMeter()
        criterion = nn.MSELoss()

        for batch in dataset:
            X, y = self.prepare_data(batch, X_idx=self.X_idx, y_idx=self.y_idx)

            X_hat = net(X, transpose=self.transpose)

            loss = criterion(X, X_hat)
            ae_val_loss.update(loss.item(), X.size(0))

        return ae_val_loss