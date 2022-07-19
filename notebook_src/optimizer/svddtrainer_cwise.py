#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2022/02/23 21:14
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : svddtrainer.py
# @Software  : PyCharm


from copy import deepcopy

import torch
from torch import nn, optim
from tqdm.auto import tqdm

from notebook_src.base import BaseTrainer
from notebook_src.model import RNNAE
from notebook_src.utils import AverageMeter, concatenate
import time

import numpy as np
from sklearn.metrics import roc_curve, classification_report, roc_auc_score

# self, optimizer_name: str, lr: float, n_epochs: int, lr_milestones: tuple,
#                  weight_decay: float, device: str, init_steps: int, cid: str, patience: int, include_pert: bool, gamma: float,
#                  radius_thresh: float, pert_steps: int, pert_step_size: float, pert_duration: int, client_wise=False, update_c=False

class DSVDDTrainer_cwise(BaseTrainer):
    def __init__(self, optimizer_name: str, lr: float, n_epochs: int, lr_milestones: tuple,
                 weight_decay: float, device: str, verbose: bool, nu: float, cid: str, random_seed, pretrain_lr, pretrain_weight_decay):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, weight_decay, device)

        self.pretrain_weight_decay = pretrain_weight_decay
        self.pretrain_lr = pretrain_lr
        self.random_seed = random_seed
        self.verbose = verbose
        self.c_idx = cid
        self.nu = nu
        self.R = torch.tensor(0.0, device=self.device)
        self.C = None

    def iter(self, net, batch, isTrain):
        batch['X'] = batch['X'].float().to(self.device)
        batch['seq_len'] = batch['seq_len'].to(self.device)
        x = batch['X']
        # print(x.data.size())
        output, enc_hidden = net(x)
        # print(output[0].data.size())
        # output = self.output_handler(output)
        # if output[0].data.size(1) != x.data.size(1):
        #     return 0, None
        # else:
        return output, enc_hidden, x

    def gamma_tune(self, data_loader, net: RNNAE = None):
        net.to(self.device)

        # R = torch.tensor(0.0, device=self.device)
        # C = torch.randn(self.model_param['encoder_embedding_dim'] * 2, device=self.device, requires_grad=True)

        criterion = nn.MSELoss()
        gamma = tune_gamma(net, criterion, train_loader=data_loader, device=self.device)
        return gamma

    def pretrain(self, data_loader, valid_loader, net: RNNAE, pretrain_epochs=20, patience=5):
        net.to(self.device)
        criterion = nn.MSELoss().to(self.device)
        optimizer = getattr(optim, self.optimizer_name)(net.parameters(), lr=self.pretrain_lr, weight_decay=self.pretrain_weight_decay)
        if not self.lr_milestones is None:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones)

        best_loss = 999999
        best_net = None

        updated_epoch = 0

        for epoch in tqdm(range(pretrain_epochs)):

            train_loss = AverageMeter()

            net.train()
            for i, (X, y) in enumerate(data_loader):
                X = X.float().to(self.device)

                X_hat = net(X)
                loss = criterion(X_hat, X)
                train_loss.update(loss.item(), X.size(0))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            net.eval()
            val_ae_loss = AverageMeter()
            for i, (X, y) in enumerate(valid_loader):
                X = X.float().to(self.device)

                X_hat = net(X)
                loss = criterion(X_hat, X)

                val_ae_loss.update(loss.item(), X.size(0))

            message = 'Epoch: {0} \t| Train Loss: {1:.8f}, Validation Loss: {2:.8f}'.format(epoch + 1, train_loss.avg,
                                                                                            val_ae_loss.avg)
            print(message)
            scheduler.step()

            if best_loss > val_ae_loss.avg:
                print('best model updated')
                best_loss = val_ae_loss.avg
                best_net = deepcopy(net)
                updated_epoch = epoch

            if epoch - updated_epoch > patience:
                print('Early stopping')
                break

        return best_net


    def train(self, data_loader, valid_loader, net: RNNAE = None, isSVDD=True, T=1, patience=5):
        global center_optimizer, gamma, scheduler

        gamma = None
        net.to(self.device)
        criterion = nn.MSELoss().to(self.device)
        optimizer = getattr(optim, self.optimizer_name)(net.encoder.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if not self.lr_milestones is None:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones)

        if self.verbose:
            print('Starting {} client training...'.format(self.c_idx))

        if self.C is None:
            self.C = self.init_center_c(net, data_loader)

        gamma = tune_gamma(net, criterion, train_loader=data_loader, device=self.device, T=T)

        best_loss = 999999
        self.best_model = None
        updated_epoch = 0

        print('done gamma update')
        print('gamma: ', gamma)

        for epoch in tqdm(range(self.n_epochs)):

            svdd_loss = AverageMeter()

            net.train()

            for i, (X, y) in enumerate(data_loader):
                batch_size = y.size(0)
                X = X.float().to(self.device)
                enc_hidden = net.encode(X)

                dist = torch.sum((enc_hidden - self.C) ** 2, dim=1)
                loss = torch.mean(dist)

                svdd_loss.update(loss.item(), batch_size)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            net.eval()
            val_ae_loss = AverageMeter()
            for i, batch in enumerate(valid_loader):
                X = X.float().to(self.device)
                enc_hidden = net.encode(X)

                dist = torch.sum((enc_hidden - self.C) ** 2, dim=1)
                loss = torch.mean(dist)

                val_ae_loss.update(loss.item(), batch_size)

            message = 'Client: {0} Epoch: {1} \t| Train Loss: {1:.8f}, Validation Loss: {2:.8f}'.format(self.c_idx, epoch + 1, svdd_loss.avg,
                                                                                            val_ae_loss.avg)
            print(message)

            scheduler.step()

            if best_loss > val_ae_loss.avg:
                print('best model updated')
                best_loss = val_ae_loss.avg
                self.best_model = deepcopy(net)
                updated_epoch = epoch

            if epoch - updated_epoch > patience:
                print("Early stopping..")
                break

        return gamma

    def pretrain_test(self, dataset, net: RNNAE, global_thresh=0.5):
        start_time = time.time()
        net.to(self.device)
        net.eval()

        scores = None
        labels = None
        num_examples_test = 0

        with torch.no_grad():
            for X, y in dataset:
                X = X.float().to(self.device)

                outputs = net(X)
                num_examples_test += X.size(0)

                dist = torch.mean((outputs.reshape(outputs.size(0), -1) - X.reshape(X.size(0), -1)) ** 2, dim=1)
                scores = concatenate(scores, dist.detach().cpu().numpy())
                labels = concatenate(labels, y.detach().cpu().numpy())

        return_type=0
        if labels.sum() > 0:
            fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
            J = tpr - fpr
            ix = np.argmax(J)
            best_thresh = thresholds[ix]
            y_prob_pred = (scores >= best_thresh).astype(bool)
            obj = classification_report(labels, y_prob_pred, output_dict=True)
            test_auc = roc_auc_score(labels, scores)

            print('Testing duration: %.4f s' % (time.time() - start_time))
        else:
            return_type = 1
            y_prob_pred = (scores >= global_thresh).astype(bool)
            obj = y_prob_pred.sum() / len(scores)
            test_auc = obj

        return num_examples_test, obj, test_auc, return_type


    def test(self, dataset, net: RNNAE, c=None, global_thresh=None):
        start_time = time.time()
        net.to(self.device)
        net.eval()

        scores = None
        labels = None
        num_examples_test = 0
        if c is None:
            c = self.C
        with torch.no_grad():
            for X, y in dataset:
                X = X.float().to(self.device)

                outputs = net.encode(X)
                num_examples_test += X.size(0)

                dist = torch.sum((outputs - c) ** 2, dim=1)
                scores = concatenate(scores, dist.detach().cpu().numpy())
                labels = concatenate(labels, y.detach().cpu().numpy())

        return_type=0
        if labels.sum() > 0:
            fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
            J = tpr - fpr
            ix = np.argmax(J)
            best_thresh = thresholds[ix]
            y_prob_pred = (scores >= best_thresh).astype(bool)
            obj = classification_report(labels, y_prob_pred, output_dict=True)
            test_auc = roc_auc_score(labels, scores)

            print('Testing duration: %.4f s' % (time.time() - start_time))
        else:
            return_type = 1
            y_prob_pred = (scores >= global_thresh).astype(bool)
            obj = y_prob_pred.sum() / len(scores)
            test_auc = obj

        return num_examples_test, obj, test_auc, return_type


    # def test(self, data_loader, gamma=0, C=None, net=None, isSVDD=True):
    #     score = []
    #     labels = []
    #     embeddings = []
    #     inputs = []
    #     criterion = nn.MSELoss().to(self.device)
    #     net.eval()
    #     for batch in data_loader:
    #         batch['X'] = batch['X'].float().to(self.device)
    #         batch['seq_len'] = batch['seq_len'].to(self.device)
    #         x = batch['X']
    #
    #         output, enc_hidden = net(x)
    #         if isSVDD:
    #             loss = criterion(output, x) + gamma * torch.sum((enc_hidden - C) ** 2, dim=1)
    #         else:
    #             loss = criterion(output, x)
    #         score.append(loss.detach().cpu().numpy())
    #         labels.append(batch['y'].detach().cpu().numpy())
    #         embeddings.append(enc_hidden.detach().cpu().numpy())
    #         inputs.append(x.detach().cpu().numpy())
    #
    #     return score, labels, embeddings, inputs

    def save_model(self, net, save_path=None):
        if save_path is None:
            save_path = './net.pt'
        torch.save(net.cpu().state_dict(), save_path)
        net.to(self.device)

    def load_model(self, net, save_path=None):
        if save_path is None:
            save_path = './net.pt'
        state_dict = torch.load(save_path)
        net.cpu().load_state_dict(state_dict)
        net.to(self.device)

    def init_center_c(self, net, train_loader, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        window_size = 128
        c = torch.zeros(net.encoder_dim * window_size, device=self.device)

        net.eval()
        # self.seq2seq.eval()

        with torch.no_grad():
            for X, y in train_loader:
                # get the inputs of the batch
                X = X.float().to(self.device)

                z = net.encode(X)

                n_samples += z.shape[0]
                c += torch.sum(z, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c

    # def get_radius(self, dist: torch.Tensor, nu: float):
    #     """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    #     return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)


def tune_gamma(model, criterion, train_loader, device="cpu", T=5):
    gamma = 0
    model.eval()
    for k in range(T):
        # model = AE(input_shape=in_shape).to(device)
        R = 0
        RE = 0
        for j, (X, y) in enumerate(tqdm(train_loader)):
            # print(j)
            # if isinstance(batch_features, list):
            #     batch_features = batch_features[0]
            # batch_features = batch_features.view(-1, in_shape).to(device)
            X = X.float().to(device)
            outputs = model(X)
            enc_hidden = model.encode(X)
            R += torch.sum((enc_hidden) ** 2, dim=1)[0].item()

            # print(R)
            RE += criterion(outputs, X).item()
        R = R / len(train_loader)
        RE = RE / len(train_loader)
        gamma += RE / R

    gamma = gamma / T
    return gamma
