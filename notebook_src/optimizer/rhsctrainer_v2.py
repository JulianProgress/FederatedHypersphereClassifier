#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2022/02/26 1:42
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : hsctrainer_r.py
# @Software  : PyCharm

from copy import deepcopy

import torch
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset

from ..base import BaseTrainer, BaseNet
from ..utils import AverageMeter, concatenate, get_confusion_matrix_values

import time
from tqdm.auto import tqdm

import numpy as np
from sklearn.metrics import roc_curve, classification_report, roc_auc_score, accuracy_score

class RHSCTrainerV2(BaseTrainer):
    def __init__(self, optimizer_name: str, lr: float, n_epochs: int, lr_milestones: tuple,
                 weight_decay: float, device: str, verbose: bool, init_steps: int, cid: str, patience: int,
                 include_pert: bool, gamma: float, is_abnormal: bool,
                 radius_thresh: float, pert_steps: int, pert_step_size: float, pert_duration: int, val_set=0,
                 client_wise=False, update_c=False, lamb: float=0., return_best=True):
        super(RHSCTrainerV2, self).__init__(optimizer_name, lr, n_epochs, lr_milestones, weight_decay, device)
        self.val_set = val_set
        self.return_best = return_best
        self.is_abnormal = is_abnormal
        self.client_wise = client_wise
        self.update_c = update_c
        self.verbose = verbose
        self.lamb = lamb
        # print(self.verbose)

        self.patience = patience
        self.pert_steps = pert_steps
        self.pert_step_size = pert_step_size
        self.include_pert = include_pert

        self.init_steps = init_steps
        self.c_idx = cid
        self.pert_gamma = gamma  # DROCC perturbation upper bound
        self.pert_radius = radius_thresh  # DROCC perturbation lower bound
        self.pert_duration = pert_duration

        self.train_loss_list = []
        self.valid_loss_list = []
        self.R_hist = []
        self.c = None
        self.R = None

    def init_optim(self, net: BaseNet) -> optim.Optimizer:
        return getattr(optim, self.optimizer_name)(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def initialize_c(self, net, dataloader, eps=0.1):
        n_samples = 0
        net.eval()
        c = torch.zeros(net.rep_dim, device=self.device)

        for X, y in dataloader:
            X = X.to(self.device)
            y = y.to(self.device)

            abnormal_idx_list = y.type(torch.bool)
            normal_X = X[~abnormal_idx_list]
            encoded = net.encode(normal_X)

            n_samples += encoded.shape[0]
            c += torch.sum(encoded, dim=0)
        c /= n_samples
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c >= 0)] = eps

        self.c = c.detach()

    def init_radius(self, net, dataloader):
        # _, radiuses = self.get_low_confidence(net, dataloader)
        # print(radiuses)
        # r, _ = self.personalized_r_gamma(radiuses)
        self.R = torch.tensor([0.1], requires_grad=True, device=self.device)
        print('done radius init')

    def get_low_confidence(self, net, dataloader, portion=0.9):

        radiuses = None
        norm_Xs = None
        net.to(self.device)
        net.eval()
        for X, y in dataloader:
            X = X.to(self.device)
            y = y.to(self.device)

            abnormal_idx_list = y.type(torch.bool)
            normal_X = X[~abnormal_idx_list]
            # if self.c_idx == '{11EEC3EB-2A31-4B80-BF40-0ECC2BB53EE4}':
            #     print(normal_X)
            encoded = net(normal_X)

            dist = torch.sqrt(torch.sum((encoded - self.c) ** 2, dim=1))

            norm_Xs = concatenate(norm_Xs, normal_X.detach().cpu().numpy())
            radiuses = concatenate(radiuses, dist.detach().cpu().numpy())
            # print(dist)


        if radiuses is None:
            radiuses = np.array([0., 0.])

        sorted_idxs = radiuses.argsort()

        norm_Xs = norm_Xs[sorted_idxs]
        low_confidence_X = norm_Xs[round(len(norm_Xs) * portion):]
        # print(low_confidence_X)
        try:
            tense = torch.tensor(low_confidence_X, dtype=torch.float32)
        except Exception as e:
            print(e)

        # print('done low confidence')
        return tense, radiuses

    def personalized_r_gamma(self, radiuses):
        # print('in r_gamma')
        radiuses.sort()
        r = radiuses[round(len(radiuses) * self.pert_radius)]
        r_max = np.max(radiuses)
        r_gamma = 1 + ((r_max - r) / r) * self.pert_gamma
        # print('r_gamma')
        return r, r_gamma

    def perturb(self, net, X, radiuses, num_gradient_steps=7, step_size=0.07):  # r=0.1, gamma=1.1
        net.to(self.device)
        net.eval()

        r, r_gamma = self.personalized_r_gamma(radiuses)

        # @TODO: fix perturbation for augmentation
        z = net.encode(X)
        zeta = torch.randn(z.shape).to(self.device).detach().requires_grad_()
        z_adv_sampled = z + zeta

        for step in range(num_gradient_steps):
            zeta.requires_grad_()
            with torch.enable_grad():
                svdd_map = net.svdd_mapping(z_adv_sampled)
                pert_dist = torch.abs(svdd_map - self.c)
                loss = torch.sum(pert_dist)
                grad = torch.autograd.grad(loss, [zeta])[0]
                grad_flattened = torch.reshape(grad, (grad.shape[0], -1))
                grad_norm = torch.norm(grad_flattened, p=2, dim=1)

                for u in range(grad.ndim - 1):
                    grad_norm = torch.unsqueeze(grad_norm, dim=u + 1)
                if grad.ndim == 2:
                    grad_norm = grad_norm.repeat(1, grad.shape[1])
                if grad.ndim == 3:
                    grad_norm = grad_norm.repeat(1, grad.shape[1], grad.shape[2])
                grad_normalized = grad / grad_norm

            with torch.no_grad():
                zeta.add_(step_size * grad_normalized)

            if (step + 1) % 3 == 0 or step == num_gradient_steps - 1:
                norm_zeta = torch.sqrt(torch.sum(zeta ** 2, dim=tuple(range(1, zeta.dim()))))
                alpha = torch.clamp(norm_zeta, r, r * r_gamma).to(self.device)

                proj = (alpha / norm_zeta).view(-1, *[1] * (zeta.dim() - 1))
                zeta = proj * zeta

                z_adv_sampled = z + zeta

        return z_adv_sampled

    def load_pretrained_model(self, ae, net):
        encoder_state = ae.encoder.rnn_layer.state_dict()
        net.encoder.rnn_layer.load_state_dict(encoder_state)

    def train(self, dataset: DataLoader, validset: DataLoader, net: BaseNet):
        # print(self.verbose)

        start_time = time.time()
        # if self.pretrain_e:
        #     self.pretrain(dataset, validset, net, lr=0.001, weight_decay=1e-6)
        net.to(self.device)

        optimizer = self.init_optim(net)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        if self.verbose:
            print('Starting {} client training...'.format(self.c_idx))

        if self.c is None:
            if self.verbose:
                print('init c..')
            self.initialize_c(net, dataset)
            # print(self.c.data)
        # radius = self.get_radius(net, dataset)
        if self.R is None:
            print('init R..')
            self.init_radius(net, dataset)

        opt_r = optim.Adam([self.R], lr=0.001)

        best_net = None
        best_loss = 999999
        best_R = 0
        best_c = None
        num_examples_train = 1
        last_update = 0

        print('start epoch')
        for epoch in range(self.n_epochs):
            train_dist = AverageMeter()
            train_loss = AverageMeter()
            valid_loss = AverageMeter()
            num_examples_train = 0

            net.train()
            """
            Training Step
            """
            if self.client_wise:
                pert_bool = epoch >= self.init_steps
            else:
                pert_bool = epoch == self.init_steps

            if self.include_pert and (pert_bool or (epoch + 1) % self.pert_duration == 0):
                # TODO: generate anomaly representations
                if self.verbose:
                    print('generate anomaly data')
                self.low_conf_X, radiuses = self.get_low_confidence(net, dataset)
                self.low_conf_X = self.low_conf_X.to(self.device)
                pert_z = self.perturb(net, self.low_conf_X.detach(), radiuses, num_gradient_steps=self.pert_steps,
                                      step_size=self.pert_step_size)
                self.pert_svdd_out = net.svdd_mapping(pert_z.detach())
                pert_y = torch.ones(self.pert_svdd_out.size(0))
                self.pert_loader = DataLoader(TensorDataset(self.pert_svdd_out.detach().cpu(), pert_y), batch_size=32,
                                              shuffle=True, drop_last=True, num_workers=0)

                net.train()

            if epoch > self.init_steps and self.include_pert:
                dataloader = zip(dataset, self.pert_loader)
            else:
                dataloader = dataset

            for i, data in enumerate(dataloader):
                if epoch > self.init_steps and self.include_pert:
                    X = data[0][0].to(self.device)
                    y = data[0][1].to(self.device)
                else:
                    X = data[0].to(self.device)
                    y = data[1].to(self.device)

                optimizer.zero_grad()
                opt_r.zero_grad()



                svdd_out = net(X)

                if epoch > self.init_steps and self.include_pert:
                    pert_in = data[1][0].to(self.device)
                    pert_y = data[1][1].to(self.device)
                    # pert_svdd_out = net.svdd_mapping(pert_in)
                    # svdd_out = torch.cat((svdd_out, pert_svdd_out))
                    svdd_out = torch.cat((svdd_out, pert_in))
                    y = torch.cat((y, pert_y))

                abnormal_idx_list = y.type(torch.bool)
                normal_len = len(X[~abnormal_idx_list])
                abnormal_len = len(X[abnormal_idx_list])
                total_len = normal_len + abnormal_len

                """
                objective
                """
                # losses = (1. - y) * (dist) - y * torch.log(1. - torch.exp(-dist))
                # losses = (1. - y) * (self.R**2 + dist) - y * torch.log(1. - torch.exp(-dist))

                num_examples_train += svdd_out.size(0)
                # print(num_examples_train)
                dist = torch.sqrt(torch.sum((svdd_out - self.c) ** 2, dim=1))
                self.R.requires_grad = False
                losses = (normal_len / total_len)*((1 - y) * ((dist - self.R) ** 2)) - (abnormal_len / total_len) * (y * torch.log(
                    1 - torch.exp(-((dist - self.R) ** 2)))) + self.lamb * torch.abs(self.R)
                # losses = (1 - y) * ((dist - self.R) ** 2) - y * torch.log(1-torch.exp(-((dist - self.R) ** 2))) + self.pert_gamma * torch.abs(self.R)
                # losses = ((1 - y) * dist - y * torch.log(1-torch.exp(-dist)))
                loss = torch.mean(losses)
                train_dist.update(torch.mean(dist).item(), X.size(0))
                train_loss.update(loss.item(), X.size(0))

                loss.backward()
                optimizer.step()

                # r loss update
                # if self.is_abnormal:
                self.R.requires_grad = True
                r_loss = (normal_len / total_len) * ((1 - y) * ((dist.data - self.R) ** 2)) - (abnormal_len / total_len) * (y * torch.log(
                    1 - torch.exp(-((dist.data - self.R) ** 2)))) + self.lamb * torch.abs(self.R)
                # r_loss = (1 - y) * torch.log((dist.data - self.R) ** 2) - y * torch.log(
                #     1 - ((dist.data - self.R) ** 2)) + self.pert_gamma * torch.abs(self.R)
                # r_loss = (1 - y) * ((dist.data - self.R) ** 2) - y * torch.log(1-torch.exp(-((dist.data - self.R) ** 2))) + self.pert_gamma * torch.abs(self.R)
                # r_loss = (1. - y) * torch.log(torch.abs(self.R - dist.data)) + y * torch.log(torch.abs(dist.data - self.R))
                r_loss = torch.mean(r_loss)

                r_loss.backward()
                opt_r.step()

                # else:
                #     self.R.data = torch.mean(dist.data)

            """
            validation step
            """
            net.eval()
            for data in validset:
                X = data[0].to(self.device)
                y = data[1].to(self.device)

                svdd_out = net(X)

                abnormal_idx_list = y.type(torch.bool)
                normal_len = len(X[~abnormal_idx_list])
                abnormal_len = len(X[abnormal_idx_list])
                total_len = normal_len + abnormal_len

                dist = torch.sqrt(torch.sum((svdd_out - self.c) ** 2, dim=1))
                # val_losses = (1 - y) * torch.log((dist - self.R) ** 2) - y * torch.log(
                #     1 - ((dist - self.R) ** 2)) + self.pert_gamma * torch.abs(self.R)
                # val_losses = (1 - y) * ((dist - self.R) ** 2) - y * torch.log(1-torch.exp(-((dist - self.R) ** 2))) + self.pert_gamma * torch.abs(self.R)
                # val_losses = ((1 - y) * dist - y * torch.log(1-torch.exp(-dist)))
                if self.val_set == 0:
                    val_losses = (normal_len / total_len)*((1 - y) * ((dist - self.R) ** 2)) - (abnormal_len / total_len) * (y * torch.log(
                    1 - torch.exp(-((dist - self.R) ** 2)))) + self.lamb * torch.abs(self.R)

                        # ((1 - y) * ((dist - self.R) ** 2) - y * torch.log(
                        # 1 - torch.exp(-((dist - self.R) ** 2)))) + self.lamb * torch.abs(self.R)
                    val_mean_loss = torch.mean(val_losses)
                else:
                    val_mean_loss = torch.mean(dist)

                valid_loss.update(val_mean_loss.item(), X.size(0))

            if self.update_c:
                self.initialize_c(net, dataset)  # update c
            # print(self.c.data)
            if self.verbose:
                print("%s Epoch %d \t| training dist: %.4f, training loss: %.8f, validation loss: %.8f, R: %.8f" % (
                    self.c_idx, epoch + 1, train_dist.avg, train_loss.avg, valid_loss.avg, self.R))

            self.train_loss_list.append(train_loss.avg)
            self.valid_loss_list.append(valid_loss.avg)
            self.R_hist.append(deepcopy(self.R.detach().cpu().numpy().tolist()))

            if valid_loss.avg < best_loss:
                if self.verbose:
                    print('best model updated')
                last_update = epoch
                best_loss = valid_loss.avg
                best_net = deepcopy(net)
                best_R = deepcopy(self.R.detach().cpu().numpy())
                best_c = deepcopy(self.c)

            scheduler.step()

            # if self.patience < epoch - last_update:
            #     print(f'Early stopping.. epoch{epoch}')
            #     print('Train duration: %.4f s'% (time.time() - start_time))
            #     break

        if self.return_best:
            return best_c, best_R, best_net, num_examples_train
        else:
            return deepcopy(self.c), self.R.detach().cpu().numpy(), deepcopy(net), num_examples_train

    def test(self, dataset: DataLoader, net: BaseNet, c=None, r=None):
        start_time = time.time()
        net.to(self.device)
        net.eval()

        scores = None
        labels = None

        num_examples_test = 0

        if c is None:
            c = self.c

        if r is None:
            r = self.R

        with torch.no_grad():
            for X, y in dataset:
                X = X.to(self.device)

                outputs = net(X)
                num_examples_test += X.size(0)

                # print(num_examples_test)
                dist = torch.sqrt(torch.sum((outputs - c) ** 2, dim=1))
                # print('test3')
                scores = concatenate(scores, dist.detach().cpu().numpy())
                # print('test4')
                labels = concatenate(labels, y.detach().cpu().numpy())
                # print('test5')

        return_type = 0
        fpr, tpr, thresholds = [], [], []

        if labels.sum() > 0:

            # fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
            # J = tpr - fpr
            # ix = np.argmax(J)
            # best_thresh = thresholds[ix]
            y_prob_pred = (scores >= r).astype(bool)

            (tp, fp, fn, tn) = get_confusion_matrix_values(labels, y_prob_pred)

            obj = classification_report(labels, y_prob_pred, output_dict=True)
            test_auc = roc_auc_score(labels, scores)

            fpr, tpr, thresholds = roc_curve(labels.ravel(), scores.ravel())

            accuracy = accuracy_score(labels, y_prob_pred)

            print('Testing duration: %.4f s' % (time.time() - start_time))
        else:
            tp, fp, fn, tn = 0, 0, 0, 0
            return_type = 1
            y_prob_pred = (scores >= r).astype(bool)
            obj = y_prob_pred.sum() / len(scores)
            test_auc = obj
            accuracy = accuracy_score(labels, y_prob_pred)

        return num_examples_test, obj, test_auc, accuracy, return_type, (tp, fp, fn, tn), fpr, tpr, thresholds


    def test_with_threshold(self, dataset: DataLoader, net: BaseNet, c=None, r=None):
        start_time = time.time()
        net.to(self.device)
        net.eval()

        scores = None
        labels = None

        num_examples_test = 0

        if c is None:
            c = self.c

        if r is None:
            r = self.R

        with torch.no_grad():
            for X, y in dataset:
                X = X.to(self.device)

                outputs = net(X)
                num_examples_test += X.size(0)

                # print(num_examples_test)
                dist = torch.sqrt(torch.sum((outputs - c) ** 2, dim=1))
                # print('test3')
                scores = concatenate(scores, dist.detach().cpu().numpy())
                # print('test4')
                labels = concatenate(labels, y.detach().cpu().numpy())
                # print('test5')

        return_type = 0
        if labels.sum() > 0:

            # fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
            # J = tpr - fpr
            # ix = np.argmax(J)
            # best_thresh = thresholds[ix]
            y_prob_pred = (scores >= r).astype(bool)
            obj = classification_report(labels, y_prob_pred, output_dict=True)
            test_auc = roc_auc_score(labels, scores)
            accuracy = accuracy_score(labels, y_prob_pred)

            fpr, tpr, thresholds = roc_curve(labels.ravel(), scores.ravel())

            print('Testing duration: %.4f s' % (time.time() - start_time))
        else:
            return_type = 1
            y_prob_pred = (scores >= r).astype(bool)
            obj = y_prob_pred.sum() / len(scores)
            test_auc = obj
            accuracy = accuracy_score(labels, y_prob_pred)

        return num_examples_test, obj, test_auc, accuracy, return_type, fpr, tpr, thresholds