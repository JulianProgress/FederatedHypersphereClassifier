#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2022/02/13 3:21 PM
# @Author    : Junhyung Kwon
# @Site      :
# @File      : HSCTrainer.py
# @Software  : PyCharm

from copy import deepcopy

import numpy as np
import torch
from base import BaseTrainer, BaseNet
from sklearn.metrics import roc_curve, classification_report, roc_auc_score
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from utils import AverageMeter, concatenate
from sklearn import metrics

import time

# from tqdm.auto import tqdm


class HSCTrainer(BaseTrainer):
    def __init__(self, optimizer_name: str, lr: float, n_epochs: int, lr_milestones: tuple,
                 weight_decay: float, device: str, verbose: bool, init_steps: int, cid: str, patience: int, include_pert: bool, gamma: float,
                 radius_thresh: float, pert_steps: int, pert_step_size: float, pert_duration: int, client_wise=False, update_c=False):
        super(HSCTrainer, self).__init__(optimizer_name, lr, n_epochs, lr_milestones, weight_decay, device)

        self.client_wise = client_wise
        self.update_c = update_c
        self.verbose = verbose
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

    def initialize_R(self, net, dataloader):
        n_samples = 0
        net.eval()




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
            encoded = net(normal_X)

            dist = torch.mean(torch.abs(encoded - self.c), dim=1)

            norm_Xs = concatenate(norm_Xs, normal_X.detach().cpu().numpy())
            radiuses = concatenate(radiuses, dist.detach().cpu().numpy())

        sorted_idxs = radiuses.argsort()
        norm_Xs = norm_Xs[sorted_idxs]
        low_confidence_X = norm_Xs[round(len(norm_Xs) * portion):]
        return torch.tensor(low_confidence_X, dtype=torch.float32), radiuses

    def personalized_r_gamma(self, radiuses):
        radiuses.sort()
        r = radiuses[round(len(radiuses)*self.pert_radius)]
        r_max = np.max(radiuses)
        r_gamma = 1 + ((r_max - r) / r)*self.pert_gamma
        return r, r_gamma


    def perturb(self, net, X, radiuses, num_gradient_steps=7, step_size=0.07): # r=0.1, gamma=1.1
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
        encoder_state = ae.encoder.state_dict()
        net.encoder.load_state_dict(encoder_state)

    def train(self, dataset: DataLoader, validset: DataLoader, net: BaseNet):
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

        best_net = None
        best_loss = 999999
        num_examples_train = 1
        last_update = 0

        train_llist = []
        valid_llist = []

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

                svdd_out = net(X)

                if epoch > self.init_steps and self.include_pert:
                    pert_in = data[1][0].to(self.device)
                    pert_y = data[1][1].to(self.device)
                    # pert_svdd_out = net.svdd_mapping(pert_in)
                    # svdd_out = torch.cat((svdd_out, pert_svdd_out))
                    svdd_out = torch.cat((svdd_out, pert_in))
                    y = torch.cat((y, pert_y))

                """
                HSC objective
                """
                # if i == 0:
                #     print('distances', (svdd_out - self.c) ** 2)

                num_examples_train += svdd_out.size(0)
                # print(num_examples_train)

                dist = torch.sum((svdd_out - self.c) ** 2, dim=1)
                losses = (1. - y) * dist - y * torch.log(1. - torch.exp(-dist))
                loss = torch.mean(losses)

                train_dist.update(torch.mean(dist).item(), X.size(0))
                train_loss.update(loss.item(), X.size(0))

                loss.backward()
                optimizer.step()


                # r loss
                # r_loss = y * torch.log()

            # print(net.encoder.encoder[0].weight)

            # radius = self.get_radius(net, dataset)

            """
            validation step
            """
            net.eval()
            for data in validset:
                X = data[0].to(self.device)
                y = data[1].to(self.device)

                svdd_out = net(X)
                dist = torch.sum((svdd_out - self.c) ** 2, dim=1)
                val_losses = (1. - y) * dist - y * torch.log(1 - torch.exp(-dist))
                val_mean_loss = torch.mean(val_losses)

                valid_loss.update(val_mean_loss.item(), X.size(0))


            if self.update_c:
                self.initialize_c(net, dataset)  # update c
            # print(self.c.data)
            if self.verbose:
                print("%s Epoch %d \t| training dist: %.4f, training loss: %.8f, validation loss: %.8f" % (
                    self.c_idx, epoch + 1, train_dist.avg, train_loss.avg, valid_loss.avg))

            train_llist.append(train_loss.avg)
            valid_llist.append(valid_loss.avg)

            if valid_loss.avg < best_loss:
                if self.verbose:
                    print('best model updated')
                last_update = epoch
                best_loss = valid_loss.avg
                best_net = deepcopy(net)

            scheduler.step()

            if self.patience < epoch - last_update:
                print(f'Early stopping.. epoch{epoch}')
                print('Train duration: %.4f s'% (time.time() - start_time))
                break


        return self.c, best_net, num_examples_train, train_llist, valid_llist

    def test(self, dataset: DataLoader, net: BaseNet, c=None, global_thresh=None):
        start_time = time.time()
        net.to(self.device)
        net.eval()

        scores = None
        labels = None

        num_examples_test = 0

        # print('test1')

        if c is None:
            c = self.c
        #
        # print('test2')

        with torch.no_grad():
            for X, y in dataset:
                X = X.to(self.device)

                outputs = net(X)
                num_examples_test += X.size(0)

                # print(num_examples_test)
                dist = torch.sum((outputs - c) ** 2, dim=1)
                # print('test3')
                scores = concatenate(scores, dist.detach().cpu().numpy())
                # print('test4')
                labels = concatenate(labels, y.detach().cpu().numpy())
                # print('test5')

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

    def test_threshold(self, val_dataset: DataLoader, test_dataset: DataLoader, net: BaseNet):
        scores = None
        labels = None

        net.eval()

        num_examples_test = 0

        for data in val_dataset:
            X = data[0].to(self.device)
            y = data[1].to(self.device)

            outputs = net(X)
            num_examples_test += X.size(0)

            probs = torch.nn.functional.sigmoid(outputs.squeeze())

            scores = concatenate(scores, probs.detach().cpu().numpy())
            labels = concatenate(labels, y.detach().cpu().numpy())

        # print(label_list.dtype, score_list.dtype)
        fpr, tpr, thresholds = roc_curve(labels.ravel(), scores.ravel())
        # print(fpr.dtype, tpr.dtype)
        roc_auc = metrics.auc(fpr, tpr)

        J = tpr - fpr
        ix = np.argmax(J)
        best_thresh = thresholds[ix]

        test_labels = None
        test_scores = None

        for X, y in test_dataset:
            X = X.to(self.device)

            outputs = net(X)
            num_examples_test += X.size(0)

            probs = torch.nn.functional.sigmoid(outputs.squeeze())

            test_scores = concatenate(test_scores, probs.detach().cpu().numpy())
            test_labels = concatenate(test_labels, y.detach().cpu().numpy())

        test_pred_list = np.zeros_like(test_scores)
        test_pred_list[test_scores > best_thresh] = 1

        test_result = classification_report(test_labels, test_pred_list, output_dict=True)

        return tpr, fpr, roc_auc, test_result, best_thresh