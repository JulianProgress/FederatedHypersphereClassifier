#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2022/02/21 11:49 PM
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : FedAvg.py
# @Software  : PyCharm

import os
import time
from abc import ABC
from collections import OrderedDict
from functools import reduce
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from datasets import ADDRepDataset
from model import SVDD
from optimizer import HSCTrainer
from torch.utils.data import DataLoader


class FederatedTraining(ABC):
    def __init__(self, num_clients: int, num_rounds: int, trainer_params, model_params, descriptor_params):
        super(FederatedTraining, self).__init__()
        self.trainer_params = trainer_params

        self.model_params = model_params

        self.descriptor_params = descriptor_params

        self.num_clients = num_clients
        self.num_rounds = num_rounds

        self.current_rnd = 0
        self.g_weights = None
        self.g_c = None
        self.keys = None
        self.initial_dict = None

        self._prepare_fit()

    def reset(self):
        self.current_rnd = 0
        self.g_weights = None
        self.g_c = None
        self.keys = None
        self.initial_dict = None

        self._prepare_fit()

    def weight_to_state_dict(self, weights):
        params_dict = zip(self.keys, weights)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        return state_dict

    def state_dict_to_weight(self, state_dict):
        return [values.cpu().numpy() for _, values in state_dict.items()]

    def _prepare_fit(self):
        net = SVDD(**self.model_params)
        self.keys = net.state_dict().keys()
        self.initial_dict = self.state_dict_to_weight(net.state_dict())

    def aggregate(self, results) -> (list, list):
        # calculate total example numbers
        num_examples_total = sum([r['num_train_ex'] for r in results])
        # calculate weighted weight parameters
        weights = [(r['c'], r['state_dict'], r['num_train_ex']) for r in results]
        weighted_weights = [
            [layer * num_ex for layer in weights] for _, weights, num_ex in weights
        ]
        # calculate weighted center c
        weighted_c = [
            c * num_ex for c, _, num_ex in weights
        ]
        # Avg total weights
        weights_prime = [
            reduce(np.add, layer_updates) / num_examples_total for layer_updates in zip(*weighted_weights)
        ]
        # find global center
        c_prime = [
            reduce(np.add, c_updates) / num_examples_total for c_updates in zip(*weighted_c)
        ]
        return weights_prime, c_prime

    def get_save_path(self, rnd):
        self.save_dir_path = self.trainer_params[
            "model_save_path"] if "model_save_path" in self.trainer_params else "../model_save/fl"
        Path(self.save_dir_path).mkdir(exist_ok=True)
        return os.path.join(self.save_dir_path,
                            f"aug{int(self.descriptor_params['include_pert'])}_l{self.model_params['num_rep_layers']}_s{self.model_params['num_svdd_layer']}_{rnd}.pt")

    def client(self, args):
        net_state_dict, g_c, trainset, validset, testset, hostid, device, kwargs, model_params, trainer_param = args[0], \
                                                                                                                args[1], \
                                                                                                                args[2], \
                                                                                                                args[3], \
                                                                                                                args[4], \
                                                                                                                args[5], \
                                                                                                                args[6], \
                                                                                                                args[7], \
                                                                                                                args[8], \
                                                                                                                args[9]

        trainl = DataLoader(trainset, **kwargs['trainloader'])
        testl = DataLoader(testset, **kwargs['testloader'])
        validl = DataLoader(validset, **kwargs['valloader'])

        print(f'hostid : {hostid}, device: {device}')

        net = SVDD(**model_params)
        global_state_dict = self.weight_to_state_dict(net_state_dict)
        net.load_state_dict(global_state_dict)  # load global state dict

        descriptor_params = self.descriptor_params

        descriptor_params['device'] = device
        # @TODO load global network

        trainer = HSCTrainer(**self.descriptor_params)
        if not g_c is None:
            trainer.c = torch.tensor(g_c)

        c, net, num_train = trainer.train(trainl, validl, net)
        num_test, obj, test_auc, return_type = trainer.test(testl, net,
                                                            global_thresh=0.5)  # return type = 1: normal acc

        return dict(
            state_dict=self.state_dict_to_weight(net.state_dict()),
            c=c.detach().cpu().numpy(),
            report=obj,
            auc=test_auc,
            return_type=return_type,
            num_train_ex=num_train,
            hostid=hostid
        )

    def round(self, rnd):
        start = time.time()
        dataset = ADDRepDataset(0.2, 0.2)

        trainset, validset, testset = dataset.create_datasets()
        host_list = dataset.window_hostid_list

        device = ['cuda:4', 'cuda:4', 'cuda:4', 'cuda:5', 'cuda:5', 'cuda:4', 'cuda:6', 'cuda:6']

        p = Pool(self.num_clients)

        if rnd == 0:  # initialize state dict
            state_dict = self.initial_dict
            global_center = None

        else:  # get from global state dict
            state_dict = self.g_weights
            global_center = self.g_c

        results = p.map(self.client,
                        zip(repeat(state_dict), repeat(global_center), trainset, validset, testset, host_list, device,
                            repeat(self.trainer_params), repeat(self.model_params), repeat(self.descriptor_params)))

        # aggregation
        global_weight, global_c = self.aggregate(results)

        print(f"Round {rnd} time: {time.time() - start} s")

        p.close()
        p.join()

        return global_weight, np.array(global_c)

    def fit(self):
        print('start training..')
        for rnd in range(self.num_rounds):
            self.current_rnd = rnd
            g_weights, g_c = self.round(rnd)
            self.g_weights = g_weights
            self.g_c = g_c

            # @TODO: Save check points

            torch.save(dict(
                rnd=self.current_rnd,
                state_dict=self.g_weights,
                c=self.g_c
            ), self.get_save_path(rnd))

        # pass


if __name__ == '__main__':
    trainer_params = {
        'trainloader': dict(
            batch_size=128,
            shuffle=True,
            drop_last=True,
            num_workers=0
        ),
        'testloader': dict(
            batch_size=128,
            shuffle=False,
            drop_last=False,
            num_workers=0
        ),
        'valloader': dict(
            batch_size=128,
            shuffle=False,
            drop_last=False,
            num_workers=0
        ),
        'model_save_path': '/workspace/code/FedHSC/model_save/fl'
    }

    model_params = {
        'rep_model_type': 'RNNEncoder',
        'in_dim': 32,
        'hidden_dims': [16, 256],  # hidden_dim, rep_dim
        'window_size': 128,
        'num_svdd_layer': 3,
        'num_rep_layer': 3
    }

    descriptor_params = {'optimizer_name': 'Adam', 'lr': 1e-5, 'n_epochs': 2, 'lr_milestones': [20, 40],
                         'weight_decay': 0, 'device': 'cpu', 'init_steps': 2, 'c_idx': 0, 'patience': 10,
                         'gamma': 1.1, 'radius': 0.1, 'pert_steps': 10, 'pert_step_size': 0.001,
                         'pert_duration': 3,
                         'include_pert': True}

    fl_trainer = FederatedTraining(8, 2, trainer_params, model_params, descriptor_params)
    fl_trainer.fit()