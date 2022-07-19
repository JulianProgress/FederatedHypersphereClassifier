#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2022/02/22 10:55
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
from torch.utils.data import DataLoader

from ..datasets import ADDRepDataset
from ..model import SVDD
from ..optimizer import HSCTrainer
from .fl_utils import weight_to_state_dict, state_dict_to_weight


class FederatedTraining(ABC):
    def __init__(self, num_clients: int, num_rounds: int, trainer_params, model_params, descriptor_params,
                 cuda_num_list, abnormal_in_val=False, rep_model='RNNAE', eval_device='cuda:0', personalization=True):
        super(FederatedTraining, self).__init__()
        self.abnormal_in_val = abnormal_in_val
        self.rep_model = rep_model
        self.cuda_num_list = cuda_num_list
        self.trainer_params = trainer_params
        self.personalization = personalization

        self.model_params = model_params
        self.eval_device = eval_device
        self.descriptor_params = descriptor_params

        self.num_clients = num_clients
        self.num_rounds = num_rounds

        self.current_rnd = 0
        self.g_weights = None
        self.g_c = None
        self.keys = None
        self.initial_dict = None
        self.c_list = []

        self.train_lr_curve_round = []
        self.valid_lr_curve_round = []
        self.train_lr_curve = []
        self.valid_lr_curve = []

        self._prepare_fit()

    def reset(self):
        self.current_rnd = 0
        self.g_weights = None
        self.g_c = None
        self.keys = None
        self.initial_dict = None

        self._prepare_fit()

    def get_global(self):
        return self.weight_to_state_dict(self.g_weights), self.g_c

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
                            f"Fed2_rep{self.rep_model}_ps{self.descriptor_params['pert_steps']}_pd{self.descriptor_params['pert_duration']}_aug{int(self.descriptor_params['include_pert'])}_l{self.model_params['num_rep_layer']}_s{self.model_params['num_svdd_layer']}_{rnd}.pt")

    def round(self, rnd, trainset, validset, testset, host_list, is_aug):
        start = time.time()
        self.descriptor_params['include_pert'] = False

        device = ['cuda:%d' % d for d in self.cuda_num_list]

        p = Pool(self.num_clients)

        if rnd == 0:  # initialize state dict
            state_dict = self.initial_dict
            global_center = None

        else:  # get from global state dict
            state_dict = self.g_weights
            global_center = None
        # g_c, trainset, validset, testset, hostid, device, trainer_params, model_params, descriptor_params, current_rnd, weights
        # print(state_dict)

        if is_aug and rnd > self.descriptor_params['init_steps'] - 1:
            self.descriptor_params['include_pert'] = True

        results = p.map(client,
                        zip(repeat(global_center),
                            trainset,
                            validset,
                            testset,
                            host_list,
                            device,
                            repeat(self.trainer_params),
                            repeat(self.model_params),
                            repeat(self.descriptor_params),
                            repeat(self.current_rnd),
                            repeat(state_dict))
                        )

        # repeat(trainer_params), repeat(model_params), repeat(descriptor_params)
        # aggregation
        global_weight, global_c = self.aggregate(results)

        train_list = [res['train_list'] for res in results]
        valid_list = [res['valid_list'] for res in results]

        c_list = [res['c'] for res in results]

        print(f"Round {rnd} time: {time.time() - start} s")

        p.close()
        p.join()

        return global_weight, np.array(global_c), train_list, valid_list, c_list

    # def eval_total_tests(self):


    def eval(self):
        dataset = ADDRepDataset(0.2, 0.2, model_name=self.rep_model, abnormal_in_val=self.abnormal_in_val)
        trainset, validset, testset = dataset.create_datasets()
        host_list = dataset.window_hostid_list

        abnormal_test_num = np.array(dataset.abnormal_test)
        test_with_abnormal = np.where(abnormal_test_num > 0)[0]

        local_auc = []
        local_f1 = []
        local_acc = []
        for i in test_with_abnormal:
            num_test, report, auc, _ = self.evaluate(testset[i], host_list[i], device=self.eval_device,
                                                     weight=self.g_weights, c=self.c_list[i])
            local_auc.append(auc)
            local_f1.append(report['1.0']['f1-score'])
            local_acc.append(report['accuracy'])

        print(f"Local results \t| auc: {local_auc}, f1: {local_f1}, acc: {local_acc}")
        print(
            f"Total results \t| auc: {np.mean(local_auc)} {np.std(local_auc)}, f1: {np.mean(local_f1)} {np.std(local_f1)}, acc: {np.mean(local_acc)} {np.std(local_acc)}")

        return local_auc, local_f1, local_acc

    def fit(self):
        print('start training..')
        start = time.time()
        dataset = ADDRepDataset(0.2, 0.2, model_name=self.rep_model, abnormal_in_val=self.abnormal_in_val)
        trainset, validset, testset = dataset.create_datasets()
        host_list = dataset.window_hostid_list

        abnormal_test_num = np.array(dataset.abnormal_test)
        test_with_abnormal = np.where(abnormal_test_num > 0)[0]

        is_aug = self.descriptor_params['include_pert']

        for rnd in range(self.num_rounds):
            self.current_rnd = rnd
            g_weights, g_c, train_list, valid_list, c_list = self.round(rnd, trainset, validset, testset, host_list,
                                        is_aug)
            # local_auc, local_f1, local_acc
            self.g_weights = g_weights
            self.g_c = g_c
            self.c_list = c_list
            self.train_lr_curve.append(train_list)
            self.valid_lr_curve.append(valid_list)
            self.train_lr_curve_round.append([np.mean(tl) for tl in train_list])
            self.valid_lr_curve_round.append([np.mean(vl) for vl in valid_list])

            torch.save(dict(
                rnd=self.current_rnd,
                state_dict=self.g_weights,
                train_lr_curve=self.train_lr_curve,
                valid_lr_curve=self.valid_lr_curve,
                train_lr_curve_round=self.train_lr_curve_round,
                valid_lr_curve_round=self.valid_lr_curve_round,
                c=self.g_c,
                c_list=c_list,
                trainer_params=self.descriptor_params
            ), self.get_save_path(rnd))

        print('total training time: ', time.time() - start)


    def resume_round(self, rnd=0):
        file_path = self.get_save_path(rnd)
        res = torch.load(file_path)
        self.current_rnd = rnd
        self.g_weights = res['state_dict']
        self.g_c = res['c']
        self.c_list = res['c_list']

    def evaluate(self, testset, hostid, device, weight=None, c=None):
        print('Evaluation..')
        # print(self.g_weights)
        testl = DataLoader(testset, **self.trainer_params['testloader'])

        net = SVDD(**self.model_params)

        if not weight is None:
            global_state_dict = self.weight_to_state_dict(weight)  # init global network
        else:
            global_state_dict = self.weight_to_state_dict(self.g_weights)  # init global network
        net.load_state_dict(global_state_dict)  # load global state dict

        self.descriptor_params['device'] = device
        self.descriptor_params['cid'] = hostid

        trainer = HSCTrainer(**self.descriptor_params)

        if not c is None:
            trainer.c = torch.tensor(c).to(device)
        else:
            trainer.c = torch.tensor(self.g_c).to(device)

        num_test, obj, test_auc, return_type = trainer.test(testl, net,
                                                            global_thresh=0.5)  # return type = 1: normal acc

        return num_test, obj, test_auc, return_type

def client(args):
    g_c, trainset, validset, testset, hostid, device, trainer_params, model_params, descriptor_params, current_rnd, weights = \
        args[0], \
        args[1], \
        args[2], \
        args[3], \
        args[4], \
        args[5], \
        args[6], \
        args[7], \
        args[8], \
        args[9], \
        args[10]

    trainl = DataLoader(trainset, **trainer_params['trainloader'])
    validl = DataLoader(validset, **trainer_params['valloader'])

    print(f'hostid : {hostid}, device: {device}')

    net = SVDD(**model_params)
    keys = net.state_dict().keys()
    global_state_dict = weight_to_state_dict(keys, weights)
    net.load_state_dict(global_state_dict)  # load global state dict

    descriptor_params['device'] = device
    descriptor_params['cid'] = hostid
    if current_rnd > 0:
        descriptor_params['init_steps'] = 0

    trainer = HSCTrainer(**descriptor_params)

    # update c with global center
    if not g_c is None:
        trainer.c = torch.tensor(g_c).to(device)

    c, net, num_train, train_llist, valid_llist = trainer.train(trainl, validl, net)
    # num_test, obj, test_auc, return_type = trainer.test(testl, net,
    #                                                     global_thresh=0.5)  # return type = 1: normal acc

    res = dict(
        state_dict=state_dict_to_weight(net.state_dict()),
        c=c.detach().cpu().numpy(),
        train_list=train_llist,
        valid_list=valid_llist,
        # report=obj,
        # auc=test_auc,
        # return_type=return_type,
        # num_test_ex=num_test,
        num_train_ex=num_train,
        hostid=hostid
    )

    return res


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
