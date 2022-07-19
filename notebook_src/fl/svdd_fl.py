#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2022/02/24 13:42
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : svdd_fl.py
# @Software  : PyCharm

from notebook_src.base import BaseFLTrainer
from notebook_src.model import RNNAE
from notebook_src.fl import weight_to_state_dict, state_dict_to_weight
from functools import reduce

import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np


class SVDDFedAvg(BaseFLTrainer):
    def __init__(self, num_clients: int, num_rounds: int, trainer_params, model_params, descriptor_params,
                 cuda_num_list, abnormal_in_val=False, rep_model='RNNAE', eval_device='cuda:0'):
        super(SVDDFedAvg, self).__init__(num_clients, num_rounds, trainer_params, model_params, descriptor_params,
                                         cuda_num_list, abnormal_in_val, rep_model, eval_device)

        self.g_weights = None # global weight
        self.g_c = None # global center
        self.initial_dict = None # initial global state_dict weight
        self.keys = None

        self._prepare_fit()

    def get_global(self):
        return weight_to_state_dict(self.g_weights), self.g_c

    def _prepare_fit(self):
        net = RNNAE(**self.model_params)
        self.keys = net.state_dict().keys()
        self.initial_dict = state_dict_to_weight(net.state_dict())

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
                            f"Fed_rep{self.rep_model}_ps{self.descriptor_params['pert_steps']}_pd{self.descriptor_params['pert_duration']}_aug{int(self.descriptor_params['include_pert'])}_l{self.model_params['num_rep_layer']}_s{self.model_params['num_svdd_layer']}_{rnd}.pt")



