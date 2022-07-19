#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2022/02/24 13:34
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : fl_trainer.py
# @Software  : PyCharm

from abc import ABC, abstractmethod

class BaseFLTrainer(ABC):
    def __init__(self, num_clients: int, num_rounds: int, trainer_params, model_params, descriptor_params,
                 cuda_num_list, abnormal_in_val=False, rep_model='RNNAE', eval_device='cuda:0'):
        super().__init__()

        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.trainer_params = trainer_params
        self.model_params = model_params
        self.descriptor_params = descriptor_params
        self.cuda_num_list = cuda_num_list
        self.abnormal_in_val = abnormal_in_val
        self.rep_model = rep_model
        self.eval_device = eval_device

    @abstractmethod
    def aggregate(self, results):
        """
        Implement aggregation method for federated learning
        :param results: parameters
        :return:
        """
        pass

    @abstractmethod
    def round(self, rnd, trainset, validset, testset, host_list, **kwargs):
        """

        :param rnd: number rounds
        :param trainset: train sets per clients
        :param validset: valid sets per clients
        :param testset: test sets per clients
        :param host_list: list of hosts
        :param kwargs: additional arguments (is aug, etc)
        """
        pass

    @abstractmethod
    def evaluate(self, testset, hostid, device, **kwargs):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def resume_round(self, rnd=0):
        pass

