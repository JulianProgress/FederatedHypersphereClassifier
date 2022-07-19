#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2022/02/13 3:33 PM
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : client.py
# @Software  : PyCharm

import sys
import torch
import flwr as fl
from collections import OrderedDict

from optimizer import HSCTrainer, AETrainer
from base import BaseDataset


ae_params = {
    'optimizer_name': 'Adam',
    'lr': 0.001,
    'n_epochs': 10,
    'lr_milestones': [20, 40],
    'weight_decay': 1e-6,
    'device': 'cpu'
}

descriptor_params = {
    'optimizer_name': 'Adam',
    'lr': 0.0001,
    'n_epochs': 10,
    'lr_milestones': [20, 40],
    'weight_decay': 0,
    'device': 'cpu',
    'init_steps': 1,
    'c_idx': 0,
    'gamma': 1.1,
    'radius': 0.5,
    'pert_steps': 20,
    'pert_step_size': 0.01,
    'pert_duration': 4
}


class FedHSCClient(fl.client.Client):
    def __init__(self, cid, model, model_params, train_loader, valid_loader, test_loader, device, flag):
        self.cid = cid
        self.model_params = model_params
        self.net = model.to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.device = device

        self.flag = flag

        self.trainer = None
        self.c = None

        self.model_params['device'] = self.device

    def pretraining(self, ae_params):
        self.pre_trainer = AETrainer(**ae_params)
        self.ae_net, num_examples_train = self.pre_trainer.train(self.train_loader, self.valid_loader, self.net)
        val_loss = self.pre_trainer.test(self.valid_loader, self.ae_net)
        return num_examples_train, val_loss.avg


    def training(self, descriptor_params):
        self.trainer = HSCTrainer(**descriptor_params)

        # Get results
        self.c, self.net, num_examples_train = self.trainer.train(self.train_loader, self.valid_loader, net=self.net)
        report, test_auc = self.trainer.test(self.test_loader, self.net, c=self.c)
        return num_examples_train, report, test_auc

    def init_network_weights_from_pretraining(self):
        """Initialize the Deep SVDD network weights from the encoder weights of the pretraining autoencoder."""
        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict()

        # Filter out decoder network keys
        ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}
        # Overwrite values in the existing state_dict
        net_dict.update(ae_net_dict)
        # Load the new state_dict
        self.net.load_state_dict(net_dict)

    def get_weights(self):
        # Return model parameters as a list of NumPy ndarrays
        return [values.cpu().numpy() for _, values in self.net.state_dict().items()]

    def get_parameters(self):
        weights = self.get_weights()
        parameters = fl.common.weights_to_parameters(weights)
        return fl.common.ParametersRes(parameters=parameters)

    def set_parameters(self, parameters):
        # Set model parameters from a list of NumPy ndarrays
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, ins):
        # Get weights
        weights = fl.common.parameters_to_weights(ins.parameters)

        # Set model parameters/weights
        self.set_parameters(weights)

        try:
            if not self.flag: # pretraining
                num_examples_train, loss = self.pretraining(ae_params=self.model_params)
                print("Pretrain client {} result: {.4f}".format(self.cid, loss))
            else: # training HSC
                num_examples_train, report, test_auc = self.training(self.model_params)
        except Exception as e:
            print(e)

        # Train model
        # num_examples_train, loss = util.train(self.model, self.train_loader, epochs=self.epochs, device=self.device,
        #                                       flag=self.flag)
        # sys.stdout.write(f"[CLIENT {self.cid.zfill(4)}] Training Loss: {loss:8.4f}" + "\r")
        # Return the refined weights and the number of examples used for training
        weights_prime = self.get_weights()
        params_prime = fl.common.weights_to_parameters(weights_prime)
        # print(fl.common.Parameters())

        return fl.common.FitRes(
            parameters=params_prime,
            num_examples=num_examples_train,
        )

    def evaluate(self, ins):
        # TODO


        return fl.common.EvaluateRes(
            loss=float(0),
            num_examples=0,
            metrics={"metrics": float(0)}
        )


        # Get weights
        # weights = fl.common.parameters_to_weights(ins.parameters)
        #
        # # Set model parameters/weights
        # self.set_parameters(weights)
        #
        # # Test model
        # loss, num_examples_test, metric = util.test(self.model, self.test_loader, device=self.device, flag=self.flag)
        # sys.stdout.write(f"[CLIENT {self.cid.zfill(4)}] Evaluation Loss: {loss:8.4f} | Metric: {metric:8.4f}" + "\r")
        # return fl.common.EvaluateRes(
        #     loss=float(loss),
        #     num_examples=num_examples_test,
        #     metrics={"metrics": float(metric)}
        # )
