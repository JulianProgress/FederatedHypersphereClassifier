#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2022/02/13 3:33 PM
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : client.py
# @Software  : PyCharm

import os
import pickle
import sys
import torch
import flwr as fl
from collections import OrderedDict
from pathlib import Path

from optimizer import HSCTrainer, AETrainer
from base import BaseDataset


class FedHSCClient(fl.client.Client):
    def __init__(self, rnd, cid, model, trainer_params, train_loader, valid_loader, test_loader, device, flag):
        self.cid = cid
        self.trainer_params = trainer_params
        self.net = model.to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.device = device

        self.flag = flag
        self.rnd = rnd

        self.trainer = None
        self.c = None

        self.trainer_params['device'] = self.device

    # @TODO implementation
    def _load_center(self):

        save_dir_path = self.trainer_params["center_save_dir_path"] if "center_save_dir_path" in self.trainer_params else "/workspace/code/FedHSC/save/center"
        Path(save_dir_path).mkdir(exist_ok=True)
        center_file_path = os.path.join(save_dir_path, f"global.pkl")
        with open(center_file_path, "rb") as f:
            global_center = pickle.load(f)
            print(f"Loaded global center, path : {center_file_path}")

        self.c = global_center

    def _save_center(self):
        save_dir_path = self.trainer_params["center_save_dir_path"] if "center_save_dir_path" in self.trainer_params else "/workspace/code/FedHSC/save/center"
        Path(save_dir_path).mkdir(exist_ok=True)

        center_file_path = os.path.join(save_dir_path, f"client_{self.cid}.pkl")

        with open(center_file_path, "wb") as f:
            pickle.dump(self.c, f)
            print(f"Complete to save center, path : {center_file_path}")

    def pretraining(self, ae_params):
        self.net.to(self.device)
        self.pre_trainer = AETrainer(**ae_params)
        self.ae_net, num_examples_train = self.pre_trainer.train(self.train_loader, self.valid_loader, self.net)
        val_loss = self.pre_trainer.test(self.valid_loader, self.ae_net)

        return num_examples_train, val_loss.avg


    def training(self, descriptor_params):
        self.net.to(self.device)
        self.trainer = HSCTrainer(**descriptor_params)

        self.trainer.c = self.c

        # Get results
        self.c, self.net, num_examples_train = self.trainer.train(self.train_loader, self.valid_loader, net=self.net)
        # report, test_auc = self.trainer.test(self.test_loader, self.net, c=self.c)
        return num_examples_train

    def test(self, descriptor_params):
        self.net.to(self.device)
        self.trainer = HSCTrainer(**descriptor_params)
        # self.c, self.net, num_examples_train = self.trainer.train(self.train_loader, self.valid_loader, net=self.net)
        try:
            print('c: ', self.c)
            num_examples_train, report, test_auc = self.trainer.test(self.test_loader, self.net, c=self.c)
        except Exception as e:
            print('6', e)
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
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, ins):

        if self.rnd == 0:
            self._load_center()


        num_examples_train = 1
        try:
            print('init weights..')
            # Get weights
            weights = fl.common.parameters_to_weights(ins.parameters)

            # print(ins.parameters)

            # Set model parameters/weights
            self.set_parameters(weights)
        except Exception as e:
            print('5', e)

        if not self.flag: # pretraining
            try:
                num_examples_train, loss = self.pretraining(ae_params=self.trainer_params)
                print("Pretrain client {} result: {}.4f".format(self.cid, loss))
            except Exception as e:
                print('4', e)
        else: # training HSC
            try:
                self.trainer_params['c_idx'] = self.cid
                num_examples_train = self.training(self.trainer_params)

            except Exception as e:
                print('3', e)

        # Return the refined weights and the number of examples used for training
        try:
            weights_prime = self.get_weights()
            params_prime = fl.common.weights_to_parameters(weights_prime)

            print('param get done')

            res = fl.common.FitRes(
                parameters=params_prime,
                num_examples=num_examples_train,
            )

            print('res done')
        except Exception as e:
            print('2', e)

        # Save center to file
        self._save_center()

        return res

    def evaluate(self, ins):
        # Get weights
        weights = fl.common.parameters_to_weights(ins.parameters)

        # load global center
        self._load_center()

        # Set model parameters/weights
        self.set_parameters(weights)

        if self.flag:
            self.trainer_params['c_idx'] = self.cid
            num_examples_train, report, test_auc = self.test(self.trainer_params)
            torch.save(
                dict(
                    report=report,
                    auc=test_auc
                ),
                f"/workspace/code/FedHSC/model_save/fl/test_result_{self.cid}.pt"
            )

            torch.save(
                self.net.cpu().state_dict(),
                f"/workspace/code/FedHSC/model_save/fl/global_svdd.pt"
            )
            # try:
            #
            # except Exception as e:
            #     print('1', e)


        return fl.common.EvaluateRes(
            loss=float(test_auc),
            num_examples=num_examples_train,
            metrics={"metrics": float(test_auc)}
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
