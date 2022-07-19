#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2022/02/13 21:31
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : server.py
# @Software  : PyCharm

import argparse
import os
import pickle
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple

import flwr as fl
import numpy as np
import torch
from torch.utils.data import DataLoader

from client import FedHSCClient
from model import AutoEncoder, SVDD
from utils import set_random_seed
from datasets import nBaIoTDataset


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
            self,
            rnd: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[BaseException],
    ) -> Optional[fl.common.Weights]:
        weights = super().aggregate_fit(rnd, results, failures)
        if weights is not None:
            # Save weights
            print(f"Saving round {rnd} weights...")
            torch.save(weights, f'../model_save/fl/test_weight_{rnd}.pt')
            np.savez(f"../model_save/fl/ae_round-{rnd}-weights.npz", *weights)
        return weights

class SaveModelStrategySVDD(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        self.strategy_params = kwargs.pop("strategy_params")
        print(kwargs.keys())
        super().__init__(*args, **kwargs)

    def _load_client_center(self):
        center_dict = dict()
        # save_dir_path = '/workspace/code/FedHSC/save/center'
        save_dir_path = self.strategy_params["center_save_dir_path"] if "center_save_dir_path" in self.strategy_params else "../save/center"

        client_center_file_path_list = [os.path.join(save_dir_path, center_file_name) for center_file_name in os.listdir(save_dir_path) if "client" in center_file_name]

        # Load centers of clients
        for client_center_file_path in client_center_file_path_list:
            client_idx = int(os.path.basename(client_center_file_path).split(".")[0].split("_")[1:][0])
            with open(client_center_file_path, "rb") as f:
                center_dict[client_idx] = pickle.load(f)

        return center_dict

    def _calculate_global_center(self, center_dict):
        global_center = None

        for client_idx, center in center_dict.items():
            if not global_center:
                global_center = center
            else:
                global_center += center

        global_center /= len(center_dict)

        return global_center

    def _save_center(self, global_center):
        # save_dir_path = '/workspace/code/FedHSC/save/center'
        save_dir_path = self.strategy_params["center_save_dir_path"] if "center_save_dir_path" in self.strategy_params else "../save/center"
        Path(save_dir_path).mkdir(exist_ok=True)

        center_file_path = os.path.join(save_dir_path, f"global.pkl")

        with open(center_file_path, "wb") as f:
            pickle.dump(global_center, f)
            print(f"Complete to save global center, path : {center_file_path}")

    def aggregate_fit(
            self,
            rnd: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[BaseException],
    ) -> Optional[fl.common.Weights]:

        # Load centers of clients
        center_dict = self._load_client_center()

        # Calculate the global center
        global_center = self._calculate_global_center(center_dict=center_dict)

        # Save global center
        self._save_center(global_center=global_center)

        weights = super().aggregate_fit(rnd, results, failures)
        if weights is not None:
            # Save weights
            print(f"Saving round {rnd} weights...")
            torch.save(weights, '../model_save/fl/test_weight_{rnd}.pt')
            np.savez(f"../model_save/fl/svdd_round-{rnd}-weights.npz", *weights)
        return weights


def client_fn(rnd, cid, net, net_params, net_vars, trainset, validset, testset, batch_size, device, flag):

    # try:
    print(net_vars)
    if not flag:
        net = net(in_dim=115, hidden_dims=[128, 64, 32])
    else:
        net = net(**net_vars)

    shuffle_train = True
    shuffle_test = False
    pin_memory = False
    num_workers = 16

    train_loader = DataLoader(dataset=trainset[int(cid)], batch_size=batch_size, shuffle=shuffle_train,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    valid_loader = DataLoader(dataset=validset[int(cid)], batch_size=batch_size, shuffle=shuffle_test,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    test_loader = DataLoader(dataset=testset[int(cid)], batch_size=batch_size, shuffle=shuffle_test,
                             num_workers=num_workers, pin_memory=pin_memory, drop_last=False)

    print(len(train_loader))

    # cid, model, model_params, train_loader, valid_loader, test_loader, device, flag
    client = FedHSCClient(rnd, cid, net, net_params, train_loader, valid_loader, test_loader, device, flag)
    return client


def get_parameters(model):
    return [values.cpu().numpy() for _, values in model.state_dict().items()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Federated Learning Simulation based on Flower")
    parser.add_argument('--seed', type=int, default=5959, help='random seed')
    parser.add_argument('--num_clients', '-N', type=int, default=9, help='number of clients')
    parser.add_argument('--fraction', '-K', type=float, default=1.,
                        help='fraction of participating clients at each round')
    parser.add_argument('--batch_size', '-B', type=int, default=128, help='batch size for local update')
    parser.add_argument('--num_epochs', '-E', type=int, default=5, help='number of local epochs')
    parser.add_argument('--num_pretrain_epochs', '-PE', type=int, default=5, help='number of local pretrain epochs')
    parser.add_argument('--num_rounds', '-R', type=int, default=10, help='number of required rounds')
    parser.add_argument('--num_pretrain_rounds', '-PR', type=int, default=5, help='number of required pretrain rounds')
    parser.add_argument('--weight_decay', '-W', type=float, default=1e-6, help='weight decay')
    parser.add_argument('--file_name', '-F', type=str, default='nBaIoT.pkl', help='data file name')
    args = parser.parse_args()

    strategy_params = {
        'center_save_dir_path': '/workspace/code/FedHSC/save/center'
    }

    ae_params = {
        'optimizer_name': 'Adam',
        'lr': 0.001,
        'n_epochs': args.num_pretrain_epochs,
        'lr_milestones': [20, 40],
        'weight_decay': 1e-6,
        'device': 'cpu'
    }

    descriptor_params = {
        'center_save_dir_path': '/workspace/code/FedHSC/save/center',
        'optimizer_name': 'Adam',
        'lr': 1e-5,
        'n_epochs': args.num_epochs,
        'lr_milestones': [20, 40],
        'weight_decay': 0,
        'device': 'cpu',
        'init_steps': 1,
        'c_idx': 0,
        'include_pert': True,
        'gamma': 1.1,
        'radius': 0.7,
        'pert_steps': 20,
        'pert_step_size': 0.001,
        'pert_duration': 4,
    }

    ae_vars = {
        'in_dim': 115,
        'hidden_dims': [128, 64, 32]
    }

    svdd_vars = {
        'in_dim': 115,
        'hidden_dims': [128, 64, 32],
        'num_svdd_layer': 3
    }

    set_random_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # torch.device("cpu")#

    print('loading datasets from {}..'.format(args.file_name))
    dataset = nBaIoTDataset(root='/workspace/code/data/', file_name=args.file_name)
    train_sets, valid_sets, test_sets = dataset.create_datasets()

    print('Initialize strategy..')
    autoencoder = AutoEncoder
    svdd = SVDD

    ae_strategy = SaveModelStrategy(
        fraction_fit=args.fraction,
        fraction_eval=args.fraction,
        min_fit_clients=int(args.num_clients * args.fraction),
        min_eval_clients=int(args.num_clients * args.fraction),
        min_available_clients=int(args.num_clients * args.fraction),
        initial_parameters=fl.common.weights_to_parameters(get_parameters(autoencoder(in_dim=115, hidden_dims=[128, 64, 32]).to(device)))
    )

    svdd_strategy = SaveModelStrategySVDD(
        fraction_fit=args.fraction,
        fraction_eval=args.fraction,
        min_fit_clients=int(args.num_clients * args.fraction),
        min_eval_clients=int(args.num_clients * args.fraction),
        min_available_clients=int(args.num_clients * args.fraction),
        initial_parameters=fl.common.weights_to_parameters(get_parameters(svdd(**svdd_vars).to(device))),
        strategy_params=strategy_params
    )

    ae_vars = {
        'in_dim': 115,
        'hidden_dims': [128, 64, 32]
    }

    svdd_vars = {
        'in_dim': 115,
        'hidden_dims': [128, 64, 32],
        'num_svdd_layer': 3,
        # 'num_rep_layer': 3
    }

    print('Initialize done..')

    # print('pre_params',autoencoder.state_dict())

    # cid, net, net_params, net_vars, trainset, validset, testset, batch_size, device, flag

    # flag = False
    # fl.simulation.start_simulation(
    #     client_fn=partial(client_fn,
    #                       net=autoencoder,
    #                       net_params=ae_params,
    #                       net_vars=ae_vars,
    #                       trainset=train_sets,
    #                       validset=valid_sets,
    #                       testset=test_sets,
    #                       batch_size=args.batch_size,
    #                       device=device,
    #                       flag=flag),
    #     num_clients=args.num_clients,
    #     num_rounds=args.num_pretrain_rounds,
    #     client_resources={"num_cpus": 16, "num_gpus": 2},
    #     strategy=ae_strategy
    # )

    flag = True
    fl.simulation.start_simulation(
        client_fn=partial(client_fn,
                          rnd=0,
                          net=svdd,
                          net_params=descriptor_params,
                          net_vars=svdd_vars,
                          trainset=train_sets,
                          validset=valid_sets,
                          testset=test_sets,
                          batch_size=args.batch_size,
                          device=device,
                          flag=flag),
        num_clients=args.num_clients,
        num_rounds=args.num_pretrain_rounds,
        client_resources={"num_cpus": 16, "num_gpus": 2},
        strategy=svdd_strategy
    )

    # fl.server.start_server(
    #     config={"num_rounds": args.num_pretrain_rounds},
    #     strategy=ae_strategy
    # )


    # print('rec')
    # print('aft_params',autoencoder.state_dict())
    # print('weights: ', fl.common.Parameters())

if __name__ == '__main__':
    main()
