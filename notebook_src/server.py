#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2022/02/13 21:31
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : server.py
# @Software  : PyCharm

import argparse
from functools import partial
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
            torch.save(weights, '../model_save/fl/test_weight_{rnd}.pt')
            np.savez(f"../model_save/fl/ae_round-{rnd}-weights.npz", *weights)
        return weights


def client_fn(cid, net, net_params, net_vars, trainset, validset, testset, batch_size, device, flag):

    # try:
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

    print(cid)

    # cid, model, model_params, train_loader, valid_loader, test_loader, device, flag
    try:
        client = FedHSCClient(cid, net, net_params, train_loader, valid_loader, test_loader, device, flag)
    except Exception as e:
        print(e)
    return client


def get_parameters(model):
    return [values.cpu().numpy() for _, values in model.state_dict().items()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Federated Learning Simulation based on Flower")
    parser.add_argument('--seed', type=int, default=5959, help='random seed')
    parser.add_argument('--num_clients', '-N', type=int, default=9, help='number of clients')
    parser.add_argument('--fraction', '-K', type=float, default=0.1,
                        help='fraction of participating clients at each round')
    parser.add_argument('--batch_size', '-B', type=int, default=128, help='batch size for local update')
    parser.add_argument('--num_epochs', '-E', type=int, default=5, help='number of local epochs')
    parser.add_argument('--num_pretrain_epochs', '-P', type=int, default=5, help='number of local pretrain epochs')
    parser.add_argument('--num_rounds', '-R', type=int, default=10, help='number of required rounds')
    parser.add_argument('--num_pretrain_rounds', '-PR', type=int, default=5, help='number of required pretrain rounds')
    parser.add_argument('--weight_decay', '-W', type=float, default=1e-6, help='weight decay')
    args = parser.parse_args()

    ae_params = {
        'optimizer_name': 'Adam',
        'lr': 0.001,
        'n_epochs': args.num_pretrain_epochs,
        'lr_milestones': [20, 40],
        'weight_decay': 1e-6,
        'device': 'cpu'
    }

    descriptor_params = {
        'optimizer_name': 'Adam',
        'lr': 5e-5,
        'n_epochs': args.num_epochs,
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

    print('loading datasets..')
    dataset = nBaIoTDataset(root='/workspace/code/data/', file_name='nBaIoT_portion10.pkl')
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
        initial_parameters=fl.common.weights_to_parameters(get_parameters(autoencoder(**ae_vars).to(device)))
    )

    print('Initialize done..')

    # print('pre_params',autoencoder.state_dict())



    flag = False
    # cid, net, net_params, trainset, validset, testset, batch_size, device, flag
    fl.simulation.start_simulation(
        client_fn=partial(client_fn,
                          net=autoencoder,
                          net_params=ae_params,
                          net_vars=ae_vars,
                          trainset=train_sets,
                          validset=valid_sets,
                          testset=test_sets,
                          batch_size=args.batch_size,
                          device=device,
                          flag=flag),
        num_clients=args.num_clients,
        num_rounds=args.num_pretrain_rounds,
        client_resources={"num_cpus": 16, "num_gpus": 2},
        strategy=ae_strategy
    )

    # fl.server.start_server(
    #     config={"num_rounds": args.num_pretrain_rounds},
    #     strategy=ae_strategy
    # )


    print('rec')
    # print('aft_params',autoencoder.state_dict())
    # print('weights: ', fl.common.Parameters())

    # TODO: SVDD training

if __name__ == '__main__':
    main()
