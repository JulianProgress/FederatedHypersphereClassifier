#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2022/02/21 10:14
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : add_server.py
# @Software  : PyCharm

from src.datasets import ADDRepDataset
from add_client import FedHSCClient
from model import SVDD


def client_fn(cid, net, net_params, net_vars, trainset, validset, testset, batch_size, device, flag):

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
    client = FedHSCClient(cid, net, net_params, train_loader, valid_loader, test_loader, device, flag)
    return client

def get_parameters(model):
    return [values.cpu().numpy() for _, values in model.state_dict().items()]



def main():
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
        'pert_duration': 4
    }

    ae_vars = {
        'in_dim': 115,
        'hidden_dims': [128, 64, 32]
    }

    svdd_vars = {
        'rep_model_type': 'RNNEncoder',
        'in_dim': 115,
        'hidden_dims': [16, 256], # hidden_dim, rep_dim
        'window_size': 128,
        'num_svdd_layer': 3
    }

    svdd = SVDD

