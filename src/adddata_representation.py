#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2022/02/20 5:21 PM
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : adddata_representation.py
# @Software  : PyCharm

import os
import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
import argparse
from utils import load_data, set_random_seed, concatenate, save_data
from datasets import add_dataloader

from optimizer import RepAETrainer
from torch import nn
from model import RNNAE, CAE


def main() -> None:
    parser = argparse.ArgumentParser(description="Federated Learning Simulation based on Flower")
    parser.add_argument('--seed', type=int, default=5959, help='random seed')
    parser.add_argument('--device', '-d', type=str, default='cpu', help='device for train')
    parser.add_argument('--max_window_size', '-w', type=int, default=128, help='device for train')
    parser.add_argument('--batch_size', '-B', type=int, default=128, help='batch size for local update')
    parser.add_argument('--num_epochs', '-E', type=int, default=50, help='number of local epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', '-W', type=float, default=1e-6, help='weight decay')
    parser.add_argument('--type', '-t', type=int, default=0, help='representation model')
    args = parser.parse_args()
    print(args)

    set_random_seed(args.seed)

    BASE_PATH = '/workspace/data/add/data/ADD/'

    DATA_PATH = "/workspace/data/add/data/ADD/final_data"
    SAVE_PATH = "/workspace/data/add/data/ADD/encoded_data"
    MODEL_PATH = "/workspace/data/add/data/ADD/model_save/representation_model"

    # host_info = load_data(os.path.join(DATA_PATH, 'host_info.pkl'))

    window_host_list = []
    window_hostid_list = []
    linux_host_list = []
    for f in os.listdir(DATA_PATH):
        if 'data' in f:
            host_id = f.split('_')[0]
            if 'windows' in f:
                window_host_list.append(os.path.join(DATA_PATH, f'{host_id}_windows_data.pkl'))
                window_hostid_list.append(host_id)
            if 'linux' in f:
                linux_host_list.append(os.path.join(DATA_PATH, f'{host_id}_linux_data.pkl'))

    print('start training.. ', window_host_list)
    for i, window_host in enumerate(window_host_list):
        data = load_data(window_host)
        max_window_size = args.max_window_size
        batch_size = args.batch_size
        train_loader = add_dataloader(data['train_X'], data['train_y'], max_window_size, random_point=True,
                                      batch_size=batch_size, shuffle=True, num_workers=16, drop_last=True)
        train_loader_forencoder = add_dataloader(data['train_X'], data['train_y'], max_window_size, random_point=False,
                                                 batch_size=batch_size, shuffle=False, num_workers=16, drop_last=False)
        test_loader = add_dataloader(data['test_X'], data['test_y'], max_window_size, random_point=False,
                                     batch_size=batch_size, shuffle=False, num_workers=16, drop_last=False)

        if args.type == 0:
            model_name = 'CAE'
            transpose = True
        else:
            model_name = 'RNNAE'
            transpose = False

        ae_params = {
            'optimizer_name': 'Adam',
            'lr': args.lr,
            'n_epochs': args.num_epochs,
            'lr_milestones': [20, 40],
            'weight_decay': args.weight_decay,
            'device': args.device,
            'X_idx': 'X',
            'y_idx': 'y',
            'patience': 4,
            'transpose': transpose
        }

        if args.type == 0:
            net = CAE(in_dim=661, num_layers=5, hidden_dim=512, filter_size=7, stride=1)
        else:
            net_params = dict(
                rnn_type='GRU',
                input_dim=661,
                encoder_dim=32,
                hidden_dim=256,
                num_layers=3,
                dropout=0,
                batch_first=True,
                device=args.device
            )

            net = RNNAE(**net_params)

        trainer = RepAETrainer(**ae_params)
        net, examples = trainer.train(train_loader, test_loader, net)

        encoded_X = None
        labels = None
        for batch in train_loader_forencoder:
            X = batch['X'].to(args.device).float()
            y = batch['y']

            z = net.encode(X)
            y = [1 if 1 in y_ else 0 for y_ in y]

            encoded_X = concatenate(encoded_X, z.detach().cpu().numpy())
            labels = concatenate(labels, np.array(y))

        for batch in test_loader:
            X = batch['X'].to(args.device).float()
            y = batch['y']

            z = net.encode(X)
            y = [1 if 1 in y_ else 0 for y_ in y]

            encoded_X = concatenate(encoded_X, z.detach().cpu().numpy())
            labels = concatenate(labels, np.array(y))

        save_data(dict(
            X=encoded_X,
            y=labels
        ), directory=SAVE_PATH, filename=f'{window_hostid_list[i]}_{model_name}_rep.pkl')
        torch.save(net.cpu().state_dict(), os.path.join(MODEL_PATH, f'{window_hostid_list[i]}_{model_name}.pt'))


if __name__ == '__main__':
    main()
