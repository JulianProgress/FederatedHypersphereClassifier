#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2022/02/18 15:41
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : add.py
# @Software  : PyCharm

import os
import random

import numpy as np
import torch
from base import FedDataset, CustomTensorDataset
from torch.utils.data import Dataset, DataLoader, TensorDataset
from utils import load_data


class ADDDataset(Dataset):
    def __init__(self, np_data, np_label, random_point=False, seq_first=True, max_window_size=64,
                 variable_length=False):
        """
        Dataset object for ADD dataset
        :param np_data: data
        :param np_label: label
        :param window_size: window size (none if variable length == True)
        :param variable_length: bool for variable length
        """
        super(ADDDataset).__init__()
        self.np_data = np_data
        self.np_label = np_label
        self.max_window_size = max_window_size
        self.variable_length = variable_length
        self.seq_first = seq_first
        self.random_point = random_point

    def __len__(self):
        return len(self.np_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        feature_data = self.np_data[idx]
        feature_label = self.np_label[idx]

        if not isinstance(feature_label, np.ndarray):
            feature_label = np.array(feature_label)

        feature_data, feature_label, feature_length = self._windowing(feature_data, feature_label)

        if not self.seq_first:
            feature_data = feature_data.T

        # if not self.variable_length:
        #     feature_data, feature_label = self._windowing(feature_data, feature_label)
        # elif len(feature_data) > self.window_limit:
        #     feature_data = feature_data[:self.window_limit]
        if self.variable_length:
            if len(feature_data) > self.max_window_size:
                feature_data = feature_data[:self.max_window_size]

        #     feature_data, feature_label = np.expand_dims(feature_data, axis=0), np.expand_dims(feature_label, axis=0)

        return {"X": torch.from_numpy(feature_data), "y": torch.from_numpy(feature_label), "seq_len": feature_length}

    def _windowing(self, data, label):
        seq_len = len(data)
        if seq_len < self.max_window_size:
            data = np.pad(data, [(0, self.max_window_size - seq_len), (0, 0)])
            label = np.pad(label, (0, self.max_window_size - len(label)))
        elif seq_len > self.max_window_size:  # log sequence data handling
            seq_len = self.max_window_size
            max_idx = seq_len - self.max_window_size + 1

            if self.random_point:
                idx = random.randint(0, max_idx)
                data = data[idx:idx + self.max_window_size]
                if type(label) == list:
                    label = np.array(label)
                label = label[idx:idx + self.max_window_size]
            else:
                data = data[0:self.max_window_size]
                label = label[0:self.max_window_size]

        return data, label, seq_len


# batch_size=4, shuffle=True, num_workers=16,
def add_dataloader(X, y, max_window_size=64, variable_length=False, random_point=False, **kwargs):
    dataset = ADDDataset(X, y, seq_first=True, random_point=random_point, max_window_size=max_window_size,
                         variable_length=variable_length)
    dataloader = DataLoader(dataset, **kwargs)
    return dataloader


def add_eventloader(X, y=None, **kwargs):
    # X = np.concatenate(X)
    if not y is None:
        # y = np.concatenate(y)
        dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
    else:
        dataset = TensorDataset(torch.from_numpy(X).float())
    dataloader = DataLoader(dataset, **kwargs)
    return dataloader


# def load_dataset():

class ADDRepDataset(FedDataset):

    def __init__(self, val_portion, val_train_portion, abnormal_in_train=True, abnormal_in_val=False, model_name='CAE',
                 root='./'):
        super().__init__(root)
        self.abnormal_in_train = abnormal_in_train
        self.abnormal_in_val = abnormal_in_val
        self.val_portion = val_portion
        self.val_train_portion = val_train_portion
        self.train_set = []
        self.valid_set = []
        self.test_set = []
        self.total_test = []

        self.abnormal_train = []
        self.abnormal_valid = []
        self.abnormal_test = []

        BASE_PATH = '/workspace/data/add/data/ADD/'

        DATA_PATH = "/workspace/data/add/data/ADD/final_data"
        SAVE_PATH = "/workspace/data/add/data/ADD/encoded_data"
        MODEL_PATH = "/workspace/data/add/data/ADD/model_save"

        # host_info = load_data(os.path.join(DATA_PATH, 'host_info.pkl'))

        window_host_list = []
        window_hostid_list = []

        linux_host_list = []
        linux_hostid_list = []
        for f in os.listdir(DATA_PATH):
            if 'data' in f:
                host_id = f.split('_')[0]
                if 'windows' in f:
                    window_host_list.append(os.path.join(DATA_PATH, f'{host_id}_windows_data.pkl'))
                    window_hostid_list.append(host_id)
                if 'linux' in f:
                    linux_host_list.append(os.path.join(DATA_PATH, f'{host_id}_linux_data.pkl'))
                    linux_hostid_list.append(host_id)

        # data = torch.load(os.path.join(root, file_name))
        self.window_hostid_list = window_hostid_list

        for host_id in window_hostid_list:
            train, test, val, test_tensor, ntrain, ntest, nval = self.load_dataset(SAVE_PATH, f'{host_id}_{model_name}_rep.pkl')
            self.abnormal_train.append(ntrain)
            self.abnormal_test.append(ntest)
            self.abnormal_valid.append(nval)
            self.train_set.append(train)
            self.valid_set.append(val)
            self.test_set.append(test)
            self.total_test.append(test_tensor)

    def return_total_test(self):
        X = None
        y = None
        for i, dtt in enumerate(self.total_test):
            if i == 0:
                X = dtt[0]
                y = dtt[1]
            else:
                X = torch.cat((X, dtt[0]))
                y = torch.cat((y, dtt[1]))

        return TensorDataset(X, y)

    def load_dataset(self, save_path, file_name):
        data = load_data(os.path.join(save_path, file_name))

        abnormal_idx = data['y'].astype(bool)
        abnormal_data = data['X'][abnormal_idx]
        normal_data = data['X'][~abnormal_idx]

        idx = np.arange(len(normal_data))
        # idx = np.random.permutation(idx)
        val_idx = idx[:int(len(idx) * self.val_portion)]
        train_idx = idx[int(len(idx) * self.val_portion):]
        normal_train_X = normal_data[train_idx]
        normal_test_X = normal_data[val_idx]

        idx = np.arange(len(abnormal_data))
        # idx = np.random.permutation(idx)
        val_idx = idx[:int(len(idx) * self.val_portion)]
        train_idx = idx[int(len(idx) * self.val_portion):]

        abnormal_train_X = abnormal_data[train_idx]
        abnormal_test_X = abnormal_data[val_idx]
        normal_train_y = np.zeros(len(normal_train_X))
        normal_test_y = np.zeros(len(normal_test_X))
        abnormal_train_y = np.ones(len(abnormal_train_X))
        abnormal_test_y = np.ones(len(abnormal_test_X))
        normal_val_X = normal_train_X[:int(len(normal_train_X) * self.val_train_portion)]
        normal_val_y = normal_train_y[:int(len(normal_train_y) * self.val_train_portion)]


        normal_train_X = normal_train_X[int(len(normal_train_X) * self.val_train_portion):]
        normal_train_y = normal_train_y[int(len(normal_train_y) * self.val_train_portion):]

        if self.abnormal_in_val:
            abnormal_val_X = abnormal_train_X[:int(len(abnormal_train_X) * self.val_train_portion)]
            abnormal_val_y = abnormal_train_y[:int(len(abnormal_train_y) * self.val_train_portion)]
            abnormal_train_X = abnormal_train_X[int(len(abnormal_train_X) * self.val_train_portion):]
            abnormal_train_y = abnormal_train_y[int(len(abnormal_train_y) * self.val_train_portion):]
            validation_X = np.concatenate((normal_val_X, abnormal_val_X))
            validation_y = np.concatenate((normal_val_y, abnormal_val_y))
        else:
            validation_X = normal_val_X
            validation_y = normal_val_y

        if self.abnormal_in_train:
            train_X = np.concatenate((normal_train_X, abnormal_train_X))
            train_y = np.concatenate((normal_train_y, abnormal_train_y))
        else:
            train_X = normal_train_X
            train_y = normal_train_y
        test_X = np.concatenate((normal_test_X, abnormal_test_X))
        test_y = np.concatenate((normal_test_y, abnormal_test_y))

        abnormal_num_train = train_y.sum()
        abnormal_num_test = test_y.sum()
        abnormal_num_val = validation_y.sum()

        trainset = TensorDataset(torch.Tensor(train_X), torch.Tensor(train_y))
        testset = TensorDataset(torch.Tensor(test_X), torch.Tensor(test_y))
        validset = TensorDataset(torch.Tensor(validation_X), torch.Tensor(validation_y))
        test_tensor = (torch.Tensor(test_X), torch.Tensor(test_y))

        return trainset, testset, validset, test_tensor, abnormal_num_train, abnormal_num_test, abnormal_num_val

    def create_datasets(self):
        return self.train_set, self.valid_set, self.test_set
