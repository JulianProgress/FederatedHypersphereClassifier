#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2022/02/13 3:45 PM
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : nbaiot.py
# @Software  : PyCharm

import fnmatch
import os

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm

from base import FedDataset, CustomTensorDataset

sns.set()

from torch.utils.data import DataLoader, TensorDataset
import torch


def create_datasets(DATA_PATH='/workspace/data/add/data/nBaIoT/', train_portion=0.8, anomaly_portion=0.3,
                    num_nonanomaly=4, ad=True, batch_size=64):
    """

    :param num_nonanomaly:
    :param ad: is anomaly detection ( label : 0 or 1(anomaly) )
    :return:
    """
    device_list = ['Danmini_Doorbell', 'Ecobee_Thermostat', 'Ennio_Doorbell', 'Philips_B120N10_Baby_Monitor',
                   'Provision_PT_737E_Security_Camera', 'Provision_PT_838_Security_Camera', 'Samsung_SNH_1011_N_Webcam',
                   'SimpleHome_XCS7_1002_WHT_Security_Camera', 'SimpleHome_XCS7_1003_WHT_Security_Camera']

    if num_nonanomaly > 0:
        non_anomaly_list = np.random.choice(device_list, num_nonanomaly, replace=False)

    # fit scaler
    scaler_list = []

    train_loader_list = []
    test_loader_list = []
    types_list = []

    train_data_list = []
    train_label_list = []
    valid_data_list = []
    valid_label_list = []
    test_data_list = []
    test_label_list = []

    for i, device in tqdm(enumerate(device_list)):
        # dataset scaling
        scaler = MinMaxScaler()
        benign_data = pd.read_csv(os.path.join(DATA_PATH, device, 'benign_traffic.csv'))
        scaler.fit(benign_data[:round(len(benign_data) * train_portion)])
        scaler_list.append(scaler)
        benign_data = scaler.transform(benign_data)

        train_data = benign_data[:round(len(benign_data) * train_portion)]
        idx = np.arange(len(train_data))
        idx = np.random.permutation(idx)
        train_data = train_data[idx]

        test_benign = benign_data[round(len(benign_data) * train_portion):]

        # import anomaly data
        flist = []
        for i, (dirpath, dirnames, filenames) in enumerate(os.walk(os.path.join(DATA_PATH, device))):
            fnl = fnmatch.filter(filenames, '*.csv')
            if i > 0:
                for fn in fnl:
                    flist.append(os.path.join(dirpath, fn))

        test_list = []
        type_list = []
        for file in flist:
            fns = file.split('/')

            ty = fns[-2].split('_')[0] + '_' + fns[-1].split('.')[0]

            test_df = pd.read_csv(file)
            test_df = scaler.transform(test_df)

            test_list.append(test_df)
            type_list.append(ty)

        types_list.append(type_list)

        test_anomaly = np.concatenate(test_list)

        if ad:
            test_label = np.concatenate([np.full((len(X_test)), 1) for i, X_test in enumerate(test_list)])
        else:
            test_label = np.concatenate([np.full((len(X_test)), i + 1) for i, X_test in enumerate(test_list)])

        idx = np.arange(len(test_label))
        idx = np.random.permutation(idx)

        test_anomaly = test_anomaly[idx]
        test_label = test_label[idx]

        # append additional anomaly data to trainset
        if num_nonanomaly > 0 and not device in non_anomaly_list:
            orig_len = len(test_anomaly)
            train_anomaly = test_anomaly[:round(orig_len * anomaly_portion)]
            train_label = test_label[:round(orig_len * anomaly_portion)]
            test_anomaly = test_anomaly[round(orig_len * anomaly_portion):]
            test_label = test_label[round(orig_len * anomaly_portion):]

            train_label = np.concatenate([np.full((len(train_data)), 0), train_label])
            train_data = np.concatenate([train_data, train_anomaly])

            idx = np.arange(len(train_label))
            idx = np.random.permutation(idx)

            train_data = train_data[idx]
            train_label = train_label[idx]

        else:
            train_label = np.full((len(train_data)), 0)

        # train dataloader
        train_data_list.append(train_data[:round(len(train_data) * 0.7)])
        train_label_list.append(train_label[:round(len(train_label) * 0.7)])
        valid_data_list.append(train_data[round(len(train_data) * 0.7):])
        valid_label_list.append(train_label[round(len(train_data) * 0.7):])

        train_data = torch.from_numpy(train_data).float()
        train_label = torch.from_numpy(train_label).float()
        train_loader = DataLoader(TensorDataset(train_data, train_label), batch_size=batch_size, num_workers=16,
                                  shuffle=True)

        # test dataloader
        test_data = np.concatenate([test_benign, test_anomaly])
        test_label = np.concatenate([np.full((len(test_benign)), 0), test_label])

        idx = np.arange(len(test_label))
        idx = np.random.permutation(idx)
        test_data = test_data[idx]
        test_label = test_label[idx]

        test_data_list.append(test_data)
        test_label_list.append(test_label)

        test_X = torch.from_numpy(test_data).float()
        test_y = torch.from_numpy(test_label).float()

        ds = TensorDataset(test_X, test_y)
        test_loader = DataLoader(ds, batch_size=batch_size, num_workers=16, shuffle=False)

        train_loader_list.append(train_loader)
        test_loader_list.append(test_loader)

        del train_label, train_data, test_anomaly, test_benign, test_label, test_y, test_X

    train_X = np.concatenate(train_data_list)
    test_X = np.concatenate(test_data_list)
    test_y = np.concatenate(test_label_list)

    train_X = torch.from_numpy(train_X)
    test_X = torch.from_numpy(test_X)
    test_y = torch.from_numpy(test_y)

    uni_trainloader = DataLoader(TensorDataset(train_X), batch_size=batch_size, num_workers=16, shuffle=True)
    uni_testloader = DataLoader(TensorDataset(test_X, test_y), batch_size=batch_size, num_workers=16, shuffle=False)

    torch.save(dict(
        train_data_list=train_data_list,
        train_label_list=train_label_list,
        valid_data_list=valid_data_list,
        valid_label_list=valid_label_list,
        test_data_list=test_data_list,
        test_label_list=test_label_list,
        non_anomaly_list=non_anomaly_list
    ), '/workspace/data/add/data/nBaIoT/dataset.pkl')

    # torch.save(dict(
    #     train_df_list=train_loader_list,
    #     test_df_list=test_loader_list,
    #     type_list=types_list,
    #     scaler_list=scaler_list
    # ), '/workspace/dataset/data/nBaIoT/dataloader_ad.pkl')

    # return train_loader_list, test_loader_list, uni_trainloader, uni_testloader, types_list, scaler_list


class nBaIoTDataset(FedDataset):

    def __init__(self, root: str, file_name: str):
        super().__init__(root)

        self.train_set = []
        self.valid_set = []
        self.test_set = []

        data = torch.load(os.path.join(root, file_name))

        train_data_list = data['train_data_list']
        train_label_list = data['train_label_list']
        valid_data_list = data['valid_data_list']
        valid_label_list = data['valid_label_list']
        test_data_list = data['test_data_list']
        test_label_list = data['test_label_list']


        train_set = [
            (torch.Tensor(train_data), torch.Tensor(train_label)) for train_data, train_label in
            zip(train_data_list, train_label_list)
        ]
        valid_set = [
            (torch.Tensor(valid_data), torch.Tensor(valid_label)) for valid_data, valid_label in
            zip(valid_data_list, valid_label_list)
        ]
        test_set = [
            (torch.Tensor(test_data), torch.Tensor(test_label)) for test_data, test_label in
            zip(test_data_list, test_label_list)
        ]

        self.train_set = [
            CustomTensorDataset(local_dataset)
            for local_dataset in train_set
        ]

        self.valid_set = [
            CustomTensorDataset(local_dataset)
            for local_dataset in valid_set
        ]

        self.test_set = [
            CustomTensorDataset(local_dataset)
            for local_dataset in test_set
        ]

    def create_datasets(self):
        return self.train_set, self.valid_set, self.test_set
