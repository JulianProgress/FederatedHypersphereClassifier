import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset
from torchvision import transforms
from torchvision.datasets import MNIST

from ..utils import load_data


class ImageDatasetModule(object):

    def __init__(self, dataset_params):
        self.params = dataset_params
        self.train_id_dataset = None
        self.train_ood_dataset = None
        self.val_id_dataset = None
        self.val_ood_dataset = None
        self.test_id_dataset = None
        self.test_ood_dataset = None

        self.normalization_variables = None
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __repr__(self):
        return f"Train ID - Size : {len(self.train_id_dataset)}\n" \
               f"Train OOD - Size : {len(self.train_ood_dataset)}\n" \
               f"Val ID - Size : {len(self.val_id_dataset)}\n" \
               f"Val OOD - Size : {len(self.val_ood_dataset)}\n" \
               f"Test ID - Size : {len(self.test_id_dataset)}\n" \
               f"Test OOD - Size : {len(self.test_ood_dataset)}\n"

    @property
    def train_dataloader(self):
        return None

    @property
    def val_dataloader(self):
        return None

    @property
    def test_dataloader(self):
        return None

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, *args, **kwargs):
        pass

    def split_dataset(self, data_list, target_list):
        train_ratio, val_ratio, test_ratio = self.params["train_val_test_ratio"]
        train_id_true_target_list = self.params["train_id_targets"]
        train_ood_true_target_list = self.params["train_ood_targets"]
        test_ood_true_target_list = self.params["test_ood_targets"]
        ood_ratio = self.params["ood_ratio"]

        # Shuffling
        if self.params["is_shuffle"]:
            shuffle_idx_list = np.arange(len(data_list))
            np.random.shuffle(shuffle_idx_list)
            data_list, target_list = data_list[shuffle_idx_list], target_list[shuffle_idx_list]
            print("Dataset shuffled")

        train_id_targets_bool = torch.full((target_list.shape[0],), False)
        train_ood_targets_bool = torch.full((target_list.shape[0],), False)
        test_ood_targets_bool = torch.full((target_list.shape[0],), False)

        # Train id targets
        for train_id_true_target in train_id_true_target_list:
            train_id_targets_bool += (target_list == train_id_true_target)

        # Train ood targets
        for train_ood_true_target in train_ood_true_target_list:
            train_ood_targets_bool += (target_list == train_ood_true_target)

        # Test ood targets
        for test_ood_true_target in test_ood_true_target_list:
            test_ood_targets_bool += (target_list == test_ood_true_target)

        id_data = data_list[train_id_targets_bool]
        id_targets = target_list[train_id_targets_bool]
        train_ood_data = data_list[train_ood_targets_bool]
        train_ood_targets = target_list[train_ood_targets_bool]
        test_ood_data = data_list[test_ood_targets_bool]
        test_ood_targets = target_list[test_ood_targets_bool]

        print(f"Original label info :\n"
              f" > ID labels : {sorted(set(id_targets.tolist()))}\n"
              f" > Train OOD labels : {sorted(set(train_ood_targets.tolist()))}\n"
              f" > Test OOD labels : {sorted(set(test_ood_targets.tolist()))}")

        # Transform targets
        # id_targets = np.array(list(map(lambda id_target: self.ce_label_dict[int(id_target)], id_targets)))
        train_ood_targets = np.array(list(map(lambda train_ood_target: -1, train_ood_targets)))
        test_ood_targets = np.array(list(map(lambda test_ood_target: -1, test_ood_targets)))

        print(f"Transformed label info :\n"
              f" > ID labels : {sorted(set(id_targets.tolist()))}\n"
              f" > Train OOD labels : {sorted(set(train_ood_targets.tolist()))}\n"
              f" > Test OOD labels : {sorted(set(test_ood_targets.tolist()))}")
        print(f"Total dataset :\n"
              f" > ID size : {len(id_data)}\n"
              f" > Train OOD size : {len(train_ood_data)}\n"
              f" > Test OOD size : {len(test_ood_data)}")

        # Train split
        id_train_size = int(len(id_data) * train_ratio)
        id_val_size = int(len(id_data) * val_ratio)
        id_test_size = int(len(id_data) * test_ratio)

        split_pivot_start = 0
        split_pivot_end = 0

        split_pivot_end += id_train_size
        id_train_data_list = id_data[split_pivot_start:split_pivot_end]
        id_train_target_list = id_targets[split_pivot_start:split_pivot_end]

        split_pivot_start += id_train_size
        split_pivot_end += id_val_size
        id_val_data_list = id_data[split_pivot_start:split_pivot_end]
        id_val_target_list = id_targets[split_pivot_start:split_pivot_end]

        split_pivot_start += id_val_size
        split_pivot_end += id_test_size
        id_test_data_list = id_data[split_pivot_start:split_pivot_end]
        id_test_target_list = id_targets[split_pivot_start:split_pivot_end]

        # Set normalized variables
        normalization_variables_dict = self.calculate_normalization_variables(data_list=id_train_data_list)
        self.set_normalization_variables(
            normalization_variables_dict=normalization_variables_dict
        )
        normalization_variables = self.normalization_variables

        # Set transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*normalization_variables)
        ])

        train_id_dataset = ImageDataset(data=id_train_data_list, targets=id_train_target_list, transform=self.transform)
        val_id_dataset = ImageDataset(data=id_val_data_list, targets=id_val_target_list, transform=self.transform)
        test_id_dataset = ImageDataset(data=id_test_data_list, targets=id_test_target_list, transform=self.transform)

        print(f"ID dataset :\n"
              f" > Train size : {len(train_id_dataset)}\n"
              f" > Val size : {len(val_id_dataset)}\n"
              f" > Test size : {len(test_id_dataset)}")

        # Val split
        val_ood_train_size = int(len(train_ood_data) * train_ratio)
        val_ood_val_size = int(len(train_ood_data) * val_ratio)
        val_ood_test_size = int(len(train_ood_data) * test_ratio)

        split_pivot_start = 0
        split_pivot_end = 0

        split_pivot_end += val_ood_train_size
        val_ood_train_data_list = train_ood_data[split_pivot_start:split_pivot_end]
        val_ood_train_target_list = train_ood_targets[split_pivot_start:split_pivot_end]

        split_pivot_start += val_ood_train_size
        split_pivot_end += val_ood_val_size
        val_ood_val_data_list = train_ood_data[split_pivot_start:split_pivot_end]
        val_ood_val_target_list = train_ood_targets[split_pivot_start:split_pivot_end]

        split_pivot_start += val_ood_val_size
        split_pivot_end += val_ood_test_size
        val_ood_test_data_list = train_ood_data[split_pivot_start:split_pivot_end]
        val_ood_test_target_list = train_ood_targets[split_pivot_start:split_pivot_end]

        # Val OOD data cutting with ratio
        if ood_ratio:
            # Train
            val_ood_train_ood_cut_idx = int(len(id_train_data_list) * ood_ratio)
            val_ood_train_data_list = val_ood_train_data_list[:val_ood_train_ood_cut_idx]
            val_ood_train_target_list = val_ood_train_target_list[:val_ood_train_ood_cut_idx]

            # Val
            val_ood_val_ood_cut_idx = int(len(id_val_data_list) * ood_ratio)
            val_ood_val_data_list = val_ood_val_data_list[:val_ood_val_ood_cut_idx]
            val_ood_val_target_list = val_ood_val_target_list[:val_ood_val_ood_cut_idx]

            # Test
            val_ood_test_ood_cut_idx = int(len(id_test_data_list) * ood_ratio)
            val_ood_test_data_list = val_ood_test_data_list[:val_ood_test_ood_cut_idx]
            val_ood_test_target_list = val_ood_test_target_list[:val_ood_test_ood_cut_idx]

        val_ood_train_dataset = ImageDataset(data=val_ood_train_data_list, targets=val_ood_train_target_list,
                                             transform=self.transform)
        val_ood_val_dataset = ImageDataset(data=val_ood_val_data_list, targets=val_ood_val_target_list,
                                           transform=self.transform)
        val_ood_test_dataset = ImageDataset(data=val_ood_test_data_list, targets=val_ood_test_target_list,
                                            transform=self.transform)

        print(f"Val-OOD dataset :\n"
              f" > Train size : {len(val_ood_train_dataset)}\n"
              f" > Val size : {len(val_ood_val_dataset)}\n"
              f" > Test size : {len(val_ood_test_dataset)}")

        # Test split
        test_ood_train_size = int(len(test_ood_data) * train_ratio)
        test_ood_val_size = int(len(test_ood_data) * val_ratio)
        test_ood_test_size = int(len(test_ood_data) * test_ratio)

        split_pivot_start = 0
        split_pivot_end = 0

        split_pivot_end += test_ood_train_size
        test_ood_train_data_list = test_ood_data[split_pivot_start:split_pivot_end]
        test_ood_train_target_list = test_ood_targets[split_pivot_start:split_pivot_end]

        split_pivot_start += test_ood_train_size
        split_pivot_end += test_ood_val_size
        test_ood_val_data_list = test_ood_data[split_pivot_start:split_pivot_end]
        test_ood_val_target_list = test_ood_targets[split_pivot_start:split_pivot_end]

        split_pivot_start += test_ood_val_size
        split_pivot_end += test_ood_test_size
        test_ood_test_data_list = test_ood_data[split_pivot_start:split_pivot_end]
        test_ood_test_target_list = test_ood_targets[split_pivot_start:split_pivot_end]

        # Val OOD data cutting with ratio
        if ood_ratio:
            # Train
            test_ood_train_ood_cut_idx = int(len(id_train_data_list) * ood_ratio)
            test_ood_train_data_list = test_ood_train_data_list[:test_ood_train_ood_cut_idx]
            test_ood_train_target_list = test_ood_train_target_list[:test_ood_train_ood_cut_idx]

            # Val
            test_ood_val_ood_cut_idx = int(len(id_val_data_list) * ood_ratio)
            test_ood_val_data_list = test_ood_val_data_list[:test_ood_val_ood_cut_idx]
            test_ood_val_target_list = test_ood_val_target_list[:test_ood_val_ood_cut_idx]

            # Test
            test_ood_test_ood_cut_idx = int(len(id_test_data_list) * ood_ratio)
            test_ood_test_data_list = test_ood_test_data_list[:test_ood_test_ood_cut_idx]
            test_ood_test_target_list = test_ood_test_target_list[:test_ood_test_ood_cut_idx]

        test_ood_train_dataset = ImageDataset(data=test_ood_train_data_list, targets=test_ood_train_target_list,
                                              transform=self.transform)
        test_ood_val_dataset = ImageDataset(data=test_ood_val_data_list, targets=test_ood_val_target_list,
                                            transform=self.transform)
        test_ood_test_dataset = ImageDataset(data=test_ood_test_data_list, targets=test_ood_test_target_list,
                                             transform=self.transform)

        print(f"Test-OOD dataset :\n"
              f" > Train size : {len(test_ood_train_dataset)}\n"
              f" > Val size : {len(test_ood_val_dataset)}\n"
              f" > Test size : {len(test_ood_test_dataset)}")

        self.train_id_dataset, self.val_id_dataset, self.test_id_dataset = \
            train_id_dataset, val_id_dataset, test_id_dataset
        self.train_ood_dataset, self.val_ood_dataset, self.test_ood_dataset = \
            val_ood_train_dataset, val_ood_val_dataset, test_ood_test_dataset

        print(f"Train dataset :\n"
              f" > ID size : {len(self.train_id_dataset)}\n"
              f" > OOD size : {len(self.train_ood_dataset)}")
        print(f"Val dataset :\n"
              f" > ID size : {len(self.val_id_dataset)}\n"
              f" > OOD size : {len(self.val_ood_dataset)}")
        print(f"Test dataset :\n"
              f" > ID size : {len(self.test_id_dataset)}\n"
              f" > OOD size : {len(self.test_ood_dataset)}")

    def calculate_normalization_variables(self, data_list):
        # Default : (n, n, 3) images, plz cascading this method.
        data_list = np.array(data_list)

        mean_arr = np.array([np.mean(data, axis=(0, 1)) for data in data_list])
        std_arr = np.array([np.std(data, axis=(0, 1)) for data in data_list])

        mean_list = np.mean(mean_arr, axis=0)
        std_list = np.mean(std_arr, axis=0)

        # If has one channel.
        if not isinstance(mean_list, np.ndarray) or not isinstance(std_list, np.ndarray):
            mean_list = [mean_list]
            std_list = [std_list]

        mean_list = list(map(lambda mean: float(mean), mean_list))
        std_list = list(map(lambda mean: float(mean), std_list))

        return dict(
            means=mean_list,
            stds=std_list,
        )

    def set_normalization_variables(self, normalization_variables_dict=None):
        mean_list = normalization_variables_dict["means"]
        std_list = normalization_variables_dict["stds"]

        print(f"Normalization variables are NOT exist")

        self.normalization_variables = [[mean for mean in mean_list], [std for std in std_list]]

        print(f"Complete to set normalization variables :\n"
              f" > Mean : {mean_list}\n"
              f" > Std : {std_list}")


class ImageDataset(Dataset):

    def __init__(self, data, targets, transform=None):
        # def __init__(self, data, targets, transform=None, id_target_transform=None, ood_target_transform=None):
        super(ImageDataset, self).__init__()
        self.data = data
        self.targets = targets
        self.transform = transform
        # self.id_target_transform = id_target_transform,
        # self.ood_target_transform = ood_target_transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        mode = None
        if len(img.shape) == 2:
            mode = 'L'

        img = Image.fromarray(img.numpy(), mode=mode)

        if self.transform is not None:
            img = self.transform(img)

        # if self.id_target_transform is not None:
        #     target = self.id_target_transform(target)
        #
        # if self.ood_target_transform is not None:
        #     target = self.ood_target_transform(target)

        return img, target


class MNISTDataModule(ImageDatasetModule):
    SAVE_DIR_PATH = "/workspace/code/data/mnist/"

    def __init__(self, dataset_params):
        super().__init__(dataset_params=dataset_params)

    def prepare_data(self, *args, **kwargs):
        MNIST(root=MNISTDataModule.SAVE_DIR_PATH, train=True, download=True)
        MNIST(root=MNISTDataModule.SAVE_DIR_PATH, train=False, download=True)

    def setup(self):
        """
        Generate Train | Validation | Test
        """
        train_entire = MNIST(
            root=MNISTDataModule.SAVE_DIR_PATH,
            train=True,
            transform=self.transform
        )
        test_entire = MNIST(
            root=MNISTDataModule.SAVE_DIR_PATH,
            train=False,
            transform=self.transform
        )

        data_list = torch.cat(tensors=[train_entire.data, test_entire.data])
        target_list = torch.cat(tensors=[train_entire.targets, test_entire.targets])

        self.split_dataset(data_list=data_list, target_list=target_list)

    @property
    def test_dataset(self):
        return ConcatDataset(datasets=[self.test_id_dataset, self.test_ood_dataset])

    @property
    def test_dataset(self):
        return ConcatDataset(datasets=[self.test_id_dataset, self.test_ood_dataset])

#
# class MNISTDataset(Dataset):
#     def __init__(self, np_data, np_label, random_point=False, seq_first=True, max_window_size=64,
#                  variable_length=False):
#         """
#         Dataset object for ADD dataset
#         :param np_data: data
#         :param np_label: label
#         :param window_size: window size (none if variable length == True)
#         :param variable_length: bool for variable length
#         """
#         super(MNISTDataset).__init__()
#         self.np_data = np_data
#         self.np_label = np_label
#         self.max_window_size = max_window_size
#         self.variable_length = variable_length
#         self.seq_first = seq_first
#         self.random_point = random_point
#
#     def __len__(self):
#         return len(self.np_data)
#
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#
#         feature_data = self.np_data[idx]
#         feature_label = self.np_label[idx]
#
#         if not isinstance(feature_label, np.ndarray):
#             feature_label = np.array(feature_label)
#
#         feature_data, feature_label, feature_length = self._windowing(feature_data, feature_label)
#
#         if not self.seq_first:
#             feature_data = feature_data.T
#
#         # if not self.variable_length:
#         #     feature_data, feature_label = self._windowing(feature_data, feature_label)
#         # elif len(feature_data) > self.window_limit:
#         #     feature_data = feature_data[:self.window_limit]
#         if self.variable_length:
#             if len(feature_data) > self.max_window_size:
#                 feature_data = feature_data[:self.max_window_size]
#
#         #     feature_data, feature_label = np.expand_dims(feature_data, axis=0), np.expand_dims(feature_label, axis=0)
#
#         return {"X": torch.from_numpy(feature_data), "y": torch.from_numpy(feature_label), "seq_len": feature_length}
#
#     def _windowing(self, data, label):
#         seq_len = len(data)
#         if seq_len < self.max_window_size:
#             data = np.pad(data, [(0, self.max_window_size - seq_len), (0, 0)])
#             label = np.pad(label, (0, self.max_window_size - len(label)))
#         elif seq_len > self.max_window_size:  # log sequence data handling
#             seq_len = self.max_window_size
#             max_idx = seq_len - self.max_window_size + 1
#
#             if self.random_point:
#                 idx = random.randint(0, max_idx)
#                 data = data[idx:idx + self.max_window_size]
#                 if type(label) == list:
#                     label = np.array(label)
#                 label = label[idx:idx + self.max_window_size]
#             else:
#                 data = data[0:self.max_window_size]
#                 label = label[0:self.max_window_size]
#
#         return data, label, seq_len
#
#
# # batch_size=4, shuffle=True, num_workers=16,
# def add_dataloader(X, y, max_window_size=64, variable_length=False, random_point=False, **kwargs):
#     dataset = MNISTDataset(X, y, seq_first=True, random_point=random_point, max_window_size=max_window_size,
#                          variable_length=variable_length)
#     dataloader = DataLoader(dataset, **kwargs)
#     return dataloader
#
#
# def add_eventloader(X, y=None, **kwargs):
#     # X = np.concatenate(X)
#     if not y is None:
#         # y = np.concatenate(y)
#         dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
#     else:
#         dataset = TensorDataset(torch.from_numpy(X).float())
#     dataloader = DataLoader(dataset, **kwargs)
#     return dataloader
#
#
# class ADDRepDataset(FedDataset):
#
#     def __init__(self, val_portion, val_train_portion, abnormal_in_train=True, abnormal_in_val=False, model_name='CAE',
#                  root='./'):
#         super().__init__(root)
#         self.abnormal_in_train = abnormal_in_train
#         self.abnormal_in_val = abnormal_in_val
#         self.val_portion = val_portion
#         self.val_train_portion = val_train_portion
#         self.train_set = []
#         self.valid_set = []
#         self.test_set = []
#         self.total_test = []
#
#         self.abnormal_train = []
#         self.abnormal_valid = []
#         self.abnormal_test = []
#
#         BASE_PATH = '/workspace/data/add/data/ADD/'
#
#         DATA_PATH = "/workspace/data/add/data/ADD/final_data"
#         SAVE_PATH = "/workspace/data/add/data/ADD/encoded_data"
#         MODEL_PATH = "/workspace/data/add/data/ADD/model_save"
#
#         # host_info = load_data(os.path.join(DATA_PATH, 'host_info.pkl'))
#
#         window_host_list = []
#         window_hostid_list = []
#
#         linux_host_list = []
#         linux_hostid_list = []
#         for f in os.listdir(DATA_PATH):
#             if 'data' in f:
#                 host_id = f.split('_')[0]
#                 if 'windows' in f:
#                     window_host_list.append(os.path.join(DATA_PATH, f'{host_id}_windows_data.pkl'))
#                     window_hostid_list.append(host_id)
#                 if 'linux' in f:
#                     linux_host_list.append(os.path.join(DATA_PATH, f'{host_id}_linux_data.pkl'))
#                     linux_hostid_list.append(host_id)
#
#         # data = torch.load(os.path.join(root, file_name))
#         self.window_hostid_list = window_hostid_list
#
#         for host_id in window_hostid_list:
#             train, test, val, test_tensor, ntrain, ntest, nval = self.load_dataset(SAVE_PATH, f'{host_id}_{model_name}_rep.pkl')
#             self.abnormal_train.append(ntrain)
#             self.abnormal_test.append(ntest)
#             self.abnormal_valid.append(nval)
#             self.train_set.append(train)
#             self.valid_set.append(val)
#             self.test_set.append(test)
#             self.total_test.append(test_tensor)
#
#     def return_total_test(self):
#         X = None
#         y = None
#         for i, dtt in enumerate(self.total_test):
#             if i == 0:
#                 X = dtt[0]
#                 y = dtt[1]
#             else:
#                 X = torch.cat((X, dtt[0]))
#                 y = torch.cat((y, dtt[1]))
#
#         return TensorDataset(X, y)
#
#     def load_dataset(self, save_path, file_name):
#         data = load_data(os.path.join(save_path, file_name))
#
#         abnormal_idx = data['y'].astype(bool)
#         abnormal_data = data['X'][abnormal_idx]
#         normal_data = data['X'][~abnormal_idx]
#
#         idx = np.arange(len(normal_data))
#         # idx = np.random.permutation(idx)
#         val_idx = idx[:int(len(idx) * self.val_portion)]
#         train_idx = idx[int(len(idx) * self.val_portion):]
#         normal_train_X = normal_data[train_idx]
#         normal_test_X = normal_data[val_idx]
#
#         idx = np.arange(len(abnormal_data))
#         # idx = np.random.permutation(idx)
#         val_idx = idx[:int(len(idx) * self.val_portion)]
#         train_idx = idx[int(len(idx) * self.val_portion):]
#
#         abnormal_train_X = abnormal_data[train_idx]
#         abnormal_test_X = abnormal_data[val_idx]
#         normal_train_y = np.zeros(len(normal_train_X))
#         normal_test_y = np.zeros(len(normal_test_X))
#         abnormal_train_y = np.ones(len(abnormal_train_X))
#         abnormal_test_y = np.ones(len(abnormal_test_X))
#         normal_val_X = normal_train_X[:int(len(normal_train_X) * self.val_train_portion)]
#         normal_val_y = normal_train_y[:int(len(normal_train_y) * self.val_train_portion)]
#
#
#         normal_train_X = normal_train_X[int(len(normal_train_X) * self.val_train_portion):]
#         normal_train_y = normal_train_y[int(len(normal_train_y) * self.val_train_portion):]
#
#         if self.abnormal_in_val:
#             abnormal_val_X = abnormal_train_X[:int(len(abnormal_train_X) * self.val_train_portion)]
#             abnormal_val_y = abnormal_train_y[:int(len(abnormal_train_y) * self.val_train_portion)]
#             abnormal_train_X = abnormal_train_X[int(len(abnormal_train_X) * self.val_train_portion):]
#             abnormal_train_y = abnormal_train_y[int(len(abnormal_train_y) * self.val_train_portion):]
#             validation_X = np.concatenate((normal_val_X, abnormal_val_X))
#             validation_y = np.concatenate((normal_val_y, abnormal_val_y))
#         else:
#             validation_X = normal_val_X
#             validation_y = normal_val_y
#
#         if self.abnormal_in_train:
#             train_X = np.concatenate((normal_train_X, abnormal_train_X))
#             train_y = np.concatenate((normal_train_y, abnormal_train_y))
#         else:
#             train_X = normal_train_X
#             train_y = normal_train_y
#         test_X = np.concatenate((normal_test_X, abnormal_test_X))
#         test_y = np.concatenate((normal_test_y, abnormal_test_y))
#
#         abnormal_num_train = train_y.sum()
#         abnormal_num_test = test_y.sum()
#         abnormal_num_val = validation_y.sum()
#
#         trainset = TensorDataset(torch.Tensor(train_X), torch.Tensor(train_y))
#         testset = TensorDataset(torch.Tensor(test_X), torch.Tensor(test_y))
#         validset = TensorDataset(torch.Tensor(validation_X), torch.Tensor(validation_y))
#         test_tensor = (torch.Tensor(test_X), torch.Tensor(test_y))
#
#         return trainset, testset, validset, test_tensor, abnormal_num_train, abnormal_num_test, abnormal_num_val
#
#     def create_datasets(self):
#         return self.train_set, self.valid_set, self.test_set
