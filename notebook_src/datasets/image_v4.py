import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset, ConcatDataset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10

from src.utils import set_random_seed


class ImageDataset(Dataset):

    def __init__(self, data, targets, transform=None):
        super(ImageDataset, self).__init__()
        self.data = data
        self.targets = targets
        self.transform = transform

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

        return img, target


class ImageDatasetModule(object):

    def __init__(self, dataset_params):
        self.params = dataset_params
        # self.train_id_dataset = None
        # self.train_ood_dataset = None
        # self.val_id_dataset = None
        # self.val_ood_dataset = None
        # self.test_id_dataset = None
        # self.test_ood_dataset = None
        self.data_dict = dict()
        self.client_dict = dict()

        self.normalization_variables = None
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def repr(self):
        for client_idx, target_dict in self.client_dict.items():
            for target, data_dict in target_dict.items():
                # print(data_dict)
                for key, data in data_dict.items():
                    print(f"Client idx : {client_idx}\t Target idx : {target}\t {key} \t Size: {len(data)}")

    @property
    def train_dataloader(self):
        return None

    @property
    def val_dataloader(self):
        return None

    @property
    def test_dataloader(self):
        return None

    @classmethod
    def split_by_client(cls, dataset, num_client):
        num_data_per_client = int(len(dataset) / num_client)
        num_data_last_client = len(dataset) - (num_data_per_client * (num_client - 1))
        num_data_per_client_list = [num_data_per_client for _ in range(num_client - 1)] + [num_data_last_client]

        return random_split(dataset, num_data_per_client_list)

    @classmethod
    def create(cls, dataset_name, dataset_params):
        dataset_module = None

        if dataset_name.lower() == "mnist":
            dataset_module = MNISTDatasetModule(dataset_params=dataset_params)
        if dataset_name.lower() == "cifar10":
            dataset_module = CIFAR10DatasetModule(dataset_params=dataset_params)

        return dataset_module

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, *args, **kwargs):
        pass

    def split_dataset(self, data_list, target_list):
        is_shuffle = self.params["is_shuffle"]
        id_targets = self.params["id_targets"]
        ood_target = self.params["ood_target"]
        num_clients = self.params["num_clients"]
        num_id_target_per_client = self.params["num_id_target_per_client"]
        num_ood_clients = self.params["num_ood_clients"]
        train_ratio = self.params["train_ratio"]
        val_ratio = self.params["val_ratio"]
        sampling_ratio = self.params["sampling_ratio"]
        is_convert_target = self.params["is_convert_target"]
        random_seed = self.params["random_seed"]

        set_random_seed(random_seed)
        print(f"Set random seed : {random_seed}")

        data_dict = {target: dict(entire_data_list=list(), entire_target_list=list()) for target in
                     set(target_list.numpy())}
        client_dict = {
            client: dict(
                entire_train_data_list=list(),
                entire_train_target_list=list(),
                entire_train_abnormal_data_list=list(),
                entire_train_abnormal_target_list=list(),
                test_data_list=list(),
                test_target_list=list()
            ) for client in range(num_clients)
        }

        idx_list = np.concatenate([np.random.permutation(id_targets) for _ in
                                   range((num_clients * num_id_target_per_client) // len(id_targets) + 1)])

        # idx_list = np.tile(np.random.permutation(id_targets), (num_clients // len(id_targets) + 1))

        for client_idx, (client_idx, client_data_dict) in enumerate(client_dict.items()):
            # normal_target_list = np.random.choice(id_targets, num_id_target_per_client, replace=False)
            normal_target_list = idx_list[client_idx * 2:client_idx * 2 + 2]
            client_data_dict["normal_target_list"] = normal_target_list

        # Shuffling
        if is_shuffle:
            shuffle_idx_list = np.arange(len(data_list))
            np.random.shuffle(shuffle_idx_list)
            data_list, target_list = data_list[shuffle_idx_list], target_list[shuffle_idx_list]
            print("Dataset shuffled")

        # Split by clients
        for data, target in zip(data_list, target_list):
            data_dict[target.item()]["entire_data_list"].append(data)
            data_dict[target.item()]["entire_target_list"].append(target)

        # Train / Test split
        for target in data_dict.keys():
            data_list_per_target = data_dict[target]["entire_data_list"]

            train_size = int(len(data_list_per_target) * train_ratio)

            data_dict[target]["train_data_list"] = data_dict[target]["entire_data_list"][:train_size]
            data_dict[target]["train_target_list"] = data_dict[target]["entire_target_list"][:train_size]
            data_dict[target]["test_data_list"] = data_dict[target]["entire_data_list"][train_size:]
            data_dict[target]["test_target_list"] = data_dict[target]["entire_target_list"][train_size:]

        total_normal_data_size = 0

        # Calculate total normal data size
        for target, target_data_dict in data_dict.items():
            if target == ood_target:
                continue
            total_normal_data_size += len(target_data_dict["entire_data_list"])

        # Calculate class weight and sampling size
        for target, target_data_dict in data_dict.items():
            if target == ood_target:
                continue
            class_weight = len(target_data_dict["entire_data_list"]) / total_normal_data_size
            target_data_dict["class_weight"] = class_weight
            target_data_dict["train_sample_size"] = round(
                len(target_data_dict["train_data_list"]) * sampling_ratio * class_weight)
            target_data_dict["test_sample_size"] = round(
                len(target_data_dict["test_data_list"]) * sampling_ratio * class_weight)
            target_data_dict["is_able_sampling"] = True

        print("============ Start Sampling ============")
        # Sampling
        while True:
            is_able_all_clients_sampling = False

            for client_idx, client_data_dict in client_dict.items():
                normal_target_list = client_data_dict["normal_target_list"]
                is_able_current_client_sampling = True

                for normal_target in normal_target_list:
                    normal_target_data_dict = data_dict[normal_target]

                    # Check the sampling
                    if not normal_target_data_dict["is_able_sampling"]:
                        is_able_current_client_sampling = False
                        continue

                    train_sample_size = normal_target_data_dict["train_sample_size"]

                    normal_train_data_list = normal_target_data_dict["train_data_list"]
                    normal_train_target_list = normal_target_data_dict["train_target_list"]

                    if len(normal_train_data_list) < train_sample_size:
                        print(f"{normal_target} Train remains - {len(normal_train_data_list)}")
                        normal_target_data_dict["is_able_sampling"] = False
                        continue

                    client_data_dict["entire_train_data_list"] += normal_train_data_list[:train_sample_size]
                    client_data_dict["entire_train_target_list"] += normal_train_target_list[:train_sample_size]
                    del normal_train_data_list[:train_sample_size]
                    del normal_train_target_list[:train_sample_size]

                    is_able_all_clients_sampling += is_able_current_client_sampling

            if not is_able_all_clients_sampling:
                break

        print("============ Complete Sampling ============")

        # Append abnormal data
        abnormal_train_data_list = data_dict[ood_target]["train_data_list"]
        abnormal_train_target_list = data_dict[ood_target]["train_target_list"]

        client_abnormal_data_size = round(len(abnormal_train_data_list) / num_clients)

        for abnormal_client_idx in range(num_ood_clients):
            client_dict[abnormal_client_idx]["entire_train_abnormal_data_list"] = abnormal_train_data_list[
                                                                                  :client_abnormal_data_size]
            client_dict[abnormal_client_idx]["entire_train_abnormal_target_list"] = abnormal_train_target_list[
                                                                                    :client_abnormal_data_size]

            del abnormal_train_data_list[:client_abnormal_data_size]
            del abnormal_train_target_list[:client_abnormal_data_size]

        # Split validation dataset
        for client_idx, client_data_dict in client_dict.items():
            entire_train_data_list = client_data_dict["entire_train_data_list"]
            entire_train_target_list = client_data_dict["entire_train_target_list"]

            entire_train_abnormal_data_list = client_data_dict["entire_train_abnormal_data_list"]
            entire_train_abnormal_target_list = client_data_dict["entire_train_abnormal_target_list"]

            val_data_size = round(len(entire_train_data_list) * val_ratio)
            val_abnormal_data_size = round(len(entire_train_abnormal_data_list) * val_ratio)

            if is_convert_target:
                entire_train_target_list = list(map(lambda target: 0, entire_train_target_list))
                entire_train_abnormal_target_list = list(map(lambda target: 1, entire_train_abnormal_target_list))

            # Append normal
            client_data_dict["val_data_list"] = entire_train_data_list[:val_data_size]
            client_data_dict["val_target_list"] = entire_train_target_list[:val_data_size]

            client_data_dict["train_data_list"] = entire_train_data_list[val_data_size:]
            client_data_dict["train_target_list"] = entire_train_target_list[val_data_size:]

            # Append abnormal
            client_data_dict["val_data_list"] += entire_train_abnormal_data_list[:val_abnormal_data_size]
            client_data_dict["val_target_list"] += entire_train_abnormal_target_list[:val_abnormal_data_size]

            client_data_dict["train_data_list"] += entire_train_abnormal_data_list[val_abnormal_data_size:]
            client_data_dict["train_target_list"] += entire_train_abnormal_target_list[val_abnormal_data_size:]

            print(f"Client {client_idx}")
            print(f"Train data normal size : {len(entire_train_data_list[val_data_size:])}")
            print(f"Train data abnormal size : {len(entire_train_abnormal_data_list[val_abnormal_data_size:])}")
            print(f"Val data normal size : {len(entire_train_data_list[:val_data_size])}")
            print(f"Val data abnormal size : {len(entire_train_abnormal_data_list[:val_abnormal_data_size])}")
            print("==================================================")

        # Append total test dataset
        total_test_data_list = list()
        total_test_target_list = list()

        print("Test dataset information")
        for target_idx, target_data_dict in data_dict.items():
            test_data_list = target_data_dict["test_data_list"]
            test_target_list = target_data_dict["test_target_list"]

            if target_idx == ood_target:
                if is_convert_target:
                    test_target_list = list(map(lambda target: 1, test_target_list))
            else:
                if is_convert_target:
                    test_target_list = list(map(lambda target: 0, test_target_list))

            total_test_data_list += test_data_list
            total_test_target_list += test_target_list

            print("===================================")
            print(f"Class {target_idx}")
            print(f"Test data size : {len(test_data_list)}")

        client_dict["total_test_data_list"] = total_test_data_list
        client_dict["total_test_target_list"] = total_test_target_list

        print(f"Total test data size : {len(total_test_data_list)}")

        self.data_dict = data_dict
        self.client_dict = client_dict

        return client_dict

    def convert_dataset(self, client_dict, client_idx):
        train_data_list = client_dict[client_idx]["train_data_list"]
        train_target_list = client_dict[client_idx]["train_target_list"]
        val_data_list = client_dict[client_idx]["val_data_list"]
        val_target_list = client_dict[client_idx]["val_target_list"]
        test_data_list = client_dict[client_idx]["test_data_list"]
        test_target_list = client_dict[client_idx]["test_target_list"]

        normalization_variables_dict = self.calculate_normalization_variables(data_list=train_data_list,
                                                                              target_list=train_target_list)
        self.set_normalization_variables(
            normalization_variables_dict=normalization_variables_dict
        )
        normalization_variables = self.normalization_variables

        # Set transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*normalization_variables)
        ])

        return dict(
            train_dataset=ImageDataset(data=train_data_list, targets=train_target_list, transform=transform),
            val_dataset=ImageDataset(data=val_data_list, targets=val_target_list, transform=transform),
            test_dataset=ImageDataset(data=test_data_list, targets=test_target_list, transform=transform)
        )

    def convert_total_dataset(self, client_dict):
        total_train_data_list = list()
        total_train_target_list = list()
        total_val_data_list = list()
        total_val_target_list = list()

        for client_idx, client_data_dict in client_dict.items():
            if not isinstance(client_idx, int):
                continue

            total_train_data_list += client_data_dict["train_data_list"]
            total_train_target_list += client_data_dict["train_target_list"]

            total_val_data_list += client_data_dict["val_data_list"]
            total_val_target_list += client_data_dict["val_target_list"]

        test_data_list = client_dict["total_test_data_list"]
        test_target_list = client_dict["total_test_target_list"]

        normalization_variables_dict = self.calculate_normalization_variables(
            data_list=total_train_data_list + total_val_data_list,
            target_list=total_train_target_list + total_val_target_list
        )
        self.set_normalization_variables(
            normalization_variables_dict=normalization_variables_dict
        )
        normalization_variables = self.normalization_variables

        # Set transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*normalization_variables)
        ])

        return dict(
            train_dataset=ImageDataset(data=total_train_data_list, targets=total_train_target_list,
                                       transform=transform),
            val_dataset=ImageDataset(data=total_val_data_list, targets=total_val_target_list, transform=transform),
            test_dataset=ImageDataset(data=test_data_list, targets=test_target_list, transform=transform)
        )

        # id_targets_bool = torch.full((target_list.shape[0],), False)
        # ood_targets_bool = torch.full((target_list.shape[0],), False)
        # # test_ood_targets_bool = torch.full((target_list.shape[0],), False)
        #
        # # Train id targets
        # for id_true_target in id_targets:
        #     id_targets_bool += (target_list == id_true_target)
        #
        # # Train ood targets
        # for ood_true_target in ood_targets:
        #     ood_targets_bool += (target_list == ood_true_target)
        #
        # # Test ood targets
        # # for test_ood_true_target in test_ood_true_target_list:
        # #     test_ood_targets_bool += (target_list == test_ood_true_target)
        #
        # id_data = data_list[id_targets_bool]
        # id_targets = target_list[id_targets_bool]
        # ood_data = data_list[ood_targets_bool]
        # ood_targets = target_list[ood_targets_bool]
        # # test_ood_data = data_list[test_ood_targets_bool]
        # # test_ood_targets = target_list[test_ood_targets_bool]
        #
        # print(f"Original label info :\n"
        #       f" > ID labels : {sorted(set(id_targets.tolist()))}\n"
        #       f" > OOD labels : {sorted(set(ood_targets.tolist()))}\n")
        #
        #
        # client_dict = dict()
        #
        # return data_list, target_list, id_data, id_targets, ood_data, ood_targets

        #
        # # Transform targets
        # # id_targets = np.array(list(map(lambda id_target: self.ce_label_dict[int(id_target)], id_targets)))
        # id_targets = np.array(list(map(lambda id_target: 0, id_targets)))
        # ood_targets = np.array(list(map(lambda train_ood_target: 1, ood_targets)))
        # test_ood_targets = np.array(list(map(lambda test_ood_target: 1, test_ood_targets)))
        #
        # print(f"Transformed label info :\n"
        #       f" > ID labels : {sorted(set(id_targets.tolist()))}\n"
        #       f" > Train OOD labels : {sorted(set(ood_targets.tolist()))}\n"
        #       f" > Test OOD labels : {sorted(set(test_ood_targets.tolist()))}")
        # print(f"Total dataset :\n"
        #       f" > ID size : {len(id_data)}\n"
        #       f" > Train OOD size : {len(ood_data)}\n"
        #       f" > Test OOD size : {len(test_ood_data)}")
        #
        # # Train split
        # id_train_size = int(len(id_data) * train_ratio)
        # id_val_size = int(len(id_data) * val_ratio)
        # id_test_size = int(len(id_data) * test_ratio)
        # print(id_test_size)
        #
        # split_pivot_start = 0
        # split_pivot_end = 0
        #
        # split_pivot_end += id_train_size
        # id_train_data_list = id_data[split_pivot_start:split_pivot_end]
        # id_train_target_list = id_targets[split_pivot_start:split_pivot_end]
        #
        # split_pivot_start += id_train_size
        # split_pivot_end += id_val_size
        # id_val_data_list = id_data[split_pivot_start:split_pivot_end]
        # id_val_target_list = id_targets[split_pivot_start:split_pivot_end]
        #
        # split_pivot_start += id_val_size
        # split_pivot_end += id_test_size
        # id_test_data_list = id_data[split_pivot_start:split_pivot_end]
        # id_test_target_list = id_targets[split_pivot_start:split_pivot_end]
        #
        # # Set normalized variables
        # normalization_variables_dict = self.calculate_normalization_variables(data_list=id_train_data_list)
        # self.set_normalization_variables(
        #     normalization_variables_dict=normalization_variables_dict
        # )
        # normalization_variables = self.normalization_variables
        #
        # # Set transforms
        # self.transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(*normalization_variables)
        # ])
        #
        # train_id_dataset = ImageDataset(data=id_train_data_list, targets=id_train_target_list, transform=self.transform)
        # val_id_dataset = ImageDataset(data=id_val_data_list, targets=id_val_target_list, transform=self.transform)
        # test_id_dataset = ImageDataset(data=id_test_data_list, targets=id_test_target_list, transform=self.transform)
        #
        # print(f"ID dataset :\n"
        #       f" > Train size : {len(train_id_dataset)}\n"
        #       f" > Val size : {len(val_id_dataset)}\n"
        #       f" > Test size : {len(test_id_dataset)}")
        #
        # # Val split
        # val_ood_train_size = int(len(ood_data) * train_ratio)
        # val_ood_val_size = int(len(ood_data) * val_ratio)
        # val_ood_test_size = int(len(ood_data) * test_ratio)
        #
        # split_pivot_start = 0
        # split_pivot_end = 0
        #
        # split_pivot_end += val_ood_train_size
        # val_ood_train_data_list = ood_data[split_pivot_start:split_pivot_end]
        # val_ood_train_target_list = ood_targets[split_pivot_start:split_pivot_end]
        #
        # split_pivot_start += val_ood_train_size
        # split_pivot_end += val_ood_val_size
        # val_ood_val_data_list = ood_data[split_pivot_start:split_pivot_end]
        # val_ood_val_target_list = ood_targets[split_pivot_start:split_pivot_end]
        #
        # split_pivot_start += val_ood_val_size
        # split_pivot_end += val_ood_test_size
        # val_ood_test_data_list = ood_data[split_pivot_start:split_pivot_end]
        # val_ood_test_target_list = ood_targets[split_pivot_start:split_pivot_end]
        #
        # # Val OOD data cutting with ratio
        # if ood_ratio:
        #     # Train
        #     val_ood_train_ood_cut_idx = int(len(id_train_data_list) * ood_ratio)
        #     val_ood_train_data_list = val_ood_train_data_list[:val_ood_train_ood_cut_idx]
        #     val_ood_train_target_list = val_ood_train_target_list[:val_ood_train_ood_cut_idx]
        #
        #     # Val
        #     val_ood_val_ood_cut_idx = int(len(id_val_data_list) * ood_ratio)
        #     val_ood_val_data_list = val_ood_val_data_list[:val_ood_val_ood_cut_idx]
        #     val_ood_val_target_list = val_ood_val_target_list[:val_ood_val_ood_cut_idx]
        #
        #     # Test
        #     val_ood_test_ood_cut_idx = int(len(id_test_data_list) * ood_ratio)
        #     val_ood_test_data_list = val_ood_test_data_list[:val_ood_test_ood_cut_idx]
        #     val_ood_test_target_list = val_ood_test_target_list[:val_ood_test_ood_cut_idx]
        #
        # val_ood_train_dataset = ImageDataset(data=val_ood_train_data_list, targets=val_ood_train_target_list,
        #                                      transform=self.transform)
        # val_ood_val_dataset = ImageDataset(data=val_ood_val_data_list, targets=val_ood_val_target_list,
        #                                    transform=self.transform)
        # val_ood_test_dataset = ImageDataset(data=val_ood_test_data_list, targets=val_ood_test_target_list,
        #                                     transform=self.transform)
        #
        # print(f"Val-OOD dataset :\n"
        #       f" > Train size : {len(val_ood_train_dataset)}\n"
        #       f" > Val size : {len(val_ood_val_dataset)}\n"
        #       f" > Test size : {len(val_ood_test_dataset)}")
        #
        # # Test split
        # test_ood_train_size = int(len(test_ood_data) * train_ratio)
        # test_ood_val_size = int(len(test_ood_data) * val_ratio)
        # test_ood_test_size = int(len(test_ood_data) * test_ratio)
        #
        # split_pivot_start = 0
        # split_pivot_end = 0
        #
        # split_pivot_end += test_ood_train_size
        # test_ood_train_data_list = test_ood_data[split_pivot_start:split_pivot_end]
        # test_ood_train_target_list = test_ood_targets[split_pivot_start:split_pivot_end]
        #
        # split_pivot_start += test_ood_train_size
        # split_pivot_end += test_ood_val_size
        # test_ood_val_data_list = test_ood_data[split_pivot_start:split_pivot_end]
        # test_ood_val_target_list = test_ood_targets[split_pivot_start:split_pivot_end]
        #
        # split_pivot_start += test_ood_val_size
        # split_pivot_end += test_ood_test_size
        # test_ood_test_data_list = test_ood_data[split_pivot_start:split_pivot_end]
        # test_ood_test_target_list = test_ood_targets[split_pivot_start:split_pivot_end]
        #
        # # Val OOD data cutting with ratio
        # if ood_ratio:
        #     # Train
        #     test_ood_train_ood_cut_idx = int(len(id_train_data_list) * ood_ratio)
        #     test_ood_train_data_list = test_ood_train_data_list[:test_ood_train_ood_cut_idx]
        #     test_ood_train_target_list = test_ood_train_target_list[:test_ood_train_ood_cut_idx]
        #
        #     # Val
        #     test_ood_val_ood_cut_idx = int(len(id_val_data_list) * ood_ratio)
        #     test_ood_val_data_list = test_ood_val_data_list[:test_ood_val_ood_cut_idx]
        #     test_ood_val_target_list = test_ood_val_target_list[:test_ood_val_ood_cut_idx]
        #
        #     # Test
        #     test_ood_test_ood_cut_idx = int(len(id_test_data_list) * ood_ratio)
        #     test_ood_test_data_list = test_ood_test_data_list[:test_ood_test_ood_cut_idx]
        #     test_ood_test_target_list = test_ood_test_target_list[:test_ood_test_ood_cut_idx]
        #
        # test_ood_train_dataset = ImageDataset(data=test_ood_train_data_list, targets=test_ood_train_target_list,
        #                                       transform=self.transform)
        # test_ood_val_dataset = ImageDataset(data=test_ood_val_data_list, targets=test_ood_val_target_list,
        #                                     transform=self.transform)
        # test_ood_test_dataset = ImageDataset(data=test_ood_test_data_list, targets=test_ood_test_target_list,
        #                                      transform=self.transform)
        #
        # print(f"Test-OOD dataset :\n"
        #       f" > Train size : {len(test_ood_train_dataset)}\n"
        #       f" > Val size : {len(test_ood_val_dataset)}\n"
        #       f" > Test size : {len(test_ood_test_dataset)}")
        #
        # self.train_id_dataset, self.val_id_dataset, self.test_id_dataset = \
        #     train_id_dataset, val_id_dataset, test_id_dataset
        # self.train_ood_dataset, self.val_ood_dataset, self.test_ood_dataset = \
        #     val_ood_train_dataset, val_ood_val_dataset, test_ood_test_dataset
        #
        # print(f"Train dataset :\n"
        #       f" > ID size : {len(self.train_id_dataset)}\n"
        #       f" > OOD size : {len(self.train_ood_dataset)}")
        # print(f"Val dataset :\n"
        #       f" > ID size : {len(self.val_id_dataset)}\n"
        #       f" > OOD size : {len(self.val_ood_dataset)}")
        # print(f"Test dataset :\n"
        #       f" > ID size : {len(self.test_id_dataset)}\n"
        #       f" > OOD size : {len(self.test_ood_dataset)}")
        #
        # return dict(
        #     train_id_dataset=train_id_dataset,
        #     train_ood_dataset=val_ood_train_dataset,
        #     val_id_dataset=val_id_dataset,
        #     val_ood_dataset=val_ood_val_dataset,
        #     test_id_dataset=test_id_dataset,
        #     test_ood_dataset=test_ood_test_dataset,
        # )

    def calculate_normalization_variables(self, data_list, target_list):
        # Default : (n, n, 3) images, plz cascading this method.
        normal_data_list = list()

        for data, target in zip(data_list, target_list):
            if target == 0:
                normal_data_list.append(data)

        # data_list = np.array(normal_data_list)

        mean_arr = np.array([np.mean(data.numpy(), axis=(0, 1)) for data in normal_data_list])
        std_arr = np.array([np.std(data.numpy(), axis=(0, 1)) for data in normal_data_list])

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


class MNISTDatasetModule(ImageDatasetModule):
    SAVE_DIR_PATH = "/workspace/code/data/"

    def __init__(self, dataset_params):
        super().__init__(dataset_params=dataset_params)

    def prepare_data(self, *args, **kwargs):
        MNIST(root=MNISTDatasetModule.SAVE_DIR_PATH, train=True, download=True)
        MNIST(root=MNISTDatasetModule.SAVE_DIR_PATH, train=False, download=True)

    def setup(self, client_idx=None, *args, **kwargs):
        """
        Generate Train | Validation | Test
        """
        dataset_dict = dict()

        train_entire = MNIST(
            root=MNISTDatasetModule.SAVE_DIR_PATH,
            train=True,
            transform=self.transform
        )
        test_entire = MNIST(
            root=MNISTDatasetModule.SAVE_DIR_PATH,
            train=False,
            transform=self.transform
        )

        data_list = torch.cat(tensors=[train_entire.data, test_entire.data])
        target_list = torch.cat(tensors=[train_entire.targets, test_entire.targets])

        if "is_client_split" in self.params and self.params["is_client_split"] is True:
            train_id_true_target_list = self.params["train_id_targets"]

            for client_idx in train_id_true_target_list:
                dataset_dict[client_idx] = self.split_dataset(data_list=data_list, target_list=target_list,
                                                              client_idx=client_idx)
                print("=========================================")

            # Check if exists abnormal clients
            if "abnormal_clients" in self.params:
                abnormal_client_list = self.params["abnormal_clients"]
            else:
                abnormal_client_list = list()

            for key, value in dataset_dict.items():
                if key in abnormal_client_list:
                    dataset_dict[key]["train_dataset"] = ConcatDataset(
                        datasets=[value["train_id_dataset"], value["train_ood_dataset"]])
                    dataset_dict[key]["val_dataset"] = ConcatDataset(
                        datasets=[value["val_id_dataset"], value["val_ood_dataset"]])
                    dataset_dict[key]["test_dataset"] = ConcatDataset(
                        datasets=[value["test_id_dataset"], value["test_ood_dataset"]])
                else:
                    dataset_dict[key]["train_dataset"] = ConcatDataset(datasets=[value["train_id_dataset"]])
                    dataset_dict[key]["val_dataset"] = ConcatDataset(datasets=[value["val_id_dataset"]])
                    dataset_dict[key]["test_dataset"] = ConcatDataset(
                        datasets=[value["test_id_dataset"], value["test_ood_dataset"]])
        else:
            dataset_dict = self.split_dataset(data_list=data_list, target_list=target_list)
        return dataset_dict

    @property
    def train_dataset(self):
        return ConcatDataset(datasets=[self.train_id_dataset, self.train_ood_dataset])

    @property
    def val_dataset(self):
        return ConcatDataset(datasets=[self.val_id_dataset, self.val_ood_dataset])

    @property
    def test_dataset(self):
        return ConcatDataset(datasets=[self.test_id_dataset, self.test_ood_dataset])


class CIFAR10DatasetModule(ImageDatasetModule):
    SAVE_DIR_PATH = "/workspace/code/data/"

    def __init__(self, dataset_params):
        super().__init__(dataset_params=dataset_params)

    def prepare_data(self, *args, **kwargs):
        CIFAR10(root=CIFAR10DatasetModule.SAVE_DIR_PATH, train=True, download=True)
        CIFAR10(root=CIFAR10DatasetModule.SAVE_DIR_PATH, train=False, download=True)

    def setup(self, *args, **kwargs):
        """
        Generate Train | Validation | Test
        """
        train_entire = CIFAR10(root=CIFAR10DatasetModule.SAVE_DIR_PATH, train=True, transform=self.transform)
        test_entire = CIFAR10(root=CIFAR10DatasetModule.SAVE_DIR_PATH, train=False, transform=self.transform)

        data_list = torch.cat(tensors=[torch.tensor(train_entire.data), torch.tensor(test_entire.data)])
        target_list = torch.cat(tensors=[torch.tensor(train_entire.targets), torch.tensor(test_entire.targets)])

        return self.split_dataset(data_list=data_list, target_list=target_list)

    @property
    def train_dataset(self):
        return ConcatDataset(datasets=[self.train_id_dataset, self.train_ood_dataset])

    @property
    def val_dataset(self):
        return ConcatDataset(datasets=[self.val_id_dataset, self.val_ood_dataset])

    @property
    def test_dataset(self):
        return ConcatDataset(datasets=[self.test_id_dataset, self.test_ood_dataset])
