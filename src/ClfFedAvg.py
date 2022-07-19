import argparse
import os
import time
from abc import ABC
from collections import OrderedDict
from functools import reduce
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import ADDRepDataset
from model import SVDD
from optimizer import HSCTrainer, RHSCTrainer
from fl import weight_to_state_dict, state_dict_to_weight
from utils import save_data, set_random_seed


class RHSCFedTraining(ABC):
    def __init__(self, num_clients: int, num_rounds: int, trainer_params, model_params, descriptor_params,
                 cuda_num_list, model_name='RHSC', abnormal_in_val=False, rep_model='RNNAE', eval_device='cuda:0',
                 personalization=True):
        super(RHSCFedTraining, self).__init__()
        self.model_name = model_name
        self.abnormal_in_val = abnormal_in_val
        self.rep_model = rep_model
        self.cuda_num_list = cuda_num_list
        self.trainer_params = trainer_params
        self.personalization = personalization

        self.model_params = model_params
        self.eval_device = eval_device
        self.descriptor_params = descriptor_params

        self.num_clients = num_clients
        self.num_rounds = num_rounds

        self.current_rnd = 0
        self.g_weights = None
        self.g_c = None
        self.g_R = None
        self.keys = None
        self.initial_dict = None

        self.train_lr_curve_round = []
        self.valid_lr_curve_round = []
        self.train_lr_curve = []
        self.valid_lr_curve = []

        self._prepare_fit()

    def reset(self):
        self.current_rnd = 0
        self.g_weights = None
        self.g_c = None
        self.g_R = None
        self.keys = None
        self.initial_dict = None

        self._prepare_fit()

    def get_global(self):
        return self.weight_to_state_dict(self.g_weights), self.g_c

    def weight_to_state_dict(self, weights):
        params_dict = zip(self.keys, weights)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        return state_dict

    def state_dict_to_weight(self, state_dict):
        return [values.cpu().numpy() for _, values in state_dict.items()]

    def _prepare_fit(self):
        net = SVDD(**self.model_params)
        self.keys = net.state_dict().keys()
        self.initial_dict = self.state_dict_to_weight(net.state_dict())

    def aggregate(self, results) -> (list, list):
        # calculate total example numbers
        num_examples_total = sum([r['num_train_ex'] for r in results])
        # calculate weighted weight parameters
        weights = [(r['R'], r['c'], r['state_dict'], r['num_train_ex']) for r in results]
        weighted_weights = [
            [layer * num_ex for layer in weights] for _, _, weights, num_ex in weights
        ]
        # calculate weighted center c
        weighted_c = [
            c * num_ex for _, c, _, num_ex in weights
        ]

        # weighted radius r
        weighted_R = [
            R * num_ex for R, _, _, num_ex in weights
        ]

        # Avg total weights
        weights_prime = [
            reduce(np.add, layer_updates) / num_examples_total for layer_updates in zip(*weighted_weights)
        ]
        # find global center
        c_prime = [
            reduce(np.add, c_updates) / num_examples_total for c_updates in zip(*weighted_c)
        ]

        print(np.sum(weighted_R) / num_examples_total)

        r_prime = np.sum(weighted_R) / num_examples_total

        return weights_prime, c_prime, r_prime

    def get_save_path(self, rnd):
        self.save_dir_path = self.trainer_params[
            "model_save_path"] if "model_save_path" in self.trainer_params else "../model_save/fl_clf"
        Path(self.save_dir_path).mkdir(exist_ok=True)
        return os.path.join(self.save_dir_path,
                            f"{self.model_name}_gamma{self.descriptor_params['gamma']}_rep{self.rep_model}_{rnd}.pt")

    def round(self, rnd, trainset, validset, testset, host_list, is_aug):
        start = time.time()
        self.descriptor_params['include_pert'] = False

        device = ['cuda:%d' % d for d in self.cuda_num_list]

        p = Pool(self.num_clients)

        if rnd == 0:  # initialize state dict
            state_dict = self.initial_dict
            global_center = None
            global_r = None

        else:  # get from global state dict
            print('retrieve global c, R')
            state_dict = self.g_weights
            global_center = self.g_c
            global_r = self.g_R

        # g_c, trainset, validset, testset, hostid, device, trainer_params, model_params, descriptor_params, current_rnd, weights
        # print(state_dict)

        if is_aug and rnd > self.descriptor_params['init_steps'] - 1:
            self.descriptor_params['include_pert'] = True

        results = p.map(RHSC_client,
                        zip(repeat(global_center),
                            repeat(global_r),
                            trainset,
                            validset,
                            testset,
                            host_list,
                            device,
                            repeat(self.trainer_params),
                            repeat(self.model_params),
                            repeat(self.descriptor_params),
                            repeat(self.current_rnd),
                            repeat(state_dict))
                        )

        # repeat(trainer_params), repeat(model_params), repeat(descriptor_params)
        # aggregation
        global_weight, global_c, global_r = self.aggregate(results)

        train_list = [res['train_list'] for res in results]
        valid_list = [res['valid_list'] for res in results]

        c_list = [res['c'] for res in results]
        R_list = [res['R'] for res in results]

        print(f"Round {rnd} time: {time.time() - start} s")

        p.close()
        p.join()

        return global_weight, np.array(global_c), global_r, train_list, valid_list, c_list

    def fit(self, test_portion):
        print('start training..')
        start = time.time()
        dataset = ADDRepDataset(test_portion, 0.2, model_name=self.rep_model, abnormal_in_val=self.abnormal_in_val)
        trainset, validset, testset = dataset.create_datasets()
        host_list = dataset.window_hostid_list

        abnormal_test_num = np.array(dataset.abnormal_test)
        test_with_abnormal = np.where(abnormal_test_num > 0)[0]

        is_aug = self.descriptor_params['include_pert']

        for rnd in range(self.num_rounds):
            g_weights, g_c, g_r, train_list, valid_list, c_list = self.round(self.current_rnd, trainset, validset, testset,
                                                                             host_list,
                                                                             is_aug)
            # local_auc, local_f1, local_acc
            self.g_weights = g_weights
            self.g_c = g_c
            self.g_R = g_r
            self.train_lr_curve.append(train_list)
            self.valid_lr_curve.append(valid_list)
            self.train_lr_curve_round.append([np.mean(tl) for tl in train_list])
            self.valid_lr_curve_round.append([np.mean(vl) for vl in valid_list])

            print(f"Global R: {self.g_R}")

            torch.save(dict(
                rnd=self.current_rnd,
                state_dict=self.g_weights,
                train_lr_curve=self.train_lr_curve,
                valid_lr_curve=self.valid_lr_curve,
                train_lr_curve_round=self.train_lr_curve_round,
                valid_lr_curve_round=self.valid_lr_curve_round,
                c=self.g_c,
                c_list=c_list,
                R=self.g_R,
                trainer_params=self.descriptor_params
            ), self.get_save_path(rnd))
            self.current_rnd += 1

        print('total training time: ', time.time() - start)

    def resume_round(self, rnd=0):
        file_path = self.get_save_path(rnd)
        res = torch.load(file_path)
        self.current_rnd = rnd
        self.g_weights = res['state_dict']
        self.g_c = res['c']
        self.g_R = res['R']

    def eval(self, test_portion):
        dataset = ADDRepDataset(test_portion, 0.2, model_name=self.rep_model, abnormal_in_val=True)
        trainset, validset, testset = dataset.create_datasets()
        total_testset = dataset.return_total_test()

        host_list = dataset.window_hostid_list

        abnormal_test_num = np.array(dataset.abnormal_test)
        test_with_abnormal = np.where(abnormal_test_num > 0)[0]

        local_auc = []
        local_f1 = []
        local_acc = []

        for i in test_with_abnormal:
            num_test, report, auc, _ = self.evaluate(testset[i], host_list[i], device=self.eval_device,
                                                     weight=self.g_weights, c=self.g_c)
            local_auc.append(auc)
            local_f1.append(report['1.0']['f1-score'])
            local_acc.append(report['accuracy'])

        print(f"Local results \t| auc: {local_auc}, f1: {local_f1}, acc: {local_acc}")
        print(
            f"Total results \t| auc: {np.mean(local_auc)} {np.std(local_auc)}, f1: {np.mean(local_f1)} {np.std(local_f1)}, acc: {np.mean(local_acc)} {np.std(local_acc)}")

        return local_auc, local_f1, local_acc

    def evaluate(self, testset, hostid, device, weight=None, c=None):
        print('Evaluation..')
        # print(self.g_weights)
        testl = DataLoader(testset, **self.trainer_params['testloader'])

        net = SVDD(**self.model_params)

        if not weight is None:
            global_state_dict = self.weight_to_state_dict(weight)  # init global network
        else:
            global_state_dict = self.weight_to_state_dict(self.g_weights)  # init global network
        net.load_state_dict(global_state_dict)  # load global state dict

        self.descriptor_params['device'] = device
        self.descriptor_params['cid'] = hostid

        trainer = RHSCTrainer(**self.descriptor_params)

        if not c is None:
            trainer.c = torch.tensor(c).to(device)
        else:
            trainer.c = torch.tensor(self.g_c).to(device)

        num_test, obj, test_auc, return_type, acc = trainer.test(testl, net, r=self.g_R)  # return type = 1: normal acc

        return num_test, obj, test_auc, return_type


def client(args):
    g_c, trainset, validset, testset, hostid, device, trainer_params, model_params, descriptor_params, current_rnd, weights = \
        args[0], \
        args[1], \
        args[2], \
        args[3], \
        args[4], \
        args[5], \
        args[6], \
        args[7], \
        args[8], \
        args[9], \
        args[10]

    trainl = DataLoader(trainset, **trainer_params['trainloader'])
    validl = DataLoader(validset, **trainer_params['valloader'])

    print(f'hostid : {hostid}, device: {device}')

    net = SVDD(**model_params)
    keys = net.state_dict().keys()
    global_state_dict = weight_to_state_dict(keys, weights)
    net.load_state_dict(global_state_dict)  # load global state dict

    descriptor_params['device'] = device
    descriptor_params['cid'] = hostid
    if current_rnd > 0:
        descriptor_params['init_steps'] = 0

    trainer = HSCTrainer(**descriptor_params)

    # update c with global center
    if not g_c is None:
        trainer.c = torch.tensor(g_c).to(device)

    c, net, num_train, train_llist, valid_llist = trainer.train(trainl, validl, net)
    # num_test, obj, test_auc, return_type = trainer.test(testl, net,
    #                                                     global_thresh=0.5)  # return type = 1: normal acc

    res = dict(
        state_dict=state_dict_to_weight(net.state_dict()),
        c=c.detach().cpu().numpy(),
        train_list=train_llist,
        valid_list=valid_llist,
        # report=obj,
        # auc=test_auc,
        # return_type=return_type,
        # num_test_ex=num_test,
        num_train_ex=num_train,
        hostid=hostid
    )

    return res


def RHSC_client(args):
    g_c, g_R, trainset, validset, testset, hostid, device, trainer_params, model_params, descriptor_params, current_rnd, weights = \
        args[0], \
        args[1], \
        args[2], \
        args[3], \
        args[4], \
        args[5], \
        args[6], \
        args[7], \
        args[8], \
        args[9], \
        args[10], \
        args[11]

    trainl = DataLoader(trainset, **trainer_params['trainloader'])
    validl = DataLoader(validset, **trainer_params['valloader'])

    print(f'hostid : {hostid}, device: {device}')

    net = SVDD(**model_params)
    keys = net.state_dict().keys()
    global_state_dict = weight_to_state_dict(keys, weights)
    net.load_state_dict(global_state_dict)  # load global state dict

    descriptor_params['device'] = device
    descriptor_params['cid'] = hostid
    if current_rnd > 0:
        descriptor_params['init_steps'] = 0

    trainer = RHSCTrainer(**descriptor_params)

    # update c with global center
    if not g_c is None:
        trainer.c = torch.tensor(g_c).to(device)

    if not g_R is None:
        print('set g r')
        trainer.R = torch.tensor(g_R, requires_grad=True)

    c, R, net, num_train = trainer.train(trainl, validl, net)
    # num_test, obj, test_auc, return_type = trainer.test(testl, net,
    #                                                     global_thresh=0.5)  # return type = 1: normal acc
    R_hist = trainer.R_hist
    res = dict(
        state_dict=state_dict_to_weight(net.state_dict()),
        c=c.detach().cpu().numpy(),
        R=R,
        R_hist=R_hist,
        train_list=trainer.train_loss_list,
        valid_list=trainer.valid_loss_list,
        # report=obj,
        # auc=test_auc,
        # return_type=return_type,
        # num_test_ex=num_test,
        num_train_ex=num_train,
        hostid=hostid
    )

    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Federated Learning Simulation based on Flower")
    parser.add_argument('--seed', type=int, default=5959, help='random seed')
    parser.add_argument('--num_clients', '-n', type=int, default=8, help='number of clients')
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='batch size for local update')
    parser.add_argument('--num_epochs', '-e', type=int, default=5, help='number of local epochs')
    parser.add_argument('--num_rounds', '-r', type=int, default=40, help='number of required rounds')
    parser.add_argument('--eval_device', '-ed', type=str, default='cuda:2', help='cuda device for evaluation')
    parser.add_argument('--weight_decay', '-w', type=float, default=0., help='weight decay')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--init_steps', '-i', type=int, default=5, help='initial steps before augmentation')
    parser.add_argument('--patience', '-p', type=int, default=10, help='patience for early stopping (no meaning)')
    parser.add_argument('--pert_steps', '-ps', type=int, default=10, help='augmentation perturbation steps')
    parser.add_argument('--pert_step_size', '-pss', type=float, default=0.001,
                        help='augmentation perturbation step sizes')
    parser.add_argument('--pert_duration', '-pd', type=int, default=2, help='augmentation duration (epochs)')
    parser.add_argument('--gamma', '-g', type=float, default=0.1, help='local gamma weight')
    parser.add_argument('--radius_thresh', '-rt', type=float, default=0.95,
                        help='threshold for local radius calculation')
    parser.add_argument('--test_portion', '-tp', type=float, default=0.8, help='portion of testset')
    parser.add_argument('--rep_model', '-rm', type=str, default="CAE", help='representation model')
    parser.add_argument('--abnormal_in_val', dest='abnormal_in_val', action='store_true')
    parser.add_argument('--include_pert', dest='include_pert', action='store_true')
    parser.add_argument('--verbose', dest='verbose', action='store_true')
    parser.add_argument('--resume_rnd', '-rr', type=int, default=-1, help='resume round')
    parser.add_argument('')


    args = parser.parse_args()

    set_random_seed(args.seed)

    trainer_params = {
        'trainloader': dict(
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0
        ),
        'testloader': dict(
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0
        ),
        'valloader': dict(
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0
        ),
        'model_save_path': '/workspace/code/FedHSC/model_save/fl_clf'
    }

    model_params = {
        'rep_model_type': 'RNNEncoder',
        'in_dim': 32,
        'hidden_dims': [16, 64],  # hidden_dim, rep_dim
        'window_size': 128,
        'num_svdd_layer': 3,
        'num_rep_layer': 3
    }

    descriptor_params = {'optimizer_name': 'Adam', 'lr': args.learning_rate, 'n_epochs': args.num_epochs,
                         'lr_milestones': [20, 40], 'verbose': args.verbose,
                         'weight_decay': args.weight_decay, 'init_steps': args.init_steps, 'patience': args.patience,
                         'pert_steps': args.pert_steps, 'pert_step_size': args.pert_step_size,
                         'pert_duration': args.pert_duration, 'gamma': args.gamma, 'radius_thresh': args.radius_thresh,
                         'include_pert': args.include_pert}

    # descriptor_params = {'optimizer_name': 'Adam', 'lr': 1e-5, 'n_epochs': num_epochs, 'lr_milestones': [20, 40],
    #                      'verbose': False,
    #                      'weight_decay': 0, 'device': 'cpu', 'init_steps': 5, 'cid': 'a', 'patience': 10,
    #                      'gamma': 1.05, 'radius_thresh': 0.95, 'pert_steps': 10, 'pert_step_size': 0.001,
    #                      'pert_duration': 2, 'include_pert': False}

    fl_trainer = RHSCFedTraining(num_clients=args.num_clients,
                                 num_rounds=args.num_rounds,
                                 rep_model=args.rep_model,
                                 abnormal_in_val=args.abnormal_in_val,
                                 trainer_params=trainer_params,
                                 model_params=model_params,
                                 descriptor_params=descriptor_params,
                                 cuda_num_list=[3, 3, 4, 4, 5, 5, 6, 6],
                                 eval_device=args.eval_device)

    if args.resume_rnd > 0:
        fl_trainer.resume_round(args.resume_rnd)

    fl_trainer.fit(args.test_portion)
    local_auc, local_f1, local_acc = fl_trainer.eval(args.test_portion)

    save_data(dict(
        auc=local_auc,
        f1=local_f1,
        acc=local_acc
    ), '/workspace/code/Fed_clf/results', f"RHSCFed_gamma{int(descriptor_params['gamma'])}_result.pkl")
